# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5 Next CANN ``key_pool`` operator adaptation tests.

Covers (plan §9.2/§10):
- an independent PyTorch reference of the operator contract (K/gate
  projection, LayerNorm, APE softmax, cross-chunk tail state, pool reduce);
- chunked-prefill vs one-shot state equivalence;
- start_pos at the pool boundaries (0, r-1, r, r+1);
- block-table sentinel (0 = no update) and out-of-order blocks;
- state cache non-contiguous 0-axis stride handling in the reference;
- ``_C_ascend`` schema / Meta shape / dtype contract.

The reference never calls the CANN op; it is used by the NPU accuracy tests
in tests/e2e/models/test_glm5_next_key_pool.py as the standalone oracle.
"""

from __future__ import annotations

import re

import pytest
import torch

from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded
from tests.ut.ops.helpers.glm5_next_cann_reference import key_pool_reference

TORCH_DTYPES = (torch.bfloat16, torch.float16)


def _run_key_pool_once(
    *,
    chunk_lens: list[int],
    start_pos: list[int],
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int,
    hidden: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = state_cache.device
    chunk_lens_t = torch.tensor(chunk_lens, dtype=torch.int64)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32), chunk_lens_t.cumsum(0).to(torch.int32)])
    if hidden is None:
        total = int(cu_seqlens[-1])
        hidden = torch.randn(total, wk.shape[1], dtype=dtype, device=dev) * 0.1
    pooled = key_pool_reference(
        hidden,
        wk,
        gate_weight,
        ape,
        state_cache,
        state_block_table,
        torch.tensor(start_pos, dtype=torch.int32, device=dev),
        cu_seqlens=cu_seqlens,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        norm_eps=norm_eps,
        cmp_ratio=cmp_ratio,
    )
    return pooled, hidden


def test_key_pool_reference_basic_single_request_complete_pools():
    head_dim, hidden_size, kpool = 8, 16, 4
    torch.manual_seed(0)
    dev = "cpu"
    state_cache = torch.zeros(4, kpool, 2 * head_dim, dtype=torch.float32, device=dev)
    state_block_table = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32, device=dev)
    wk = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    gate_weight = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(kpool, head_dim, dtype=torch.float32, device=dev) * 0.1
    # Sequence of 7 tokens starting at position 0: the first pool (0..3) is
    # completed inside this chunk, the second pool (4..7) is not.
    pooled, hidden = _run_key_pool_once(
        chunk_lens=[7],
        start_pos=[0],
        state_cache=state_cache,
        state_block_table=state_block_table,
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
    )
    # Output capacity follows the padded block-table width (4 pages).
    assert pooled.shape == (1, 4, head_dim)

    # Manually recompute the completed pool row.
    k = hidden[:4] @ wk.t().to(hidden.dtype)
    gate = hidden[:4] @ gate_weight.t().to(hidden.dtype)
    scores = (gate.float() + ape).softmax(dim=0)
    expected = (scores * k.float()).sum(dim=0).to(hidden.dtype)
    torch.testing.assert_close(pooled[0, 0], expected, rtol=1e-2, atol=1e-2)

    # The incomplete pool's tail (positions 4..6) must be persisted into the
    # FP32 state cache at block 2 (page 1) offsets 0..2.
    k5 = hidden[5] @ wk.t().to(hidden.dtype)
    g5 = hidden[5] @ gate_weight.t().to(hidden.dtype)
    torch.testing.assert_close(
        state_cache[2, 1, :head_dim].to(hidden.dtype),
        k5,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        state_cache[2, 1, head_dim:].to(hidden.dtype),
        g5,
        rtol=1e-2,
        atol=1e-2,
    )


def test_key_pool_reference_cross_chunk_equivalence():
    """chunked prefill must equal one-shot prefill (plan §9.2)."""
    head_dim, hidden_size, kpool = 8, 16, 4
    torch.manual_seed(1)
    dev = "cpu"
    seq_len = 13  # 3 complete pools + a 1-token tail
    hidden = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=dev) * 0.1
    num_blocks = 5  # 4 logical pages map to state blocks 1..4 (+ dummy 0)

    def fresh_caches() -> torch.Tensor:
        return torch.zeros(num_blocks, kpool, 2 * head_dim, dtype=torch.float32, device=dev)

    def make_table() -> torch.Tensor:
        width = (seq_len + kpool - 1) // kpool
        return torch.tensor([list(range(1, width + 1)) + [0] * (8 - width)], dtype=torch.int32, device=dev)

    wk = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    gate_weight = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(kpool, head_dim, dtype=torch.float32, device=dev) * 0.1
    norm_weight = torch.randn(head_dim, dtype=torch.float32, device=dev)
    norm_bias = torch.randn(head_dim, dtype=torch.float32, device=dev)

    # One-shot prefill.
    pooled_one, _ = _run_key_pool_once(
        chunk_lens=[seq_len],
        start_pos=[0],
        state_cache=fresh_caches(),
        state_block_table=make_table(),
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
        hidden=hidden,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
    )

    # Chunked prefill: [0..5), [5..9), [9..13). Pool 0 completes in chunk 0
    # (rows placed at relative row pool-first_pool), pools 1..2 in later
    # chunks; accumulating by absolute pool row reproduces the one-shot
    # buffer. The incomplete tail of pool 3 lives in the state cache.
    state_chunk = fresh_caches()
    pooled_chunk = torch.zeros_like(pooled_one, dtype=torch.float32)
    for start, length in ((0, 5), (5, 4), (9, 4)):
        p, _ = _run_key_pool_once(
            chunk_lens=[length],
            start_pos=[start],
            state_cache=state_chunk,
            state_block_table=make_table(),
            wk=wk,
            gate_weight=gate_weight,
            ape=ape,
            cmp_ratio=kpool,
            hidden=hidden[start : start + length],
            norm_weight=norm_weight,
            norm_bias=norm_bias,
        )
        # Each chunk's rows are relative to its own first pool; place them at
        # their absolute pool rows. Accumulate in FP32: a per-chunk output is
        # BF16-rounded, so a BF16 accumulation would add ulp-level rounding.
        first_pool = start // kpool
        width = pooled_chunk.shape[1] - first_pool
        pooled_chunk[:, first_pool:] += p.float()[:, :width]

    torch.testing.assert_close(pooled_chunk, pooled_one.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("start_pos", [0, 3, 4, 5])
def test_key_pool_reference_start_pos_boundaries(start_pos: int):
    """start_pos = 0, r-1, r, r+1 (plan §9.2)."""
    head_dim, hidden_size, kpool = 8, 16, 4
    torch.manual_seed(2)
    dev = "cpu"
    chunk_len = 5
    seq_len = start_pos + chunk_len
    width = (seq_len + kpool - 1) // kpool
    state_cache = torch.zeros(4, kpool, 2 * head_dim, dtype=torch.float32, device=dev)
    state_block_table = torch.tensor(
        [list(range(1, width + 1))],
        dtype=torch.int32,
        device=dev,
    )
    wk = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    gate_weight = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(kpool, head_dim, dtype=torch.float32, device=dev) * 0.1
    # Prefill history so the tail state is non-empty for start_pos > 0.
    if start_pos > 0:
        hist = torch.randn(start_pos, hidden_size, dtype=torch.bfloat16, device=dev) * 0.1
        _run_key_pool_once(
            chunk_lens=[start_pos],
            start_pos=[0],
            state_cache=state_cache,
            state_block_table=state_block_table,
            wk=wk,
            gate_weight=gate_weight,
            ape=ape,
            cmp_ratio=kpool,
            hidden=hist,
        )
    pooled, _ = _run_key_pool_once(
        chunk_lens=[chunk_len],
        start_pos=[start_pos],
        state_cache=state_cache,
        state_block_table=state_block_table,
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
    )
    completed = (start_pos + chunk_len) // kpool
    assert completed >= 1
    assert pooled.shape == (1, width, head_dim)
    assert not torch.isnan(pooled).any()


def test_key_pool_reference_block_table_sentinel_and_out_of_order():
    """State block 0 must be a no-op; tables may be out of order (plan §9.2)."""
    head_dim, hidden_size, kpool = 8, 16, 4
    torch.manual_seed(3)
    dev = "cpu"
    state_cache = torch.zeros(4, kpool, 2 * head_dim, dtype=torch.float32, device=dev)
    # Row order deliberately shuffled: request pages map to blocks [3, 1, 0(invalid)].
    state_block_table = torch.tensor([[3, 1, 0]], dtype=torch.int32, device=dev)
    wk = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    gate_weight = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    ape = torch.zeros(kpool, head_dim, dtype=torch.float32, device=dev)

    # 9 tokens: pools 0 and 1 complete; token 8 lands in the page whose table
    # entry is 0 (invalid) and must not be written to state block 0.
    pooled, hidden = _run_key_pool_once(
        chunk_lens=[9],
        start_pos=[0],
        state_cache=state_cache,
        state_block_table=state_block_table,
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
    )
    k = hidden @ wk.t().to(hidden.dtype)
    gate = hidden @ gate_weight.t().to(hidden.dtype)
    # Pool 0 uses block 3 at offsets 0..3; recompute directly (softmax over
    # the gate scores, weighted K sum).
    expected0 = (gate[:4].float().softmax(dim=0) * k[:4].float()).sum(0).to(hidden.dtype)
    torch.testing.assert_close(pooled[0, 0], expected0, rtol=1e-2, atol=1e-2)
    assert torch.count_nonzero(state_cache[0]).item() == 0


def test_key_pool_reference_state_cache_noncontiguous_stride0():
    """The state cache supports a non-contiguous 0-axis (plan §9.2); the
    framework forwards stride(0) to the op, the reference must too."""
    head_dim, hidden_size, kpool = 8, 16, 4
    torch.manual_seed(4)
    dev = "cpu"
    # Storage with an extra leading page: the real cache is an as_strided view
    # whose stride(0) is larger than the contiguous stride.
    raw = torch.zeros(5, kpool, 2 * head_dim, dtype=torch.float32, device=dev)
    state_cache = torch.as_strided(
        raw,
        (4, kpool, 2 * head_dim),
        (kpool * 2 * head_dim + 2, 2 * head_dim, 1),
    )
    assert state_cache.stride(0) > kpool * 2 * head_dim
    state_block_table = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32, device=dev)
    wk = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    gate_weight = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    ape = torch.zeros(kpool, head_dim, dtype=torch.float32, device=dev)
    _run_key_pool_once(
        chunk_lens=[7],
        start_pos=[0],
        state_cache=state_cache,
        state_block_table=state_block_table,
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
    )
    # The incomplete pool's tail (positions 4..6) must land in view block 2
    # (page 1 -> table id 2) of the as_strided cache: exactly 3 [K, gate]
    # rows, i.e. 3*2*head_dim non-zero elements anywhere in the storage.
    assert torch.count_nonzero(raw).item() == 3 * 2 * head_dim
    assert torch.count_nonzero(state_cache[2, 2]).item() == 2 * head_dim


@pytest.mark.parametrize("dtype", TORCH_DTYPES)
def test_key_pool_reference_supports_bf16_and_fp16(dtype: torch.dtype):
    head_dim, hidden_size, kpool = 8, 16, 4
    torch.manual_seed(5)
    dev = "cpu"
    state_cache = torch.zeros(2, kpool, 2 * head_dim, dtype=torch.float32, device=dev)
    state_block_table = torch.tensor([[1]], dtype=torch.int32, device=dev)
    wk = torch.randn(head_dim, hidden_size, dtype=dtype, device=dev)
    gate_weight = torch.randn(head_dim, hidden_size, dtype=dtype, device=dev)
    ape = torch.zeros(kpool, head_dim, dtype=torch.float32, device=dev)
    pooled, _ = _run_key_pool_once(
        chunk_lens=[4],
        start_pos=[0],
        state_cache=state_cache,
        state_block_table=state_block_table,
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
        dtype=dtype,
    )
    assert pooled.dtype == dtype
    assert pooled.shape == (1, 1, head_dim)


def test_key_pool_reference_layernorm_toggle():
    head_dim, hidden_size, kpool = 8, 16, 4
    torch.manual_seed(6)
    dev = "cpu"
    state_cache = torch.zeros(2, kpool, 2 * head_dim, dtype=torch.float32, device=dev)
    state_block_table = torch.tensor([[1, 0]], dtype=torch.int32, device=dev)
    wk = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    gate_weight = torch.randn(head_dim, hidden_size, dtype=torch.bfloat16, device=dev)
    ape = torch.zeros(kpool, head_dim, dtype=torch.float32, device=dev)
    norm_weight = torch.randn(head_dim, dtype=torch.float32, device=dev)
    norm_bias = torch.randn(head_dim, dtype=torch.float32, device=dev)
    hidden = torch.randn(4, hidden_size, dtype=torch.bfloat16, device=dev) * 0.1

    pooled_ln, _ = _run_key_pool_once(
        chunk_lens=[4],
        start_pos=[0],
        state_cache=state_cache.clone(),
        state_block_table=state_block_table,
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
        hidden=hidden,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
    )
    pooled_no, _ = _run_key_pool_once(
        chunk_lens=[4],
        start_pos=[0],
        state_cache=state_cache.clone(),
        state_block_table=state_block_table,
        wk=wk,
        gate_weight=gate_weight,
        ape=ape,
        cmp_ratio=kpool,
        hidden=hidden,
        norm_weight=None,
        norm_bias=None,
    )
    k = hidden @ wk.t().to(hidden.dtype)
    gate = hidden @ gate_weight.t().to(hidden.dtype)
    ln_k = torch.nn.functional.layer_norm(k.float(), (head_dim,), norm_weight, norm_bias, 1e-6).to(hidden.dtype)
    ref_no_ln = (gate[:4].float().softmax(dim=0) * k[:4].float()).sum(0).to(hidden.dtype)
    ref_ln = (gate[:4].float().softmax(dim=0) * ln_k[:4].float()).sum(0).to(hidden.dtype)
    torch.testing.assert_close(pooled_no[0, 0], ref_no_ln, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pooled_ln[0, 0], ref_ln, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# _C_ascend schema / Meta contract (requires the built extension)
# ---------------------------------------------------------------------------


def _key_pool_schema():
    return torch._C._dispatch_find_schema_or_throw("_C_ascend::key_pool", "")


def test_key_pool_schema_abi_matches_cann_wrapper():
    ensure_c_ascend_loaded()
    text = str(_key_pool_schema())
    # Frozen ABI from torch_extension/cann_ops_transformer/ops/key_pool.py
    # (plan §3): required tensors + keyword-only optional attrs.
    assert "Tensor hidden_states" in text
    assert "Tensor wk" in text
    assert "Tensor gate_weight" in text
    assert "Tensor ape" in text
    assert "Tensor(a!) state_cache" in text  # mutation alias
    assert "Tensor cache_block_table" in text
    assert "Tensor start_pos" in text
    assert "Tensor? norm_weight" in text
    assert "Tensor? norm_bias" in text
    assert "Tensor? cos" in text
    assert "Tensor? sin" in text
    assert "Tensor? cu_seqlens" in text
    assert "Tensor? seqused" in text
    assert re.search(r"int cmp_ratio=4", text)
    # The schema parser normalizes float defaults (1e-6 prints as
    # 9.9999999999999995e-07 on some torch versions), so only the name/type
    # is asserted here.
    assert "float norm_eps=" in text
    assert re.search(r"int rotary_mode=1", text)
    assert "-> Tensor" in text


def test_key_pool_meta_shapes_match_contract():
    """Meta output must be [B, ceil(cols * state_block_size / cmp_ratio), D]
    with the same dtype as hidden_states (plan §4.2)."""
    ensure_c_ascend_loaded()
    hidden = torch.empty(6, 16, dtype=torch.bfloat16, device="meta")
    wk = torch.empty(8, 16, dtype=torch.bfloat16, device="meta")
    gate = torch.empty(8, 16, dtype=torch.bfloat16, device="meta")
    ape = torch.empty(4, 8, dtype=torch.float32, device="meta")
    state = torch.empty(3, 4, 16, dtype=torch.float32, device="meta")
    table = torch.empty(2, 4, dtype=torch.int32, device="meta")
    start = torch.empty(2, dtype=torch.int32, device="meta")
    cu = torch.empty(3, dtype=torch.int32, device="meta")

    pooled = torch.ops._C_ascend.key_pool(
        hidden,
        wk,
        gate,
        ape,
        state,
        table,
        start,
        norm_weight=None,
        norm_bias=None,
        cos=None,
        sin=None,
        cu_seqlens=cu,
        seqused=None,
        cmp_ratio=4,
        norm_eps=1e-6,
        rotary_mode=1,
    )
    assert pooled.shape == (2, 4, 8)
    assert pooled.dtype == torch.bfloat16


def test_key_pool_meta_rejects_mismatched_norm_pair():
    ensure_c_ascend_loaded()
    hidden = torch.empty(6, 16, dtype=torch.bfloat16, device="meta")
    wk = torch.empty(8, 16, dtype=torch.bfloat16, device="meta")
    gate = torch.empty(8, 16, dtype=torch.bfloat16, device="meta")
    ape = torch.empty(4, 8, dtype=torch.float32, device="meta")
    state = torch.empty(3, 4, 16, dtype=torch.float32, device="meta")
    table = torch.empty(2, 4, dtype=torch.int32, device="meta")
    start = torch.empty(2, dtype=torch.int32, device="meta")
    norm_weight = torch.empty(8, dtype=torch.float32, device="meta")

    with pytest.raises(RuntimeError, match="pair"):
        torch.ops._C_ascend.key_pool(
            hidden,
            wk,
            gate,
            ape,
            state,
            table,
            start,
            norm_weight=norm_weight,
            norm_bias=None,
            cos=None,
            sin=None,
            cu_seqlens=None,
            seqused=None,
            cmp_ratio=4,
            norm_eps=1e-6,
            rotary_mode=1,
        )
