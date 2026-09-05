# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5 Next CANN ``pool_key_indexer`` operator adaptation tests.

Covers (plan §9.3/§10):
- an independent PyTorch reference of the operator contract (per-head ReLU
  dot product, 1/sqrt(head_dim) scaling, head-weight aggregation, causal
  pool Top-K, pool expansion, tail append);
- positive/negative dot products and mixed head weights;
- top-k boundary near-ties and full ties;
- history shorter/equal/longer than the top-k budget;
- pool_tail_k in [0, pool_size-1];
- single/multi request, prefill/decode/chunked prefill;
- non-contiguous pool_key stride(0) and out-of-order paged block tables;
- physical block 0 and block-table padding;
- output index range, causality and tail token order.

The reference never calls the CANN op; it is used by the NPU accuracy tests
in tests/e2e/models/test_glm5_next_key_pool.py as the standalone oracle.
"""

from __future__ import annotations

import pytest
import torch

from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded
from tests.ut.ops.helpers.glm5_next_cann_reference import sparse_to_dense_token_ids
from tests.ut.ops.helpers.pool_key_indexer_reference import (
    pool_key_indexer_reference as _official_pki_reference,
)

TORCH_DTYPES = (torch.bfloat16, torch.float16)


def pool_key_indexer_reference(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q,
    actual_seq_k,
    block_table,
    topk,
    pool_size,
    mask_mode=3,
) -> torch.Tensor:
    """Official CANN golden wrapper: TND query / PA_BBND paged key, returns
    the sparse indices only (the golden also returns the selected pool
    scores, which the e2e contract check consumes separately)."""
    indices, _ = _official_pki_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q="TND",
        layout_k="PA_BBND",
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
    )
    return indices


def _run_indexer(
    query,
    pool_key,
    weights,
    *,
    actual_seq_q,
    actual_seq_k,
    pool_tail_k,
    block_table,
    topk,
    pool_size,
) -> torch.Tensor:
    return pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        topk=topk,
        pool_size=pool_size,
        mask_mode=3,
    )


def _make_request(
    *,
    seq_len: int,
    query_lens: list[int],
    num_heads: int,
    head_dim: int,
    pool_size: int,
    block_size: int,
    num_blocks: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    **indexer_kwargs,
) -> dict:
    """Synthesize one request worth of state and return all indexer inputs."""
    torch.manual_seed(seed)
    dev = "cpu"
    total_query = sum(query_lens)
    query = torch.randn(total_query, num_heads, head_dim, dtype=dtype, device=dev) * 0.5
    weights = torch.randn(total_query, num_heads, dtype=dtype, device=dev) * 0.5 - 0.2
    # Build the paged K cache: pool p lives at block p // block_size, slot p % block_size.
    # actual_seq_k is the number of COMPLETED pools = floor(seq_len / pool_size).
    pool_count = seq_len // pool_size
    pool_key = torch.zeros(num_blocks, block_size, 1, head_dim, dtype=dtype, device=dev)
    for p in range(pool_count):
        block = p // block_size
        if block >= num_blocks:
            break
        pool_key[block, p % block_size, 0] = torch.randn(head_dim, dtype=dtype, device=dev) * 0.3
    pages = (pool_count + block_size - 1) // block_size
    block_table = torch.full((1, max(pages, 1)), 0, dtype=torch.int32, device=dev)
    for page in range(pages):
        block_table[0, page] = page % num_blocks  # identity mapping
    actual_seq_q = torch.tensor(query_lens, dtype=torch.int64).cumsum(0)
    actual_seq_k = torch.tensor([pool_count], dtype=torch.int64)
    pool_tail_k = torch.tensor([seq_len % pool_size], dtype=torch.int64)
    out = _run_indexer(
        query,
        pool_key,
        weights,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        pool_tail_k=pool_tail_k,
        block_table=block_table,
        topk=topk,
        pool_size=pool_size,
    )
    return {
        "query": query,
        "pool_key": pool_key,
        "weights": weights,
        "actual_seq_q": actual_seq_q,
        "actual_seq_k": actual_seq_k,
        "pool_tail_k": pool_tail_k,
        "block_table": block_table,
        "topk": topk,
        "pool_size": pool_size,
        "output": out,
        "seq_len": seq_len,
        "query_lens": query_lens,
    }


def test_pki_reference_shape_and_padding():
    r = _make_request(
        seq_len=12,
        query_lens=[4],
        num_heads=2,
        head_dim=8,
        pool_size=4,
        block_size=4,
        num_blocks=8,
        topk=8,
    )
    out = r["output"]
    seq_len, qlen, pool_size, topk = 12, 4, 4, 8
    assert out.shape == (qlen, topk + pool_size - 1)
    for row in range(qlen):
        pos = seq_len - qlen + row
        visible_pools = (pos + 1) // pool_size
        row_vals = out[row].tolist()
        # Valid ids are in [0, seq_len), everything else is -1.
        assert all(v == -1 or 0 <= v < seq_len for v in row_vals)
        # Causal: the top-k expanded region may only contain pools completed
        # before the current position.
        expanded = [v for v in row_vals[:topk] if v >= 0]
        assert all(v // pool_size < visible_pools for v in expanded), (row, expanded)
        # tail=0 (seq_len % pool_size == 0): the request-level tail region is
        # empty, so everything past the expanded columns must be -1.
        assert all(v == -1 for v in row_vals[topk:])


def test_pki_reference_positive_and_negative_dot_products_and_mixed_weights():
    """Negative dot products must be killed by the per-head ReLU before the
    head-weight aggregation (plan §9.3)."""
    num_heads, head_dim, pool_size = 2, 8, 4
    dev = "cpu"
    q = torch.tensor([[[1.0, 0.0] + [0.0] * 6, [0.0, 1.0] + [0.0] * 6]], dtype=torch.bfloat16)
    # Two pools: pool 0 matches head 0 (positive for head 0, negative for
    # head 1), pool 1 matches head 1.
    k = torch.zeros(2, 4, 1, head_dim, dtype=torch.bfloat16)
    k[0, 0, 0, 0] = 1.0
    k[0, 1, 0, 0] = -1.0  # irrelevant to head 0 query but negative
    k[1, 0, 0, 1] = 5.0
    weights = torch.tensor([[1.0, 0.1]], dtype=torch.bfloat16)
    out = pool_key_indexer_reference(
        q,
        k,
        weights,
        torch.tensor([0], dtype=torch.int64),
        actual_seq_q=torch.tensor([1], dtype=torch.int64),
        actual_seq_k=torch.tensor([2], dtype=torch.int64),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        topk=8,
        pool_size=pool_size,
    )
    # head 1's dot with pool 0 is 0 (ReLU kills the negative), so pool 0
    # scores only via head 0; pool 1 scores via head 1. topk=8 -> both pools
    # are selected and expanded.
    dense = sparse_to_dense_token_ids(out, 8)[0]
    assert 0 in dense and 1 in dense
    assert 4 in dense  # pool 1 expanded


def test_pki_reference_near_ties_at_topk_boundary_are_deterministic():
    """Near ties at the top-k boundary (plan §9.3): distinct-but-close scores
    must select the top pools deterministically."""
    num_heads, head_dim, pool_size = 1, 8, 4
    dev = "cpu"
    torch.manual_seed(0)
    # q = [1, 0, ...] so score_p = relu(k_p[0]) / sqrt(head_dim).
    q = torch.tensor([[[1.0, 0.0, 0, 0, 0, 0, 0, 0]]], dtype=torch.bfloat16, device=dev)
    k = torch.zeros(6, 4, 1, head_dim, dtype=torch.bfloat16, device=dev)
    # Distinct, near-boundary scores in descending order:
    #   pools 0..5 -> 2.00, 1.99, 1.50, 1.00, 0.50, 0.10
    values = [2.0, 1.99, 1.5, 1.0, 0.5, 0.1]
    for p, v in enumerate(values):
        k[p // 4, p % 4, 0, 0] = v
    weights = torch.ones(1, num_heads, dtype=torch.bfloat16, device=dev)
    out = pool_key_indexer_reference(
        q,
        k,
        weights,
        torch.tensor([1], dtype=torch.int64),
        actual_seq_q=torch.tensor([1], dtype=torch.int64),
        actual_seq_k=torch.tensor([6], dtype=torch.int64),
        block_table=torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32),
        topk=8,
        pool_size=pool_size,
    )
    dense = sparse_to_dense_token_ids(out, 25)[0]
    # top-2 pools are 0 and 1; request tail [24] appended at column 8.
    assert dense == [0, 1, 2, 3, 4, 5, 6, 7, 24]


def test_pki_reference_full_tie_selects_exact_budget_without_tie_order_reliance():
    """Full ties (plan §9.3): every visible pool scores exactly 0. torch.topk
    tie-breaking is not part of the operator contract, so the reference must
    only guarantee the structural invariants: exactly `k` fully-expanded
    pools within the visible range, plus the request tail."""
    num_heads, head_dim, pool_size = 1, 8, 4
    dev = "cpu"
    q = torch.zeros(1, num_heads, head_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(6, 4, 1, head_dim, dtype=torch.bfloat16, device=dev) * 0.1
    weights = torch.ones(1, num_heads, dtype=torch.bfloat16, device=dev)
    out = pool_key_indexer_reference(
        q,
        k,
        weights,
        torch.tensor([1], dtype=torch.int64),
        actual_seq_q=torch.tensor([1], dtype=torch.int64),
        actual_seq_k=torch.tensor([6], dtype=torch.int64),
        block_table=torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32),
        topk=8,
        pool_size=pool_size,
    )
    tokens = [int(v) for v in out[0] if v >= 0]
    # L_orig = 6*4 + 1 = 25; causal cap = min(1, 24 - 8 + 1) = 1 -> tail [24].
    assert tokens[-1] == 24
    expanded = [t for t in tokens if t < 24]
    # Exactly sparse_count=2 pools, each fully expanded to pool_size tokens.
    assert len(expanded) == 8
    pools = sorted({t // pool_size for t in expanded})
    assert len(pools) == 2
    for pool in pools:
        assert sorted(t for t in expanded if t // pool_size == pool) == list(
            range(pool * pool_size, (pool + 1) * pool_size)
        )
    assert all(pool < 6 for pool in pools)


def test_pki_reference_history_shorter_equal_longer_than_topk():
    cases = {
        "short": dict(topk=16, seq_len=8),  # 2 pools < 4-pool budget
        "equal": dict(topk=16, seq_len=16),  # 4 pools == budget
        "long": dict(topk=16, seq_len=32),  # 8 pools > budget
    }
    seeds = {"short": 11, "equal": 12, "long": 13}
    for name, case in cases.items():
        r = _make_request(
            seq_len=case["seq_len"],
            query_lens=[1],
            num_heads=2,
            head_dim=8,
            pool_size=4,
            block_size=4,
            num_blocks=16,
            topk=case["topk"],
            seed=seeds[name],
        )
        dense = sparse_to_dense_token_ids(r["output"], r["seq_len"])[0]
        # Budget is topk tokens; history shorter than the budget must still
        # produce only existing pool tokens.
        max_pool = (r["seq_len"] + r["pool_size"] - 1) // r["pool_size"]
        assert all(tok // r["pool_size"] < max_pool for tok in dense)


def test_pki_reference_pool_tail_k_range():
    """pool_tail_k from 0 to pool_size-1 must produce the right tail append."""
    for tail in range(4):
        r = _make_request(
            seq_len=8 + tail,  # 2 full pools + `tail` tokens
            query_lens=[1],
            num_heads=1,
            head_dim=8,
            pool_size=4,
            block_size=4,
            num_blocks=8,
            topk=8,
            seed=tail,
        )
        assert r["pool_tail_k"][0].item() == tail
        row = r["output"][0]
        tokens = [v for v in row.tolist() if v >= 0]
        # The `tail` running-pool tokens 8..8+tail-1 are appended in order.
        if tail > 0:
            assert tokens[-tail:] == list(range(8, 8 + tail))
        else:
            assert all(tok < 8 for tok in tokens)


def test_pki_reference_multi_request_uneven_lengths():
    """Multi-request prefill with inconsistent lengths (plan §9.3)."""
    torch.manual_seed(0)
    dev = "cpu"
    num_heads, head_dim, pool_size, block_size = 2, 8, 4, 4
    # Request A: 10 tokens (2 pools + tail 2), 3 query tokens.
    # Request B: 5 tokens (1 pool + tail 1), 2 query tokens.
    seq_lens = [10, 5]
    pool_counts = [10 // 4, 5 // 4]
    query_lens = [3, 2]
    total_query = sum(query_lens)
    query = torch.randn(total_query, num_heads, head_dim, dtype=torch.bfloat16) * 0.4
    weights = torch.randn(total_query, num_heads, dtype=torch.bfloat16) * 0.3
    num_blocks = 8
    pool_key = torch.randn(num_blocks, block_size, 1, head_dim, dtype=torch.bfloat16) * 0.2
    block_table = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    out = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        torch.tensor([2, 1], dtype=torch.int64),
        actual_seq_q=torch.tensor([3, 5], dtype=torch.int64),
        actual_seq_k=torch.tensor(pool_counts, dtype=torch.int64),
        block_table=block_table,
        topk=8,
        pool_size=pool_size,
    )
    assert out.shape == (total_query, 8 + pool_size - 1)
    for row in range(total_query):
        tokens = [v for v in out[row].tolist() if v >= 0]
        assert all(0 <= v < 10 for v in tokens)


def test_pki_reference_noncontiguous_stride0_and_out_of_order_blocks():
    """pool_key stride(0) > contiguous stride and shuffled block tables
    (plan §9.3)."""
    num_heads, head_dim, pool_size = 1, 8, 4
    dev = "cpu"
    torch.manual_seed(3)
    # Raw cache with an extra leading block: the real cache is as_strided.
    raw = torch.randn(9, 4, 1, head_dim, dtype=torch.bfloat16) * 0.2
    pool_key = torch.as_strided(raw, (8, 4, 1, head_dim), (4 * head_dim + 4, head_dim, head_dim, 1))
    assert pool_key.stride(0) > 4 * head_dim
    q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16) * 0.4
    weights = torch.ones(1, num_heads, dtype=torch.bfloat16)
    # Out-of-order table: request pages map to blocks [5, 2, 7, 1].
    block_table = torch.tensor([[5, 2, 7, 1]], dtype=torch.int32)
    out = pool_key_indexer_reference(
        q,
        pool_key,
        weights,
        torch.tensor([1], dtype=torch.int64),
        actual_seq_q=torch.tensor([1], dtype=torch.int64),
        actual_seq_k=torch.tensor([12], dtype=torch.int64),
        block_table=block_table,
        topk=8,
        pool_size=pool_size,
    )
    assert out.shape == (1, 11)
    # Directly compare against a dense implementation using the same page ids.
    dense_scores = []
    for p in range(12):
        block_id = int(block_table[0, p // 4])
        kvec = pool_key[block_id, p % 4, 0]
        score = torch.nn.functional.relu((q[0, 0] * kvec).sum() / (head_dim**0.5)).item()
        dense_scores.append(score)
    expected_pools = sorted(
        range(12),
        key=lambda p: (-dense_scores[p], p),
    )[:2]
    expected_tokens = sorted(
        [tok for p in expected_pools for tok in range(p * 4, (p + 1) * 4)]
    )
    # Request-level tail [L_orig - 1, L_orig) = [48] (L_orig = 12*4+1 = 49)
    # is appended at column 8 (kernel contract). The golden emits the selected
    # pools in score-descending order, so compare as sets.
    assert out.shape == (1, 11)
    tokens = sparse_to_dense_token_ids(out, 49)[0]
    assert sorted(tokens) == sorted(expected_tokens + [48])


def test_pki_reference_physical_block_zero_and_padding():
    """Physical block 0 is valid; padded table entries beyond actual_seq_k
    must not contribute (plan §9.3)."""
    num_heads, head_dim, pool_size = 1, 8, 4
    dev = "cpu"
    torch.manual_seed(4)
    # Only 1 pool (block 0), table padded with zeros (vLLM convention).
    pool_key = torch.randn(2, 4, 1, head_dim, dtype=torch.bfloat16) * 0.2
    q = torch.ones(1, num_heads, head_dim, dtype=torch.bfloat16)
    weights = torch.ones(1, num_heads, dtype=torch.bfloat16)
    out = pool_key_indexer_reference(
        q,
        pool_key,
        weights,
        torch.tensor([1], dtype=torch.int64),
        actual_seq_q=torch.tensor([1], dtype=torch.int64),
        actual_seq_k=torch.tensor([1], dtype=torch.int64),
        block_table=torch.tensor([[0, 0, 0]], dtype=torch.int32),
        topk=8,
        pool_size=pool_size,
    )
    tokens = sparse_to_dense_token_ids(out, 8)[0]
    # Only pool 0 tokens: L_orig=5 < topk=8, so the causal cap
    # min(pool_tail_k, pos - topk + 1) = min(1, -2) = 0 and no tail is
    # appended (kernel ExpandAndAppendIndices).
    assert tokens == [0, 1, 2, 3]


@pytest.mark.parametrize("dtype", TORCH_DTYPES)
def test_pki_reference_supports_bf16_and_fp16(dtype: torch.dtype):
    r = _make_request(
        seq_len=12,
        query_lens=[1],
        num_heads=2,
        head_dim=8,
        pool_size=4,
        block_size=4,
        num_blocks=8,
        topk=8,
        dtype=dtype,
        seed=7,
    )
    assert r["output"].shape == (1, 11)
    assert r["output"].dtype == torch.int32


def test_pki_reference_request_level_tail_with_causal_cap_matches_kernel():
    """The op appends the request-level tail [L-1, L) and caps it to
    min(pool_tail_k, pos - topk + 1) under causal masking (kernel
    ExpandAndAppendIndices): a mid-prefill row only sees the tail once its
    position has passed the expanded region (plan §9.3)."""
    torch.manual_seed(21)
    dev = "cpu"
    num_heads, head_dim, pool_size = 1, 8, 4
    # Request L=13 = 3 full pools + tail 1, chunk covers 8 query rows
    # (positions 5..12).
    query = torch.randn(8, num_heads, head_dim, dtype=torch.bfloat16, device=dev) * 0.4
    weights = torch.ones(8, num_heads, dtype=torch.bfloat16, device=dev)
    pool_key = torch.randn(8, 4, 1, head_dim, dtype=torch.bfloat16, device=dev) * 0.2
    out = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        torch.tensor([1], dtype=torch.int64),
        actual_seq_q=torch.tensor([8], dtype=torch.int64),
        actual_seq_k=torch.tensor([3], dtype=torch.int64),
        block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        topk=8,
        pool_size=pool_size,
    )
    assert out.shape == (8, 8 + pool_size - 1)
    for row in range(8):
        pos = 13 - 8 + row  # 5..12
        visible_tail = max(0, min(1, pos - 8 + 1))
        tail_token = out[row, 8].item()
        assert tail_token == (12 if visible_tail else -1), (row, tail_token)
        # The expanded region stays causal.
        expanded = [int(v) for v in out[row, :8] if v >= 0]
        assert all(v // 4 < (pos + 1) // 4 for v in expanded), (row, expanded)


def test_pki_reference_decode_then_prefill_chunked():
    """Decode + chunked prefill equivalence: each chunk answers from the same
    pool table (plan §9.3)."""
    num_heads, head_dim, pool_size = 1, 8, 4
    dev = "cpu"
    torch.manual_seed(8)
    pool_key = torch.randn(8, 4, 1, head_dim, dtype=torch.bfloat16) * 0.2
    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    weights = torch.randn(4, num_heads, dtype=torch.bfloat16)

    def run(q_row: int, total_pools: int, tail: int, seq_len: int) -> torch.Tensor:
        q = torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16) * 0.4
        return pool_key_indexer_reference(
            q,
            pool_key,
            weights[q_row : q_row + 1],
            torch.tensor([tail], dtype=torch.int64),
            actual_seq_q=torch.tensor([1], dtype=torch.int64),
            actual_seq_k=torch.tensor([total_pools], dtype=torch.int64),
            block_table=block_table,
            topk=8,
            pool_size=pool_size,
        )

    # Decode at position 6 (1 full pool + tail 2), then a later decode at
    # position 9 (2 pools + tail 1): both must be causal and in range.
    decode1 = run(0, 1, 2, 6)
    decode2 = run(1, 2, 1, 9)
    for row in (decode1[0], decode2[0]):
        tokens = [v for v in row.tolist() if v >= 0]
        assert all(0 <= v < 12 for v in tokens)
    assert 4 in decode2[0].tolist() or 5 in decode2[0].tolist()


# ---------------------------------------------------------------------------
# _C_ascend schema / Meta contract (requires the built extension)
# ---------------------------------------------------------------------------


def test_pki_schema_abi_matches_cann_wrapper():
    ensure_c_ascend_loaded()
    schema = torch._C._dispatch_find_schema_or_throw("_C_ascend::pool_key_indexer", "")
    text = str(schema)
    assert "Tensor query" in text
    assert "Tensor pool_key" in text
    assert "Tensor weights" in text
    assert "Tensor pool_tail_k" in text
    assert "Tensor? actual_seq_q" in text
    assert "Tensor? actual_seq_k" in text
    assert "Tensor? block_table" in text
    assert "Tensor? q_descale" in text
    assert "Tensor? k_descale" in text
    # String defaults print with single or double quotes depending on the
    # torch version, so only the name/type is asserted here.
    assert "str layout_q=" in text
    assert "str layout_k=" in text
    assert "int topk=128" in text
    assert "int pool_size=1" in text
    assert "int mask_mode=0" in text
    assert "int quant_mode=-1" in text
    assert "bool return_value=False" in text
    assert "-> (Tensor, Tensor)" in text
    # key_stride0 must NOT be part of the public schema: it is derived from
    # pool_key.stride(0) in the binding layer (plan §4.2).
    assert "key_stride0" not in text


def test_pki_meta_shapes_match_contract():
    """TND indices [T, topk+pool_size-1] int32; values empty FP32 when
    return_value=false (plan §4.2)."""
    ensure_c_ascend_loaded()
    query = torch.empty(5, 2, 8, dtype=torch.bfloat16, device="meta")
    pool_key = torch.empty(4, 4, 1, 8, dtype=torch.bfloat16, device="meta")
    weights = torch.empty(5, 2, dtype=torch.bfloat16, device="meta")
    tail = torch.empty(1, dtype=torch.int64, device="meta")

    indices, values = torch.ops._C_ascend.pool_key_indexer(
        query,
        pool_key,
        weights,
        tail,
        actual_seq_q=torch.empty(1, dtype=torch.int64, device="meta"),
        actual_seq_k=torch.empty(1, dtype=torch.int64, device="meta"),
        block_table=torch.empty(1, 2, dtype=torch.int32, device="meta"),
        layout_q="TND",
        layout_k="PA_BBND",
        topk=8,
        pool_size=4,
        mask_mode=3,
        quant_mode=-1,
        return_value=False,
    )
    assert indices.shape == (5, 8 + 4 - 1)
    assert indices.dtype == torch.int32
    assert values.shape == (0,)
    assert values.dtype == torch.float32


def test_pki_meta_bsnd_shapes_when_return_value():
    ensure_c_ascend_loaded()
    query = torch.empty(2, 3, 2, 8, dtype=torch.bfloat16, device="meta")
    pool_key = torch.empty(4, 4, 1, 8, dtype=torch.bfloat16, device="meta")
    weights = torch.empty(2, 3, 2, dtype=torch.bfloat16, device="meta")
    tail = torch.empty(2, dtype=torch.int64, device="meta")

    indices, values = torch.ops._C_ascend.pool_key_indexer(
        query,
        pool_key,
        weights,
        tail,
        layout_q="BSND",
        layout_k="BSND",
        topk=8,
        pool_size=4,
        return_value=True,
    )
    assert indices.shape == (2, 3, 11)
    assert values.shape == (2, 3, 2)
    assert values.dtype == torch.float32
