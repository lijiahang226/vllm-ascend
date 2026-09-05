# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone PyTorch reference for the GLM-5 Next CANN ``key_pool`` op.

This implements the key_pool operator contract (plan §9.1) independently of
the CANN op and of the framework wrappers it replaces. It is CPU-run in the
UTs here and serves as the oracle for the NPU accuracy tests in
``tests/e2e/models/test_glm5_next_key_pool.py``.

The ``pool_key_indexer`` oracle is the official CANN golden vendored at
``tests/ut/ops/helpers/pool_key_indexer_reference.py``.

Contract notes (from the ``aclnnKeyPool`` doc and the op_host tiling):
- ``key_pool`` writes each incoming chunk's ``[K, gate]`` rows into an FP32
  state cache addressed through the state block table (``0`` = no update),
  then compresses every pool that becomes complete inside the chunk: a
  per-row softmax over the pool axis of ``gate + ape`` followed by a weighted
  sum of K.
"""

from __future__ import annotations

import math

import torch


def _layer_norm(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    if weight is None or bias is None:
        return x
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight, bias, eps).to(x.dtype)


def key_pool_reference(
    x: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    cmp_ratio: int = 4,
) -> torch.Tensor:
    """Reference of the CANN ``key_pool`` contract (TND layout, coff=1).

    Args:
        x: ``[T, H]`` chunk of hidden states (BF16/FP16).
        wk: ``[D, H]`` K projection weight (rows = head dim).
        gate_weight: ``[D, H]`` gate projection weight.
        ape: ``[cmp_ratio, D]`` positional bias (FP32).
        state_cache: ``[num_blocks, block_size, 2 * D]`` FP32 ``[K, gate]``
            tail state; block id 0 is reserved as the invalid sentinel and
            never written.
        state_block_table: ``[B, pages]`` INT32; values are the framework
            converted key_pool block ids (vLLM b -> b+1, -1 -> 0).
        start_pos: ``[B]`` INT32 absolute position the chunk starts at.
        cu_seqlens: ``[B + 1]`` INT32 ``[0, cumsum(query_lens)]``; required
            for the TND layout.
        norm_weight / norm_bias / norm_eps: optional LayerNorm for K.
        cmp_ratio: pool size (``index_kpool``).

    Returns:
        ``[B, ceil(pages * block_size / cmp_ratio), D]`` pooled keys; only
        pools that complete in this call are filled.
    """
    input_dtype = x.dtype  # captured before the computation cast below
    x = x.float()
    wk = wk.float()
    gate_weight = gate_weight.float()
    ape = ape.float()
    if cu_seqlens is None:
        raise ValueError("key_pool_reference requires cu_seqlens for the TND layout")
    chunk_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch = state_block_table.shape[0]
    block_size = state_cache.shape[1]
    head_dim = wk.shape[0]
    pool_capacity = (
        state_block_table.shape[1] * block_size + cmp_ratio - 1
    ) // cmp_ratio
    pooled = torch.zeros(batch, pool_capacity, head_dim, dtype=torch.float32)

    k = _layer_norm(x @ wk.t(), norm_weight, norm_bias, norm_eps)
    gate = x @ gate_weight.t()

    # Split the chunk per request.
    chunk_rows: list[torch.Tensor] = []
    offset = 0
    for length in chunk_lens.tolist():
        chunk_rows.append((offset, offset + int(length)))
        offset += int(length)
    assert offset == x.shape[0], "cu_seqlens[-1] must equal the chunk length"

    # 1) Persist the incomplete-pool tail of this chunk into the FP32 state
    #    cache (cross-chunk state, plan §9.1): only the last
    #    ``len % cmp_ratio`` tokens need to survive for the next call.
    #    Matches the op golden (key_pool_golden._write_cache).
    for b in range(batch):
        length = int(chunk_lens[b])
        end_pos = int(start_pos[b]) + length
        tail_len = end_pos % cmp_ratio
        if tail_len == 0:
            continue
        tail_start = max(int(start_pos[b]), end_pos - tail_len)
        for local in range(length):
            abs_pos = int(start_pos[b]) + local
            if abs_pos < tail_start:
                continue
            page = abs_pos // block_size
            block_id = int(state_block_table[b, page])
            if block_id == 0:
                continue
            row_start, row_end = chunk_rows[b]
            state_cache[block_id, abs_pos % block_size, :head_dim] = k[row_start + local]
            state_cache[block_id, abs_pos % block_size, head_dim:] = gate[row_start + local]

    # 2) Compress every pool that completes inside this chunk. Tokens before
    #    start_pos are read back from the state cache (cross-chunk tail).
    for b in range(batch):
        length = int(chunk_lens[b])
        chunk_start = int(start_pos[b])
        chunk_end = chunk_start + length
        first_pool = chunk_start // cmp_ratio
        last_pool = (chunk_end - 1) // cmp_ratio
        for pool in range(first_pool, last_pool + 1):
            pool_end = (pool + 1) * cmp_ratio
            if pool_end > chunk_end:
                continue  # pool is still incomplete
            pool_k = torch.zeros(cmp_ratio, head_dim, dtype=torch.float32)
            pool_gate = torch.zeros(cmp_ratio, head_dim, dtype=torch.float32)
            fill = 0
            for token_pos in range(pool * cmp_ratio, pool_end):
                if token_pos < chunk_start:
                    page = token_pos // block_size
                    block_id = int(state_block_table[b, page])
                    if block_id == 0:
                        raise ValueError(
                            f"request {b}: history token {token_pos} has an invalid state block"
                        )
                    pool_k[fill] = state_cache[block_id, token_pos % block_size, :head_dim]
                    pool_gate[fill] = state_cache[block_id, token_pos % block_size, head_dim:]
                else:
                    row_start, _ = chunk_rows[b]
                    pool_k[fill] = k[row_start + token_pos - chunk_start]
                    pool_gate[fill] = gate[row_start + token_pos - chunk_start]
                fill += 1
            scores = (pool_gate + ape).softmax(dim=0)
            pooled[b, pool - first_pool] = (scores * pool_k).sum(dim=0)

    return pooled.to(input_dtype)


def sparse_to_dense_token_ids(
    sparse_indices: torch.Tensor,
    seq_len: int,
) -> list[list[int]]:
    """Expand sparse indices to unique valid token ids (test helper)."""
    valid = (sparse_indices >= 0) & (sparse_indices < seq_len)
    return [sparse_indices[i][valid[i]].tolist() for i in range(sparse_indices.shape[0])]
