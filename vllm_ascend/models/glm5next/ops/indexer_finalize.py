# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finalize GLM sparse indices without changing scoring or pool selection."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

INDEXER_FINALIZE_BLOCK_SIZE = 256
INDEXER_FINALIZE_DECODE_ROWS = 16
INDEXER_FINALIZE_PREFILL_BLOCK_SIZE = 1024


@triton.jit
def _finalize_indices(
    indices,
    positions,
    query_ends,
    output,
    index_stride,
    position_stride,
    output_stride,
    rows,
    DECODE_ROWS: tl.constexpr,
    PERSISTENT: tl.constexpr,
    LAST_QUERY: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    POOL: tl.constexpr,
    COPY_OUTPUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live_rows = tl.load(query_ends + LAST_QUERY)
    # Prefill shapes vary with every scheduler batch. Keep their row count
    # dynamic to reuse compiled code, while specializing captured decode sizes.
    row_end = rows if PERSISTENT else DECODE_ROWS
    row_step = tl.num_programs(0) if PERSISTENT else DECODE_ROWS
    for row in range(tl.program_id(0), row_end, row_step):
        live = row < live_rows
        position = tl.load(positions + row * position_stride).to(tl.int64)
        values = tl.load(indices + row * index_stride + columns, columns < SOURCE_WIDTH, other=-1)
        if POOL > 1:
            tail_start = (position + 1) // POOL * POOL
            # Keep absolute-position arithmetic in int64, but bounded column
            # offsets and stored indices in int32 for Ascend vector execution.
            tail_offset = tl.minimum(tail_start, TOPK).to(tl.int32)
            tail_column = columns - tail_offset
            is_tail = (tail_column >= 0) & (tail_column < POOL - 1)
            tail_size = (position + 1 - tail_start).to(tl.int32)
            tail_values = tl.where(tail_column < tail_size, tail_start.to(tl.int32) + tail_column, -1)
            values = tl.where(columns >= TOPK, -1, values)
            values = tl.where(is_tail, tail_values, values)
        values = tl.where(live, values, -1).to(indices.dtype.element_ty)
        tl.store(indices + row * index_stride + columns, values, columns < SOURCE_WIDTH)
        if COPY_OUTPUT:
            values = tl.where(columns < SOURCE_WIDTH, values, -1)
            tl.store(output + row * output_stride + columns, values, columns < OUTPUT_WIDTH)


def finalize_indices(
    indices: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
    query_ends: torch.Tensor,
    output: torch.Tensor | None = None,
) -> None:
    """Pack the causal tail, mask padding and optionally fill the SFA buffer."""
    if output is not None and (output.shape[0] != indices.shape[0] or output.shape[1] < indices.shape[1]):
        raise ValueError("GLM sparse output buffer must cover all index rows and columns.")
    if indices.shape[0] == 0:
        return
    width = indices.shape[1] if output is None else output.shape[1]
    destination = indices if output is None else output
    rows = indices.shape[0]
    persistent = rows > INDEXER_FINALIZE_DECODE_ROWS
    block = (
        min(next_power_of_2(width), INDEXER_FINALIZE_PREFILL_BLOCK_SIZE) if persistent else INDEXER_FINALIZE_BLOCK_SIZE
    )
    column_tiles = triton.cdiv(width, block)
    row_programs = min(rows, max(1, get_vectorcore_num() // column_tiles)) if persistent else rows
    grid = (row_programs, column_tiles)
    _finalize_indices[grid](
        indices,
        positions,
        query_ends,
        destination,
        indices.stride(0),
        positions.stride(0),
        destination.stride(0),
        rows,
        0 if persistent else rows,
        persistent,
        query_ends.numel() - 1,
        indices.shape[1],
        width,
        topk_tokens,
        pool_size,
        output is not None,
        block,
    )
