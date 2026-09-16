# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

MAX_CACHE_WRITE_TILE_SIZE = 1024
NOPE_HEAD_DIM = 512
CACHE_WRITE_ROW_GROUP_SIZE = 8


@triton.jit
def _scatter_paged_cache_kernel(
    cache,
    slots,
    values,
    capacity,
    num_tokens,
    page_stride: tl.constexpr,
    token_stride: tl.constexpr,
    head_stride: tl.constexpr,
    dim_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_dim_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    ROW_GROUP_SIZE: tl.constexpr,
):
    cols = tl.program_id(1) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    heads = cols // HEAD_DIM
    dims = cols % HEAD_DIM
    # Keep contiguous features affine: head/dim div/mod can make Ascend
    # lower a row copy into slow, element-wise indirect loads and stores.
    if NUM_HEADS == 1 or head_stride == HEAD_DIM * dim_stride:
        cache_cols = cols * dim_stride
    else:
        cache_cols = heads * head_stride + dims * dim_stride
    if NUM_HEADS == 1 or value_head_stride == HEAD_DIM * value_dim_stride:
        value_cols = cols * value_dim_stride
    else:
        value_cols = heads * value_head_stride + dims * value_dim_stride
    mask = cols < NUM_HEADS * HEAD_DIM
    # Distribute adjacent row groups cyclically. This reduces loop overhead
    # and balances periodic invalid rows across cores, while retaining the
    # affine one-row accesses used by the Ascend vector backend.
    for first in range(tl.program_id(0) * ROW_GROUP_SIZE, num_tokens, tl.num_programs(0) * ROW_GROUP_SIZE):
        for offset in tl.static_range(ROW_GROUP_SIZE):
            row = first + offset
            if ROW_GROUP_SIZE == 1 or row < num_tokens:
                slot = tl.load(slots + row * slot_stride).to(tl.int64)
                # Device-side control flow; invalid graph rows never access the cache.
                if (slot >= 0) & (slot < capacity):
                    value = tl.load(
                        values + row * value_row_stride + value_cols,
                        mask=mask,
                        other=0,
                    )
                    address = (
                        cache + (slot // BLOCK_SIZE) * page_stride + (slot % BLOCK_SIZE) * token_stride + cache_cols
                    )
                    tl.store(address, value, mask=mask)


def scatter_paged_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """Scatter [T, H, D] values into a strided [pages, B, H, D] cache.

    Valid slots must be unique. Negative and out-of-range slots are no-ops.
    Values and slots may be strided; stores convert values to the cache dtype.
    """
    num_tokens, num_heads, head_dim = values.shape
    if num_tokens == 0 or cache.numel() == 0:
        return
    tile_size = min(triton.next_power_of_2(num_heads * head_dim), MAX_CACHE_WRITE_TILE_SIZE)
    vector_cores = get_vectorcore_num()
    row_group_size = 1
    if (
        num_heads == 1
        and head_dim == NOPE_HEAD_DIM
        and num_tokens >= vector_cores * CACHE_WRITE_ROW_GROUP_SIZE
        and cache.stride(-1) == values.stride(-1) == 1
    ):
        row_group_size = CACHE_WRITE_ROW_GROUP_SIZE
    _scatter_paged_cache_kernel[
        (min(triton.cdiv(num_tokens, row_group_size), vector_cores), triton.cdiv(num_heads * head_dim, tile_size))
    ](
        cache,
        slots,
        values,
        cache.shape[0] * cache.shape[1],
        num_tokens,
        *cache.stride(),
        slots.stride(0),
        *values.stride(),
        cache.shape[1],
        num_heads,
        head_dim,
        tile_size,
        row_group_size,
    )
