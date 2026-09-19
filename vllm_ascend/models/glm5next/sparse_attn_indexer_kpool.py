# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor orchestration for the GLM-Next Triton KPool indexer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_ascend.models.glm5next.ops.indexer_finalize import finalize_indices
from vllm_ascend.ops.triton.glm5_next_kpool_tail_compress import (  # type: ignore[import-untyped]
    glm5_next_kpool_tail_compress_and_write_cache_triton,
)
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (  # type: ignore[import-untyped]
    glm5_next_lightning_indexer_triton,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolTailMetadata,
    )


class SparseAttnIndexerKpool(nn.Module):
    """Update KPool caches and optionally select sparse token indices.

    Cache binding and forward-context lookup belong to the model-side backend.
    This helper receives explicit tensors and typed metadata so the cache update
    can be tested independently from the vLLM attention wrapper.
    """

    def __init__(self, topk_tokens: int, head_dim: int) -> None:
        super().__init__()
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim

    def forward(
        self,
        k: torch.Tensor,
        q_values: torch.Tensor | None,
        weights: torch.Tensor | None,
        positions: torch.Tensor,
        indexer_cache: torch.Tensor,
        tail_cache: torch.Tensor,
        indexer_metadata: AscendIndexerKPoolMetadata,
        tail_metadata: AscendIndexerKPoolTailMetadata,
        *,
        gate_score: torch.Tensor,
        compress_ape: torch.Tensor,
        index_kpool: int,
        max_pool_seq_len: int,
        compute_topk: bool,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        num_tokens = k.shape[0]
        if index_kpool <= 0 or self.topk_tokens % index_kpool:
            raise ValueError("KPool top-k must be divisible by its positive pool size.")
        if num_tokens == 0:
            return (
                None
                if not compute_topk
                else torch.empty((0, 1, self.topk_tokens + index_kpool - 1), dtype=torch.int32, device=k.device)
            )
        if indexer_metadata.cum_query_lens is None or indexer_metadata.raw_seq_lens is None:
            raise ValueError("GLM KPool metadata requires cum_query_lens and raw_seq_lens.")
        if indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM KPool compressed cache must be bfloat16.")
        if tail_cache.dtype != torch.float32 or k.dtype != torch.float32 or gate_score.dtype != torch.float32:
            raise TypeError("GLM KPool keys, gates and compressor tail must be float32.")
        if (
            tail_cache.ndim != 4
            or tail_cache.shape[1] != 2
            or tail_cache.shape[2] < index_kpool
            or tail_cache.shape[3] != self.head_dim
            or gate_score.shape != k.shape
        ):
            raise ValueError("GLM KPool tail requires [blocks, 2, capacity, head_dim] K/gate storage.")
        if tail_metadata.block_size != tail_cache.shape[2]:
            raise ValueError("GLM KPool tail metadata capacity must match the bound cache.")
        if compress_ape.shape != (index_kpool, self.head_dim) or compress_ape.dtype != torch.float32:
            raise ValueError("GLM KPool APE must be FP32 with shape [pool_size, head_dim].")

        glm5_next_kpool_tail_compress_and_write_cache_triton(
            tail_cache,
            indexer_cache,
            k,
            gate_score,
            compress_ape,
            positions,
            indexer_metadata.cum_query_lens,
            indexer_metadata.raw_seq_lens,
            tail_metadata.slot_mapping[:num_tokens],
            tail_metadata.block_table,
            indexer_metadata.slot_mapping[:num_tokens],
            index_kpool,
        )
        # Sharing top-k still advances the compressed cache and raw tail.
        if not compute_topk:
            return None
        if q_values is None or weights is None:
            raise ValueError("GLM KPool top-k requires query and head weights.")
        indices = glm5_next_lightning_indexer_triton(
            q_values,
            indexer_cache,
            weights.to(q_values.dtype),
            indexer_metadata.cum_query_lens,
            indexer_metadata.seq_lens,
            indexer_metadata.block_table,
            positions,
            index_topk=self.topk_tokens,
            index_kpool=index_kpool,
            max_pool_seq_len=max_pool_seq_len,
        )
        # A2/A3 SFA requires a contiguous valid prefix; the reference indexer
        # puts the running tail at the fixed top-k column for short requests.
        finalize_indices(
            indices[:, 0],
            positions,
            self.topk_tokens,
            index_kpool,
            indexer_metadata.cum_query_lens,
            output_buffer,
        )
        return indices
