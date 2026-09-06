# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash kpool tail-cache backend for vllm 0.27.1.

vLLM main ships ``vllm.v1.attention.backends.mla.indexer.KpoolTailBackend``
(and the ``tokens_per_state`` KV-spec field) for the GLM kpool tail scratch
cache. vllm 0.27.1 has neither, so this module provides the plugin-side
equivalents: a storage-only backend plus a metadata builder that maps every
token to its request's single circular tail block (``compute_kpool_tail_slot_mapping``
is a pure-torch port of the vLLM main implementation).

``Glm5NextTailCache.get_attn_backend`` prefers the vLLM symbol when present
and falls back to this module otherwise.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map every token to its request's one circular tail block.

    The tail cache allocates exactly one block of ``kpool`` slots per request
    (``KpoolTailManager``); the slot for token ``pos`` is
    ``own_block * kpool + pos % kpool``.
    """
    if out is None:
        out = slot_mapping.clone()
    else:
        assert out.shape == slot_mapping.shape
        out.copy_(slot_mapping)
    if num_actual_tokens == 0:
        return out
    tokens = torch.arange(num_actual_tokens, device=slot_mapping.device)
    req = torch.searchsorted(query_start_loc, tokens, right=True) - 1
    req = req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return out


class Glm5NextTailMetadataBuilder(AttentionMetadataBuilder):
    """Build only the circular slot mapping needed by the storage-only tail."""

    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.slot_mapping_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        del common_prefix_len, fast_build
        slot_mapping = common_attn_metadata.slot_mapping
        positions = getattr(common_attn_metadata, "positions", None)
        if positions is not None:
            slot_mapping_buffer = self.slot_mapping_buffer[: slot_mapping.numel()].view_as(slot_mapping)
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
                out=slot_mapping_buffer,
            )
        # vllm 0.27.1's CommonAttentionMetadata has no num_decodes /
        # num_prefills split fields; derive them from is_prefilling.
        if getattr(common_attn_metadata, "is_prefilling", False):
            num_decodes, num_prefills = 0, common_attn_metadata.num_reqs
            num_decode_tokens, num_prefill_tokens = 0, common_attn_metadata.num_actual_tokens
        else:
            num_decodes, num_prefills = common_attn_metadata.num_reqs, 0
            num_decode_tokens, num_prefill_tokens = common_attn_metadata.num_actual_tokens, 0
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )


class Glm5NextTailBackend(DeepseekV32IndexerBackend):
    """Storage-only backend for the GLM-5.3-Flash kpool tail cache.

    vllm 0.27.1 equivalent of vLLM main's ``KpoolTailBackend``. The tail
    cache tensor is ``[num_blocks, index_kpool, 2, head_dim]`` (the two head
    slots hold the raw ``[K, gate]`` pair), so the platform backend overrides
    the upstream 3-D indexer cache shape.
    """

    @staticmethod
    def get_name() -> str:
        return "GLM5_NEXT_KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[Glm5NextTailMetadataBuilder]:
        return Glm5NextTailMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2, 3)


# Re-export so attention.py can use a single symbol name for the fallback.
__all__ = ["Glm5NextTailBackend", "Glm5NextTailMetadataBuilder", "compute_kpool_tail_slot_mapping"]
