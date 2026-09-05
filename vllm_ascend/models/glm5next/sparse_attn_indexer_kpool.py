# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer layer for the GLM-5.3-Flash kpool indexer.

Ascend implementation of vLLM's ``SparseAttnIndexerKpool``: the upstream
CUDA FP8/Triton kernels (block-FP8 MQA logits through DeepGEMM, paged MQA
logits, radix top-k) are replaced by the CANN ``key_pool`` and
``pool_key_indexer`` operators:

- ``key_pool`` owns the K/gate projection, the optional LayerNorm, the
  cross-chunk tail state and the KPool compression (plan phases 6);
- ``pool_key_indexer`` owns the pool-level MQA scoring (with ``1/sqrt(d)``
  and the per-head ReLU applied internally), the pool Top-K, the expansion
  to token ids and the request-final tail append (plan phase 7).

The public constructor follows the upstream custom-op layer; ``forward``
extends the optional keyword arguments with the inputs ``key_pool`` needs
internally (``wk`` / ``gate_weight`` / ``norm_weight`` / ``norm_bias``).
Without them the operator keeps the original "not implemented on Ascend"
failure so a legacy checkpoint can never silently fall back to CUDA code.
"""

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.custom_op import CustomOp

from vllm_ascend.core.kv_cache_interface import format_indexer_kpool_slot_mapping
from vllm_ascend.device.device_op import DeviceOperator

_UNSUPPORTED_MESSAGE = (
    "GLM-5.3-Flash sparse (kpool) attention indexing requires the CANN "
    "key_pool / pool_key_indexer operators, but this SparseAttnIndexerKpool "
    "was constructed without the key_pool weight inputs (wk / gate_weight). "
    "Serve a checkpoint whose config leaves `index_topk` unset, or wire the "
    "indexer through the CANN path."
)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Ascend implementation of the GLM-5.3-Flash kpool indexer operator.

    The scoring path runs on CANN ``key_pool`` + ``pool_key_indexer``. The
    compressed indexer K cache is BF16 ``[blocks, block, 1, head_dim]`` and
    the compressor state cache is FP32 ``[blocks + 1, index_kpool, 2D]``
    with an all-zero dummy physical block 0 (vLLM block ``b`` maps to the
    operator's ``b + 1``, ``-1`` to ``0``).
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
        *,
        logical_block_size: int = 0,
        attn_layer_name: str | None = None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.logical_block_size = logical_block_size
        self.attn_layer_name = attn_layer_name

    @staticmethod
    def _bound_cache(layer) -> torch.Tensor | tuple[torch.Tensor, ...]:
        context = get_forward_context()
        virtual_engine = getattr(context, "virtual_engine", 0) or 0
        cache = layer.kv_cache
        if isinstance(cache, (list, tuple)):
            cache = cache[virtual_engine]
        if isinstance(cache, (list, tuple)):
            if len(cache) == 1:
                cache = cache[0]
            elif all(isinstance(tensor, torch.Tensor) for tensor in cache):
                return tuple(cache)
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"GLM-5 Indexer cache {type(layer).__name__} is not bound.")
        return cache

    @staticmethod
    def _scatter_paged_cache(
        cache: torch.Tensor,
        slots: torch.Tensor,
        values: torch.Tensor,
        block_size: int,
    ) -> None:
        """Scatter fixed-shape rows while treating invalid slots as no-ops.

        ACLGraph-capture safe: no scatter op and no dynamic ``.nonzero()``
        slicing. Padded rows are routed to a fixed sentinel row (slot 0 of
        physical block 0), which is restored immediately afterwards; the
        sentinel contributes to no query because invalid rows are masked.
        """
        if cache.shape[1] != block_size:
            raise ValueError(
                f"Cache block size mismatch: expected {block_size}, got {cache.shape[1]}."
            )
        values = values.reshape(values.shape[0], *cache.shape[2:])
        valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
        block_ids = torch.div(
            safe_slots,
            block_size,
            rounding_mode="floor",
        )
        block_offsets = torch.remainder(safe_slots, block_size)
        row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
        row_zero = cache[0, 0].clone()
        safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
        row_zero_mask = valid & (slots == 0)
        update_zero = torch.where(
            row_zero_mask.view(-1, *([1] * (values.ndim - 1))),
            values,
            torch.zeros_like(values),
        ).sum(dim=0)
        expected_zero = torch.where(
            row_zero_mask.any(),
            update_zero,
            row_zero,
        )
        cache[block_ids, block_offsets] = safe_values
        cache[0, 0].copy_(expected_zero)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
        wk: torch.Tensor | None = None,
        gate_weight: torch.Tensor | None = None,
        norm_weight: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run the CANN key_pool compress and pool_key_indexer select sequence."""
        if wk is None or gate_weight is None:
            raise NotImplementedError(_UNSUPPORTED_MESSAGE)
        if self.use_fp4_cache:
            raise ValueError("Ascend GLM-5 Indexer uses BF16 Q, not FP4.")
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
            if q_scale is not None:
                q_values = q_values * q_scale.unsqueeze(-1).to(q_values.dtype)
        else:
            q_values = q_quant
        if compress_ape is None or positions is None:
            raise ValueError("GLM-5 kpool requires compress_ape and positions.")
        if self.tail_cache is None:
            raise RuntimeError("GLM-5 kpool requires the compressor tail cache.")
        if self.logical_block_size <= 0:
            raise RuntimeError("GLM-5 kpool requires logical_block_size.")

        context = get_forward_context()
        metadata = context.attn_metadata
        if not isinstance(metadata, dict):
            raise TypeError("GLM-5 Indexer requires per-layer attention metadata.")
        if self.attn_layer_name is None or self.attn_layer_name not in metadata:
            raise RuntimeError(
                f"Missing attention metadata for GLM-5 Indexer layer {self.attn_layer_name!r}."
            )
        attn_metadata = metadata[self.attn_layer_name]
        tail_metadata = metadata.get(self.tail_cache.prefix)
        state_cache = self._bound_cache(self.tail_cache)
        indexer_cache = self._bound_cache(self.k_cache)
        if not isinstance(state_cache, torch.Tensor):
            raise TypeError("GLM-5 compressor state cache must be one tensor.")
        if state_cache.dtype != torch.float32:
            raise TypeError(
                f"GLM-5 compressor state cache must be float32 for CANN key_pool, got {state_cache.dtype}."
            )
        if not isinstance(indexer_cache, torch.Tensor) or indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM-5 indexer cache must be one bfloat16 K tensor.")
        # [blocks, kpool, num_kv_heads=2, head_dim] -> [blocks, kpool, 2 * head_dim]
        if state_cache.ndim == 4:
            state_cache = state_cache.view(state_cache.shape[0], state_cache.shape[1], -1)

        is_full_graph = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        # Eager MTP keeps the first-pass buffer length for later draft steps,
        # while the per-step attention metadata contains only the real query
        # rows. Do not feed those padded rows into cache/indexer addressing.
        # Full graphs must retain their captured fixed shape instead.
        num_tokens = (
            positions.shape[0]
            if is_full_graph
            else min(getattr(attn_metadata, "num_actual_tokens", positions.shape[0]), positions.shape[0])
        )
        if num_tokens == 0:
            if self.topk_indices_buffer is not None:
                return self.topk_indices_buffer[:0].unsqueeze(1)
            return torch.empty(
                (0, 1, self.topk_tokens + index_kpool - 1),
                dtype=torch.int32,
                device=hidden_states.device,
            )

        # ---- Per-request quantities (plan §5.2) ----
        seq_lens = attn_metadata.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = seq_lens.shape[0]
        seq_lens = seq_lens[:num_reqs]
        query_start_loc = query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        # ---- KeyPool: K/gate projection + LayerNorm + cross-chunk tail ----
        # state_cache is mutated in place (Tensor(a!) alias). KeyPool has no
        # RoPE support: the model layer gates qk_rope_head_dim > 0 before
        # reaching this path. The state block table maps vLLM block b to
        # key_pool block b+1 (-1 -> 0, physical block 0 is the dummy sentinel).
        if tail_metadata is not None and hasattr(tail_metadata, "block_table"):
            raw_state_block_table = tail_metadata.block_table
        else:
            raw_state_block_table = attn_metadata.block_tables
        state_block_table = torch.where(
            raw_state_block_table >= 0,
            raw_state_block_table + 1,
            torch.zeros_like(raw_state_block_table),
        )
        start_pos = (seq_lens - query_lens).to(torch.int32)
        cu_seqlens = query_start_loc.to(torch.int32)
        pooled_key = DeviceOperator.kpool_key_pool_compress(
            hidden_states[:num_tokens],
            wk,
            gate_weight,
            compress_ape,
            state_cache,
            state_block_table,
            start_pos,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            cu_seqlens=cu_seqlens,
            cmp_ratio=index_kpool,
        )

        # ---- Write this call's newly completed pools into the paged K cache.
        # Fixed-shape mapping/mask only: rows whose slot_mapping is -1 are
        # no-ops, so no dynamic .nonzero() compression is needed (plan §6).
        token_ids = torch.arange(num_tokens, device=hidden_states.device)
        request_ids = torch.bucketize(
            token_ids,
            query_start_loc,
            right=True,
        ).clamp_max(pooled_key.shape[0] - 1)
        first_pool = torch.div(
            start_pos,
            index_kpool,
            rounding_mode="floor",
        )
        pool_idx = torch.div(
            positions[:num_tokens],
            index_kpool,
            rounding_mode="floor",
        )
        rows = pooled_key[
            request_ids,
            (pool_idx - first_pool[request_ids]).clamp(0, pooled_key.shape[1] - 1),
        ]
        slots = format_indexer_kpool_slot_mapping(
            attn_metadata.slot_mapping[:num_tokens],
            positions[:num_tokens],
            self.logical_block_size,
            index_kpool,
        )
        self._scatter_paged_cache(
            indexer_cache,
            slots.to(torch.int64),
            rows,
            indexer_cache.shape[1],
        )

        # ---- PoolKeyIndexer: pool Top-K + expand ----
        # The op applies 1/sqrt(head_dim) and per-head ReLU internally;
        # weights only carry the model-level num_heads**-0.5 factor (plan §7).
        indices = DeviceOperator.kpool_pool_key_indexer(
            q_values[:num_tokens],
            indexer_cache,
            weights[:num_tokens].to(q_values.dtype),
            torch.remainder(seq_lens.to(torch.int64), index_kpool),
            actual_seq_q=query_start_loc[1:].to(torch.int64),
            actual_seq_k=torch.div(
                seq_lens,
                index_kpool,
                rounding_mode="floor",
            ).to(torch.int64),
            block_table=attn_metadata.block_tables,
            topk=self.topk_tokens,
            pool_size=index_kpool,
            mask_mode=3,
        )

        # ---- Tail: restore the old per-query running-pool semantics ----
        # The CANN op appends the request-final tail [L - pool_tail_k, L)
        # causally capped by (pos - topk + 1), so early prefill rows (and any
        # row of a short request with seq_len < kpool) would be all -1. The
        # framework must overwrite the last kpool-1 columns per query position
        # with the query's OWN running pool tail so the current token always
        # enters the attention (the sparse flash attention kernel cannot
        # consume zero-length rows).
        tail_width = index_kpool - 1
        if tail_width > 0:
            positions_i64 = positions[:num_tokens].to(torch.int64)
            tail_start = (
                torch.div(
                    positions_i64 + 1,
                    index_kpool,
                    rounding_mode="floor",
                )
                * index_kpool
            )
            tail_count = positions_i64 + 1 - tail_start  # [0, kpool-1]
            tail_cols = torch.arange(
                tail_width,
                device=indices.device,
                dtype=torch.int64,
            )
            is_tail = tail_cols.unsqueeze(0) < tail_count.unsqueeze(1)
            tail_tokens = (
                tail_start.unsqueeze(1) + tail_cols.unsqueeze(0)
            ).to(torch.int32)
            tail_out = torch.where(
                is_tail,
                tail_tokens,
                torch.full_like(tail_tokens, -1),
            )
            indices[:, self.topk_tokens : self.topk_tokens + tail_width] = tail_out

        if self.skip_k_cache_insert:
            logger.warning("k_cache_insert option is not supported by the CANN kpool path.")
        return indices.unsqueeze(1)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
        wk: torch.Tensor | None = None,
        gate_weight: torch.Tensor | None = None,
        norm_weight: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> torch.Tensor:
        # The CANN path is the native path; keep the module consistent with
        # the opaque-kernel convention (``forward_native`` selected on the
        # current device family).
        return self.forward_oot(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
            wk=wk,
            gate_weight=gate_weight,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
        )