# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from vllm.config import (
    CacheConfig,
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.models.glm5next.config import Glm5NextConfig
from vllm_ascend.models.glm5next.kv_cache import KpoolTailSpec
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
from vllm_ascend.utils import glm5_next_uses_cann_kpool


class Glm5NextIndexerBackend(DeepseekV32IndexerBackend):
    """GLM-5 compressed indexer K cache backend.

    The CANN ``key_pool`` / ``pool_key_indexer`` operators address the
    compressed cache in the 4-D ``[blocks, block, 1, head_dim]`` layout, so
    the platform backend overrides the upstream 3-D indexer cache shape.
    """

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    """Indexer K cache that stores kpool-compressed BF16 entries.

    Setting ``tokens_per_state = index_kpool`` on the KV cache spec makes vLLM's
    indexer metadata builder emit pool-granular ``slot_mapping`` /
    ``seq_lens`` / ``cu_seq_lens`` / ``page_table`` for free, and shrinks the
    cache allocation store one state per ``index_kpool`` tokens. The pool
    *content* (softmax-weighted sum vs keep-every-Nth) is computed by the CANN
    ``key_pool`` operator inside the indexer op — the cache only provides the
    addressing, which is identical for both schemes.

    The indexer shares one block with the co-located MLA (a single
    ``MLAAttentionSpec`` / block_table), so ``block_size`` is the model-wide
    ``cache_config.block_size``. Compressed K is stored plain BF16 (the CANN
    kpool path allocates no FP8 quant/scale cache).
    """

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config)
        assert index_kpool > 1, "Glm5NextIndexerCache expects index_kpool > 1"
        # Keep chunked-prefill boundaries aligned to complete pools.
        assert cache_config.block_size % index_kpool == 0, (
            "Glm5NextIndexerCache: cache_config.block_size "
            f"({cache_config.block_size}) must be a multiple of index_kpool "
            f"({index_kpool}) so chunked-prefill boundaries stay pool-aligned."
        )
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        from dataclasses import replace

        spec = super().get_kv_cache_spec(vllm_config)
        # ``tokens_per_state`` is the KV-spec representation of kpool
        # compression in the current cache-layout API.
        assert isinstance(spec, MLAAttentionSpec)
        return replace(spec, tokens_per_state=self._index_kpool)

    def get_attn_backend(self):
        return Glm5NextIndexerBackend


class Glm5NextTailCache(DeepseekV32IndexerCache):
    """Paged circular buffer for the kpool indexer's in-progress (tail) pool.

    Holds the trailing incomplete pool's raw K + gate score: one block of
    ``index_kpool`` slots per request, overwritten in place by ``pos % kpool``
    as decode/spec-decode advances. Prefill seeds it (instead of discarding the
    tail raw K+gate); the connector transfers it across PD; decode reads it to
    compress the boundary pool correctly. ``KpoolTailSpec`` /
    ``KpoolTailManager`` provide the no-prune, 1-block/req allocation that lets
    the in-progress pool survive across steps and across transfer.

    Stores raw bf16 K (``head_dim``) as the "K" half of each block and the
    bf16 gate score (``head_dim``) as the "V" half -- not the fp8-compressed
    entry, which lives in ``Glm5NextIndexerCache``.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config)
        assert index_kpool > 1, "Glm5NextTailCache expects index_kpool > 1"
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # The two head slots form [K, gate score] in the generic
        # [block, head, state, content] cache view.
        return KpoolTailSpec(
            block_size=self._index_kpool,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.float32,
            sliding_window=self._index_kpool,
        )

    def get_attn_backend(self):
        from vllm.v1.attention.backends.mla.indexer import KpoolTailBackend

        return KpoolTailBackend


class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        # Indexer is only constructed for v32 configs, where these sparse-indexer
        # fields are guaranteed populated; narrow away the `int | None` declared
        # on Glm5NextConfig for the optional-indexer case.
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        if not glm5_next_uses_cann_kpool(vllm_config.model_config):
            raise RuntimeError(
                "GLM-5.3-Flash sparse (kpool) indexing requires the CANN "
                "key_pool / pool_key_indexer operators, which are not "
                "available on this hardware profile. Serve a checkpoint "
                "whose config leaves `index_topk` unset."
            )
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank  # 1536

        # kpool
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32))
        # Keep the checkpoint name ``index_kpool_compress_gate`` without a
        # ``.weight`` suffix. F.linear consumes its [head_dim, hidden_size] shape.
        self.index_kpool_compress_gate = nn.Parameter(torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16))

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        # Kept for the upstream MLA wrapper (vllm_ascend.ops.mla.IndexerWrapper
        # reads ``indexer.softmax_scale``); the CANN key_pool path does not use
        # it as a framework-side scale (the operator applies 1/sqrt(head_dim)
        # internally).
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = None
        self.quant_block_size = self.head_dim
        self.topk_indices_buffer = topk_indices_buffer

        # Compressed indexer K cache: plain BF16 rows (the CANN kpool path
        # allocates no FP8 quant/scale cache; key_pool writes the compressed
        # entries and pool_key_indexer reads them in the PA_BBND layout).
        self.k_cache = Glm5NextIndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            index_kpool=self.index_kpool,
        )
        # Paged tail cache (in-progress pool's raw K + gate score) doubles as
        # the CANN key_pool state cache: one FP32 [K, gate] row per token,
        # addressed through the +1 block table with a dummy physical block 0.
        self.tail_cache = Glm5NextTailCache(
            head_dim=self.head_dim,
            dtype=torch.float32,
            prefix=f"{prefix}.tail_cache",
            cache_config=cache_config,
            index_kpool=self.index_kpool,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexerKpool(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            tail_cache=self.tail_cache,
            logical_block_size=cache_config.block_size if cache_config is not None else 0,
            attn_layer_name=prefix.removesuffix(".indexer"),
        )

    def forward(self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb) -> torch.Tensor:
        if self.rope_dim > 0:
            raise NotImplementedError(
                "GLM-5 CANN key_pool path requires qk_rope_head_dim == 0, "
                f"got {self.rope_dim}. The AscendC key_pool operator does not "
                "support RoPE inputs in this stage."
            )
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim).to(torch.bfloat16)

        # Indexer head weights only: use the last n_head rows of the merged
        # projection (KeyPool projects K from the first head_dim rows inside).
        weights = torch.nn.functional.linear(
            hidden_states,
            self.wk_weights_proj.weight[self.head_dim :],
        )
        # PoolKeyIndexer applies 1/sqrt(head_dim) and the per-head ReLU
        # internally, so the framework only carries n_head**-0.5 here.
        weights = weights * self.n_head**-0.5

        indices = self.indexer_op(
            hidden_states,
            q,
            None,
            weights,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
            wk=self.wk_weights_proj.weight[: self.head_dim],
            gate_weight=self.index_kpool_compress_gate,
            norm_weight=self.k_norm.weight,
            norm_bias=self.k_norm.bias,
            norm_eps=self.k_norm.eps,
        )
        # The upstream MLA wrapper calls the indexer for its side effect: the
        # sparse attention backend consumes the top-k buffer.
        num_tokens = indices.shape[0]
        if self.topk_indices_buffer is not None:
            if num_tokens > self.topk_indices_buffer.shape[0]:
                raise RuntimeError(
                    "GLM-5 indexer output exceeds the topk buffer rows: "
                    f"{num_tokens} > {self.topk_indices_buffer.shape[0]}."
                )
            self.topk_indices_buffer[:num_tokens].copy_(indices.view(num_tokens, -1))
        return indices


class Glm5NextMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        skip_rope: bool | None = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Use input_size for projection input dimensions if provided,
        # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA)
        proj_input_size = input_size if input_size is not None else self.hidden_size

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if not skip_rope:
            assert config.rope_parameters is not None
            if config.rope_parameters["rope_type"] != "default":
                config.rope_parameters["rope_type"] = (
                    "deepseek_yarn"
                    if config.rope_parameters.get("apply_yarn_scaling", True)
                    else "deepseek_llama_scaling"
                )

            self.rotary_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )

            if (
                config.rope_parameters["rope_type"] != "default"
                and config.rope_parameters["rope_type"] == "deepseek_yarn"
            ):
                mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
                scaling_factor = config.rope_parameters["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        self.is_v32 = config.index_topk is not None

        if self.is_v32:
            self.indexer_rope_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not config.indexer_rope_interleave,
            )
            # The sparse indexer projects from the MLA q-lora rank, which is
            # always set for v32 MLA configs; narrow away the `int | None`.
            assert q_lora_rank is not None
            self.indexer: Indexer | None = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )

        else:
            self.indexer_rope_emb = None
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj if self.q_lora_rank is not None else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa if self.q_lora_rank is None else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            skip_topk=False,
            fuse_qkv_rmsnorm=True,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # The wrapper also runs the sparse indexer before MLA attention.
        return self.mla_attn(positions, hidden_states)
