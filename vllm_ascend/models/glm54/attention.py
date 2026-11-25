# SPDX-License-Identifier: Apache-2.0

import weakref

import torch
from torch import nn
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.mla_v1 import AscendMLAAttention
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl
from vllm_ascend.models.glm54.cache_config import Glm54WindowCacheSpec
from vllm_ascend.models.glm54.loop import two_bank_mla


class LoopKVCache(AscendMLAAttention):
    """An independently managed window cache for the second loop pass."""

    def __init__(self, source: AscendMLAAttention, prefix: str, vllm_config):
        nn.Module.__init__(self)
        self._source_ref = weakref.ref(source)
        self.layer_name = prefix
        self.kv_cache = torch.tensor([])
        self.impl = source.impl
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate loop cache: {prefix}")
        context[prefix] = self

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            source = self._source_ref()
            if source is None:
                raise RuntimeError("Loop cache outlived its physical layer") from None
            return getattr(source, name)

    def get_attn_backend(self):
        return self._source_ref().get_attn_backend()

    def get_kv_cache_spec(self, vllm_config):
        source = self._source_ref().get_kv_cache_spec(vllm_config)
        return Glm54WindowCacheSpec(
            block_size=source.block_size,
            num_kv_heads=source.num_kv_heads,
            head_size=source.head_size,
            dtype=source.dtype,
            sliding_window=self.impl.loop_window,
        )

    def process_weights_after_loading(self, act_dtype=None):
        # The physical layer owns projection weights and their postprocessing.
        pass


class Glm54SFABackend(AscendSFABackend):
    @staticmethod
    def is_mla():
        return True

    @staticmethod
    def get_impl_cls():
        return Glm54SFAImpl


class Glm54SFAImpl(AscendSFAImpl):
    loop_pass: int = 0
    loop_window: int = 0
    loop_cache: LoopKVCache | None = None
    saved_indices: torch.Tensor | None = None

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        if self.loop_cache is not None:
            # Index groups reuse writable buffers; keep this call's indices
            # until the corresponding physical layer runs in pass two.
            self.saved_indices = topk_indices.clone()
        return super()._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
        )

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
        if self.loop_pass != 1 or attn_metadata is None:
            return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)
        assert output is not None and self.loop_cache is not None and self.saved_indices is not None
        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens == 0:
            return output.zero_()
        hidden_states = hidden_states[:num_tokens]
        local_metadata = get_forward_context().attn_metadata[self.loop_cache.layer_name]
        local_cache = self._compose_sfa_kv_cache(self.loop_cache.kv_cache)
        first_cache = self._compose_sfa_kv_cache(kv_cache)
        assert local_cache is not None and first_cache is not None
        qkv = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv = qkv.split((self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim), dim=-1)
        q_c = self.q_a_layernorm(q_c)
        self.exec_kv(
            kv,
            attn_metadata.cos[:num_tokens],
            attn_metadata.sin[:num_tokens],
            local_cache,
            local_metadata.slot_mapping[:num_tokens],
            local_metadata,
        )
        q_latent, q_rope = self._q_proj_and_k_up_proj(q_c)
        q_rope = self.rope_single(q_rope, attn_metadata.cos[:num_tokens], attn_metadata.sin[:num_tokens])
        latent_output = two_bank_mla(
            q_latent,
            q_rope,
            first_cache[:2],
            local_cache[:2],
            self.saved_indices[:num_tokens],
            attn_metadata.block_table,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            self.loop_window,
            self.scale,
            local_block_table=local_metadata.block_table,
        )
        self._start_o_proj_weight_gather()
        self._finalize_o_proj(self._v_up_proj(latent_output), output[:num_tokens])
        return output
