# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.models.deepseek_mtp import AscendGlmMoeDsaForCausalLM
from vllm_ascend.models.glm54.attention import Glm54SFAImpl, LoopKVCache
from vllm_ascend.models.glm54.loop import DampGate, LoopConfig


def validate_dummy_config(vllm_config):
    config = vllm_config.model_config.hf_config
    loop = LoopConfig.from_hf_config(config)
    parallel = vllm_config.parallel_config
    if (
        any(
            getattr(parallel, name) != 1
            for name in (
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        or parallel.enable_expert_parallel
    ):
        raise ValueError("GLM-5.4 dummy currently requires PP=DP=PCP=DCP=1 without EP")
    if config.num_attention_heads % parallel.tensor_parallel_size:
        raise ValueError("GLM-5.4 dummy attention heads must be divisible by TP size")
    if parallel.use_sequence_parallel_moe:
        raise ValueError("GLM-5.4 dummy does not yet support sequence-parallel MoE")
    if not vllm_config.model_config.enforce_eager:
        raise ValueError("GLM-5.4 dummy currently requires enforce_eager=True")
    if vllm_config.quant_config is not None or vllm_config.cache_config.cache_dtype != "auto":
        raise ValueError("GLM-5.4 dummy currently requires unquantized weights and auto KV dtype")
    if vllm_config.model_config.dtype != torch.bfloat16 or not config.q_lora_rank or not config.qk_rope_head_dim:
        raise ValueError("GLM-5.4 dummy requires BF16 with Q-LoRA and RoPE")
    if vllm_config.speculative_config is not None or vllm_config.kv_transfer_config is not None:
        raise ValueError("GLM-5.4 dummy does not yet support speculative decoding or KV transfer")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("GLM-5.4 dummy prefix caching has not been validated; disable it")
    if getattr(config, "llama_4_scaling", None) is not None:
        raise ValueError("GLM-5.4 dummy does not support llama_4_scaling")
    return loop


def expand_loop_moe_calls(compilation_config, first_owned_entry, loop):
    registered = compilation_config.static_all_moe_layers
    owned = registered[first_owned_entry:]
    prelude = [name for name in owned if extract_layer_index(name) < loop.start]
    repeated = [name for name in owned if loop.start <= extract_layer_index(name) < loop.end]
    coda = [name for name in owned if extract_layer_index(name) >= loop.end]
    # The dispatcher consumes an entry per invocation, including reused layers.
    registered[first_owned_entry:] = prelude + repeated + repeated + coda


class Glm54DummyModel(DeepseekV2Model):
    def __init__(self, *, vllm_config, prefix=""):
        self.loop_config = validate_dummy_config(vllm_config)
        first_moe_entry = len(vllm_config.compilation_config.static_all_moe_layers)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        expand_loop_moe_calls(vllm_config.compilation_config, first_moe_entry, self.loop_config)
        self.damp_gates = nn.ModuleList(DampGate(self.hidden_size, self.config.rms_norm_eps) for _ in range(2))
        self.loop_caches = nn.ModuleList()
        for idx in range(self.loop_config.start, self.loop_config.end):
            attention = self.layers[idx].self_attn.mla_attn.mla_attn
            impl = attention.impl
            if not isinstance(impl, Glm54SFAImpl):
                raise ValueError("GLM-5.4 dummy requires its two-bank SFA backend")
            cache = LoopKVCache(attention, f"{prefix}.layers.{idx}.loop_kv", vllm_config)
            self.loop_caches.append(cache)
            impl.loop_cache = cache
            impl.loop_window = self.loop_config.window

    def _run_layers(self, positions, hidden_states, residual, start, end):
        for idx in range(start, end):
            hidden_states, residual = self.layers[idx](positions, hidden_states, residual, None)
        return hidden_states, residual

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("GLM-5.4 dummy does not support pipeline intermediate tensors")
        hidden_states = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        start, end = self.loop_config.start, self.loop_config.end
        hidden_states, residual = self._run_layers(positions, hidden_states, None, 0, start)
        hidden_states = hidden_states if residual is None else hidden_states + residual
        try:
            for pass_idx, gate in enumerate(self.damp_gates):
                for cache in self.loop_caches:
                    cache.impl.loop_pass = pass_idx
                inputs = hidden_states
                candidate, residual = self._run_layers(positions, inputs, None, start, end)
                candidate = candidate if residual is None else candidate + residual
                hidden_states = gate(inputs, candidate)
        finally:
            for cache in self.loop_caches:
                cache.impl.loop_pass = 0
                cache.impl.saved_indices = None
        hidden_states, residual = self._run_layers(positions, hidden_states, None, end, self.end_layer)
        return self.norm(hidden_states) if residual is None else self.norm(hidden_states, residual)[0]


class Glm54DummyForCausalLM(AscendGlmMoeDsaForCausalLM):
    model_cls = Glm54DummyModel
