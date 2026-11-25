# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models.glm54.attention import LoopKVCache
from vllm_ascend.models.glm54.cache_config import Glm54WindowCacheSpec
from vllm_ascend.models.glm54.loop import DampGate, LoopConfig
from vllm_ascend.models.glm54.model import Glm54DummyModel, expand_loop_moe_calls, validate_dummy_config


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                glm54_dummy=True,
                loop_start_layer=2,
                loop_end_layer=4,
                loop_window_size=128,
                num_hidden_layers=6,
                q_lora_rank=128,
                qk_rope_head_dim=64,
                num_attention_heads=16,
            ),
            enforce_eager=True,
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            use_sequence_parallel_moe=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        quant_config=None,
        speculative_config=None,
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("field", ["enforce_eager", "prefix", "pp", "tp_heads", "sp", "quant", "mtp", "pd", "dtype"])
def test_unsupported_execution_is_rejected(field):
    value = config()
    if field == "enforce_eager":
        value.model_config.enforce_eager = False
    elif field == "prefix":
        value.cache_config.enable_prefix_caching = True
    elif field == "pp":
        value.parallel_config.pipeline_parallel_size = 2
    elif field == "tp_heads":
        value.parallel_config.tensor_parallel_size = 3
    elif field == "sp":
        value.parallel_config.use_sequence_parallel_moe = True
    elif field == "quant":
        value.quant_config = object()
    elif field == "mtp":
        value.speculative_config = object()
    elif field == "pd":
        value.kv_transfer_config = object()
    else:
        value.model_config.dtype = torch.float16
    with pytest.raises(ValueError):
        validate_dummy_config(value)


@pytest.mark.parametrize("tp_size", [1, 8])
def test_supported_config(tp_size):
    value = config()
    value.parallel_config.tensor_parallel_size = tp_size
    assert validate_dummy_config(value) == LoopConfig(2, 4, 128)


def test_moe_dispatcher_tracks_invocations_without_duplicating_physical_layers():
    names = [f"model.layers.{index}.mlp.experts" for index in range(1, 6)]
    compilation = SimpleNamespace(static_all_moe_layers=["other.model.layers.0.mlp.experts", *names])
    expand_loop_moe_calls(compilation, 1, LoopConfig(2, 4, 128))
    assert compilation.static_all_moe_layers == [
        "other.model.layers.0.mlp.experts",
        names[0],
        names[1],
        names[2],
        names[1],
        names[2],
        names[3],
        names[4],
    ]
    assert len(set(compilation.static_all_moe_layers[1:])) == 5


def test_loop_cache_owns_no_weights_and_has_a_separate_allocation_identity():
    class Source(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.impl = SimpleNamespace(loop_window=128)
            self.qk_rope_head_dim = 64

        def get_kv_cache_spec(self, config):
            return AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)

        def get_attn_backend(self):
            return "physical-backend"

    source = Source()
    value = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    cache = LoopKVCache(source, "model.layers.2.loop_kv", value)
    cache.process_weights_after_loading()
    cache.process_weights_after_loading(torch.bfloat16)
    assert list(cache.parameters()) == []
    spec = cache.get_kv_cache_spec(value)
    assert isinstance(spec, Glm54WindowCacheSpec)
    assert spec.sliding_window == 128
    assert spec.page_size_bytes == 128 * 576 * 2
    assert cache.qk_rope_head_dim == 64
    assert value.compilation_config.static_forward_context[cache.layer_name] is cache


@pytest.mark.parametrize("fail_second_pass", [False, True])
def test_loop_reuses_layers_materializes_residuals_and_clears_step_state(fail_second_pass):
    model = Glm54DummyModel.__new__(Glm54DummyModel)
    nn.Module.__init__(model)
    model.loop_config = LoopConfig(2, 4, 128)
    model.end_layer = 6
    model.loop_caches = [SimpleNamespace(impl=SimpleNamespace(loop_pass=0, saved_indices=object())) for _ in range(2)]
    calls = []

    class Layer(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, positions, hidden, residual, scaling):
            calls.append((self.index, id(self)))
            if fail_second_pass and self.index == 3 and model.loop_caches[0].impl.loop_pass == 1:
                raise RuntimeError("interrupted loop")
            inputs = hidden if residual is None else hidden + residual
            return torch.full_like(inputs, self.index + 1), inputs

    class Norm(nn.Module):
        def forward(self, hidden, residual=None):
            return hidden if residual is None else (hidden + residual, residual)

    model.layers = nn.ModuleList(Layer(index) for index in range(6))
    model.damp_gates = nn.ModuleList(DampGate(4, 1e-6) for _ in range(2))
    for gate in model.damp_gates:
        with torch.no_grad():
            gate.proj.weight.zero_()
    model.norm = Norm()
    inputs = torch.randn(3, 4)
    if fail_second_pass:
        with pytest.raises(RuntimeError, match="interrupted loop"):
            model.forward(None, torch.arange(3), inputs_embeds=inputs)
    else:
        output = model.forward(None, torch.arange(3), inputs_embeds=inputs)
        assert [index for index, _ in calls] == [0, 1, 2, 3, 2, 3, 4, 5]
        assert calls[2] == calls[4] and calls[3] == calls[5]
        torch.testing.assert_close(output, inputs + 21)
    for cache in model.loop_caches:
        assert cache.impl.loop_pass == 0
        assert cache.impl.saved_indices is None
