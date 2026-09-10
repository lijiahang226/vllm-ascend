# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_ascend.models.glm5next.model as glm
import vllm_ascend.models.glm5next.mtp as mtp
from vllm_ascend.models.glm5next.config import Glm5NextTextConfig
from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, get_quant_type_for_layer


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_modelslim_resolves_mtp_and_multimodal_quant_keys(wrapped, packed):
    model_type = "glm5_next_mtp"
    base = "model.layers.1.self_attn."
    checkpoint = "model.language_model.layers.1.self_attn." if wrapped else base
    shards = ["q_a_proj", "kv_a_proj_with_mqa"] if packed else ["o_proj"]
    config = AscendModelSlimConfig({checkpoint + shard + ".weight": "W8A8_DYNAMIC" for shard in shards})
    mapper = glm.Glm5NextForConditionalGeneration.hf_to_vllm_mapper if wrapped else glm.GLM5_WEIGHTS_MAPPER
    config.apply_vllm_mapper(mapper)
    config.packed_modules_mapping = dict(glm.GLM5_PACKED_MODULES_MAPPING)
    config._update_packed_modules_mapping(model_type)
    prefix = "model.layers.1.mtp_block.self_attn." + ("fused_qkv_a_proj" if packed else "o_proj")
    prefix = config.quant_prefix_mapper(model_type, prefix)
    assert prefix == ("language_model." if wrapped else "") + base + ("fused_qkv_a_proj" if packed else "o_proj")
    assert get_quant_type_for_layer(config.quant_description, prefix, config.packed_modules_mapping) == "W8A8_DYNAMIC"


def test_direct_quant_keys_take_precedence_and_mixed_shards_fail():
    base = "model.layers.1.self_attn."
    config = AscendModelSlimConfig(
        {
            base + "q_a_proj.weight": "FLOAT",
            base + "kv_a_proj_with_mqa.weight": "W8A8_DYNAMIC",
            "language_model." + base + "q_a_proj.weight": "W8A8_DYNAMIC",
            "language_model." + base + "kv_a_proj_with_mqa.weight": "W8A8_DYNAMIC",
        }
    )
    config.packed_modules_mapping = dict(glm.GLM5_PACKED_MODULES_MAPPING)
    prefix = config.quant_prefix_mapper("glm5_next", base + "fused_qkv_a_proj")
    assert prefix == base + "fused_qkv_a_proj"
    with pytest.raises(ValueError, match="same quant type"):
        get_quant_type_for_layer(config.quant_description, prefix, config.packed_modules_mapping)


def _module(cls=nn.Module):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def _config():
    return SimpleNamespace(
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        n_routed_experts=0,
        is_moe=False,
        mla_nope=True,
        qk_rope_head_dim=0,
        kv_lora_rank=2,
    )


@pytest.mark.parametrize("modelslim", [False, True])
def test_mla_receives_modelslim_config_and_preserves_native_fp8(monkeypatch, modelslim):
    quant = (
        AscendModelSlimConfig({})
        if modelslim
        else AscendFp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128]
        )
    )
    config = Glm5NextTextConfig(num_hidden_layers=1, layer_types=["deepseek_sparse_attention"])
    runtime = SimpleNamespace(
        cache_config=None, quant_config=quant, parallel_config=SimpleNamespace(use_sequence_parallel_moe=False)
    )

    class CapturedProjectionConfig(Exception):
        pass

    def attention(**kwargs):
        assert kwargs["quant_config"] is (quant if modelslim else None)
        raise CapturedProjectionConfig

    monkeypatch.setattr(glm, "Glm5NextMLAAttention", attention)
    with pytest.raises(CapturedProjectionConfig):
        glm.Glm5NextDecoderLayer(runtime, config, 0)


@pytest.mark.parametrize("wrapped", [False, True])
def test_target_loader_skips_mtp_rotation_and_maps_forget_gate(wrapped):
    target = _module(glm.Glm5NextForCausalLM)
    target.model = _module(glm.Glm5NextModel)
    target.model.config = _config()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.f_b_proj = nn.Linear(2, 2, bias=False)
    target.model.layers = nn.ModuleList([layer])
    wrapper = _module(glm.Glm5NextForConditionalGeneration) if wrapped else target
    if wrapped:
        wrapper.language_model = target
    prefix = "model.language_model." if wrapped else "model."
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    loaded = wrapper.load_weights(
        [
            ("rot.weight", torch.eye(2)),
            (prefix + "layers.0.self_attn.forget_gate.f_b_proj.weight", value),
        ]
    )
    torch.testing.assert_close(layer.self_attn.f_b_proj.weight, value)
    assert loaded == {("language_model." if wrapped else "") + "model.layers.0.self_attn.f_b_proj.weight"}


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_modelslim_weight_and_scale_reach_the_packed_loader(monkeypatch, draft, dtype):
    layer = nn.Module()
    layer.is_rot_used = False
    attn = nn.Module()
    attn.fused_qkv_a_proj = nn.Module()
    calls = []
    for suffix, param_dtype in [("weight", dtype), ("weight_scale", torch.float32)]:
        param = nn.Parameter(torch.zeros(4, 2, dtype=param_dtype), requires_grad=False)

        def loader(param, value, shard, suffix=suffix):
            param.data[shard * 2 : (shard + 1) * 2].copy_(value)
            calls.append((suffix, shard))

        param.weight_loader = loader
        attn.fused_qkv_a_proj.register_parameter(suffix, param)
    if draft:
        layer.mtp_block = nn.Module()
        layer.mtp_block.self_attn = attn
        model = _module(mtp.Glm5NextMTP)
        model.model = nn.Module()
        model.model.layers = nn.ModuleDict({"1": layer})
        model.model.mtp_start_layer_idx = model.model.num_mtp_layers = 1
        monkeypatch.setattr(mtp, "fused_moe_make_expert_params_mapping", lambda *a, **k: [])
        prefix = "model.layers.1.self_attn."
    else:
        layer.self_attn = attn
        model = _module(glm.Glm5NextModel)
        model.layers = nn.ModuleList([layer])
        prefix = "layers.0.self_attn."
    model.config = _config()
    weights = [
        (
            prefix + shard + "." + suffix,
            torch.full((2, 2), float(index + 1)).to(dtype if suffix == "weight" else torch.float32),
        )
        for index, shard in enumerate(["q_a_proj", "kv_a_proj_with_mqa"])
        for suffix in ["weight", "weight_scale"]
    ]
    model.load_weights(weights)
    assert calls == [("weight", 0), ("weight_scale", 0), ("weight", 1), ("weight_scale", 1)]
    torch.testing.assert_close(
        attn.fused_qkv_a_proj.weight.float(), torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
    )


def test_native_block_fp8_still_dequantizes_into_bf16():
    value = torch.ones(2, 2).to(torch.float8_e4m3fn)
    target = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False)
    target.weight_loader = lambda param, weight, shard: param.data.copy_(weight)
    params = {"layer.fused_qkv_a_proj.weight": target}
    pending, loaded = {}, set()
    for suffix, tensor in [("weight", value), ("weight_scale_inv", torch.tensor([[2.0]]))]:
        assert glm._try_load_fp8_attn_proj("layer.q_a_proj." + suffix, tensor, pending, params, loaded, 0)
    torch.testing.assert_close(target, torch.full_like(target, 2))
    assert loaded == set(params)


@pytest.mark.parametrize("enabled", [False, True])
def test_mtp_rotation_is_applied_before_hidden_state_norm(monkeypatch, enabled):
    config = SimpleNamespace(hidden_size=2, rms_norm_eps=1e-5, index_topk=128, index_kpool=4)
    runtime = SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={"is_rot_used": enabled}),
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config)),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2),
    )
    monkeypatch.setattr(mtp, "current_platform", SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(mtp, "RMSNorm", lambda *a, **k: SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5))
    monkeypatch.setattr(mtp, "SharedHead", lambda **k: SimpleNamespace(norm=lambda x, residual: (x, None)))
    monkeypatch.setattr(
        mtp, "Glm5NextDecoderLayer", lambda **k: lambda **inputs: (inputs["hidden_states"], None, None, None)
    )

    def norm(positions, embeds, hidden, ew, hw, eps):
        normalized = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)
        return torch.cat((embeds, normalized), dim=-1)

    monkeypatch.setattr(mtp, "fused_eh_norm", norm)
    layer = mtp.Glm5NextMultiTokenPredictorLayer(runtime, "model.layers.1")
    layer.eh_proj.weight.data.copy_(torch.cat((torch.zeros(2, 2), torch.eye(2)), dim=1))
    previous = torch.tensor([[1.0, 3.0]])
    transformed = previous
    if enabled:
        layer.rot.weight.data.copy_(torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))
        transformed = torch.tensor([[3.0, -1.0]])
    else:
        assert not hasattr(layer, "rot")
    result, recycled = layer(None, torch.tensor([1]), previous, torch.zeros(1, 2))
    torch.testing.assert_close(result, transformed * torch.rsqrt(transformed.square().mean(-1, keepdim=True) + 1e-5))
    assert recycled is result


@pytest.mark.parametrize("missing", [False, True])
def test_mtp_loads_required_rotation_and_rejects_missing_weight(monkeypatch, missing):
    model = _module(mtp.Glm5NextMTP)
    model.config = _config()
    model.model = nn.Module()
    layer = nn.Module()
    layer.is_rot_used = True
    layer.rot = nn.Linear(2, 2, bias=False)
    layer.eh_proj = nn.Linear(4, 2, bias=False)
    model.model.layers = nn.ModuleDict({"1": layer})
    model.model.mtp_start_layer_idx = model.model.num_mtp_layers = 1
    monkeypatch.setattr(mtp, "fused_moe_make_expert_params_mapping", lambda *a, **k: [])
    weights = [("layers.1.eh_proj.weight", torch.ones(2, 4))]
    if missing:
        with pytest.raises(ValueError, match="requires rot.weight"):
            model.load_weights(weights)
    else:
        weights.append(("rot.weight", torch.eye(2)))
        assert "model.layers.1.rot.weight" in model.load_weights(weights)
        torch.testing.assert_close(layer.rot.weight, torch.eye(2))


@pytest.mark.parametrize(
    "name,shape,padded",
    [("weight", (2, 3), True), ("weight", (4, 3), False), ("weight_scale", (2, 1), False), ("input_scale", (), False)],
)
def test_nope_padding_does_not_resize_quantization_metadata(name, shape, padded):
    config = SimpleNamespace(mla_nope=True, qk_rope_head_dim=2, kv_lora_rank=2)
    weight = torch.ones(shape)
    result = glm._pad_nope_kv_a_weight(config, "layer.kv_a_proj_with_mqa." + name, weight)
    if padded:
        torch.testing.assert_close(result, torch.cat((weight, torch.zeros_like(weight))))
    else:
        assert result is weight


@pytest.mark.parametrize("prefix", ["model.layers.", "model.language_model.layers."])
def test_mtp_mapped_weights_preserve_loader_kwargs(prefix):
    model = _module(mtp.Glm5NextMTP)
    model.config = _config()
    model.config.n_routed_experts = None
    model.model = nn.Module()
    layer = nn.Module()
    layer.is_rot_used = False
    layer.mtp_block = nn.Module()
    layer.mtp_block.self_attn = nn.Module()
    layer.mtp_block.self_attn.f_b_proj = nn.Linear(2, 2, bias=False)
    model.model.layers = nn.ModuleDict({"1": layer})
    model.model.mtp_start_layer_idx = model.model.num_mtp_layers = 1
    target = layer.mtp_block.self_attn.f_b_proj.weight
    calls = []

    def loader(param, tensor, *, load_metadata):
        calls.append(load_metadata)
        param.data.copy_(tensor)

    target.weight_loader = loader
    tensor = torch.full((2, 2), 3.0)
    name = prefix + "1.self_attn.forget_gate.f_b_proj.weight"
    loaded = model.load_weights([(name, tensor, {"load_metadata": "modelslim"})])
    assert loaded == {"model.layers.1.mtp_block.self_attn.f_b_proj.weight"}
    assert calls == ["modelslim"]
    torch.testing.assert_close(target, tensor)
