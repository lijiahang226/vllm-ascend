# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import pytest
import torch
import vllm.config.speculative as speculative_config

import vllm_ascend.models.glm5next.mtp as mtp_module
from vllm_ascend.models.glm5next.config import Glm5NextTextConfig
from vllm_ascend.models.glm5next.model import (
    Glm5NextForConditionalGeneration,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.models.glm5next.mtp import Glm5NextMTP
from vllm_ascend.patch.platform.patch_speculative_config import (
    _normalize_legacy_qwen3_dspark_config,
)


def test_get_spec_layer_idx_accepts_checkpoint_prefixes():
    config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=2,
    )

    assert get_spec_layer_idx_from_weight_name(config, "model.layers.45.enorm.weight") == 45
    assert get_spec_layer_idx_from_weight_name(config, "layers.46.self_attn.q_a_proj.weight") == 46
    assert get_spec_layer_idx_from_weight_name(config, "model.layers.44.mlp.weight") is None
    assert get_spec_layer_idx_from_weight_name(config, "rot.weight") is None


def test_mtp_rewrites_layer_and_shared_weight_names():
    mtp = object.__new__(Glm5NextMTP)

    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.self_attn.q_a_proj.weight",
        )
        == "model.layers.45.mtp_block.self_attn.q_a_proj.weight"
    )
    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.shared_head.norm.weight",
        )
        == "model.layers.45.shared_head.norm.weight"
    )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_mtp_eh_projection_loads_tp_shard_and_gathers_full_output(monkeypatch, tp_size):
    hidden_size = 8
    tp_rank = tp_size - 1
    tp_group = MagicMock(world_size=tp_size, rank_in_group=tp_rank)
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: tp_group)
    monkeypatch.setattr("vllm.distributed.communication_op.get_tp_group", lambda: tp_group)
    monkeypatch.setattr("vllm_ascend.ops.linear_op.get_tp_group", lambda: tp_group)
    monkeypatch.setattr("vllm_ascend.ops.linear_op.enable_dsa_cp", lambda: False)
    monkeypatch.setattr(mtp_module, "current_platform", SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(mtp_module, "RMSNorm", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(mtp_module, "SharedHead", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(mtp_module, "Glm5NextDecoderLayer", lambda **kwargs: torch.nn.Identity())
    config = SimpleNamespace(hidden_size=hidden_size, rms_norm_eps=1e-5, index_topk=128, index_kpool=1)
    quant_config = MagicMock()
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config)),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        quant_config=quant_config,
    )
    layer = mtp_module.Glm5NextMultiTokenPredictorLayer(vllm_config, "model.layers.45")
    projection = layer.eh_proj
    full_weight = torch.arange(hidden_size * hidden_size * 2, dtype=torch.float32).reshape(hidden_size, -1) / 128
    projection.weight.weight_loader(projection.weight, full_weight)
    shard_size = hidden_size // tp_size
    torch.testing.assert_close(projection.weight, full_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size])
    quant_config.get_quant_method.assert_not_called()
    monkeypatch.setattr(
        projection.quant_method,
        "apply",
        lambda layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    inputs = torch.arange(3 * hidden_size * 2, dtype=torch.float32).reshape(3, -1) / 16
    expected = torch.nn.functional.linear(inputs, full_weight)
    tp_group.all_gather.return_value = expected

    output = projection(inputs)

    torch.testing.assert_close(output, expected)
    if tp_size > 1:
        local_output = tp_group.all_gather.call_args.args[0]
        torch.testing.assert_close(local_output, expected[:, tp_rank * shard_size : (tp_rank + 1) * shard_size])
        tp_group.all_gather.assert_called_once()
    else:
        tp_group.all_gather.assert_not_called()


def test_multimodal_mapper_flattens_modelslim_forget_gate_prefix():
    weight_name = "model.language_model.layers.0.self_attn.forget_gate.f_b_proj.weight"

    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name(weight_name)
        == "language_model.model.layers.0.self_attn.f_b_proj.weight"
    )

    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name("model.language_model.layers.1.attn_hc.fn")
        == "language_model.model.layers.1.hc_attn_fn"
    )
    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name("model.language_model.layers.1.ffn_hc.scale")
        == "language_model.model.layers.1.hc_ffn_scale"
    )


def test_glm5_speculative_config_selects_mtp_architecture():
    config = Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        num_nextn_predict_layers=2,
    )

    result = _normalize_legacy_qwen3_dspark_config(config)

    assert result.model_type == "glm5_next_mtp"
    assert result.n_predict == 2
    assert result.architectures == ["Glm5NextMTPModel"]
    assert "glm5_next_mtp" in get_args(speculative_config.MTPModelTypes)
