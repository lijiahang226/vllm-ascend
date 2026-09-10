# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import vllm_ascend.models.glm5next.multimodal as vision


@pytest.mark.parametrize("vision_limit,text_limit,expected", [(None, 7, 7), (5, 7, 5), (None, None, None)])
def test_vision_swiglu_config_fallback(monkeypatch, vision_limit, text_limit, expected):
    config = SimpleNamespace(
        swiglu_limit=vision_limit,
        hidden_size=8,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=8,
        in_channels=3,
        attention_bias=False,
        intermediate_size=16,
        depth=1,
        projection_intermediate_size=8,
    )
    monkeypatch.setattr(vision, "is_vit_use_data_parallel", lambda: True)
    for name in ("Glm5NextVisionPatchEmbed", "get_rope", "Conv2dLayer", "get_vit_attn_backend"):
        monkeypatch.setattr(vision, name, lambda **kwargs: nn.Identity())
    block = MagicMock(side_effect=lambda **kwargs: nn.Identity())
    merger = MagicMock(side_effect=lambda **kwargs: nn.Identity())
    monkeypatch.setattr(vision, "AscendGlm5NextVisionBlock", block)
    monkeypatch.setattr(vision, "AscendGlm5NextVisionPatchMerger", merger)
    if expected is None:
        with pytest.raises(ValueError, match="swiglu_limit"):
            vision.AscendGlm5NextVisionTransformer(SimpleNamespace(swiglu_limit=text_limit), config)
        block.assert_not_called()
    else:
        tower = vision.AscendGlm5NextVisionTransformer(SimpleNamespace(swiglu_limit=text_limit), config, norm_eps=1e-6)
        assert block.call_args.kwargs["swiglu_limit"] == expected
        assert merger.call_args.kwargs["swiglu_limit"] == expected
        assert block.call_args.kwargs["norm_eps"] == 1e-6
        assert tower.post_layernorm.variance_epsilon == 1e-6


def test_vision_norm_and_clamped_swiglu_bfloat16():
    norm = vision.Glm5NextVisionRMSNorm(4, eps=1e-6).to(dtype=torch.bfloat16)
    norm.weight.data.fill_(1)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)
    output = norm(x)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    torch.testing.assert_close(output, expected)
    assert output.dtype == x.dtype
    assert set(dict(norm.named_parameters())) == {"weight"}
    gate_up = torch.tensor([[20, -20, 20, -20]], dtype=torch.bfloat16)
    actual = vision.Glm5NextSiluAndMul(5)(gate_up)
    gate, up = gate_up.chunk(2, -1)
    expected = torch.nn.functional.silu(gate.clamp(max=5)) * up.clamp(min=-5, max=5)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "source,shard",
    [
        ("attn.q.weight", "q"),
        ("attn.k.weight", "k"),
        ("attn.v.weight", "v"),
        ("mlp.gate_proj.weight", 0),
        ("mlp.up_proj.weight", 1),
    ],
)
def test_vision_loader_preserves_packed_shards(source, shard):
    tower = vision.AscendGlm5NextVisionTransformer.__new__(vision.AscendGlm5NextVisionTransformer)
    nn.Module.__init__(tower)
    target, mapped_shard = tower._map_weight_name("blocks.0." + source)
    loader = MagicMock()
    param = SimpleNamespace(weight_loader=loader)
    tower.named_parameters = lambda **kwargs: [(target, param)]
    tensor = torch.ones(2, 2)
    assert tower.load_weights([("blocks.0." + source, tensor)]) == {target}
    assert mapped_shard == shard
    loader.assert_called_once_with(param, tensor, shard)


def test_vision_loader_rejects_duplicate_and_unknown_weights():
    tower = vision.AscendGlm5NextVisionTransformer.__new__(vision.AscendGlm5NextVisionTransformer)
    nn.Module.__init__(tower)
    param = nn.Parameter(torch.empty(2))
    tower.named_parameters = lambda **kwargs: [("post_layernorm.weight", param)]
    with pytest.raises(ValueError, match="Duplicate"):
        tower.load_weights([("post_layernorm.weight", torch.ones(2))] * 2)
    with pytest.raises(ValueError, match="Unexpected"):
        tower.load_weights([("post_layernorm.bias", torch.ones(2))])
