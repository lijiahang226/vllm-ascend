# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 checkpoint loading contract for the GLM-5.3-Flash MLA projections.

``Glm5NextDecoderLayer`` builds the MLA attention with ``quant_config=None``, so
``q_a_proj`` / ``kv_a_proj_with_mqa`` (fused into ``fused_qkv_a_proj``),
``q_b_proj`` and ``o_proj`` stay BF16 in the model. A ModelSlim W8A8_MXFP8
checkpoint ships those as ``weight`` (e4m3) + ``weight_scale`` (uint8 E8M0, one
exponent per 32-column group) instead of the native FP8 ``weight_scale_inv``
pair, so the loader has to dequantize both conventions.

Before the fix the MXFP8 scale name was not recognized: the scale fell through
the stacked mapping (the BF16 target has no ``weight_scale`` param) into
``params_dict[name]`` and raised
``KeyError: 'layers.<i>.self_attn.q_a_proj.weight_scale'`` /
``...kv_a_proj_with_mqa.weight_scale``, while the fp8 ``weight`` half was
buffered and dropped silently.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.glm5next.model import (
    Glm5NextModel,
    _dequant_mxfp8_block,
    _raise_if_unpaired_fp8_weights,
    _try_load_fp8_attn_proj,
)

HIDDEN = 64
Q_LORA = 4
KV_LORA = 2
MXFP8_GROUP = 32
NOPE_PAD = 2
BLOCK_FP8_BLOCK = 128


def _mxfp8_pair(out_dim, in_dim, seed):
    """Return an MXFP8 (e4m3 weight, uint8 E8M0 scale) checkpoint pair."""
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(out_dim, in_dim, generator=generator) * 4.0).to(torch.float8_e4m3fn)
    groups = -(-in_dim // MXFP8_GROUP)
    exponent = torch.randint(120, 132, (out_dim, groups), generator=generator, dtype=torch.int32)
    return weight, exponent.to(torch.uint8)


def _expected_mxfp8(weight, scale):
    """Independent reference decode: ``2 ** (exponent - 127)`` over each group."""
    expected = torch.zeros(weight.shape, dtype=torch.bfloat16)
    for row in range(weight.shape[0]):
        for group in range(scale.shape[1]):
            start = group * MXFP8_GROUP
            if start >= weight.shape[1]:
                break
            stop = min(start + MXFP8_GROUP, weight.shape[1])
            multiplier = float(2.0 ** (int(scale[row, group]) - 127))
            expected[row, start:stop] = (weight[row, start:stop].to(torch.float32) * multiplier).to(torch.bfloat16)
    return expected


def _expected_block_fp8(weight, scale_inv):
    """Independent reference decode of a 128x128 block-FP8 weight."""
    out_dim, in_dim = weight.shape
    expected = torch.zeros(weight.shape, dtype=torch.bfloat16)
    for out_block in range(scale_inv.shape[0]):
        rows = slice(out_block * BLOCK_FP8_BLOCK, min((out_block + 1) * BLOCK_FP8_BLOCK, out_dim))
        if rows.start >= out_dim:
            break
        for in_block in range(scale_inv.shape[1]):
            cols = slice(in_block * BLOCK_FP8_BLOCK, min((in_block + 1) * BLOCK_FP8_BLOCK, in_dim))
            if cols.start >= in_dim:
                break
            multiplier = float(scale_inv[out_block, in_block])
            expected[rows, cols] = (weight[rows, cols].to(torch.float32) * multiplier).to(torch.bfloat16)
    return expected


def _fused_param(calls, in_dim=HIDDEN, kv_pad=0):
    """Stand-in for the merged ``fused_qkv_a_proj`` parameter."""
    param = nn.Parameter(torch.zeros(Q_LORA + KV_LORA + kv_pad, in_dim, dtype=torch.bfloat16), requires_grad=False)

    def weight_loader(loaded_param, loaded_weight, shard_id):
        calls.append(("fused_qkv_a_proj", shard_id))
        offset = 0 if shard_id == 0 else Q_LORA
        loaded_param.data[offset : offset + loaded_weight.shape[0]].copy_(loaded_weight)

    param.weight_loader = weight_loader
    return param


def _plain_param(name, shape, calls):
    param = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)

    def weight_loader(loaded_param, loaded_weight, shard_id=None):
        calls.append((name, shard_id))
        loaded_param.data.copy_(loaded_weight)

    param.weight_loader = weight_loader
    return param


def _targets(calls, with_scale_params=False, kv_pad=0):
    """BF16 MLA targets plus the checkpoint-named pairs that feed them."""
    params_dict = {
        "layers.0.self_attn.fused_qkv_a_proj.weight": _fused_param(calls, kv_pad=kv_pad),
        "layers.0.self_attn.q_b_proj.weight": _plain_param("q_b_proj", (8, Q_LORA), calls),
        "layers.0.self_attn.o_proj.weight": _plain_param("o_proj", (HIDDEN, 8), calls),
    }
    if with_scale_params:
        # Quantized target: the normal stacked/direct path owns the pair.
        params_dict["layers.0.self_attn.fused_qkv_a_proj.weight_scale"] = nn.Parameter(torch.zeros(1))
        params_dict["layers.0.self_attn.q_b_proj.weight_scale"] = nn.Parameter(torch.zeros(1))
        params_dict["layers.0.self_attn.o_proj.weight_scale"] = nn.Parameter(torch.zeros(1))
    pairs = [
        ("layers.0.self_attn.q_a_proj", *_mxfp8_pair(Q_LORA, HIDDEN, 1)),
        ("layers.0.self_attn.kv_a_proj_with_mqa", *_mxfp8_pair(KV_LORA, HIDDEN, 2)),
        ("layers.0.self_attn.q_b_proj", *_mxfp8_pair(8, Q_LORA, 3)),
        ("layers.0.self_attn.o_proj", *_mxfp8_pair(HIDDEN, 8, 4)),
    ]
    return params_dict, pairs


def _weights(pairs, scale_suffix=".weight_scale"):
    weights = []
    for prefix, weight, scale in pairs:
        weights.append((f"{prefix}.weight", weight))
        weights.append((f"{prefix}{scale_suffix}", scale))
    return weights


def _run(params_dict, pairs, kv_a_pad_size=0, scale_suffix=".weight_scale"):
    buf: dict = {}
    loaded: set[str] = set()
    consumed = []
    for name, tensor in _weights(pairs, scale_suffix):
        consumed.append(_try_load_fp8_attn_proj(name, tensor, buf, params_dict, loaded, kv_a_pad_size))
    return consumed, buf, loaded


def test_dequant_mxfp8_block_matches_reference_including_partial_group():
    # in_dim = 40 spans a full 32-column group plus an 8-column tail.
    weight, scale = _mxfp8_pair(3, 40, 7)
    decoded = _dequant_mxfp8_block(weight, scale)
    assert decoded.dtype == torch.bfloat16
    torch.testing.assert_close(decoded, _expected_mxfp8(weight, scale))


def test_dequant_mxfp8_block_recovers_group_size_from_shapes():
    """An export with a different grouping still decodes in whole groups."""
    in_dim, group = 128, 64
    weight, _ = _mxfp8_pair(2, in_dim, 9)
    scale = torch.randint(120, 132, (2, in_dim // group), dtype=torch.int32).to(torch.uint8)
    decoded = _dequant_mxfp8_block(weight, scale)

    expected = torch.zeros(weight.shape, dtype=torch.bfloat16)
    for row in range(weight.shape[0]):
        for group_idx in range(scale.shape[1]):
            start = group_idx * group
            multiplier = float(2.0 ** (int(scale[row, group_idx]) - 127))
            expected[row, start : start + group] = (
                weight[row, start : start + group].to(torch.float32) * multiplier
            ).to(torch.bfloat16)
    torch.testing.assert_close(decoded, expected)


def test_mxfp8_pair_dequantizes_into_bf16_targets():
    calls: list = []
    params_dict, pairs = _targets(calls)
    consumed, buf, loaded = _run(params_dict, pairs)

    assert consumed == [True] * len(consumed)
    assert buf == {}
    assert loaded == {
        "layers.0.self_attn.fused_qkv_a_proj.weight",
        "layers.0.self_attn.q_b_proj.weight",
        "layers.0.self_attn.o_proj.weight",
    }
    # q_a / kv_a land in the fused projection with their shard ids.
    assert calls == [("fused_qkv_a_proj", 0), ("fused_qkv_a_proj", 1), ("q_b_proj", None), ("o_proj", None)]

    fused = params_dict["layers.0.self_attn.fused_qkv_a_proj.weight"]
    q_a_weight, q_a_scale = pairs[0][1], pairs[0][2]
    kv_a_weight, kv_a_scale = pairs[1][1], pairs[1][2]
    torch.testing.assert_close(fused[:Q_LORA], _expected_mxfp8(q_a_weight, q_a_scale))
    torch.testing.assert_close(fused[Q_LORA:], _expected_mxfp8(kv_a_weight, kv_a_scale))
    torch.testing.assert_close(
        params_dict["layers.0.self_attn.q_b_proj.weight"], _expected_mxfp8(pairs[2][1], pairs[2][2])
    )
    torch.testing.assert_close(
        params_dict["layers.0.self_attn.o_proj.weight"], _expected_mxfp8(pairs[3][1], pairs[3][2])
    )


def test_mxfp8_scale_never_falls_through_to_params_dict():
    """The reported failure: the scale must be consumed, not looked up by name."""
    calls: list = []
    params_dict, pairs = _targets(calls)
    buf: dict = {}
    loaded: set[str] = set()
    for name, tensor in _weights(pairs):
        if name.endswith(".weight_scale"):
            # No BF16 MLA target owns a scale, which is why the old loader fell
            # through to ``params_dict[name]``.
            assert name not in params_dict
        assert _try_load_fp8_attn_proj(name, tensor, buf, params_dict, loaded, 0) is True
    assert buf == {}
    assert loaded == {
        "layers.0.self_attn.fused_qkv_a_proj.weight",
        "layers.0.self_attn.q_b_proj.weight",
        "layers.0.self_attn.o_proj.weight",
    }


def test_native_block_fp8_pair_still_dequantizes():
    calls: list = []
    in_dim = 2 * BLOCK_FP8_BLOCK
    params_dict = {"layers.0.self_attn.fused_qkv_a_proj.weight": _fused_param(calls, in_dim=in_dim)}
    generator = torch.Generator().manual_seed(11)
    weight = (torch.randn(KV_LORA, in_dim, generator=generator) * 4.0).to(torch.float8_e4m3fn)
    scale_inv = (torch.rand(1, 2, generator=generator) + 0.01).to(torch.float32)
    buf: dict = {}
    loaded: set[str] = set()
    assert _try_load_fp8_attn_proj("layers.0.self_attn.kv_a_proj_with_mqa.weight", weight, buf, params_dict, loaded, 0)
    assert _try_load_fp8_attn_proj(
        "layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv", scale_inv, buf, params_dict, loaded, 0
    )
    assert buf == {}
    assert calls == [("fused_qkv_a_proj", 1)]
    fused = params_dict["layers.0.self_attn.fused_qkv_a_proj.weight"]
    torch.testing.assert_close(fused[Q_LORA:], _expected_block_fp8(weight, scale_inv))


def test_quantized_target_defers_to_the_normal_path():
    calls: list = []
    params_dict, pairs = _targets(calls, with_scale_params=True)
    buf: dict = {}
    loaded: set[str] = set()
    for name, tensor in _weights(pairs):
        assert _try_load_fp8_attn_proj(name, tensor, buf, params_dict, loaded, 0) is False
    assert buf == {}
    assert loaded == set()
    assert calls == []


def test_mxfp8_kv_a_keeps_nope_rope_padding():
    calls: list = []
    params_dict, pairs = _targets(calls, kv_pad=NOPE_PAD)
    _, buf, _ = _run(params_dict, pairs, kv_a_pad_size=NOPE_PAD)
    assert buf == {}
    fused = params_dict["layers.0.self_attn.fused_qkv_a_proj.weight"]
    kv_a_weight, kv_a_scale = pairs[1][1], pairs[1][2]
    torch.testing.assert_close(fused[Q_LORA : Q_LORA + KV_LORA], _expected_mxfp8(kv_a_weight, kv_a_scale))
    assert torch.count_nonzero(fused[Q_LORA + KV_LORA :]) == 0


def test_unpaired_fp8_weight_is_reported():
    pending = {"layers.0.self_attn": {"o_proj": {"weight": torch.zeros(2, 2)}}}
    with pytest.raises(ValueError, match="layers.0.self_attn.o_proj"):
        _raise_if_unpaired_fp8_weights(pending)
    _raise_if_unpaired_fp8_weights({})


@pytest.mark.parametrize(
    ("in_dim", "scale_cols"),
    [
        (4096, 128),  # q_a_proj / kv_a_proj_with_mqa of the released checkpoint
        (1536, 48),  # q_b_proj
        (16384, 512),  # o_proj
    ],
)
def test_mxfp8_group_size_matches_released_checkpoint_geometry(in_dim, scale_cols):
    """The exported scale column count must decode as 32-column MX groups."""
    weight, _ = _mxfp8_pair(1, in_dim, 5)
    exponent = torch.randint(120, 132, (1, scale_cols), dtype=torch.int32)
    scale = exponent.to(torch.uint8)
    decoded = _dequant_mxfp8_block(weight, scale)
    torch.testing.assert_close(decoded, _expected_mxfp8(weight, scale))


def _stub_model():
    """Minimal Glm5NextModel tree holding the three BF16 MLA targets."""
    calls: list = []
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        is_moe=False,
        is_linear_attn=False,
        mla_nope=False,
        qk_rope_head_dim=0,
        num_nextn_predict_layers=0,
    )
    attention = nn.Module()
    for name, param in (
        ("fused_qkv_a_proj", _fused_param(calls)),
        ("q_b_proj", _plain_param("q_b_proj", (8, Q_LORA), calls)),
        ("o_proj", _plain_param("o_proj", (HIDDEN, 8), calls)),
    ):
        holder = nn.Module()
        holder.weight = param
        attention.add_module(name, holder)
    layer = nn.Module()
    layer.add_module("self_attn", attention)
    model.add_module("layers", nn.ModuleList([layer]))
    return model


def test_glm5next_model_accepts_mxfp8_checkpoint():
    """End-to-end regression for the reported KeyError on the raw checkpoint name."""
    model = _stub_model()
    calls: list = []
    _, pairs = _targets(calls)
    loaded = Glm5NextModel.load_weights(model, _weights(pairs))

    assert loaded == {
        "layers.0.self_attn.fused_qkv_a_proj.weight",
        "layers.0.self_attn.q_b_proj.weight",
        "layers.0.self_attn.o_proj.weight",
    }
    q_a_weight, q_a_scale = pairs[0][1], pairs[0][2]
    kv_a_weight, kv_a_scale = pairs[1][1], pairs[1][2]
    parameters = dict(model.named_parameters())
    fused = parameters["layers.0.self_attn.fused_qkv_a_proj.weight"]
    torch.testing.assert_close(fused[:Q_LORA], _expected_mxfp8(q_a_weight, q_a_scale))
    torch.testing.assert_close(fused[Q_LORA:], _expected_mxfp8(kv_a_weight, kv_a_scale))
    assert "layers.0.self_attn.kv_a_proj_with_mqa.weight_scale" not in parameters
