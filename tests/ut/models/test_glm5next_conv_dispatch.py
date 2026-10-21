# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from unittest.mock import patch

import pytest
import torch

conv_ops = importlib.import_module("vllm_ascend.models.glm5next.ops.causal_conv1d")


@pytest.mark.parametrize("layout", ["page_strided", "row_strided", "dim_first"])
@pytest.mark.parametrize("run_mode", [0, 1])
def test_conv_dispatch_passes_contiguous_indices_and_stages_only_ds(monkeypatch, layout, run_mode):
    dim_first = layout == "dim_first"
    if dim_first:
        state = torch.zeros(4, 8, 3).transpose(1, 2)
    elif layout == "row_strided":
        state = torch.zeros(8, 6, 8)[::2, ::2]
    else:
        state = torch.zeros(8, 3, 8)[::2]
    indices = torch.tensor([[1, 3, 2, 0], [2, 1, 3, 0]], dtype=torch.int32)
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    x = torch.ones(2, 8)
    calls = []

    def copy_state(cache, staging, ids, query_starts, staging_ids, *, write_back):
        calls.append(write_back)
        if not write_back:
            staging.copy_(cache[1:3])
            staging_ids.copy_(torch.tensor([0, 1], dtype=torch.int32))
        else:
            cache[1:3].copy_(staging)

    def prefill(x, weight, bias, **kwargs):
        kernel_indices = kwargs["cache_indices"]
        assert kernel_indices.ndim == 1 and kernel_indices.stride() == (1,)
        assert kwargs["null_block_id"] == (-1 if dim_first else 0)
        expected = torch.tensor([0, 1] if dim_first else [1, 2], dtype=torch.int32)
        torch.testing.assert_close(kernel_indices, expected)
        assert torch.all(kernel_indices != kwargs["null_block_id"])
        if not dim_first:
            assert kwargs["conv_states"] is state
        first_slot = 0 if dim_first else 1
        kwargs["conv_states"][first_slot : first_slot + 2].fill_(5)
        return x + 2

    def decode(x, state, weight, **kwargs):
        output = kwargs.pop("out")
        assert output.is_contiguous() and output.shape == x.shape
        assert output.dtype == x.dtype and output.device == x.device
        assert output.data_ptr() != x.data_ptr()
        kwargs["cache_indices"] = kwargs["conv_state_indices"]
        kwargs["conv_states"] = state
        output.copy_(prefill(x, weight, **kwargs))
        return output

    monkeypatch.setattr(conv_ops, "copy_conv_state", copy_state)
    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", prefill)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", decode)
    with patch.object(torch, "zeros_like", wraps=torch.zeros_like) as zeros:
        result = conv_ops.causal_conv1d(x, torch.ones(4, 8), state, starts, indices, run_mode=run_mode)
    zeros.assert_not_called()
    torch.testing.assert_close(result, x + 2)
    torch.testing.assert_close(x, torch.ones_like(x))
    assert calls == ([False, True] if dim_first else [])
    torch.testing.assert_close(state[1:3], torch.full_like(state[1:3], 5))
    assert torch.count_nonzero(state[0]).item() == 0
    assert torch.count_nonzero(state[3:]).item() == 0


@pytest.mark.parametrize("run_mode", [0, 1])
def test_empty_request_batch_returns_zero_output(monkeypatch, run_mode):
    def unexpected(*args, **kwargs):
        raise AssertionError("An empty request batch must not launch convolution.")

    monkeypatch.setattr(conv_ops, "causal_conv1d_fn", unexpected)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", unexpected)
    output = conv_ops.causal_conv1d(
        torch.full((2, 8), float("nan")),
        torch.ones(4, 8),
        torch.zeros(4, 3, 8),
        torch.tensor([0], dtype=torch.int32),
        torch.empty((0, 1), dtype=torch.int32),
        run_mode=run_mode,
    )
    assert torch.count_nonzero(output) == 0
