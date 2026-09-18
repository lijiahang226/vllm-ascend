# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.models.glm5next.ops.causal_conv1d as conv


@pytest.mark.parametrize("layout", ["contiguous", "paged", "transposed"])
@pytest.mark.parametrize("mode", ["prefill", "decode", "spec"])
def test_conv_passes_original_state_view_and_indices_to_kernel(monkeypatch, layout, mode):
    slots, state_len, dim = 5, 5, 64
    page_size = state_len * dim + (64 if layout != "contiguous" else 0)
    # A nonzero storage offset catches accidental access to the backing's base.
    backing = torch.full((32 + slots * page_size,), -7.0, dtype=torch.bfloat16)
    strides = (page_size, 1, state_len) if layout == "transposed" else (page_size, dim, 1)
    state = backing.as_strided((slots, state_len, dim), strides, storage_offset=32)
    saved = backing.clone()
    tokens_per_request = 3 if mode == "spec" else 1
    starts = torch.tensor([0, tokens_per_request, 2 * tokens_per_request, 2 * tokens_per_request])
    indices = torch.tensor([[3, 4], [-1, -1], [1, 2]], dtype=torch.int32)
    x = torch.ones(2 * tokens_per_request + 1, dim, dtype=state.dtype)
    weight = torch.ones(4, dim, dtype=state.dtype)
    initial = torch.tensor([True, False, True]) if mode == "prefill" else None
    accepted = torch.tensor([2, 1, 1]) if mode == "spec" else None

    def kernel(output, x_arg, weight_arg, **kwargs):
        kernel_state = kwargs["conv_state"]
        kernel_indices = kwargs["cache_indices_opt"]
        assert kernel_state is state and kernel_indices is indices
        assert x_arg is x and weight_arg is weight
        assert kwargs["query_start_loc_opt"] is starts
        assert kwargs["initial_state_mode_opt"] is initial
        assert kwargs["num_accepted_tokens_opt"] is accepted
        assert kwargs["run_mode"] == (0 if mode == "prefill" else 1)
        assert torch.count_nonzero(output) == 0
        kernel_state[3].fill_(17)
        # Returning a different tensor ensures graph functionalization's op
        # result is consumed; padding and skipped requests remain zero.
        result = output.clone()
        result[:tokens_per_request] = 2
        return result

    op = Mock(side_effect=kernel)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_causal_conv1d_custom", op, raising=False)
    out = conv.causal_conv1d(
        x,
        weight,
        state,
        starts,
        indices,
        run_mode=0 if mode == "prefill" else 1,
        initial_state_mode=initial,
        num_accepted_tokens=accepted,
    )
    op.assert_called_once()
    expected = saved.as_strided(state.shape, strides, storage_offset=32)
    expected[3].fill_(17)
    torch.testing.assert_close(backing, saved)
    torch.testing.assert_close(out[:tokens_per_request], torch.full_like(out[:tokens_per_request], 2))
    assert torch.count_nonzero(out[tokens_per_request:]) == 0


def test_empty_batch_does_not_call_kernel(monkeypatch):
    op = Mock(side_effect=AssertionError("Empty batches must not launch a kernel"))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_causal_conv1d_custom", op, raising=False)
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    out = conv.causal_conv1d(
        x,
        torch.ones(4, 64, dtype=x.dtype),
        torch.zeros(2, 3, 64, dtype=x.dtype),
        torch.zeros(1, dtype=torch.int32),
        torch.empty(0, 1, dtype=torch.int32),
        run_mode=1,
    )
    torch.testing.assert_close(out, torch.zeros_like(x))
    op.assert_not_called()
