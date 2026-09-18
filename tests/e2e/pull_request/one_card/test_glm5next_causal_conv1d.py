# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.utils import enable_custom_op


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["prefill", "decode", "spec"])
@pytest.mark.parametrize("dim", [384, 6144])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("invalid_slot", [-1, 7])
def test_paged_conv_state_matches_contiguous_cache(mode, dim, dtype, use_graph, invalid_slot, monkeypatch):
    enable_custom_op()
    torch.manual_seed(0)
    slots, width, state_len = 5, 4, 5
    page_size = state_len * dim + 64
    storage_offset = 32
    backing = torch.full((storage_offset + slots * page_size,), -7, device="npu", dtype=dtype)
    state = backing.as_strided((slots, state_len, dim), (page_size, dim, 1), storage_offset=storage_offset)
    initial_state = torch.randn(state.shape, device="npu", dtype=dtype)
    state.copy_(initial_state)
    original = backing.clone()
    reference = initial_state.clone()
    lengths = [65, 0, 7, 130] if mode == "prefill" else ([3, 0, 3, 3] if mode == "spec" else [1, 0, 1, 1])
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="npu", dtype=torch.int32)
    indices = torch.tensor([[3, 4], [2, 3], [invalid_slot, invalid_slot], [1, 2]], device="npu", dtype=torch.int32)
    x = torch.randn(sum(lengths) + 1, dim, device="npu", dtype=dtype)
    weight = torch.randn(width, dim, device="npu", dtype=dtype)
    initial = torch.tensor([True, True, False, False], device="npu") if mode == "prefill" else None
    accepted = torch.tensor([2, 1, 1, 3], device="npu", dtype=torch.int32) if mode == "spec" else None

    # A page gap must use the original cache directly, including graph capture.
    monkeypatch.setattr("vllm_ascend.models.glm5next.ops.causal_conv1d._copy_conv_state", None)

    def run(cache):
        return causal_conv1d(
            x,
            weight,
            cache,
            starts,
            indices,
            run_mode=0 if mode == "prefill" else 1,
            initial_state_mode=initial,
            num_accepted_tokens=accepted,
        )

    if use_graph:
        for _ in range(2):
            run(state)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run(state)
        state.copy_(initial_state)

    # Reuse the updated state to expose missing writes as well as wrong reads.
    for _ in range(2):
        expected = run(reference)
        if use_graph:
            graph.replay()
        else:
            actual = run(state)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, reference, rtol=0, atol=0)
        expected_backing = original.clone()
        expected_backing.as_strided(state.shape, state.stride(), storage_offset=storage_offset).copy_(reference)
        torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
        # Slots 0, 2 (zero query length), and 4 must retain their old history.
        torch.testing.assert_close(state[[0, 2, 4]], initial_state[[0, 2, 4]], rtol=0, atol=0)
        skipped_start = lengths[0]
        assert torch.count_nonzero(actual[skipped_start : skipped_start + lengths[2]]).item() == 0
        assert torch.count_nonzero(actual[-1]).item() == 0
