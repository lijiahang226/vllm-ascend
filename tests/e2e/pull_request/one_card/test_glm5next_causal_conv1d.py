# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.utils import enable_custom_op


def _reference_update(x, weight, state, indices, lengths, accepted, initial):
    """Independent CPU reference, including the sliding MTP history window."""
    output = torch.zeros_like(x)
    width = weight.shape[0]
    start = 0
    for request, (slot, length) in enumerate(zip(indices, lengths)):
        if slot >= 0 and length > 0:
            offset = 0 if accepted is None else accepted[request] - 1
            history = state[slot, offset : offset + width - 1].clone()
            if initial is not None and not initial[request]:
                history.zero_()
            tokens = x[start : start + length]
            sequence = torch.cat((history, tokens))
            convolved = F.conv1d(sequence.T.unsqueeze(0).float(), weight.T.unsqueeze(1).float(), groups=x.shape[1])
            output[start : start + length] = F.silu(convolved).squeeze(0).T.to(x.dtype)
            updated = sequence[-(width - 1) :] if accepted is None else sequence[1:]
            state[slot, : updated.shape[0]].copy_(updated)
        start += length
    return output


@torch.inference_mode()
@pytest.mark.parametrize("layout", ["contiguous", "paged", "transposed"])
@pytest.mark.parametrize("mode", ["prefill", "decode", "spec"])
@pytest.mark.parametrize("dim", [384, 6144])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
def test_conv_output_and_original_state_writeback(layout, mode, dim, dtype, use_graph):
    enable_custom_op()
    torch.manual_seed(0)
    slots, width, state_len = 5, 4, 6
    page_size = state_len * dim + (64 if layout != "contiguous" else 0)
    storage_offset = 32
    strides = (page_size, 1, state_len) if layout == "transposed" else (page_size, dim, 1)
    backing = torch.full((storage_offset + slots * page_size,), -7, device="npu", dtype=dtype)
    state = backing.as_strided((slots, state_len, dim), strides, storage_offset=storage_offset)
    initial_state = torch.randn(state.shape, dtype=dtype)
    state.copy_(initial_state)
    original = backing.cpu()
    reference = initial_state.clone()
    indices_cpu = [3, -1, 1, 2]
    lengths = {"prefill": [7, 1, 4, 0], "decode": [1, 1, 1, 0], "spec": [4, 0, 2, 1]}[mode]
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="npu", dtype=torch.int32)
    # GDN metadata can pass a block table; convolution uses its first column.
    indices = torch.tensor([[slot, -1] for slot in indices_cpu], device="npu", dtype=torch.int32)
    x_cpu = torch.randn(sum(lengths), dim, dtype=dtype)
    weight_cpu = torch.randn(width, dim, dtype=dtype)
    x, weight = x_cpu.to("npu"), weight_cpu.to("npu")
    # Include full acceptance (4) and rejection to a shorter history window.
    accepted_cpu = [4, 1, 2, 1] if mode == "spec" else None
    accepted = torch.tensor(accepted_cpu, device="npu", dtype=torch.int32) if accepted_cpu is not None else None
    initial_cpu = [True, False, False, True] if mode == "prefill" else None
    initial = torch.tensor(initial_cpu, device="npu", dtype=torch.bool) if initial_cpu is not None else None

    def run():
        return causal_conv1d(
            x,
            weight,
            state,
            starts,
            indices,
            run_mode=0 if mode == "prefill" else 1,
            initial_state_mode=initial,
            num_accepted_tokens=accepted,
        )

    if use_graph:
        for _ in range(2):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        backing.copy_(original)

    for _ in range(2):
        expected = _reference_update(x_cpu, weight_cpu, reference, indices_cpu, lengths, accepted_cpu, initial_cpu)
        if use_graph:
            graph.replay()
        else:
            actual = run()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state.cpu(), reference, rtol=0, atol=0)
        expected_backing = original.clone()
        expected_backing.as_strided(state.shape, strides, storage_offset=storage_offset).copy_(reference)
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)
