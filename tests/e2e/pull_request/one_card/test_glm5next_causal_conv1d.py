# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.utils import enable_custom_op, is_950


def _reference_update(x, weight, state, indices, lengths, accepted):
    """Independent CPU reference, including the sliding MTP history window."""
    output = torch.zeros_like(x)
    width = weight.shape[0]
    start = 0
    for request, (slot, length) in enumerate(zip(indices, lengths)):
        if slot >= 0 and length > 0:
            offset = 0 if accepted is None else accepted[request] - 1
            history = state[slot, offset : offset + width - 1].clone()
            tokens = x[start : start + length]
            sequence = torch.cat((history, tokens))
            convolved = F.conv1d(sequence.T.unsqueeze(0).float(), weight.T.unsqueeze(1).float(), groups=x.shape[1])
            output[start : start + length] = F.silu(convolved).squeeze(0).T.to(x.dtype)
            updated = sequence[-(width - 1) :] if accepted is None else sequence[1:]
            state[slot, : updated.shape[0]].copy_(updated)
        start += length
    return output


@torch.inference_mode()
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("dim", [384, 6144])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
def test_cann_update_output_and_state_writeback(speculative, dim, dtype, use_graph):
    if not is_950():
        pytest.skip("CANN CausalConv1dUpdate currently supports Ascend 950 only")
    enable_custom_op()
    torch.manual_seed(0)
    slots, width, state_len = 5, 4, 5
    page_size = state_len * dim
    storage_offset = 32
    strides = (page_size, dim, 1)
    backing = torch.full((storage_offset + slots * page_size,), -7, device="npu", dtype=dtype)
    state = backing.as_strided((slots, state_len, dim), strides, storage_offset=storage_offset)
    initial_state = torch.randn(state.shape, dtype=dtype)
    state.copy_(initial_state)
    original = backing.cpu()
    reference = initial_state.clone()
    indices_cpu = [3, -1, 1, 2]
    lengths = [3, 0, 2, 1] if speculative else [1, 1, 1, 1]
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="npu", dtype=torch.int32)
    indices = torch.tensor(indices_cpu, device="npu", dtype=torch.int32)
    x_cpu = torch.randn(sum(lengths), dim, dtype=dtype)
    weight_cpu = torch.randn(width, dim, dtype=dtype)
    x, weight = x_cpu.to("npu"), weight_cpu.to("npu")
    # Include full acceptance (3) as well as rejection to a shorter history.
    accepted_cpu = [3, 1, 2, 1] if speculative else None
    accepted = torch.tensor(accepted_cpu, device="npu", dtype=torch.int32) if speculative else None
    kernel_x = x if speculative else x.unsqueeze(1)
    output = torch.zeros_like(kernel_x)

    def run():
        return torch.ops._C_ascend.npu_causal_conv1d_update(
            output,
            kernel_x,
            weight,
            state,
            bias=None,
            query_start_loc=starts if speculative else None,
            cache_indices=indices,
            num_accepted_tokens=accepted,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="silu",
            null_block_id=-1,
            max_query_len=max(lengths),
        )

    if use_graph:
        for _ in range(2):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = run()
        state.copy_(initial_state)

    for _ in range(2):
        expected = _reference_update(x_cpu, weight_cpu, reference, indices_cpu, lengths, accepted_cpu)
        if use_graph:
            graph.replay()
        else:
            actual = run()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.reshape_as(x).cpu(), expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state.cpu(), reference, rtol=0, atol=0)
        expected_backing = original.clone()
        expected_backing.as_strided(state.shape, strides, storage_offset=storage_offset).copy_(reference)
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)


def test_cann_update_meta_preserves_output_alias():
    enable_custom_op()
    output = torch.empty(2, 1, 384, device="meta")
    result = torch.ops._C_ascend.npu_causal_conv1d_update(
        output,
        torch.empty_like(output),
        torch.empty(4, 384, device="meta"),
        torch.empty(5, 3, 384, device="meta"),
        None,
        None,
        None,
        None,
        None,
        None,
        "silu",
        -1,
        1,
    )
    assert result is output
