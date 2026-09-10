# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded
from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d


@pytest.fixture(scope="module", autouse=True)
def _load_ascendc():
    ensure_c_ascend_loaded("npu_causal_conv1d_custom")


def _reference(x, weight, state, starts, slots, initial, accepted):
    """Independent per-token FP32 depthwise convolution and exact state update."""
    out = torch.zeros_like(x)
    updated = state.clone()
    width = weight.shape[0]
    for row, slot in enumerate(slots):
        begin, end = starts[row : row + 2]
        if slot < 0 or begin == end:
            continue
        offset = 0 if accepted is None else accepted[row] - 1
        history = state[slot, offset : offset + width - 1].clone()
        if initial is not None and not initial[row]:
            history.zero_()
        original_history = history.clone()
        for token in range(begin, end):
            window = torch.cat((history, x[token : token + 1]))
            value = (window.float() * weight.float()).sum(0)
            out[token] = torch.nn.functional.silu(value).to(x.dtype)
            history = window[1:]
        if accepted is None:
            updated[slot, : width - 1] = history
        else:
            saved = torch.cat((original_history[1:], x[begin:end]))
            updated[slot, : saved.shape[0]] = saved
    return out, updated


def _case(mode, layout, dim):
    torch.manual_seed(17)
    x = (torch.randn(10, dim) * 0.2).bfloat16()
    weight = (torch.randn(4, dim) * 0.2).bfloat16()
    initial_state = (torch.randn(5, 6, dim) * 0.2).bfloat16()
    if layout == "SD":
        storage = torch.empty((5, 6, dim), dtype=torch.bfloat16, device="npu")
        state = storage
    elif layout == "DS":
        storage = torch.empty((5, dim, 6), dtype=torch.bfloat16, device="npu")
        state = storage.transpose(1, 2)
    else:
        storage = torch.full((5, 6 * dim + 64), 17, dtype=torch.bfloat16, device="npu")
        state = storage[:, : 6 * dim].view(5, 6, dim)
    state.copy_(initial_state)
    starts = {"prefill": [0, 5, 6, 6, 8], "decode": [0, 1, 2, 2, 3], "mtp": [0, 4, 6, 6, 7]}[mode]
    slots = [2, 0, 3, -1]
    initial = [True, False, True, False] if mode == "prefill" else None
    accepted = [1, 4, 2, 1] if mode == "mtp" else None
    # Match the builder's multi-column cache-index metadata: only column zero
    # selects the conv state; other columns are recurrent-state draft slots.
    ids = torch.tensor([[slot, 4, 1, 3] for slot in slots], dtype=torch.int32, device="npu")
    kwargs = dict(
        run_mode=0 if mode == "prefill" else 1,
        initial_state_mode=None if initial is None else torch.tensor(initial, device="npu"),
        num_accepted_tokens=None if accepted is None else torch.tensor(accepted, dtype=torch.int32, device="npu"),
    )
    device_args = (x.npu(), weight.npu(), state, torch.tensor(starts, dtype=torch.int32, device="npu"), ids)
    reference_args = (x, weight, initial_state, starts, slots, initial, accepted)
    return device_args, kwargs, reference_args, storage


@pytest.mark.parametrize("mode", ["prefill", "decode", "mtp"])
@pytest.mark.parametrize("layout", ["SD", "DS", "padded_SD"])
@pytest.mark.parametrize("dim", [384, 12288])
def test_glm_conv_output_and_original_cache_storage(mode, layout, dim):
    args, kwargs, reference_args, storage = _case(mode, layout, dim)
    address = args[2].data_ptr()
    expected, expected_state = _reference(*reference_args)
    actual = causal_conv1d(*args, **kwargs)
    torch.npu.synchronize()
    assert args[2].data_ptr() == address
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(args[2].cpu(), expected_state, rtol=0, atol=0)
    if layout == "padded_SD":
        assert torch.all(storage[:, -64:] == 17)


@pytest.mark.parametrize("mode", ["prefill", "decode", "mtp"])
@pytest.mark.parametrize("layout", ["SD", "DS", "padded_SD"])
def test_glm_conv_graph_replay_updates_inputs_metadata_and_state(mode, layout):
    args, kwargs, reference_args, storage = _case(mode, layout, 12288)
    x, weight, initial_state, starts, slots, initial, accepted = reference_args
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        causal_conv1d(*args, **kwargs)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = causal_conv1d(*args, **kwargs)
    torch.npu.synchronize()
    for iteration in range(2):
        if iteration:
            x = -x
            slots = [1, 4, 3, -1]
            args[0].copy_(x)
            args[4][:, 0].copy_(torch.tensor(slots, dtype=torch.int32, device="npu"))
            if accepted is not None:
                accepted = [4, 1, 2, 1]
                kwargs["num_accepted_tokens"].copy_(torch.tensor(accepted, dtype=torch.int32, device="npu"))
        args[2].copy_(initial_state)
        expected, expected_state = _reference(x, weight, initial_state, starts, slots, initial, accepted)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(args[2].cpu(), expected_state, rtol=0, atol=0)
        if layout == "padded_SD":
            assert torch.all(storage[:, -64:] == 17)
