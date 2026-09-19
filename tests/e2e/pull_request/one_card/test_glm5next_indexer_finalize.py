# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.ops.indexer_finalize import finalize_indices
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture(scope="module", autouse=True)
def initialize_triton_device_properties():
    init_device_properties_triton()


def _case(positions, topk, pool, valid):
    source = torch.full((len(positions), topk + pool - 1), -1, dtype=torch.int32)
    expected = source.clone()
    for row, position in enumerate(positions):
        start = (position + 1) // pool * pool
        groups = list(reversed(range(min(start // pool, topk // pool))))
        history = [group * pool + offset for group in groups for offset in range(pool)]
        tail = list(range(start, position + 1))
        source[row, : len(history)] = torch.tensor(history, dtype=torch.int32)
        source[row, topk : topk + len(tail)] = torch.tensor(tail, dtype=torch.int64).to(torch.int32)
        if row < valid:
            expected[row, : len(history) + len(tail)] = torch.tensor(history + tail, dtype=torch.int64).to(torch.int32)
    return source, expected


@pytest.mark.parametrize("pool", [1, 4, 16])
@pytest.mark.parametrize("topk", [16, 2048])
@pytest.mark.parametrize("copy_output", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("row_repeats", [1, 13])
def test_finalize_indices_exact(pool, topk, copy_output, strided, row_repeats):
    positions = [0, 1, pool - 1, pool, pool * 2 - 2, topk - 1, topk, 131069, 2**31 - 2, 2**32 + 6, 0, 0]
    positions *= row_repeats
    valid = len(positions) - 2
    source, expected = _case(positions, topk, pool, valid)
    rows, width = source.shape
    backing = torch.full((rows + 2, width + (19 if strided else 0)), 91, device="npu", dtype=torch.int32)
    indices = backing[1:-1, :width]
    indices.copy_(source)
    output_width = (width + 127) // 128 * 128
    output_backing = torch.full((rows + 2, output_width + 23), 97, device="npu", dtype=torch.int32)
    output = output_backing[1:-1, :output_width] if copy_output else None
    device_positions = torch.tensor(positions, device="npu", dtype=torch.int64)
    query_ends = torch.tensor([1, valid], device="npu", dtype=torch.int32)

    finalize_indices(indices, device_positions, topk, pool, query_ends, output)

    assert torch.equal(indices.cpu(), expected)
    assert (backing[[0, -1]] == 91).all()
    if strided:
        assert (backing[:, width:] == 91).all()
    if copy_output:
        expected_output = torch.full((rows, output_width), -1, dtype=torch.int32)
        expected_output[:, :width] = expected
        assert torch.equal(output.cpu(), expected_output)
        assert (output_backing[[0, -1]] == 97).all()
        assert (output_backing[:, output_width:] == 97).all()


def test_finalize_indices_graph_replay_reads_updated_metadata():
    pool, topk = 4, 16
    positions = torch.zeros(8, device="npu", dtype=torch.int64)
    query_ends = torch.tensor([0, 8], device="npu", dtype=torch.int32)
    indices = torch.full((8, topk + pool - 1), -1, device="npu", dtype=torch.int32)
    output = torch.empty((8, 128), device="npu", dtype=torch.int32)
    for _ in range(3):
        finalize_indices(indices, positions, topk, pool, query_ends, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        finalize_indices(indices, positions, topk, pool, query_ends, output)
    for count in [8, 1, 7, 0, 8]:
        values = [count + step for step in range(8)]
        source, expected = _case(values, topk, pool, count)
        positions.copy_(torch.tensor(values, dtype=torch.int64))
        query_ends.copy_(torch.tensor([0, count], dtype=torch.int32))
        indices.copy_(source)
        output.fill_(89)
        graph.replay()
        assert torch.equal(indices.cpu(), expected)
        assert torch.equal(output[:, : indices.shape[1]].cpu(), expected)
        assert (output[:, indices.shape[1] :] == -1).all()


def test_finalize_indices_empty_and_small_output():
    indices = torch.empty((0, 19), device="npu", dtype=torch.int32)
    positions = torch.empty(0, device="npu", dtype=torch.int64)
    ends = torch.tensor([0], device="npu", dtype=torch.int32)
    finalize_indices(indices, positions, 16, 4, ends)
    with pytest.raises(ValueError, match="cover all index rows and columns"):
        finalize_indices(indices, positions, 16, 4, ends, torch.empty((0, 16), device="npu", dtype=torch.int32))
