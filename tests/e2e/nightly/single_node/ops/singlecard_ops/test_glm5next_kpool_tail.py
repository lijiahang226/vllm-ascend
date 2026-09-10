# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_sfa_nope_integration import (  # noqa: E402
    _assert_matches_reference,
    _copy_case_,
    _make_cpu_case,
    _run_sparse_attention,
    _to_npu,
)
from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded  # noqa: E402
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import append_causal_tail  # noqa: E402


@pytest.fixture(scope="module")
def sparse_attention_op():
    ensure_c_ascend_loaded(required_op="npu_sparse_flash_attention")
    return torch.ops._C_ascend.npu_sparse_flash_attention


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("query_lens", [(3, 3), (1, 1)])
@torch.inference_mode()
def test_glm5next_sparse_mla_includes_unpooled_tail(sparse_attention_op, query_lens, use_graph) -> None:
    def make_case(shift):
        seq_lens = (query_lens[0] + shift, query_lens[1] + 4 + shift)
        positions = [position for length, end in zip(query_lens, seq_lens) for position in range(end - length, end)]
        cpu_case = _make_cpu_case(
            seed=47 + shift,
            query_lens=query_lens,
            seq_lens=seq_lens,
            block_table=[[2, 0], [3, 1]],
            selected_tokens=[list(range(position + 1)) for position in positions],
        )
        # Reproduce PKI's separate history and tail regions, including rows
        # with no complete pool. The dense causal reference includes all tokens.
        raw_indices = torch.full_like(cpu_case["sparse_indices"], -1)
        for row, position in enumerate(positions):
            full_pool_tokens = (position + 1) // 4 * 4
            raw_indices[row, 0, :full_pool_tokens] = torch.arange(full_pool_tokens, dtype=torch.int32)
            tail = torch.arange(full_pool_tokens, position + 1, dtype=torch.int32)
            raw_indices[row, 0, 2048 : 2048 + tail.numel()] = tail
        return cpu_case, raw_indices, torch.tensor(positions)

    cpu_case, raw_cpu, positions_cpu = make_case(0)
    inputs = _to_npu(cpu_case)
    raw_indices = raw_cpu.npu()
    positions = positions_cpu.npu()

    def run_attention():
        inputs["sparse_indices"].copy_(raw_indices)
        append_causal_tail(inputs["sparse_indices"].squeeze(1), positions, 2048, 4)
        return _run_sparse_attention(sparse_attention_op, inputs)[0]

    _assert_matches_reference(run_attention(), cpu_case)
    graph = torch.npu.NPUGraph() if use_graph else None
    try:
        if graph is not None:
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                output = run_attention()
        for shift in (0, 4):
            cpu_case, raw_cpu, positions_cpu = make_case(shift)
            _copy_case_(inputs, cpu_case)
            raw_indices.copy_(raw_cpu.npu())
            positions.copy_(positions_cpu.npu())
            if graph is not None:
                graph.replay()
            else:
                output = run_attention()
            torch_npu.npu.synchronize()
            torch.testing.assert_close(inputs["sparse_indices"].cpu(), cpu_case["sparse_indices"])
            _assert_matches_reference(output, cpu_case)
    finally:
        if graph is not None:
            graph.reset()
        torch_npu.npu.synchronize()
