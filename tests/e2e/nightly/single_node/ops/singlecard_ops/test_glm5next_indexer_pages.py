# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer addressing across complete physical pages and graph replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("storage_size", [24, 1536])
@pytest.mark.parametrize("page_padding", [0, 128])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_indexer_reads_complete_pages(storage_size, page_padding, use_graph):
    dim, pool_size, topk = 128, 16, 32
    logical_size = storage_size * pool_size
    split = logical_size // 128
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2, max_num_seqs=1),
        model_config=SimpleNamespace(max_model_len=logical_size * 2),
    )
    spec = MLAAttentionSpec(
        block_size=logical_size,
        num_kv_heads=1,
        head_size=dim,
        dtype=torch.bfloat16,
        **({"compress_ratio": pool_size} if vllm_version_is("0.28.0") else {"tokens_per_state": pool_size}),
        model_version="glm5_next",
    )
    builder = AscendIndexerKPoolMetadataBuilder(spec, ["layer.indexer.k_cache"], config, torch.device("npu"))
    pages = torch.tensor([[2, 0]], dtype=torch.int32, device="npu")
    common = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=2,
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        seq_lens=torch.tensor([logical_size + 64], dtype=torch.int32, device="npu"),
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
        positions=torch.tensor([logical_size + 63, 0], device="npu"),
        slot_mapping=torch.tensor([63, -1], device="npu"),
        block_table_tensor=(pages.unsqueeze(-1) * split + torch.arange(split, device="npu")).reshape(1, -1).int(),
    )
    backing = torch.zeros(4, storage_size * dim + page_padding, dtype=torch.bfloat16, device="npu")
    cache = backing[:, : storage_size * dim].view(4, storage_size, 1, dim)
    # Exact BF16 scores select pools on both sides of the logical page boundary.
    cache[2, storage_size - 1, 0, 0] = 2
    cache[0, 2, 0, 0] = 3
    cache[1, 1, 0, 0] = 4
    query = torch.zeros(2, 1, dim, dtype=torch.bfloat16, device="npu")
    query[:, :, 0] = 1
    weights = torch.ones(2, 1, dtype=torch.bfloat16, device="npu")
    metadata = builder.build(0, common)
    table_address = metadata.block_table.data_ptr()

    def run():
        return glm5_next_lightning_indexer_triton(
            query,
            cache,
            weights,
            metadata.cum_query_lens,
            metadata.seq_lens,
            metadata.block_table,
            common.positions,
            index_topk=topk,
            index_kpool=pool_size,
            max_pool_seq_len=2 * storage_size,
        )

    run()
    graph = torch.npu.NPUGraph() if use_graph else None
    try:
        if graph is not None:
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                result = run()
        for step in range(3):
            common.positions[0] = logical_size + 63 + step
            common.seq_lens[0] = logical_size + 64 + step
            common.block_table_tensor[0, :split] = (2 if step == 0 else 1) * split + torch.arange(split, device="npu")
            refreshed = builder.build(0, common)
            assert refreshed.block_table.data_ptr() == table_address
            if graph is None:
                result = run()
            else:
                graph.replay()
            torch.npu.synchronize()
            selected_pools = [storage_size - 1, storage_size + 2] if step == 0 else [1, storage_size + 2]
            expected = {p * pool_size + i for p in selected_pools for i in range(pool_size)}
            expected.update(range(logical_size + 64, logical_size + 64 + step))
            actual = [v for v in result[0, 0].cpu().tolist() if v >= 0]
            assert len(actual) == len(expected)
            assert set(actual) == expected
    finally:
        if graph is not None:
            graph.reset()
        torch.npu.synchronize()
