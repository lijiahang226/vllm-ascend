# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton


@pytest.mark.parametrize("max_pool_seq_len", [0, 4, 2050])
def test_pool_selection_matches_dense_scores_and_causal_tail(max_pool_seq_len):
    torch.manual_seed(19)
    dim, pool, topk = 128, 4, 8
    cache = torch.randn(max(1, max_pool_seq_len), 1, 1, dim, dtype=torch.bfloat16, device="npu")
    query = torch.randn(3, 2, dim, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(3, 2, dtype=torch.bfloat16, device="npu")
    positions = torch.tensor([0, 2, max_pool_seq_len * pool + 2], device="npu")
    table = torch.arange(max_pool_seq_len, dtype=torch.int32, device="npu").view(1, -1)
    result = glm5_next_lightning_indexer_triton(
        query,
        cache,
        weights,
        torch.tensor([3], dtype=torch.int32, device="npu"),
        torch.tensor([max_pool_seq_len], dtype=torch.int32, device="npu"),
        table,
        positions,
        index_topk=topk,
        index_kpool=pool,
        max_pool_seq_len=max_pool_seq_len,
    )
    qbar = (query.float() * weights.float().unsqueeze(-1)).sum(1).cpu()
    scores = qbar @ cache[:, 0, 0].float().cpu().T
    for row, pos in enumerate(positions.cpu().tolist()):
        count = min((pos + 1) // pool, max_pool_seq_len)
        selected = torch.topk(scores[row, :count], min(topk // pool, count)).indices.tolist()
        expected = [p * pool + i for p in selected for i in range(pool)]
        expected += list(range((pos + 1) // pool * pool, pos + 1))
        actual = result[row, 0].cpu().tolist()
        assert set(v for v in actual if v >= 0) == set(expected)
        assert sum(v >= 0 for v in actual) == len(expected)
