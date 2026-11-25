# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.glm54.loop import DampGate, LoopConfig, two_bank_mla


@pytest.mark.parametrize("window", [1, 3, 128])
def test_two_bank_matches_concatenated_attention_with_ragged_paged_history(window):
    torch.manual_seed(17)
    page_size, latent_dim, rope_dim, heads = 4, 8, 4, 2
    first = tuple(torch.randn(7, page_size, 1, dim) for dim in (latent_dim, rope_dim))
    local = tuple(torch.randn_like(value) for value in first)
    first_before = tuple(value.clone() for value in first)
    block_table = torch.tensor([[4, 1, 6], [3, 5, 2]])
    local_table = torch.tensor([[1, 6, 4], [5, 2, 3]])
    starts = torch.tensor([0, 3, 4])
    lengths = torch.tensor([7, 10])
    q_latent = torch.randn(4, heads, latent_dim)
    q_rope = torch.randn(4, heads, rope_dim)
    indices = torch.tensor([[[0, 4, -1, 6]], [[1, 5, 6, -1]], [[0, 6, 2, -1]], [[9, 3, 1, -1]]])
    expected = []
    for token, (request, position) in enumerate([(0, 4), (0, 5), (0, 6), (1, 9)]):
        keys, values = [], []
        sparse_positions = [value for value in indices[token].flatten().tolist() if 0 <= value <= position]
        local_positions = list(range(max(0, position - window + 1), position + 1))
        for bank, positions, table in ((first, sparse_positions, block_table), (local, local_positions, local_table)):
            for logical_position in positions:
                page = table[request, logical_position // page_size]
                offset = logical_position % page_size
                latent, rope = (component[page, offset, 0] for component in bank)
                keys.append(torch.cat((latent, rope)))
                values.append(latent)
        query = torch.cat((q_latent[token], q_rope[token]), dim=-1)
        scores = query @ torch.stack(keys).T * 0.3
        expected.append(scores.softmax(-1) @ torch.stack(values))
    result = two_bank_mla(
        q_latent,
        q_rope,
        first,
        local,
        indices,
        block_table,
        starts,
        lengths,
        window,
        0.3,
        local_block_table=local_table,
    )
    torch.testing.assert_close(result, torch.stack(expected))
    for actual, original in zip(first, first_before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_overlap_preserves_both_representations_and_excludes_future():
    first = (torch.tensor([[[[2.0]], [[float("nan")]]]]), torch.zeros(1, 2, 1, 1))
    local = (torch.tensor([[[[6.0]], [[float("nan")]]]]), torch.zeros(1, 2, 1, 1))
    result = two_bank_mla(
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        first,
        local,
        torch.tensor([[[0, 1, -1]]]),
        torch.tensor([[0]]),
        torch.tensor([0, 1]),
        torch.tensor([1]),
        128,
        1.0,
    )
    torch.testing.assert_close(result, torch.tensor([[[4.0]]]))


def test_split_lse_merge_matches_union():
    torch.manual_seed(29)
    scores = torch.randn(2, 9) * 30
    values = torch.randn(9, 5)
    left, right = scores[:, :4], scores[:, 4:]
    outputs = torch.stack((left.softmax(-1) @ values[:4], right.softmax(-1) @ values[4:]), dim=1)
    lse = torch.stack((left.logsumexp(-1), right.logsumexp(-1)), dim=1)
    merged = (outputs * lse.softmax(-1).unsqueeze(-1)).sum(1)
    torch.testing.assert_close(merged, scores.softmax(-1) @ values)


def test_empty_queries():
    q = torch.empty(0, 2, 8)
    bank = (torch.empty(0, 4, 1, 8), torch.empty(0, 4, 1, 4))
    result = two_bank_mla(
        q,
        torch.empty(0, 2, 4),
        bank,
        bank,
        torch.empty(0, 1, 4, dtype=torch.int32),
        torch.empty(0, 0, dtype=torch.int32),
        torch.tensor([0]),
        torch.tensor([]),
        128,
        1.0,
    )
    assert result.shape == q.shape


def test_gate_uses_pass_input_and_independent_parameters():
    gate = DampGate(4, 1e-6)
    other = DampGate(4, 1e-6)
    with torch.no_grad():
        gate.proj.weight.zero_()
    inputs = torch.randn(3, 4)
    candidate = torch.randn(3, 4)
    torch.testing.assert_close(gate(inputs, candidate), (inputs + candidate) / 2)
    assert gate.proj.weight.data_ptr() != other.proj.weight.data_ptr()


@pytest.mark.parametrize("start,end,window", [(-1, 4, 128), (2, 2, 128), (2, 7, 128), (2, 4, 0), (True, 4, 128)])
def test_invalid_loop_config(start, end, window):
    config = SimpleNamespace(
        glm54_dummy=True, loop_start_layer=start, loop_end_layer=end, loop_window_size=window, num_hidden_layers=6
    )
    with pytest.raises(ValueError):
        LoopConfig.from_hf_config(config)


def test_real_checkpoints_are_not_accepted_as_dummy():
    with pytest.raises(ValueError, match="dummy checkpoint"):
        LoopConfig.from_hf_config(SimpleNamespace(glm54_dummy=False))
