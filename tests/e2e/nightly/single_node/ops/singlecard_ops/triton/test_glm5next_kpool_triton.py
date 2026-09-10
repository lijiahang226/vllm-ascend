# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.glm5_next_kpool_state_compress import glm5_next_kpool_state_compress_and_write_cache_triton


@pytest.mark.parametrize("pool", [4, 8])
@pytest.mark.parametrize("use_graph", [False, True])
def test_paged_state_long_prefill_padding_and_rollback(pool, use_graph):
    torch.manual_seed(13)
    dim, capacity = 128, pool
    # Physical pages deliberately have padding and request IDs are reordered.
    state = torch.zeros(32, capacity + 2, 2 * dim, device="npu")[:, :capacity]
    cache = torch.zeros(3, 18, 1, dim, dtype=torch.bfloat16, device="npu")[:, :16]
    expected_state, expected_cache = state.cpu(), cache.cpu()
    ape = torch.randn(pool, dim) * 0.1
    state_table = torch.tensor([[5, 9, 3, 12, 7, 16, 2, 20], [11, 8, 1, 18, 4, 22, 15, 23]], dtype=torch.int32)
    cache_blocks = [1, 2]
    history = [{}, {}]
    graph, captured_args = None, None
    graph_capacity = 32

    def run(starts, lengths, invalidate=False):
        nonlocal graph, captured_args
        positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, lengths)])
        num_tokens = positions.numel()
        keys, gates = torch.randn(num_tokens, dim), torch.randn(num_tokens, dim) * 0.1
        ends = torch.tensor(lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
        seq_lens = torch.tensor([s + n for s, n in zip(starts, lengths)], dtype=torch.int32)
        state_slots, indexer_slots = [], []
        cursor = 0
        for req, (start, length) in enumerate(zip(starts, lengths)):
            for local in range(length):
                row, pos = cursor + local, start + local
                history[req][pos] = (keys[row], gates[row])
                state_slots.append(int(state_table[req, pos // capacity]) * capacity + pos % capacity)
                slot = cache_blocks[req] * 16 + pos // pool if (pos + 1) % pool == 0 else -1
                indexer_slots.append(slot)
                if slot >= 0:
                    window = [history[req][p] for p in range(pos - pool + 1, pos + 1)]
                    pooled = (
                        torch.softmax(torch.stack([g for _, g in window]) + ape, dim=0)
                        * torch.stack([k for k, _ in window])
                    ).sum(0)
                    expected_cache[cache_blocks[req], pos // pool, 0] = pooled.bfloat16()
            cursor += length
        if invalidate:
            state_slots[0] = -1
        # Each logical position has its own scheduler-provided physical slot.
        cursor = 0
        for req, (start, length) in enumerate(zip(starts, lengths)):
            for local in range(length):
                row, pos = cursor + local, start + local
                if state_slots[row] >= 0:
                    expected_state[state_slots[row] // capacity, pos % capacity] = torch.cat((keys[row], gates[row]))
            cursor += length
        # Graph-capacity rows have no owning request, even if a stale slot is positive.
        padded_keys = torch.nn.functional.pad(keys, (0, 0, 0, graph_capacity - num_tokens)).npu()
        padded_gates = torch.nn.functional.pad(gates, (0, 0, 0, graph_capacity - num_tokens)).npu()
        args = (
            state,
            cache,
            padded_keys,
            padded_gates,
            ape.npu(),
            torch.cat((positions, torch.zeros(graph_capacity - num_tokens))).long().npu(),
            ends.npu(),
            seq_lens.npu(),
            torch.tensor(state_slots + [0] + [-1] * (graph_capacity - num_tokens - 1), device="npu"),
            state_table.npu(),
            torch.tensor(indexer_slots + [-1] * (graph_capacity - num_tokens), device="npu"),
            pool,
        )
        if use_graph:
            if graph is None:
                captured_args = args
                glm5_next_kpool_state_compress_and_write_cache_triton(*args)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    glm5_next_kpool_state_compress_and_write_cache_triton(*captured_args)
            else:
                for target, value in zip(captured_args[2:-1], args[2:-1]):
                    target.copy_(value)
            graph.replay()
        else:
            glm5_next_kpool_state_compress_and_write_cache_triton(*args)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
        torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0.02, atol=0.02)

    run([0, 0], [1, 3], invalidate=True)
    run([0, 3], [2, 13])  # Historical tail plus a long query spans several state pages.
    run([2, 16], [3, 3])  # Verification writes future candidates into separate paged slots.
    run([3, 17], [1, 3])  # Reject candidates and overwrite their positions.
    state_table[1, : 16 // pool] = 0  # Evicted old pages must not be consulted.
    run([4, 20], [pool, pool])
