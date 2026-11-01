# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from tests.ut.models.test_glm5next_kda_projection_tp import CollectiveReference
from vllm_ascend.models.glm5next import kda_state_tp


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("window", [1, 6])
@pytest.mark.parametrize("conv", [False, True])
def test_state_transfer_preserves_strided_owner_pages_and_selected_rollback(monkeypatch, size, window, conv):
    coordinator = CollectiveReference(size)
    heads, dim, requests, history = 8, 2, 3, 8
    generator = torch.Generator().manual_seed(741)
    shape = (history, 3 * heads * dim) if conv else (heads, dim, dim)
    owners = [torch.randn(9, *shape, 2, generator=generator)[..., 0] for _ in range(size)]
    originals = [cache.clone() for cache in owners]
    monkeypatch.setattr(
        kda_state_tp.KDATPLayout,
        "_all_to_all",
        lambda self, x: coordinator.exchange(self.group.rank_in_group, "all_to_all", x),
    )

    def worker(rank):
        group = SimpleNamespace(world_size=size, rank_in_group=rank)
        group.all_gather = lambda x, dim: coordinator.exchange(rank, "gather", x)
        helper = kda_state_tp.KDAStateTP(group, heads, dim, max_requests=requests, state_window=window)
        local_shape = (history, 3 * heads * dim // size) if conv else (heads // size, dim, dim)
        cache = torch.full((1 + size * requests * window, *local_shape, 2), -999.0)[..., 0]
        owner_ids = torch.tensor([2, 5, 0], dtype=torch.int32)
        slots = torch.tensor([1, 0, -1], dtype=torch.int32)
        helper.load_pages(owners[rank], cache, owner_ids, slots, conv=conv)
        gathered_indices = helper.gather_state_indices(slots)
        expected_indices = torch.cat(
            [
                torch.tensor(
                    [
                        [1 + owner * requests * window + slot * window + offset for offset in range(window)]
                        if slot >= 0
                        else [0] * window
                        for slot in slots.tolist()
                    ]
                )
                for owner in range(size)
            ]
        )
        torch.testing.assert_close(gathered_indices.long(), expected_indices)
        for offset in range(window):
            for owner in range(size):
                for slot in (0, 1):
                    first = 1 + owner * requests * window + slot * window
                    cache[first + offset].copy_(cache[first] + offset)
        offsets = torch.tensor([window - 1, 0, 0], dtype=torch.int32)
        destinations = torch.tensor([7, 4, 0], dtype=torch.int32)
        helper.store_pages(cache, owners[rank], destinations, slots, offsets, conv=conv)
        expected = originals[rank].clone()
        expected[7].copy_(originals[rank][2] + (window - 1))
        expected[4].copy_(originals[rank][5])
        torch.testing.assert_close(owners[rank], expected)
        torch.testing.assert_close(cache[0], torch.full(local_shape, -999.0))

        # A reused request slot must import the new owner's prefix, including
        # when another rank only supplies padded admission rows.
        reused_slots = (
            torch.tensor([1, -1, -1], dtype=torch.int32) if rank % 2 == 0 else torch.full((3,), -1, dtype=torch.int32)
        )
        reused_ids = torch.tensor([3, 0, 0], dtype=torch.int32)
        helper.load_pages(owners[rank], cache, reused_ids, reused_slots, conv=conv)
        helper.store_pages(
            cache, owners[rank], reused_ids + 3, reused_slots, torch.zeros(3, dtype=torch.int32), conv=conv
        )
        if rank % 2 == 0:
            expected[6].copy_(originals[rank][3])
        torch.testing.assert_close(owners[rank], expected)

        rollback_slots = torch.tensor([2, -1, -1], dtype=torch.int32)
        rollback_ids = torch.tensor([8, 0, 0], dtype=torch.int32)
        rollback_offsets = torch.full((3,), window - 1, dtype=torch.int32)
        helper.load_pages(owners[rank], cache, rollback_ids, rollback_slots, conv=conv, state_offsets=rollback_offsets)
        helper.store_pages(cache, owners[rank], torch.tensor([1, 0, 0]), rollback_slots, rollback_offsets, conv=conv)
        expected[1].copy_(originals[rank][8])
        torch.testing.assert_close(owners[rank], expected)
        return True

    with ThreadPoolExecutor(max_workers=size) as executor:
        results = list(executor.map(worker, range(size)))
    assert all(results)


@pytest.mark.parametrize("requests,window", [(0, 1), (1, 0), (-1, 6)])
def test_state_transfer_rejects_invalid_capacity(requests, window):
    group = SimpleNamespace(world_size=2)
    with pytest.raises(ValueError, match="positive"):
        kda_state_tp.KDAStateTP(group, 8, 2, max_requests=requests, state_window=window)
