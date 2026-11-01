# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer request-owned KDA pages to and from a persistent head partition."""

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from .kda_projection_tp import KDATPLayout


class KDAStateTP(KDATPLayout):
    def __init__(self, group: GroupCoordinator, num_heads: int, head_dim: int, *, max_requests: int, state_window: int):
        super().__init__(group, num_heads, head_dim)
        if max_requests < 1 or state_window < 1:
            raise ValueError("KDA state TP needs positive request capacity and state window")
        self.max_requests = max_requests
        self.state_window = state_window
        self.owner_pages = max_requests * state_window

    def state_indices(self, request_slots: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.state_window, device=request_slots.device, dtype=request_slots.dtype)
        return 1 + request_slots.unsqueeze(-1) * self.state_window + offsets

    def gather_state_indices(self, request_slots: torch.Tensor) -> torch.Tensor:
        slots = self.group.all_gather(request_slots, dim=0).reshape(self.size, -1)
        owner_offsets = torch.arange(self.size, device=slots.device, dtype=slots.dtype) * self.owner_pages
        indices = self.state_indices(slots) + owner_offsets[:, None, None]
        return torch.where(slots.unsqueeze(-1) >= 0, indices, 0).flatten(0, 1)

    def _shard_pages(self, pages: torch.Tensor, *, conv: bool) -> torch.Tensor:
        count = pages.shape[0]
        if conv:
            history = pages.shape[1]
            packed = pages.reshape(count, history, 3, self.size, self.width).permute(3, 0, 1, 2, 4)
        else:
            packed = pages.reshape(count, self.size, self.heads, self.head_dim, self.head_dim).transpose(0, 1)
        return self._all_to_all(packed.contiguous()).flatten(0, 1)

    def _restore_pages(self, pages: torch.Tensor, *, conv: bool) -> torch.Tensor:
        count = pages.shape[0] // self.size
        if conv:
            history = pages.shape[1]
            packed = pages.reshape(self.size, count, history, 3, self.width)
            received = self._all_to_all(packed)
            return received.permute(1, 2, 3, 0, 4).reshape(count, history, -1)
        packed = pages.reshape(self.size, count, self.heads, self.head_dim, self.head_dim)
        received = self._all_to_all(packed)
        return received.transpose(0, 1).reshape(count, -1, self.head_dim, self.head_dim)

    def load_pages(
        self,
        owner_cache: torch.Tensor,
        sharded_cache: torch.Tensor,
        owner_indices: torch.Tensor,
        request_slots: torch.Tensor,
        *,
        conv: bool = False,
        state_offsets: torch.Tensor | None = None,
    ) -> None:
        """Import an admission or restored prefix after its owner-side KV load.

        All ranks provide equally padded plans. Invalid rows never overwrite
        a cache page. Imports and exports run outside the decode graph.
        """
        valid = (owner_indices > 0) & (request_slots >= 0)
        pages = owner_cache.index_select(0, owner_indices.clamp_min(0).long())
        pages = pages.masked_fill(~valid.reshape(-1, *([1] * (pages.ndim - 1))), 0)
        received = self._shard_pages(pages, conv=conv)
        if state_offsets is None:
            state_offsets = torch.zeros_like(request_slots)
        plan = torch.stack((request_slots, state_offsets, valid.to(request_slots.dtype)), dim=-1)
        gathered = self.group.all_gather(plan, dim=0).reshape(self.size, -1, 3)
        slots, offsets, valid = gathered.unbind(-1)
        valid = valid.flatten().bool()
        owner_offsets = torch.arange(self.size, device=slots.device, dtype=slots.dtype) * self.owner_pages
        indices = (1 + slots * self.state_window + offsets + owner_offsets[:, None]).flatten()
        sharded_cache.index_copy_(0, indices[valid].long(), received[valid].reshape(-1, *sharded_cache.shape[1:]))

    def store_pages(
        self,
        sharded_cache: torch.Tensor,
        owner_cache: torch.Tensor,
        owner_indices: torch.Tensor,
        request_slots: torch.Tensor,
        state_offsets: torch.Tensor,
        *,
        conv: bool = False,
    ) -> None:
        """Export selected accepted states before owner-side checkpoint copies."""
        valid = (owner_indices > 0) & (request_slots >= 0)
        plan = torch.stack((request_slots, state_offsets), dim=-1)
        gathered = self.group.all_gather(plan, dim=0).reshape(self.size, -1, 2)
        owner_offsets = torch.arange(self.size, device=plan.device, dtype=plan.dtype) * self.owner_pages
        indices = (1 + gathered[:, :, 0] * self.state_window + gathered[:, :, 1] + owner_offsets[:, None]).clamp_min(0)
        pages = sharded_cache.index_select(0, indices.flatten().long())
        restored = self._restore_pages(pages, conv=conv)
        owner_cache.index_copy_(0, owner_indices[valid].long(), restored[valid])
