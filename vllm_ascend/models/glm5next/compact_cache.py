# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local physical block addresses for GLM-Next attention caches."""

from dataclasses import replace
from heapq import heappop, heappush

import torch
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import validate_additional_config_bool
from vllm_ascend.models.glm5next.cache_config import _get_glm5_next_cache_layout


class _BlockMap:
    def __init__(self, capacity: int, logical_capacity: int, first_live_block: int = 1):
        self.capacity = capacity
        self.logical_capacity = logical_capacity
        self.free = list(range(first_live_block, capacity))
        self.requests: dict[str, dict[int, int]] = {}
        self.owners: dict[int, str] = {}

    def release(self, req_id: str) -> None:
        for logical, physical in self.requests.pop(req_id, {}).items():
            del self.owners[logical]
            heappush(self.free, physical)

    def append(self, req_id: str, logical_ids: list[int]) -> tuple[list[int], list[int]]:
        mapping = self.requests.setdefault(req_id, {})
        result, added = [], []
        for logical in logical_ids:
            if logical == 0:
                result.append(0)
                continue
            if not 0 < logical < self.logical_capacity:
                raise ValueError("GLM-Next logical block is outside the scheduler pool.")
            if self.owners.get(logical) == req_id:
                # Resumable streaming can resend the full existing block list.
                # Preserve its contents and address, clearing only fresh pages.
                result.append(mapping[logical])
                continue
            if logical in self.owners:
                raise ValueError("GLM-Next compact cache received an already owned block.")
            if not self.free:
                raise RuntimeError("GLM-Next compact attention cache capacity exceeded.")
            physical = heappop(self.free)
            mapping[logical] = physical
            self.owners[logical] = req_id
            result.append(physical)
            added.append(physical)
        return result, added


class Glm5NextCompactCache:
    """Translate scheduler deltas on CPU; keep state groups in global block IDs.

    Cache and block-table storage addresses remain fixed during graph replay.
    Slot release follows finished/preempted notifications, never batch removal.
    All device zeroing uses the runner's current stream, before the next forward.
    """

    def __init__(self, config: KVCacheConfig, group_ids: tuple[int, int], capacity: int, max_num_seqs: int):
        self.capacity = capacity
        self.group_ids = group_ids
        self.maps = {gid: _BlockMap(capacity, config.num_blocks) for gid in group_ids}
        # The runner's dummy/capture tail path clears rows 1..max_num_seqs.
        # Those rows must never contain state of a live but unscheduled request.
        self.maps[group_ids[1]] = _BlockMap(capacity, config.num_blocks, 1 + max_num_seqs)
        self.layer_names = {gid: tuple(config.kv_cache_groups[gid].layer_names) for gid in group_ids}
        self.layer_capacities = {name: capacity for names in self.layer_names.values() for name in names}
        self.views: dict[int, list[torch.Tensor]] = {}

    @classmethod
    def create(cls, vllm_config: VllmConfig, config: KVCacheConfig) -> "Glm5NextCompactCache | None":
        additional = getattr(vllm_config, "additional_config", None) or {}
        if not validate_additional_config_bool(
            additional.get("glm5_next_compact_attention_cache", False),
            "additional_config.glm5_next_compact_attention_cache",
        ):
            return None
        layout = _get_glm5_next_cache_layout(config.kv_cache_groups)
        parallel = vllm_config.parallel_config
        if (
            layout is None
            or not layout.state_slot_count
            or vllm_config.cache_config.enable_prefix_caching
            or vllm_config.kv_transfer_config is not None
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "Compact GLM-Next attention caches require contiguous states, TP1/PP1/CP1, "
                "and no prefix caching or KV transfer."
            )
        # Include the configured in-flight token allowance, even at a model-length
        # boundary. Each pool also owns a permanent null block for graph padding.
        max_len = vllm_config.model_config.max_model_len + vllm_config.max_in_flight_tokens
        per_request = max(
            spec.max_num_blocks_per_req(vllm_config, max_len)
            for group in (layout.full_group, layout.tail_group)
            for spec in group.kv_cache_spec.kv_cache_specs.values()
        )
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        capacity = 1 + max_num_seqs * max(per_request, 2)
        if capacity > config.num_blocks:
            raise ValueError("Compact GLM-Next attention caches require enough global blocks for dummy and live slots.")
        group_ids = tuple(config.kv_cache_groups.index(group) for group in (layout.full_group, layout.tail_group))
        compact = cls(config, group_ids, capacity, max_num_seqs)
        # Apply only after upstream has normalized the global block count across
        # workers. The scheduler's original capacity accounting remains conservative.
        for tensor in config.kv_cache_tensors:
            if any(name in compact.layer_capacities for name in tensor.layers):
                if not all(name in compact.layer_capacities for name in tensor.layers):
                    raise ValueError("Compact attention storage cannot alias recurrent states.")
                tensor.size = capacity * tensor.block_stride
        return compact

    def bind_views(self, caches: dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]]) -> None:
        for gid, names in self.layer_names.items():
            views = []
            for name in names:
                tensors = caches[name]
                if isinstance(tensors, torch.Tensor):
                    tensors = (tensors,)
                for tensor in tensors:
                    if tensor.numel():
                        # Indexer and tail occupy separate packed regions of one
                        # backing. Clear logical views, never raw page-size slices.
                        views.append(tensor.view(self.capacity, -1))
            self.views[gid] = views

    def translate(self, output: SchedulerOutput) -> SchedulerOutput:
        if output.kv_cache_block_copies:
            # CoW is outside the initial no-prefix-cache/no-state-checkpoint scope.
            raise ValueError("GLM-Next compact attention caches do not support block copies.")
        cached = output.scheduled_cached_reqs
        retired = output.finished_req_ids | (output.preempted_req_ids or set()) | cached.resumed_req_ids
        for mapping in self.maps.values():
            for req_id in retired:
                mapping.release(req_id)
        added: dict[int, list[int]] = {gid: [] for gid in self.maps}

        def translate_blocks(req_id: str, blocks: tuple[list[int], ...]) -> tuple[list[int], ...]:
            # The parent runner retains these lists and later extends them. Never
            # lend it lists from the original scheduler payload, even for states.
            result = [list(ids) for ids in blocks]
            for gid, mapping in self.maps.items():
                result[gid], fresh = mapping.append(req_id, blocks[gid])
                added[gid].extend(fresh)
            return tuple(result)

        new_reqs = [
            replace(req, block_ids=translate_blocks(req.req_id, req.block_ids)) for req in output.scheduled_new_reqs
        ]
        new_blocks = [
            translate_blocks(req_id, blocks) if blocks is not None else None
            for req_id, blocks in zip(cached.req_ids, cached.new_block_ids)
        ]
        for gid, rows in added.items():
            for view in self.views[gid]:
                for row in rows:
                    view[row].zero_()
        return replace(
            output,
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=replace(cached, new_block_ids=new_blocks),
            # All new compact pages were cleared through their logical views.
            # The existing Ascend zeroer skips recurrent-state caches.
            new_block_ids_to_zero=[],
        )
