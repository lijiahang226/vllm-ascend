# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP metadata for GLM's full-history and persistent compressor caches."""

from copy import copy

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolStateSpec
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class AscendGlm5NextMTPProposer(AscendEagleProposer):
    """Keep each cache's addressing while advancing draft positions once.

    The target and draft use the same request ordering. Full-history MLA and
    compressed keys share scheduler blocks; compressor state uses its own sliding-window
    page table from the original cache manager. GLM drafts retain the base class's eager mode.
    """

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        all_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        groups = {}
        for gid, cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for name in sorted(self._draft_attn_layer_names.intersection(cache_group.layer_names)):
                spec = cache_group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                backend = all_layers[name].get_attn_backend()
                key = (gid, backend.full_cls_name(), spec)
                if key not in groups:
                    groups[key] = AttentionGroup(backend, [], spec, gid)
                groups[key].layer_names.append(name)

        self.draft_attn_groups = sorted(groups.values(), key=lambda group: group.backend is not AscendSFABackend)
        if not self.draft_attn_groups or self.draft_attn_groups[0].backend is not AscendSFABackend:
            raise ValueError("GLM MTP requires a main MLA cache group.")
        # The runner selects draft common metadata using the first layer name.
        self.attn_layer_names = [name for group in self.draft_attn_groups for name in group.layer_names]
        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        for group in self.draft_attn_groups:
            is_main = group.backend is AscendSFABackend
            if not isinstance(group.kv_cache_spec, AscendIndexerKPoolStateSpec):
                if group.kv_cache_group_id != self.kv_cache_gid:
                    raise ValueError("GLM MTP main and compressed keys must share scheduler blocks.")
            # Main and compressed metadata both recover physical pages from
            # the common C128 table. Rewriting either spec to C128 loses the
            # original page geometry; only position updates use kernel sizes.
            group.create_metadata_builders(
                self.vllm_config,
                self.device,
                kernel_block_size=None,
                # All draft steps are built before execution. Each step needs
                # independent buffers for compressed lengths and addresses.
                num_metadata_builders=1 if is_main else self.num_speculative_tokens,
            )
        self.block_size = (
            kernel_block_sizes[self.kv_cache_gid]
            if kernel_block_sizes
            else self.draft_attn_groups[0].kv_cache_spec.block_size
        )
        self.kernel_block_size = self.block_size

    def _cache_group_common_metadata(self, common, group):
        if not isinstance(group.kv_cache_spec, AscendIndexerKPoolStateSpec):
            return common
        state_common = copy(common)
        table = self.runner.input_batch.block_table[group.kv_cache_group_id].get_device_tensor()
        table = table[: common.num_reqs]
        state_common.block_table_tensor = table
        positions = common.positions[: common.num_input_tokens].long()
        slots = torch.full_like(positions, -1)
        if common.num_reqs and table.shape[1]:
            rows = torch.arange(positions.numel(), device=positions.device)
            request = torch.searchsorted(common.query_start_loc[1 : common.num_reqs + 1], rows, right=True)
            page = positions // group.kv_cache_spec.block_size
            valid = (request < common.num_reqs) & (positions >= 0) & (page < table.shape[1])
            valid &= common.slot_mapping[: positions.numel()] >= 0
            physical = table[request.clamp(max=common.num_reqs - 1), page.clamp(0, table.shape[1] - 1)].long()
            valid &= physical >= 0
            slots = torch.where(
                valid,
                physical * group.kv_cache_spec.block_size + positions % group.kv_cache_spec.block_size,
                slots,
            )
        # Each draft keeps its own mapping; subsequent steps must not change it.
        state_common.slot_mapping = slots
        return state_common

    def build_draft_attn_metadata(self, common_attn_metadata, num_input_tokens, num_actual_tokens):
        per_layer = {}
        for group in self.draft_attn_groups:
            metadata = group.get_metadata_builder().build(
                0, self._cache_group_common_metadata(common_attn_metadata, group)
            )
            for name in group.layer_names:
                per_layer[name] = metadata
        return [per_layer], per_layer[self.attn_layer_names[0]]

    def attn_update_stack_num_spec_norm(
        self,
        draft_index,
        old_common_metadata,
        batch_size,
        input_batch_size,
        used_update_positions,
        aclgraph_runtime_mode,
        **kwargs,
    ):
        group = kwargs["attn_group"]
        if group.backend is not AscendSFABackend:
            metadata = group.get_metadata_builder(draft_index).build(
                0, self._cache_group_common_metadata(old_common_metadata, group)
            )
            return old_common_metadata, metadata

        if draft_index == 1:
            old_common_metadata = copy(old_common_metadata)
            old_common_metadata.seq_lens = old_common_metadata.seq_lens.clone()
            # The base proposer selects positions at the accepted endpoint.
            # Reuse that selection without retaining per-batch rejection state.
            old_common_metadata.seq_lens[:batch_size] = used_update_positions[:batch_size] + 1
            old_common_metadata.seq_lens_cpu = None
            old_common_metadata._seq_lens_cpu = None
            old_common_metadata.num_computed_tokens_cpu = None
            old_common_metadata._num_computed_tokens_cpu = None
            old_common_metadata._num_computed_tokens_cache = None
        # SFA is first; only it advances the common positions/lengths. The
        # remaining groups build metadata from those same updated tensors.
        return super().attn_update_stack_num_spec_norm(
            draft_index,
            old_common_metadata,
            batch_size,
            input_batch_size,
            used_update_positions,
            aclgraph_runtime_mode,
            **kwargs,
        )
