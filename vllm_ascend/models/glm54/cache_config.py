# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass, replace

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)


@dataclass(frozen=True, kw_only=True)
class Glm54WindowCacheSpec(AscendSlidingWindowMLASpec):
    max_tp_shards: int | None = 1


def is_glm54_dummy(config):
    model = getattr(config, "model_config", None)
    return getattr(getattr(model, "hf_config", None), "architectures", None) == ["Glm54DummyForCausalLM"]


def is_glm54_cache_groups(groups):
    return any(
        isinstance(spec, Glm54WindowCacheSpec)
        for group in groups
        for spec in (
            group.kv_cache_spec.kv_cache_specs.values()
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
    )


def glm54_cache_groups(config, specs):
    if config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("GLM-5.4 dummy requires the hybrid KV cache manager")
    main = sorted(
        (name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)), key=extract_layer_index
    )
    indexers = sorted(
        (name for name, spec in specs.items() if isinstance(spec, AscendSFAIndexerCacheSpec)), key=extract_layer_index
    )
    window = sorted(
        (name for name, spec in specs.items() if isinstance(spec, Glm54WindowCacheSpec)), key=extract_layer_index
    )
    if not main or not window or len(main) + len(indexers) + len(window) != len(specs):
        raise ValueError("Unexpected GLM-5.4 cache roles")
    if len({spec.block_size for spec in specs.values()}) != 1:
        raise ValueError("GLM-5.4 cache owners must share one logical block size")
    specs.update((name, replace(spec, page_size_padded=None)) for name, spec in list(specs.items()))
    owners = {extract_layer_index(name) for name in main}
    for name in indexers:
        if extract_layer_index(name) not in owners:
            raise ValueError(f"Indexer has no physical attention owner: {name}")
    windows = defaultdict(list)
    for name in window:
        if extract_layer_index(name) not in owners:
            raise ValueError(f"Loop cache has no physical attention owner: {name}")
        windows[specs[name].sliding_window].append(name)
    names = [main + indexers, *(windows[size] for size in sorted(windows))]
    groups = []
    for group_names in names:
        uniform = UniformTypeKVCacheSpecs.from_specs({name: specs[name] for name in group_names})
        if uniform is None:
            raise ValueError("GLM-5.4 group members must have identical block ownership")
        groups.append(KVCacheGroupSpec(group_names, uniform))
    return groups


def glm54_cache_slots(groups):
    slots = defaultdict(list)
    for group in groups:
        for name in group.layer_names:
            spec = group.kv_cache_spec.kv_cache_specs[name]
            role = "indexer" if isinstance(spec, AscendSFAIndexerCacheSpec) else "mla"
            key = (role, spec.page_size_bytes, extract_layer_index(name))
            slots[key].append(name)
    return slots


def glm54_pool_bytes(groups):
    return sum(page_bytes for _, page_bytes, _ in glm54_cache_slots(groups))


def glm54_cache_config(config, groups, available_memory):
    slots = glm54_cache_slots(groups)
    num_blocks = may_override_num_blocks(config, available_memory // glm54_pool_bytes(groups))
    tensors = [
        KVCacheTensor(size=num_blocks * page_bytes, layers=names, block_stride=page_bytes, layer_stride=0)
        for (_, page_bytes, _), names in slots.items()
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
    )


def allocate_glm54_cache(config, cache_config, allocate):
    specs = {
        name: group.kv_cache_spec.kv_cache_specs[name]
        for group in cache_config.kv_cache_groups
        for name in group.layer_names
    }
    context = config.compilation_config.static_forward_context
    result = {}
    # Different groups own disjoint pool IDs. Their layer views may share a
    # backing slot, but layers in the same group always use different slots.
    for tensor in cache_config.kv_cache_tensors:
        name = tensor.layers[0]
        spec = specs[name]
        if isinstance(spec, AscendSFAIndexerCacheSpec):
            if spec.scale_dim:
                raise ValueError("GLM-5.4 dummy requires unquantized indexer pages")
            parts = (allocate(tensor.size),)
        else:
            layer = context[name]
            if spec.head_size != layer.kv_lora_rank + layer.qk_rope_head_dim:
                raise ValueError("GLM-5.4 dummy requires split latent/RoPE cache pages")
            latent_bytes = (
                cache_config.num_blocks * spec.block_size * spec.num_kv_heads * layer.kv_lora_rank * spec.dtype.itemsize
            )
            parts = (allocate(latent_bytes), allocate(tensor.size - latent_bytes))
        for owner in tensor.layers:
            result[owner] = parts
    return result
