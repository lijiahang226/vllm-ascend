# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block_from_builder
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.glm54.cache_config import (
    Glm54WindowCacheSpec,
    allocate_glm54_cache,
    glm54_cache_config,
    glm54_cache_groups,
    glm54_pool_bytes,
)
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.worker import NPUWorker


def make_specs(layers=6, index_stride=2, loop_layers=(2, 3)):
    specs = {}
    for index in range(layers):
        name = f"model.layers.{index}"
        specs[name + ".attn"] = AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        )
        if index % index_stride == 0:
            specs[name + ".indexer"] = AscendSFAIndexerCacheSpec(
                block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
            )
        if index in loop_layers:
            specs[name + ".loop_kv"] = Glm54WindowCacheSpec(
                block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, sliding_window=128
            )
    return specs


def make_config(specs):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Glm54DummyForCausalLM"], model_type="glm_moe_dsa"),
            max_model_len=1024,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
        compilation_config=SimpleNamespace(
            static_forward_context={name: SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64) for name in specs}
        ),
    )


def test_native_pages_and_indexer_ownership_survive_grouping():
    specs = make_specs()
    config = make_config(specs)
    specs = {name: replace(spec, page_size_padded=128 * 576 * 2) for name, spec in specs.items()}
    groups = glm54_cache_groups(config, specs)
    assert [len(group.layer_names) for group in groups] == [9, 2]
    assert {spec.page_size_bytes for spec in specs.values()} == {147456, 32768}
    assert all(spec.page_size_padded is None for spec in specs.values())
    for name in groups[0].layer_names:
        if name.endswith(".indexer"):
            assert name.removesuffix(".indexer") + ".attn" in groups[0].layer_names
    assert isinstance(groups[-1].kv_cache_spec.first_spec, SlidingWindowSpec)
    assert glm54_pool_bytes(groups) == 983040


@pytest.mark.parametrize(
    "layers,index_stride,loop_layers",
    [(6, 2, (2, 3)), (7, 2, (2, 3, 4)), (8, 1, (1, 2, 3)), (78, 4, tuple(range(20, 60)))],
)
def test_every_group_has_distinct_slots_and_the_plan_fits_the_budget(layers, index_stride, loop_layers):
    specs = make_specs(layers, index_stride, loop_layers)
    config = make_config(specs)
    groups = glm54_cache_groups(config, specs)
    assert len(groups) == 2
    assert set(groups[0].layer_names) == {name for name in specs if not name.endswith(".loop_kv")}
    assert set(groups[1].layer_names) == {name for name in specs if name.endswith(".loop_kv")}
    cost = glm54_pool_bytes(groups)
    plan = glm54_cache_config(config, groups, cost * 7 + cost // 2)
    assert plan.num_blocks == 7
    assert sum(tensor.size for tensor in plan.kv_cache_tensors) == 7 * cost
    owners = {name: slot for slot, tensor in enumerate(plan.kv_cache_tensors) for name in tensor.layers}
    assert owners.keys() == specs.keys()
    for group in groups:
        assert len({owners[name] for name in group.layer_names}) == len(group.layer_names)
    assert all(tensor.size == plan.num_blocks * tensor.block_stride for tensor in plan.kv_cache_tensors)


def test_grouping_depends_on_retention_and_not_registration_order():
    specs = make_specs()
    specs["model.layers.3.loop_kv"] = replace(specs["model.layers.3.loop_kv"], sliding_window=256)
    groups = glm54_cache_groups(make_config(specs), dict(reversed(list(specs.items()))))
    assert len(groups) == 3
    assert [group.kv_cache_spec.first_spec.sliding_window for group in groups[1:]] == [128, 256]
    assert set(groups[1].layer_names) == {"model.layers.2.loop_kv"}
    assert set(groups[2].layer_names) == {"model.layers.3.loop_kv"}


def test_shared_backings_preserve_live_first_pass_pages():
    specs = make_specs()
    config = make_config(specs)
    groups = glm54_cache_groups(config, specs)
    plan = glm54_cache_config(config, groups, glm54_pool_bytes(groups) * 8)
    sizes = []

    def allocate(size):
        sizes.append(size)
        return torch.zeros(size, dtype=torch.int8)

    caches = allocate_glm54_cache(config, plan, allocate)
    assert sum(sizes) == sum(tensor.size for tensor in plan.kv_cache_tensors)
    assert caches["model.layers.2.attn"][0] is caches["model.layers.2.loop_kv"][0]
    first = caches["model.layers.2.attn"][0].view(torch.bfloat16).view(8, 128, 1, 512)
    second = caches["model.layers.2.loop_kv"][0].view(torch.bfloat16).view(8, 128, 1, 512)
    first[2].fill_(3)
    saved = first[2].clone()
    second[4].fill_(7)
    torch.testing.assert_close(first[2], saved)
    assert second[4, 0, 0, 0] == 7
    assert first.is_contiguous() and second.is_contiguous()


def test_window_admission_keeps_the_entire_prefill_chunk():
    spec = make_specs()["model.layers.2.loop_kv"]
    assert spec.num_retained_tokens == 127
    assert spec.max_admission_blocks_per_request(max_in_flight_tokens=32, max_model_len=1024) == 3
    assert spec.max_admission_blocks_per_request(max_in_flight_tokens=256, max_model_len=4096) == 4


def test_window_manager_recycles_pages_without_reusing_live_full_ids():
    specs = make_specs()
    pool = BlockPool(num_gpu_blocks=64, enable_caching=False, hash_block_size=128)
    full = FullAttentionManager(
        specs["model.layers.2.attn"],
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=128,
    )
    window = SlidingWindowManager(
        specs["model.layers.2.loop_kv"],
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=1,
        scheduler_block_size=128,
        max_admission_blocks_per_request=4,
    )
    computed = 0
    for total in (256, 512, 768, 895, 903):
        window.remove_skipped_blocks("request", computed)
        full.allocate_new_blocks("request", total, total)
        window.allocate_new_blocks("request", total, total)
        full_ids = {block.block_id for block in full.req_to_blocks["request"] if not block.is_null}
        window_ids = {block.block_id for block in window.req_to_blocks["request"] if not block.is_null}
        assert not full_ids & window_ids
        assert len(window_ids) <= 4
        computed = total
    assert len(full_ids) == 8


def test_worker_reshapes_window_pages_as_split_mla(monkeypatch):
    specs = make_specs()
    config = make_config(specs)
    config.model_config.hf_text_config = config.model_config.hf_config
    config.kv_transfer_config = None
    groups = glm54_cache_groups(config, specs)
    plan = glm54_cache_config(config, groups, glm54_pool_bytes(groups) * 8)
    raw = allocate_glm54_cache(config, plan, lambda size: torch.zeros(size, dtype=torch.int8))
    group = groups[-1]
    spec = group.kv_cache_spec.first_spec
    attention = AttentionGroup(
        backend=AscendSFABackend,
        layer_names=group.layer_names,
        kv_cache_spec=spec,
        kv_cache_group_id=len(groups) - 1,
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *args: (512, 64))
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: True)
    caches = attn_utils._reshape_kv_cache_v2([attention], raw, "auto", [128] * len(groups), {}, plan)
    for name in group.layer_names:
        latent, rope = caches[name]
        assert latent.shape == (8, 128, 1, 512)
        assert rope.shape == (8, 128, 1, 64)
        assert latent.is_contiguous() and rope.is_contiguous()
        assert latent.untyped_storage().data_ptr() == raw[name][0].untyped_storage().data_ptr()


def test_builder_uses_the_same_native_cost_for_planning_and_materialization():
    specs = make_specs()
    config = make_config(specs)
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(config, specs)
    config.attention_config = SimpleNamespace(hisparse_config=None)
    config.kv_transfer_config = None
    assert _pool_bytes_per_block_from_builder(builder, config, groups) == 983040
    assert builder._get_native_pool_bytes_per_block(config, groups) == 983040
    plan = builder.get_kv_cache_config_from_groups(config, groups, num_blocks=11)
    assert plan.num_blocks == 11
    assert sum(tensor.size for tensor in plan.kv_cache_tensors) == 983040 * 11


def test_worker_does_not_scale_the_shared_pool_budget_twice(monkeypatch):
    specs = make_specs()
    config = make_config(specs)
    config.attention_config = SimpleNamespace(hisparse_config=None)
    groups = glm54_cache_groups(config, specs)
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = config
    worker.get_kv_cache_spec = lambda: specs
    monkeypatch.setattr("vllm_ascend.worker.worker.get_kv_cache_groups", lambda *args: groups)
    budget = 128 * 1024 * 1024
    assert worker._scale_kv_cache_memory_for_multi_group(budget) == budget
    assert glm54_cache_config(config, groups, budget).num_blocks == 136


def test_disabled_hybrid_management_is_rejected():
    specs = make_specs()
    config = make_config(specs)
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    with pytest.raises(ValueError, match="hybrid"):
        glm54_cache_groups(config, specs)
