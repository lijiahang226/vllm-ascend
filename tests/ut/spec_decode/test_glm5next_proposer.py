# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolStateBackend,
    AscendIndexerKPoolStateMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadataBuilder
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolStateSpec, AscendMLAAttentionSpec
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.glm5next_proposer import AscendGlm5NextMTPProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import vllm_version_is

MAIN = "draft.layers.45.self_attn.mla_attn"
INDEXER = "draft.layers.45.self_attn.indexer.k_cache"
STATE = "draft.layers.45.self_attn.indexer.state_cache"


def _specs(block_size=256):
    main = AscendMLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=512, dtype=torch.bfloat16, model_version="glm5_next"
    )
    indexer = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        **({"compress_ratio": 8} if vllm_version_is("0.28.0") else {"tokens_per_state": 8}),
        model_version="glm5_next",
    )
    state = AscendIndexerKPoolStateSpec(
        block_size=8, sliding_window=8, num_kv_heads=1, head_size=256, dtype=torch.float32
    )
    return main, indexer, state


def test_initialize_keeps_logical_indexer_geometry_and_independent_step_buffers():
    main, indexer, state = _specs()
    # Put state first to catch accidental use of its eight-token geometry for
    # the main MLA's 128-token kernel block table.
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[STATE], kv_cache_spec=state),
            SimpleNamespace(
                layer_names=[INDEXER, MAIN],
                kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs={MAIN: main, INDEXER: indexer}),
            ),
        ]
    )
    layers = {
        MAIN: SimpleNamespace(get_attn_backend=lambda: AscendSFABackend),
        INDEXER: SimpleNamespace(get_attn_backend=lambda: AscendIndexerKPoolBackend),
        STATE: SimpleNamespace(get_attn_backend=lambda: AscendIndexerKPoolStateBackend),
    }
    proposer = object.__new__(AscendGlm5NextMTPProposer)
    proposer.vllm_config = object()
    proposer.device = torch.device("cpu")
    proposer.num_speculative_tokens = 3
    proposer._draft_attn_layer_names = {MAIN, INDEXER, STATE}
    with (
        patch("vllm_ascend.spec_decode.glm5next_proposer.get_layers_from_vllm_config", return_value=layers),
        patch.object(AttentionGroup, "create_metadata_builders", autospec=True) as create,
    ):
        proposer.initialize_attn_backend(config, [8, 128])
    assert proposer.kv_cache_gid == 1
    assert proposer.attn_layer_names[0] == MAIN
    assert proposer.block_size == proposer.kernel_block_size == 128
    calls = {call.args[0].layer_names[0]: call.kwargs for call in create.call_args_list}
    assert calls[MAIN] == {"kernel_block_size": None, "num_metadata_builders": 1}
    assert calls[INDEXER] == calls[STATE] == {"kernel_block_size": None, "num_metadata_builders": 3}


@pytest.mark.parametrize("block_size", [128, 256, 384, 2304])
def test_initialize_preserves_physical_pages_with_real_metadata_builders(block_size):
    main, indexer, state = _specs(block_size)
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[MAIN, INDEXER],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=block_size, kv_cache_specs={MAIN: main, INDEXER: indexer}
                ),
            ),
            SimpleNamespace(layer_names=[STATE], kv_cache_spec=state),
        ]
    )
    layers = {
        MAIN: SimpleNamespace(get_attn_backend=lambda: AscendSFABackend),
        INDEXER: SimpleNamespace(get_attn_backend=lambda: AscendIndexerKPoolBackend),
        STATE: SimpleNamespace(get_attn_backend=lambda: AscendIndexerKPoolStateBackend),
    }
    proposer = object.__new__(AscendGlm5NextMTPProposer)
    max_model_len = 131072
    proposer.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=8),
        model_config=SimpleNamespace(max_model_len=max_model_len, get_head_size=lambda: 512),
        compilation_config=SimpleNamespace(
            static_forward_context={MAIN: SimpleNamespace(qk_rope_head_dim=0, impl=SimpleNamespace(indexer=None))}
        ),
        speculative_config=None,
    )
    proposer.device = torch.device("cpu")
    proposer.num_speculative_tokens = 3
    proposer._draft_attn_layer_names = {MAIN, INDEXER, STATE}

    def base_init(self, spec, names, cfg, device, metadata_cls, supports_dcp):
        self.kv_cache_spec, self.vllm_config, self.device = spec, cfg, device
        self.model_config, self.metadata_cls = cfg.model_config, metadata_cls

    with (
        patch("vllm_ascend.spec_decode.glm5next_proposer.get_layers_from_vllm_config", return_value=layers),
        patch.object(AscendSFABackend, "get_builder_cls", return_value=AscendSFAMetadataBuilder),
        patch.object(MLACommonMetadataBuilder, "__init__", base_init),
        patch(
            "vllm_ascend.attention.sfa_v1.get_ascend_config",
            return_value=SimpleNamespace(c8_reshape_optim_enabled=False),
        ),
        patch("vllm_ascend.attention.sfa_v1.select_common_block_size", return_value=128),
        patch("vllm_ascend.attention.sfa_v1.AttentionMaskBuilder"),
    ):
        proposer.initialize_attn_backend(config, [128, 8])
    # Position updates use C128, while supported cache storage pages remain
    # intact. 128K also leaves a partial final C384 page.
    assert proposer.block_size == proposer.kernel_block_size == 128
    width = (max_model_len + block_size - 1) // block_size
    pages = torch.arange(2 * width, dtype=torch.int32).reshape(2, width)
    split = block_size // 128
    expanded = (pages.unsqueeze(-1) * split + torch.arange(split, dtype=torch.int32)).reshape(2, -1)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2,
        num_input_tokens=2,
        max_query_len=1,
        max_seq_len=max_model_len,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([max_model_len, 1], dtype=torch.int32),
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
        attn_state=None,
        causal=True,
        group_len=None,
        group_key_idx=None,
        group_key_cache_idx=None,
        block_table_tensor=expanded,
        slot_mapping=torch.tensor([max_model_len - 1, width * block_size]),
        positions=torch.tensor([max_model_len - 1, 0]),
    )
    builder = proposer.draft_attn_groups[0].get_metadata_builder()
    with (
        patch(
            "vllm_ascend.attention.sfa_v1.get_ascend_config",
            return_value=SimpleNamespace(c8_reshape_optim_enabled=False),
        ),
        patch(
            "vllm_ascend.attention.sparse_mla.get_current_hardware_profile",
            return_value=SimpleNamespace(device_adaptor_family=None),
        ),
    ):
        metadata = builder.build(0, common)
    torch.testing.assert_close(metadata.block_table, pages if block_size <= 1024 else expanded)
    assert metadata.block_size == (block_size if block_size <= 1024 else 128)
    physical = metadata.block_table[torch.arange(2), common.positions // metadata.block_size]
    physical_token = physical.long() * metadata.block_size + common.positions % metadata.block_size
    torch.testing.assert_close(physical_token, common.slot_mapping)


@pytest.mark.parametrize("positions", [[6, 7, 8, 9, 0], [2, 3, 4, 5, 0]])
def test_state_slots_follow_draft_positions_and_request_blocks_after_rollback(positions):
    _, _, spec = _specs()
    group = AttentionGroup(AscendIndexerKPoolStateBackend, [STATE], spec, 1)
    state_table = torch.tensor([[9, 90], [3, 30]], dtype=torch.int32)
    proposer = object.__new__(AscendGlm5NextMTPProposer)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=[None, SimpleNamespace(get_device_tensor=lambda: state_table)])
    )
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=5,
        num_actual_tokens=4,
        positions=torch.tensor(positions),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        block_table_tensor=torch.tensor([[100, 101], [200, 201]]),
        slot_mapping=torch.tensor([1000, -1, 2000, 2001, -1]),
    )
    state_common = proposer._cache_group_common_metadata(common, group)
    assert state_common is not common
    builder = object.__new__(AscendIndexerKPoolStateMetadataBuilder)
    builder.block_size = 8
    builder.cache_role = "indexer_state"
    result = builder.build(0, state_common)
    torch.testing.assert_close(result.block_table, state_table)
    torch.testing.assert_close(
        result.slot_mapping,
        torch.tensor(
            [
                9 * 8 + positions[0] % 8,
                -1,
                int(state_table[1, positions[2] // 8]) * 8 + positions[2] % 8,
                int(state_table[1, positions[3] // 8]) * 8 + positions[3] % 8,
                -1,
            ]
        ),
    )
    assert common.slot_mapping.tolist() == [1000, -1, 2000, 2001, -1]
    state_table.copy_(state_table.flip(0))
    reordered = builder.build(0, proposer._cache_group_common_metadata(common, group))
    assert reordered.slot_mapping[0] == 3 * 8 + positions[0] % 8
    assert reordered.slot_mapping[2] == state_table[1, positions[2] // 8] * 8 + positions[2] % 8
    assert reordered.slot_mapping.data_ptr() != result.slot_mapping.data_ptr()


def test_state_metadata_masks_idle_rank_padding():
    builder = object.__new__(AscendIndexerKPoolStateMetadataBuilder)
    builder.block_size = 8
    builder.cache_role = "indexer_state"
    common = SimpleNamespace(
        num_reqs=0,
        num_actual_tokens=0,
        num_input_tokens=3,
        block_table_tensor=torch.empty(0, 1, dtype=torch.int32),
        positions=torch.zeros(3, dtype=torch.int64),
        slot_mapping=torch.full((3,), -1, dtype=torch.int64),
    )
    result = builder.build(0, common)
    assert result.block_table.shape == (0, 1)
    assert result.slot_mapping.tolist() == [-1, -1, -1]


@pytest.mark.parametrize("rejected", [[0, 0], [2, 0], [3, 3]])
@pytest.mark.parametrize("num_input_tokens, max_model_len", [(8, 128), (12, 128), (12, 24)])
def test_base_propose_keeps_each_group_and_draft_step_at_the_accepted_endpoint(
    rejected, num_input_tokens, max_model_len
):
    main, indexer, state = _specs()
    groups = [
        AttentionGroup(AscendSFABackend, [MAIN], main, 0),
        AttentionGroup(AscendIndexerKPoolBackend, [INDEXER], indexer, 0),
        AttentionGroup(AscendIndexerKPoolStateBackend, [STATE], state, 1),
    ]

    def metadata(common, *args):
        return SimpleNamespace(
            num_prefills=0,
            seq_lens=common.seq_lens,
            positions=common.positions,
            slot_mapping=common.slot_mapping,
        )

    main_builder = Mock(
        build=Mock(side_effect=lambda _, common: metadata(common)), build_for_drafting=Mock(side_effect=metadata)
    )
    groups[0].metadata_builders = [main_builder]
    for group in groups[1:]:
        group.metadata_builders = [Mock(build=Mock(side_effect=lambda _, common: metadata(common))) for _ in range(3)]
    proposer = object.__new__(AscendGlm5NextMTPProposer)
    proposer.draft_attn_groups = groups
    proposer.attn_layer_names = [MAIN, INDEXER, STATE]
    state_table = torch.tensor([[9, 90, 900, 91], [3, 30, 300, 31]], dtype=torch.int32)
    proposer.runner = SimpleNamespace(
        dcp_manager=None,
        dynamic_eplb=False,
        input_batch=SimpleNamespace(
            lora_id_to_lora_request={}, block_table=[None, SimpleNamespace(get_device_tensor=lambda: state_table)]
        ),
        _sync_metadata_across_dp=Mock(return_value=(None, torch.tensor([num_input_tokens]), None)),
    )
    proposer.method = "mtp"
    proposer.dcp_size = 1
    proposer.dp_rank = 0
    proposer.use_cuda_graph = proposer.uses_mrope = proposer.supports_mm_inputs = False
    proposer.parallel_drafting = proposer.has_gdn = proposer.use_compress = False
    proposer.draft_window_size = proposer.sliding_window = None
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_mla=True))
    proposer.block_size = 128
    proposer.max_model_len = max_model_len
    proposer.arange = torch.arange(9, dtype=torch.int32)
    proposer.token_arange_np = proposer.arange.numpy()
    proposer.slot_mapping_group = [torch.full((num_input_tokens,), -1, dtype=torch.int32) for _ in range(3)]
    proposer.seq_lens_group = [torch.zeros(2, dtype=torch.int32) for _ in range(3)]
    proposer.query_start_loc_group = [torch.zeros(3, dtype=torch.int32) for _ in range(3)]
    proposer.token_indices_to_sample = torch.zeros(2, dtype=torch.int32)
    proposer._pad_draft_buffers = Mock()
    # Reuse the same proposer for the next batch, with no retained rejection state.
    for rejects in (rejected, [0, 0]):
        for group in groups[1:]:
            for builder in group.metadata_builders:
                builder.build.reset_mock()
        positions = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int32)
        proposer.positions = torch.zeros(num_input_tokens, dtype=torch.int32)
        proposer.positions[:8].copy_(positions)
        indices = torch.tensor([3, 7]) - torch.tensor(rejects)
        lengths = torch.tensor([12, 24], dtype=torch.int32)
        common = SimpleNamespace(
            batch_size=lambda: 2,
            num_reqs=2,
            num_actual_tokens=8,
            num_input_tokens=8,
            positions=proposer.positions,
            seq_lens=lengths,
            seq_lens_cpu=lengths.clone(),
            _seq_lens_cpu=lengths.clone(),
            num_computed_tokens_cpu=lengths.clone(),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
            block_table_tensor=torch.tensor([[4], [6]], dtype=torch.int32),
            slot_mapping=torch.tensor([520, 521, 522, 523, 788, 789, 790, 791], dtype=torch.int32),
        )
        proposer.set_inputs_first_pass = Mock(return_value=(8, indices, common, None))
        with (
            patch(
                "vllm_ascend.spec_decode.llm_base_proposer.set_ascend_forward_context",
                side_effect=RuntimeError("metadata ready"),
            ) as context,
            pytest.raises(RuntimeError, match="metadata ready"),
        ):
            AscendSpecDecodeBaseProposer._propose(
                proposer,
                3,
                target_token_ids=torch.ones(8, dtype=torch.int64),
                target_positions=positions,
                target_hidden_states=torch.ones(8, 2),
                next_token_ids=torch.ones(2, dtype=torch.int64),
                token_indices_to_sample=indices,
                common_attn_metadata=common,
                target_model_batch_desc=SimpleNamespace(uniform=True),
                sampling_metadata=None,
            )
        steps = context.call_args.kwargs["draft_attn_metadatas"]
        assert len(steps) == 3
        for step, per_layer in enumerate(steps):
            assert set(per_layer) == {MAIN, INDEXER, STATE}
            assert len({id(value) for value in per_layer.values()}) == 3
            raw_lengths = lengths - torch.tensor(rejects, dtype=lengths.dtype) + step
            expected = lengths if step == 0 else (raw_lengths - 1) % max_model_len + 1
            for value in per_layer.values():
                torch.testing.assert_close(value.seq_lens, expected)
                assert (value.slot_mapping[2 if step else 8 :] == -1).all()
            if step:
                expected_positions = torch.where(raw_lengths > max_model_len, 0, raw_lengths - 1)
                torch.testing.assert_close(per_layer[MAIN].positions[:2], expected_positions)
                for value in per_layer.values():
                    assert (value.slot_mapping[:2][raw_lengths > max_model_len] == -1).all()
        for group in groups[1:]:
            for builder in group.metadata_builders:
                builder.build.assert_called_once()
        for name in (MAIN, INDEXER, STATE):
            assert len({step[name].slot_mapping.data_ptr() for step in steps}) == 3
        torch.testing.assert_close(lengths, torch.tensor([12, 24], dtype=torch.int32))
        assert not hasattr(proposer, "_num_rejected_tokens")


def test_factory_selects_glm_proposer_for_nested_multimodal_text_config():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="glm5_next"),
            hf_text_config=SimpleNamespace(model_type="glm5_next_text"),
        ),
        speculative_config=SimpleNamespace(use_step3p5_mtp=lambda: False),
    )
    with patch("vllm_ascend.spec_decode.AscendGlm5NextMTPProposer") as proposer:
        result = get_spec_decode_method("mtp", config, "cpu", None)
    proposer.assert_called_once_with(config, "cpu", None)
    assert result is proposer.return_value


@pytest.mark.parametrize(
    "table, positions, num_reqs",
    [
        ([], [0, 0, 0], 0),
        ([[]], [0, 1, 2], 1),
        ([[5, -1]], [-1, 8, 16], 1),
    ],
)
def test_proposer_state_slots_mask_idle_and_invalid_pages(table, positions, num_reqs):
    _, _, spec = _specs()
    group = AttentionGroup(AscendIndexerKPoolStateBackend, [STATE], spec, 0)
    state_table = torch.tensor(table, dtype=torch.int32).reshape(
        num_reqs, 0 if not table or not table[0] else len(table[0])
    )
    proposer = object.__new__(AscendGlm5NextMTPProposer)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=[SimpleNamespace(get_device_tensor=lambda: state_table)])
    )
    common = SimpleNamespace(
        num_reqs=num_reqs,
        num_input_tokens=3,
        positions=torch.tensor(positions),
        query_start_loc=torch.tensor([0, 3][: num_reqs + 1]),
        slot_mapping=torch.tensor([0, 1, 2]),
    )
    metadata = proposer._cache_group_common_metadata(common, group)
    assert metadata.slot_mapping.tolist() == [-1, -1, -1]
    assert common.slot_mapping.tolist() == [0, 1, 2]
