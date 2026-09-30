# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.attention.context_parallel.sfa_cp as sfa_cp
import vllm_ascend.attention.sfa_v1 as sfa
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder, PreprocessType
from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def test_shared_mtp_boundary_survives_rejection_and_length_updates():
    step_zero = torch.tensor([8, 2049, 4096, 0], dtype=torch.int32)
    proposer = SimpleNamespace(mtp_shared_seq_lens=torch.zeros_like(step_zero))
    common = SimpleNamespace(seq_lens=step_zero.clone())
    AscendSpecDecodeBaseProposer._set_mtp_shared_seq_lens(proposer, common, 0)
    assert common.mtp_shared_seq_lens is None

    # The first request selects an earlier query after rejecting three tokens.
    common.seq_lens.sub_(torch.tensor([3, 2, 0, 0], dtype=torch.int32))
    AscendSpecDecodeBaseProposer._prepare_mtp_shared_seq_lens(proposer, common)
    snapshot_ptr = proposer.mtp_shared_seq_lens.data_ptr()
    for draft_index in (1, 2):
        common.seq_lens[:3].add_(1)
        AscendSpecDecodeBaseProposer._set_mtp_shared_seq_lens(proposer, common, draft_index)
        assert common.mtp_shared_seq_lens.data_ptr() == snapshot_ptr
        torch.testing.assert_close(common.mtp_shared_seq_lens, torch.tensor([5, 2047, 4096, 0], dtype=torch.int32))
    torch.testing.assert_close(common.seq_lens, torch.tensor([7, 2049, 4098, 0], dtype=torch.int32))

    # A new graph replay changes values in the same buffer and may use fewer requests.
    common.seq_lens = torch.tensor([9, 0], dtype=torch.int32)
    AscendSpecDecodeBaseProposer._prepare_mtp_shared_seq_lens(proposer, common)
    common.seq_lens[0].add_(1)
    AscendSpecDecodeBaseProposer._set_mtp_shared_seq_lens(proposer, common, 1)
    assert common.mtp_shared_seq_lens.data_ptr() == snapshot_ptr
    torch.testing.assert_close(common.mtp_shared_seq_lens, torch.tensor([9, 0], dtype=torch.int32))
    assert torch.count_nonzero(proposer.mtp_shared_seq_lens[2:]) == 0
    torch.testing.assert_close(step_zero, torch.tensor([8, 2049, 4096, 0], dtype=torch.int32))

    # No snapshot is allocated when sharing is disabled or the KV cache is sharded.
    proposer.mtp_shared_seq_lens = None
    AscendSpecDecodeBaseProposer._prepare_mtp_shared_seq_lens(proposer, common)
    AscendSpecDecodeBaseProposer._set_mtp_shared_seq_lens(proposer, common, 1)
    assert common.mtp_shared_seq_lens is None


@pytest.mark.parametrize("nope", [False, True])
def test_sfa_metadata_preserves_live_lengths_and_shared_boundary(monkeypatch, nope):
    live_lengths = torch.tensor([6, 4097, 0], dtype=torch.int32)
    shared_lengths = torch.tensor([5, 4096, 0], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=2,
        num_input_tokens=3,
        block_table_tensor=torch.zeros(3, 257, dtype=torch.int32),
        slot_mapping=torch.tensor([5, 4096, -1]),
        positions=torch.tensor([5, 4096, 0]),
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        seq_lens=live_lengths,
        mtp_shared_seq_lens=shared_lengths,
        _seq_lens_cpu=live_lengths,
        causal=True,
        attn_state=sfa.AscendAttentionState.SpecDecoding,
        max_query_len=1,
        max_seq_len=4097,
    )
    builder = SimpleNamespace(
        kernel_block_size=16,
        nope=nope,
        metadata_cls=AscendSFAMetadata,
        model_config=SimpleNamespace(get_head_size=lambda: 8),
        attn_mask_builder=SimpleNamespace(get_attention_mask=lambda *_args: None),
        decode_threshold=1,
        use_pcp=False,
        nope_states={1: Mock()},
        _prepare_parallel_metadata=lambda _common, cos, sin, slots, *_args: (cos, sin, slots, {}),
    )
    monkeypatch.setattr(sfa, "get_cos_and_sin_mla", lambda *_args, **_kwargs: (torch.zeros(3, 1), torch.zeros(3, 1)))
    monkeypatch.setattr(sfa, "split_decodes_and_prefills", lambda *_args, **_kwargs: (3, 0, 3, 0))

    metadata = AscendSFAMetadataBuilder._build(builder, common, draft_index=1)

    assert metadata.seq_lens.data_ptr() == live_lengths.data_ptr()
    assert metadata.slot_mapping.data_ptr() == common.slot_mapping.data_ptr()
    if nope:
        assert metadata.mtp_shared_seq_lens is None
    else:
        assert metadata.mtp_shared_seq_lens.data_ptr() == shared_lengths.data_ptr()


def test_dsa_cp_keeps_shared_boundary_aligned_with_local_queries(monkeypatch):
    monkeypatch.setattr(sfa_cp, "get_tp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=1))
    builder = AscendSFADSACPMetadataBuilder.__new__(AscendSFADSACPMetadataBuilder)
    builder.nope = False
    builder.dsa_cp_spec_actual_seq_lengths_query = [torch.empty(4, dtype=torch.int32)]
    builder.dsa_cp_spec_actual_seq_lengths_key = [torch.empty(4, dtype=torch.int32)]
    common = SimpleNamespace(
        num_reqs=4,
        num_input_tokens=4,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32),
        mtp_shared_seq_lens=torch.tensor([5, 2047, 4096, 0], dtype=torch.int32),
    )
    live_lengths = torch.tensor([6, 2048, 4097, 0], dtype=torch.int32)
    cos = torch.zeros(4, 1, 1, 2)
    slots = torch.tensor([5, 2047, 4096, -1])

    _, _, _, extra = builder._prepare_parallel_metadata(
        common, cos, cos, slots, common.query_start_loc[1:], live_lengths, 1
    )

    context = extra["dsa_cp_context"]
    torch.testing.assert_close(context.actual_seq_lengths_query, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(context.actual_seq_lengths_key, torch.tensor([0, 0, 4096, 0], dtype=torch.int32))
    torch.testing.assert_close(context.slot_mapping_cp, slots[2:])
    torch.testing.assert_close(live_lengths, torch.tensor([6, 2048, 4097, 0], dtype=torch.int32))


@pytest.mark.parametrize("is_mtp", [True, False])
def test_forward_reuses_boundary_without_changing_indices_or_cache_slots(monkeypatch, is_mtp):
    buffer = torch.full((1, 2048), -1, dtype=torch.int32)
    buffer[0, :5] = torch.tensor([4, 2, 0, 3, 1], dtype=torch.int32)
    original_indices = buffer.clone()
    metadata = SimpleNamespace(
        positions=torch.tensor([5]),
        num_actual_tokens=1,
        num_input_tokens=1,
        slot_mapping=torch.tensor([5]),
        cum_query_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        mtp_shared_seq_lens=torch.tensor([5], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        cos=None,
        sin=None,
    )
    hidden = torch.zeros(1, 1, 8)
    q_pe = torch.zeros(1, 1, 2)
    preprocess = Mock(return_value=(hidden, hidden, q_pe, hidden))
    impl = SimpleNamespace(
        qk_rope_head_dim=2,
        g_proj=None,
        scale=0.125,
        layer_name="mtp" if is_mtp else "shared",
        layerwise_kv_cache_hook=None,
        preprocess_type=PreprocessType.MLAPO,
        runtime_has_indexer=is_mtp,
        _is_mtp_layer=is_mtp,
        skip_topk=True,
        topk_indices_buffer=buffer,
        _compose_sfa_kv_cache=lambda cache: cache,
        _get_sfa_kv_slot_mapping=lambda meta: meta.slot_mapping,
        _get_indexer_attn_metadata=lambda: metadata if is_mtp else None,
        _sfa_preprocess_mlapo=preprocess,
        _prepare_indexer_metadata=lambda *_args: None,
        indexer=Mock(return_value=None),
        _v_up_proj=lambda x: x,
        _finalize_o_proj=lambda x, out, _: out.copy_(x),
    )
    impl._get_parallel_forward_context = lambda *args: AscendSFAImpl._get_parallel_forward_context(impl, *args)
    impl._get_indexcache_topk_indices = lambda n: AscendSFAImpl._get_indexcache_topk_indices(impl, n)
    impl._execute_sparse_flash_attention_process = lambda *args: (
        BaseDeviceAdaptor.execute_sparse_flash_attention_process(impl, *args)
    )
    op = Mock(return_value=hidden)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention", op, raising=False)
    for name in ("wait_for_kv_layer_from_connector", "notify_kv_cache_written", "maybe_save_kv_layer_to_connector"):
        monkeypatch.setattr(sfa, name, Mock())
    monkeypatch.setattr(sfa, "attention_transfer_window", nullcontext)

    output = torch.empty_like(hidden)
    for live_length in (6, 7):
        metadata.seq_lens.fill_(live_length)
        metadata.positions.fill_(live_length - 1)
        metadata.slot_mapping.fill_(live_length - 1)
        AscendSFAImpl.forward(
            impl, impl.layer_name, hidden, (torch.zeros(1, 8, 1, 10, dtype=torch.int8),), metadata, output
        )

        kwargs = op.call_args.kwargs
        torch.testing.assert_close(kwargs["actual_seq_lengths_query"], torch.tensor([1], dtype=torch.int32))
        expected_length = 5 if is_mtp else live_length
        torch.testing.assert_close(kwargs["actual_seq_lengths_kv"], torch.tensor([expected_length], dtype=torch.int32))
        assert kwargs["sparse_indices"].data_ptr() == buffer.data_ptr()
        torch.testing.assert_close(buffer, original_indices)
        if is_mtp:
            # The op's active prefix contains no -1, without selecting the new draft KV.
            assert (kwargs["sparse_indices"][0, 0, :expected_length] >= 0).all()
            assert impl.indexer.call_args.kwargs["compute_topk"] is False
        assert preprocess.call_args.kwargs["slot_mapping"][0] == live_length - 1
        assert metadata.seq_lens[0] == live_length
    assert impl.indexer.call_count == (2 if is_mtp else 0)
    torch.testing.assert_close(output, hidden)
