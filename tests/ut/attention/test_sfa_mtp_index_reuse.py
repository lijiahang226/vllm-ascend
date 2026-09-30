# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.attention.sfa_v1 as sfa
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
from vllm_ascend.device.device_op import BaseDeviceAdaptor


def test_mtp_reuse_extends_short_selection_in_place():
    # Step 0 saw five keys; subsequent single-query steps see six and seven.
    buffer = torch.full((1, 2048), -1, dtype=torch.int32)
    buffer[0, :5] = torch.tensor([4, 2, 0, 3, 1])
    indices = buffer.unsqueeze(1)
    pointer = buffer.data_ptr()
    for position in (5, 6):
        metadata = SimpleNamespace(seq_lens=torch.tensor([position + 1], dtype=torch.int32))
        AscendSFAImpl._refresh_mtp_topk_indices(indices, metadata)
        torch.testing.assert_close(buffer[0, : position + 1], torch.arange(position + 1, dtype=torch.int32))
        assert (buffer[0, position + 1 :] == -1).all()
        assert indices.data_ptr() == pointer


@pytest.mark.parametrize("dsa_cp", [False, True])
def test_mtp_reuse_handles_topk_boundary_and_graph_padding(dsa_cp):
    topk = 2048
    # A strided shared buffer, two short rows, a full ranked row and padding.
    backing = torch.full((8, topk + 2), -99, dtype=torch.int32)
    buffer = backing[::2, :topk]
    buffer.fill_(-1)
    buffer[0, : topk - 2] = torch.arange(topk - 2, dtype=torch.int32)
    buffer[1, : topk - 1] = torch.arange(topk - 1, dtype=torch.int32)
    buffer[2] = torch.arange(4095, 2047, -1, dtype=torch.int32)
    buffer[3].fill_(0)
    ranked = buffer[2].clone()
    seq_lens = torch.tensor([topk - 1, topk + 1, 4097, 0], dtype=torch.int32)
    local_start = 4 if dsa_cp else 0
    metadata = SimpleNamespace(
        seq_lens=torch.cat([torch.zeros(local_start, dtype=torch.int32), seq_lens[:3] if dsa_cp else seq_lens]),
    )
    if dsa_cp:
        metadata.dsa_cp_context = SimpleNamespace(local_start=local_start)

    AscendSFAImpl._refresh_mtp_topk_indices(buffer.unsqueeze(1), metadata)

    torch.testing.assert_close(buffer[0, : topk - 1], torch.arange(topk - 1, dtype=torch.int32))
    assert buffer[0, -1] == -1
    torch.testing.assert_close(buffer[1], torch.arange(topk, dtype=torch.int32))
    torch.testing.assert_close(buffer[2], ranked)
    assert (buffer[3] == -1).all()
    assert (backing[1::2] == -99).all()
    assert (backing[:, topk:] == -99).all()

    # The same graph bucket can contain fewer active requests on its next run.
    metadata.seq_lens[local_start + 2] = 0
    AscendSFAImpl._refresh_mtp_topk_indices(buffer.unsqueeze(1), metadata)
    assert (buffer[2:] == -1).all()

    # The proposer clamps requests beyond max_model_len to a single key.
    buffer[2].copy_(ranked)
    metadata.seq_lens[local_start + 2] = 1
    AscendSFAImpl._refresh_mtp_topk_indices(buffer.unsqueeze(1), metadata)
    assert buffer[2, 0] == 0
    assert (buffer[2, 1:] == -1).all()


@pytest.mark.parametrize("is_mtp", [True, False])
def test_forward_refreshes_mtp_indices_before_kv_quant_attention(monkeypatch, is_mtp):
    buffer = torch.full((1, 2048), -1, dtype=torch.int32)
    buffer[0, :5] = torch.arange(5, dtype=torch.int32)
    metadata = SimpleNamespace(
        positions=torch.tensor([5]),
        num_actual_tokens=1,
        num_input_tokens=1,
        slot_mapping=torch.tensor([5]),
        cum_query_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        cos=None,
        sin=None,
    )
    hidden = torch.zeros(1, 1, 8)
    q_pe = torch.zeros(1, 1, 2)
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
        _get_parallel_forward_context=lambda *_args: SimpleNamespace(
            actual_seq_lengths_query=metadata.cum_query_lens,
            actual_seq_lengths_key=metadata.seq_lens,
            topk_num_tokens=1,
            gather_full_o_proj=False,
        ),
        _sfa_preprocess_mlapo=lambda **_kwargs: (hidden, hidden, q_pe, hidden),
        _prepare_indexer_metadata=lambda *_args: None,
        indexer=Mock(return_value=None),
        _refresh_mtp_topk_indices=AscendSFAImpl._refresh_mtp_topk_indices,
        _v_up_proj=lambda x: x,
        _finalize_o_proj=lambda x, out, _: out.copy_(x),
    )
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
    AscendSFAImpl.forward(
        impl, impl.layer_name, hidden, (torch.zeros(1, 8, 1, 10, dtype=torch.int8),), metadata, output
    )

    op.assert_called_once()
    kwargs = op.call_args.kwargs
    torch.testing.assert_close(kwargs["actual_seq_lengths_query"], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(kwargs["actual_seq_lengths_kv"], torch.tensor([6], dtype=torch.int32))
    assert kwargs["sparse_indices"].data_ptr() == buffer.data_ptr()
    assert kwargs["sparse_indices"][0, 0, 5] == (5 if is_mtp else -1)
    if is_mtp:
        impl.indexer.assert_called_once()
        assert impl.indexer.call_args.kwargs["compute_topk"] is False
    else:
        impl.indexer.assert_not_called()
    torch.testing.assert_close(output, hidden)
