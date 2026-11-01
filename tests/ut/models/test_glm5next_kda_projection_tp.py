# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend import ascend_forward_context
from vllm_ascend.models.glm5next import kda_projection_tp


@pytest.mark.parametrize("ai_cpu", [False, True])
def test_fused_projection_uses_kda_group_and_returns_owner_token_shard(monkeypatch, ai_cpu):
    group = SimpleNamespace(world_size=8, rank_in_group=3, device_group=MagicMock())
    group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "kda-dp-axis"
    profile = MagicMock()
    profile.supports.side_effect = lambda capability: ai_cpu or capability.name == "GRAPH_MM_REDUCE_SCATTER_FUSION"
    monkeypatch.setattr(kda_projection_tp, "get_current_hardware_profile", lambda: profile)
    helper = kda_projection_tp.KDAProjectionTP(group, 64, 128, fuse_o_proj=True)
    weight = torch.randn(16, 1024, dtype=torch.bfloat16)
    x = torch.randn(48, 1024, dtype=torch.bfloat16)
    expected = torch.randn(6, 16, dtype=x.dtype)
    fused = MagicMock(return_value=(expected, None) if ai_cpu else expected)
    name = "npu_quant_mm_reduce_scatter" if ai_cpu else "npu_mm_reduce_scatter_base"
    monkeypatch.setattr(kda_projection_tp.torch_npu, name, fused, raising=False)
    layer = MagicMock(weight=weight, bias=None, quant_method=kda_projection_tp.UnquantizedLinearMethod())
    assert helper.project_output(layer, x) is expected
    args, kwargs = fused.call_args
    assert args[0] is x
    torch.testing.assert_close(args[1], weight.t())
    assert args[2:] == ("kda-dp-axis", 8)
    assert kwargs["comm_mode"] == ("ai_cpu" if ai_cpu else "aiv")
    group.device_group._get_backend.return_value.get_hccl_comm_name.assert_called_once_with(3)
    layer.assert_not_called()


@pytest.mark.parametrize("reason", ["disabled", "quantized", "bias", "dtype", "small_k", "strided_weight"])
def test_unsupported_fusion_preserves_projection_and_reduce_scatter(reason):
    group = SimpleNamespace(world_size=8, reduce_scatter=MagicMock(return_value=torch.zeros(3, 16)))
    helper = kda_projection_tp.KDAProjectionTP(group, 64, 128)
    helper.mmrs_hcom = None if reason == "disabled" else "kda-group"
    width = 128 if reason == "small_k" else 1024
    weight = torch.zeros(16, width, dtype=torch.float32 if reason == "dtype" else torch.bfloat16)
    if reason == "strided_weight":
        weight = torch.zeros(32, width * 2, dtype=torch.bfloat16)[::2, ::2]
    x = torch.zeros(24, width, dtype=torch.bfloat16)
    partial = torch.ones(24, 16)
    layer = MagicMock(
        weight=weight,
        bias=torch.ones(16) if reason == "bias" else None,
        quant_method=object() if reason == "quantized" else kda_projection_tp.UnquantizedLinearMethod(),
        return_value=(partial, None),
    )
    assert helper.project_output(layer, x) is group.reduce_scatter.return_value
    layer.assert_called_once_with(x)
    group.reduce_scatter.assert_called_once_with(partial, dim=0)


class CollectiveReference:
    def __init__(self, size):
        self.size = size
        self.barrier = Barrier(size, timeout=20)
        self.inputs = {}
        self.outputs = {}

    def exchange(self, rank, kind, tensor):
        self.inputs[rank] = tensor.clone()
        self.barrier.wait()
        if rank == 0:
            tensors = [self.inputs[i] for i in range(self.size)]
            for destination in range(self.size):
                if kind == "gather":
                    result = torch.cat(tensors)
                elif kind == "scatter":
                    result = torch.stack(tensors).sum(0).chunk(self.size)[destination]
                else:
                    result = torch.stack([source[destination] for source in tensors])
                self.outputs[destination] = result
        self.barrier.wait()
        result = self.outputs[rank].clone()
        self.barrier.wait()
        return result


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("runner_v2", [False, True])
def test_projection_tp_matches_dense_weights_with_ragged_and_idle_owners(monkeypatch, size, runner_v2):
    generator = torch.Generator().manual_seed(91)
    hidden_dim, heads, dim = 7, 8, 2
    width = heads * dim
    weights = [torch.randn(n, hidden_dim, generator=generator) * 0.1 for n in (width, width, width, heads, dim, dim)]
    forget_weight = torch.randn(width, dim, generator=generator) * 0.1
    gate_weight = torch.randn(width, dim, generator=generator) * 0.1
    output_weight = torch.randn(hidden_dim, width, generator=generator) * 0.1
    initial = torch.randn(3, heads, dim, generator=generator) * 0.1
    states = [initial.clone() for _ in range(size)]
    dense_states = [initial.clone() for _ in range(size)]
    counts = ([3, 0, 1, 2][:size], [1, 2, 0, 3][:size], [0] * size)
    inputs = [[torch.randn(n, hidden_dim, generator=generator) for n in step] for step in counts]
    coordinator = CollectiveReference(size)
    thread = local()

    def core(qkv, forget, beta, gate, output, prefix):
        count = thread.count
        q, k, v = qkv[:count].reshape(count, 3, heads, dim).unbind(1)
        state = thread.state
        state[:count].mul_(forget[0, :count].sigmoid()).add_(q * k + v * beta[0, :count, :, None].sigmoid())
        output[0, :count].copy_(state[:count] * gate[:count].sigmoid())

    monkeypatch.setattr(ascend_forward_context, "get_forward_context", lambda: thread.context)
    monkeypatch.setattr(ascend_forward_context, "_USE_V2_EXTRA_KWARGS", runner_v2)
    monkeypatch.setattr(torch.ops.vllm, "glm5next_kda_forward", core, raising=False)

    def all_to_all(output, tensor, group):
        output.copy_(coordinator.exchange(group.rank, "all_to_all", tensor))

    monkeypatch.setattr(kda_projection_tp.dist, "all_to_all_single", all_to_all)

    def linear(weight):
        return lambda tensor: (F.linear(tensor, weight), None)

    def worker(rank):
        group = SimpleNamespace(world_size=size, rank_in_group=rank, device_group=SimpleNamespace(rank=rank))
        group.all_gather = lambda tensor, dim: coordinator.exchange(rank, "gather", tensor)
        group.reduce_scatter = lambda tensor, dim: coordinator.exchange(rank, "scatter", tensor)
        sharded = [w.chunk(size)[rank] if index < 4 else w for index, w in enumerate(weights)]
        layer = SimpleNamespace(
            prefix="kda",
            num_heads=heads,
            in_proj_qkvbfg_a=linear(torch.cat(sharded)),
            f_b_proj=linear(forget_weight.chunk(size)[rank]),
            g_b_proj=linear(gate_weight.chunk(size)[rank]),
            o_proj=linear(output_weight.chunk(size, dim=1)[rank]),
        )
        helper = kda_projection_tp.KDAProjectionTP(group, heads, dim)
        for step, batches in enumerate(inputs):
            hidden = batches[rank]
            extras = {"max_tokens_across_dp": max(counts[step])}
            thread.context = SimpleNamespace(additional_kwargs=extras) if runner_v2 else SimpleNamespace(**extras)
            thread.count, thread.state = hidden.shape[0], states[rank]
            output = torch.empty_like(hidden)
            helper.forward(layer, hidden, output)

            projected = F.linear(hidden, torch.cat(weights))
            qkv, beta, forget_a, gate_a = projected.split((3 * width, heads, dim, dim), dim=-1)
            dense_output = torch.zeros(1, hidden.shape[0], heads, dim)
            thread.state = dense_states[rank]
            core(
                qkv,
                F.linear(forget_a, forget_weight).reshape(1, -1, heads, dim),
                beta.unsqueeze(0),
                F.linear(gate_a, gate_weight).reshape(-1, heads, dim),
                dense_output,
                "kda",
            )
            expected = F.linear(dense_output.flatten(2).squeeze(0), output_weight)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(states[rank], dense_states[rank])

    with ThreadPoolExecutor(max_workers=size) as executor:
        list(executor.map(worker, range(size)))
