# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.models.glm5next.model import Glm5NextMoE
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_glm5_moe_routes_hidden_states_once_with_internal_gate(dtype):
    torch.manual_seed(3)
    hidden = torch.randn(4, 16, dtype=dtype)
    gate = nn.Linear(16, 6, bias=False, dtype=torch.float32)
    gate.forward = Mock(side_effect=lambda values: (F.linear(values.float(), gate.weight), None))
    routed = Mock(forward_impl=Mock(return_value=hidden))
    runner = SimpleNamespace(
        _sequence_parallel_context=nullcontext,
        ascend_shared_experts=None,
        is_internal_router=True,
        gate=gate,
        routed_experts=routed,
    )
    moe = Glm5NextMoE.__new__(Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = False
    moe.gate = gate
    moe.experts = Mock(
        side_effect=lambda **kwargs: AscendMoERunner._forward_impl(runner, **kwargs, shared_experts_input=None)
    )

    output = moe(hidden)

    gate.forward.assert_not_called()
    routed.forward_impl.assert_called_once()
    actual = routed.forward_impl.call_args.kwargs
    assert actual["hidden_states"] is hidden
    torch.testing.assert_close(actual["router_logits"], F.linear(hidden.float(), gate.weight))
    torch.testing.assert_close(output, hidden)
