# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA projection sharding with request-local recurrent state."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

if TYPE_CHECKING:
    from .kda import Glm5NextLinearAttention


class KDAProjectionTP:
    def __init__(self, group: GroupCoordinator, num_heads: int, head_dim: int):
        self.group = group
        self.size = group.world_size
        self.heads = num_heads // self.size
        self.head_dim = head_dim
        self.width = self.heads * head_dim

    def _all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, group=self.group.device_group)
        return output

    def gather_projected_heads(
        self, qkv: torch.Tensor, forget: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = qkv.shape[0] // self.size
        packed = torch.cat((qkv, forget, gate, beta), dim=-1)
        received = self._all_to_all(packed.view(self.size, tokens, -1))
        qkv, forget, gate, beta = received.split((3 * self.width, self.width, self.width, self.heads), dim=-1)
        qkv = qkv.view(self.size, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3, 4).reshape(tokens, -1)
        forget = forget.permute(1, 0, 2).reshape(1, tokens, -1, self.head_dim)
        gate = gate.permute(1, 0, 2).reshape(tokens, -1, self.head_dim)
        beta = beta.permute(1, 0, 2).reshape(1, tokens, -1)
        return qkv, forget, gate, beta

    def gather_output_tokens(self, output: torch.Tensor) -> torch.Tensor:
        tokens = output.shape[0]
        by_head = output.view(tokens, self.size, self.width).permute(1, 0, 2).contiguous()
        return self._all_to_all(by_head).reshape(self.size * tokens, self.width)

    def forward(self, layer: "Glm5NextLinearAttention", hidden: torch.Tensor, output: torch.Tensor) -> None:
        local_tokens = hidden.shape[0]
        capacity = max(local_tokens, _EXTRA_CTX.max_tokens_across_dp or local_tokens)
        if capacity == 0:
            return
        hidden = F.pad(hidden, (0, 0, 0, capacity - local_tokens))
        gathered = self.group.all_gather(hidden, dim=0)
        projected = layer.in_proj_qkvbfg_a(gathered)[0]
        qkv, beta, forget_a, gate_a = projected.split(
            (3 * self.width, self.heads, self.head_dim, self.head_dim), dim=-1
        )
        forget = layer.f_b_proj(forget_a)[0]
        gate = layer.g_b_proj(gate_a)[0]
        qkv, forget, gate, beta = self.gather_projected_heads(qkv, forget, gate, beta)
        # Every owner retains the original cache indices, MTP rollback and PD
        # load/copy ordering; only projected activations cross the DP ranks.
        core = hidden.new_zeros((1, capacity, layer.num_heads, self.head_dim))
        torch.ops.vllm.glm5next_kda_forward(qkv, forget, beta, gate, core, layer.prefix)
        sharded = self.gather_output_tokens(core.flatten(2).squeeze(0))
        partial = layer.o_proj(sharded)[0]
        reduced = self.group.reduce_scatter(partial, dim=0)
        output.copy_(reduced[:local_tokens])
