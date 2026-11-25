# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoopConfig:
    start: int
    end: int
    window: int

    @classmethod
    def from_hf_config(cls, config):
        if getattr(config, "glm54_dummy", False) is not True:
            raise ValueError("GLM-5.4 prototype requires an explicitly marked dummy checkpoint")
        values = (config.loop_start_layer, config.loop_end_layer, config.loop_window_size)
        if any(type(value) is not int for value in values):
            raise ValueError("Loop bounds and window size must be integers")
        start, end, window = values
        if not 0 <= start < end <= config.num_hidden_layers or window <= 0:
            raise ValueError(
                "Expected 0 <= loop_start_layer < loop_end_layer <= num_hidden_layers and a positive window"
            )
        return cls(start, end, window)


class DampGate(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        beta = self.proj(self.norm(inputs)).float().sigmoid()
        return (inputs.float() + beta * (candidate.float() - inputs.float())).to(inputs.dtype)


def two_bank_mla(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    first_cache: tuple[torch.Tensor, torch.Tensor],
    local_cache: tuple[torch.Tensor, torch.Tensor],
    first_indices: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    window: int,
    scale: float,
    local_block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager reference MLA over indexed first-pass KV and causal second-pass KV.

    Both banks use engine-managed pages and separate logical block tables.
    The local table must retain the windows for every query in this chunk,
    including earlier queries when the chunk is larger than the window.
    """
    num_tokens = q_latent.shape[0]
    if num_tokens == 0:
        return torch.empty_like(q_latent)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    request_ids = torch.repeat_interleave(
        torch.arange(query_lens.numel(), device=q_latent.device), query_lens, output_size=num_tokens
    )
    query_positions = (
        seq_lens[request_ids]
        - query_lens[request_ids]
        + torch.arange(num_tokens, device=q_latent.device)
        - query_start_loc[request_ids]
    )
    sparse_positions = first_indices.reshape(num_tokens, -1).long()
    local_positions = query_positions[:, None] - torch.arange(window - 1, -1, -1, device=q_latent.device)
    sparse_mask = (sparse_positions >= 0) & (sparse_positions <= query_positions[:, None])
    local_mask = local_positions >= 0
    page_size = first_cache[0].shape[1]

    def gather(cache, positions, table, valid):
        safe_positions = torch.where(valid, positions, 0).clamp(0, table.shape[1] * page_size - 1)
        pages = table[request_ids[:, None], safe_positions // page_size].long()
        slots = pages * page_size + safe_positions % page_size
        return tuple(component.reshape(-1, component.shape[-1])[slots] for component in cache)

    sparse_latent, sparse_rope = gather(first_cache, sparse_positions, block_table, sparse_mask)
    local_latent, local_rope = gather(
        local_cache, local_positions, block_table if local_block_table is None else local_block_table, local_mask
    )
    latent = torch.cat((sparse_latent, local_latent), dim=1).float()
    rope = torch.cat((sparse_rope, local_rope), dim=1).float()
    mask = torch.cat((sparse_mask, local_mask), dim=1)
    latent = latent.masked_fill(~mask[:, :, None], 0)
    rope = rope.masked_fill(~mask[:, :, None], 0)
    scores = torch.einsum("thd,tkd->thk", q_latent.float(), latent)
    scores += torch.einsum("thd,tkd->thk", q_rope.float(), rope)
    scores = (scores * scale).masked_fill(~mask[:, None, :], -torch.inf)
    probabilities = scores.softmax(dim=-1)
    # Same-position entries from different passes remain separate softmax terms.
    return torch.einsum("thk,tkd->thd", probabilities, latent).to(q_latent.dtype)
