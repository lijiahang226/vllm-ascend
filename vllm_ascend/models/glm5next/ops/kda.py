# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM bounded-gate contracts for the AscendC KDA operators."""

import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.models.glm5next.ops.state_ops import gather_initial_states, scatter_states

KDA_CHUNK_SIZE = 64
KDA_MAX_RECURRENT_TOKENS = 8


def recurrent_kda(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    state,
    cu_seqlens,
    state_indices,
    a_log,
    dt_bias,
    lower_bound,
    num_accepted_tokens=None,
):
    """Update the selected VK state slots, including MTP rejection rollback."""
    num_seqs = cu_seqlens.numel() - 1
    state_indices = state_indices[:num_seqs]
    if num_accepted_tokens is not None:
        num_accepted_tokens = num_accepted_tokens[:num_seqs]
    output = torch.ops._C_ascend.recurrent_kda(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        raw_gate.contiguous(),
        raw_beta.contiguous(),
        state,
        cu_seqlens,
        state_indices,
        a_log.reshape(-1).contiguous(),
        dt_bias.contiguous(),
        num_accepted_tokens=num_accepted_tokens,
        scale=q.shape[-1] ** -0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
    )
    valid = torch.arange(q.shape[1], device=q.device) < cu_seqlens[-1]
    return output.masked_fill(~valid[None, :, None, None], 0)


def chunk_kda(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    state,
    state_indices,
    has_initial_state,
    metadata,
    a_log,
    dt_bias,
    lower_bound,
):
    """Run prefill with CPU chunk descriptors prepared by the GDN builder."""
    if metadata.keep_meta is not None:
        state_indices = state_indices[metadata.keep_meta]
        has_initial_state = has_initial_state[metadata.keep_meta]
    initial_state = gather_initial_states(state, state_indices, has_initial_state).float().contiguous()
    cu_seqlens = metadata.cu_seqlens_host if metadata.cu_seqlens_kern is None else metadata.cu_seqlens_kern
    output, final_state, *_ = torch.ops._C_ascend.chunk_kda_fwd(
        l2norm_fwd(q.contiguous()),
        l2norm_fwd(k.contiguous()),
        v.contiguous(),
        raw_gate.contiguous(),
        raw_beta.float().sigmoid().contiguous(),
        q.shape[-1] ** -0.5,
        KDA_CHUNK_SIZE,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=metadata.chunk_indices_chunk64_host,
        safe_gate=True,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        state_v_first=True,
    )
    scatter_states(state, final_state.to(state.dtype), state_indices)
    return output
