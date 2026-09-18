# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC short convolution for GLM prefill, decode and MTP verification."""

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    run_mode: int,
    initial_state_mode: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume GDN metadata and update the caller's [cache, state_len, dim] state."""
    # Padded requests can be skipped by the kernel; their output must stay zero.
    output = torch.zeros_like(x)
    if cache_indices.shape[0] == 0:
        return output
    # Match Ascend K3: pass the original state view and cache indices to the
    # existing operator for both prefill and update.
    # Return the declared result so graph functionalization retains the call.
    return torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=conv_state,
        bias_opt=None,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=initial_state_mode,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=run_mode,
    )
