/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef CAUSAL_CONV1D_UPDATE_TORCH_ADPT_H
#define CAUSAL_CONV1D_UPDATE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_causal_conv1d_update(
    const at::Tensor &output,
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &conv_state,
    const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &cache_indices,
    const c10::optional<at::Tensor> &num_accepted_tokens,
    const c10::optional<at::Tensor> &block_idx_last_scheduled_token,
    const c10::optional<at::Tensor> &initial_state_idx,
    const std::string &activation,
    int64_t null_block_id,
    int64_t max_query_len)
{
    // Use the standard CANN API, whose convStatesRef argument is input/output.
    // Callers must qualify state writeback for their CANN version and layout.
    char *activation_ptr = const_cast<char *>(activation.c_str());
    EXEC_NPU_CMD(aclnnCausalConv1dUpdate, x, weight, conv_state, bias,
                 query_start_loc, cache_indices, num_accepted_tokens,
                 block_idx_last_scheduled_token, initial_state_idx,
                 activation_ptr, null_block_id, max_query_len, output);
    return output;
}

} // namespace vllm_ascend

#endif
