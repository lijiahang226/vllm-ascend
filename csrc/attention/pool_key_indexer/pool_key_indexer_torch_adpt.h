/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef POOL_KEY_INDEXER_TORCH_ADPT_H
#define POOL_KEY_INDEXER_TORCH_ADPT_H

namespace vllm_ascend {

constexpr int64_t POOL_KEY_INDEXER_SIZE = 8;
constexpr int64_t POOL_KEY_INDEXER_DIM_0 = 0;
constexpr int64_t POOL_KEY_INDEXER_DIM_1 = 1;

// Derive output shapes from query shape + layout.
//   BSND: query (B,S1,N1,D) -> indices (B,S1,topk+poolSize-1) / values (B,S1,topk/poolSize)
//   TND:  query (T1,N1,D)   -> indices (T1,topk+poolSize-1)     / values (T1,topk/poolSize)
// Note: PKI output has no N2 dimension (N2 is fixed to 1), unlike LIV2 which emits keyHeadNum.
std::tuple<at::Tensor, at::Tensor> ConstructPoolKeyIndexerOutputTensor(
    const at::Tensor& query, int64_t topk, int64_t pool_size,
    const std::string& query_layout_str, bool return_value)
{
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    TORCH_CHECK(pool_size > 0,
                "pool_size should be greater than 0, but now is ", pool_size);
    TORCH_CHECK(topk % pool_size == 0, "topk(", topk,
                ") should be divisible by pool_size(", pool_size, ").");

    int64_t indices_len = topk + pool_size - 1;
    int64_t values_len = topk / pool_size;

    at::SmallVector<int64_t, POOL_KEY_INDEXER_SIZE> indices_shape;
    at::SmallVector<int64_t, POOL_KEY_INDEXER_SIZE> values_shape;

    if (query_layout_str == "BSND") {
        indices_shape = {query.size(POOL_KEY_INDEXER_DIM_0),
                         query.size(POOL_KEY_INDEXER_DIM_1), indices_len};
        values_shape = {query.size(POOL_KEY_INDEXER_DIM_0),
                        query.size(POOL_KEY_INDEXER_DIM_1), values_len};
    } else {
        indices_shape = {query.size(POOL_KEY_INDEXER_DIM_0), indices_len};
        values_shape = {query.size(POOL_KEY_INDEXER_DIM_0), values_len};
    }

    at::Tensor sparse_indices_out =
        // KNOWN ISSUE (temporary safety net): the op may leave parts of the
        // output unwritten in some configs; -1-initialize so unwritten
        // entries are masked instead of garbage reaching
        // npu_sparse_flash_mla. Costs one device-side fill per call; revisit
        // once the op's output coverage is confirmed.
        at::full(indices_shape, -1, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        // Spec requires sparseValuesOut to be FLOAT (not query dtype)
        sparse_values_out =
            at::empty(values_shape, query.options().dtype(at::kFloat));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                              sparse_values_out);
}

std::vector<bool> IsContiguousAxes(const at::Tensor& tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();
    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguous_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguous_stride[i] = contiguous_stride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguous_stride[i]);
    }
    return result;
}

// pool_tail_k / actual_seq_q / actual_seq_k are device ValueDepend tensors
// in vLLM.  The generated host API, aclnnPoolKeyIndexerGetWorkspaceSize,
// accepts aclIntArray* for those arguments, whereas the graph-safe tensor API
// accepts aclTensor*.  EXEC_NPU_CMD derives the workspace API name from the run
// API and therefore cannot express this split-name pair.  Calling the host API
// with converted aclTensor* arguments corrupts its array lengths and can surface
// as std::bad_alloc before the kernel is launched.
template <typename... Ts>
void RunPoolKeyIndexerTensor(const at::Device& compute_device, Ts&... args)
{
    using namespace at_npu::native;

    const c10::OptionalDeviceGuard device_guard(compute_device);
    static const auto get_workspace_size_func_addr =
        GetOpApiFuncAddr("aclnnPoolKeyIndexerTensorGetWorkspaceSize");
    static const auto op_api_func_addr =
        GetOpApiFuncAddr("aclnnPoolKeyIndexer");
    static const auto init_mem_addr =
        GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto uninit_mem_addr =
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto release_mem_addr = GetOpApiFuncAddr("ReleaseHugeMem");

    TORCH_CHECK(get_workspace_size_func_addr != nullptr &&
                    op_api_func_addr != nullptr,
                "aclnnPoolKeyIndexerTensorGetWorkspaceSize or "
                "aclnnPoolKeyIndexer not found in opapi libraries");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    uint64_t* workspace_size_addr = &workspace_size;
    aclOpExecutor* executor = nullptr;
    aclOpExecutor** executor_addr = &executor;
    InitHugeMemThreadLocal init_mem_func =
        reinterpret_cast<InitHugeMemThreadLocal>(init_mem_addr);
    UnInitHugeMemThreadLocal uninit_mem_func =
        reinterpret_cast<UnInitHugeMemThreadLocal>(uninit_mem_addr);
    ReleaseHugeMem release_mem_func =
        reinterpret_cast<ReleaseHugeMem>(release_mem_addr);

    if (init_mem_func != nullptr) {
        init_mem_func(nullptr, false);
    }

    auto converted_params =
        ConvertTypes(args..., workspace_size_addr, executor_addr);
    auto get_workspace_size_func = ConvertToOpApiFunc(
        converted_params, get_workspace_size_func_addr);
    auto workspace_status = call(get_workspace_size_func, converted_params);
    if (workspace_status != 0) {
        ReleaseConvertTypes(converted_params);
        if (release_mem_func != nullptr) {
            release_mem_func(nullptr, false);
        }
        if (uninit_mem_func != nullptr) {
            uninit_mem_func(nullptr, false);
        }
        TORCH_CHECK(false,
                    "call aclnnPoolKeyIndexerTensorGetWorkspaceSize failed, "
                    "detail:", aclGetRecentErrMsg());
    }

    // Keep the tensor alive until OpCommand has launched the custom handler.
    at::Tensor workspace_tensor;
    void* workspace_addr = nullptr;
    if (workspace_size != 0) {
        at::TensorOptions options =
            at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspace_tensor =
            at::empty({static_cast<int64_t>(workspace_size)},
                      options.dtype(at::kByte));
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }

    auto acl_call = [converted_params, workspace_addr, workspace_size,
                     acl_stream, executor, op_api_func_addr,
                     release_mem_func]() mutable -> int {
        using OpApiFunc =
            int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
        auto op_api_func = reinterpret_cast<OpApiFunc>(op_api_func_addr);
        auto api_status =
            op_api_func(workspace_addr, workspace_size, executor, acl_stream);
        ReleaseConvertTypes(converted_params);
        if (release_mem_func != nullptr) {
            release_mem_func(nullptr, false);
        }
        TORCH_CHECK(api_status == 0,
                    "call aclnnPoolKeyIndexer failed, detail:",
                    aclGetRecentErrMsg());
        return api_status;
    };

    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnPoolKeyIndexer");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    if (uninit_mem_func != nullptr) {
        uninit_mem_func(nullptr, false);
    }
}

std::tuple<at::Tensor, at::Tensor> pool_key_indexer(
    const at::Tensor& query, const at::Tensor& pool_key,
    const at::Tensor& weights, const at::Tensor& pool_tail_k,
    const c10::optional<at::Tensor>& actual_seq_q,
    const c10::optional<at::Tensor>& actual_seq_k,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& q_descale,
    const c10::optional<at::Tensor>& k_descale, c10::string_view layout_q,
    c10::string_view layout_k, int64_t topk, int64_t pool_size,
    int64_t mask_mode, int64_t quant_mode, bool return_value)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(pool_key.numel() > 0, "Tensor pool_key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
    TORCH_CHECK(pool_tail_k.numel() > 0, "Tensor pool_tail_k is empty.");

    std::string query_layout_str = std::string(layout_q);
    std::string key_layout_str = std::string(layout_k);

    TORCH_CHECK(query_layout_str == "BSND" || query_layout_str == "TND",
                "layout_q must be BSND or TND, got ", query_layout_str);
    TORCH_CHECK(key_layout_str == "BSND" || key_layout_str == "TND" ||
                    key_layout_str == "PA_BBND",
                "layout_k must be BSND, TND or PA_BBND, got ",
                key_layout_str);

    auto check_same_device = [&query](const at::Tensor& tensor,
                                      const char* name) {
        TORCH_CHECK(tensor.device() == query.device(), name,
                    " must be on the same device as query (", query.device(),
                    "), got ", tensor.device());
    };
    check_same_device(pool_key, "pool_key");
    check_same_device(weights, "weights");
    check_same_device(pool_tail_k, "pool_tail_k");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 ||
                    query.scalar_type() == at::kHalf,
                "query must be BF16 or FP16");
    TORCH_CHECK(pool_key.scalar_type() == query.scalar_type() &&
                    weights.scalar_type() == query.scalar_type(),
                "pool_key and weights dtype must match query");
    TORCH_CHECK(pool_tail_k.scalar_type() == at::kLong &&
                    pool_tail_k.dim() == 1,
                "pool_tail_k must be rank-1 INT64");
    TORCH_CHECK(pool_tail_k.is_contiguous(),
                "pool_tail_k must be contiguous");

    if (actual_seq_q.has_value()) {
        check_same_device(*actual_seq_q, "actual_seq_q");
        TORCH_CHECK(actual_seq_q->scalar_type() == at::kLong &&
                        actual_seq_q->dim() == 1,
                    "actual_seq_q must be rank-1 INT64");
        TORCH_CHECK(actual_seq_q->is_contiguous(),
                    "actual_seq_q must be contiguous");
    }
    if (actual_seq_k.has_value()) {
        check_same_device(*actual_seq_k, "actual_seq_k");
        TORCH_CHECK(actual_seq_k->scalar_type() == at::kLong &&
                        actual_seq_k->dim() == 1,
                    "actual_seq_k must be rank-1 INT64");
        TORCH_CHECK(actual_seq_k->is_contiguous(),
                    "actual_seq_k must be contiguous");
    }
    if (block_table.has_value()) {
        check_same_device(*block_table, "block_table");
        TORCH_CHECK(block_table->scalar_type() == at::kInt &&
                        block_table->dim() == 2,
                    "block_table must be rank-2 INT32");
    }
    if (q_descale.has_value()) {
        check_same_device(*q_descale, "q_descale");
    }
    if (k_descale.has_value()) {
        check_same_device(*k_descale, "k_descale");
    }

    const bool page_attention = key_layout_str == "PA_BBND";
    if (query_layout_str == "TND") {
        TORCH_CHECK(actual_seq_q.has_value(),
                    "actual_seq_q is required for TND query layout");
    } else {
        TORCH_CHECK(!actual_seq_q.has_value(),
                    "actual_seq_q must be None for BSND query layout");
    }
    if (key_layout_str == "TND" || page_attention) {
        TORCH_CHECK(actual_seq_k.has_value(),
                    "actual_seq_k is required for TND/PA key layout");
    }
    TORCH_CHECK(page_attention == block_table.has_value(),
                "block_table must be provided exactly for PA_BBND layout");

    int64_t batch_size = query_layout_str == "BSND"
                             ? query.size(0)
                             : actual_seq_q->numel();
    TORCH_CHECK(pool_tail_k.numel() == batch_size,
                "pool_tail_k length must equal batch size");
    if (actual_seq_k.has_value()) {
        TORCH_CHECK(actual_seq_k->numel() == batch_size,
                    "actual_seq_k length must equal batch size");
    }
    if (block_table.has_value()) {
        TORCH_CHECK(block_table->size(0) == batch_size,
                    "block_table rows must equal batch size");
    }

    std::tuple<at::Tensor, at::Tensor> pool_key_indexer_output =
        ConstructPoolKeyIndexerOutputTensor(query, topk, pool_size,
                                            query_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(pool_key_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(pool_key_indexer_output);

    char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
    char* key_layout_ptr = const_cast<char*>(key_layout_str.c_str());

    // Non-contiguous pool_key on axis 0 (following the compressor approach):
    // the eager/aclnn tiling context does not report the runtime stride, so
    // the torch extension reads at::Tensor::stride(0) directly and passes it
    // down as the key_stride0 attribute; non-zero axes must be contiguous.
    // In PA_BBND, axis-0 stride >= blockSize*N2*D means non-contiguous
    // addressing, otherwise it is equivalent to contiguous.
    int64_t key_stride0 = -1;
    if (!pool_key.is_contiguous()) {
        auto contiguous_axes = IsContiguousAxes(pool_key);
        bool is_pa_bbnd = (key_layout_str == "PA_BBND");
        for (int64_t i = 1; i < static_cast<int64_t>(contiguous_axes.size());
             i++) {
            TORCH_CHECK(contiguous_axes[i],
                        "pool_key only supports non-contiguous tensor on the 0-axis, axis[",
                        i, "] stride is not contiguous");
        }
        key_stride0 = pool_key.stride(0);
        if (is_pa_bbnd) {
            int64_t contiguous_stride0 =
                pool_key.size(1) * pool_key.size(2) * pool_key.size(3);
            TORCH_CHECK(key_stride0 >= contiguous_stride0,
                        "pool_key stride0(", key_stride0,
                        ") must be >= contiguous stride(", contiguous_stride0,
                        ") in PA_BBND scenarios");
        }
    }

    // These ValueDepend buffers are persistent contiguous NPU tensors owned by
    // the metadata builder.  Pass them through without copies so their fixed
    // addresses and graph-replay updates are preserved.
    RunPoolKeyIndexerTensor(
        query.device(), query, pool_key, weights, pool_tail_k, actual_seq_q,
        actual_seq_k, block_table, q_descale, k_descale, topk, pool_size,
        query_layout_ptr, key_layout_ptr, mask_mode, quant_mode, return_value,
        key_stride0, sparse_indices_out, sparse_values_out);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out,
                                              sparse_values_out);
}
}  // namespace vllm_ascend

#endif  // POOL_KEY_INDEXER_TORCH_ADPT_H
