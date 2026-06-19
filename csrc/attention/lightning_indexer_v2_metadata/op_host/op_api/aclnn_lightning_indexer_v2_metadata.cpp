/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
  *\file aclnn_lightning_indexer_v2_metadata.cpp
  *\brief
  */

#include "aclnn_lightning_indexer_v2_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "l0_lightning_indexer_v2_metadata.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

int64_t LI_NO_MASK_MODE = 0;
int64_t LI_CAUSAL_MASK_MODE = 3;
int64_t LI_CMP_RATIO_LOWER_BOUND = 1;
int64_t LI_CMP_RATIO_UPPER_BOUND = 128;

inline bool IsTensorExistLiV2(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumLiV2(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeLiV2(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

int64_t GetQueryBatchSizeLiV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
    const aclTensor *sequsedQOptional, const char *layoutQOptional)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    if (IsTensorExistLiV2(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedQOptional 没传，使用cuSeqlensQOptional获取BatchSize
    if (IsTensorExistLiV2(cuSeqlensQOptional)) {
        return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
    }
    // 如果是 TND，必须使用 cuSeqlensQOptional获取BatchSize
    if (strcmp(layoutQOptional, "TND") == 0) {
        return 0;
    }
    // 3. 如果不是 TND，或者 cuSeqlensQOptional 为空，使用batchSize
    return batchSize;
}

int64_t GetKeyBatchSizeLiV2(int64_t batchSize, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedKOptional, const char *layoutKOptional)
{
    // 1. 如果sequsedKOptional 传了，使用sequsedKOptional获取BatchSize
    if (IsTensorExistLiV2(sequsedKOptional)) {
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedKOptional 没传，使用cuSeqlensKOptional获取BatchSize
    if (IsTensorExistLiV2(cuSeqlensKOptional)) {
        return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
    }
    // 如果是 TND，必须使用 cuSeqlensKOptional获取BatchSize
    if (strcmp(layoutKOptional, "TND") == 0) {
        return 0;
    }
    // 3. 如果不是 TND，或者 cuSeqlensKOptional 为空，使用batchSize
    return batchSize;
}

aclnnStatus CheckSingleParamLiV2(int64_t numHeadsQ, int64_t numHeadsK, int64_t topk,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
    const char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
    uint32_t aicCoreNum, uint32_t aivCoreNum, const std::string &socVersion)
{
    // num_heads_q 校验
    CHECK_COND(numHeadsQ >= 0, ACLNN_ERR_PARAM_INVALID,
        "num_heads_q should not be negative, but got %d", numHeadsQ);
    // num_heads_k 校验
    CHECK_COND(numHeadsK >= 0, ACLNN_ERR_PARAM_INVALID,
        "num_heads_kv should not be negative, but got %d", numHeadsK);
    // topk 校验
    CHECK_COND(topk >= 0, ACLNN_ERR_PARAM_INVALID,
        "topk should not be negative, but got %d", topk);
    // batch_size 非负校验
    CHECK_COND(batchSize >= 0, ACLNN_ERR_PARAM_INVALID,
        "batch_size should not be negative, but got %d", batchSize);
    // max_seqlen_q 校验
    CHECK_COND(maxSeqlenQ >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_q should not be negative, but got %d", maxSeqlenQ);
    // max_seqlen_k 校验
    CHECK_COND(maxSeqlenK >= 0, ACLNN_ERR_PARAM_INVALID,
        "max_seqlen_k should not be negative, but got %d", maxSeqlenK);
    // mask_mode 校验
    CHECK_COND((maskMode == LI_NO_MASK_MODE) || (maskMode == LI_CAUSAL_MASK_MODE),
        ACLNN_ERR_PARAM_INVALID,
        "mask_mode should be %d/%d, but got %d", LI_NO_MASK_MODE, LI_CAUSAL_MASK_MODE, maskMode);
    // cmp_ratio 校验
    CHECK_COND((cmpRatio >= LI_CMP_RATIO_LOWER_BOUND) && (cmpRatio <= LI_CMP_RATIO_UPPER_BOUND),
        ACLNN_ERR_PARAM_INVALID,
        "cmp_ratio should be between [%d, %d], but got %d",
        LI_CMP_RATIO_LOWER_BOUND, LI_CMP_RATIO_UPPER_BOUND, cmpRatio);
    // layout_q 校验
    CHECK_COND((strcmp(layoutQOptional, "TND") == 0) || (strcmp(layoutQOptional, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
        "layout_q must be TND or BSND, but got %s", layoutQOptional);
    // layout_k 校验
    CHECK_COND((strcmp(layoutKOptional, "TND") == 0) || (strcmp(layoutKOptional, "BSND") == 0) || \
        (strcmp(layoutKOptional, "PA_BBND") == 0),
        ACLNN_ERR_PARAM_INVALID, "layout_k must be TND/BSND/PA_BBND, but got %s", layoutKOptional);
    if ((strcmp(layoutKOptional, "PA_BBND") != 0)) {
        CHECK_COND((strcmp(layoutQOptional, layoutKOptional) == 0), ACLNN_ERR_PARAM_INVALID,
            "For layout_k != PA_BBND, layout_q and layout_k must be the same!");
    }
    // 核心数校验
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID,
        "AIC num should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_PARAM_INVALID,
        "AIV num should be larger than 0, but got %u", aivCoreNum);
    CHECK_COND(socVersion.find("Ascend950") != std::string::npos, ACLNN_ERR_PARAM_INVALID,
        "This operator supports [Ascend950], but now is on %s", socVersion.c_str());
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceLiV2(int64_t maskMode, int64_t cmpRatio,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *cmpResidualKOptional,
    const char *layoutQOptional, const char *layoutKOptional)
{
    // Query 存在性逻辑
    if (strcmp(layoutQOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistLiV2(cuSeqlensQOptional), ACLNN_ERR_PARAM_INVALID,
            "For layout_q TND, cu_seqlens_q must be provided!");
    }
    // Key 存在性逻辑
    if (strcmp(layoutKOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistLiV2(cuSeqlensKOptional), ACLNN_ERR_PARAM_INVALID,
            "For layout_k TND, cu_seqlens_k must be provided!");
    }
    // cmp_residual_k 存在性逻辑
    if (cmpRatio != LI_CMP_RATIO_LOWER_BOUND && maskMode == LI_CAUSAL_MASK_MODE) {
        CHECK_COND(IsTensorExistLiV2(cmpResidualKOptional), ACLNN_ERR_PARAM_INVALID,
            "When cmp_ratio is not 1 and mask_mode is 3, cmp_residual_k must be provided!");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyLiV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, const char *layoutQOptional, const char *layoutKOptional)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    int64_t batch = GetQueryBatchSizeLiV2(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional);
    // 校验 cu_seqlens_q
    if (IsTensorExistLiV2(cuSeqlensQOptional)) {
        dimNum = GetDimNumLiV2(cuSeqlensQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %ld", dimNum);
        dataType = GetDataTypeLiV2(cuSeqlensQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistLiV2(cuSeqlensKOptional)) {
        dimNum = GetDimNumLiV2(cuSeqlensKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_k must be 1, but got %ld", dimNum);
        dataType = GetDataTypeLiV2(cuSeqlensKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cu_seqlens_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_q
    if (IsTensorExistLiV2(sequsedQOptional)) {
        dimNum = GetDimNumLiV2(sequsedQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %ld", dimNum);
        dataType = GetDataTypeLiV2(sequsedQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 seqused_k
    if (IsTensorExistLiV2(sequsedKOptional)) {
        dimNum = GetDimNumLiV2(sequsedKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_k must be 1, but got %ld", dimNum);
        dataType = GetDataTypeLiV2(sequsedKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of seqused_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    // 校验 cmp_residual_k
    if (IsTensorExistLiV2(cmpResidualKOptional)) {
        dimNum = GetDimNumLiV2(cmpResidualKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID,
            "The dim num of cmp_residual_k must be 1, but got %ld", dimNum);
        dataType = GetDataTypeLiV2(cmpResidualKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
            "The data type of cmp_residual_k must be int32, but got %d", static_cast<int32_t>(dataType));
        // 校验batch
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        CHECK_COND(cmpResidualKBatch == batchSize, ACLNN_ERR_PARAM_INVALID,
            "The batch_size of cmp_residual_k should match the one of q tensor, but got %ld and %ld",
            cmpResidualKBatch, batchSize);
    }

    // 校验batch
    int64_t keyBatch = GetKeyBatchSizeLiV2(batchSize, cuSeqlensKOptional, sequsedKOptional,
        layoutKOptional);
    CHECK_COND(batch == keyBatch, ACLNN_ERR_PARAM_INVALID,
        "The batch_size obtained from q Tensor should be the same as that obtained from k tensor, but got %ld and %ld",
        batch, keyBatch);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFeatureLiV2()
{
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckLiV2(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t topk,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
    const char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
    uint32_t aicCoreNum, uint32_t aivCoreNum, const std::string &socVersion)
{
    auto ret = CheckSingleParamLiV2(numHeadsQ, numHeadsK, topk,
        batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, maskMode,
        cmpRatio,
        aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceLiV2(maskMode, cmpRatio,
        cuSeqlensQOptional, cuSeqlensKOptional, cmpResidualKOptional, layoutQOptional, layoutKOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyLiV2(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
        sequsedKOptional, cmpResidualKOptional, layoutQOptional, layoutKOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckFeatureLiV2();
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

}

aclnnStatus aclnnLightningIndexerV2MetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, char *layoutQOptional,
    char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
    const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnLightningIndexerV2Metadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                          cmpResidualKOptional,
                          numHeadsQ, numHeadsK, headDim, topk,
                          batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional,
                          maskMode, cmpRatio),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const std::string socVersion = npuInfo.GetSocLongVersion();

    auto ret = ParamsCheckLiV2(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                               cmpResidualKOptional,
                               numHeadsQ, numHeadsK, topk, batchSize, maxSeqlenQ, maxSeqlenK,
                               layoutQOptional, layoutKOptional, maskMode, cmpRatio,
                               aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *cuSeqlensQOptionalContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQOptionalContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqlensQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cuSeqlensKOptionalContiguous = nullptr;
    if (cuSeqlensKOptional != nullptr) {
        cuSeqlensKOptionalContiguous = l0op::Contiguous(cuSeqlensKOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqlensKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedQOptionalContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        CHECK_RET(sequsedQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedKOptionalContiguous = nullptr;
    if (sequsedKOptional != nullptr) {
        sequsedKOptionalContiguous = l0op::Contiguous(sequsedKOptional, uniqueExecutor.get());
        CHECK_RET(sequsedKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cmpResidualLKOptionalContiguous = nullptr;
    if (cmpResidualKOptional != nullptr) {
        cmpResidualLKOptionalContiguous = l0op::Contiguous(cmpResidualKOptional, uniqueExecutor.get());
        CHECK_RET(cmpResidualLKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto output = l0op::LightningIndexerV2Metadata(
        cuSeqlensQOptionalContiguous, cuSeqlensKOptionalContiguous, sequsedQOptionalContiguous,
        sequsedKOptionalContiguous, cmpResidualLKOptionalContiguous,
        numHeadsQ, numHeadsK, headDim, topk,
        batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, maskMode, cmpRatio,
        aicCoreNum, aivCoreNum, socVersion.c_str(), metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLightningIndexerV2Metadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLightningIndexerV2Metadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
