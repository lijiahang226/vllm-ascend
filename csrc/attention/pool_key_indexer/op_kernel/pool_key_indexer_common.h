/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_COMMON_H
#define POOL_KEY_INDEXER_COMMON_H

using namespace AscendC;
namespace PkiCommon {

enum class PkiLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

template <typename Q_T, typename K_T, typename OUT_T,
          PkiLayout LAYOUT_T = PkiLayout::BSND,
          PkiLayout K_LAYOUT_T = PkiLayout::BSND,
          bool DT_W_FLAG = false>
struct PkiType {
    static constexpr bool weightsTypeFlag = DT_W_FLAG;
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = (K_LAYOUT_T == PkiLayout::PA_BSND);
    static constexpr PkiLayout layout = LAYOUT_T;
    static constexpr PkiLayout keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1; // L_orig (压缩前 token 总数)
    int32_t poolTailK = 0;      // 当前 batch 的尾部 token 数
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
};

struct ConstInfo {
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    static constexpr uint32_t PKI_SYNC_MODE4 = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    static constexpr int INVALID_IDX = -1;
    uint32_t INVALID_VAL = 0xFF800000;

    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    uint32_t mBaseSize = 1ULL;
    uint32_t mBaseSizeAlign = 1ULL;
    uint32_t mBaseSizeMax = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;
    uint32_t trunkLen = 8192;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum = 1ULL;
    uint64_t headDim = 128ULL;
    float qkScale = 1.0f; // 1/sqrt(headDim), 池级分数缩放系数
    uint64_t sparseCount = 0ULL;
    uint64_t topk = 0ULL;
    uint64_t poolSize = 1ULL;
    uint64_t kSeqSize = 0ULL;
    uint64_t qSeqSize = 1ULL;

    uint32_t kCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t keyStride0 = 0U;
    uint32_t keyDequantScaleStride0 = 0U;

    PkiLayout outputLayout = PkiLayout::BSND;
    bool attenMaskFlag = false;
    uint32_t maskMode = 0;
    int32_t quantMode = -1;
    bool returnValue = false;
    bool isSparseCountOver2K = false;
    bool isLDOpen = false;
    bool splitMFlag = false;

    uint32_t actualLenQDims = 0U;
    uint32_t actualLenDims = 0U;
    bool isAccumSeqS1 = false;
    bool isAccumSeqS2 = false;

    // workspace offsets
    uint64_t wsOffScore = 0;
    uint64_t wsOffLdScore = 0;
    uint64_t wsOffLdIdx = 0;
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;
    bool isCoreEnable = false;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace PkiCommon

#define UB_BLOCK 32
#define UB_BANK_GROUPS 8
#define UB_BANKS 2
#define UB_BANK_DEPTH 512
#define UB_BANK_GROUP_STRIDE UB_BLOCK
#define UB_BANK_STRIDE (UB_BANK_GROUPS * UB_BLOCK)
#define UB_BANK_DEPTH_STRIDE (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK)

#endif // POOL_KEY_INDEXER_COMMON_H
