/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef KEY_POOL_LAYER_NORM_H
#define KEY_POOL_LAYER_NORM_H

namespace KeyPool {

__aicore__ inline void KeyPoolLayerNorm(const LocalTensor<float> &data, const LocalTensor<float> &in,
                                       const LocalTensor<float> &out, const LocalTensor<float> &mean,
                                       const LocalTensor<float> &gamma, const LocalTensor<float> &beta, float eps,
                                       uint32_t count)
{
    const float reciprocal = 1.0f / static_cast<float>(static_cast<int64_t>(count));

    Duplicate(mean, reciprocal, count);
    PipeBarrier<PIPE_V>();
    Mul(in, data, mean, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(in, in, in, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float meanValue = in.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(mean, meanValue, count);
    PipeBarrier<PIPE_V>();

    Sub(data, data, mean, count);
    PipeBarrier<PIPE_V>();
    Mul(out, data, data, count);
    PipeBarrier<PIPE_V>();
    Muls(out, out, reciprocal, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(out, out, out, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float variance = out.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, variance, count);
    PipeBarrier<PIPE_V>();
    Adds(out, out, eps, count);
    PipeBarrier<PIPE_V>();
    Sqrt(out, out, count);
    PipeBarrier<PIPE_V>();

    Div(data, data, out, count);
    PipeBarrier<PIPE_V>();
    Mul(data, data, gamma, count);
    PipeBarrier<PIPE_V>();
    Add(data, data, beta, count);
    PipeBarrier<PIPE_V>();
}

// Keep the legacy cache-transform helper source-compatible with the in-place
// implementation. The helper is retained on the master branch but is not used
// by the current pooling path.
__aicore__ inline void KeyPoolLayerNorm(const LocalTensor<float> &out, const LocalTensor<float> &in,
                                       const LocalTensor<float> &mean, const LocalTensor<float> &gamma,
                                       const LocalTensor<float> &beta, float eps, uint32_t count)
{
    DataCopy(out, in, count);
    PipeBarrier<PIPE_V>();
    KeyPoolLayerNorm(out, in, out, mean, gamma, beta, eps, count);
}

// Normalize complete current K rows before they enter the pooling path. The
// caller provides a scratch tensor with room for input/output and three
// vectors: mean, gamma, and beta.
__aicore__ inline void KeyPoolLayerNormRowsInplace(
    const LocalTensor<float> &data, const LocalTensor<float> &scratch,
    const GlobalTensor<float> &normWeight, const GlobalTensor<float> &normBias,
    float eps, uint32_t rowCount, uint32_t rowStride, uint32_t headDim)
{
    if (rowCount == 0 || rowStride < headDim) {
        return;
    }
    LocalTensor<float> input = scratch;
    LocalTensor<float> output = input[headDim];
    LocalTensor<float> mean = output[headDim];
    LocalTensor<float> gamma = mean[headDim];
    LocalTensor<float> beta = gamma[headDim];
    DataCopy(gamma, normWeight, headDim);
    DataCopy(beta, normBias, headDim);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t row = 0; row < rowCount; ++row) {
        KeyPoolLayerNorm(data[static_cast<uint64_t>(row) * rowStride], input, output, mean, gamma, beta, eps,
                         headDim);
    }
}

__aicore__ inline void ApplyKeyPoolRotaryPlaceholder(const LocalTensor<float> &, uint32_t)
{
    // RoPE inputs are rejected by the Host in this version. Keep the transform
    // boundary explicit so a later RoPE implementation is inserted between
    // normalization and cache writeback without changing the Pool stage.
}

template <typename INPUT_QUEUE, typename HIDDEN_STATES_T>
__aicore__ inline void WriteTransformedKeyToStateCache(
    const GlobalTensor<float> &stateCache, uint64_t stateOffset, uint32_t headDim, uint32_t coff,
    INPUT_QUEUE &inputQue, const LocalTensor<float> &key)
{
    LocalTensor<HIDDEN_STATES_T> rounded = inputQue.template AllocTensor<HIDDEN_STATES_T>();
    Cast(rounded, key, RoundMode::CAST_ROUND, coff * headDim);
    PipeBarrier<PIPE_V>();
    Cast(key, rounded, RoundMode::CAST_NONE, coff * headDim);
    PipeBarrier<PIPE_V>();
    inputQue.FreeTensor(rounded);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(stateCache[stateOffset], key, headDim);
    if (coff == 2) {
        DataCopy(stateCache[stateOffset + headDim], key[headDim], headDim);
    }
    PipeBarrier<PIPE_MTE3>();
}

template <typename COMP, typename INPUT_QUEUE, typename HIDDEN_STATES_T>
__aicore__ inline void TransformKeyCacheBeforePool(
    const GlobalTensor<float> &stateCache, const GlobalTensor<int32_t> &blockTable,
    const GlobalTensor<float> &normWeight, const GlobalTensor<float> &normBias, KeyPoolTools<COMP> &tools,
     const ConstInfo &constInfo, uint32_t coff, INPUT_QUEUE &inputQue, const LocalTensor<float> &output,
     const LocalTensor<float> &mean, const LocalTensor<float> &gamma, const LocalTensor<float> &beta)
{
    const uint32_t headDim = constInfo.headDim;
    const uint32_t hiddenDim = coff * headDim;
    const uint32_t vectorId = constInfo.aiCoreIdx * 2 + GetSubBlockIdx();
    const uint32_t vectorCount = constInfo.usedCoreNum * 2;

    DataCopy(gamma, normWeight, hiddenDim);
    DataCopy(beta, normBias, hiddenDim);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t bIdx = vectorId; bIdx < constInfo.batchSize; bIdx += vectorCount) {
        const uint32_t seqLength = tools.GetSeqLength(bIdx);
        const uint64_t startPos = tools.GetStartPos(bIdx);
        const uint64_t blockTableOffset = static_cast<uint64_t>(bIdx) * constInfo.maxBlockNumPerBatch;

        for (uint32_t sIdx = 0; sIdx < seqLength; ++sIdx) {
            const uint64_t logicalPos = startPos + sIdx;
            const uint64_t logicalBlock = logicalPos / constInfo.blockSize;
            const uint64_t blockOffset = logicalPos % constInfo.blockSize;
            const uint64_t physicalBlock = blockTable.GetValue(blockTableOffset + logicalBlock);
            const uint64_t stateOffset =
                physicalBlock * constInfo.stateCacheStrideDim0 +
                blockOffset * STATE_INTERLEAVE_FACTOR * hiddenDim;

            LocalTensor<float> input = inputQue.template AllocTensor<float>();
            DataCopy(input, stateCache[stateOffset], headDim);
            if (coff == 2) {
                DataCopy(input[headDim], stateCache[stateOffset + headDim], headDim);
            }
            inputQue.EnQue(input);
            input = inputQue.template DeQue<float>();
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            PipeBarrier<PIPE_ALL>();

            KeyPoolLayerNorm(output, input, mean, gamma, beta, constInfo.normEps, hiddenDim);

            inputQue.FreeTensor(input);
            ApplyKeyPoolRotaryPlaceholder(output, hiddenDim);
            WriteTransformedKeyToStateCache<INPUT_QUEUE, HIDDEN_STATES_T>(
                stateCache, stateOffset, headDim, coff, inputQue, output);
        }
    }
}

} // namespace KeyPool

#endif // KEY_POOL_LAYER_NORM_H
