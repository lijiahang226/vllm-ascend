#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer 独立 CPU 标杆实现(golden reference)。

标杆与算子 aclnnPoolKeyIndexer(非量化路径, quantMode=-1)的计算过程一致:

  1. 池级分数计算(逐 batch, float32):
        dot[i, n, j] = dot(query[i, n, :], pool_key[j, :]) / sqrt(headDim)  # 缩放点积
        score[i, j]  = sum_n weights[i, n] * relu(dot[i, n, j])             # 多头加权聚合
      说明: 算子 cube 侧 mm 输出经 ReLU(Fixpipe reluPre)后, vector 侧按头乘
      weights 再跨头求和(DoScale + DoReduce), 全程 float32;
      点积按算子文档公式乘 1/sqrt(headDim) 缩放(缩放为正数, 与 ReLU/加权
      求和可交换, 算子在聚合后统一乘缩放系数, 数值等价)。

  2. 因果可见池数(mask_mode=3, rightDownCausal 右对齐因果):
       L          = 池数 * pool_size + pool_tail_k[b]            # 原始 token 总数
       valid[i]   = clamp(floor((L - S1 + i + 1) / pool_size), 0, 池数)
     即 query 第 i 行只能看到前 valid[i] 个池; mask_mode=0 时所有池可见。

  3. 池级 TopK: 取 k = topk // pool_size 个池, 分数降序选取
     (标杆在同分时按下标升序取; NPU 并列分数取舍可能不同, 由集合比对吸收)。

  4. 索引展开: 选中池 p 展开为 token 索引
       p * pool_size, p * pool_size + 1, ..., p * pool_size + pool_size - 1

  5. 尾块追加(输出 topk 位置起, 容量 pool_size - 1):
       mask_mode=0: visible = pool_tail_k
       mask_mode=3: visible = max(0, min(pool_tail_k, gpos - topk + 1))
                     其中 gpos = L - S1 + i (query 行的全局 token 位置)
       第 t 个可见尾 token 的索引为 L - pool_tail_k + t

  6. 输出:
       indices: (B, S1, topk+pool_size-1) 或 (T1, topk+pool_size-1), int32, 无效 -1
       values:  (B, S1, topk//pool_size)   或 (T1, topk//pool_size),  float32, 无效 -inf

可直接运行本文件做最小自验:  python pool_key_indexer_reference.py
"""

import math

import torch

__all__ = ["pool_key_indexer_reference"]

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _as_int_list(x):
    """int tensor / list / 标量 -> list[int]"""
    if x is None:
        return None
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def _prefix_to_counts(prefix):
    """前缀和序列 -> 每 batch 数量列表"""
    p = _as_int_list(prefix)
    return [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))]


# ---------------------------------------------------------------------------
# 布局归一化: 逐 batch 拆分输入
# ---------------------------------------------------------------------------


def _split_query(query, weights, layout_q, actual_seq_q):
    """拆分 query/weights 为逐 batch 的 (Sb, N1, D) / (Sb, N1) float32 列表。

    返回 (q_list, w_list, counts, batch, s1_dim):
      - counts: 每 batch 有效 query token 数
      - s1_dim: BSND 输出的 S1 维(query.shape[1], 含无效行)
    """
    if layout_q == "BSND":
        batch, s1_dim, n1, d = query.shape
        if actual_seq_q is not None:
            counts = _as_int_list(actual_seq_q)
        else:
            counts = [s1_dim] * batch
        q_list = [query[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        w_list = [weights[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        return q_list, w_list, counts, batch, s1_dim
    if layout_q == "TND":
        t1, n1, d = query.shape
        if actual_seq_q is None:
            raise ValueError("layout_q='TND' requires actual_seq_q (prefix sums)")
        counts = _prefix_to_counts(actual_seq_q)
        q_list, w_list, start = [], [], 0
        for c in counts:
            q_list.append(query[start : start + c].to(torch.float32))
            w_list.append(weights[start : start + c].to(torch.float32))
            start += c
        return q_list, w_list, counts, len(counts), t1
    raise ValueError(f"unsupported layout_q: {layout_q}")


def _split_key(pool_key, layout_k, actual_seq_k, block_table):
    """拆分 pool_key 为逐 batch 的 (Pb, D) float32 列表。

    - BSND:    pool_key (B, S2, N2, D), actual_seq_k 可选(每 batch 池数, 非前缀和);
    - TND:     pool_key (T2, N2, D),    actual_seq_k 必传(池数前缀和);
    - PA_BBND: pool_key (blockNum, blockSize, N2, D), 块内每行一个池,
               需 block_table(B, maxBlocks) 与 actual_seq_k(每 batch 池数, 非前缀和);
               逻辑池 p 属于物理块 block_table[b][p // blockSize] 的第 p % blockSize 行,
               空槽(-1)跳过。
    """
    if layout_k == "BSND":
        batch, s2, n2, d = pool_key.shape
        if actual_seq_k is not None:
            counts = _as_int_list(actual_seq_k)
        else:
            counts = [s2] * batch
        return [pool_key[b, :c, 0, :].to(torch.float32) for b, c in enumerate(counts)]
    if layout_k == "TND":
        t2, n2, d = pool_key.shape
        if actual_seq_k is None:
            raise ValueError("layout_k='TND' requires actual_seq_k (prefix sums)")
        counts = _prefix_to_counts(actual_seq_k)
        keys, start = [], 0
        for c in counts:
            keys.append(pool_key[start : start + c, 0, :].to(torch.float32))
            start += c
        return keys
    if layout_k == "PA_BBND":
        if block_table is None:
            raise ValueError("layout_k='PA_BBND' requires block_table")
        if actual_seq_k is None:
            raise ValueError(
                "layout_k='PA_BBND' requires actual_seq_k (pools per batch)"
            )
        block_num, block_size, n2, d = pool_key.shape
        bt = block_table.detach().cpu()
        batch, max_blocks = bt.shape
        counts = _as_int_list(actual_seq_k)
        keys = []
        for b in range(batch):
            need, rows, pos = counts[b], [], 0
            for t in range(max_blocks):
                if pos >= need:
                    break
                blk = int(bt[b, t])
                if blk < 0:
                    continue
                take = min(block_size, need - pos)
                rows.append(pool_key[blk, :take, 0, :].to(torch.float32))
                pos += take
            if rows:
                keys.append(torch.cat(rows, dim=0))
            else:
                keys.append(pool_key.new_zeros((0, d)).to(torch.float32))
        return keys
    raise ValueError(f"unsupported layout_k: {layout_k}")


# ---------------------------------------------------------------------------
# 逐 batch 核心计算
# ---------------------------------------------------------------------------


def _compute_batch(q_b, w_b, k_b, tail_k, topk, pool_size, mask_mode):
    """单个 batch 的池级 TopK 选择与索引展开。

    入参:
      q_b: (S, N1, D) float32;  w_b: (S, N1) float32;  k_b: (P, D) float32;
      tail_k: 尾部有效 token 数; topk/pool_size/mask_mode: 算子属性。

    返回:
      token_idx: (S, topk + pool_size - 1) int32, 无效 -1
      pool_val:  (S, topk // pool_size) float32,   无效 -inf
    """
    s_len, n1, d = q_b.shape
    p_len = k_b.shape[0]
    sparse_count = topk // pool_size
    out_len = topk + pool_size - 1
    l_orig = p_len * pool_size + tail_k  # 原始 token 总数

    token_idx = torch.full((s_len, out_len), -1, dtype=torch.int32)
    pool_val = torch.full((s_len, sparse_count), _NEG_INF, dtype=torch.float32)
    if s_len == 0 or p_len == 0:
        return token_idx, pool_val

    # 步骤 1: 池级分数 score[i, j] = sum_n w[i, n] * relu(q[i, n] . k[j] / sqrt(d))
    # (算子文档: S = Q @ K_pool^T * 1/sqrt(headDim) -> ReLU -> 多头加权)
    scale = 1.0 / math.sqrt(d)
    dot = torch.einsum("snd,pd->snp", q_b, k_b) * scale  # (S, N1, P)
    scores = torch.einsum("sn,snp->sp", w_b, dot.clamp_min(0.0))  # (S, P)

    arange_pool = torch.arange(pool_size, dtype=torch.int64)
    for i in range(s_len):
        # 步骤 2: 因果可见池数
        if mask_mode == 3:
            valid = (l_orig - s_len + i + 1) // pool_size  # floor 除法
            valid = max(0, min(valid, p_len))
        else:
            valid = p_len
        select = min(valid, sparse_count)
        if select > 0:
            # 步骤 3: 池级 TopK(分数降序, 同分按下标升序)
            order = torch.argsort(scores[i, :valid], descending=True, stable=True)
            picked = order[:select]
            # 步骤 4: 展开为 token 索引(依次填入, 总长 select*pool_size <= topk)
            tokens = picked.to(torch.int64).unsqueeze(1) * pool_size + arange_pool
            flat = tokens.reshape(-1)
            token_idx[i, : flat.numel()] = flat.to(torch.int32)
            pool_val[i, :select] = scores[i, picked]

        # 步骤 5: 尾块追加(输出 topk 位置起, 容量 pool_size - 1)
        if pool_size > 1 and tail_k > 0:
            if mask_mode == 0:
                visible = tail_k
            else:
                gpos = l_orig - s_len + i
                visible = max(0, min(tail_k, gpos - topk + 1))
            if visible > 0:
                tail_tokens = torch.arange(
                    l_orig - tail_k, l_orig - tail_k + visible, dtype=torch.int32
                )
                token_idx[i, topk : topk + visible] = tail_tokens

    return token_idx, pool_val


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def pool_key_indexer_reference(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=16,
    mask_mode=0,
    return_value=False,
):
    """PoolKeyIndexer CPU 标杆实现(与算子非量化路径计算过程一致)。

    入参说明:
      query: Query 张量, dtype float16/bfloat16/float32(内部统一转 float32 计算)。
        - layout_q="BSND": shape (B, S1, N1, D);
        - layout_q="TND":  shape (T1, N1, D), T1 为所有 batch 的 query token 总和。
      pool_key: 池级 Key 张量(每行一个 pool 的 key), dtype 与 query 一致:
        - layout_k="BSND":    shape (B, S2, N2, D), S2 为每 batch 池数;
        - layout_k="TND":     shape (T2, N2, D), T2 为所有 batch 池数总和;
        - layout_k="PA_BBND": shape (blockNum, blockSize, N2, D), 块内每行一个池,
          需同时传入 block_table 与 actual_seq_k; 支持 0 轴非连续(strided view)。
      weights: 多头权重, shape (B, S1, N1)(BSND) 或 (T1, N1)(TND)。
      pool_tail_k: 每 batch 尾部不完整 pool 的有效 token 数, shape (B,), int64,
        取值范围 [0, pool_size-1]。
      actual_seq_q: 可选, 每 batch query 有效 token 数:
        - layout_q="TND":  必传, 前缀和形式(长度 B, 末项等于 T1);
        - layout_q="BSND": 可传, 每 batch 有效数(非前缀和); None 表示全部有效。
      actual_seq_k: 可选, 每 batch 有效池数:
        - layout_k="TND":     必传, 前缀和形式(末项等于 T2);
        - layout_k="PA_BBND": 必传, 每 batch 池数(非前缀和);
        - layout_k="BSND":    可传, 每 batch 池数(非前缀和); None 表示全部有效。
      block_table: layout_k="PA_BBND" 时必传, shape (B, maxBlockNumPerSeq), int32,
        元素为物理块编号, 无效槽位 -1。
      layout_q: query 布局, "BSND" 或 "TND"。
      layout_k: pool_key 布局, "BSND"、"TND" 或 "PA_BBND"。
      topk: 展开后保留的 token 数, 需满足 topk % pool_size == 0。
      pool_size: 每个 pool 包含的 token 数, 取值 [1, 128]。
      mask_mode: 0 = defaultMask(无掩码); 3 = rightDownCausal(右对齐因果)。
      return_value: True 时输出 values; False 时 values 为空 tensor。

    返回值:
      (indices, values):
        indices: int32; BSND 时 (B, S1, topk+pool_size-1), TND 时
          (T1, topk+pool_size-1); 无效位置填 -1。
        values: float32; return_value=True 时 BSND 为 (B, S1, topk//pool_size),
          TND 为 (T1, topk//pool_size), 无效位置填 -inf; return_value=False 时
          为空 tensor (0,)。
    """
    # ---- 参数校验 ----
    if layout_q not in ("BSND", "TND"):
        raise ValueError(f"unsupported layout_q: {layout_q}")
    if layout_k not in ("BSND", "TND", "PA_BBND"):
        raise ValueError(f"unsupported layout_k: {layout_k}")
    if mask_mode not in (0, 3):
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if topk % pool_size != 0:
        raise ValueError(f"topk({topk}) must be divisible by pool_size({pool_size})")
    if pool_key.shape[-2] != 1:
        raise ValueError(f"pool_key N2 must be 1, got {pool_key.shape[-2]}")
    if query.shape[-1] != pool_key.shape[-1]:
        raise ValueError(
            f"headDim mismatch: query {query.shape[-1]} vs pool_key {pool_key.shape[-1]}"
        )

    # ---- 布局归一化: 逐 batch 拆分 ----
    q_list, w_list, q_counts, batch, s1_dim = _split_query(
        query, weights, layout_q, actual_seq_q
    )
    k_list = _split_key(pool_key, layout_k, actual_seq_k, block_table)
    if len(k_list) != batch:
        raise ValueError(f"batch mismatch: query {batch} vs pool_key {len(k_list)}")
    tail_list = _as_int_list(pool_tail_k) if pool_tail_k is not None else [0] * batch
    if len(tail_list) != batch:
        raise ValueError(f"pool_tail_k length {len(tail_list)} != batch {batch}")

    # ---- 逐 batch 计算 ----
    idx_rows, val_rows = [], []
    for b in range(batch):
        token_idx, pool_val = _compute_batch(
            q_list[b], w_list[b], k_list[b], tail_list[b], topk, pool_size, mask_mode
        )
        idx_rows.append(token_idx)
        val_rows.append(pool_val)

    # ---- 组装输出 ----
    out_len = topk + pool_size - 1
    sparse_count = topk // pool_size
    if layout_q == "BSND":
        indices = torch.full((batch, s1_dim, out_len), -1, dtype=torch.int32)
        values = torch.full(
            (batch, s1_dim, sparse_count), _NEG_INF, dtype=torch.float32
        )
        for b in range(batch):
            c = q_counts[b]
            indices[b, :c] = idx_rows[b]
            values[b, :c] = val_rows[b]
    else:  # TND: 所有行有效, 按序拼接
        indices = (
            torch.cat(idx_rows, dim=0)
            if idx_rows
            else torch.zeros((0, out_len), dtype=torch.int32)
        )
        values = (
            torch.cat(val_rows, dim=0)
            if val_rows
            else torch.zeros((0, sparse_count), dtype=torch.float32)
        )

    if not return_value:
        values = torch.empty(0, dtype=torch.float32)
    return indices, values


# ---------------------------------------------------------------------------
# 独立运行入口: 最小自验
# ---------------------------------------------------------------------------


def _pack_pa_demo_key(pool_key_bsnd, k_pool_lens, block_size=16):
    """将逻辑 (sum_pools, N2, D) 池数据打包为 PA 物理块布局。

    与算子 LIV2 语义一致: 块内每行 = 1 个池(key 不做 token 展开),
    每 batch 的池独立连续填块, 物理块随机排布, 尾块不足补零。
    返回 (key_pa, block_table):
      - key_pa: (blockNum, block_size, N2, D) float16
      - block_table: (B, maxBlocks) int32, 空槽 -1
    """
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    n2, d = pool_key_bsnd.shape[1], pool_key_bsnd.shape[2]
    blocks = torch.zeros(block_num, block_size, n2, d, dtype=pool_key_bsnd.dtype)
    pool_start = blk_start = 0
    for pools, nb in zip(k_pool_lens, blocks_per_batch):
        seg = pool_key_bsnd[pool_start : pool_start + pools]
        pad = nb * block_size - pools
        if pad:  # 尾块不足补零
            seg = torch.cat([seg, torch.zeros(pad, n2, d, dtype=seg.dtype)], dim=0)
        blocks[blk_start : blk_start + nb] = seg.view(nb, block_size, n2, d)
        pool_start += pools
        blk_start += nb
    perm = torch.randperm(block_num)
    # 物理块重排后按新块号重建 block_table
    bt = torch.full((len(k_pool_lens), max(blocks_per_batch)), -1, dtype=torch.int32)
    cur = 0
    for b, nb in enumerate(blocks_per_batch):
        for i in range(nb):
            bt[b, i] = perm[cur]
            cur += 1
    return blocks[perm].contiguous(), bt


# ---------------------------------------------------------------------------
# demo 公共助手: 输入构造 / 期望 shape / 输出自检
# ---------------------------------------------------------------------------


def _make_demo_query_weights(layout_q, b, s1, n1, d):
    """按 query 布局构造随机 (query, weights)。"""
    if layout_q == "BSND":
        return (
            torch.randn(b, s1, n1, d, dtype=torch.float16),
            torch.randn(b, s1, n1, dtype=torch.float16),
        )
    return (
        torch.randn(b * s1, n1, d, dtype=torch.float16),
        torch.randn(b * s1, n1, dtype=torch.float16),
    )


def _make_dense_demo_inputs(layout_q, b, s1, n1, d, pools, n2=1):
    """构造 BSND/TND 密集布局随机输入。

    返回 (query, pool_key, weights, actual_seq_q, actual_seq_k);
    BSND 时两个 actual_seq 为 None, TND 时为前缀和形式。
    """
    query, weights = _make_demo_query_weights(layout_q, b, s1, n1, d)
    if layout_q == "BSND":
        pool_key = torch.randn(b, pools, n2, d, dtype=torch.float16)
        return query, pool_key, weights, None, None
    pool_key = torch.randn(b * pools, n2, d, dtype=torch.float16)
    asq = torch.tensor([s1] * b, dtype=torch.int64).cumsum(0)
    ask = torch.tensor([pools] * b, dtype=torch.int64).cumsum(0)
    return query, pool_key, weights, asq, ask


def _demo_expected_shapes(layout_q, b, s1, topk, pool_size):
    """期望输出 shape: BSND 带 (B, S1) 维, TND 为扁平 T1 行。"""
    out_len = topk + pool_size - 1
    sparse_count = topk // pool_size
    if layout_q == "BSND":
        return (b, s1, out_len), (b, s1, sparse_count)
    return (b * s1, out_len), (b * s1, sparse_count)


def _check_demo_output(indices, values, exp_idx_shape, exp_val_shape, max_token, tag):
    """公共自检: 输出 shape / 有效索引范围 / values 有限性, 并打印 OK 日志。"""
    assert indices.shape == exp_idx_shape, (indices.shape, exp_idx_shape)
    assert values.shape == exp_val_shape, (values.shape, exp_val_shape)
    valid = indices[indices >= 0]
    assert valid.numel() > 0
    assert int(valid.min()) >= 0
    assert int(valid.max()) < max_token
    assert torch.isfinite(values[torch.isfinite(values)]).all()
    print(
        f"[demo] {tag} indices{tuple(indices.shape)} valid={valid.numel()} "
        f"values{tuple(values.shape)} -> OK"
    )


# ---------------------------------------------------------------------------
# demo 场景: 密集布局 / PA 布局
# ---------------------------------------------------------------------------


def _demo_dense():
    """BSND/TND 密集布局自验: mask_mode 0/3, 验证输出 shape/取值范围。"""
    torch.manual_seed(0)
    b, s1, n1, d = 2, 8, 4, 128
    topk, pool_size, pools = 32, 8, 12
    tail = torch.tensor([3, 5], dtype=torch.int64)
    max_token = pools * pool_size + int(max(tail))

    for layout_q, layout_k in [("BSND", "BSND"), ("TND", "TND")]:
        query, pool_key, weights, asq, ask = _make_dense_demo_inputs(
            layout_q, b, s1, n1, d, pools
        )
        exp_idx, exp_val = _demo_expected_shapes(layout_q, b, s1, topk, pool_size)
        for mask_mode in (0, 3):
            indices, values = pool_key_indexer_reference(
                query,
                pool_key,
                weights,
                tail,
                actual_seq_q=asq,
                actual_seq_k=ask,
                layout_q=layout_q,
                layout_k=layout_k,
                topk=topk,
                pool_size=pool_size,
                mask_mode=mask_mode,
                return_value=True,
            )
            _check_demo_output(
                indices,
                values,
                exp_idx,
                exp_val,
                max_token,
                f"layout=({layout_q},{layout_k}) mask={mask_mode}",
            )


def _demo_pa():
    """PA_BBND 场景自验: 不等池数(-1 空槽+尾块 padding)、block_size 16/32、
    mask_mode 0/3、BSND/TND 两种 query 布局, 并用同一份逻辑池数据
    交叉验证 PA 输出 == BSND 输出。"""
    torch.manual_seed(7)
    b, s1, n1, d, n2 = 2, 8, 4, 128, 1
    topk, pool_size = 32, 8
    k_pool_lens = [12, 5]  # 每 batch 池数不等 -> -1 空槽与尾块 padding
    total_pools = sum(k_pool_lens)
    tail = torch.tensor([3, 5], dtype=torch.int64)
    max_token = total_pools * pool_size + int(max(tail))

    # 逻辑池数据 (与 BSND 交叉验证共用): batch0 前 12 池, batch1 后 5 池
    logical_key = torch.randn(total_pools, n2, d, dtype=torch.float16)
    ask = torch.tensor(k_pool_lens, dtype=torch.int64)
    asq_t = torch.tensor([s1] * b, dtype=torch.int64).cumsum(0)
    # 交叉验证用 BSND key (B, S2max, N2, D): 各 batch 前 pools 行为有效池
    s2max = max(k_pool_lens)
    key_bsnd = torch.zeros(b, s2max, n2, d, dtype=logical_key.dtype)
    start = 0
    for bi, pools in enumerate(k_pool_lens):
        key_bsnd[bi, :pools] = logical_key[start : start + pools]
        start += pools

    for block_size in (16, 32):
        key_pa, bt = _pack_pa_demo_key(logical_key, k_pool_lens, block_size)
        for layout_q in ("BSND", "TND"):
            query, weights = _make_demo_query_weights(layout_q, b, s1, n1, d)
            asq = asq_t if layout_q == "TND" else None
            exp_idx, exp_val = _demo_expected_shapes(layout_q, b, s1, topk, pool_size)
            for mask_mode in (0, 3):
                indices, values = pool_key_indexer_reference(
                    query,
                    key_pa,
                    weights,
                    tail,
                    actual_seq_q=asq,
                    actual_seq_k=ask,
                    block_table=bt,
                    layout_q=layout_q,
                    layout_k="PA_BBND",
                    topk=topk,
                    pool_size=pool_size,
                    mask_mode=mask_mode,
                    return_value=True,
                )
                _check_demo_output(
                    indices,
                    values,
                    exp_idx,
                    exp_val,
                    max_token,
                    f"block_size={block_size} mask={mask_mode} ({layout_q},PA_BBND)",
                )
                if layout_q == "BSND":
                    # batch1(仅5池)选中池数被截断到 5
                    per_row_cnt = (indices[1] >= 0).sum(dim=-1)
                    assert int(
                        per_row_cnt.max()
                    ) <= topk // pool_size * pool_size + min(int(tail[1]), topk), (
                        per_row_cnt.max()
                    )
                    # 交叉验证: PA 输出 == BSND 输出(同一份逻辑池数据)
                    idx_b, val_b = pool_key_indexer_reference(
                        query,
                        key_bsnd,
                        weights,
                        tail,
                        actual_seq_k=ask,
                        layout_q="BSND",
                        layout_k="BSND",
                        topk=topk,
                        pool_size=pool_size,
                        mask_mode=mask_mode,
                        return_value=True,
                    )
                    assert torch.equal(indices, idx_b), "PA indices != BSND indices"
                    assert torch.equal(values, val_b), "PA values != BSND values"


def _demo():
    """构造随机输入自验标杆, 验证输出 shape / 取值范围 / 布局一致性。"""
    _demo_dense()
    print("pool_key_indexer_reference dense demo passed.")
    _demo_pa()
    print("pool_key_indexer_reference PA demo passed.")


if __name__ == "__main__":
    _demo()
