# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Lightning Indexer Prolog Quant registration module.

This module registers the fused lightning_indexer_prolog_quant_mxfp8 operator
via torch.library when pypto is available. The operator fuses query/key/weights
computation and KV cache scatter into a single optimized kernel.
"""

import logging

import torch

try:
    from vllm_ascend.attention.lightning_indexer_prolog_quant_mxfp8_impl import (
        lightning_indexer_prolog_quant,
    )

    HAS_PYPTO = True
except Exception:
    HAS_PYPTO = False
    logging.debug("pypto not available, lightning_indexer_prolog_quant disabled")

if HAS_PYPTO:
    try:
        from torch._dynamo import allow_in_graph
    except Exception:

        def allow_in_graph(fn):
            return fn

    _pyptolib = torch.library.Library("pypto", "FRAGMENT")
    _pyptolib.define(
        "lightning_indexer_prolog_quant_mxfp8(Tensor x, Tensor q_norm, Tensor q_norm_scale, "
        "Tensor w_qb, Tensor w_qb_scale, Tensor wk, Tensor w_proj, Tensor gamma_k, "
        "Tensor cos_idx_rope, Tensor sin_idx_rope, Tensor hadamard_q, Tensor hadamard_k, Tensor k_cache, "
        "Tensor k_scale_cache, Tensor k_cache_index, Tensor k_scale_cache_index) -> "
        "(Tensor q_fp8e4m3, Tensor q_scale, Tensor k_fp8e4m3, Tensor k_scale, Tensor weights)"
    )

    @torch.library.impl(_pyptolib, "lightning_indexer_prolog_quant_mxfp8", "Meta")
    def _lightning_indexer_prolog_quant_mxfp8_meta(
        x,
        q_norm,
        q_norm_scale,
        w_qb,
        w_qb_scale,
        wk,
        w_proj,
        gamma_k,
        cos_idx_rope,
        sin_idx_rope,
        hadamard_q,
        hadamard_k,
        k_cache,
        k_scale_cache,
        k_cache_index,
        k_scale_cache_index,
    ):
        t = x.shape[0]
        head_num = w_proj.shape[1]
        block_num, block_size, n_kv, head_dim = k_cache.shape
        q_fp8e4m3 = torch.empty((t, head_num, head_dim), device="meta", dtype=torch.float8_e4m3fn)
        q_scale = torch.empty((t, head_num, 1), device="meta", dtype=torch.float32)
        k_fp8e4m3 = k_cache
        k_scale = k_scale_cache
        weights = torch.empty((t, head_num), device="meta", dtype=torch.bfloat16)
        return q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights

    def _lightning_indexer_prolog_quant_mxfp8_pypto(
        x,
        q_norm,
        q_norm_scale,
        w_qb,
        w_qb_scale,
        wk,
        w_proj,
        gamma_k,
        cos_idx_rope,
        sin_idx_rope,
        hadamard_q,
        hadamard_k,
        k_cache,
        k_scale_cache,
        k_cache_index,
        k_scale_cache_index,
    ):
        t = x.shape[0]
        head_num = w_proj.shape[1]
        block_num, block_size, n_kv, head_dim = k_cache.shape

        k_storage_offset = k_cache.storage_offset()
        k_scale_storage_offset = k_scale_cache.storage_offset()

        if not k_cache.is_contiguous():
            page_size = k_cache.stride()[0] // (n_kv * block_size)
            pg_cache = torch.as_strided(
                k_cache,
                size=(block_num, block_size, n_kv, page_size),
                stride=(block_size * n_kv * page_size, n_kv * page_size, page_size, 1),
                storage_offset=0,
            )
            k_cache = pg_cache.view(torch.float8_e4m3fn)
            k_scale_cache = pg_cache.view(torch.float32)
            pg_cache_index = (
                k_cache_index // block_size * (k_cache.shape[-1] // head_dim) * block_size
                + k_cache_index % block_size
                + k_storage_offset // head_dim
            )
            pg_scale_cache_index = (
                k_scale_cache_index // block_size * k_scale_cache.shape[-1] * block_size
                + k_scale_cache_index % block_size
                + k_scale_storage_offset
            )

        k_cache = k_cache.view(block_num, block_size * (k_cache.shape[-1] // head_dim), n_kv, head_dim)
        k_scale_cache = k_scale_cache.view(block_num, block_size * k_scale_cache.shape[-1], n_kv, 1)
        k_cache_index = pg_cache_index.reshape(t, 1)
        k_scale_cache_index = pg_scale_cache_index.reshape(t, 1)

        device = x.device
        q_fp8e4m3 = torch.empty((t * head_num, head_dim), device=device, dtype=torch.float8_e4m3fn)
        q_scale = torch.empty((t * head_num, 1), device=device, dtype=torch.float32)
        k_fp8e4m3 = k_cache
        k_scale = k_scale_cache
        weights = torch.empty((t, head_num), device=device, dtype=torch.bfloat16)

        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(x, FakeTensor):
            return q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights

        lightning_indexer_prolog_quant(
            x,
            q_norm,
            q_norm_scale,
            w_qb,
            w_qb_scale,
            wk,
            w_proj,
            gamma_k,
            cos_idx_rope,
            sin_idx_rope,
            hadamard_q,
            hadamard_k,
            k_cache,
            k_scale_cache,
            k_cache_index,
            k_scale_cache_index,
            q_fp8e4m3,
            q_scale,
            k_fp8e4m3,
            k_scale,
            weights,
        )

        k_fp8e4m3 = k_fp8e4m3.view(block_num, -1)[
            :, k_storage_offset : k_storage_offset + block_size * n_kv * head_dim
        ].view(block_num, block_size, n_kv, head_dim)
        k_scale = k_scale.view(block_num, -1)[
            :, k_scale_storage_offset : k_scale_storage_offset + block_size * n_kv * 1
        ].view(block_num, block_size, n_kv, 1)

        q_fp8e4m3 = q_fp8e4m3.view(t, head_num, head_dim)
        q_scale = q_scale.view(t, head_num, 1)

        return q_fp8e4m3, q_scale, k_fp8e4m3, k_scale, weights

    _lightning_indexer_prolog_quant_mxfp8_pypto = allow_in_graph(_lightning_indexer_prolog_quant_mxfp8_pypto)

    try:
        torch.library.impl(_pyptolib, "lightning_indexer_prolog_quant_mxfp8", "NPU")(
            _lightning_indexer_prolog_quant_mxfp8_pypto
        )
    except Exception as e:
        if "could not parse dispatch key: NPU" in str(e):
            logging.warning(
                "Skip: torchair not installed, skip NPU registration for "
                "operator 'lightning_indexer_prolog_quant_mxfp8'"
            )
        else:
            logging.warning("Skip: Unexpected error during NPU registration: %s", e)

    def lightning_indexer_prolog_quant_mxfp8(
        x,
        q_norm,
        q_norm_scale,
        w_qb,
        w_qb_scale,
        wk,
        w_proj,
        gamma_k,
        cos_idx_rope,
        sin_idx_rope,
        hadamard_q,
        hadamard_k,
        k_cache,
        k_scale_cache,
        k_cache_index,
        k_scale_cache_index,
    ):
        return torch.ops.pypto.lightning_indexer_prolog_quant_mxfp8(
            x,
            q_norm,
            q_norm_scale,
            w_qb,
            w_qb_scale,
            wk,
            w_proj,
            gamma_k,
            cos_idx_rope,
            sin_idx_rope,
            hadamard_q,
            hadamard_k,
            k_cache,
            k_scale_cache,
            k_cache_index,
            k_scale_cache_index,
        )
else:

    def lightning_indexer_prolog_quant_mxfp8(*args, **kwargs):
        raise RuntimeError("pypto is not available, lightning_indexer_prolog_quant_mxfp8 is disabled")
