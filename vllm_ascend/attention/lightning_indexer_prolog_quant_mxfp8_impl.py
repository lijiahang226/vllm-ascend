#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lightning Indexer Prolog Quantization Module (MXFP8).

This module implements the Lightning Indexer Prolog quantization computation
using pypto frontend JIT compilation. It fuses query/key/weights computation
and KV cache scatter into a single optimized kernel.
"""

import pypto

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3
COS_SIN_DIM = 2
SCATTER_DIM = -2


def quant_rms_norm(x: pypto.Tensor, gamma: pypto.Tensor, dim: int, epsilon: float):
    assert (dim == len(x.shape) - 1) or (dim == -1)
    actual_dim = dim + len(x.shape) if dim < 0 else dim
    x_dtype = x.dtype

    x_fp32 = pypto.cast(x, pypto.DT_FP32)
    x2 = x_fp32 * x_fp32
    x2_scaled = x2 * (1.0 / x.shape[actual_dim])
    mean_square = pypto.sum(x2_scaled, actual_dim, keepdim=True)

    rms = pypto.sqrt(mean_square + epsilon)
    res32 = pypto.div(x_fp32, rms, pypto.PrecisionType.INTRINSIC)
    gamma32 = pypto.cast(gamma, pypto.DT_FP32)
    return pypto.cast((res32 * gamma32), x_dtype)


def quant_rope_2d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor):
    key_rope_dim = 2
    x_dtype = x.dtype
    t_tile = x.shape[0]
    rope_dim = x.shape[1]
    assert len(x.shape) == key_rope_dim and len(cos.shape) == COS_SIN_DIM and len(sin.shape) == COS_SIN_DIM

    pypto.set_vec_tile_shapes(t_tile, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    x_view = pypto.cast(x, pypto.DT_FP32)

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


def prolog_quant(x: pypto.Tensor):
    pypto.experimental.set_operation_options(combine_axis=True)

    fp8_max_value = 448.0
    fp8_one_value = 1.0
    input_fp32 = pypto.cast(x, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

    abs_res = pypto.abs(input_fp32)
    max_value = pypto.amax(abs_res, dim=-1, keepdim=True)

    scale_dequant = max_value * (fp8_one_value / fp8_max_value)
    out_fp32 = pypto.div(input_fp32, scale_dequant, pypto.PrecisionType.INTRINSIC)
    out_fp8 = pypto.cast(out_fp32, pypto.DT_FP8E4M3, satmode=pypto.SaturationMode.ON)
    return (out_fp8, scale_dequant)


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % chunk_size == 0
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    head_num_axis = 1
    head_dim_axis = 2
    assert len(x.shape) == SHAPE_DIM_3 and len(cos.shape) == SHAPE_DIM_2 and len(sin.shape) == SHAPE_DIM_2

    x_dtype = x.dtype
    t_tile = x.shape[0]
    head_num = x.shape[head_num_axis]
    rope_dim = x.shape[head_dim_axis]

    pypto.set_vec_tile_shapes(8, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [t_tile, 1, rope_dim])
    cast_sin = pypto.reshape(cast_sin, [t_tile, 1, rope_dim])

    pypto.set_vec_tile_shapes(8, head_num, rope_dim)
    x_view = pypto.cast(x, pypto.DT_FP32)

    x_embed = (x_view * cast_cos) + ((rotate_half(x_view)) * cast_sin)
    res = pypto.cast(x_embed, x_dtype)
    return res


@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {0: 2, 1: 4, 5: 16, 7: 8, -2: 1},
        "cube_l1_reuse_setting": {-1: 8},
    },
    runtime_options={"device_sched_mode": 1},
)
def lightning_indexer_prolog_quant(
    x_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    q_norm_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP8E4M3, format=pypto.TileOpFormat.TILEOP_ND),
    q_norm_scale_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP8E8M0, format=pypto.TileOpFormat.TILEOP_ND),
    w_qb_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP8E4M3, format=pypto.TileOpFormat.TILEOP_ND),
    w_qb_scale_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP8E8M0, format=pypto.TileOpFormat.TILEOP_ND),
    wk_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    w_proj_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    gamma_k_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    cos_idx_rope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    sin_idx_rope_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    hadamard_q_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    hadamard_k_in: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
    k_quant_in: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP8E4M3, format=pypto.TileOpFormat.TILEOP_ND
    ),
    k_scale_in: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND
    ),
    k_cache_index_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT64, format=pypto.TileOpFormat.TILEOP_ND),
    k_scale_cache_index_in: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT64, format=pypto.TileOpFormat.TILEOP_ND
    ),
    q_quant_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP8E4M3, format=pypto.TileOpFormat.TILEOP_ND),
    q_scale_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    k_quant_out: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP8E4M3, format=pypto.TileOpFormat.TILEOP_ND
    ),
    k_scale_out: pypto.Tensor(
        [pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND
    ),
    weights_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16, format=pypto.TileOpFormat.TILEOP_ND),
):
    x_dtype = x_in.dtype
    t = x_in.shape[0]
    h = x_in.shape[1]
    q_lora_rank = q_norm_in.shape[1]
    head_num = w_proj_in.shape[1]
    head_dim = hadamard_q_in.shape[0]
    rope_head_dim = cos_idx_rope_in.shape[1]

    unroll_list = [128, 64, 32, 16, 8, 4, 2, 1]
    for t_idx, unroll_length in pypto.loop_unroll(
        0, t, 1, name="IndexerPrologQuantQuantLoop", idx_name="t_idx", unroll_list=unroll_list
    ):
        t_tile = unroll_length
        pypto.set_semantic_label("Query-Dequant-Linear")
        q_norm = pypto.view(q_norm_in, [t_tile, q_lora_rank], [t_idx, 0])
        q_norm_scale = pypto.view(q_norm_scale_in, [t_tile, q_lora_rank // 64, 2], [t_idx, 0, 0])
        pypto.set_cube_tile_shapes([128, 128], [256, 1024], [128, 128])
        q_scaled_mm = pypto.scaled_mm(q_norm, w_qb_in, x_dtype, q_norm_scale, w_qb_scale_in)

        pypto.set_semantic_label("Query-Rope")
        pypto.set_vec_tile_shapes(8, head_num * head_dim)
        q_bf16 = pypto.reshape(q_scaled_mm, [t_tile, head_num, head_dim])
        q_rope = pypto.view(q_bf16, [t_tile, head_num, rope_head_dim], [0, 0, 0])
        q_nope = pypto.view(q_bf16, [t_tile, head_num, head_dim - rope_head_dim], [0, 0, rope_head_dim])
        rope_cos = pypto.view(cos_idx_rope_in, [t_tile, rope_head_dim], [t_idx, 0])
        rope_sin = pypto.view(sin_idx_rope_in, [t_tile, rope_head_dim], [t_idx, 0])
        q_roped = rope_3d(q_rope, rope_cos, rope_sin)
        q_cat = pypto.concat([q_roped, q_nope], -1)
        pypto.set_vec_tile_shapes(8, head_num, head_dim)
        q_cat_2d = pypto.reshape(q_cat, [t_tile * head_num, head_dim])

        pypto.set_semantic_label("Query-Hadamard")
        pypto.set_cube_tile_shapes([256, 256], [128, 128], [128, 128])
        q_hadamard = pypto.matmul(q_cat_2d, hadamard_q_in, x_dtype)

        pypto.set_semantic_label("Query-Quant")
        pypto.set_vec_tile_shapes(128, head_dim)
        q_res = prolog_quant(q_hadamard)
        pypto.assemble(q_res[0], [t_idx * head_num, 0], q_quant_out)
        pypto.assemble(q_res[1], [t_idx * head_num, 0], q_scale_out)

        pypto.set_semantic_label("Key-Linear")
        pypto.set_cube_tile_shapes([128, 128], [256, 1024], [128, 128])
        x = pypto.view(x_in, [t_tile, h], [t_idx, 0])
        k_proj = pypto.matmul(x, wk_in, x_dtype)

        pypto.set_semantic_label("Key-RmsNorm")
        pypto.set_vec_tile_shapes(128, head_dim)
        k_rms_norm = pypto.cast(quant_rms_norm(k_proj, gamma_k_in, -1, 1e-6), x_dtype)

        pypto.set_semantic_label("Key-Rope")
        k_rope = pypto.view(k_rms_norm, [t_tile, rope_head_dim], [0, 0])
        k_nope = pypto.view(k_rms_norm, [t_tile, head_dim - rope_head_dim], [0, rope_head_dim])
        k_roped = quant_rope_2d(k_rope, rope_cos, rope_sin)
        pypto.set_vec_tile_shapes(128, head_dim)
        k_concat = pypto.concat([k_roped, k_nope], -1)

        pypto.set_semantic_label("Key-Hadamard")
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        hadamard_k = pypto.matmul(k_concat, hadamard_k_in, x_dtype)

        pypto.set_semantic_label("Key-Quant")
        pypto.set_vec_tile_shapes(128, head_dim)
        k_res = prolog_quant(hadamard_k)
        k_cache_4d = pypto.reshape(k_res[0], [t_tile, 1, 1, head_dim])
        k_scale_4d = pypto.reshape(k_res[1], [t_tile, 1, 1, 1])

        index = pypto.view(k_cache_index_in, [t_tile, 1], [t_idx, 0])
        scale_index = pypto.view(k_scale_cache_index_in, [t_tile, 1], [t_idx, 0])
        pypto.set_vec_tile_shapes(128, 1, 1, head_dim)
        k_quant_out.move(pypto.scatter_update(k_quant_in, SCATTER_DIM, index, k_cache_4d))
        k_scale_out.move(pypto.scatter_update(k_scale_in, SCATTER_DIM, scale_index, k_scale_4d))

        pypto.set_semantic_label("Weight-Linear")
        pypto.set_cube_tile_shapes([32, 32], [1024, 1024], [32, 32])
        pypto.set_vec_tile_shapes(128, head_num)
        weights = pypto.cast(pypto.matmul(x, w_proj_in, x_dtype), pypto.DT_FP32)
        weights = pypto.cast(pypto.cast(weights * (head_num**-0.5), pypto.DT_BF16), pypto.DT_FP32)
        weights = pypto.cast(weights * (head_dim**-0.5), pypto.DT_BF16)
        pypto.assemble(weights, [t_idx, 0], weights_out)
