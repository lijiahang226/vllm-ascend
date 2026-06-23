import math

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def split_kv_rmsnorm_rope_kernel(
    rms_in_ptr,
    rope_in_ptr,
    gamma_ptr,
    cos_sin_ptr,
    k_nope_out_ptr,
    k_rope_out_ptr,
    total_tokens,
    eps: tl.constexpr,
    num_vectorcore: tl.constexpr,
    tokens_per_program: tl.constexpr,
    RMS_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * tokens_per_program
    row_end = min(row_start + tokens_per_program, total_tokens)

    col_rms = tl.arange(0, RMS_DIM)
    gamma = tl.load(gamma_ptr + col_rms).to(tl.float32)

    pair_offsets = 2 * tl.arange(0, HALF_ROPE_DIM)[:, None] + tl.arange(0, 2)[None, :]
    pair_offsets_wide = 4 * tl.arange(0, HALF_ROPE_DIM)[:, None] + tl.arange(0, 4)[None, :]
    CS_STRIDE = ROPE_DIM * 2

    for row_idx in tl.range(row_start, row_end):
        # --- RMSNorm ---
        rms_vals = tl.load(rms_in_ptr + row_idx * RMS_DIM + col_rms).to(tl.float32)
        denom = tl.sqrt(tl.sum(rms_vals * rms_vals) / RMS_DIM + eps)
        tl.store(k_nope_out_ptr + row_idx * RMS_DIM + col_rms, (rms_vals / denom) * gamma)

        # --- Interleaved RoPE ---
        rope_base = rope_in_ptr + row_idx * ROPE_DIM
        cs_base = cos_sin_ptr + row_idx * CS_STRIDE
        out_base = k_rope_out_ptr + row_idx * ROPE_DIM

        rope_tile = tl.load(rope_base + pair_offsets).to(tl.float32)
        cs_tile = tl.load(cs_base + pair_offsets_wide).to(tl.float32)
        cos_tile = cs_tile[:, 0]
        sin_tile = cs_tile[:, 1]

        x_even, x_odd = tl.split(rope_tile)
        o_even = x_even * cos_tile - x_odd * sin_tile
        o_odd = x_odd * cos_tile + x_even * sin_tile
        k_rope_tile = tl.join(o_even, o_odd)

        tl.store(out_base + pair_offsets, k_rope_tile)


def _prepare_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    total_tokens: int,
    rope_dim: int,
) -> torch.Tensor:
    cos_2d = cos.reshape(total_tokens, rope_dim)
    sin_2d = sin.reshape(total_tokens, rope_dim)
    merged = torch.empty(total_tokens, rope_dim * 2, dtype=torch.float32, device=cos.device)
    merged[:, 0::2] = cos_2d
    merged[:, 1::2] = sin_2d
    return merged


def split_kv_rmsnorm_rope_impl(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    epsilon: float = 1e-5,
    rms_dim: int = 512,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv.ndim >= 2
    assert kv.shape[-1] == rms_dim + rope_dim

    rms_in, rope_in = kv.split([rms_dim, rope_dim], dim=-1)
    total_tokens = rms_in.numel() // rms_dim
    rms_in_2d = rms_in.reshape(total_tokens, rms_dim).contiguous()
    rope_in_2d = rope_in.reshape(total_tokens, rope_dim).contiguous()
    gamma_1d = gamma.reshape(-1).contiguous()
    cos_sin = _prepare_cos_sin(cos, sin, total_tokens, rope_dim)

    k_nope_out = torch.empty(total_tokens, rms_dim, dtype=torch.float32, device=kv.device)
    k_rope_out = torch.empty(total_tokens, rope_dim, dtype=torch.float32, device=kv.device)

    num_vectorcore = get_vectorcore_num()
    UB_SIZE = 87040
    factor = rms_dim * 2 + rope_dim + rope_dim * 2 + rms_dim + rope_dim
    tokens_per_program = max(1, int(UB_SIZE / 4) // factor)
    tokens_per_program = max(1, min(tokens_per_program, 32))
    total_tokens_per_program = math.ceil(total_tokens / num_vectorcore)
    tokens_per_program = max(1, min(tokens_per_program, total_tokens_per_program))

    grid = (num_vectorcore, 1, 1)
    split_kv_rmsnorm_rope_kernel[grid](
        rms_in_2d,
        rope_in_2d,
        gamma_1d,
        cos_sin,
        k_nope_out,
        k_rope_out,
        total_tokens,
        epsilon,
        num_vectorcore,
        tokens_per_program,
        RMS_DIM=rms_dim,
        ROPE_DIM=rope_dim,
        HALF_ROPE_DIM=rope_dim // 2,
    )

    return k_nope_out, k_rope_out


def split_kv_rmsnorm_rope_impl_fake(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    epsilon: float = 1e-5,
    rms_dim: int = 512,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = kv.numel() // (rms_dim + rope_dim)
    k_nope = torch.empty(total_tokens, rms_dim, dtype=torch.float32, device=kv.device)
    k_rope = torch.empty(total_tokens, rope_dim, dtype=torch.float32, device=kv.device)
    return k_nope, k_rope


direct_register_custom_op(
    op_name="split_kv_rmsnorm_rope",
    op_func=split_kv_rmsnorm_rope_impl,
    fake_impl=split_kv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
