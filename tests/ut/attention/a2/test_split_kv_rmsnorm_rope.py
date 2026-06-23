import gc

import numpy as np
import pytest
import torch

NUM_TOKENS = [1, 4, 8, 128]
RMS_DIM = 512
ROPE_DIM = 64
EPS = [1e-5, 1e-6]
DTYPES = [torch.bfloat16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input = input.to(torch.float32)
    weight = weight.to(torch.float32)
    reciprocal_std = 1 / torch.sqrt(torch.mean(input**2, axis=-1, keepdims=True) + eps)
    return input * reciprocal_std * weight


def interleave_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rope_dim = cos.shape[-1]
    x = x.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    c_pair = cos[..., 0::2]
    s_pair = cos[..., 1::2]
    o_even = x_even * c_pair - x_odd * s_pair
    o_odd = x_odd * c_pair + x_even * s_pair
    out = torch.empty_like(x)
    out[..., 0::2] = o_even.to(out.dtype)
    out[..., 1::2] = o_odd.to(out.dtype)
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_kv_rmsnorm_rope(num_tokens, eps, dtype, seed, device):
    torch.manual_seed(seed)
    torch.set_default_device(device)

    kv = torch.randn(num_tokens, RMS_DIM + ROPE_DIM, dtype=dtype, device=device)
    gamma = torch.randn(RMS_DIM, dtype=dtype, device=device)
    cos = torch.from_numpy(np.random.uniform(-1, 1, [num_tokens, ROPE_DIM])).to(dtype).npu()
    sin = torch.from_numpy(np.random.uniform(-1, 1, [num_tokens, ROPE_DIM])).to(dtype).npu()

    k_nope, k_rope = torch.ops.vllm.split_kv_rmsnorm_rope(kv, gamma, cos, sin, eps)

    rms_in, rope_in = kv.cpu().split([RMS_DIM, ROPE_DIM], dim=-1)
    k_nope_gold = rms_norm(rms_in, gamma.cpu(), eps)
    k_rope_gold = interleave_rope(rope_in, cos.cpu(), sin.cpu())

    torch.testing.assert_close(
        k_nope.to(torch.float32).cpu(), k_nope_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )
    torch.testing.assert_close(
        k_rope.to(torch.float32).cpu(), k_rope_gold, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_split_kv_rmsnorm_rope_nda_input(num_tokens=8):
    torch.set_default_device("npu:0")

    kv = torch.randn(2, num_tokens // 2, RMS_DIM + ROPE_DIM, dtype=torch.bfloat16)
    gamma = torch.randn(RMS_DIM, dtype=torch.bfloat16)
    cos = torch.randn(2, num_tokens // 2, ROPE_DIM, dtype=torch.bfloat16)
    sin = torch.randn(2, num_tokens // 2, ROPE_DIM, dtype=torch.bfloat16)

    k_nope, k_rope = torch.ops.vllm.split_kv_rmsnorm_rope(kv, gamma, cos, sin, 1e-5)

    assert k_nope.shape == (num_tokens, RMS_DIM)
    assert k_rope.shape == (num_tokens, ROPE_DIM)

    gc.collect()
    torch.npu.empty_cache()


@torch.inference_mode()
def test_split_kv_rmsnorm_rope_shape_mismatch():
    torch.set_default_device("npu:0")

    kv = torch.randn(4, 256, dtype=torch.bfloat16)
    gamma = torch.randn(RMS_DIM, dtype=torch.bfloat16)
    cos = torch.randn(4, ROPE_DIM, dtype=torch.bfloat16)
    sin = torch.randn(4, ROPE_DIM, dtype=torch.bfloat16)

    with pytest.raises((AssertionError, RuntimeError)):
        torch.ops.vllm.split_kv_rmsnorm_rope(kv, gamma, cos, sin, 1e-5)

    gc.collect()
    torch.npu.empty_cache()
