# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu  # noqa: F401

from vllm_ascend.quantization.methods.w8a8.fp8_block import resolve_block_scales


def test_fp8_decode_all_bits_with_default_npu_device():
    raw = torch.arange(256, dtype=torch.uint8, device="cpu").reshape(16, 16)
    weight = raw.view(torch.float8_e4m3fn)
    expected = weight.float()
    device_weight = raw.npu().view(torch.float8_e4m3fn)
    scale = torch.ones(1, 1, dtype=torch.float32, device="npu")

    with torch.device("npu"):
        actual = resolve_block_scales(device_weight, scale, 16, 16, torch.float32)

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
