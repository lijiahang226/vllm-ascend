# SPDX-License-Identifier: Apache-2.0
"""Compare W4A4 MXFP4 linear ND/NZ paths on A5, without vLLM or a model.

Run: python verify_w4a4_mxfp4_nz.py --device 0
NZ conversion matches PR #16616: cast the uint8 carrier, then transpose.
"""

import argparse

import torch
import torch_npu

ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29
MX_GROUP_SIZE = 32


def describe(name, tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"stride={tensor.stride()}, format={torch_npu.get_npu_format(tensor)}",
        flush=True,
    )


def make_input(rows, k, device):
    # Vary scales across both rows and K groups to expose layout mistakes.
    values = torch.randn(rows, k // MX_GROUP_SIZE, MX_GROUP_SIZE)
    exponents = torch.randint(-2, 3, (rows, k // MX_GROUP_SIZE, 1))
    return (values * torch.exp2(exponents.float())).reshape(rows, k).to(device=device, dtype=torch.bfloat16)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()
    if min(args.m, args.n) <= 0 or args.k <= 64 or args.k % 64 or args.n % 32:
        parser.error("Use M > 0, K > 64 divisible by 64, and N > 0 divisible by 32.")
    if args.rtol < 0 or args.atol < 0:
        parser.error("Tolerances must be nonnegative.")

    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(args.seed)
    print(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}, device={device}", flush=True)
    x = make_input(args.m, args.k, device)
    w = make_input(args.n, args.k, device)  # Logical [N, K].
    xq, xs = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2, round_mode="round")
    wq, ws = torch_npu.npu_dynamic_mx_quant(w, dst_type=torch_npu.float4_e2m1fn_x2, round_mode="round")
    assert wq.dtype == torch.uint8 and tuple(wq.shape) == (args.n, args.k // 2)
    assert tuple(ws.shape) == (args.n, args.k // 64, 2)
    describe("xq", xq)
    describe("x_scale", xs)
    describe("packed_weight", wq)

    # Keep the packing along K. Do not make the transposed FP4 bytes contiguous.
    w_nd = wq.transpose(0, 1)
    scale = ws.transpose(0, 1)

    def matmul(weight):
        return torch_npu.npu_quant_matmul(
            xq,
            weight,
            scale,
            pertoken_scale=xs,
            output_dtype=torch.bfloat16,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, MX_GROUP_SIZE],
        )

    describe("weight_ND", w_nd)
    describe("weight_scale", scale)
    assert torch_npu.get_npu_format(w_nd) == ACL_FORMAT_ND
    print("Running ND matmul...", flush=True)
    y_nd = matmul(w_nd).cpu().float()  # Copy also waits for the NPU result.

    print("Casting packed uint8 weight to NZ (no dtype overrides)...", flush=True)
    w_nz = torch_npu.npu_format_cast(wq, ACL_FORMAT_FRACTAL_NZ).transpose(0, 1)
    describe("weight_NZ", w_nz)
    assert torch_npu.get_npu_format(w_nz) == ACL_FORMAT_FRACTAL_NZ
    print("Running NZ matmul...", flush=True)
    y_nz = matmul(w_nz).cpu().float()

    assert y_nd.shape == y_nz.shape == (args.m, args.n)
    assert torch.isfinite(y_nd).all() and torch.isfinite(y_nz).all(), "Non-finite output"
    diff = y_nz - y_nd
    rel_l2 = diff.norm() / y_nd.norm().clamp_min(1e-12)
    print(f"output={tuple(y_nz.shape)}, max_abs={diff.abs().max().item():.6g}, rel_l2={rel_l2.item():.6g}")
    torch.testing.assert_close(y_nz, y_nd, rtol=args.rtol, atol=args.atol)
    print(f"PASS: W4A4 MXFP4 NZ matches ND (rtol={args.rtol}, atol={args.atol})")


if __name__ == "__main__":
    main()
