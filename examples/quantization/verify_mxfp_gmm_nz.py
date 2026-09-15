# SPDX-License-Identifier: Apache-2.0
"""Compare MXFP4/MXFP8 grouped-matmul ND/NZ paths on A5 without vLLM.

python examples/quantization/verify_mxfp_gmm_nz.py --quant-type mxfp4
python examples/quantization/verify_mxfp_gmm_nz.py --quant-type mxfp8

Checks ordinary npu_grouped_matmul, not fused SwiGLU or EPLB. Tokens are
already grouped by expert. Default groups [16, 0, 48] include an empty expert.
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


def make_input(shape, device):
    # Give different experts, rows and K groups different quantization scales.
    grouped_shape = (*shape[:-1], shape[-1] // MX_GROUP_SIZE, MX_GROUP_SIZE)
    values = torch.randn(grouped_shape)
    exponents = torch.randint(-2, 3, (*grouped_shape[:-1], 1))
    return (values * torch.exp2(exponents.float())).reshape(shape).to(device=device, dtype=torch.bfloat16)


def check_close(name, actual, expected, args):
    actual, expected = actual.cpu().float(), expected.cpu().float()
    assert actual.shape == expected.shape == (sum(args.group_sizes), args.n)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), "Non-finite output"
    diff = actual - expected
    rel_l2 = diff.norm() / expected.norm().clamp_min(1e-12)
    print(f"{name}: max_abs={diff.abs().max().item():.6g}, rel_l2={rel_l2.item():.6g}", flush=True)
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)


def expert_reference(xq, xs, wq, ws, args):
    """Independent group slicing, using ND linear matmul for each expert."""
    logical_dtypes = (
        {"x1_dtype": torch_npu.float4_e2m1fn_x2, "x2_dtype": torch_npu.float4_e2m1fn_x2}
        if args.quant_type == "mxfp4"
        else {}
    )
    outputs = []
    start = 0
    for expert, count in enumerate(args.group_sizes):
        if count:
            outputs.append(
                torch_npu.npu_quant_matmul(
                    xq[start : start + count],
                    wq[expert].transpose(0, 1),
                    ws[expert].transpose(0, 1),
                    pertoken_scale=xs[start : start + count],
                    output_dtype=torch.bfloat16,
                    scale_dtype=torch_npu.float8_e8m0fnu,
                    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                    group_sizes=[1, 1, MX_GROUP_SIZE],
                    **logical_dtypes,
                ).cpu()
            )
        start += count
    return torch.cat(outputs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-type", choices=("mxfp4", "mxfp8"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[16, 0, 48])
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()
    if not 2 <= len(args.group_sizes) <= 1024 or min(args.group_sizes) < 0 or sum(args.group_sizes) <= 0:
        parser.error("Use 2..1024 nonnegative expert group sizes with a positive total.")
    if args.k <= 64 or args.k % 64 or args.n <= 0 or args.n % 32:
        parser.error("Use K > 64 divisible by 64, and N > 0 divisible by 32.")
    if args.rtol < 0 or args.atol < 0:
        parser.error("Tolerances must be nonnegative.")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(args.seed)
    print(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}, device={device}", flush=True)
    m, experts = sum(args.group_sizes), len(args.group_sizes)
    is_fp4 = args.quant_type == "mxfp4"
    quant_args = (
        {"dst_type": torch_npu.float4_e2m1fn_x2, "round_mode": "round"}
        if is_fp4
        else {"dst_type": torch.float8_e4m3fn, "scale_alg": 0}
    )
    xq, xs = torch_npu.npu_dynamic_mx_quant(make_input((m, args.k), device), **quant_args)
    wq, ws = torch_npu.npu_dynamic_mx_quant(make_input((experts, args.n, args.k), device), **quant_args)
    assert wq.dtype == (torch.uint8 if is_fp4 else torch.float8_e4m3fn)
    assert tuple(wq.shape) == (experts, args.n, args.k // (2 if is_fp4 else 1))
    assert tuple(ws.shape) == (experts, args.n, args.k // 64, 2)
    assert tuple(xs.shape) == (m, args.k // 64, 2)
    describe("xq", xq)
    describe("x_scale", xs)

    w_nd, scale_nd = wq.transpose(1, 2), ws.transpose(1, 2)
    describe("weight_ND", w_nd)
    describe("weight_scale_ND", scale_nd)
    assert torch_npu.get_npu_format(w_nd) == ACL_FORMAT_ND
    logical_dtypes = (
        {"x_dtype": torch_npu.float4_e2m1fn_x2, "weight_dtype": torch_npu.float4_e2m1fn_x2} if is_fp4 else {}
    )

    def gmm(weight, scale, groups, group_list_type):
        return torch_npu.npu_grouped_matmul(
            x=[xq],
            weight=[weight],
            scale=[scale],
            per_token_scale=[xs],
            group_list=groups,
            group_list_type=group_list_type,
            split_item=2,
            group_type=0,
            output_dtype=torch.bfloat16,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            **logical_dtypes,
        )[0].cpu()

    print("Running per-expert ND linear reference...", flush=True)
    reference = expert_reference(xq, xs, wq, ws, args)
    counts = torch.tensor(args.group_sizes, dtype=torch.int64)
    groups_by_type = (counts.cumsum(0).to(device), counts.to(device))
    nd_outputs = []
    for group_list_type, groups in enumerate(groups_by_type):
        print(f"Running ND GMM: group_list_type={group_list_type}, counts={args.group_sizes}", flush=True)
        nd_outputs.append(gmm(w_nd, scale_nd, groups, group_list_type))
        check_close(f"ND GMM vs expert reference (type={group_list_type})", nd_outputs[-1], reference, args)

    print("Casting GMM weights to NZ...", flush=True)
    if is_fp4:
        # uint8 packs two K values per byte: cast before transpose, no dtype overrides.
        w_nz = torch_npu.npu_format_cast(wq, ACL_FORMAT_FRACTAL_NZ).transpose(1, 2)
        scale_nz = scale_nd
    else:
        # Match the MXFP8 MoE NZ weight and scale layout.
        w_nz = torch_npu.npu_format_cast(w_nd.contiguous(), ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn)
        scale_nz = scale_nd.contiguous()
    describe("weight_NZ", w_nz)
    describe("weight_scale_NZ", scale_nz)
    assert torch_npu.get_npu_format(w_nz) == ACL_FORMAT_FRACTAL_NZ
    for group_list_type, groups in enumerate(groups_by_type):
        print(f"Running NZ GMM: group_list_type={group_list_type}", flush=True)
        y_nz = gmm(w_nz, scale_nz, groups, group_list_type)
        check_close(f"NZ vs ND GMM (type={group_list_type})", y_nz, nd_outputs[group_list_type], args)
        check_close(f"NZ GMM vs expert reference (type={group_list_type})", y_nz, reference, args)
    print(f"PASS: {args.quant_type} GMM ND/NZ, group_list_type=0/1 (rtol={args.rtol}, atol={args.atol})")


if __name__ == "__main__":
    main()
