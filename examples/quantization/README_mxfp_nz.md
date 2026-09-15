# A5 MXFP ND/NZ validation

These standalone examples require `torch` and `torch_npu` on an A5 machine
with CANN configured. They do not require vLLM or model checkpoints.
Run from the repository root on the `verify-mxfp-nz` branch:

```bash
python examples/quantization/verify_w4a4_mxfp4_nz.py --device 0
python examples/quantization/verify_w8a8_mxfp8_nz.py --device 0
python examples/quantization/verify_mxfp_gmm_nz.py --quant-type mxfp4 --device 0
python examples/quantization/verify_mxfp_gmm_nz.py --quant-type mxfp8 --device 0
```

Each example quantizes its inputs once and compares outputs using the same
quantized values and scales. It prints tensor shapes, strides, formats,
maximum absolute error and relative L2 error. Success prints `PASS`;
operator errors, invalid outputs or numerical mismatches cause a nonzero exit.
The default tolerance is `rtol=0.01, atol=0.01`; use `--rtol 0 --atol 0` to
require exact equality.

The GMM example uses ordinary `npu_grouped_matmul` with one packed token
tensor and one 3D expert-weight tensor. It verifies both cumulative boundaries
(`group_list_type=0`) and token counts (`group_list_type=1`). The default
expert token counts `[16, 0, 48]` exercise unequal groups and an empty expert.
Both ND and NZ results are also checked against concatenated per-expert ND
`npu_quant_matmul` results. Tokens are assumed to be already sorted by expert;
routing and sorting are outside this example.

To test different shapes or groups:

```bash
python examples/quantization/verify_mxfp_gmm_nz.py --quant-type mxfp4 --group-sizes 8 24 --n 256 --k 1024
python examples/quantization/verify_mxfp_gmm_nz.py --quant-type mxfp8 --group-sizes 0 1 31 0 --n 256 --k 1024
```

For FP4, NZ conversion casts the packed `uint8` carrier before transposing,
without format-cast dtype overrides; the operators receive their FP4 logical
dtype arguments. For FP8, NZ conversion transposes the weight, makes it
contiguous and casts with `customize_dtype=torch.float8_e4m3fn`; weight scales
are also made contiguous. Only weights are converted to NZ.

These are single-operator checks. They do not validate fused SwiGLU GMM,
EPLB, graph capture, tensor-parallel loading or full-model accuracy. The FP4
GMM NZ case tests the operator conversion directly; it does not enable NZ in
the model's FP4 MoE implementation. Local validation covered syntax, lint
and static review; actual A5 execution remains to be performed.

Reference: [Ascend grouped-matmul API](https://github.com/Ascend/op-plugin/blob/d4569a7fa8e051703773b33cb3190749f499bb41/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md).
