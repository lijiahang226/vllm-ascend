# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay captured, normalized KDA inputs directly at the native operator boundary."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def load_case(path):
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key].copy() for key in z.files}
    for key in ("q", "k", "v", "raw_gate", "baseline_output", "padded_output"):
        bits = values.pop(key + "_bf16_bits")
        values[key] = (bits.astype(np.uint32) << 16).view(np.float32)
    return values


def reference(case):
    q, k, v, gate, beta = [case[key].astype(np.float64) for key in ("q", "k", "v", "raw_gate", "beta")]
    a_log, bias = case["A_log"].astype(np.float64), case["dt_bias"].astype(np.float64)
    gate = float(case["lower_bound"]) / (
        1 + np.exp(-np.exp(a_log).reshape(1, 1, 8, 1) * (gate + bias.reshape(1, 1, 8, 128)))
    )
    state = case["initial_state"][0].astype(np.float64).copy()
    output = np.empty_like(v)
    for token in range(581):
        state *= np.exp(gate[0, token])[:, None, :]
        residual = v[0, token] - np.sum(state * k[0, token][:, None, :], axis=-1)
        state += (beta[0, token, :, None] * residual)[:, :, None] * k[0, token][:, None, :]
        output[0, token] = np.sum(state * q[0, token][:, None, :], axis=-1) * 128**-0.5
    return output, state[None]


def relative(actual, expected):
    return float(
        np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2)) / max(np.sqrt(np.mean(expected**2)), 1e-30)
    )


def metrics(output, state, expected, expected_state):
    return {
        "output_relative_RMS_percent": 100 * relative(output, expected),
        "tail_5_output_relative_RMS_percent": 100 * relative(output[:, 576:581], expected[:, 576:581]),
        "final_state_relative_RMS_percent": 100 * relative(state, expected_state),
    }


def native_replay(args, case, expected, expected_state, report):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    if args.backend == "fla":
        from fla_npu.ops.ascendc import chunk_kda_fwd
    else:
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        chunk_kda_fwd = torch.ops._C_ascend.chunk_kda_fwd
    for package in ("torch", "torch-npu", "flash-linear-attention-npu-a5", "vllm", "vllm-ascend"):
        try:
            report.setdefault("package_versions", {})[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report.setdefault("package_versions", {})[package] = None
    report.update(backend=args.backend, device=args.device, repeats=args.repeats)
    with torch.inference_mode():
        for padded in (False, True):
            length = 640 if padded else 581
            tensors = []
            for key in ("q", "k", "v", "raw_gate", "beta"):
                dtype = torch.float32 if key == "beta" else torch.bfloat16
                tensor = torch.from_numpy(case[key]).to(device=args.device, dtype=dtype)
                if padded:
                    fill = float("-inf") if key == "raw_gate" else 0
                    tail = tensor.new_full((1, 59, *tensor.shape[2:]), fill)
                    tensor = torch.cat((tensor, tail), dim=1)
                tensors.append(tensor.contiguous())
            a_log = torch.from_numpy(case["A_log"]).to(args.device)
            dt_bias = torch.from_numpy(case["dt_bias"]).to(args.device)
            initial = torch.from_numpy(case["initial_state"]).to(args.device)
            first = None
            for repeat in range(args.repeats):
                outputs = chunk_kda_fwd(
                    *tensors,
                    128**-0.5,
                    64,
                    layout="BSND",
                    initial_state=initial.clone(),
                    output_final_state=True,
                    cu_seqlens=[0, length],
                    chunk_indices=[part for chunk in range(10) for part in (0, chunk)],
                    safe_gate=True,
                    lower_bound=-5.0,
                    use_gate_in_kernel=True,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    disable_recompute=False,
                    return_intermediate_states=False,
                    state_v_first=True,
                )
                torch.npu.synchronize()
                actual = outputs[0][:, :581].float().cpu().numpy()
                final = outputs[1].float().cpu().numpy()
                assert np.isfinite(actual).all() and np.isfinite(final).all()
                row = {
                    "padded": padded,
                    "repeat": repeat,
                    "input_length": length,
                    "chunk_count": 10,
                    "bitwise_equal_to_first": first is None or bool(np.array_equal(first, actual)),
                    **metrics(actual, final, expected, expected_state),
                }
                report["new_native_results"].append(row)
                np.savez_compressed(
                    args.output / f"native-{'padded' if padded else 'baseline'}-{repeat}.npz",
                    output=actual,
                    final_state=final,
                )
                if first is None:
                    first = actual.copy()
                print(json.dumps(row), flush=True)
                (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("kda581_case.npz"))
    parser.add_argument("--output", type=Path, default=Path("kda581-results"))
    parser.add_argument(
        "--audit-only", action="store_true", help="Recompute archived outputs on CPU; no torch/NPU required"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--backend", choices=("fla", "vllm-ascend"), default="fla")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.inputs.with_name("case_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(args.inputs.read_bytes()).hexdigest() == manifest["npz_sha256"]
    case = load_case(args.inputs)
    assert case["q"].shape == case["k"].shape == case["v"].shape == case["raw_gate"].shape == (1, 581, 8, 128)
    assert case["beta"].shape == (1, 581, 8)
    expected, expected_state = reference(case)
    audit = {
        name: metrics(case[name + "_output"], case[name + "_state"], expected, expected_state)
        for name in ("baseline", "padded")
    }
    assert audit["baseline"]["tail_5_output_relative_RMS_percent"] > 50
    assert audit["padded"]["tail_5_output_relative_RMS_percent"] < 0.3
    report = {
        "mode": "saved_outputs_CPU_audit" if args.audit_only else "new_native_replay",
        "archived_output_audit": audit,
        "new_native_results": [],
    }
    np.savez_compressed(args.output / "FP64_reference.npz", output=expected, final_state=expected_state)
    if not args.audit_only:
        native_replay(args, case, expected, expected_state, report)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
