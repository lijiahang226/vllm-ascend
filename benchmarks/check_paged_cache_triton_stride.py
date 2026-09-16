# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure Triton on the layout for which native scatter rejects the cache."""

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from statistics import median

import torch
import torch_npu


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("comparison", root / "compare_paged_cache_scatter_nd.py")
    comparison = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = comparison
    spec.loader.exec_module(comparison)
    comparison.torch, comparison.torch_npu = torch, torch_npu
    torch.npu.set_device(0)
    triton_writer = comparison.load_triton_reference()
    case = next(case for case in comparison.all_cases() if case.name == "t4096_feature_stride2")
    inputs = comparison.make_inputs(case, "npu:0")
    fn = partial(triton_writer, inputs["cache"], inputs["slots"], inputs["values"])
    result = {"case": case.name, "triton_reference_sha256": comparison.TRITON_REFERENCE_SHA256, "modes": {}}
    for mode in ("eager", "graph"):
        accuracy, _ = comparison.check_accuracy(case, fn, inputs, mode)
        assert accuracy["pass"], accuracy
        graph = comparison.capture(fn, 32) if mode == "graph" else None
        call = graph.replay if graph else fn
        samples = [comparison.measure(call, 8, 32 if graph else 1) for _ in range(15)]
        assert comparison.byte_difference(inputs["backing"], comparison.reference(inputs)) == 0
        assert comparison.byte_difference(inputs["source"], inputs["source_cpu"]) == 0
        assert comparison.byte_difference(inputs["slot_storage"], inputs["slot_cpu"]) == 0
        result["modes"][mode] = {
            "accuracy": accuracy,
            "triton_us": median(sample["event_us"] for sample in samples),
            "samples": samples,
        }
        (root / "triton_stride.json").write_text(json.dumps(result, indent=2))
        print("TRITON_STRIDE", mode, result["modes"][mode]["triton_us"], flush=True)


if __name__ == "__main__":
    main()
