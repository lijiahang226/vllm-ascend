# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Paged-cache baselines vs native scatter with complete preprocessing costs.

python compare_paged_cache_scatter_nd.py --suite smoke --device 0 --output results
Use --suite full for boundary/layout/dtype coverage. Each case runs in its own
process, so a rejected native negative index cannot poison subsequent cases.
The default small_ops baseline needs no vLLM/Triton installation. Optional
--baseline triton uses the hash-verified sibling reference and requires both.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from statistics import median

BASELINE_COMMIT = "517e0e5b8969e44077ad96d81e66f9bf76aa3f0f"
TRITON_REFERENCE_SHA256 = "69dbe58d37c84dcf5d95285a41dba282c1c3fb526f98f88944e9f64c7ff57c1b"
METHODS = ("small_ops", "scatter_nd")
REPLAYS = 4


def small_ops(cache, slots, values):
    # PR #16252 attention/utils.py, with the original caller's conversions.
    # Keep the reduction/slot-zero restore intact; do not repair the baseline.
    slots, values = slots.long(), values.to(cache.dtype)
    values = values.reshape(values.shape[0], *cache.shape[2:])
    block_size = cache.shape[1]
    valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_ids = torch.div(safe_slots, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(safe_slots, block_size)
    row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
    old_zero = cache[0, 0].clone()
    safe_values = torch.where(row_mask, values, old_zero.unsqueeze(0))
    writes_zero = valid & (slots == 0)
    zero_value = torch.where(writes_zero.view(-1, *([1] * (values.ndim - 1))), values, torch.zeros_like(values)).sum(
        dim=0
    )
    expected_zero = torch.where(writes_zero.any(), zero_value, old_zero)
    cache[block_ids, block_offsets] = safe_values
    cache[0, 0].copy_(expected_zero)


def scatter_nd(cache, slots, values):
    # Complete candidate: all coordinate/mask/cast work remains inside timing.
    values = values.reshape(values.shape[0], *cache.shape[2:])
    if values.shape[0] == 0 or cache.numel() == 0:
        return
    slots = slots.to(torch.int64)
    valid = (slots >= 0) & (slots < cache.shape[0] * cache.shape[1])
    safe_slots = torch.where(valid, slots, 0)
    indices = torch.stack(
        (torch.div(safe_slots, cache.shape[1], rounding_mode="floor"), safe_slots % cache.shape[1]), dim=-1
    )
    indices = torch.where(valid[:, None], indices, -1)
    # Whether this CANN version ignores [-1, -1] is tested, never assumed.
    torch_npu.npu_scatter_nd_update_(cache, indices, values.to(cache.dtype))


def load_triton_reference():
    # Lazy initialization stays inside the isolated NPU worker, before timing.
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    source = Path(__file__).with_name("paged_cache_triton_reference.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == TRITON_REFERENCE_SHA256
    init_device_properties_triton()
    spec = importlib.util.spec_from_file_location("paged_cache_triton_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.scatter_paged_cache


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int = 64
    block: int = 640
    dim: int = 512
    heads: int = 1
    padding: int = 47616  # Elements between pages, not bytes.
    dtype: str = "bfloat16"
    cache_dtype: str = "bfloat16"
    slot_dtype: str = "int64"
    invalid: str = "none"
    slot_zero: bool = True
    strided_slots: bool = False
    feature_stride: int = 1
    source_padding: int = 0
    special: bool = False


def all_cases():
    cases = [
        Case(f"t{tokens}_pad{padding}_{invalid}", tokens=tokens, padding=padding, invalid=invalid)
        for tokens in (1, 64, 256, 2048, 4096)
        for padding in (0, 47616)
        for invalid in (("none",) if tokens == 1 else ("none", "mixed"))
    ]
    for tokens in (64, 4096):
        cases.extend(
            (
                Case(f"t{tokens}_fp16", tokens=tokens, dtype="float16", cache_dtype="float16"),
                Case(f"t{tokens}_fp32", tokens=tokens, dtype="float32", cache_dtype="float32"),
                Case(f"t{tokens}_fp32_to_bf16", tokens=tokens, dtype="float32"),
                Case(f"t{tokens}_all_invalid", tokens=tokens, invalid="all"),
                Case(f"t{tokens}_strided_slots", tokens=tokens, strided_slots=True),
            )
        )
    cases.extend(
        (
            Case("t4096_projection_slice", tokens=4096, source_padding=512),
            Case("t4096_feature_stride2", tokens=4096, feature_stride=2),
            Case("t64_int32_mixed", slot_dtype="int32", invalid="mixed"),
            Case("t64_no_slot_zero", invalid="mixed", slot_zero=False),
            Case("t64_two_heads", heads=2, invalid="mixed"),
            Case("t64_block384", block=384, invalid="mixed"),
            Case("empty", tokens=0),
        )
    )
    for dtype in ("bfloat16", "float16"):
        for zero in (False, True):
            cases.append(
                Case(
                    f"bits_{dtype}_zero{int(zero)}",
                    tokens=128,
                    dtype=dtype,
                    cache_dtype=dtype,
                    special=True,
                    slot_zero=zero,
                    invalid="mixed",
                )
            )
    return cases


def make_inputs(case, device):
    """Generate once on CPU; both methods get identical independent allocations."""
    torch.manual_seed(16252)
    pages = max(3, (case.tokens + case.block - 1) // case.block + 2)
    head_stride = case.dim * case.feature_stride
    row_stride = case.heads * head_stride
    page_stride = case.block * row_stride + case.padding
    shape = (pages, case.block, case.heads, case.dim)
    stride = (page_stride, row_stride, head_stride, case.feature_stride)
    prefix, suffix = 19, 23
    initial = torch.randn(prefix + pages * page_stride + suffix, dtype=getattr(torch, case.cache_dtype))
    source_stride = row_stride + case.source_padding
    source = torch.randn(7 + case.tokens * source_stride + 17, dtype=getattr(torch, case.dtype))
    values_cpu = source.as_strided(
        (case.tokens, case.heads, case.dim), (source_stride, head_stride, case.feature_stride), 7
    )
    if case.special:
        patterns = torch.arange(values_cpu.numel(), dtype=torch.int32).to(torch.int16).view(values_cpu.dtype)
        values_cpu.copy_(patterns.reshape(values_cpu.shape))
        # Include -0, infinities and NaN payloads in the row targeting slot zero.
        bits = (0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 1, 0x8001)
        if case.dtype == "float16":
            bits = (0, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7C01, 1, 0x8001)
        values_cpu[0, 0, :8].copy_(torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(values_cpu.dtype))
    capacity = pages * case.block
    slots_cpu = (torch.randperm(capacity - 1)[: case.tokens] + 1).to(getattr(torch, case.slot_dtype))
    if case.tokens and case.slot_zero:
        slots_cpu[0] = 0
    if case.invalid == "mixed":
        slots_cpu[1::4] = -1
        slots_cpu[2::8] = capacity
        slots_cpu[3::8] = torch.iinfo(slots_cpu.dtype).min
        slots_cpu[-1] = torch.iinfo(slots_cpu.dtype).max
    elif case.invalid == "all":
        slots_cpu.fill_(-1)
    slot_stride = 2 if case.strided_slots else 1
    slot_storage = torch.full((1 + case.tokens * slot_stride,), -9, dtype=slots_cpu.dtype)
    slot_storage[1::slot_stride].copy_(slots_cpu)
    backing = initial.to(device)
    source_device, slot_device = source.to(device), slot_storage.to(device)
    return {
        "initial": initial,
        "backing": backing,
        "cache": backing.as_strided(shape, stride, prefix),
        "source_cpu": source,
        "source": source_device,
        "values": source_device.as_strided(values_cpu.shape, values_cpu.stride(), 7),
        "slot_cpu": slot_storage,
        "slot_storage": slot_device,
        "slots": slot_device[1::slot_stride],
    }


def reference(inputs):
    # CPU boolean indexing deliberately differs from both device algorithms.
    cache = inputs["cache"]
    expected = inputs["initial"].clone()
    view = expected.as_strided(cache.shape, cache.stride(), cache.storage_offset())
    slots = inputs["slots"].cpu().long()
    valid = (slots >= 0) & (slots < cache.shape[0] * cache.shape[1])
    assert slots[valid].unique().numel() == int(valid.sum()), "Valid destinations must be unique"
    values = inputs["values"].cpu().to(cache.dtype)
    view[slots[valid] // cache.shape[1], slots[valid] % cache.shape[1]] = values[valid]
    return expected


def byte_difference(actual, expected):
    actual, expected = actual.cpu().contiguous(), expected.cpu().contiguous()
    return int((actual.view(torch.uint8) != expected.view(torch.uint8)).sum())


def accuracy_metrics(actual, expected):
    actual = actual.cpu()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = (actual[finite].double() - expected[finite].double()).abs()
    return {
        "different_bytes": byte_difference(actual, expected),
        "max_abs_finite": float(difference.max()) if difference.numel() else 0.0,
        "nonfinite_mask_differences": int((torch.isfinite(actual) != torch.isfinite(expected)).sum()),
    }


def capture(fn, unroll):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(5):
            fn()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(unroll):
            fn()
    graph.replay()
    torch.npu.synchronize()
    return graph


def check_accuracy(case, fn, inputs, mode):
    graph = capture(fn, 1) if mode == "graph" and case.tokens else None
    assert byte_difference(inputs["source"], inputs["source_cpu"]) == 0, "Capture mutated source"
    assert byte_difference(inputs["slot_storage"], inputs["slot_cpu"]) == 0, "Capture mutated slots"
    records, outputs = [], []
    # Never use a previous output as the next expected cache.
    original_values, original_slots = inputs["values"].cpu(), inputs["slots"].cpu()
    for replay in range(REPLAYS if mode == "graph" else 1):
        inputs["values"].copy_(original_values.roll(replay, dims=0).roll(replay, dims=-1))
        # Different rotations change destinations as well as values. Rotating
        # both equally would preserve the mapping and could miss stale graphs.
        inputs["slots"].copy_(original_slots.roll(2 * replay, dims=0))
        inputs["backing"].copy_(inputs["initial"])
        source_before, slots_before = inputs["source"].cpu(), inputs["slot_storage"].cpu()
        expected = reference(inputs)
        graph.replay() if graph else fn()
        actual = inputs["backing"].cpu()
        record = accuracy_metrics(actual, expected)
        record["source_changed_bytes"] = byte_difference(inputs["source"], source_before)
        record["slots_changed_bytes"] = byte_difference(inputs["slot_storage"], slots_before)
        record["pass"] = all(
            record[key] == 0 for key in ("different_bytes", "source_changed_bytes", "slots_changed_bytes")
        )
        records.append(record)
        outputs.append(actual)
    inputs["values"].copy_(original_values)
    inputs["slots"].copy_(original_slots)
    inputs["backing"].copy_(inputs["initial"])
    return {"pass": all(record["pass"] for record in records), "replays": records}, outputs


def measure(fn, calls, unroll):
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    torch.npu.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    end.synchronize()
    return {
        "event_us": start.elapsed_time(end) * 1000 / calls / unroll,
        "wall_us": (time.perf_counter() - wall_start) * 1e6 / calls / unroll,
    }


def benchmark(functions, mode, args):
    unroll = args.unroll if mode == "graph" else 1
    graphs = [capture(fn, unroll) for fn in functions] if mode == "graph" else []
    calls = [graph.replay for graph in graphs] if graphs else functions
    for fn in calls:
        for _ in range(5):
            fn()
    samples = [[], []]
    for repeat in range(args.repeats):
        for side in (0, 1) if repeat % 2 == 0 else (1, 0):
            samples[side].append(measure(calls[side], args.iterations, unroll))
    old, new = [median(sample["event_us"] for sample in side) for side in samples]
    wall = [median(sample["wall_us"] for sample in side) for side in samples]
    if old <= 0 or new <= 0:
        raise RuntimeError("Timer resolution too low; increase --iterations/--unroll")
    return {
        f"{args.baseline}_us": old,
        "scatter_nd_us": new,
        "speedup": old / new,
        "change_percent": (new / old - 1) * 100,
        f"{args.baseline}_wall_us": wall[0],
        "scatter_nd_wall_us": wall[1],
        "faster_pairs": sum(b["event_us"] < a["event_us"] for a, b in zip(*samples)),
        "pairs": args.repeats,
        "unroll": unroll,
        "samples": dict(zip((args.baseline, "scatter_nd"), samples)),
    }


def save(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def run_worker(args):
    methods = (args.baseline, "scatter_nd")
    case = next(case for case in all_cases() if case.name == args.worker)
    path = args.output / f"{case.name}.json"
    result = {"case": asdict(case), "status": "running", "accuracy": {}, "performance": {}}
    save(path, result)
    try:
        torch.npu.set_device(args.device)
        result["runtime"] = {
            "device": torch.npu.get_device_name(args.device),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        }
        baseline = load_triton_reference() if args.baseline == "triton" else small_ops
        if args.baseline == "triton":
            result["runtime"]["triton_reference_sha256"] = TRITON_REFERENCE_SHA256
        inputs = [make_inputs(case, f"npu:{args.device}") for _ in methods]
        functions = [
            partial(writer, data["cache"], data["slots"], data["values"])
            for writer, data in zip((baseline, scatter_nd), inputs)
        ]
        for mode in args.modes:
            outputs = []
            result["accuracy"][mode] = {}
            for name, fn, data in zip(methods, functions, inputs):
                result["stage"] = f"{mode}/{name}/accuracy"
                save(path, result)
                record, actual = check_accuracy(case, fn, data, mode)
                result["accuracy"][mode][name] = record
                outputs.append(actual)
                save(path, result)
            result["accuracy"][mode]["pair_different_bytes"] = [byte_difference(a, b) for a, b in zip(*outputs)]
            passed = all(result["accuracy"][mode][name]["pass"] for name in methods)
            if passed and case.tokens and not case.special and not args.accuracy_only:
                result["stage"] = f"{mode}/performance"
                save(path, result)
                performance = benchmark(functions, mode, args)
                # Validate the timed graph too: its unroll is larger than the
                # graph used for changing-input accuracy checks.
                for data in inputs:
                    assert byte_difference(data["backing"], reference(data)) == 0, "Timed calls wrote wrong cache"
                    assert byte_difference(data["source"], data["source_cpu"]) == 0, "Timed calls mutated source"
                    assert byte_difference(data["slot_storage"], data["slot_cpu"]) == 0, "Timed calls mutated slots"
                result["performance"][mode] = performance
            else:
                result["performance"][mode] = {"skipped": "accuracy failed, empty/special case, or --accuracy-only"}
            save(path, result)
        result["status"] = (
            "pass"
            if all(mode[name]["pass"] for mode in result["accuracy"].values() for name in methods)
            else "accuracy_failed"
        )
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    save(path, result)
    return 0 if result["status"] == "pass" else 1


def write_summary(output, records, args):
    save(
        output / "results.json",
        {
            "baseline": args.baseline,
            "baseline_commit": BASELINE_COMMIT if args.baseline == "small_ops" else None,
            "triton_reference_sha256": TRITON_REFERENCE_SHA256 if args.baseline == "triton" else None,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "timing": {"iterations": args.iterations, "repeats": args.repeats, "unroll": args.unroll},
            "results": records,
        },
    )
    fields = (
        "case",
        "mode",
        "status",
        f"{args.baseline}_bytes",
        "scatter_nd_bytes",
        "pair_bytes",
        f"{args.baseline}_us",
        "scatter_nd_us",
        "change_percent",
        "speedup",
        "faster_pairs",
        "pairs",
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for mode in args.modes:
                accuracy = record.get("accuracy", {}).get(mode, {})
                row = {"case": record["case"]["name"], "mode": mode, "status": record["status"]}
                for name in (args.baseline, "scatter_nd"):
                    replays = accuracy.get(name, {}).get("replays", [])
                    row[f"{name}_bytes"] = max((r["different_bytes"] for r in replays), default="")
                row["pair_bytes"] = max(accuracy.get("pair_different_bytes", []), default="")
                performance = record.get("performance", {}).get(mode, {})
                row.update({key: value for key, value in performance.items() if key in fields})
                writer.writerow(row)


def run_parent(args):
    cases = all_cases()
    if args.case:
        unknown = set(args.case) - {case.name for case in cases}
        if unknown:
            raise SystemExit(f"Unknown cases: {sorted(unknown)}; use --list-cases")
        cases = [case for case in cases if case.name in args.case]
    elif args.suite == "smoke":
        names = {"t64_pad0_none", "t64_pad47616_mixed", "t4096_pad47616_none", "t4096_pad47616_mixed"}
        cases = [case for case in cases if case.name in names]
    if args.list_cases:
        for case in cases:
            print(json.dumps(asdict(case)))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in cases:
        print(f"START {case.name}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            case.name,
            "--device",
            str(args.device),
            "--output",
            str(args.output.resolve()),
            "--iterations",
            str(args.iterations),
            "--repeats",
            str(args.repeats),
            "--unroll",
            str(args.unroll),
            "--baseline",
            args.baseline,
            "--modes",
            *args.modes,
        ]
        if args.accuracy_only:
            command.append("--accuracy-only")
        with (args.output / f"{case.name}.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = "timeout"
        path = args.output / f"{case.name}.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"case": asdict(case)}
        record["exit_code"] = exit_code
        if exit_code != 0 and record.get("status") not in ("accuracy_failed", "error"):
            record["status"] = "timeout" if exit_code == "timeout" else "process_failed"
        records.append(record)
        write_summary(args.output, records, args)
        print(f"RESULT {case.name}: {record['status']}", flush=True)
        for mode, row in record.get("performance", {}).items():
            if "speedup" in row:
                print(
                    f"  {mode} ({args.baseline} -> scatter_nd): "
                    f"{row[f'{args.baseline}_us']:.3f} -> {row['scatter_nd_us']:.3f} us; "
                    f"{row['change_percent']:+.2f}%; {row['speedup']:.3f}x",
                    flush=True,
                )
    print(f"Saved {args.output.resolve() / 'summary.csv'}", flush=True)
    return 0 if all(record["status"] == "pass" for record in records) else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", type=int, default=0, help="Logical device after ASCEND_RT_VISIBLE_DEVICES filtering"
    )
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--baseline", choices=("small_ops", "triton"), default="small_ops")
    parser.add_argument("--case", nargs="+", help="Run exact names from --suite full --list-cases")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("eager", "graph"), default=["eager", "graph"])
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=8, help="Calls/replays per timed sample")
    parser.add_argument("--repeats", type=int, default=15, help="Alternating A/B sample pairs")
    parser.add_argument("--unroll", type=int, default=32, help="Complete writer calls per captured graph")
    parser.add_argument("--timeout", type=int, default=900, help="Per-case process timeout in seconds")
    parser.add_argument("--output", type=Path, default=Path("scatter_nd_comparison"), help="A new output directory")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.unroll, args.timeout) < 1 or args.device < 0:
        parser.error("Timing counts must be positive and device must be nonnegative")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        # Lazy imports isolate NPU runtimes in child processes. Listing cases,
        # CLI help, and result collection need only the Python standard library.
        import torch
        import torch_npu

        with torch.inference_mode():
            sys.exit(run_worker(arguments))
    sys.exit(run_parent(arguments))
