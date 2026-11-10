"""Isolate GPU contention between the serving reconstruction kernel and FFN."""

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slora.common.lowra_reconstruct_kv import lowra_reconstruct_kv_batched
from slora.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from overlap_analysis import analyze_trace, compare, compare_traces


MODES = ("reconstruct", "ffn", "serial_rf", "serial_fr", "overlap_rf", "overlap_fr")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[4608])
    parser.add_argument("--unique-loras", type=int, nargs="+", default=[1, 8, 20])
    parser.add_argument("--ffn-tokens", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--ranks", type=int, nargs="+", default=[16])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--row-stride", type=int, default=1,
                        help="2 leaves unused rows between prefix destinations")
    parser.add_argument("--launch-modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace", action="store_true", help="export separate traces for each paired mode/shape")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(*args.prefix_lengths, *args.unique_loras, *args.ffn_tokens, *args.ranks,
           args.hidden_size, args.intermediate_size, args.row_stride, args.warmup, args.iterations) <= 0:
        parser.error("all dimensions and repetition counts must be positive")
    if max(args.ranks) > args.hidden_size:
        parser.error("rank must not exceed hidden size")
    return args


def random_tensor(shape, device):
    return torch.empty(shape, device=device, dtype=torch.float16).normal_(0, 0.02)


class Workload:
    def __init__(self, args, prefix, count, tokens, rank):
        h, i, device = args.hidden_size, args.intermediate_size, args.device
        self.shape = dict(prefix_length=prefix, unique_loras=count, ffn_tokens=tokens,
                          rank=rank, hidden_size=h, intermediate_size=i, row_stride=args.row_stride)
        self.base = [random_tensor((prefix, h), device) for _ in range(2)]
        self.mini = [random_tensor((prefix, count * rank), device) for _ in range(2)]
        self.b = random_tensor((2 * count * rank, h), device)
        self.locations = (torch.arange(count * prefix, device=device) * args.row_stride).view(count, prefix)
        self.b_locations = torch.arange(2 * count * rank, device=device).view(2, count, rank)
        self.starts = torch.arange(count, device=device) * rank
        self.ranks = torch.full((count,), rank, device=device, dtype=torch.long)
        self.scales = torch.linspace(0.5, 1.5, count, device=device, dtype=torch.float16)
        self.outputs = [torch.full((count * prefix * args.row_stride, h), -7.0,
                                   device=device, dtype=torch.float16) for _ in range(2)]
        self.inputs = random_tensor((tokens, h), device)
        self.weights = SimpleNamespace(gate_proj=random_tensor((h, i), device),
                                       up_proj=random_tensor((h, i), device),
                                       down_proj=random_tensor((i, h), device))
        self.ffn_self = SimpleNamespace(embed_dim_=h)
        self.ffn_output = None

    def reconstruct(self):
        lowra_reconstruct_kv_batched(*self.base, *self.mini, self.b, *self.outputs,
                                    self.locations, *self.b_locations, self.starts,
                                    self.ranks, self.scales)

    def ffn(self):
        # Call the serving FFN, including its real allocation/operator sequence.
        self.ffn_output = LlamaTransformerLayerInfer._ffn(self.ffn_self, self.inputs, None, self.weights)

    def make_reference(self):
        self.ffn()
        self.ffn_reference = self.ffn_output.clone()
        p, u, r = (self.shape[key] for key in ("prefix_length", "unique_loras", "rank"))
        self.check_rows = torch.linspace(0, p - 1, min(p, 17), device=self.inputs.device).long().unique()
        self.prefix_reference = []
        for kv in range(2):
            expected = []
            for adapter in range(u):
                mini = self.mini[kv][self.check_rows, adapter * r:(adapter + 1) * r].float()
                b = self.b[self.b_locations[kv, adapter]].float()
                expected.append((self.base[kv][self.check_rows].float()
                                 + (mini @ b) * self.scales[adapter].float()).half())
            self.prefix_reference.append(torch.stack(expected))
        self.reconstruct()
        self.validate(True, True)

    def validate(self, reconstruction, ffn):
        if ffn:
            torch.testing.assert_close(self.ffn_output, self.ffn_reference, atol=1e-4, rtol=1e-3)
        if reconstruction:
            for output, expected in zip(self.outputs, self.prefix_reference):
                actual = output[self.locations[:, self.check_rows]]
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
                for offset in range(1, self.shape["row_stride"]):
                    if not bool((output[offset::self.shape["row_stride"]] == -7).all()):
                        raise AssertionError("reconstruction overwrote unused destination rows")


class Runner:
    def __init__(self, workload, mode, launch):
        self.workload, self.mode, self.launch = workload, mode, launch
        self.compute = torch.cuda.Stream()
        self.side = torch.cuda.Stream()
        self.events = {name: torch.cuda.Event(enable_timing=launch == "eager")
                       for name in ("origin", "rs", "re", "fs", "fe", "done")}
        self.graph_timers = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.has_r = mode != "ffn"
        self.has_f = mode != "reconstruct"
        self.graph = None
        torch.cuda.synchronize()
        if launch == "graph":
            # Warm cuBLAS, Triton, event handles and both streams outside capture.
            for _ in range(3):
                self.body()
                self.events["done"].synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.compute):
                self.body()
            self.graph_output = workload.ffn_output
            torch.cuda.synchronize()

    def branch(self, kind, stream):
        with torch.cuda.stream(stream):
            self.events[kind + "s"].record(stream)
            if kind == "r":
                self.workload.reconstruct()
            else:
                self.workload.ffn()
            self.events[kind + "e"].record(stream)

    def body(self):
        e, main, side = self.events, self.compute, self.side
        with torch.cuda.stream(main):
            e["origin"].record(main)
            if self.mode.startswith("overlap") or self.mode == "reconstruct":
                # Same fork/join as layer_prefetch; neither branch waits for the other.
                side.wait_event(e["origin"])
                if self.mode == "overlap_fr":
                    self.branch("f", main)
                self.branch("r", side)
                if self.mode == "overlap_rf":
                    self.branch("f", main)
                main.wait_event(e["re"])
            elif self.mode == "ffn":
                self.branch("f", main)
            else:
                for kind in self.mode[-2:]:
                    self.branch(kind, main)
            e["done"].record(main)

    def run(self):
        start = time.perf_counter()
        if self.graph is None:
            self.body()
        else:
            with torch.cuda.stream(self.compute):
                self.graph_timers[0].record(self.compute)
                self.graph.replay()
                self.graph_timers[1].record(self.compute)
            self.workload.ffn_output = self.graph_output
        submitted = time.perf_counter()
        done = self.events["done"] if self.graph is None else self.graph_timers[1]
        done.synchronize()
        finish = time.perf_counter()
        e = self.events
        total = (e["origin"].elapsed_time(e["done"]) if self.graph is None
                 else self.graph_timers[0].elapsed_time(self.graph_timers[1]))
        row = dict(total_ms=total,
                   host_enqueue_ms=(submitted - start) * 1000, wall_ms=(finish - start) * 1000)
        # Torch 2.0 turns captured event dependencies into graph edges. Their
        # event handles cannot reliably be timed/synchronized after replay.
        # Graph branch durations therefore come only from the kernel trace.
        if self.graph is not None:
            if total <= 0:
                raise RuntimeError("invalid CUDA graph timing")
            return row
        for kind, enabled, name in (("r", self.has_r, "reconstruct"), ("f", self.has_f, "ffn")):
            if enabled:
                row[name + "_ms"] = e[kind + "s"].elapsed_time(e[kind + "e"])
                row[name + "_start_ms"] = e["origin"].elapsed_time(e[kind + "s"])
                row[name + "_end_ms"] = e["origin"].elapsed_time(e[kind + "e"])
        if self.has_r and self.has_f:
            row["event_interval_overlap_ms"] = max(0.0, min(row["reconstruct_end_ms"], row["ffn_end_ms"])
                                                   - max(row["reconstruct_start_ms"], row["ffn_start_ms"]))
        if row["total_ms"] <= 0 or any(row[name + "_ms"] <= 0 for name, yes in
                                       (("reconstruct", self.has_r), ("ffn", self.has_f)) if yes):
            raise RuntimeError("invalid CUDA event timing (including graph capture); do not use these results")
        return row


def capture_trace(runner, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.profiler.record_function("overlap_diagnostic/" + runner.mode):
            runner.run()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return dict(path=str(path), **analyze_trace(json.loads(path.read_text())))


def gpu_snapshot():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                                        "--format=csv"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    rows = []
    for row in report["rows"]:
        flat = dict(**row["shape"], launch=row["launch"])
        flat.update({mode + "_median_ms": row["timings_ms"][mode]["total_ms"]["median"] for mode in MODES})
        flat.update({"speedup_" + order: row["comparisons"][order]["serial_over_parallel_speedup"]
                     for order in ("rf", "fr")})
        rows.append(flat)
    if rows:
        temporary = path.with_name("summary.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path.with_name("summary.csv"))


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with the slora environment and a visible GPU")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "result.json"
    if destination.exists():
        raise FileExistsError("use a new --output-dir; result.json already exists")
    paths = [Path(__file__), Path(__file__).with_name("overlap_analysis.py"),
             ROOT / "slora/common/lowra_reconstruct_kv.py",
             ROOT / "slora/models/llama/layer_infer/transformer_layer_infer.py"]
    report = dict(status="running", torch=torch.__version__, triton=triton.__version__,
                  gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                  gpu_before=gpu_snapshot(),
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope="Single-layer GPU-resident synthetic FP16 data; real fused reconstruction and serving FFN core. "
                        "Not full serving: excludes attention, norm/residual, O/QKV, admission and CPU transfers. "
                        "Event intervals include scheduling/launch gaps; only trace kernel intersections prove overlap. "
                        "Hardware bottleneck attribution requires hardware counters, not these timings alone.", rows=[])
    write_report(destination, report)
    shapes = list(itertools.product(args.prefix_lengths, args.unique_loras, args.ffn_tokens, args.ranks))
    try:
        for index, (prefix, count, tokens, rank) in enumerate(shapes):
            torch.manual_seed(args.seed + index)
            workload = Workload(args, prefix, count, tokens, rank)
            workload.make_reference()
            torch.cuda.synchronize()
            for launch in args.launch_modes:
                runners = {mode: Runner(workload, mode, launch) for mode in MODES}
                samples = {mode: [] for mode in MODES}
                for runner in runners.values():
                    for _ in range(args.warmup):
                        runner.run()
                    if runner.has_r:
                        for output in workload.outputs:
                            output.fill_(-7)
                    if runner.has_f and runner.graph is not None:
                        runner.graph_output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    runner.run()
                    workload.validate(runner.has_r, runner.has_f)
                rng = random.Random(args.seed + index)
                for iteration in range(args.iterations):
                    order = list(MODES)
                    rng.shuffle(order)
                    for mode in order:
                        samples[mode].append(runners[mode].run())
                row = dict(shape=workload.shape, launch=launch, correctness_passed=True,
                           correctness_scope="All FFN outputs; 17 prefix positions per adapter, all columns, K and V; "
                                             "unused destination rows guarded; outputs poisoned before checking",
                           reconstruction_gflops=4 * count * prefix * rank * args.hidden_size / 1e9,
                           ffn_gflops=6 * tokens * args.hidden_size * args.intermediate_size / 1e9,
                           logical_prefix_output_bytes=4 * count * prefix * args.hidden_size,
                           raw_samples=samples, **compare(samples))
                # Run after all timed samples, without profiler instrumentation in the measurements.
                if args.trace:
                    row["traces"] = {}
                    for mode in MODES:
                        name = "p%s_u%s_t%s_r%s_%s_%s.json" % (prefix, count, tokens, rank, launch, mode)
                        row["traces"][mode] = capture_trace(runners[mode], args.output_dir / name)
                    row["kernel_trace_comparisons"] = compare_traces(row["traces"])
                report["rows"].append(row)
                write_report(destination, report)
                print(json.dumps(dict(shape=workload.shape, launch=launch, comparisons=row["comparisons"])), flush=True)
                # Graphs own activation buffers. Release each shape's captures before moving on.
                del runner, runners
                workload.ffn_output = None
                gc.collect()
                torch.cuda.synchronize()
            del workload
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["gpu_after"] = gpu_snapshot()
        write_report(destination, report)
    print("Results: %s" % destination, flush=True)


if __name__ == "__main__":
    main(parse_args())
