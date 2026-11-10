"""Run paired retention or matched-kernel serial/prefetch experiments on one GPU."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.lowra_saturation_exp import write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=("retention", "overlap", "prefetch-top1"), default="retention",
                        help="overlap: serial/prefetch; prefetch-top1: prefetch/dynamic Top-1, with bounded kernels.")
    parser.add_argument("--reconstruction-workers", type=int, default=1296)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--port", type=int, default=8270)
    parser.add_argument("--nccl-port", type=int, default=30270)
    parser.add_argument("--internal-port-start", type=int, default=18610)
    parser.add_argument("--dry-run", action="store_true", help="Print both commands without starting servers.")
    args = parser.parse_args(argv)
    if args.output_len <= 0 or args.reconstruction_workers <= 0:
        parser.error("output-len and reconstruction-workers must be positive")
    return args


def build_commands(args):
    root = args.output_dir.resolve()
    common = ["--model-dir", "huggyllama/llama-7b", "--adapter-base", "dummy-lora-7b-rank-16",
              "--num-adapters", "50", "--prefix-len", "4608", "--query-len", "480",
              "--output-len", str(args.output_len), "--max-context-len", "10000",
              "--num-token", "36864", "--batch-max-tokens", "36864", "--running-max-req-size", "256",
              "--request-rates", "8", "--duration", "120", "--warmup", "100", "--steady-end", "180",
              "--alpha", "1.0", "--cv", "1.0", "--seeds", "42", "--model-seed", "42",
              "--request-timeout", "10800", "--drain-timeout", "10800", "--startup-timeout", "1800",
              "--port", str(args.port), "--nccl-port", str(args.nccl_port),
              "--internal-port-start", str(args.internal_port_start),
              "--trace-dir", str(root / "traces"), "--measure-saturation", "--dummy"]
    if args.comparison in ("overlap", "prefetch-top1"):
        common += ["--reconstruction-kernel", "bounded",
                   "--reconstruction-workers", str(args.reconstruction_workers)]
        if args.comparison == "overlap":
            methods = (("serial_bounded", "minikv-layer_serial", []),
                       ("parallel_bounded", "layer_prefetch", []))
        else:
            methods = (("parallel_bounded", "layer_prefetch", []),
                       ("dynamic_top1_bounded", "minikv-layer_retained",
                        ["--retain-layers", "32", "--retain-top-k", "1"]))
    else:
        methods = (("serial_all", "minikv-layer_serial", []),
                   ("dynamic_top1", "minikv-layer_retained",
                    ["--retain-layers", "32", "--retain-top-k", "1"]))
    return {name: [sys.executable, "-u", str(ROOT / "benchmarks/lowra_saturation_exp.py"),
                   "--mode", mode, "--output-dir", str(root / name)] + extra + common
            for name, mode, extra in methods}


def main():
    args = parse_args()
    commands = build_commands(args)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    root = args.output_dir.resolve()
    if (root / "launcher.json").exists():
        raise FileExistsError("output already contains a run; choose a fresh --output-dir")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "launcher.json", dict(pid=os.getpid(), comparison=args.comparison,
               commands=commands, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    child = None

    def interrupt(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for name, command in commands.items():
            print("Starting %s: decode=%s, complete-cycle capacity measurement" % (name, args.output_len), flush=True)
            child = subprocess.Popen(command, cwd=ROOT)
            code = child.wait()
            if code:
                raise RuntimeError("%s exited with status %s" % (name, code))
            child = None
        points = {name: json.loads((root / name / "rate8_seed42.json").read_text())
                  for name in commands}
        if len({point["trace_sha256"] for point in points.values()}) != 1:
            raise RuntimeError("comparison request traces do not match")
        comparison = dict(comparison=args.comparison,
                          trace_sha256=next(iter(points.values()))["trace_sha256"], methods={})
        for name, point in points.items():
            result = point["result"]
            comparison["methods"][name] = {key: result[key] for key in
                ("total_requests", "completed_requests", "failed_requests", "benchmark_time",
                 "completed_req_per_s_total", "completed_req_per_s_steady",
                 "completed_req_per_s_saturated", "completed_req_per_s_peak",
                 "latency_mean", "ttft_mean", "saturation")}
            comparison["methods"][name]["valid"] = result["valid"]
            if args.comparison in ("overlap", "prefetch-top1"):
                configs = point["runtime"]["reconstruction_configurations"]
                expected = [dict(kernel="bounded", workers=args.reconstruction_workers)]
                if configs != expected:
                    raise RuntimeError("%s used unexpected reconstruction configuration: %s" % (name, configs))
                comparison["methods"][name]["reconstruction_configurations"] = configs
                comparison["methods"][name]["peak_allocated_without_base_model_bytes"] = point["runtime"][
                    "peak_allocated_without_base_model_bytes"]
                comparison["methods"][name]["prefix_cache_stats"] = point["runtime"].get("prefix_cache_stats")
        write_json(root / "comparison.json", comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        print("Finished both methods: %s" % (root / "comparison.json"), flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
