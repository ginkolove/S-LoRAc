"""Replay V3, optionally paired with the original V2, on one physical H20."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
from statistics import mean
import subprocess

import run_h20_adaptive_v2 as v2
from benchmarks.h20_adaptive_v3 import OBJECTIVE, POLICY, POLICY_REVISION
import h20_runtime as h20
from h20_pair import replay_env, wait_idle
from run_h20_rankmix_suite import ADAPTIVE_POINTS
from slora.common.space_plan_config import space_plan_options


def stage_v3_source(source):
    v2.stage_v2_source(source)
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "h20_adaptive_v3.py", source / "slora/server/router")
    shutil.copy2(here / "h20_adaptive_v3_runtime.py", source / "slora/common")
    subprocess.run(["git", "apply", "--no-index", "-"], cwd=source,
                   input=(here / "h20_adaptive_v3.patch").read_text(), text=True, check=True,
                   env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(source.parent)))


def paired_summary(dest):
    methods = {}
    reference = None
    for mode in (v2.MODE, POLICY):
        path = dest / mode / "rate3_seed42.json"
        point = json.loads(path.read_text())
        result = point["result"]
        if (point["mode"] != mode or not result["valid"] or point["reset"]["status"] != "confirmed"
                or result["total_requests"] != result["completed_requests"] or not point["trace_sha256"]):
            raise ValueError(f"Incomplete replay: {path}")
        with path.with_name("rate3_seed42_requests.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        workload = {r["req_id"]: (r["scheduled_time"], r["source_id"], r["adapter_dir"],
                                  r["query_tokens"], r["output_tokens"]) for r in rows}
        if (len(rows) != len(workload) or len(workload) != result["completed_requests"]
                or not all(r["success"] and r["output_tokens"] == r["expected_output_tokens"] for r in rows)):
            raise ValueError(f"Request logs do not match completed replay: {path}")
        signature = (point["trace_sha256"], workload)
        if reference is not None and signature != reference:
            raise ValueError(f"Different traces or completed workloads in {dest}")
        reference = signature
        metrics = {k: result[k] for k in ("completed_requests", "benchmark_time", "completed_req_per_s_total",
                                          "ttft_mean", "ttft_p95", "latency_mean", "latency_p95")}
        tpot = [(r["end_time"] - r["first_token_time"]) * 1000. / (r["output_tokens"] - 1)
                for r in rows if r["output_tokens"] > 1]
        metrics["mean_request_tpot_ms"] = mean(tpot) if tpot else None
        metrics["peak_output_window_10s"] = result.get("peak_output_window_10s")
        methods[mode] = metrics
    ratio = methods[POLICY]["completed_req_per_s_total"] / methods[v2.MODE]["completed_req_per_s_total"]
    return dict(trace_sha256=reference[0], methods=methods, v3_throughput_change_pct=100. * (ratio - 1.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="parent for new per-point result directories")
    parser.add_argument("--gpu", type=int, choices=range(8), required=True, help="physical H20 index")
    parser.add_argument("--point", nargs="+", choices=ADAPTIVE_POINTS, metavar="MODEL:PREFIX", required=True)
    parser.add_argument("--compare-v2", nargs="?", const="v2-first", choices=("v2-first", "v3-first"),
                        help="run both versions sequentially on the same GPU and compare completed traces")
    parser.add_argument("--space-plan-window", type=int, default=8,
                        help="upper bound on V3 layout reuse in decode steps (default 8)")
    parser.add_argument("--space-plan-interval", type=int, default=10, help="both versions' profiling interval")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(space_plan_hysteresis=0., space_plan_max_step_ms=0., space_plan_max_wait_ms=0.)
    args = parser.parse_args()
    if len(set(args.point)) != len(args.point):
        parser.error("--point entries must be distinct")
    try:
        space_plan_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    modes = [POLICY]
    if args.compare_v2:
        modes = [v2.MODE, POLICY] if args.compare_v2 == "v2-first" else [POLICY, v2.MODE]
    shim = str(Path(h20.__file__).resolve())
    env = replay_env(dict(os.environ, SLORA_PYTHON=shim, SLORA_TRACE_PYTHON=shim,
                          SLORA_H20_PHYSICAL_GPU=str(args.gpu)))
    jobs = {}
    output = args.output.resolve()
    for point in args.point:
        size, prefix = ADAPTIVE_POINTS[point]
        dest = output / f"{size}_p{prefix}"
        if dest.exists() and not args.dry_run:
            parser.error(f"Point already exists: {dest}. Choose a fresh output directory.")
        jobs[dest] = []
        for mode in modes:
            options = argparse.Namespace(**vars(args))
            if mode == v2.MODE:
                options.space_plan_window = 1
            command = v2.build_command(dest, size, prefix, env, options)
            command[command.index("--mode") + 1] = mode
            command[command.index("--output-dir") + 1] = str(dest / mode)
            source = dest / ("source_v2" if mode == v2.MODE else "source_v3")
            jobs[dest].append((mode, source, command, space_plan_options(options)))
            if args.dry_run:
                print(f"Physical GPU {args.gpu} {point} {mode} cwd={source}")
                print(shlex.join(h20.command(command[1:], plan_gpu=args.gpu)))
    if args.dry_run:
        return

    info = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name,uuid",
                                    "--format=csv,noheader"], text=True).strip()
    name, uuid = (field.strip() for field in info.split(","))
    if "H20" not in name:
        parser.error(f"Expected an H20 on GPU {args.gpu}, found {name}")
    h20.prepare(sorted({ADAPTIVE_POINTS[p][0] for p in args.point}))
    output.mkdir(parents=True, exist_ok=True)
    lock_path = h20.ROOT / "benchmarks/results" / f"lmsys_preemption_gpu{args.gpu}.lock"
    with lock_path.open("a") as lock:
        print(f"Waiting for GPU {args.gpu} ({uuid})", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for dest, runs in jobs.items():
            dest.mkdir(exist_ok=False)
            for mode, source, command, options in runs:
                wait_idle(args.gpu)
                (v2.stage_v2_source if mode == v2.MODE else stage_v3_source)(source)
                launch = dict(mode=mode, policy_revision=v2.POLICY_REVISION if mode == v2.MODE else POLICY_REVISION,
                              admission_policy="fcfs_max_feasible",
                              objective="fixed_batch_steady_decode" if mode == v2.MODE else OBJECTIVE,
                              physical_gpu=args.gpu, gpu_uuid=uuid, source=str(source), space_plan=options,
                              trace=command[command.index("--workload-trace") + 1], command=command)
                (dest / f"launch_{mode}.json").write_text(json.dumps(launch, indent=2) + "\n")
                run_env = dict(env, CUDA_VISIBLE_DEVICES=uuid,
                               PYTHONPATH=os.pathsep.join(filter(None, [str(source), env.get("PYTHONPATH")])))
                print(f"GPU {args.gpu} START {dest.name} {mode}", flush=True)
                subprocess.run(command, cwd=source, env=run_env, check=True)
                print(dest / mode / "rate3_seed42.json", flush=True)
            if args.compare_v2:
                summary = paired_summary(dest)
                pending = dest / ".comparison.pending.json"
                pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
                pending.replace(dest / "comparison.json")
                print(f"{dest.name} V3 throughput change {summary['v3_throughput_change_pct']:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
