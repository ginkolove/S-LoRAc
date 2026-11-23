"""Replay one physical H20's assigned prefix points using the frozen HF rank mix."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess

import h20_runtime as h20
from compare_prefix_sources import compare
from h20_pair import pair_plan, replay_env, wait_idle
from lmsys_maf_trace import HF_PROFILE, load_trace, trace_profile
from run_h20_minikv_actual import stage_source
from run_prefix_sources import point_launcher


RESULTS = h20.ROOT / "benchmarks/results"
TRACE_ROOT = RESULTS / "h20_hf_llama7b_rankmix_120s_traces"
GPU_POINTS = {
    0: ("7b", (5000, 8000, 15000)),
    1: ("7b", (10000, 12000)),
    2: ("13b", (5000, 8000)),
    3: ("13b", (10000,)),
    4: ("30b", (1000, 2000)),
    5: ("30b", (3000,)),
}
RECONSTRUCTION = ("minikv-full_reconstruct", "minikv-layer_serial")
MODES = ("active-reprefill", "cpu-read", "cpu-read-fixed1-r64", *RECONSTRUCTION)
ADAPTIVE = "minikv-adaptive"
ADAPTIVE_POINTS = {f"{size}:{p}": (size, p) for size, prefixes in GPU_POINTS.values() for p in prefixes}
RANK_SPLIT_MODES = {f"minikv-serial-le{r}-full-gt{r}": r for r in (16, 32)}
RANK_SPLIT_POINTS = {
    f"{size}:{prefix}:{limit}": (size, prefix, mode)
    for size, prefixes in GPU_POINTS.values() for prefix in prefixes
    for mode, limit in RANK_SPLIT_MODES.items()
}


def point_commands(dest, size, prefix, env, *, rank_split=False, adaptive=False):
    trace_path = TRACE_ROOT / dest.name / "trace.json"
    trace, _ = load_trace(trace_path)
    expected = dict(num_adapters=50, prefix_len=prefix, duration=120, request_rate=3,
                    seed=42, data_seed=42, tokenizer=str(h20.model_dir(size)))
    if trace_profile(trace) != HF_PROFILE or any(trace["config"][k] != v for k, v in expected.items()):
        raise ValueError(f"Frozen trace does not match the requested rank mix or point: {trace_path}")
    _, groups = pair_plan(point_launcher(dest, size, prefix, 120, profile=HF_PROFILE), env)
    by_mode = {cmd[cmd.index("--mode") + 1]: cmd for group in groups for cmd in group}
    commands = []
    modes = (ADAPTIVE,) if adaptive else (RANK_SPLIT_MODES if rank_split else MODES)
    for mode in modes:
        if rank_split or adaptive:
            cmd = by_mode["minikv-layer_serial"].copy()
            cmd[cmd.index("--mode") + 1] = mode
            cmd[cmd.index("--output-dir") + 1] = str(dest / mode)
            if rank_split:
                cmd.append("--reuse-next-reconstructed")
        else:
            cmd = by_mode[mode]
        cmd[cmd.index("--workload-trace") + 1] = str(trace_path)
        commands.append((mode, cmd))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="common results directory for the selected queues")
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--dry-run", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--rank-split", action="store_true",
                           help="run both MiniKV rank cutoffs at this GPU's assigned prefix lengths")
    selection.add_argument("--rank-split-point", choices=RANK_SPLIT_POINTS, metavar="MODEL:PREFIX:CUTOFF",
                           help="run one MiniKV cutoff on the selected GPU, e.g. 30b:1000:16")
    selection.add_argument("--adaptive", action="store_true",
                           help="run adaptive MiniKV actual/16 at this GPU's assigned prefix lengths")
    selection.add_argument("--adaptive-point", choices=ADAPTIVE_POINTS, metavar="MODEL:PREFIX",
                           help="run one adaptive MiniKV point on the selected GPU, e.g. 7b:5000")
    args = parser.parse_args()
    gpu = args.gpu
    selected_mode = None
    adaptive = args.adaptive or args.adaptive_point is not None
    if args.adaptive_point:
        size, prefix = ADAPTIVE_POINTS[args.adaptive_point]
        prefixes = (prefix,)
    elif args.rank_split_point:
        size, prefix, selected_mode = RANK_SPLIT_POINTS[args.rank_split_point]
        prefixes = (prefix,)
    elif gpu in GPU_POINTS:
        size, prefixes = GPU_POINTS[gpu]
    else:
        parser.error(f"GPU {gpu} has no default queue; select --adaptive-point or --rank-split-point")
    rank_split = args.rank_split or selected_mode is not None
    output = args.output.resolve()
    source = output / f"gpu{gpu}_actual16_source"
    shim = str(Path(h20.__file__).resolve())
    env = replay_env(dict(os.environ, SLORA_PYTHON=shim, SLORA_TRACE_PYTHON=shim,
                          SLORA_H20_PHYSICAL_GPU=str(gpu)))
    plan = {}
    for prefix in prefixes:
        dest = output / f"{size}_p{prefix}"
        commands = point_commands(dest, size, prefix, env, rank_split=rank_split, adaptive=adaptive)
        plan[dest] = [(selected_mode, dict(commands)[selected_mode])] if selected_mode else commands

    if args.dry_run:
        for dest, commands in plan.items():
            for mode, cmd in commands:
                cwd = source if adaptive or rank_split or mode in RECONSTRUCTION else h20.ROOT
                print(f"GPU {gpu} {dest.name} {mode} cwd={cwd}")
                print(shlex.join(h20.command(cmd[1:], plan_gpu=gpu)))
        return

    info = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=name,uuid",
                                    "--format=csv,noheader"], text=True).strip()
    name, uuid = (part.strip() for part in info.split(","))
    if "H20" not in name:
        parser.error(f"Expected H20 on GPU {gpu}, found {name}")
    env["CUDA_VISIBLE_DEVICES"] = uuid
    h20.prepare([size])
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    with (RESULTS / f"lmsys_preemption_gpu{gpu}.lock").open("a") as lock:
        print(f"GPU {gpu} ({uuid}) waiting for lock and idle device", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait_idle(gpu)
        stage_source(source, 16, rank_split=rank_split, adaptive=adaptive)
        for dest, commands in plan.items():
            dest.mkdir(exist_ok=selected_mode is not None)
            if selected_mode:
                (dest / selected_mode).mkdir(exist_ok=False)
            failed = False
            for mode, cmd in commands:
                wait_idle(gpu, timeout=60)
                cwd = source if adaptive or rank_split or mode in RECONSTRUCTION else h20.ROOT
                print(f"GPU {gpu} START {dest.name} {mode}", flush=True)
                rc = subprocess.run(cmd, cwd=cwd, env=env).returncode
                print(f"GPU {gpu} EXIT {dest.name} {mode} rc={rc}", flush=True)
                if rc:
                    failures.append(f"{dest.name}/{mode}")
                    failed = True
            if not failed and rank_split:
                # Independent GPUs share a prefix directory. The last finisher compares both.
                if selected_mode and not all((dest / mode / "rate3_seed42.json").is_file()
                                             for mode in RANK_SPLIT_MODES):
                    continue
                try:
                    summary = compare(dest, modes=RANK_SPLIT_MODES)
                    for mode, cutoff in RANK_SPLIT_MODES.items():
                        if summary["methods"][mode]["cache"]["serial_max_rank"] != cutoff:
                            raise ValueError(f"{mode} runtime reported an unexpected rank cutoff")
                    pending = dest / f".comparison_gpu{gpu}.tmp"
                    pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
                    pending.replace(dest / "comparison.json")
                except (KeyError, ValueError) as exc:
                    print(f"GPU {gpu} {dest.name} comparison failed: {exc}", flush=True)
                    failures.append(f"{dest.name}/comparison")
            elif not failed and not adaptive:
                compare_command = [shim, "benchmarks/compare_prefix_sources.py", str(dest),
                           "--fixed-mode", "cpu-read-fixed1-r64"]
                if subprocess.run(compare_command, cwd=h20.ROOT, env=env).returncode:
                    failures.append(f"{dest.name}/comparison")
    if failures:
        raise SystemExit(f"GPU {gpu} failed: {', '.join(failures)}")
    print(f"GPU {gpu} DONE", flush=True)


if __name__ == "__main__":
    main()
