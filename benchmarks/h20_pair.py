"""Run the existing prefix-source plan on a selected pair of physical H20 GPUs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
import os
import shlex
import subprocess
import time

import h20_runtime as h20
from compare_prefix_sources import compare as compare_modes


FOUR_METHODS = ("active-reprefill", "minikv-full_reconstruct", "cpu-read", "cpu-read-fixed1-r64")


def replay_env(env):
    return dict(env, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")


def prepare_point(dest, trace, env):
    dest.mkdir(parents=True, exist_ok=False)
    temporary = dest / ".trace.pending.json"
    cmd = list(trace)
    cmd[cmd.index("--output") + 1] = str(temporary)
    with (dest / ".trace.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(cmd, cwd=h20.ROOT, env=env, check=True)
        os.replace(temporary, dest / "trace.json")


def pair_plan(launcher, env):
    lines = subprocess.check_output(launcher + ["--dry-run"], cwd=h20.ROOT, env=env, text=True)
    trace_commands = []
    groups = [[], []]
    side = None
    for line in lines.splitlines():
        if line.startswith("CUDA_VISIBLE_DEVICES="):
            side = int(line.split("=", 1)[1].split(";")[0])
        words = shlex.split(line) if line.startswith("/") else []
        if words[:2] != [env["SLORA_PYTHON"], "-u"]:
            continue
        if words[2] == "benchmarks/prepare_lmsys_maf_trace.py":
            trace_commands.append(words)
        elif words[2] == "benchmarks/lowra_saturation_exp.py":
            groups[side].append(words)
    if (len(trace_commands) != 2 or trace_commands[0] != trace_commands[1]
            or [len(group) for group in groups] != [3, 2]):
        raise RuntimeError("Expected a shared trace and five prefix-source modes")
    return trace_commands[0], groups


def wait_idle(gpu, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                                  "--format=csv,noheader,nounits"], text=True).strip():
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(f"GPU {gpu} is busy")
        time.sleep(2)


def run_group(gpu, uuid, commands, dest, env):
    env = dict(env, CUDA_VISIBLE_DEVICES=uuid, SLORA_H20_PHYSICAL_GPU=str(gpu))
    with (dest / f"gpu{gpu}.log").open("x") as log:
        for cmd in commands:
            wait_idle(gpu, timeout=60)
            print(f"GPU {gpu}: {cmd[cmd.index('--mode') + 1]}", file=log, flush=True)
            subprocess.run(cmd, cwd=h20.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        wait_idle(gpu, timeout=60)


def run(launcher, env, dest, gpus, dry_run=False, start_now=False, four_methods=False):
    env = replay_env(env)
    trace, groups = pair_plan(launcher, env)
    if four_methods:
        by_mode = {cmd[cmd.index("--mode") + 1]: cmd for group in groups for cmd in group}
        groups = ([by_mode[m] for m in FOUR_METHODS[:3]], [by_mode[FOUR_METHODS[3]]])
    compare = [env["SLORA_PYTHON"], "benchmarks/compare_prefix_sources.py", str(dest),
               "--fixed-mode", "cpu-read-fixed1-r64"]
    if dry_run:
        print(shlex.join(h20.command(trace[1:])))
        for gpu, commands in zip(gpus, groups):
            print(f"Physical GPU {gpu}")
            for cmd in commands:
                print(shlex.join(h20.command(cmd[1:], plan_gpu=gpu)))
        if four_methods:
            print(f"Compare the four selected methods in {dest}")
        else:
            print(shlex.join(h20.command(compare[1:])))
        return

    results = h20.ROOT / "benchmarks/results"
    results.mkdir(exist_ok=True)
    with ExitStack() as locks:
        for gpu in sorted(gpus):
            lock = locks.enter_context((results / f"lmsys_preemption_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if start_now else 0))
        uuids = []
        for gpu in gpus:
            wait_idle(gpu, timeout=0 if start_now else None)
            uuids.append(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                                  "--format=csv,noheader"], text=True).strip())
        # Exclusive directory creation also respects an earlier legacy waiter's claim.
        prepare_point(dest, trace, env)
        if (dest / "waiter.pid").exists():
            raise RuntimeError(f"Legacy launcher already owns {dest}")
        with ThreadPoolExecutor(max_workers=2) as workers:
            jobs = [workers.submit(run_group, gpu, uuid, commands, dest, env)
                    for gpu, uuid, commands in zip(gpus, uuids, groups)]
            for job in jobs:
                job.result()
        if four_methods:
            summary = compare_modes(dest, modes=FOUR_METHODS)
            pending = dest / ".comparison.pending.json"
            pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            pending.replace(dest / "comparison.json")
            print(dest / "comparison.json", flush=True)
        else:
            subprocess.run(compare, cwd=h20.ROOT, env=env, check=True)
