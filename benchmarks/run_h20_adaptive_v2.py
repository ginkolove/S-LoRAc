"""Run FCFS/local-layout adaptive-v2 on one physical H20 and frozen trace."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h20_runtime as h20
from h20_pair import replay_env, wait_idle
from run_h20_minikv_actual import stage_source
from run_h20_rankmix_suite import ADAPTIVE_POINTS, point_commands
from slora.common.space_plan_config import add_space_plan_arguments, space_plan_command, space_plan_options


MODE = "minikv-adaptive-v2"
POLICY_REVISION = "fcfs-local-layout-v1"
PATCH = Path(__file__).with_name("h20_adaptive_v2.patch")


def stage_v2_source(source):
    """Register v2 only in this run's snapshot; preserve the original methods."""
    stage_source(source, 16, adaptive=True)
    shutil.copy2(PATCH.with_suffix(".py"), source / "slora/server/router/h20_adaptive_v2.py")
    subprocess.run(["git", "apply", "--no-index", "-"], cwd=source,
                   input=PATCH.read_text(), text=True, check=True,
                   env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(source.parent)))


def build_command(dest, size, prefix, env, args):
    (_, command), = point_commands(dest, size, prefix, env, adaptive=True)
    command[command.index("--mode") + 1] = MODE
    command[command.index("--output-dir") + 1] = str(dest / MODE)
    command.extend(space_plan_command(args))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Planning is triggered by batch changes or memory pressure. interval controls "
        "profiling and hysteresis only controls layout changes. Legacy window, group, "
        "beam, candidates, budget-ms, max-wait-ms and max-step-ms options do not gate admission."))
    parser.add_argument("output", type=Path, help="new common results directory")
    parser.add_argument("--gpu", type=int, choices=range(8), required=True,
                        help="physical nvidia-smi GPU index")
    parser.add_argument("--point", choices=ADAPTIVE_POINTS, required=True, metavar="MODEL:PREFIX")
    parser.add_argument("--dry-run", action="store_true")
    add_space_plan_arguments(parser)
    parser.set_defaults(space_plan_window=1, space_plan_max_wait_ms=0.,
                        space_plan_max_step_ms=0., space_plan_hysteresis=0.)
    args = parser.parse_args()
    try:
        options = space_plan_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    size, prefix = ADAPTIVE_POINTS[args.point]
    output = args.output.resolve()
    dest = output / f"{size}_p{prefix}"
    source = output / f"gpu{args.gpu}_adaptive_v2_source"
    shim = str(Path(h20.__file__).resolve())
    env = replay_env(dict(os.environ, SLORA_PYTHON=shim, SLORA_TRACE_PYTHON=shim,
                          SLORA_H20_PHYSICAL_GPU=str(args.gpu)))
    # The service, worker subprocesses and benchmark must import this snapshot,
    # even if the caller's environment has the working checkout on PYTHONPATH.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(source), env.get("PYTHONPATH")]))
    command = build_command(dest, size, prefix, env, args)
    if args.dry_run:
        print(f"Physical GPU {args.gpu}: {args.point} {MODE} ({POLICY_REVISION}); cwd={source}")
        print(shlex.join(h20.command(command[1:], plan_gpu=args.gpu)))
        return
    if dest.exists() or source.exists():
        parser.error("point or source directory already exists; choose a new output directory")
    info = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name,uuid",
                                    "--format=csv,noheader"], text=True).strip()
    name, uuid = (part.strip() for part in info.split(","))
    if "H20" not in name:
        parser.error(f"Expected H20 on physical GPU {args.gpu}, found {name}")
    env["CUDA_VISIBLE_DEVICES"] = uuid
    h20.prepare([size])
    output.mkdir(parents=True, exist_ok=True)
    lock_path = h20.ROOT / "benchmarks/results" / f"lmsys_preemption_gpu{args.gpu}.lock"
    with lock_path.open("a") as lock:
        print(f"GPU {args.gpu} ({uuid}) waiting for lock and idle device", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait_idle(args.gpu)
        dest.mkdir(exist_ok=False)
        stage_v2_source(source)
        (dest / "launch.json").write_text(json.dumps(dict(
            mode=MODE, policy_revision=POLICY_REVISION,
            admission_policy="fcfs_max_feasible", objective="fixed_batch_steady_decode",
            physical_gpu=args.gpu, gpu_uuid=uuid, source=str(source),
            trace=command[command.index("--workload-trace") + 1],
            space_plan=options, command=command), indent=2) + "\n")
        print(f"GPU {args.gpu} START {dest.name} {MODE} ({POLICY_REVISION})", flush=True)
        subprocess.run(command, cwd=source, env=env, check=True)
    print(f"GPU {args.gpu} DONE: {dest / MODE / 'rate3_seed42.json'}", flush=True)


if __name__ == "__main__":
    main()
