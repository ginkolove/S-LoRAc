"""Replay MiniKV reconstruction with actual-rank loops on one H20 GPU."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from h20_pair import pair_plan, replay_env, wait_idle
from lmsys_maf_trace import trace_profile, load_trace
import h20_runtime as h20
from run_prefix_sources import MATRIX, point_launcher


MODES = {"full": "minikv-full_reconstruct", "serial": "minikv-layer_serial"}
PATCH = Path(__file__).with_name("h20_actual_rank.patch.in")


def stage_source(source, block_r, *, rank_split=False, adaptive=False):
    """Keep the A100 checkout intact and retain the exact H20 sources beside each run."""
    if adaptive and (block_r != 16 or rank_split):
        raise ValueError("the adaptive layout uses its own actual/16 runtime")
    source.mkdir(parents=True, exist_ok=False)
    shutil.copytree(h20.ROOT / "slora", source / "slora", ignore=shutil.ignore_patterns("__pycache__"))
    bench = source / "benchmarks"
    bench.mkdir()
    for filename in ("lowra_saturation_exp.py", "lowra_shared_ratio_exp.py", "lmsys_maf_trace.py",
                     "saturation_cycles.py", "prepare_lmsys_maf_trace.py"):
        shutil.copy2(h20.ROOT / "benchmarks" / filename, bench)
    patches = [PATCH.read_text().replace("@BLOCK_R@", str(block_r))]
    if rank_split:
        shutil.copy2(PATCH.with_name("h20_rank_split_runtime.py"), source / "slora/common")
        patches.append(PATCH.with_name("h20_rank_split.patch").read_text())
    if adaptive:
        shutil.copy2(PATCH.with_name("h20_adaptive_layout.py"), source / "slora/common")
        patches.append(PATCH.with_name("h20_adaptive.patch").read_text())
    for patch in patches:
        subprocess.run(["git", "apply", "--no-index", "-"], cwd=source, input=patch, text=True,
                       env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(source.parent)), check=True)


def plan(mode, trace_root, output, env, prefix_lens=None, *, model_size="7b", block_widths=(32, 16)):
    if prefix_lens is None:
        prefix_lens = MATRIX[model_size]
    for block_r in block_widths:
        variant = output / mode / f"actual{block_r}"
        for prefix in prefix_lens:
            point = f"{model_size}_p{prefix}"
            trace = trace_root / point / "trace.json"
            if not trace.is_file():
                raise FileNotFoundError(trace)
            dest = variant / point
            frozen, _ = load_trace(trace)
            profile = trace_profile(frozen)
            _, groups = pair_plan(point_launcher(dest, model_size, prefix, 120, profile=profile), env)
            cmd, = (cmd for group in groups for cmd in group if cmd[cmd.index("--mode") + 1] == mode)
            cmd[cmd.index("--workload-trace") + 1] = str(trace)
            cmd[cmd.index("--output-dir") + 1] = str(dest)
            yield block_r, variant / "source", prefix, cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trace-root", type=Path, required=True, help="previous 120s suite directory")
    parser.add_argument("--model-size", choices=MATRIX, default="7b")
    parser.add_argument("--gpu", type=int, choices=range(8), required=True, help="physical nvidia-smi index")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--prefix-lens", type=int, nargs="+",
                        help="prefix lengths in tokens, in replay order for each block width")
    parser.add_argument("--block-r", type=int, choices=(16, 32),
                        help="run only this rank tile width (default: sweep 32, then 16)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prefix_lens = args.prefix_lens or MATRIX[args.model_size]
    if not set(prefix_lens).issubset(MATRIX[args.model_size]):
        parser.error(f"--prefix-lens for {args.model_size} must be chosen from {MATRIX[args.model_size]}")
    if len(set(prefix_lens)) != len(prefix_lens):
        parser.error("--prefix-lens must contain distinct lengths")
    mode = MODES[args.mode]
    output = args.output.resolve()
    shim = str(Path(h20.__file__).resolve())
    env = replay_env(dict(os.environ, SLORA_PYTHON=shim, SLORA_TRACE_PYTHON=shim,
                          SLORA_H20_PHYSICAL_GPU=str(args.gpu)))
    widths = (args.block_r,) if args.block_r is not None else (32, 16)
    jobs = list(plan(mode, args.trace_root.resolve(), output, env, prefix_lens,
                     model_size=args.model_size, block_widths=widths))
    if args.dry_run:
        for block_r, source, prefix, cmd in jobs:
            print(f"GPU {args.gpu} {mode} actual/{block_r} prefix={prefix} cwd={source}")
            print(shlex.join(h20.command(cmd[1:], plan_gpu=args.gpu)))
        return

    gpu_info = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name,uuid",
                                        "--format=csv,noheader"], text=True).strip()
    name, uuid = (field.strip() for field in gpu_info.split(","))
    if "H20" not in name:
        parser.error(f"Expected H20 on physical GPU {args.gpu}, found {name}")
    env["CUDA_VISIBLE_DEVICES"] = uuid
    h20.prepare([args.model_size])
    (output / mode).mkdir(parents=True, exist_ok=False)
    lock_path = h20.ROOT / "benchmarks/results" / f"lmsys_preemption_gpu{args.gpu}.lock"
    with lock_path.open("a") as lock:
        print(f"Waiting for GPU {args.gpu} ({uuid})", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait_idle(args.gpu)
        previous_width = None
        for block_r, source, prefix, cmd in jobs:
            if previous_width != block_r:
                stage_source(source, block_r)
                previous_width = block_r
            wait_idle(args.gpu, timeout=60)
            print(f"GPU {args.gpu} {mode} actual/{block_r} prefix={prefix}", flush=True)
            subprocess.run(cmd, cwd=source, env=env, check=True)


if __name__ == "__main__":
    main()
