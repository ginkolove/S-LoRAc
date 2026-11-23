"""Run the paired five-method sweep on H20 or A100."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

import h20_runtime as h20
import h20_pair
from lmsys_maf_trace import HF_PROFILE

ROOT = Path(__file__).resolve().parents[1]
MATRIX = {"7b": (5000, 8000, 10000, 12000, 15000), "13b": (5000, 8000, 10000),
          "30b": (1000, 2000, 3000)}
REPREFILL_POOL = {"7b": 120000, "13b": 62000}


def point_launcher(dest, size, prefix, duration, *, profile, start_now=False):
    cmd = ["bash", "benchmarks/run_prefix_sources_after_idle.sh", str(dest),
           "--start-now" if start_now else "--queue", "--model-size", size,
           "--prefix-len", str(prefix), "--duration", str(duration), "--adapter-profile",
           profile, "--fixed-rank64", "--replay-timeout", "86400"]
    if size in REPREFILL_POOL:
        cmd += ["--reprefill-num-token", str(REPREFILL_POOL[size])]
    return cmd


def dry_run(cmd, env, hardware):
    plan = subprocess.check_output(cmd + ["--dry-run"], env=env, cwd=ROOT, text=True)
    if hardware == "a100":
        print(plan, end="")
        return
    gpu = 0
    for line in plan.splitlines():
        if line.startswith("CUDA_VISIBLE_DEVICES="):
            gpu = int(line.split("=", 1)[1].split(";")[0])
        if line.startswith(env["SLORA_PYTHON"] + " "):
            print(shlex.join(h20.command(shlex.split(line)[1:], plan_gpu=gpu)))
        elif line.startswith("After both succeed: "):
            args = shlex.split(line.removeprefix("After both succeed: "))[1:]
            print("After both succeed:", shlex.join(h20.command(args, plan_gpu=gpu)))
        elif not line.startswith("After both full groups finish:"):
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="H20 caches its pinned tokenizer files and uvloop under benchmarks/results/.h20. "
               "Its dry run shows the first candidate ports for the selected GPUs.")
    parser.add_argument("--hardware", type=str.lower, choices=("a100", "h20"), required=True)
    parser.add_argument("output", type=Path, help="fresh output directory")
    parser.add_argument("--duration", type=int, help="send window in seconds (H20 120, A100 300)")
    parser.add_argument("--adapter-profile", choices=(HF_PROFILE, "chameleon"),
                        help="rank distribution (H20 default: hf-llama7b, A100 default: chameleon)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--point", choices=[f"{s}:{p}" for s, lengths in MATRIX.items() for p in lengths],
                           help="run just one model/prefix combination")
    selection.add_argument("--model-size", choices=MATRIX, help="run every prefix length for one model")
    parser.add_argument("--gpus", type=int, nargs=2, metavar=("FIRST", "SECOND"),
                        help="physical nvidia-smi indices for the two H20 workers, in order")
    parser.add_argument("--four-methods", action="store_true",
                        help="first GPU runs re-prefill, full reconstruction and CPU-read; "
                             "second GPU runs fixed rank-64 Top-1 CPU-read")
    parser.add_argument("--start-now", action="store_true", help="fail immediately if GPUs are busy")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without using GPUs")
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (120 if args.hardware == "h20" else 300)
    profile = args.adapter_profile or (HF_PROFILE if args.hardware == "h20" else "chameleon")
    if duration < 1:
        parser.error("--duration must be positive")
    if args.four_methods and not args.gpus:
        parser.error("--four-methods requires --gpus FIRST SECOND")
    if args.gpus and (args.hardware != "h20" or len(set(args.gpus)) != 2
                      or min(args.gpus) < 0 or max(args.gpus) > 7):
        parser.error("--gpus requires H20 and two distinct physical indices in 0..7")
    points = [(s, p) for s, lengths in MATRIX.items() for p in lengths
              if (not args.model_size or args.model_size == s) and (not args.point or args.point == f"{s}:{p}")]
    env = os.environ.copy()
    if args.hardware == "h20":
        shim = str(Path(h20.__file__).resolve())
        env.update(SLORA_PYTHON=shim, SLORA_TRACE_PYTHON=shim)
        if not args.dry_run:
            indices = ",".join(map(str, args.gpus or (0, 1)))
            names = subprocess.check_output(["nvidia-smi", "-i", indices, "--query-gpu=name",
                                             "--format=csv,noheader"], text=True).splitlines()
            if len(names) != 2 or not all("H20" in name for name in names):
                parser.error(f"Expected H20 on GPUs {indices}, found {names}")
            h20.prepare(dict.fromkeys(s for s, _ in points))
    for size, prefix in points:
        dest = args.output.resolve() / f"{size}_p{prefix}"
        cmd = point_launcher(dest, size, prefix, duration, profile=profile, start_now=args.start_now)
        print(f"{args.hardware.upper()} Llama-{size}, prefix={prefix}, send={duration}s, profile={profile}", flush=True)
        if args.gpus:
            h20_pair.run(cmd, env, dest, args.gpus, args.dry_run, args.start_now, args.four_methods)
        elif args.dry_run:
            dry_run(cmd, env, args.hardware)
        else:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
