"""Load the TP=2 model before preparing Alpaca-Cleaned Base/Full/MiniKV data."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .data.prepare_alpaca_cleaned import add_arguments, prepare, validate_arguments
    from .run_alpaca_cleaned import mark_partial
else:
    from data.prepare_alpaca_cleaned import add_arguments, prepare, validate_arguments
    from run_alpaca_cleaned import mark_partial


def evaluation_command(args, data, result, check_only=False):
    command = [sys.executable, "-u"]
    if not check_only:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    "--monitor-interval", "1"]
    command += [str(Path(__file__).with_name("eval_tp.py")),
                "--model", str(args.model), "--lora", str(args.lora), "--data", str(data),
                "--dataset-name", "alpaca_cleaned", "--model-weight-format", args.model_weight_format,
                "--adapter-weight-format", args.adapter_weight_format, "--dtype", args.dtype,
                "--batch-size", str(args.batch_size), "--max-new-tokens", str(args.max_new_tokens),
                "--seed", str(args.seed), "--output", str(result)]
    if args.tokenizer:
        command += ["--tokenizer", args.tokenizer]
    if check_only:
        command.append("--check-only")
    else:
        command += ["--prepare-alpaca-cleaned", "--dataset-dir", str(args.dataset_dir),
                    "--sample-size", str(args.sample_size), "--num-shots", str(args.num_shots),
                    "--prefix-max-tokens", str(args.prefix_max_tokens)]
    return command


def launcher_environment():
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment["PYTHONUNBUFFERED"] = "1"
    # Preserve nohup's inherited SIGHUP ignore instead of letting Elastic replace it.
    environment["TORCHELASTIC_SIGNALS_TO_HANDLE"] = "SIGTERM,SIGINT,SIGQUIT"
    return environment


def validate_gpu_memory(devices, minimum_gib):
    if len(devices) != 2:
        raise RuntimeError(f"TP=2 needs exactly two visible GPUs; found {len(devices)}")
    insufficient = [device for device in devices if device["free_bytes"] < minimum_gib * 2**30]
    if insufficient:
        detail = "; ".join(f"cuda:{device['device']} ({device['name']}): "
                           f"{device['free_bytes']/2**30:.2f} GiB free" for device in insufficient)
        raise RuntimeError(f"Insufficient free GPU memory; require {minimum_gib:g} GiB per GPU. "
                           f"{detail}. Wait until both selected GPUs are free; no model was loaded.")
    return devices


def check_gpu_memory(minimum_gib):
    # A separate probe honors CUDA's device mapping without retaining parent CUDA contexts.
    probe = """import json
import torch
devices = []
for device in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(device)
    properties = torch.cuda.get_device_properties(device)
    devices.append(dict(device=device, name=properties.name, uuid=str(getattr(properties, "uuid", "")),
                        free_bytes=free, total_bytes=total))
print(json.dumps(devices))
"""
    result = subprocess.run([sys.executable, "-c", probe], env=launcher_environment(),
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"GPU memory probe failed before model loading: {result.stderr.strip()}")
    return validate_gpu_memory(json.loads(result.stdout), minimum_gib)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--lora", default="/data/hub/alpaca_cleaned-lora/65b")
    parser.add_argument("--model-weight-format", choices=("auto", "safetensors", "bin"), default="safetensors")
    parser.add_argument("--adapter-weight-format", choices=("auto", "safetensors", "bin"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--min-free-gpu-gib", type=float, default=76,
                        help="Required free memory per visible GPU before TP startup; 0 is for tiny-model tests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data and validate checkpoints without loading GPUs")
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not math.isfinite(args.min_free_gpu_gib) or args.min_free_gpu_gib < 0:
        parser.error("min-free-gpu-gib must be finite and nonnegative")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Launch this script with plain Python; it starts torchrun itself")
    return args


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = output / "status.json"
        if status_path.exists():
            raise FileExistsError(f"Use a fresh output directory: {output}")
        state = dict(state="starting", pid=os.getpid(), started_at_unix=time.time())

        def update(**changes):
            state.update(changes, updated_at_unix=time.time())
            temporary = output / "status.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(status_path)
            print(json.dumps(changes), flush=True)

        result = output / "result.json"
        try:
            data = output / f"alpaca_cleaned_random{args.sample_size}_seed{args.seed}.jsonl"
            update(state="starting", dataset=str(data), result=str(result),
                   requested_samples=args.sample_size, batch_size=args.batch_size, tp_size=2,
                   startup_order="model_then_data")
            environment = launcher_environment()
            if args.prepare_only:
                update(state="preparing")
                sampling = prepare(args, data)
                subprocess.run(evaluation_command(args, data, result, check_only=True),
                               env=environment, stdin=subprocess.DEVNULL, check=True)
                update(state="prepared", samples=sampling["selected"],
                       prefix_tokens=sampling["prefix_tokens"], data_sha256=sampling["output_sha256"])
                return
            update(state="checking_gpu", min_free_gpu_gib=args.min_free_gpu_gib)
            devices = check_gpu_memory(args.min_free_gpu_gib)
            update(state="starting", gpu_memory=devices)
            command = evaluation_command(args, data, result)
            with subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL) as process:
                update(state="running", child_pid=process.pid, command=command)
                code = process.wait()
            if code:
                mark_partial(result, code)
                raise RuntimeError(f"TP evaluation exited with code {code}; see run.log")
            sampling = json.loads(Path(str(data) + ".meta.json").read_text())
            report = json.loads(result.read_text())
            modes = {"base", "lora_full", "lora_minikv"}
            if (report["experiment"]["samples"] != sampling["selected"] or
                    (args.sample_size and sampling["selected"] != args.sample_size) or
                    report["experiment"]["data_sha256"] != sampling["output_sha256"] or
                    len(report["predictions"]) != sampling["selected"] or
                    any(set(row["predictions"]) != modes for row in report["predictions"])):
                raise RuntimeError("TP evaluation did not return all samples in all three modes")
            update(state="complete", child_pid=None, samples=sampling["selected"],
                   prefix_tokens=sampling["prefix_tokens"], data_sha256=sampling["output_sha256"],
                   scores=report["scores"], finished_at_unix=time.time())
        except BaseException as error:
            update(state="failed", error=str(error))
            raise


if __name__ == "__main__":
    run(parse_args())
