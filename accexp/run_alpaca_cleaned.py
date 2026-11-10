"""Run the three Alpaca-Cleaned evaluations sequentially with GPU admission checks."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def free_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--id=0,1", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 2:
        raise RuntimeError("Expected free-memory readings for GPU 0 and GPU 1")
    return values


def sample_jsonl(source, destination, count, seed):
    rows, digest = [], hashlib.sha256()
    with source.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows.append(json.loads(line))
    if count < 1 or count > len(rows):
        raise ValueError(f"Sample size must be between 1 and {len(rows)}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    selected = [rows[index] for index in indices]
    if len({row["id"] for row in selected}) != count:
        raise ValueError("Selected sample IDs are not unique")
    with destination.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(source=str(source), source_sha256=digest.hexdigest(), population=len(rows),
                  selected=count, seed=seed, sampling="uniform without replacement; source order retained",
                  indices=indices, sample_ids=[row["id"] for row in selected],
                  unique_prefixes=len({row["prefix"] for row in selected}),
                  output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    Path(str(destination) + ".meta.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def mark_partial(destination, code):
    path = destination.with_suffix(".partial.json")
    if path.is_file():
        summary = json.loads(path.read_text())
        summary.update(state="interrupted" if code < 0 else "failed", exit_code=code,
                       updated_at_unix=time.time())
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = root / "accexp/results/alpaca_cleaned_staged.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another managed Alpaca-Cleaned suite is already running")
    source = root / "accexp/results/alpaca_cleaned_3shot_all_seed42.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    data = output / f"alpaca_cleaned_random{args.sample_size}_seed{args.seed}.jsonl"
    sampling = sample_jsonl(source, data, args.sample_size, args.seed)
    state = dict(state="starting", pid=os.getpid(), completed_models=[], started_at=timestamp(),
                 dataset=str(data), max_new_tokens=256, max_samples=0, seed=args.seed,
                 sample_size=args.sample_size, population=sampling["population"])

    def update(**changes):
        state.update(changes, updated_at=timestamp())
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "status.json")
        print(timestamp(), json.dumps(changes), flush=True)

    update(state="starting")
    try:
        for size, batch, required_mib in (("7b", 8, 26 * 1024),
                                          ("13b", 4, 40 * 1024),
                                          ("30b", 2, 74 * 1024)):
            destination = output / f"alpaca_cleaned_{size}_random{args.sample_size}_seed{args.seed}.json"
            log_path = destination.with_suffix(".log")
            if destination.exists() or log_path.exists():
                raise FileExistsError(f"Refusing to overwrite results for {size}: {output}")
            while True:
                free = free_gpu_memory()
                update(state="waiting_gpu", model=size, batch_size=batch,
                       required_free_mib=required_mib, free_gpu_mib=free, child_pid=None)
                if min(free) >= required_mib:
                    break
                time.sleep(30)
            command = [
                sys.executable, "-u", str(root / "accexp/eval.py"),
                "--model", f"huggyllama/llama-{size}",
                "--lora", f"/data/hub/alpaca_cleaned-lora/{size}",
                "--data", str(data), "--device", "cuda:0", "--lora-device", "cuda:1",
                "--batch-size", str(batch), "--minikv-batch-size", str(batch),
                "--weight-format", "bin", "--dtype", "float16", "--max-new-tokens", "256",
                "--max-samples", "0", "--seed", str(args.seed), "--local-files-only",
                "--output", str(destination)]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", PYTHONUNBUFFERED="1")
            with log_path.open("x") as log:
                process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                update(state="running", model=size, child_pid=process.pid,
                       log=str(log_path), result=str(destination), command=command)
                code = process.wait()
            if code:
                mark_partial(destination, code)
                raise RuntimeError(f"{size} evaluation exited with code {code}; see {log_path}")
            result = json.loads(destination.read_text())
            expected = result["experiment"]["samples"]
            if expected != args.sample_size or len(result["predictions"]) != expected or any(
                    set(row["predictions"]) != {"base", "lora_full", "lora_minikv"}
                    for row in result["predictions"]):
                raise RuntimeError(f"Incomplete evaluation result for {size}")
            state["completed_models"].append(size)
            update(state="model_complete", model=size, child_pid=None, samples=expected)
        update(state="complete", finished_at=timestamp())
    except BaseException as error:
        update(state="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
