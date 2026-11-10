"""Run fixed-English-prefix multilingual accuracy on 7B, 13B and 30B in order."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_alpaca_cleaned import free_gpu_memory, mark_partial, timestamp
from rescore_turn_boundaries import digest, load_boundaries, metric_delta, run as rescore_turn_boundaries


def validate_data(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(Path(str(path) + ".meta.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["output_sha256"]:
        raise ValueError("Prepared data hash differs from its metadata")
    counts = Counter(row["metadata"]["language"] for row in rows)
    expected = {language: report["samples_per_language"] for language in report["languages"]}
    if counts != expected or "en" in counts or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Prepared sample IDs/language counts are invalid")
    prefixes = {row["prefix"] for row in rows}
    if len(prefixes) != 1 or hashlib.sha256(next(iter(prefixes)).encode()).hexdigest() != report["prefix_sha256"]:
        raise ValueError("All samples must share the recorded English prefix")
    expected_ids = set(report["selected_source_indices"])
    if any({row["metadata"]["source_index"] for row in rows
            if row["metadata"]["language"] == language} != expected_ids for language in counts):
        raise ValueError("Language samples are not aligned to the same source tasks")
    return rows, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sizes", nargs="+", choices=["7b", "13b", "30b"], default=["7b", "13b", "30b"])
    parser.add_argument("--template-loader", type=Path, default=Path("/data/hub/alpaca_mt/alpaca_mt.py"),
                        help="Local Alpaca-MT templates used for automatic final answer filtering")
    args = parser.parse_args(argv)
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("Model sizes must not be repeated")
    root = Path(__file__).resolve().parents[1]
    data = Path(args.data).resolve()
    rows, report = validate_data(data)
    template_loader = args.template_loader.resolve()
    load_boundaries(template_loader)
    if report.get("template_sha256") and digest(template_loader) != report["template_sha256"]:
        raise ValueError("Filtering templates differ from those used to prepare the data")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        raise FileExistsError("Use a new output directory; automatic resume is not implemented")
    lock_path = root / "accexp/results/alpaca_mt_staged.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit("Another managed Alpaca-MT suite is already running")
    state = dict(state="starting", pid=os.getpid(), completed_models=[], started_at=timestamp(),
                 dataset=str(data), samples=len(rows), languages=report["languages"],
                 samples_per_language=report["samples_per_language"], seed=report["seed"],
                 physical_gpus=[0, 1], template_loader=str(template_loader))

    def update(**changes):
        state.update(changes, updated_at=timestamp())
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "status.json")
        print(timestamp(), json.dumps(changes), flush=True)

    update(state="starting")
    summary = {}
    destinations = {}
    try:
        settings = {"7b": (8, 26), "13b": (4, 40), "30b": (2, 74)}
        for size in args.sizes:
            model_id = f"huggyllama/llama-{size}"
            if model_id not in report["models"]:
                raise ValueError(f"Prepared data was not validated against {model_id}")
            batch, required_gib = settings[size]
            destination = output / f"alpaca_mt_{size}_non_en_seed{report['seed']}.json"
            log_path = destination.with_suffix(".log")
            if destination.exists() or log_path.exists():
                raise FileExistsError(f"Refusing to overwrite results for {size}")
            while True:
                free = free_gpu_memory()
                update(state="waiting_gpu", model=size, required_free_mib=required_gib * 1024,
                       free_gpu_mib=free, child_pid=None)
                if min(free) >= required_gib * 1024:
                    break
                time.sleep(30)
            command = [sys.executable, "-u", str(root / "accexp/eval.py"),
                       "--model", model_id, "--lora", f"/data/hub/alpaca_mt-lora/{size}",
                       "--data", str(data), "--device", "cuda:0", "--lora-device", "cuda:1",
                       "--batch-size", str(batch), "--minikv-batch-size", str(batch),
                       "--weight-format", "bin", "--dtype", "float16",
                       "--max-new-tokens", str(report["max_new_tokens"]),
                       "--max-samples", "0", "--seed", str(report["seed"]), "--local-files-only",
                       "--scorer", str(root / "accexp/multilingual_score.py") + ":score",
                       "--output", str(destination)]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", PYTHONUNBUFFERED="1")
            with log_path.open("x") as log:
                process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                update(state="running", model=size, child_pid=process.pid, log=str(log_path),
                       result=str(destination), command=command)
                code = process.wait()
            if code:
                mark_partial(destination, code)
                raise RuntimeError(f"{size} evaluation exited with code {code}; see {log_path}")
            result = json.loads(destination.read_text())
            if result["experiment"]["samples"] != len(rows) or [row["id"] for row in result["predictions"]] != [
                    row["id"] for row in rows] or any(set(row["predictions"]) != {
                        "base", "lora_full", "lora_minikv"} for row in result["predictions"]):
                raise RuntimeError(f"Incomplete or mismatched evaluation result for {size}")
            summary[size] = dict(scores=result["scores"], score_deltas=result["score_deltas"],
                                 minikv_minus_lora_by_language={
                                     language: {metric: result["scores"]["lora_minikv"]["by_language"][language][metric] -
                                                result["scores"]["lora_full"]["by_language"][language][metric]
                                                for metric in ("exact_match", "generic_token_f1")}
                                     for language in report["languages"]})
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            state["completed_models"].append(size)
            destinations[size] = destination
            update(state="model_complete", model=size, child_pid=None)
        filtered_dir = output / "turn_boundary_v1"
        update(state="filtering", child_pid=None, filtered_summary=str(filtered_dir / "summary.json"))
        filtered = rescore_turn_boundaries(argparse.Namespace(
            results=list(destinations.values()), data=data, template_loader=template_loader,
            output_dir=filtered_dir, audit_per_group=4, seed=report["seed"]))
        for size, destination in destinations.items():
            model = filtered["models"][destination.stem]
            scores = model["filtered_scores"]
            summary[size].update(
                filtered_scores=scores,
                filtered_score_deltas=dict(lora_minus_base=metric_delta(scores["lora_full"], scores["base"]),
                                           minikv_minus_lora=model["filtered_minikv_minus_lora"]),
                filtered_minikv_minus_lora_by_language={
                    language: metric_delta(scores["lora_minikv"]["by_language"][language],
                                           scores["lora_full"]["by_language"][language])
                    for language in report["languages"]},
                filter_counts=model["counts"], filtered_predictions=model["extracted_predictions"],
                filtering_summary=str(filtered_dir / "summary.json"))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        update(state="complete", finished_at=timestamp(), filtering_policy=filtered["policy"]["name"])
    except BaseException as error:
        update(state="failed", error=str(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
