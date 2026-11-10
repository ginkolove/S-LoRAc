"""Evaluate Alpaca-MT on one TP=2 model, then uniformly filter all three modes."""

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
    from .data.prepare_alpaca_mt import LANGUAGES, prepare_for_tp
    from .rescore_turn_boundaries import digest, metric_delta, run as rescore_turn_boundaries
    from .run_alpaca_cleaned import mark_partial
    from .run_alpaca_cleaned_65b import check_gpu_memory, launcher_environment
    from .run_alpaca_mt import validate_data
else:
    from data.prepare_alpaca_mt import LANGUAGES, prepare_for_tp
    from rescore_turn_boundaries import digest, metric_delta, run as rescore_turn_boundaries
    from run_alpaca_cleaned import mark_partial
    from run_alpaca_cleaned_65b import check_gpu_memory, launcher_environment
    from run_alpaca_mt import validate_data


def evaluation_command(args, data, result, check_only=False):
    command = [sys.executable, "-u"]
    if not check_only:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    "--monitor-interval", "1"]
    command += [str(Path(__file__).with_name("eval_tp.py")),
                "--model", args.model, "--lora", args.lora, "--data", str(data),
                "--dataset-name", "alpaca_mt", "--model-weight-format", args.model_weight_format,
                "--adapter-weight-format", args.adapter_weight_format, "--dtype", args.dtype,
                "--batch-size", str(args.batch_size), "--max-new-tokens", str(args.max_new_tokens),
                "--seed", str(args.seed), "--output", str(result),
                "--scorer", str(Path(__file__).with_name("multilingual_score.py")) + ":score"]
    if check_only:
        command.append("--check-only")
    elif args.data is None:
        command += ["--prepare-alpaca-mt", "--english-file", str(args.english_file),
                    "--prefix-data", str(args.prefix_data), "--raw-dir", str(args.raw_dir),
                    "--template-file", str(args.template_file), "--languages", *args.languages,
                    "--samples-per-language", str(args.samples_per_language)]
    return command


def validate_prepared_data(args, data):
    rows, report = validate_data(data)
    if (set(report["languages"]) != set(args.languages) or
            report["samples_per_language"] != args.samples_per_language or
            report["seed"] != args.seed or report["max_new_tokens"] != args.max_new_tokens):
        raise ValueError("Prepared MT sampling/language/output settings differ from this run")
    if args.model not in report["models"]:
        raise ValueError(f"Prepared MT inputs were not length-checked against {args.model}")
    if report["template_sha256"] != digest(args.template_file):
        raise ValueError("Answer filtering templates differ from the data preparation templates")
    return rows, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model", default="/data/hub/Llama-65B")
    parser.add_argument("--lora", default="/data/hub/alpaca_mt-lora/65b")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/data/hub/alpaca_mt"))
    parser.add_argument("--template-file", type=Path, help="Default: <dataset-dir>/alpaca_mt.py")
    parser.add_argument("--english-file", type=Path, default=root / "accexp/results/alpaca_mt_en_raw.json")
    parser.add_argument("--prefix-data", type=Path,
                        default=root / "accexp/results/alpaca_mt_en_3shot_all_seed42.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=root / "accexp/results/alpaca_mt_raw")
    parser.add_argument("--languages", nargs="+", choices=tuple(LANGUAGES), default=["pt", "es", "ca"])
    parser.add_argument("--samples-per-language", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--min-free-gpu-gib", type=float, default=76)
    parser.add_argument("--model-weight-format", choices=("auto", "safetensors", "bin"), default="safetensors")
    parser.add_argument("--adapter-weight-format", choices=("auto", "safetensors", "bin"), default="bin")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--data", type=Path, help="Reuse prepared MT JSONL; read only after model loading")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and validate without GPU model loading")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.template_file = args.template_file or args.dataset_dir / "alpaca_mt.py"
    if min(args.samples_per_language, args.max_new_tokens, args.batch_size) < 1:
        parser.error("Sample/output/batch sizes must be positive")
    if len(set(args.languages)) != len(args.languages):
        parser.error("Languages must not repeat")
    if not math.isfinite(args.min_free_gpu_gib) or args.min_free_gpu_gib < 0:
        parser.error("min-free-gpu-gib must be finite and nonnegative")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use plain Python; this launcher starts torchrun itself")
    return args


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = output / "status.json"
        if status_path.exists():
            raise FileExistsError(f"Use a fresh output directory: {output}")
        state = dict(pid=os.getpid(), started_at_unix=time.time())

        def update(**changes):
            state.update(changes, updated_at_unix=time.time())
            temporary = output / "status.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(status_path)
            print(json.dumps(changes), flush=True)

        label = "_".join(args.languages)
        data = (args.data.resolve() if args.data else
                output / f"alpaca_mt_{label}_{args.samples_per_language}_seed{args.seed}.jsonl")
        result_path = output / "result.json"
        try:
            update(state="starting", dataset=str(data), result=str(result_path), languages=args.languages,
                   samples_per_language=args.samples_per_language, batch_size=args.batch_size,
                   tp_size=2, startup_order="model_then_data")
            environment = launcher_environment()
            if args.prepare_only:
                update(state="preparing")
                if args.data is None:
                    prepare_for_tp(args, data)
                rows, report = validate_prepared_data(args, data)
                subprocess.run(evaluation_command(args, data, result_path, check_only=True),
                               env=environment, stdin=subprocess.DEVNULL, check=True)
                update(state="prepared", samples=len(rows), models=report["models"],
                       data_sha256=report["output_sha256"])
                return
            update(state="checking_gpu", min_free_gpu_gib=args.min_free_gpu_gib)
            devices = check_gpu_memory(args.min_free_gpu_gib)
            command = evaluation_command(args, data, result_path)
            with subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL) as process:
                update(state="running", gpu_memory=devices, child_pid=process.pid, command=command)
                code = process.wait()
            if code:
                mark_partial(result_path, code)
                raise RuntimeError(f"TP evaluation exited with code {code}; see run.log")
            rows, report = validate_prepared_data(args, data)
            result = json.loads(result_path.read_text())
            modes = {"base", "lora_full", "lora_minikv"}
            if (result["experiment"]["samples"] != len(rows) or
                    result["experiment"]["data_sha256"] != report["output_sha256"] or
                    [row["id"] for row in result["predictions"]] != [row["id"] for row in rows] or
                    any(set(row["predictions"]) != modes for row in result["predictions"])):
                raise RuntimeError("Incomplete or mismatched MT result")
            filtered_dir = output / "turn_boundary_v1"
            update(state="filtering", child_pid=None)
            filtered = rescore_turn_boundaries(argparse.Namespace(
                results=[result_path], data=data, template_loader=args.template_file,
                output_dir=filtered_dir, audit_per_group=4, seed=args.seed))
            filtered_model = filtered["models"][result_path.stem]
            scores = filtered_model["filtered_scores"]
            summary = dict(samples=len(rows), languages=args.languages,
                samples_per_language=args.samples_per_language, seed=args.seed,
                scores=result["scores"], score_deltas=result["score_deltas"],
                minikv_minus_lora_by_language={language: metric_delta(
                    result["scores"]["lora_minikv"]["by_language"][language],
                    result["scores"]["lora_full"]["by_language"][language]) for language in args.languages},
                filtered_scores=scores,
                filtered_score_deltas=dict(lora_minus_base=metric_delta(scores["lora_full"], scores["base"]),
                                           minikv_minus_lora=metric_delta(scores["lora_minikv"], scores["lora_full"])),
                filtered_minikv_minus_lora_by_language={language: metric_delta(
                    scores["lora_minikv"]["by_language"][language],
                    scores["lora_full"]["by_language"][language]) for language in args.languages},
                filter_counts=filtered_model["counts"],
                filtered_predictions=filtered_model["extracted_predictions"],
                filtering_policy=filtered["policy"], raw_result=str(result_path),
                filtering_summary=str(filtered_dir / "summary.json"))
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            update(state="complete", child_pid=None, samples=len(rows), scores=result["scores"],
                   filtered_scores=scores, finished_at_unix=time.time())
        except BaseException as error:
            update(state="failed", error=str(error))
            raise


if __name__ == "__main__":
    run(parse_args())
