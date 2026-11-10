"""Prepare local CNN/DailyMail and evaluate three modes sequentially on one GPU."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .cnn_dailymail_score import score
    from .data.cnn_dailymail import add_arguments, prepare, validate_arguments
else:
    from cnn_dailymail_score import score
    from data.cnn_dailymail import add_arguments, prepare, validate_arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--lora", default="/data/hub/cnn_dailmail-lora")
    parser.add_argument("--output-dir", default="accexp/results/mistral_cnn_" +
                        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--weight-format", choices=["bin", "safetensors", "auto"], default="bin")
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return args


def evaluator_command(args, data, result):
    directory = Path(__file__).resolve().parent
    command = [
        sys.executable, "-u", str(directory / "eval.py"),
        "--model", args.model, "--lora", args.lora, "--data", str(data),
        "--device", args.device, "--batch-size", str(args.batch_size),
        "--dtype", args.dtype, "--weight-format", args.weight_format,
        "--attn-implementation", args.attn_implementation,
        "--max-new-tokens", str(args.max_new_tokens), "--max-samples", "0",
        "--seed", str(args.seed), "--local-files-only",
        "--minikv-independent-prefill",
        "--scorer", str(directory / "cnn_dailymail_score.py") + ":score",
        "--output", str(result)]
    return command


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, result, status = (output / name for name in ("samples.jsonl", "result.json", "status.json"))
    with status.open("x", encoding="utf-8") as handle:
        json.dump(dict(state="starting", config=vars(args)), handle, indent=2)

    def update(state, **details):
        value = dict(state=state, config=vars(args), **details)
        temporary = status.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(status)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        if not args.prepare_only:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(args.model, local_files_only=True)
            if config.model_type != "mistral":
                raise ValueError(f"This launcher requires a Mistral model, got {config.model_type}")
            score(["dependency check"], [["dependency check"]], [{}])
        update("preparing", data=str(data))
        report = prepare(args, data)
        if args.prepare_only:
            update("prepared", data=str(data), samples=report["selected"],
                   prefix_tokens=report["prefix_tokens"], context_limit=report["context_limit"])
            return
        command = evaluator_command(args, data, result)
        update("running", command=command, result=str(result), samples=report["selected"],
               prefix_tokens=report["prefix_tokens"], context_limit=report["context_limit"])
        subprocess.run(command, check=True)
        value = json.loads(result.read_text())
        if value["experiment"]["parallel_base_lora"] or not value["experiment"]["minikv_independent_prefill"]:
            raise RuntimeError("Expected single-device, independent third-stage prefix preparation")
        if len(value["predictions"]) != report["selected"] or any(
                set(row["predictions"]) != {"base", "lora_full", "lora_minikv"}
                for row in value["predictions"]):
            raise RuntimeError("The evaluator did not complete all samples in all three modes")
        value["dataset_preparation"] = report
        value["scorer_metadata"] = dict(
            implementation="rouge-score", version=importlib.metadata.version("rouge-score"),
            metrics=["rouge1", "rouge2", "rougeL"], aggregation="macro mean F1",
            use_stemmer=True, scale="fractions in [0, 1]", output_filtering="none")
        value["experiment"]["scores_scale"] = "ROUGE F1 fractions in [0, 1]; multiply by 100 for percentages"
        value["generation_diagnostics"] = {
            mode: dict(samples=len(value["predictions"]),
                       stopped_on_eos=sum(row["predictions"][mode]["stopped_on_eos"]
                                          for row in value["predictions"]),
                       reached_output_limit=sum(not row["predictions"][mode]["stopped_on_eos"] and
                                                row["predictions"][mode]["output_tokens"] == args.max_new_tokens
                                                for row in value["predictions"]))
            for mode in ("base", "lora_full", "lora_minikv")}
        temporary = result.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(result)
        update("complete", result=str(result), scores=value["scores"],
               generation_diagnostics=value["generation_diagnostics"])
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
