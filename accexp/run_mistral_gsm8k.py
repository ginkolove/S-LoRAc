"""Single-device sequential Base, LoRA Full and independent MiniKV evaluation on GSM8K."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .data.gsm8k import add_arguments, prepare, validate_arguments
    from .gsm8k_score import POLICY_NAME, REFERENCE, extract_prediction, score_unfiltered
else:
    from data.gsm8k import add_arguments, prepare, validate_arguments
    from gsm8k_score import POLICY_NAME, REFERENCE, extract_prediction, score_unfiltered


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--lora", default="/data/hub/gsm8k-lora")
    parser.add_argument("--output-dir", default="accexp/results/mistral_gsm8k_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=20)
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
    return [sys.executable, "-u", str(directory / "eval.py"),
            "--model", args.model, "--lora", args.lora, "--data", str(data),
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--dtype", args.dtype, "--weight-format", args.weight_format,
            "--attn-implementation", args.attn_implementation,
            "--max-new-tokens", str(args.max_new_tokens), "--max-samples", "0",
            "--seed", str(args.seed), "--local-files-only", "--minikv-independent-prefill",
            "--scorer", str(directory / "gsm8k_score.py") + ":score", "--output", str(result)]


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
                raise ValueError(f"This launcher requires Mistral, got {config.model_type}")
        update("preparing", data=str(data))
        report = prepare(args, data)
        if args.prepare_only:
            update("prepared", samples=report["selected"], prefix_tokens=report["prefix_tokens"],
                   context_limit=report["context_limit"])
            return
        command = evaluator_command(args, data, result)
        update("running", command=command, samples=report["selected"], prefix_tokens=report["prefix_tokens"],
               context_limit=report["context_limit"], result=str(result))
        subprocess.run(command, check=True)
        value = json.loads(result.read_text())
        if value["experiment"]["parallel_base_lora"] or not value["experiment"]["minikv_independent_prefill"]:
            raise RuntimeError("Expected single-device sequential evaluation with independent MiniKV sources")
        modes = ("base", "lora_full", "lora_minikv")
        if [row["id"] for row in value["predictions"]] != report["sample_ids"] or any(
                set(row["predictions"]) != set(modes) for row in value["predictions"]):
            raise RuntimeError("Missing or mismatched GSM8K samples/modes")
        samples = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        references = [sample["references"] for sample in samples]
        diagnostics, unfiltered = {}, {}
        for mode in modes:
            raw = [row["predictions"][mode]["text"] for row in value["predictions"]]
            extracted = [extract_prediction(text) for text in raw]
            unfiltered[mode] = score_unfiltered(raw, references, samples)
            for row, parsed in zip(value["predictions"], extracted):
                row["predictions"][mode]["answer_extraction"] = dict(parsed)
            diagnostics[mode] = dict(
                samples=len(raw), boundary_trimmed=sum(item["boundary_trimmed"] for item in extracted),
                missing_or_invalid_strict_answer=sum(item["strict"] is None for item in extracted),
                flexible_fallback_used=sum(item["reason"] == "last_number_fallback" for item in extracted),
                stopped_on_eos=sum(row["predictions"][mode]["stopped_on_eos"] for row in value["predictions"]),
                reached_output_limit=sum(not row["predictions"][mode]["stopped_on_eos"] and
                                         row["predictions"][mode]["output_tokens"] == args.max_new_tokens
                                         for row in value["predictions"]))
        value["dataset_preparation"] = report
        value["scorer_metadata"] = dict(
            policy=POLICY_NAME, final_marker_reference=REFERENCE, reference_independent=True,
            metrics=["exact_match", "flexible_exact_match"], primary_metric="exact_match",
            strict="first #### numeric prefix; optional currency/grouping/units; reject expression prefixes",
            flexible="strict answer, or last numeric token before the next question only when #### is absent",
            boundary="line-start adapter prompt or Question:/### Question:",
            normalization="exact Decimal numeric equality; no expression evaluation or ground-truth matching",
            comparability="local few-shot protocol, not an identical official leaderboard setup")
        value["experiment"]["scores_scale"] = "GSM8K exact-match fractions in [0, 1]; multiply by 100 for percentages"
        value["unfiltered_scores"] = unfiltered
        value["generation_diagnostics"] = diagnostics
        temporary = result.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(result)
        update("complete", result=str(result), scores=value["scores"], generation_diagnostics=diagnostics)
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
