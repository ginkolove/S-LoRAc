"""Qwen CNN/DailyMail accuracy: native generation, sequential Base/Full/MiniKV."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .cnn_dailymail_score import METRICS, TURN_POLICY, extract_summary, score, score_turn_filtered
    from .data.cnn_dailymail import add_arguments, validate_arguments
    from .data.qwen_cnn import LORA, MODEL, PROTOCOL, prepare
    from .qwen_utils import check_adapter
else:
    from cnn_dailymail_score import METRICS, TURN_POLICY, extract_summary, score, score_turn_filtered
    from data.cnn_dailymail import add_arguments, validate_arguments
    from data.qwen_cnn import LORA, MODEL, PROTOCOL, prepare
    from qwen_utils import check_adapter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(model=MODEL, max_context_len=4096)
    parser.add_argument("--lora", default=LORA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Conservative FP32 batch for complete CNN/DailyMail articles")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32",
                        help="FP32 keeps the strict Qwen prefix-cache consistency check")
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
        "--model", args.model, "--tokenizer", args.model, "--lora", args.lora, "--data", str(data),
        "--device", args.device, "--batch-size", str(args.batch_size), "--dtype", args.dtype,
        "--weight-format", "safetensors", "--attn-implementation", args.attn_implementation,
        "--max-new-tokens", str(args.max_new_tokens), "--max-samples", "0", "--seed", str(args.seed),
        "--num-shots", str(args.num_shots), "--prefix-max-tokens", str(args.prefix_max_tokens),
        "--no-add-special-tokens", "--local-files-only", "--minikv-independent-prefill",
        "--scorer", str(directory / "cnn_dailymail_score.py") + ":score_turn_filtered", "--output", str(result)]


def finalize(result, samples, report, compatibility, args):
    value = json.loads(result.read_text())
    modes = ("base", "lora_full", "lora_minikv")
    experiment = value["experiment"]
    if (experiment.get("generation_engine") != "transformers.generate" or experiment["parallel_base_lora"]
            or not experiment["minikv_independent_prefill"] or experiment["unique_prefixes"] != 1
            or [r["id"] for r in value["predictions"]] != report["sample_ids"]
            or any(set(r["predictions"]) != set(modes) for r in value["predictions"])):
        raise RuntimeError("Expected every prepared sample/mode, native generate and independent sequential evaluation")
    if (len(samples) != report["selected"] or len({s["prefix"] for s in samples}) != 1
            or hashlib.sha256(samples[0]["prefix"].encode()).hexdigest() != report["prefix_sha256"]):
        raise RuntimeError("Prepared fixed prefix or sample count differs from its manifest")
    references = [s["references"] for s in samples]
    unfiltered, diagnostics = {}, {}
    for mode in modes:
        raw = [r["predictions"][mode]["text"] for r in value["predictions"]]
        parsed = [extract_summary(text) for text in raw]
        if value["scores"][mode] != score_turn_filtered(raw, references, samples):
            raise RuntimeError(f"ROUGE independent recomputation differs: {mode}")
        unfiltered[mode] = score(raw, references, samples)
        for row, summary in zip(value["predictions"], parsed):
            row["predictions"][mode]["summary_extraction"] = summary
        diagnostics[mode] = dict(samples=len(raw), boundary_trimmed=sum(p["boundary_trimmed"] for p in parsed),
            stopped_on_eos=sum(r["predictions"][mode]["stopped_on_eos"] for r in value["predictions"]),
            reached_output_limit=sum(not r["predictions"][mode]["stopped_on_eos"] and
                r["predictions"][mode]["output_tokens"] == args.max_new_tokens for r in value["predictions"]))
    value.update(dataset_preparation=report, adapter_compatibility=compatibility,
                 generation_diagnostics=diagnostics, unfiltered_scores=unfiltered)
    value["config"]["prompt_protocol"] = PROTOCOL
    value["scorer_metadata"] = dict(implementation="rouge-score", version=importlib.metadata.version("rouge-score"),
        primary_metric="rougeL", metrics=list(METRICS), aggregation="macro mean F1", use_stemmer=True,
        scale="fractions in [0, 1]", policy=TURN_POLICY, reference_independent_filter=True,
        output_filtering="only explicit line-start next-article/instruction boundaries; full multiline summary retained",
        comparability="local fixed train few-shots; adapter training template/membership not independently verified")
    experiment["scores_scale"] = "ROUGE F1 fractions in [0, 1]; multiply by 100 for percentages"
    temporary = result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(result)
    return value


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, result, status = (output / name for name in ("samples.jsonl", "result.json", "status.json"))
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite results: {result}")
    with status.open("x", encoding="utf-8") as handle:
        json.dump(dict(state="starting", config=vars(args)), handle, indent=2)

    def update(state, **details):
        value = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(), config=vars(args), **details)
        temporary = status.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(status)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        update("checking")
        compatibility = check_adapter(args, allow_template_mismatch=True)
        score(["dependency check"], [["dependency check"]], [{}])
        update("preparing", adapter_compatibility=compatibility)
        report = prepare(args, data)
        args.prefix_max_tokens = report["prefix_max_tokens"]
        if args.prepare_only:
            update("prepared", data=str(data), samples=report["selected"], prefix_tokens=report["prefix_tokens"],
                   input_tokens_max=report["input_tokens_max"], context_limit=report["context_limit"])
            return
        command = evaluator_command(args, data, result)
        update("running", command=command, samples=report["selected"], prefix_tokens=report["prefix_tokens"])
        subprocess.run(command, check=True)
        if hashlib.sha256(data.read_bytes()).hexdigest() != report["output_sha256"]:
            raise RuntimeError("Prepared data changed during evaluation")
        samples = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        value = finalize(result, samples, report, compatibility, args)
        update("complete", result=str(result), scores=value["scores"], generation_diagnostics=value["generation_diagnostics"])
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
