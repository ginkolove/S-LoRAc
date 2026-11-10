"""Single-GPU sequential Base, LoRA Full and independent MiniKV GSM8K evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .data.gsm8k import add_arguments, validate_arguments
    from .data.qwen_gsm8k import LORA, MODEL, PROTOCOL, prepare
    from .gsm8k_score import gold_number
    from .qwen_gsm8k_score import POLICY_NAME, extract_prediction, score, score_unfiltered
    from .qwen_utils import check_adapter
else:
    from data.gsm8k import add_arguments, validate_arguments
    from data.qwen_gsm8k import LORA, MODEL, PROTOCOL, prepare
    from gsm8k_score import gold_number
    from qwen_gsm8k_score import POLICY_NAME, extract_prediction, score, score_unfiltered
    from qwen_utils import check_adapter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(model=MODEL, max_context_len=4096)
    parser.add_argument("--lora", default=LORA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32",
                        help="FP32 default: real Qwen low-precision split prefill exceeds the strict cache-check tolerance")
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
        "--device", args.device, "--batch-size", str(args.batch_size), "--dtype", args.dtype,
        "--weight-format", "safetensors", "--attn-implementation", args.attn_implementation,
        "--max-new-tokens", str(args.max_new_tokens), "--max-samples", "0", "--seed", str(args.seed),
        "--num-shots", str(args.num_shots), "--prefix-max-tokens", str(args.prefix_max_tokens),
        "--no-add-special-tokens", "--local-files-only", "--minikv-independent-prefill",
        "--scorer", str(directory / "qwen_gsm8k_score.py") + ":score", "--output", str(result)]


def finalize(result, samples, report, compatibility, args):
    value = json.loads(result.read_text())
    modes = ("base", "lora_full", "lora_minikv")
    if (value["experiment"]["parallel_base_lora"] or not value["experiment"]["minikv_independent_prefill"]
            or [r["id"] for r in value["predictions"]] != report["sample_ids"]
            or any(set(r["predictions"]) != set(modes) for r in value["predictions"])):
        raise RuntimeError("Missing/mismatched samples or nonsequential/independent evaluation")
    if len({s["prefix"] for s in samples}) != 1 or len(samples) != report["selected"]:
        raise RuntimeError("Expected one common fixed prefix and all prepared samples")
    references = [s["references"] for s in samples]
    if any(s["references"] != [gold_number(s["metadata"]["gold_solution"])] for s in samples):
        raise ValueError("Numeric references differ from original GSM8K solutions")
    diagnostics, unfiltered = {}, {}
    for mode in modes:
        raw = [r["predictions"][mode]["text"] for r in value["predictions"]]
        parsed = [extract_prediction(text) for text in raw]
        if value["scores"][mode] != score(raw, references, samples):
            raise RuntimeError(f"Independent score recomputation differs: {mode}")
        unfiltered[mode] = score_unfiltered(raw, references, samples)
        for row, answer in zip(value["predictions"], parsed):
            row["predictions"][mode]["answer_extraction"] = answer
        diagnostics[mode] = dict(samples=len(raw), boundary_trimmed=sum(p["boundary_trimmed"] for p in parsed),
            missing_or_invalid_strict_answer=sum(p["strict"] is None for p in parsed),
            flexible_fallback_used=sum(p["reason"] == "last_number_fallback" for p in parsed),
            stopped_on_eos=sum(r["predictions"][mode]["stopped_on_eos"] for r in value["predictions"]),
            reached_output_limit=sum(not r["predictions"][mode]["stopped_on_eos"] and
                r["predictions"][mode]["output_tokens"] == args.max_new_tokens for r in value["predictions"]))
    value.update(dataset_preparation=report, adapter_compatibility=compatibility,
                 generation_diagnostics=diagnostics, unfiltered_scores=unfiltered)
    value["config"]["prompt_protocol"] = PROTOCOL
    value["scorer_metadata"] = dict(policy=POLICY_NAME, primary_metric="exact_match",
        reference_independent=True, scale="fractions in [0, 1]",
        strict="first declared #### / boxed / answer tag / The answer is numeric answer; reject malformed declarations",
        flexible="strict, or last numeric token only when no final declaration exists",
        normalization="exact Decimal equality; no arithmetic evaluation or reference-based answer selection",
        comparability="local fixed train few-shots; not the undocumented adapter training prompt or a leaderboard reproduction")
    value["experiment"]["scores_scale"] = "GSM8K numeric EM fractions in [0, 1]; multiply by 100 for percentages"
    temporary = result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(result)
    return value


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, result, status = (output / name for name in ("samples.jsonl", "result.json", "status.json"))
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
        compatibility = check_adapter(args)
        update("preparing", adapter_compatibility=compatibility)
        report = prepare(args, data)
        if args.prepare_only:
            update("prepared", samples=report["selected"], prefix_tokens=report["prefix_tokens"], data=str(data))
            return
        command = evaluator_command(args, data, result)
        update("running", command=command, samples=report["selected"], prefix_tokens=report["prefix_tokens"])
        subprocess.run(command, check=True)
        samples = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        if hashlib.sha256(data.read_bytes()).hexdigest() != report["output_sha256"]:
            raise RuntimeError("Prepared data changed during evaluation")
        value = finalize(result, samples, report, compatibility, args)
        update("complete", result=str(result), scores=value["scores"], generation_diagnostics=value["generation_diagnostics"])
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
