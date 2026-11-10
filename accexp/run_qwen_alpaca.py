"""Single-device Qwen Alpaca evaluation: Base, LoRA Full, independent MiniKV."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .data.qwen_alpaca import CAVEAT, LORA, PROTOCOL, add_arguments, prepare, validate_arguments
    from .qwen_alpaca_score import BOUNDARIES, METRICS, POLICY, extract_prediction, score, score_unfiltered
    from .qwen_utils import check_adapter
else:
    from data.qwen_alpaca import CAVEAT, LORA, PROTOCOL, add_arguments, prepare, validate_arguments
    from qwen_alpaca_score import BOUNDARIES, METRICS, POLICY, extract_prediction, score, score_unfiltered
    from qwen_utils import check_adapter


MODES = ("base", "lora_full", "lora_minikv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--lora", default=LORA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32",
                        help="Same FP32 default and strict cache checks as existing Qwen experiments")
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
        "--scorer", str(directory / "qwen_alpaca_score.py") + ":score", "--output", str(result)]


def finalize(result, samples, report, compatibility, args):
    value = json.loads(result.read_text())
    experiment, records = value["experiment"], value["predictions"]
    if (experiment.get("generation_engine") != "transformers.generate" or experiment["parallel_base_lora"]
            or not experiment["minikv_independent_prefill"] or experiment["unique_prefixes"] != 1
            or [r["id"] for r in records] != report["sample_ids"]
            or any(set(r["predictions"]) != set(MODES) for r in records)):
        raise RuntimeError("Expected identical samples and native sequential Base/Full/independent MiniKV")
    if (len(samples) != report["selected"] or len({s["prefix"] for s in samples}) != 1
            or hashlib.sha256(samples[0]["prefix"].encode()).hexdigest() != report["prefix_sha256"]):
        raise RuntimeError("Prepared fixed prefix or sample count differs from manifest")
    for row, sample in zip(records, samples):
        if (row["references"] != sample["references"] or row["prefix_key"] != report["prefix_key"]
                or row["input_tokens"] + args.max_new_tokens > report["context_limit"]):
            raise RuntimeError("Reference, cached prefix or context budget differs from prepared input")
    references = [sample["references"] for sample in samples]
    unfiltered, diagnostics = {}, {}
    for mode in MODES:
        raw = [row["predictions"][mode]["text"] for row in records]
        parsed = [extract_prediction(text) for text in raw]
        if value["scores"][mode] != score(raw, references, samples):
            raise RuntimeError(f"Independent score recomputation differs: {mode}")
        unfiltered[mode] = score_unfiltered(raw, references, samples)
        for row, answer in zip(records, parsed):
            row["predictions"][mode]["answer_extraction"] = answer
        diagnostics[mode] = dict(samples=len(raw), boundary_trimmed=sum(p["truncated"] for p in parsed),
            empty_answers=sum(p["empty_answer"] for p in parsed),
            stopped_on_eos=sum(row["predictions"][mode]["stopped_on_eos"] for row in records),
            reached_output_limit=sum(not row["predictions"][mode]["stopped_on_eos"] and
                row["predictions"][mode]["output_tokens"] == args.max_new_tokens for row in records))
    value.update(dataset_preparation=report, adapter_compatibility=compatibility,
                 generation_diagnostics=diagnostics, unfiltered_scores=unfiltered)
    value["config"]["prompt_protocol"] = PROTOCOL
    value["scorer_metadata"] = dict(policy=POLICY, primary_metric="generic_token_f1", metrics=list(METRICS),
        aggregation="macro mean over every selected sample, including empty/capped outputs",
        scale="fractions in [0, 1]", reference_independent=True, boundaries=BOUNDARIES,
        normalization="EM/token F1: lowercase, remove ASCII punctuation and English articles, split whitespace",
        rouge=dict(implementation="rouge-score", version=importlib.metadata.version("rouge-score"),
                   metric="rougeL", use_stemmer=True, aggregation="macro F1"),
        filtering="only explicit next Alpaca instruction outside fenced/indented code; same rule for all modes",
        limitation="Reserved headers can be legitimate answer content; unmarked continuation is retained. Raw scores also saved.",
        comparability=CAVEAT)
    experiment["scores_scale"] = "Generic EM/token F1 and ROUGE-L F1 fractions in [0, 1]"
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
        compatibility = check_adapter(args)
        compatibility["tokenizer_policy"] = (
            "base vocabulary and special tokens for all modes; chat-template equality checked for compatibility "
            "but chat wrapping is not used; prompts follow the Alpaca instruction format")
        score(["dependency check"], [["dependency check"]], [{}])
        update("preparing", adapter_compatibility=compatibility)
        report = prepare(args, data)
        if args.prepare_only:
            update("prepared", data=str(data), samples=report["selected"],
                   prefix_tokens=report["cached_prefix_tokens"], input_tokens_max=report["input_tokens_max"])
            return
        command = evaluator_command(args, data, result)
        update("running", command=command, samples=report["selected"], prefix_tokens=report["cached_prefix_tokens"])
        subprocess.run(command, check=True)
        if hashlib.sha256(data.read_bytes()).hexdigest() != report["output_sha256"]:
            raise RuntimeError("Prepared samples changed during evaluation")
        samples = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        value = finalize(result, samples, report, compatibility, args)
        update("complete", result=str(result), scores=value["scores"], generation_diagnostics=value["generation_diagnostics"])
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
