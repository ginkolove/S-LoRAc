"""Sequential Base, LoRA Full and independent MiniKV on shared Llama-2 GSM8K data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .data.llama2_gsm8k import MODEL_PATHS, PROMPT_SOURCE
    from .gsm8k_score import gold_number
    from .llama2_gsm8k_score import POLICY_NAME, extract_prediction, score, score_unfiltered
else:
    from data.llama2_gsm8k import MODEL_PATHS, PROMPT_SOURCE
    from gsm8k_score import gold_number
    from llama2_gsm8k_score import POLICY_NAME, extract_prediction, score, score_unfiltered


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=list(MODEL_PATHS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--lora")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--weight-format", choices=["bin", "safetensors", "auto"])
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, help="Default to the common prepared dataset's budget")
    args = parser.parse_args(argv)
    args.model = args.model or MODEL_PATHS[args.size]
    args.lora = args.lora or f"/data/hub/gsm8k-lora/llama2-{args.size}"
    args.batch_size = args.batch_size if args.batch_size is not None else (8 if args.size == "7b" else 4)
    args.weight_format = args.weight_format or ("safetensors" if args.size == "7b" else "bin")
    if args.batch_size < 1 or args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("batch-size and max-new-tokens must be positive")
    return args


def load_prepared(args):
    path = Path(args.data).resolve()
    report = json.loads(Path(str(path) + ".meta.json").read_text())
    if (report.get("dataset") != "openai/gsm8k" or report.get("evaluation_split") != "test"
            or report.get("prompt_protocol") != "lema_fixed_fewshot"
            or hashlib.sha256(path.read_bytes()).hexdigest() != report["output_sha256"]):
        raise ValueError("Llama-2 GSM8K data, prompt protocol or preparation checksum does not match")
    if str(Path(args.model).resolve()) not in report["context_limits"]:
        raise ValueError("Prepare GSM8K with this model included in --models first")
    if args.max_new_tokens is not None and args.max_new_tokens != report["max_new_tokens"]:
        raise ValueError("Output budget must match the common prepared dataset")
    for field in ("max_new_tokens", "num_shots", "prefix_max_tokens"):
        setattr(args, field, report[field])
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (not samples or len(samples) != report["selected"] or [s["id"] for s in samples] != report["sample_ids"]
            or len({s["id"] for s in samples}) != len(samples)
            or len({s["prefix"] for s in samples}) != 1
            or hashlib.sha256(samples[0]["prefix"].encode()).hexdigest() != report["prefix_sha256"]
            or report["num_shots"] != len(report["demonstrations"])):
        raise ValueError("GSM8K sample IDs, demonstration count or fixed prefix differ from preparation report")
    for sample in samples:
        if sample["references"] != [gold_number(sample["metadata"]["gold_solution"])]:
            raise ValueError("GSM8K numeric references differ from original solutions")
    score([""] * len(samples), [s["references"] for s in samples], samples)
    return samples, report


def evaluator_command(args, result, seed):
    directory = Path(__file__).resolve().parent
    return [sys.executable, "-u", str(directory / "eval.py"),
            "--model", args.model, "--lora", args.lora, "--data", str(Path(args.data).resolve()),
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--dtype", args.dtype, "--weight-format", args.weight_format,
            "--attn-implementation", args.attn_implementation,
            "--max-new-tokens", str(args.max_new_tokens), "--max-samples", "0", "--seed", str(seed),
            "--num-shots", str(args.num_shots), "--prefix-max-tokens", str(args.prefix_max_tokens),
            "--local-files-only", "--minikv-independent-prefill",
            "--scorer", str(directory / "llama2_gsm8k_score.py") + ":score", "--output", str(result)]


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result, status = output / "result.json", output / "status.json"
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
        from transformers import AutoConfig
        from peft import PeftConfig

        update("checking")
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        adapter = PeftConfig.from_pretrained(args.lora, local_files_only=True)
        expected = f"meta-llama/Llama-2-{args.size}-hf"
        if config.model_type != "llama" or adapter.base_model_name_or_path != expected:
            raise ValueError(f"Expected Llama-2-{args.size} and adapter base {expected}; got {adapter.base_model_name_or_path}")
        shape = (config.hidden_size, config.num_hidden_layers, config.num_attention_heads)
        expected_shape = (4096, 32, 32) if args.size == "7b" else (5120, 40, 40)
        if shape != expected_shape:
            raise ValueError(f"Model shape {shape} does not match Llama-2-{args.size} {expected_shape}")
        samples, report = load_prepared(args)
        command = evaluator_command(args, result, report["seed"])
        update("running", command=command, samples=len(samples), result=str(result))
        subprocess.run(command, check=True)
        value = json.loads(result.read_text())
        modes = ("base", "lora_full", "lora_minikv")
        if value["experiment"]["parallel_base_lora"] or not value["experiment"]["minikv_independent_prefill"]:
            raise RuntimeError("Expected independent single-device sequential evaluation")
        if [r["id"] for r in value["predictions"]] != report["sample_ids"] or any(
                set(r["predictions"]) != set(modes) for r in value["predictions"]):
            raise RuntimeError("Missing or mismatched GSM8K samples/modes")
        references = [s["references"] for s in samples]
        unfiltered, diagnostics = {}, {}
        for mode in modes:
            raw = [row["predictions"][mode]["text"] for row in value["predictions"]]
            parsed = [extract_prediction(text) for text in raw]
            if value["scores"][mode] != score(raw, references, samples):
                raise RuntimeError(f"GSM8K {mode} scores differ on independent recomputation")
            unfiltered[mode] = score_unfiltered(raw, references, samples)
            for row, answer in zip(value["predictions"], parsed):
                row["predictions"][mode]["answer_extraction"] = answer
            diagnostics[mode] = dict(samples=len(raw), boundary_trimmed=sum(p["boundary_trimmed"] for p in parsed),
                missing_or_invalid_strict_answer=sum(p["strict"] is None for p in parsed),
                flexible_fallback_used=sum(p["reason"] == "last_number_fallback" for p in parsed),
                stopped_on_eos=sum(r["predictions"][mode]["stopped_on_eos"] for r in value["predictions"]),
                reached_output_limit=sum(not r["predictions"][mode]["stopped_on_eos"] and
                                         r["predictions"][mode]["output_tokens"] == args.max_new_tokens
                                         for r in value["predictions"]))
        value["config"].update(num_shots=report["num_shots"], prefix_max_tokens=report["prefix_max_tokens"],
                               prompt_protocol=report["prompt_protocol"])
        value["dataset_preparation"] = report
        value["unfiltered_scores"] = unfiltered
        value["generation_diagnostics"] = diagnostics
        value["scorer_metadata"] = dict(policy=POLICY_NAME, prompt_source=PROMPT_SOURCE,
            primary_metric="exact_match", metrics=["exact_match", "flexible_exact_match"],
            scale="fractions in [0, 1]", reference_independent=True,
            strict="numeric prefix after first The answer is / The final answer is / ####; malformed first answer is binding",
            flexible="strict answer, or last numeric token only when no declared answer exists",
            boundary="line-start next LEMA instruction or Question:/### Question:",
            normalization="exact Decimal equality; no arithmetic evaluation or ground-truth matching",
            comparability="local fixed train few-shots and output budget, not an identical official LEMA setup")
        value["experiment"]["scores_scale"] = "GSM8K numeric EM fractions in [0, 1]; multiply by 100 for percentages"
        temporary = result.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(result)
        update("complete", result=str(result), scores=value["scores"], generation_diagnostics=diagnostics)
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
