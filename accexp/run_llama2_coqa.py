"""Evaluate Llama-2 Base, LoRA Full and independent MiniKV sequentially on one GPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from .coqa_score import OFFICIAL_SCORER, POLICY_NAME, extract_answer, score, score_unfiltered
    from .data.coqa import MODEL_PATHS
else:
    from coqa_score import OFFICIAL_SCORER, POLICY_NAME, extract_answer, score, score_unfiltered
    from data.coqa import MODEL_PATHS


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
    parser.add_argument("--max-new-tokens", type=int, help="Default to the shared prepared dataset's budget")
    args = parser.parse_args(argv)
    args.model = args.model or MODEL_PATHS[args.size]
    args.lora = args.lora or f"/data/hub/coqa-lora/llama2-{args.size}"
    args.batch_size = args.batch_size if args.batch_size is not None else (20 if args.size == "7b" else 8)
    args.weight_format = args.weight_format or ("safetensors" if args.size == "7b" else "bin")
    if args.batch_size < 1 or args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("batch-size and max-new-tokens must be positive")
    return args


def load_prepared(args):
    path = Path(args.data).resolve()
    report = json.loads(Path(str(path) + ".meta.json").read_text())
    if report.get("dataset") != "stanfordnlp/coqa" or hashlib.sha256(path.read_bytes()).hexdigest() != report["output_sha256"]:
        raise ValueError("CoQA data or its preparation checksum does not match")
    if str(Path(args.model).resolve()) not in report["context_limits"]:
        raise ValueError("Prepare CoQA with this model included in --models first")
    if args.max_new_tokens is not None and args.max_new_tokens != report["max_new_tokens"]:
        raise ValueError("Output budget must match the common prepared dataset")
    args.max_new_tokens = report["max_new_tokens"]
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (not samples or len(samples) != report["selected"] or [s["id"] for s in samples] != report["sample_ids"]
            or len({s["id"] for s in samples}) != len(samples)
            or len({s["prefix"] for s in samples}) != 1
            or hashlib.sha256(samples[0]["prefix"].encode()).hexdigest() != report["prefix_sha256"]):
        raise ValueError("CoQA sample IDs or fixed prefix differ from preparation report")
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
            "--local-files-only", "--minikv-independent-prefill",
            "--scorer", str(directory / "coqa_score.py") + ":score", "--output", str(result)]


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result, status = output / "result.json", output / "status.json"
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite results: {result}")
    with status.open("x") as handle:
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
        if config.model_type != "llama":
            raise ValueError(f"Expected Llama model, got {config.model_type}")
        adapter = PeftConfig.from_pretrained(args.lora, local_files_only=True)
        expected = f"meta-llama/Llama-2-{args.size}-hf"
        if adapter.base_model_name_or_path != expected:
            raise ValueError(f"Adapter base is {adapter.base_model_name_or_path}, expected {expected}")
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
            raise RuntimeError("Missing or mismatched CoQA samples/modes")
        references = [s["references"] for s in samples]
        unfiltered, diagnostics = {}, {}
        for mode in modes:
            raw = [row["predictions"][mode]["text"] for row in value["predictions"]]
            parsed = [extract_answer(text) for text in raw]
            if value["scores"][mode] != score(raw, references, samples):
                raise RuntimeError(f"CoQA {mode} scores differ on independent recomputation")
            unfiltered[mode] = score_unfiltered(raw, references, samples)
            for row, answer in zip(value["predictions"], parsed):
                row["predictions"][mode]["answer_extraction"] = answer
            diagnostics[mode] = dict(samples=len(raw), boundary_trimmed=sum(p["boundary_trimmed"] for p in parsed),
                stopped_on_eos=sum(r["predictions"][mode]["stopped_on_eos"] for r in value["predictions"]),
                reached_output_limit=sum(not r["predictions"][mode]["stopped_on_eos"] and
                                         r["predictions"][mode]["output_tokens"] == args.max_new_tokens
                                         for r in value["predictions"]))
        value["dataset_preparation"] = report
        value["unfiltered_scores"] = unfiltered
        value["generation_diagnostics"] = diagnostics
        value["scorer_metadata"] = dict(source=OFFICIAL_SCORER, primary_metric="coqa_f1", policy=POLICY_NAME,
            metrics=["exact_match", "coqa_f1"], scale="fractions in [0, 1]",
            aggregation="official leave-one-annotator-out maxima averaged over annotators, then turns",
            reference_independent_filter=True, output_filtering="only explicit line-start next-question/passage boundaries",
            comparability="local few-shot dev subset with gold dialogue history; not official test leaderboard")
        temporary = result.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(result)
        update("complete", result=str(result), scores=value["scores"], generation_diagnostics=diagnostics)
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
