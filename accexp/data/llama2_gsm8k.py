"""Shared Llama-2 GSM8K samples with a fixed train prefix and LEMA-style prompts."""

import argparse
import hashlib
import json
from pathlib import Path
import re

if __package__:
    from .gsm8k import add_arguments, build_samples, read_split, validate_arguments
    if __package__ == "data":
        from gsm8k_score import gold_number
    else:
        from ..gsm8k_score import gold_number
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.gsm8k import add_arguments, build_samples, read_split, validate_arguments
    from gsm8k_score import gold_number


MODEL_PATHS = {"7b": "/data/models_2/llama-2-7b-hf", "13b": "/data/hub/Llama2-13b-hf"}
PROMPT_PREFIX = "Please solve the following math problem."
PROMPT_SOURCE = "https://github.com/microsoft/LEMA/blob/main/test_data/gsm8k_test.jsonl"
PROMPT_FORMAT = "LEMA question/step-by-step answer envelope with local fixed train few-shots"


def question_prompt(question):
    return PROMPT_PREFIX + "\nQuestion: " + question + "\nAnswer: Let's think step by step.\n"


def demonstration_answer(solution):
    final = gold_number(solution)
    reasoning = re.sub(r"<<[^<>]*>>", "", solution.split("####", 1)[0]).strip()
    return reasoning + "\nThe answer is " + final + "."


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    tokenizers, limits = {}, {}
    for model in args.models:
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        declared = [int(v) for v in (getattr(config, "max_position_embeddings", None),
                                    getattr(config, "max_sequence_length", None)) if v]
        if not declared:
            raise ValueError(f"No model context limit: {model}")
        native = min(declared)
        limit = args.max_context_len or native
        if not 0 < limit <= native:
            raise ValueError(f"Requested context {limit} exceeds {model} context {native}")
        name = str(Path(model).resolve())
        limits[name] = limit
        tokenizers[name] = AutoTokenizer.from_pretrained(model, local_files_only=True)
    train, train_files = read_split(args.dataset_dir, "train", args.dataset_config)
    test, test_files = read_split(args.dataset_dir, "test", args.dataset_config)
    first, *others = tokenizers.values()
    samples, report = build_samples(train, test, first, additional_tokenizers=others,
        prompt_builder=question_prompt, answer_formatter=demonstration_answer, prompt_format=PROMPT_FORMAT,
        context_limit=min(limits.values()), max_new_tokens=args.max_new_tokens,
        num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
        sample_size=args.sample_size, seed=args.seed, dataset_config=args.dataset_config)
    for sample in samples:
        sample["metadata"].update(context_limits=limits, num_shots=args.num_shots, prompt_protocol="lema_fixed_fewshot")
    report.update(context_limits=limits, prompt_protocol="lema_fixed_fewshot", prompt_source=PROMPT_SOURCE,
        prompt_caveat="Published LEMA question envelope; local train few-shots, not identical to official zero-shot evaluation",
        demonstration_format="complete reasoning without redundant <<calculator>> annotations; final number as The answer is ...",
        input_tokens_semantics="maximum over all prepared tokenizers; common minimum context limit",
        prefix_tokens_by_model={name: len(t.encode(samples[0]["prefix"], add_special_tokens=True))
                                for name, t in tokenizers.items()},
        source_files=dict(train=train_files, test=test_files),
        cache_content="fixed train question/solution examples only; current test question is in query")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, include_model=False)
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS.values()))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    return args


if __name__ == "__main__":
    args = parse_args()
    report = prepare(args, args.output)
    print(json.dumps({key: report[key] for key in ("selected", "num_shots", "prefix_tokens_by_model",
        "context_limits", "test_eligible", "test_overlength", "max_new_tokens", "prefix_sha256")}, indent=2))
