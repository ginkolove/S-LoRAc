"""Small shared helpers for reproducible few-shot accuracy inputs."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

from tqdm.auto import tqdm


DATASETS = ("guanaco", "alpaca_cleaned", "alpaca_mt")
ALPACA_PROMPTS = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def read_json(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected a JSON array or JSONL objects")
    return rows


def hf_json(repo, filename, local_files_only=False):
    from huggingface_hub import hf_hub_download

    return read_json(hf_hub_download(repo, filename, repo_type="dataset",
                                    local_files_only=local_files_only))


def fingerprint(text):
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def alpaca_example(row, index, templates=ALPACA_PROMPTS):
    instruction, context, answer = (row.get(key) for key in ("instruction", "input", "output"))
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(answer, str) or not answer.strip():
        return None
    if context is None:
        context = ""
    if not isinstance(context, str):
        return None
    template = templates["prompt_input" if context.strip() else "prompt_no_input"]
    query = template.format(instruction=instruction, input=context)
    return dict(id=str(row.get("id", index)), query=query, references=[answer.strip()],
                group_key=fingerprint(json.dumps([instruction, context], ensure_ascii=False)))


def clean_examples(rows, formatter):
    examples, seen, seen_ids = [], set(), set()
    invalid = duplicates = 0
    for index, row in enumerate(rows):
        item = formatter(row, index)
        if item is None:
            invalid += 1
            continue
        key = fingerprint(item["query"])
        if key in seen:
            duplicates += 1
            continue
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate source id for different questions: {item['id']}")
        seen.add(key)
        seen_ids.add(item["id"])
        item["query_key"] = key
        examples.append(item)
    return examples, dict(raw=len(rows), invalid=invalid, duplicate_queries=duplicates)


def make_fewshot(train, test, tokenizer, *, dataset, context_limit, max_new_tokens,
                 num_shots=3, prefix_max_tokens=1024, seed=42, max_samples=0,
                 holdout_fraction=None, add_special_tokens=True, split_info=None):
    if num_shots < 1 or prefix_max_tokens < 1 or max_new_tokens < 1 or max_samples < 0:
        raise ValueError("Shots, prefix budget and output budget must be positive; sample limit must be nonnegative")
    if holdout_fraction is not None and not 0 < holdout_fraction < 1:
        raise ValueError("holdout-fraction must be strictly between 0 and 1")
    info = dict(split_info or {})
    use_remainder = test is None and holdout_fraction is None
    if use_remainder:
        info.update(test_source="dataset_remainder_after_fewshot")
    elif test is None:
        keys = sorted({item["group_key"] for item in train})
        if len(keys) < 2:
            raise ValueError("Need at least two distinct question/conversation groups for a holdout")
        random.Random(seed).shuffle(keys)
        count = max(1, min(len(keys) - 1, round(len(keys) * holdout_fraction)))
        held_out = set(keys[:count])
        test = [item for item in train if item["group_key"] in held_out]
        train = [item for item in train if item["group_key"] not in held_out]
        info.update(test_source="explicit_train_holdout", holdout_fraction=holdout_fraction)
    elif holdout_fraction is not None:
        raise ValueError("Use either a test split/file or a train holdout, not both")
    if not train or (not use_remainder and not test):
        raise ValueError("Both a nonempty train pool and a nonempty test pool are required")

    # Only question/history identities are used here, never test answers. Keep
    # sibling Guanaco paths from the same root conversation out of demonstrations.
    test_groups = {item["group_key"] for item in test or ()}
    test_queries = {item["query_key"] for item in test or ()}
    candidates = [item for item in train if item["group_key"] not in test_groups
                  and item["query_key"] not in test_queries]
    overlap_count = len(train) - len(candidates)
    random.Random(seed).shuffle(candidates)
    prefix_budget = min(prefix_max_tokens, context_limit - max_new_tokens - 1)
    if prefix_budget <= 0:
        raise ValueError("Output budget leaves no room for a prefix and query")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=add_special_tokens)
    prefix, demos, selected_groups = "", [], set()
    for item in candidates:
        if item["group_key"] in selected_groups:
            continue
        proposed = prefix + item["query"] + item["references"][0] + "\n\n"
        if len(encode(proposed)) <= prefix_budget:
            prefix = proposed
            demos.append(item)
            selected_groups.add(item["group_key"])
        if len(demos) == num_shots:
            break
    if len(demos) != num_shots:
        raise ValueError(
            f"Only {len(demos)} of {num_shots} train demonstrations fit prefix budget {prefix_budget}. "
            "Reduce --num-shots or increase --prefix-max-tokens."
        )

    excluded_from_test = 0
    if use_remainder:
        demo_queries = {item["query_key"] for item in demos}
        test = [item for item in train if item["group_key"] not in selected_groups
                and item["query_key"] not in demo_queries]
        excluded_from_test = len(train) - len(test)
        if not test:
            raise ValueError("No test examples remain after selecting the few-shot demonstrations")

    samples, too_long = [], 0
    prefix_tokens = len(encode(prefix))
    demo_ids = [item["id"] for item in demos]
    for item in tqdm(test, desc=f"Prepare {dataset} test", unit="sample"):
        length = len(encode(prefix + item["query"]))
        if length + max_new_tokens > context_limit:
            too_long += 1
            continue
        samples.append(dict(
            id=f"{dataset}:test:{item['id']}", prefix=prefix, query=item["query"],
            references=item["references"],
            metadata=dict(dataset=dataset, source_id=item["id"], fewshot_train_ids=demo_ids,
                          seed=seed, context_limit=context_limit, reserved_max_new_tokens=max_new_tokens,
                          input_tokens=length, test_source=info.get("test_source", "provided_test"))))
    eligible = len(samples)
    if max_samples:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("No test requests fit the fixed few-shot prefix and output budget")
    report = dict(dataset=dataset, seed=seed, num_shots=num_shots, fewshot_train_ids=demo_ids,
                  prefix_tokens=prefix_tokens, prefix_max_tokens=prefix_budget,
                  prefix_sha256=hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
                  context_limit=context_limit, max_new_tokens=max_new_tokens,
                  add_special_tokens=add_special_tokens,
                  train_examples=len(train), train_overlap_excluded=overlap_count,
                  fewshot_examples_excluded_from_test=excluded_from_test,
                  test_examples=len(test), test_overlength=too_long, test_eligible=eligible,
                  selected_test_examples=len(samples), split_info=info,
                  adapter_training_disjointness="not_verified")
    return samples, report


def add_dataset_arguments(parser):
    parser.add_argument("--train-file", help="Optional raw train JSON/JSONL override")
    parser.add_argument("--test-file", help="Optional raw held-out test JSON/JSONL")
    parser.add_argument("--holdout-fraction", type=float,
                        help="Explicitly split a train-only dataset; does not prove LoRA training disjointness")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--prefix-max-tokens", type=int, default=1024)
    parser.add_argument("--language", default="en", choices=["en", "pt", "es", "ca", "eu", "gl", "at"])


def validate_dataset_arguments(parser, args):
    if args.num_shots < 1 or args.prefix_max_tokens < 1:
        parser.error("num-shots and prefix-max-tokens must be positive")
    if args.holdout_fraction is not None and not 0 < args.holdout_fraction < 1:
        parser.error("holdout-fraction must be strictly between 0 and 1")
    if args.holdout_fraction is not None and args.test_file:
        parser.error("test-file and holdout-fraction are mutually exclusive")


def load_dataset_samples(args, tokenizer, context_limit):
    reader = importlib.import_module(f"{__package__}.{args.dataset}")
    return prepare_with_reader(reader.read_splits, args, tokenizer, context_limit, args.dataset)


def prepare_with_reader(read_splits, args, tokenizer, context_limit, dataset):
    train, test, info = read_splits(args)
    samples, report = make_fewshot(
        train, test, tokenizer, dataset=dataset, context_limit=context_limit,
        max_new_tokens=args.max_new_tokens, num_shots=args.num_shots,
        prefix_max_tokens=args.prefix_max_tokens, seed=args.seed, max_samples=args.max_samples,
        holdout_fraction=args.holdout_fraction, add_special_tokens=not args.no_add_special_tokens,
        split_info=info)
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
    return samples, report


def export_cli(read_splits, dataset):
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(description=f"Prepare fixed train few-shot + {dataset} test inputs")
    parser.add_argument("--model", required=True, help="Read context length and tokenizer from this model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", required=True, help="Prepared JSONL for accexp/eval.py --data")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    add_dataset_arguments(parser)
    args = parser.parse_args()
    validate_dataset_arguments(parser, args)
    if args.max_new_tokens < 1 or args.max_samples < 0:
        parser.error("Output budget must be positive; sample limit must be nonnegative")
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    limits = [int(value) for value in (getattr(config, "max_position_embeddings", None),
                                      getattr(config, "max_sequence_length", None)) if value]
    if not limits:
        parser.error("Model configuration does not declare a context length")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                              local_files_only=args.local_files_only)
    samples, report = prepare_with_reader(read_splits, args, tokenizer, min(limits), dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    Path(str(output) + ".meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(f"Prepared {len(samples)} samples: {output}")
