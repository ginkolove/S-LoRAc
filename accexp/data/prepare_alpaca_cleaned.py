"""Prepare a seeded Alpaca-Cleaned subset with one fixed few-shot prefix."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

if __package__:
    from .alpaca_cleaned import read_splits
    from .common import make_fewshot
else:
    from alpaca_cleaned import read_splits
    from common import make_fewshot


def select_samples(samples, count, seed):
    if count < 0 or count > len(samples):
        raise ValueError(f"sample-size must be between 0 (all) and {len(samples)}")
    indices = sorted(random.Random(seed).sample(range(len(samples)), count)) if count else list(range(len(samples)))
    selected = [samples[index] for index in indices]
    if len({sample["id"] for sample in selected}) != len(selected):
        raise ValueError("Selected test IDs are not unique")
    return selected, indices


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    raw = Path(args.dataset_dir) / "alpaca_data_cleaned.json"
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    limits = [int(value) for value in (getattr(config, "max_position_embeddings", None),
                                       getattr(config, "max_sequence_length", None)) if value]
    if not limits:
        raise ValueError("Model configuration does not declare a context limit")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, local_files_only=True)
    train, test, split_info = read_splits(SimpleNamespace(
        train_file=str(raw), test_file=None, local_files_only=True))
    samples, report = make_fewshot(
        train, test, tokenizer, dataset="alpaca_cleaned", context_limit=min(limits),
        max_new_tokens=args.max_new_tokens, num_shots=args.num_shots,
        prefix_max_tokens=args.prefix_max_tokens, seed=args.seed, max_samples=0,
        add_special_tokens=True, split_info=split_info)
    selected, indices = select_samples(samples, args.sample_size, args.seed)
    report.update(model=str(args.model), tokenizer=str(args.tokenizer or args.model),
                  raw_file=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                  population=len(samples), selected=len(selected), selected_test_examples=len(selected),
                  sampling="uniform without replacement; source order retained" if args.sample_size else "all eligible",
                  indices=indices, sample_ids=[sample["id"] for sample in selected],
                  unique_prefixes=len({sample["prefix"] for sample in selected}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def add_arguments(parser):
    parser.add_argument("--model", default="/data/hub/Llama-65B")
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset-dir", default="/data/hub/alpaca_cleaned")
    parser.add_argument("--sample-size", type=int, default=1000, help="0 uses all eligible non-demonstration examples")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--prefix-max-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)


def validate_arguments(parser, args):
    if args.sample_size < 0 or min(args.num_shots, args.prefix_max_tokens, args.max_new_tokens) < 1:
        parser.error("sample-size must be nonnegative; shots and token budgets must be positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_arguments(parser, args)
    report = prepare(args, args.output)
    print(json.dumps(dict(output=args.output, samples=report["selected"],
                          prefix_tokens=report["prefix_tokens"], context_limit=report["context_limit"],
                          prefix_sha256=report["prefix_sha256"]), indent=2))


if __name__ == "__main__":
    main()
