"""Original Alpaca: fixed demonstrations and seeded samples from the remainder."""

import argparse
import hashlib
import json
from pathlib import Path
import random

if __package__:
    from .common import ALPACA_PROMPTS, alpaca_example, clean_examples, make_fewshot
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.common import ALPACA_PROMPTS, alpaca_example, clean_examples, make_fewshot


MODEL = "/data/hub/qwen2.5-7b-instruct"
LORA = "/data/hub/alpaca-lora"
DATASET = "/data/hub/alpaca"
PROTOCOL = "qwen_alpaca_fixed_fewshot_v1"
CAVEAT = ("Original Alpaca has only a train split. The adapter reports training on this dataset "
          "with val_set_size=0.0. Excluding demonstrations does not establish independence from "
          "adapter training. These scores measure reference overlap and MiniKV quality retention, "
          "not held-out generalization or an official instruction-following accuracy.")


def read_local(dataset_dir):
    import pyarrow.parquet as parquet

    root = Path(dataset_dir).resolve()
    files = sorted((root / "data").glob("train-*.parquet"))
    if not files:
        files = sorted(root.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No original Alpaca train parquet shards under {root}")
    rows, sources = [], []
    for path in files:
        table = parquet.read_table(path)
        if not {"instruction", "input", "output"} <= set(table.column_names):
            raise ValueError(f"Not an Alpaca instruction/input/output table: {path}")
        rows.extend(table.select(["instruction", "input", "output"]).to_pylist())
        sources.append(dict(path=str(path), rows=table.num_rows,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows, sources


def build_samples(rows, tokenizer, *, context_limit, max_new_tokens, num_shots,
                  prefix_max_tokens, sample_size, seed):
    if sample_size < 0:
        raise ValueError("sample-size must be nonnegative; 0 means all eligible examples")
    examples, cleaning = clean_examples(rows, alpaca_example)
    samples, report = make_fewshot(examples, None, tokenizer, dataset="alpaca",
        context_limit=context_limit, max_new_tokens=max_new_tokens, num_shots=num_shots,
        prefix_max_tokens=prefix_max_tokens, seed=seed, max_samples=0, add_special_tokens=False,
        split_info=dict(source="tatsu-lab/alpaca", official_splits=["train"], cleaning=cleaning))
    if sample_size > len(samples):
        raise ValueError(f"Requested {sample_size} examples, only {len(samples)} eligible after exclusions")
    if sample_size:
        samples = random.Random(seed).sample(samples, sample_size)
    for sample in samples:
        sample["metadata"].update(prompt_protocol=PROTOCOL, num_shots=num_shots,
                                  official_source_split="train", adapter_training_disjoint=False)
    by_id = {item["id"]: item for item in examples}
    report.update(selected=len(samples), sample_ids=[sample["id"] for sample in samples],
        selection="seeded uniform sample without replacement; no truncation or score-based selection",
        prompt_protocol=PROTOCOL, prompt_templates=ALPACA_PROMPTS,
        prompt_policy="Alpaca instruction/input/response format, as indicated by the adapter's Axolotl type: alpaca; no chat wrapping",
        demonstrations=[by_id[key] for key in report["fewshot_train_ids"]], training_overlap_caveat=CAVEAT,
        input_tokens_min=min(s["metadata"]["input_tokens"] for s in samples),
        input_tokens_max=max(s["metadata"]["input_tokens"] for s in samples))
    report["selected_test_examples"] = len(samples)
    report["adapter_training_disjointness"] = "not_held_out_from_adapter_training"
    return samples, report


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer
    if __package__ == "accexp.data":
        from ..eval_minikv import prepare_inputs
    else:
        from eval_minikv import prepare_inputs

    destination = Path(destination)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared samples: {destination}")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen2":
        raise ValueError(f"Expected Qwen2/Qwen2.5, got {config.model_type}")
    native = int(config.max_position_embeddings)
    limit = args.max_context_len or native
    if not 0 < limit <= native:
        raise ValueError(f"Requested context {limit} exceeds model context {native}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows, sources = read_local(args.dataset_dir)
    samples, report = build_samples(rows, tokenizer, context_limit=limit,
        max_new_tokens=args.max_new_tokens, num_shots=args.num_shots,
        prefix_max_tokens=args.prefix_max_tokens, sample_size=args.sample_size, seed=args.seed)
    prepared = prepare_inputs(samples, tokenizer, args.max_new_tokens, limit, add_special_tokens=False)
    if len({item["prefix_key"] for item in prepared}) != 1:
        raise ValueError("The text prefix does not yield one shared token prefix")
    for item in prepared:
        item["sample"]["metadata"].update(cached_prefix_tokens=len(item["prefix_ids"]),
                                          query_tokens=len(item["query_ids"]))
    report.update(source_files=sources, model=str(Path(args.model).resolve()), model_context_limit=native,
        cached_prefix_tokens=len(prepared[0]["prefix_ids"]), prefix_key=prepared[0]["prefix_key"],
        query_tokens_min=min(len(item["query_ids"]) for item in prepared),
        query_tokens_max=max(len(item["query_ids"]) for item in prepared),
        tokenizer_source="base model for all three evaluation modes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def add_arguments(parser):
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset-dir", default=DATASET)
    parser.add_argument("--sample-size", type=int, default=1000, help="0 uses all eligible examples")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--prefix-max-tokens", type=int, default=1024)
    parser.add_argument("--max-context-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)


def validate_arguments(parser, args):
    if min(args.num_shots, args.prefix_max_tokens, args.max_new_tokens) < 1 or args.sample_size < 0:
        parser.error("shots, prefix and output budgets must be positive; sample-size must be nonnegative")
    if args.max_context_len < 0 or 0 < args.max_context_len <= args.max_new_tokens + 1:
        parser.error("max-context-len must leave room for prefix, query and output; 0 uses model limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_arguments(parser, args)
    report = prepare(args, args.output)
    print(json.dumps({key: report[key] for key in ("selected", "num_shots", "prefix_tokens",
        "cached_prefix_tokens", "input_tokens_min", "input_tokens_max", "context_limit")}, indent=2))
