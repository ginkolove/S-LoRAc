"""Prepare fixed train few-shots and local GSM8K main test questions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from tqdm.auto import tqdm

if __package__:
    from .common import fingerprint
    if __package__ == "data":
        from gsm8k_score import PROMPT_PREFIX, gold_number
    else:
        from ..gsm8k_score import PROMPT_PREFIX, gold_number
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.common import fingerprint
    from gsm8k_score import PROMPT_PREFIX, gold_number


def question_prompt(question):
    return PROMPT_PREFIX + question + "\nAnswer:"


def read_split(directory, split, config="main"):
    import pyarrow.parquet as pq

    root = Path(directory)
    if (root / config).is_dir():
        root = root / config
    elif root.name in ("main", "socratic") and root.name != config:
        raise ValueError(f"Requested GSM8K {config}, but supplied the {root.name} directory")
    paths = sorted(root.glob(f"{split}-*.parquet"))
    if not paths and (root / f"{split}.parquet").is_file():
        paths = [root / f"{split}.parquet"]
    if not paths:
        raise FileNotFoundError(f"No GSM8K {config}/{split} Parquet files under {root}")
    shards = [re.fullmatch(rf"{split}-(\d+)-of-(\d+)\.parquet", path.name) for path in paths]
    if all(shards):
        totals = {int(match[2]) for match in shards}
        if len(totals) != 1 or [int(match[1]) for match in shards] != list(range(totals.pop())):
            raise ValueError(f"Incomplete GSM8K {split} Parquet shards")
    rows, manifest = [], []
    for path in paths:
        source = pq.ParquetFile(path)
        if not {"question", "answer"} <= set(source.schema_arrow.names):
            raise ValueError(f"{path} lacks question/answer fields")
        rows.extend(source.read(columns=["question", "answer"]).to_pylist())
        manifest.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, rows=source.metadata.num_rows))
    if not rows:
        raise ValueError(f"Empty GSM8K {split}")
    return rows, manifest


def clean_row(row):
    if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("question", "answer")):
        return None
    return {key: row[key].strip() for key in ("question", "answer")}


def build_samples(train, test, tokenizer, *, context_limit, max_new_tokens,
                  num_shots=3, prefix_max_tokens=1536, sample_size=1000, seed=42,
                  dataset_config="main", additional_tokenizers=(),
                  prompt_builder=question_prompt, answer_formatter=None,
                  prompt_format="predibase/gsm8k", prefix_builder=None):
    if min(context_limit, max_new_tokens, num_shots, prefix_max_tokens) < 1 or sample_size < 0:
        raise ValueError("Invalid GSM8K sampling/token budgets")
    budget = min(prefix_max_tokens, context_limit - max_new_tokens - 1)
    tokenizers = [tokenizer, *additional_tokenizers]
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    prefix = prefix_builder([]) if prefix_builder else ""
    demonstrations, demo_questions, demo_turns = [], set(), []
    for index in indices:
        row = clean_row(train[index])
        if row is None or fingerprint(row["question"]) in demo_questions:
            continue
        gold_number(row["answer"])
        answer = answer_formatter(row["answer"]) if answer_formatter else row["answer"]
        turns = demo_turns + [(row["question"], answer)]
        proposed = (prefix_builder(turns) if prefix_builder else
                    prefix + prompt_builder(row["question"]) + " " + answer + "\n\n")
        if any(len(t.encode(proposed, add_special_tokens=True)) > budget * (len(demonstrations) + 1) // num_shots
               for t in tokenizers):
            continue
        prefix = proposed
        demo_turns = turns
        demonstrations.append(dict(source_index=index, question_sha256=fingerprint(row["question"])))
        demo_questions.add(fingerprint(row["question"]))
        if len(demonstrations) == num_shots:
            break
    if len(demonstrations) != num_shots:
        raise ValueError(f"Only {len(demonstrations)}/{num_shots} complete GSM8K demonstrations fit {budget} tokens")

    eligible, counts, seen = [], Counter(), set()
    for start in tqdm(range(0, len(test), 64), desc="Check GSM8K test lengths", unit="batch"):
        valid = []
        for index in range(start, min(start + 64, len(test))):
            row = clean_row(test[index])
            if row is None:
                counts["invalid"] += 1
                continue
            final = gold_number(row["answer"])
            key = fingerprint(row["question"])
            if key in seen:
                raise ValueError(f"Duplicate GSM8K test question at index {index}")
            seen.add(key)
            if key in demo_questions:
                counts["demo_overlap"] += 1
                continue
            valid.append((index, row, final, prompt_builder(row["question"])))
        if not valid:
            continue
        lengths = [[len(ids) for ids in t([prefix + query for _, _, _, query in valid], add_special_tokens=True,
                    truncation=False, padding=False, return_attention_mask=False)["input_ids"]] for t in tokenizers]
        for (index, row, final, query), input_tokens in zip(valid, map(max, zip(*lengths))):
            if input_tokens + max_new_tokens > context_limit:
                counts["overlength"] += 1
                continue
            eligible.append(dict(
                id=f"gsm8k:{dataset_config}:test:{index}", prefix=prefix, query=query, references=[final],
                metadata=dict(dataset="openai/gsm8k", dataset_config=dataset_config, split="test",
                              source_index=index, question_sha256=fingerprint(row["question"]), seed=seed,
                              fewshot_train_indices=[demo["source_index"] for demo in demonstrations],
                              gold_solution=row["answer"], gold_final_answer=final,
                              input_tokens=input_tokens, context_limit=context_limit,
                              reserved_max_new_tokens=max_new_tokens, question_truncated=False,
                              prompt_format=prompt_format)))
    if not eligible or sample_size > len(eligible):
        raise ValueError(f"Requested {sample_size} samples, but only {len(eligible)} complete GSM8K test questions fit")
    selected_indices = (sorted(random.Random(seed).sample(range(len(eligible)), sample_size))
                        if sample_size else list(range(len(eligible))))
    samples = [eligible[index] for index in selected_indices]
    return samples, dict(
        dataset="openai/gsm8k", dataset_config=dataset_config, evaluation_split="test",
        train_examples=len(train), test_examples=len(test), seed=seed, num_shots=num_shots,
        demonstrations=demonstrations, prefix_max_tokens=budget,
        prefix_tokens=len(tokenizer.encode(prefix, add_special_tokens=True)),
        prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(), unique_prefixes=1,
        context_limit=context_limit, max_new_tokens=max_new_tokens, test_eligible=len(eligible),
        test_invalid=counts["invalid"], test_demo_overlap=counts["demo_overlap"],
        test_overlength=counts["overlength"], selected=len(samples), indices=selected_indices,
        sample_ids=[sample["id"] for sample in samples],
        sampling="uniform without replacement; source order retained" if sample_size else "all eligible",
        prompt_format=prompt_format, demonstrations_include_complete_reasoning=True,
        question_truncation="none; overlength inputs excluded", reference_truncation="none",
        adapter_training_disjointness="official test split; adapter training membership not independently verified")


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    limits = [int(value) for value in (getattr(config, "max_position_embeddings", None),
                                      getattr(config, "max_sequence_length", None)) if value]
    if not limits:
        raise ValueError("Model has no declared context limit")
    native = min(limits)
    window = getattr(config, "sliding_window", None)
    limit = args.max_context_len or min(native, window or native)
    if not 0 < limit <= native:
        raise ValueError(f"Requested context {limit} exceeds model context {native}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train, train_files = read_split(args.dataset_dir, "train", args.dataset_config)
    test, test_files = read_split(args.dataset_dir, "test", args.dataset_config)
    samples, report = build_samples(train, test, tokenizer, context_limit=limit,
                                   max_new_tokens=args.max_new_tokens, num_shots=args.num_shots,
                                   prefix_max_tokens=args.prefix_max_tokens, sample_size=args.sample_size,
                                   seed=args.seed, dataset_config=args.dataset_config)
    report.update(model=str(args.model), model_context_limit=native, sliding_window=window,
                  prefix_may_leave_sliding_window=bool(window and limit > window),
                  source_files=dict(train=train_files, test=test_files))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def add_arguments(parser, include_model=True):
    if include_model:
        parser.add_argument("--model", default="/data/hub/mistral-7b-v0.1")
    parser.add_argument("--dataset-dir", default="/data/hub/gsm8k")
    parser.add_argument("--dataset-config", choices=["main", "socratic"], default="main")
    parser.add_argument("--sample-size", type=int, default=1000, help="0 uses all eligible test questions")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--prefix-max-tokens", type=int, default=1536)
    parser.add_argument("--max-context-len", type=int, help="Default min(model context, sliding window)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)


def validate_arguments(parser, args):
    if args.sample_size < 0 or min(args.num_shots, args.prefix_max_tokens, args.max_new_tokens) < 1:
        parser.error("sample-size must be nonnegative; shots/token budgets must be positive")
    if args.max_context_len is not None and args.max_context_len < 1:
        parser.error("max-context-len must be positive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_arguments(parser, args)
    print(json.dumps(prepare(args, args.output), ensure_ascii=False, indent=2))
