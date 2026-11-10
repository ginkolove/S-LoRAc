"""Prepare local CNN/DailyMail Parquet data for the three-way KV experiment."""

import argparse
from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re

from tqdm.auto import tqdm

if __package__:
    from .common import fingerprint
else:
    from common import fingerprint


DATASET = "abisee/cnn_dailymail"
INSTRUCTION = (
    "You are given a news article below. Please summarize the article, "
    "including only its highlights."
)


def article_prompt(article):
    return INSTRUCTION + "\n\n### Article: " + article + "\n\n### Summary: "


class ParquetSplit:
    """Read only the needed train row groups and stream the test split."""

    def __init__(self, directory, version, split):
        import pyarrow.parquet as pq

        directory = Path(directory)
        root = directory / version if (directory / version).is_dir() else directory
        self.paths = sorted(root.glob(f"{split}-*.parquet"))
        if not self.paths and (root / f"{split}.parquet").is_file():
            self.paths = [root / f"{split}.parquet"]
        if not self.paths:
            raise FileNotFoundError(f"No {split} Parquet files under {root}")
        shards = [re.fullmatch(rf"{split}-(\d+)-of-(\d+)\.parquet", path.name)
                  for path in self.paths]
        if all(shards):
            expected = {int(match[2]) for match in shards}
            if len(expected) != 1 or [int(match[1]) for match in shards] != list(range(expected.pop())):
                raise ValueError(f"Missing or inconsistent {split} Parquet shards under {root}")
        self.files = [pq.ParquetFile(path) for path in self.paths]
        self.groups, self.ends = [], []
        size = 0
        for file_index, source in enumerate(self.files):
            if not {"id", "article", "highlights"} <= set(source.schema_arrow.names):
                raise ValueError(f"{self.paths[file_index]} lacks id/article/highlights")
            for group in range(source.num_row_groups):
                size += source.metadata.row_group(group).num_rows
                self.groups.append((file_index, group))
                self.ends.append(size)
        self.size = size
        if not size:
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return self.size

    @lru_cache(maxsize=2)
    def _rows(self, file_index, group):
        return self.files[file_index].read_row_group(
            group, columns=["id", "article", "highlights"]).to_pylist()

    def get(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        group = bisect_right(self.ends, index)
        start = self.ends[group - 1] if group else 0
        return self._rows(*self.groups[group])[index - start]

    def batches(self, batch_size=64):
        index = 0
        for source in self.files:
            for batch in source.iter_batches(batch_size=batch_size, columns=["id", "article", "highlights"]):
                rows = batch.to_pylist()
                yield index, rows
                index += len(rows)

    def manifest(self):
        return [dict(path=str(path.resolve()), bytes=path.stat().st_size,
                     rows=source.metadata.num_rows)
                for path, source in zip(self.paths, self.files)]


def clean_row(row):
    if not all(isinstance(row.get(key), str) and row[key].strip()
               for key in ("id", "article", "highlights")):
        return None
    return {key: row[key].strip() for key in ("id", "article", "highlights")}


def build_samples(train, test, tokenizer, *, context_limit, max_new_tokens,
                  num_shots=3, prefix_max_tokens=1536, sample_size=1000, seed=42,
                  dataset_version="3.0.0", test_split="test", additional_tokenizers=(),
                  prompt_builder=article_prompt, progress_desc="Check CNN/DailyMail test lengths",
                  prefix_builder=None, prompt_format="predibase/cnn"):
    if min(context_limit, max_new_tokens, num_shots, prefix_max_tokens) < 1 or sample_size < 0:
        raise ValueError("Token budgets/shots must be positive; sample-size must be nonnegative")
    prefix_budget = min(prefix_max_tokens, context_limit - max_new_tokens - 1)
    if prefix_budget <= 0:
        raise ValueError("The output budget leaves no space for a prefix and article")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=True)
    tokenizers = [tokenizer, *additional_tokenizers]
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    prefix = prefix_builder([]) if prefix_builder else ""
    demos, demo_articles, demo_turns = [], set(), []
    checked = 0
    for index in indices:
        checked += 1
        row = clean_row(train.get(index))
        if row is None or fingerprint(row["article"]) in demo_articles:
            continue
        turns = demo_turns + [(row["article"], row["highlights"])]
        proposed = (prefix_builder(turns) if prefix_builder else
                    prefix + prompt_builder(row["article"]) + row["highlights"] + "\n\n")
        # Reserve room for each complete demonstration rather than truncating news.
        shot_budget = prefix_budget * (len(demos) + 1) // num_shots
        if any(len(t.encode(proposed, add_special_tokens=True)) > shot_budget for t in tokenizers):
            continue
        prefix = proposed
        demo_turns = turns
        demos.append(dict(id=row["id"], source_index=index))
        demo_articles.add(fingerprint(row["article"]))
        if len(demos) == num_shots:
            break
    if len(demos) != num_shots:
        raise ValueError(f"Only {len(demos)}/{num_shots} complete demonstrations fit {prefix_budget} tokens")

    samples, invalid, overlap, too_long, seen_ids = [], 0, 0, 0, set()
    demo_ids = [row["id"] for row in demos]
    with tqdm(total=len(test), desc=progress_desc, unit="article") as progress:
        for start, batch in test.batches():
            valid = []
            for offset, raw in enumerate(batch):
                row = clean_row(raw)
                if row is None:
                    invalid += 1
                    continue
                if row["id"] in seen_ids:
                    raise ValueError(f"Duplicate test source id: {row['id']}")
                seen_ids.add(row["id"])
                if row["id"] in demo_ids or fingerprint(row["article"]) in demo_articles:
                    overlap += 1
                    continue
                valid.append((start + offset, row, prompt_builder(row["article"])))
            if valid:
                lengths = [[len(ids) for ids in t([prefix + query for _, _, query in valid],
                            add_special_tokens=True, truncation=False, padding=False,
                            return_attention_mask=False)["input_ids"]] for t in tokenizers]
                for (index, row, query), input_tokens in zip(valid, map(max, zip(*lengths))):
                    if input_tokens + max_new_tokens > context_limit:
                        too_long += 1
                        continue
                    samples.append(dict(
                        id=f"cnn_dailymail:{test_split}:{row['id']}", prefix=prefix,
                        query=query, references=[row["highlights"]],
                        metadata=dict(dataset=DATASET, dataset_version=dataset_version,
                                      test_split=test_split, source_id=row["id"], source_index=index,
                                      fewshot_train_ids=demo_ids, seed=seed, input_tokens=input_tokens,
                                      context_limit=context_limit, reserved_max_new_tokens=max_new_tokens,
                                      prompt_format=prompt_format, article_truncated=False)))
            progress.update(len(batch))
    eligible = len(samples)
    if not eligible or sample_size > eligible:
        raise ValueError(f"Requested {sample_size} samples, but only {eligible} complete test articles fit")
    selected_indices = (sorted(random.Random(seed).sample(range(eligible), sample_size))
                        if sample_size else list(range(eligible)))
    selected = [samples[index] for index in selected_indices]
    report = dict(
        dataset=DATASET, dataset_version=dataset_version, test_split=test_split,
        seed=seed, num_shots=num_shots, fewshot_train_ids=demo_ids, demonstrations=demos,
        train_examples=len(train), train_candidates_checked=checked, prefix_max_tokens=prefix_budget,
        prefix_tokens=len(encode(prefix)), prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        context_limit=context_limit, max_new_tokens=max_new_tokens, test_examples=len(test),
        test_invalid=invalid, test_demo_overlap=overlap, test_overlength=too_long,
        test_eligible=eligible, selected=len(selected), selected_test_examples=len(selected),
        sampling="uniform without replacement; source order retained" if sample_size else "all eligible",
        indices=selected_indices, sample_ids=[row["id"] for row in selected], unique_prefixes=1,
        prompt_format=prompt_format + "; complete train article/summary demonstrations",
        article_truncation="none; overlength inputs excluded", reference_truncation="none",
        adapter_training_disjointness="official test split; adapter training membership not independently verified")
    return selected, report


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
        raise ValueError("The model configuration does not declare a context limit")
    native_limit = min(limits)
    window = getattr(config, "sliding_window", None)
    requested = args.max_context_len
    context_limit = min(native_limit, requested if requested else (window or native_limit))
    if requested and requested > native_limit:
        raise ValueError(f"Requested context {requested} exceeds model context {native_limit}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train = ParquetSplit(args.dataset_dir, args.dataset_version, "train")
    test = ParquetSplit(args.dataset_dir, args.dataset_version, args.test_split)
    samples, report = build_samples(
        train, test, tokenizer, context_limit=context_limit, max_new_tokens=args.max_new_tokens,
        num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
        sample_size=args.sample_size, seed=args.seed, dataset_version=args.dataset_version,
        test_split=args.test_split)
    report.update(model=str(args.model), model_context_limit=native_limit, sliding_window=window,
                  prefix_may_leave_sliding_window=bool(window and context_limit > window),
                  source_files=dict(train=train.manifest(), test=test.manifest()))
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
    parser.add_argument("--dataset-dir", default="/data/hub/cnn_dailmail")
    parser.add_argument("--dataset-version", default="3.0.0")
    parser.add_argument("--test-split", choices=["test", "validation"], default="test")
    parser.add_argument("--sample-size", type=int, default=1000, help="0 uses every eligible test article")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--prefix-max-tokens", type=int, default=1536)
    parser.add_argument("--max-context-len", type=int,
                        help="Default min(model context, sliding window); overlength articles are excluded")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)


def validate_arguments(parser, args):
    if args.sample_size < 0 or min(args.num_shots, args.prefix_max_tokens, args.max_new_tokens) < 1:
        parser.error("sample-size must be nonnegative; shots/token budgets must be positive")
    if args.max_context_len is not None and args.max_context_len < 1:
        parser.error("max-context-len must be positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validate_arguments(parser, args)
    report = prepare(args, args.output)
    print(json.dumps(dict(output=args.output, samples=report["selected"],
                          prefix_tokens=report["prefix_tokens"], context_limit=report["context_limit"],
                          test_eligible=report["test_eligible"], test_overlength=report["test_overlength"]), indent=2))


if __name__ == "__main__":
    main()
