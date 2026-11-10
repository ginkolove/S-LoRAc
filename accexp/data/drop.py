"""Prepare local DROP validation with complete train demonstrations and original gold groups."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen
import zipfile

from tqdm.auto import tqdm

if __package__:
    from .common import fingerprint
    if __package__ == "data":
        from drop_score import answer_spans
    else:
        from ..drop_score import answer_spans
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from drop_score import answer_spans
    from data.common import fingerprint


GOLD_URL = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip"
DEFAULT_GOLD = Path(__file__).resolve().parents[1] / "results/drop_annotations/drop_dataset_dev.json"
INSTRUCTION = (
    "Given a passage, you need to accurately identify and extract relevant spans of text that answer "
    "specific questions. Provide concise and coherent responses based on the information present in the passage."
)
OUTPUT_FORMAT = "Give only the answer. If multiple answer spans are required, return a JSON array of strings.\n\n"


def question_prompt(passage, question):
    return INSTRUCTION + "\n\n### Passage: " + passage + "\n### Question: " + question + "\n### Answer:"


def ensure_gold_file(dataset_dir, gold_file=None, archive=None, download=False):
    destination = Path(gold_file) if gold_file else DEFAULT_GOLD
    candidates = ([destination] if gold_file else [Path(dataset_dir) / "drop_dataset_dev.json",
                  Path(dataset_dir) / "drop_dataset/drop_dataset_dev.json", destination])
    for path in candidates:
        if path.is_file():
            return path
    if archive:
        data = Path(archive).read_bytes()
    elif download:
        with urlopen(GOLD_URL, timeout=120) as response:
            data = response.read(32 * 1024 * 1024)
    else:
        raise FileNotFoundError(
            "Original DROP validation annotations are required: use --gold-file PATH, "
            "--gold-archive PATH, or --download-gold once. The HF Parquet flattens multi-span groups."
        )
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        members = [name for name in source.namelist() if name.endswith("/drop_dataset_dev.json")
                   or name == "drop_dataset_dev.json"]
        if len(members) != 1 or source.getinfo(members[0]).file_size > 32 * 1024 * 1024:
            raise ValueError("Archive must contain exactly one reasonably sized DROP dev annotation file")
        content = source.read(members[0])
    if not isinstance(json.loads(content), dict):
        raise ValueError("Invalid original DROP annotations")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(content)
    provenance = dict(source_url=GOLD_URL, archive_sha256=hashlib.sha256(data).hexdigest(),
                      gold_sha256=hashlib.sha256(content).hexdigest())
    destination.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination


def read_split(directory, split):
    import pyarrow.parquet as pq

    root = Path(directory)
    if (root / "data").is_dir():
        root = root / "data"
    paths = sorted(root.glob(f"{split}-*.parquet"))
    if not paths and (root / f"{split}.parquet").is_file():
        paths = [root / f"{split}.parquet"]
    if not paths:
        raise FileNotFoundError(f"No DROP {split} Parquet files under {root}")
    shards = [re.fullmatch(rf"{split}-(\d+)-of-(\d+)\.parquet", path.name) for path in paths]
    if all(shards):
        totals = {int(match[2]) for match in shards}
        if len(totals) != 1 or [int(match[1]) for match in shards] != list(range(totals.pop())):
            raise ValueError(f"Incomplete {split} Parquet shards")
    rows, manifest = [], []
    columns = ["section_id", "query_id", "passage", "question", "answers_spans"]
    for path in paths:
        source = pq.ParquetFile(path)
        if not set(columns) <= set(source.schema_arrow.names):
            raise ValueError(f"{path} lacks DROP fields")
        rows.extend(source.read(columns=columns).to_pylist())
        manifest.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, rows=source.metadata.num_rows))
    if not rows:
        raise ValueError(f"Empty DROP {split}")
    return rows, manifest


def load_gold(path):
    source = json.loads(Path(path).read_text())
    result = {}
    for section_id, section in source.items():
        for qa in section["qa_pairs"]:
            key = (section_id, qa["query_id"], qa["question"].strip())
            if key in result:
                raise ValueError(f"Duplicate original DROP question key: {key}")
            answers, types = [], []
            for annotation in [qa["answer"]] + (qa.get("validated_answers") or []):
                try:
                    spans, kind = answer_spans(annotation)
                except ValueError:
                    continue
                if spans[0].strip() and spans not in answers:
                    answers.append(spans)
                    types.append(kind)
            result[key] = dict(passage=section["passage"].strip(), answers=answers, types=types)
    return result


def clean_row(row):
    fields = ("section_id", "query_id", "passage", "question")
    if not all(isinstance(row.get(name), str) and row[name].strip() for name in fields):
        return None
    return {**row, **{name: row[name].strip() for name in fields}}


def train_answer(row):
    answer = row.get("answers_spans", {})
    spans, kinds = answer.get("spans", []), answer.get("types", [])
    if not spans or len(spans) != len(kinds) or not all(isinstance(span, str) and span.strip() for span in spans):
        raise ValueError("Malformed train answer")
    if not (all(kind == "span" for kind in kinds) or len(kinds) == 1 and kinds[0] in ("number", "date")):
        raise ValueError("Ambiguous train answer types")
    return spans[0] if len(spans) == 1 else json.dumps(spans, ensure_ascii=False)


def build_samples(train, validation, gold, tokenizer, *, context_limit, max_new_tokens,
                  num_shots=3, prefix_max_tokens=1536, sample_size=1000, seed=42,
                  additional_tokenizers=()):
    if min(context_limit, max_new_tokens, num_shots, prefix_max_tokens) < 1 or sample_size < 0:
        raise ValueError("Invalid DROP sampling/token budgets")
    prefix_budget = min(prefix_max_tokens, context_limit - max_new_tokens - 1)
    tokenizers = [tokenizer, *additional_tokenizers]
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    prefix, demos, demo_passages, demo_ids = OUTPUT_FORMAT, [], set(), set()
    for index in indices:
        row = clean_row(train[index])
        if row is None or fingerprint(row["passage"]) in demo_passages:
            continue
        try:
            answer = train_answer(row)
        except ValueError:
            continue
        proposed = prefix + question_prompt(row["passage"], row["question"]) + " " + answer + "\n\n"
        if any(len(t.encode(proposed, add_special_tokens=True)) > prefix_budget * (len(demos) + 1) // num_shots
               for t in tokenizers):
            continue
        prefix = proposed
        demos.append(dict(section_id=row["section_id"], query_id=row["query_id"], source_index=index))
        demo_ids.add(row["query_id"])
        demo_passages.add(fingerprint(row["passage"]))
        if len(demos) == num_shots:
            break
    if len(demos) != num_shots:
        raise ValueError(f"Only {len(demos)}/{num_shots} complete DROP demonstrations fit {prefix_budget} tokens")

    eligible, counts, seen = [], Counter(), set()
    query_ids = Counter(row.get("query_id") for row in validation)
    for start in tqdm(range(0, len(validation), 64), desc="Check DROP validation lengths", unit="batch"):
        valid = []
        for index in range(start, min(start + 64, len(validation))):
            row = clean_row(validation[index])
            if row is None:
                counts["invalid"] += 1
                continue
            key = (row["section_id"], row["query_id"], row["question"])
            if key in seen:
                raise ValueError(f"Duplicate validation question: {key}")
            seen.add(key)
            annotation = gold.get(key)
            if annotation is None or annotation["passage"] != row["passage"]:
                raise ValueError(f"Original gold does not match local validation question: {key}")
            if not annotation["answers"]:
                counts["empty_gold"] += 1
                continue
            if row["query_id"] in demo_ids or fingerprint(row["passage"]) in demo_passages:
                counts["demo_overlap"] += 1
                continue
            valid.append((index, row, annotation, question_prompt(row["passage"], row["question"])))
        if not valid:
            continue
        lengths = [[len(ids) for ids in t([prefix + query for _, _, _, query in valid], add_special_tokens=True,
                    truncation=False, padding=False, return_attention_mask=False)["input_ids"]] for t in tokenizers]
        for (index, row, annotation, query), input_tokens in zip(valid, map(max, zip(*lengths))):
            if input_tokens + max_new_tokens > context_limit:
                counts["overlength"] += 1
                continue
            eligible.append(dict(
                id=f"drop:validation:{row['section_id']}:{row['query_id']}:{index}",
                prefix=prefix, query=query,
                references=["; ".join(answer) for answer in annotation["answers"]],
                metadata=dict(dataset="ucinlp/drop", split="validation", section_id=row["section_id"],
                              query_id=row["query_id"], source_index=index, seed=seed,
                              fewshot_train_ids=[demo["query_id"] for demo in demos],
                              drop_gold_answers=[list(answer) for answer in annotation["answers"]],
                              drop_gold_types=annotation["types"], input_tokens=input_tokens,
                              context_limit=context_limit, reserved_max_new_tokens=max_new_tokens,
                              passage_truncated=False, prompt_format="predibase/drop + fixed multi-span output instruction")))
    if not eligible or sample_size > len(eligible):
        raise ValueError(f"Requested {sample_size} samples, but only {len(eligible)} complete validation questions fit")
    selected_indices = sorted(random.Random(seed).sample(range(len(eligible)), sample_size)) if sample_size else list(range(len(eligible)))
    samples = [eligible[index] for index in selected_indices]
    return samples, dict(
        dataset="ucinlp/drop", evaluation_split="validation", train_examples=len(train),
        validation_examples=len(validation), seed=seed, num_shots=num_shots, demonstrations=demos,
        prefix_tokens=len(tokenizer.encode(prefix, add_special_tokens=True)), prefix_max_tokens=prefix_budget,
        prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(), unique_prefixes=1,
        context_limit=context_limit, max_new_tokens=max_new_tokens, validation_eligible=len(eligible),
        validation_invalid=counts["invalid"], validation_empty_gold=counts["empty_gold"],
        validation_demo_overlap=counts["demo_overlap"], validation_overlength=counts["overlength"],
        repeated_source_query_ids=sum(count - 1 for count in query_ids.values() if count > 1),
        selected=len(samples), indices=selected_indices, sample_ids=[row["id"] for row in samples],
        sampling="uniform without replacement; source order retained" if sample_size else "all eligible",
        answer_groups="original answer plus validated_answers; multi-span groups retained",
        prompt_format="predibase/drop + fixed multi-span output instruction",
        passage_truncation="none; overlength inputs excluded", reference_truncation="none",
        adapter_training_disjointness="official validation; adapter training membership not independently verified")


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    gold_path = ensure_gold_file(args.dataset_dir, args.gold_file, args.gold_archive, args.download_gold)
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
    train, train_files = read_split(args.dataset_dir, "train")
    validation, validation_files = read_split(args.dataset_dir, "validation")
    samples, report = build_samples(train, validation, load_gold(gold_path), tokenizer,
                                   context_limit=limit, max_new_tokens=args.max_new_tokens,
                                   num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
                                   sample_size=args.sample_size, seed=args.seed)
    report.update(model=str(args.model), model_context_limit=native, sliding_window=window,
                  prefix_may_leave_sliding_window=bool(window and limit > window),
                  source_files=dict(train=train_files, validation=validation_files),
                  gold_file=str(gold_path.resolve()), gold_url=GOLD_URL,
                  gold_sha256=hashlib.sha256(gold_path.read_bytes()).hexdigest())
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
    parser.add_argument("--dataset-dir", default="/data/hub/drop")
    parser.add_argument("--gold-file", help="Original DROP dev JSON; otherwise use local/cache lookup")
    parser.add_argument("--gold-archive", help="Local official ZIP to extract dev JSON once")
    parser.add_argument("--download-gold", action="store_true", help="Allow downloading the official 8 MB ZIP if gold is absent")
    parser.add_argument("--sample-size", type=int, default=1000, help="0 uses all eligible validation questions")
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
    report = prepare(args, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
