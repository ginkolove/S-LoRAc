"""Prepare one fixed train prefix and the same CoQA dev turns for multiple models."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen

from tqdm.auto import tqdm

if __package__:
    if __package__ == "data":
        from coqa_score import INSTRUCTION
    else:
        from ..coqa_score import INSTRUCTION
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from coqa_score import INSTRUCTION


MODEL_PATHS = {"7b": "/data/models_2/llama-2-7b-hf", "13b": "/data/hub/Llama2-13b-hf"}
GOLD_URL = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"
GOLD_SHA256 = "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a"
DEFAULT_GOLD = Path(__file__).resolve().parents[1] / "results/coqa_annotations/coqa-dev-v1.0.json"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_gold_file(path=None, download=False):
    destination = Path(path) if path else DEFAULT_GOLD
    if destination.is_file():
        return destination
    if not download:
        raise FileNotFoundError("Original CoQA dev annotations are required; supply --gold-file or --download-gold")
    with urlopen(GOLD_URL, timeout=120) as response:
        content = response.read(16 * 1024 * 1024)
    if hashlib.sha256(content).hexdigest() != GOLD_SHA256:
        raise ValueError("CoQA dev download does not match the published dataset checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(content)
    destination.with_suffix(".source.json").write_text(json.dumps(
        dict(url=GOLD_URL, sha256=GOLD_SHA256), indent=2) + "\n")
    return destination


def read_split(directory, split):
    import pyarrow.parquet as pq

    root = Path(directory)
    root = root / "data" if (root / "data").is_dir() else root
    paths = sorted(root.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No CoQA {split} Parquet shards under {root}")
    matches = [re.fullmatch(rf"{split}-(\d+)-of-(\d+)\.parquet", path.name) for path in paths]
    if not all(matches) or len({int(m[2]) for m in matches}) != 1 or [int(m[1]) for m in matches] != list(range(int(matches[0][2]))):
        raise ValueError(f"Incomplete CoQA {split} shards")
    rows, manifest = [], []
    columns = ["story", "questions", "answers", "source"]
    for path in paths:
        source = pq.ParquetFile(path)
        if not set(columns) <= set(source.schema_arrow.names):
            raise ValueError(f"Missing CoQA fields in {path}")
        rows.extend(source.read(columns=columns).to_pylist())
        manifest.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, rows=source.metadata.num_rows))
    if not rows:
        raise ValueError(f"Empty CoQA {split}")
    return rows, manifest


def validate_conversation(row):
    questions, answers = row.get("questions"), row.get("answers", {}).get("input_text")
    if (not isinstance(row.get("story"), str) or not row["story"].strip()
            or not isinstance(questions, list) or not isinstance(answers, list)
            or not questions or len(questions) != len(answers)
            or not all(isinstance(x, str) and x.strip() for x in questions + answers)):
        raise ValueError("Malformed CoQA conversation")
    return questions, answers


def conversation_key(story, questions, answers):
    return digest(json.dumps([story, questions, answers], ensure_ascii=False))


def load_gold(path):
    result, seen_ids = {}, set()
    for row in json.loads(Path(path).read_text())["data"]:
        questions = [q["input_text"] for q in row["questions"]]
        primary = [a["input_text"] for a in row["answers"]]
        if [q["turn_id"] for q in row["questions"]] != list(range(1, len(questions) + 1)):
            raise ValueError("Nonsequential CoQA question turns")
        annotations = [row["answers"]] + list(row["additional_answers"].values())
        for answers in annotations:
            if len(answers) != len(questions) or [a["turn_id"] for a in answers] != list(range(1, len(questions) + 1)):
                raise ValueError("CoQA annotation turn IDs do not match questions")
            if not all(isinstance(a["input_text"], str) for a in answers):
                raise ValueError("Malformed CoQA reference")
        key = conversation_key(row["story"], questions, primary)
        if key in result or row["id"] in seen_ids:
            raise ValueError("Duplicate official CoQA conversation")
        seen_ids.add(row["id"])
        result[key] = dict(id=row["id"], source=row["source"],
                           references=[[answers[i]["input_text"] for answers in annotations]
                                       for i in range(len(questions))])
    if not result:
        raise ValueError("Empty CoQA annotations")
    return result


def question_prompt(row, turn_index):
    questions, answers = validate_conversation(row)
    if not 0 <= turn_index < len(questions):
        raise ValueError("Invalid CoQA turn index")
    text = "Passage: " + row["story"] + "\n\n"
    for index in range(turn_index):
        text += "Q: " + questions[index] + "\nA: " + answers[index] + "\n\n"
    return text + "Q: " + questions[turn_index] + "\nA:"


def build_samples(train, validation, gold, tokenizers, context_limits, *, max_new_tokens=64,
                  num_shots=3, demo_turns=2, prefix_max_tokens=1536, sample_size=1000, seed=42):
    if (not tokenizers or set(tokenizers) != set(context_limits) or sample_size < 0
            or min(max_new_tokens, num_shots, demo_turns, prefix_max_tokens, *context_limits.values()) < 1):
        raise ValueError("Invalid CoQA sampling/token budgets")
    budget = min(prefix_max_tokens, min(context_limits.values()) - max_new_tokens - 1)
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    prefix, demonstrations, demo_stories = INSTRUCTION + "\n\n", [], set()
    for index in order:
        row = train[index]
        questions, answers = validate_conversation(row)
        if digest(row["story"]) in demo_stories:
            continue
        turns = min(demo_turns, len(questions))
        candidate = prefix + question_prompt(row, turns - 1) + " " + answers[turns - 1] + "\n\n"
        allowance = budget * (len(demonstrations) + 1) // num_shots
        if any(len(tokenizer.encode(candidate, add_special_tokens=True)) > allowance for tokenizer in tokenizers.values()):
            continue
        prefix = candidate
        demonstrations.append(dict(source_index=index, story_sha256=digest(row["story"]), turns=turns))
        demo_stories.add(digest(row["story"]))
        if len(demonstrations) == num_shots:
            break
    if len(demonstrations) != num_shots:
        raise ValueError(f"Only {len(demonstrations)}/{num_shots} complete CoQA demonstrations fit {budget} tokens")

    candidates, seen, counts = [], set(), Counter()
    for index, row in enumerate(validation):
        questions, answers = validate_conversation(row)
        key = conversation_key(row["story"], questions, answers)
        if key in seen:
            raise ValueError("Duplicate local CoQA conversation")
        seen.add(key)
        annotation = gold.get(key)
        if annotation is None or annotation["source"] != row["source"]:
            raise ValueError(f"Original CoQA annotations do not match local conversation {index}")
        if digest(row["story"]) in demo_stories:
            counts["demo_overlap_turns"] += len(questions)
            continue
        for turn in range(len(questions)):
            refs = annotation["references"][turn]
            candidates.append(dict(
                id=f"coqa:validation:{annotation['id']}:{turn + 1}", prefix=prefix,
                query=question_prompt(row, turn), references=list(refs),
                metadata=dict(dataset="stanfordnlp/coqa", split="validation", story_id=annotation["id"],
                              source=row["source"], source_index=index, turn_id=turn + 1,
                              history_turns=turn, history_policy="all previous primary gold answers",
                              coqa_gold_answers=list(refs), seed=seed, reserved_max_new_tokens=max_new_tokens,
                              context_limits=context_limits, story_truncated=False, history_truncated=False)))
    eligible = []
    for start in tqdm(range(0, len(candidates), 64), desc="Check CoQA lengths for all models", unit="batch"):
        group = candidates[start:start + 64]
        lengths = {name: [len(tokens) for tokens in tokenizer(
            [prefix + sample["query"] for sample in group], add_special_tokens=True, truncation=False,
            padding=False, return_attention_mask=False)["input_ids"]] for name, tokenizer in tokenizers.items()}
        for i, sample in enumerate(group):
            if any(lengths[name][i] + max_new_tokens > context_limits[name] for name in tokenizers):
                counts["overlength_turns"] += 1
                continue
            sample["metadata"]["input_tokens_by_model"] = {name: lengths[name][i] for name in tokenizers}
            eligible.append(sample)
    if not eligible or sample_size > len(eligible):
        raise ValueError(f"Requested {sample_size} turns but only {len(eligible)} complete CoQA turns fit all models")
    chosen = sorted(random.Random(seed).sample(range(len(eligible)), sample_size)) if sample_size else list(range(len(eligible)))
    samples = [eligible[i] for i in chosen]
    return samples, dict(
        dataset="stanfordnlp/coqa", evaluation_split="validation", train_conversations=len(train),
        validation_conversations=len(validation), validation_turns=sum(len(r["questions"]) for r in validation),
        eligible_turns=len(eligible), excluded=dict(counts), selected=len(samples), seed=seed,
        sampling="uniform turns without replacement; source order retained" if sample_size else "all eligible turns",
        num_shots=num_shots, demo_turns=demo_turns, demonstrations=demonstrations,
        prefix_max_tokens=budget, prefix_sha256=digest(prefix), unique_prefixes=1,
        prefix_tokens_by_model={name: len(t.encode(prefix, add_special_tokens=True)) for name, t in tokenizers.items()},
        context_limits=context_limits, max_new_tokens=max_new_tokens,
        sample_ids=[s["id"] for s in samples], selected_turns_by_source=dict(Counter(s["metadata"]["source"] for s in samples)),
        history_policy="all previous primary gold Q/A; no current answer or future turns; not free-running dialogue",
        prompt_format="local Passage/Q/A few-shot format; adapter's exact training template is undocumented",
        cache_content="fixed train demonstrations only; current passage and dialogue history are in query",
        truncation="none; exclude turns that exceed any model's context budget",
        adapter_training_disjointness="official dev; public adapter training membership not independently verified")


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    metadata = Path(str(destination) + ".meta.json")
    if destination.exists() or metadata.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    tokenizers, limits = {}, {}
    for model in args.models:
        name = str(Path(model).resolve())
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        native = int(config.max_position_embeddings)
        limit = args.max_context_len or native
        if not 0 < limit <= native:
            raise ValueError(f"Requested context {limit} exceeds {model} context {native}")
        tokenizers[name] = AutoTokenizer.from_pretrained(model, local_files_only=True)
        limits[name] = limit
    gold_path = ensure_gold_file(args.gold_file, args.download_gold)
    train, train_files = read_split(args.dataset_dir, "train")
    validation, dev_files = read_split(args.dataset_dir, "validation")
    samples, report = build_samples(train, validation, load_gold(gold_path), tokenizers, limits,
                                   max_new_tokens=args.max_new_tokens, num_shots=args.num_shots,
                                   demo_turns=args.demo_turns, prefix_max_tokens=args.prefix_max_tokens,
                                   sample_size=args.sample_size, seed=args.seed)
    report.update(source_files=dict(train=train_files, validation=dev_files), gold_file=str(gold_path.resolve()),
                  gold_sha256=hashlib.sha256(gold_path.read_bytes()).hexdigest(), gold_source=GOLD_URL)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with metadata.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS.values()))
    parser.add_argument("--dataset-dir", default="/data/hub/coqa")
    parser.add_argument("--gold-file")
    parser.add_argument("--download-gold", action="store_true")
    parser.add_argument("--sample-size", type=int, default=1000, help="0 uses every eligible dev turn")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--demo-turns", type=int, default=2)
    parser.add_argument("--prefix-max-tokens", type=int, default=1536)
    parser.add_argument("--max-context-len", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.sample_size < 0 or min(args.num_shots, args.demo_turns, args.prefix_max_tokens, args.max_new_tokens) < 1:
        parser.error("sample-size must be nonnegative; shots/turns/token budgets must be positive")
    if args.max_context_len is not None and args.max_context_len < 1:
        parser.error("max-context-len must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    report = prepare(args, args.output)
    print(json.dumps({key: report[key] for key in ("selected", "prefix_tokens_by_model", "context_limits",
                                                 "eligible_turns", "excluded", "prefix_sha256")}, indent=2))
