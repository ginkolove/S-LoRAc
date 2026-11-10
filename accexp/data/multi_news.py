"""Prepare complete Multi-News text pairs and identical examples for both Llama-2 models."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

if __package__:
    from .cnn_dailymail import build_samples
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.cnn_dailymail import build_samples


DATASET = "alexfabbri/multi_news"
MODEL_PATHS = {"7b": "/data/models_2/llama-2-7b-hf", "13b": "/data/hub/Llama2-13b-hf"}
INSTRUCTION = "Write a concise summary of the following news articles, combining their key information without repetition."
PROMPT_FORMAT = "local Multi-News Documents/Summary few-shot template; adapter training template undocumented"
ALPACA_SOURCE = "https://github.com/StyxXuan/LoraRetriever/blob/main/templates/alpaca.json"
ALPACA_PREAMBLE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROTOCOLS = ("legacy", "alpaca")


def document_prompt(document, protocol="legacy"):
    if protocol == "legacy":
        return INSTRUCTION + "\n\n### Documents:\n" + document + "\n\n### Summary: "
    if protocol == "alpaca":
        return ALPACA_PREAMBLE + "\n\n### Instruction:\n" + INSTRUCTION + "\n\n" + document + "\n\n### Response:\n"
    raise ValueError(f"Unknown Multi-News prompt protocol: {protocol}")


def document_count(document):
    return sum(bool(part.strip()) for part in document.split("|||||"))


class TextPairSplit:
    """Index line offsets without loading hundreds of MB of training news into RAM."""

    def __init__(self, directory, split, min_documents=2):
        root = Path(directory)
        root = root / "data" if (root / "data").is_dir() else root
        self.split, self.min_documents = split, min_documents
        if split not in ("train", "validation", "test") or min_documents < 1:
            raise ValueError("Invalid Multi-News split or minimum document count")
        stem = "val" if split == "validation" else split
        self.paths = [root / f"{stem}.src.cleaned", root / f"{stem}.tgt"]
        self.offsets, self.hashes = [], []
        for path in self.paths:
            positions, checksum = [], hashlib.sha256()
            with path.open("rb") as handle:
                while True:
                    position = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    positions.append(position)
                    checksum.update(line)
            self.offsets.append(positions)
            self.hashes.append(checksum.hexdigest())
        if not self.offsets[0] or len(self.offsets[0]) != len(self.offsets[1]):
            raise ValueError(f"Multi-News {split} source/target line counts differ or are empty")

    def __len__(self):
        return len(self.offsets[0])

    def get(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        texts = []
        for path, positions in zip(self.paths, self.offsets):
            with path.open("rb") as handle:
                handle.seek(positions[index])
                texts.append(handle.readline().decode("utf-8").strip())
        document, summary = texts
        # Match the original dataset reader, preserving the document separators.
        document = document.replace("NEWLINE_CHAR", "\n")
        count = document_count(document)
        return dict(id=f"{self.split}:{index}", article=document if count >= self.min_documents else "",
                    highlights=summary, num_documents=count)

    def batches(self, batch_size=64):
        if batch_size < 1:
            raise ValueError("batch-size must be positive")
        for start in range(0, len(self), batch_size):
            yield start, [self.get(i) for i in range(start, min(start + batch_size, len(self)))]

    def manifest(self):
        return [dict(path=str(path.resolve()), bytes=path.stat().st_size, lines=len(positions), sha256=checksum)
                for path, positions, checksum in zip(self.paths, self.offsets, self.hashes)]


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    metadata = Path(str(destination) + ".meta.json")
    if destination.exists() or metadata.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    tokenizers, limits = {}, {}
    for model in args.models:
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        declared = [int(value) for value in (getattr(config, "max_position_embeddings", None),
                                            getattr(config, "max_sequence_length", None)) if value]
        if not declared:
            raise ValueError(f"Model has no declared context limit: {model}")
        native = min(declared)
        limit = args.max_context_len or native
        if not 0 < limit <= native:
            raise ValueError(f"Requested context {limit} exceeds {model} context {native}")
        name = str(Path(model).resolve())
        limits[name] = limit
        tokenizers[name] = AutoTokenizer.from_pretrained(model, local_files_only=True)
    train = TextPairSplit(args.dataset_dir, "train", args.min_documents)
    test = TextPairSplit(args.dataset_dir, args.test_split, args.min_documents)
    first, *others = tokenizers.values()
    # Reuse complete-example selection, overlap exclusion and joint length checks.
    protocol = getattr(args, "prompt_format", "alpaca")
    prompt_format = (PROMPT_FORMAT if protocol == "legacy" else
                     "published Alpaca no-input outer wrapper; local Multi-News instruction and complete train demonstrations")
    samples, report = build_samples(train, test, first, additional_tokenizers=others,
        prompt_builder=partial(document_prompt, protocol=protocol), progress_desc=f"Check Multi-News {args.test_split} lengths",
        context_limit=min(limits.values()), max_new_tokens=args.max_new_tokens,
        num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
        sample_size=args.sample_size, seed=args.seed, test_split=args.test_split)
    for sample in samples:
        sample["id"] = f"multi_news:{args.test_split}:{sample['metadata']['source_index']}"
        meta = sample["metadata"]
        meta.pop("dataset_version")
        meta.pop("article_truncated")
        meta.update(dataset=DATASET, prompt_format=prompt_format, prompt_protocol=protocol,
                    num_shots=args.num_shots, context_limits=limits,
                    document_truncated=False, num_documents=test.get(meta["source_index"])["num_documents"])
    report.pop("dataset_version")
    report.pop("article_truncation")
    report.update(dataset=DATASET, context_limits=limits, prompt_format=prompt_format,
        prompt_protocol=protocol, outer_template_source=ALPACA_SOURCE if protocol == "alpaca" else None,
        prompt_caveat="outer wrapper matches published inference; task wording and few-shot extension are local, not verified training prompts",
        sample_ids=[sample["id"] for sample in samples], min_documents=args.min_documents,
        invalid_definition="empty document/summary or fewer than min_documents nonempty news documents",
        input_tokens_semantics="maximum over all prepared tokenizers; common minimum context limit",
        prefix_tokens_by_model={name: len(t.encode(samples[0]["prefix"], add_special_tokens=True))
                                for name, t in tokenizers.items()},
        source_files=dict(train=train.manifest(), test=test.manifest()),
        source_decoding="one source/target pair per line; restore NEWLINE_CHAR; preserve ||||| separators",
        document_truncation="none; exclude overlength complete examples",
        adapter_training_disjointness=f"official {args.test_split} split; adapter training membership not independently verified",
        cache_content="fixed complete train document/summary demonstrations; current news documents are in query")
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
    parser.add_argument("--dataset-dir", default="/data/hub/multi_news")
    parser.add_argument("--test-split", choices=["test", "validation"], default="test")
    parser.add_argument("--prompt-format", choices=PROTOCOLS, default="alpaca")
    parser.add_argument("--sample-size", type=int, default=1000, help="0 uses all eligible examples")
    parser.add_argument("--num-shots", type=int, default=1)
    parser.add_argument("--min-documents", type=int, default=2)
    parser.add_argument("--prefix-max-tokens", type=int, default=1024)
    parser.add_argument("--max-context-len", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.sample_size < 0 or min(args.num_shots, args.min_documents, args.prefix_max_tokens, args.max_new_tokens) < 1:
        parser.error("sample-size must be nonnegative; shots/documents/token budgets must be positive")
    if args.max_context_len is not None and args.max_context_len < 1:
        parser.error("max-context-len must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    report = prepare(args, args.output)
    print(json.dumps({key: report[key] for key in ("selected", "prefix_tokens_by_model", "context_limits",
        "test_eligible", "test_invalid", "test_overlength", "prefix_sha256")}, indent=2))
