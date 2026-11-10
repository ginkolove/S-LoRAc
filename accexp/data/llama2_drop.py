"""Prepare identical DROP examples and a fixed train prefix for both local Llama-2 models."""

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .drop import add_arguments, build_samples, ensure_gold_file, load_gold, read_split, validate_arguments, GOLD_URL
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.drop import add_arguments, build_samples, ensure_gold_file, load_gold, read_split, validate_arguments, GOLD_URL


MODEL_PATHS = {"7b": "/data/models_2/llama-2-7b-hf", "13b": "/data/hub/Llama2-13b-hf"}
PROMPT_FORMAT = "local fixed DROP few-shot Passage/Question/Answer template; adapter training template undocumented"


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    metadata = Path(str(destination) + ".meta.json")
    if destination.exists() or metadata.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    tokenizers, limits = {}, {}
    for model in args.models:
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        native = min(int(value) for value in (getattr(config, "max_position_embeddings", None),
                                            getattr(config, "max_sequence_length", None)) if value)
        limit = args.max_context_len or native
        if not 0 < limit <= native:
            raise ValueError(f"Requested context {limit} exceeds {model} context {native}")
        key = str(Path(model).resolve())
        limits[key] = limit
        tokenizers[key] = AutoTokenizer.from_pretrained(model, local_files_only=True)
    gold_path = ensure_gold_file(args.dataset_dir, args.gold_file, args.gold_archive, args.download_gold)
    train, train_files = read_split(args.dataset_dir, "train")
    validation, validation_files = read_split(args.dataset_dir, "validation")
    first, *others = tokenizers.values()
    samples, report = build_samples(train, validation, load_gold(gold_path), first,
        additional_tokenizers=others, context_limit=min(limits.values()), max_new_tokens=args.max_new_tokens,
        num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
        sample_size=args.sample_size, seed=args.seed)
    for sample in samples:
        sample["metadata"]["prompt_format"] = PROMPT_FORMAT
        sample["metadata"]["context_limits"] = limits
    report.update(context_limits=limits, prompt_format=PROMPT_FORMAT,
        input_tokens_semantics="maximum over all prepared tokenizers; common minimum context limit",
        prefix_tokens_by_model={name: len(t.encode(samples[0]["prefix"], add_special_tokens=True))
                                for name, t in tokenizers.items()},
        source_files=dict(train=train_files, validation=validation_files),
        gold_file=str(gold_path.resolve()), gold_url=GOLD_URL,
        gold_sha256=hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        cache_content="fixed train demonstrations only; current evidence passage is in query")
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
    add_arguments(parser, include_model=False)
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS.values()))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    return args


if __name__ == "__main__":
    args = parse_args()
    report = prepare(args, args.output)
    print(json.dumps({key: report[key] for key in ("selected", "prefix_tokens_by_model", "context_limits",
                                                 "validation_eligible", "validation_overlength", "prefix_sha256")}, indent=2))
