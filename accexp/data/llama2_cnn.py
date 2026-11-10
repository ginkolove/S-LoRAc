"""Prepare one fixed CNN/DailyMail prefix and identical samples for both Llama-2 models."""

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .cnn_dailymail import ParquetSplit, add_arguments, build_samples, validate_arguments
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.cnn_dailymail import ParquetSplit, add_arguments, build_samples, validate_arguments


MODEL_PATHS = {"7b": "/data/models_2/llama-2-7b-hf", "13b": "/data/hub/Llama2-13b-hf"}
PROMPT_FORMAT = "local fixed CNN/DailyMail Article/Summary few-shot template; adapter training template undocumented"


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
        key = str(Path(model).resolve())
        limits[key] = limit
        tokenizers[key] = AutoTokenizer.from_pretrained(model, local_files_only=True)
    train = ParquetSplit(args.dataset_dir, args.dataset_version, "train")
    test = ParquetSplit(args.dataset_dir, args.dataset_version, args.test_split)
    first, *others = tokenizers.values()
    samples, report = build_samples(train, test, first, additional_tokenizers=others,
        context_limit=min(limits.values()), max_new_tokens=args.max_new_tokens,
        num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
        sample_size=args.sample_size, seed=args.seed, dataset_version=args.dataset_version,
        test_split=args.test_split)
    for sample in samples:
        sample["metadata"]["prompt_format"] = PROMPT_FORMAT
        sample["metadata"]["context_limits"] = limits
    report.update(context_limits=limits, prompt_format=PROMPT_FORMAT,
        input_tokens_semantics="maximum over all prepared tokenizers; common minimum context limit",
        prefix_tokens_by_model={name: len(t.encode(samples[0]["prefix"], add_special_tokens=True))
                                for name, t in tokenizers.items()},
        source_files=dict(train=train.manifest(), test=test.manifest()),
        cache_content="fixed complete train article/summary demonstrations; current article is in query")
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
                                                 "test_eligible", "test_overlength", "prefix_sha256")}, indent=2))
