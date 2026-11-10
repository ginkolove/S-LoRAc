"""Qwen-native CNN/DailyMail: fixed complete train demonstrations, held-out articles."""

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .cnn_dailymail import ParquetSplit, add_arguments, build_samples, validate_arguments
    if __package__ == "data":
        from qwen_utils import chat_prompt_builders
    else:
        from ..qwen_utils import chat_prompt_builders
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.cnn_dailymail import ParquetSplit, add_arguments, build_samples, validate_arguments
    from qwen_utils import chat_prompt_builders


MODEL = "/data/hub/qwen2.5-7b-instruct"
LORA = "/data/hub/cnn_dailmail-lora/qwen2.5-7b"
PROTOCOL = "qwen_chat_cnn_fixed_fewshot_v1"
SYSTEM = ("You are a helpful assistant. Summarize the news article in a few concise sentences, "
          "including only its main highlights. Return only the summary.")


def article_user(article):
    return "Article:\n" + article


def prompt_builders(tokenizer):
    return chat_prompt_builders(tokenizer, SYSTEM, article_user)


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen2":
        raise ValueError(f"Expected Qwen2/Qwen2.5, got {config.model_type}")
    native = int(config.max_position_embeddings)
    limit = args.max_context_len or native
    if not 0 < limit <= native:
        raise ValueError(f"Requested context {limit} exceeds native context {native}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefix_builder, query_builder = prompt_builders(tokenizer)
    train = ParquetSplit(args.dataset_dir, args.dataset_version, "train")
    test = ParquetSplit(args.dataset_dir, args.dataset_version, args.test_split)
    samples, report = build_samples(train, test, tokenizer, context_limit=limit,
        max_new_tokens=args.max_new_tokens, num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
        sample_size=args.sample_size, seed=args.seed, dataset_version=args.dataset_version,
        test_split=args.test_split, prefix_builder=prefix_builder, prompt_builder=query_builder, prompt_format=PROTOCOL)
    messages = [dict(role="system", content=SYSTEM)]
    for demo in report["demonstrations"]:
        row = train.get(demo["source_index"])
        messages.extend((dict(role="user", content=article_user(row["article"].strip())),
                         dict(role="assistant", content=row["highlights"].strip())))
    for sample in samples:
        article = test.get(sample["metadata"]["source_index"])["article"].strip()
        expected = tokenizer.apply_chat_template(messages + [dict(role="user", content=article_user(article))],
                                                 tokenize=False, add_generation_prompt=True)
        text = sample["prefix"] + sample["query"]
        if text != expected or tokenizer.encode(text, add_special_tokens=False) != tokenizer.encode(text, add_special_tokens=True):
            raise ValueError("Prepared input differs from native chat template or duplicates special tokens")
        sample["metadata"].update(prompt_protocol=PROTOCOL, num_shots=args.num_shots)
    report.update(model=str(Path(args.model).resolve()), model_context_limit=native, prompt_protocol=PROTOCOL,
        system_prompt=SYSTEM, chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        tokenizer_source="base model, including for LoRA Full and MiniKV",
        prompt_caveat="Local fixed train few-shots using only the base model's native chat template; adapter training prompt unverified",
        input_tokens_min=min(s["metadata"]["input_tokens"] for s in samples),
        input_tokens_max=max(s["metadata"]["input_tokens"] for s in samples),
        source_files=dict(train=train.manifest(), test=test.manifest()),
        cache_content="system instruction and complete fixed train article/summary turns; current article only in query")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(model=MODEL, max_context_len=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    return args


if __name__ == "__main__":
    args = parse_args()
    report = prepare(args, args.output)
    print(json.dumps({key: report[key] for key in ("selected", "num_shots", "prefix_tokens", "context_limit",
        "input_tokens_min", "input_tokens_max", "test_eligible", "test_overlength", "max_new_tokens")}, indent=2))
