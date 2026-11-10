"""Qwen chat-template GSM8K with one immutable prefix of complete train examples."""

import argparse
import hashlib
import json
from pathlib import Path
import re

if __package__:
    from .gsm8k import add_arguments, build_samples, read_split, validate_arguments
    if __package__ == "data":
        from gsm8k_score import gold_number
        from qwen_utils import chat_prompt_builders
    else:
        from ..gsm8k_score import gold_number
        from ..qwen_utils import chat_prompt_builders
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.gsm8k import add_arguments, build_samples, read_split, validate_arguments
    from gsm8k_score import gold_number
    from qwen_utils import chat_prompt_builders


MODEL = "/data/hub/qwen2.5-7b-instruct"
LORA = "/data/hub/gsm8k-lora/qwen2.5-7b"
PROTOCOL = "qwen_chat_fixed_fewshot_v1"
SYSTEM = ("You are a helpful assistant. Solve the math problem step by step. "
          "End your response with '#### <number>' containing the final numeric answer.")


def demonstration_answer(solution):
    reasoning = re.sub(r"<<[^<>]*>>", "", solution.split("####", 1)[0]).strip()
    return reasoning + "\n#### " + gold_number(solution)


def prompt_builders(tokenizer):
    return chat_prompt_builders(tokenizer, SYSTEM)


def prepare(args, destination):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(destination)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen2":
        raise ValueError(f"Expected a Qwen2/Qwen2.5 model, got {config.model_type}")
    native = int(config.max_position_embeddings)
    limit = args.max_context_len or native
    if not 0 < limit <= native:
        raise ValueError(f"Requested context {limit} exceeds native context {native}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefix_builder, query_builder = prompt_builders(tokenizer)
    train, train_files = read_split(args.dataset_dir, "train", args.dataset_config)
    test, test_files = read_split(args.dataset_dir, "test", args.dataset_config)
    samples, report = build_samples(train, test, tokenizer, context_limit=limit,
        max_new_tokens=args.max_new_tokens, num_shots=args.num_shots, prefix_max_tokens=args.prefix_max_tokens,
        sample_size=args.sample_size, seed=args.seed, dataset_config=args.dataset_config,
        prompt_builder=query_builder, prefix_builder=prefix_builder, answer_formatter=demonstration_answer,
        prompt_format=PROTOCOL)
    demo_messages = [(train[d["source_index"]]["question"].strip(),
                      demonstration_answer(train[d["source_index"]]["answer"].strip()))
                     for d in report["demonstrations"]]
    for sample in samples:
        question = test[sample["metadata"]["source_index"]]["question"].strip()
        messages = [dict(role="system", content=SYSTEM)]
        for q, answer in demo_messages:
            messages.extend((dict(role="user", content=q), dict(role="assistant", content=answer)))
        messages.append(dict(role="user", content=question))
        text = sample["prefix"] + sample["query"]
        expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text != expected or tokenizer.encode(text, add_special_tokens=False) != tokenizer.encode(text, add_special_tokens=True):
            raise ValueError("Prepared input differs from native chat template or would duplicate special tokens")
        sample["metadata"].update(prompt_protocol=PROTOCOL, num_shots=args.num_shots)
    report.update(model=str(Path(args.model).resolve()), model_context_limit=native, prompt_protocol=PROTOCOL,
        system_prompt=SYSTEM, chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        tokenizer_vocab_sha256=hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
        prompt_caveat="Local fixed train few-shots; adapter README does not document its exact GRPO training prompt",
        input_tokens_min=min(s["metadata"]["input_tokens"] for s in samples),
        input_tokens_max=max(s["metadata"]["input_tokens"] for s in samples),
        source_files=dict(train=train_files, test=test_files),
        cache_content="system instruction and complete fixed train examples; held-out test question only in query")
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
        "model_context_limit", "input_tokens_max", "max_new_tokens", "prefix_sha256")}, indent=2))
