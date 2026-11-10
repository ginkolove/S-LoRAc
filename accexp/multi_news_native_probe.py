"""Locked validation-only native HF comparison of Multi-News prompts and demonstrations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from tqdm.auto import tqdm

if __package__:
    from .data.common import fingerprint
    from .data.cnn_dailymail import clean_row
    from .data.multi_news import ALPACA_SOURCE, DATASET, MODEL_PATHS, PROTOCOLS, TextPairSplit, document_prompt
    from .multi_news_score import TURN_POLICY, extract_summary, score, score_turn_filtered
else:
    from data.common import fingerprint
    from data.cnn_dailymail import clean_row
    from data.multi_news import ALPACA_SOURCE, DATASET, MODEL_PATHS, PROTOCOLS, TextPairSplit, document_prompt
    from multi_news_score import TURN_POLICY, extract_summary, score, score_turn_filtered


CONDITIONS = ("legacy_zero_shot", "legacy_fixed_prefix", "alpaca_zero_shot", "alpaca_fixed_prefix")


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer

    destination = Path(args.data)
    manifest = Path(str(destination) + ".meta.json")
    if destination.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite probe data: {destination}")
    tokenizers, limits = {}, {}
    for model in args.models:
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        declared = [int(v) for v in (getattr(config, "max_position_embeddings", None),
                                    getattr(config, "max_sequence_length", None)) if v]
        if not declared:
            raise ValueError(f"No model context limit: {model}")
        name = str(Path(model).resolve())
        limits[name] = min(declared)
        tokenizers[name] = AutoTokenizer.from_pretrained(model, local_files_only=True)
    limit = min(limits.values())
    budget = min(args.prefix_max_tokens, limit - args.max_new_tokens - 1)
    train = TextPairSplit(args.dataset_dir, "train", args.min_documents)
    validation = TextPairSplit(args.dataset_dir, "validation", args.min_documents)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    prefixes, demonstrations, demo_documents = dict.fromkeys(PROTOCOLS, ""), [], set()
    for index in order:
        row = clean_row(train.get(index))
        if row is None or fingerprint(row["article"]) in demo_documents:
            continue
        proposed = {p: prefixes[p] + document_prompt(row["article"], p) + row["highlights"] + "\n\n" for p in PROTOCOLS}
        allowance = budget * (len(demonstrations) + 1) // args.num_shots
        if any(len(t.encode(text, add_special_tokens=True)) > allowance
               for text in proposed.values() for t in tokenizers.values()):
            continue
        prefixes = proposed
        demonstrations.append(dict(source_index=index, id=row["id"]))
        demo_documents.add(fingerprint(row["article"]))
        if len(demonstrations) == args.num_shots:
            break
    if len(demonstrations) != args.num_shots:
        raise ValueError("Complete common demonstrations do not fit all prompt protocols/models")
    eligible, invalid, overlap, overlength = [], 0, 0, 0
    for start, rows in tqdm(validation.batches(), total=(len(validation) + 63) // 64,
                            desc="Check common validation prompts", unit="batch"):
        batch = []
        for offset, raw in enumerate(rows):
            row = clean_row(raw)
            if row is None:
                invalid += 1
                continue
            if fingerprint(row["article"]) in demo_documents:
                overlap += 1
                continue
            prompts = {}
            for protocol in PROTOCOLS:
                query = document_prompt(row["article"], protocol)
                prompts[protocol + "_zero_shot"] = query
                prompts[protocol + "_fixed_prefix"] = prefixes[protocol] + query
            batch.append(dict(id=f"multi_news:validation:{start + offset}", references=[row["highlights"]],
                prompts=prompts, metadata=dict(source_index=start + offset, num_documents=raw["num_documents"])))
        if not batch:
            continue
        lengths = {name: {condition: [len(ids) for ids in tokenizer(
            [sample["prompts"][condition] for sample in batch], add_special_tokens=True,
            truncation=False, padding=False, return_attention_mask=False)["input_ids"]]
            for condition in CONDITIONS} for name, tokenizer in tokenizers.items()}
        for i, sample in enumerate(batch):
            if any(lengths[name][condition][i] + args.max_new_tokens > limits[name]
                   for name in tokenizers for condition in CONDITIONS):
                overlength += 1
                continue
            sample["metadata"]["input_tokens"] = {name: {c: lengths[name][c][i] for c in CONDITIONS} for name in tokenizers}
            eligible.append(sample)
    if args.sample_size > len(eligible):
        raise ValueError(f"Requested {args.sample_size} validation samples, only {len(eligible)} fit every condition/model")
    selected = sorted(random.Random(args.seed).sample(range(len(eligible)), args.sample_size))
    samples = [eligible[i] for i in selected]
    report = dict(dataset=DATASET, split="validation", selected=len(samples), seed=args.seed,
        sample_ids=[s["id"] for s in samples], conditions=list(CONDITIONS), context_limits=limits,
        condition_num_shots={c: 0 if c.endswith("zero_shot") else args.num_shots for c in CONDITIONS},
        max_new_tokens=args.max_new_tokens, num_shots=args.num_shots, prefix_max_tokens=budget,
        demonstrations=demonstrations, source_files=dict(train=train.manifest(), validation=validation.manifest()),
        validation_examples=len(validation), eligible=len(eligible), invalid=invalid, demo_overlap=overlap,
        overlength=overlength, min_documents=args.min_documents, outer_template_source=ALPACA_SOURCE,
        prefix_sha256={p: hashlib.sha256(text.encode()).hexdigest() for p, text in prefixes.items()},
        protocol_caveat="published Alpaca outer wrapper only; task wording/few-shot extension are local",
        sampling="same seeded validation IDs for all models/conditions; common eligibility before sampling",
        selection_policy="no score-based prompt choice; no test examples; original references never truncated")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(dict(state="prepared", data=str(destination), samples=len(samples),
                         eligible=len(eligible), conditions=CONDITIONS), ensure_ascii=False), flush=True)
    return samples, report


def load_data(args):
    path = Path(args.data)
    report = json.loads(Path(str(path) + ".meta.json").read_text())
    if report.get("dataset") != DATASET or report.get("split") != "validation":
        raise ValueError("The native probe accepts only its prepared validation data")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["output_sha256"]:
        raise ValueError("Probe dataset checksum differs from its preparation report")
    if str(Path(args.model).resolve()) not in report["context_limits"]:
        raise ValueError("Prepare the common validation subset with this model included first")
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (not samples or len(samples) != report["selected"] or len({s["id"] for s in samples}) != len(samples)
            or [s["id"] for s in samples] != report["sample_ids"]
            or report["conditions"] != list(CONDITIONS)
            or any(set(s["prompts"]) != set(CONDITIONS) for s in samples)):
        raise ValueError("Missing or mismatched probe IDs/conditions")
    if args.max_new_tokens is not None and args.max_new_tokens != report["max_new_tokens"]:
        raise ValueError("Output budget must match the frozen validation data")
    for field, value in dict(max_new_tokens=report["max_new_tokens"], num_shots=report["num_shots"],
                            prefix_max_tokens=report["prefix_max_tokens"], seed=report["seed"],
                            sample_size=report["selected"]).items():
        setattr(args, field, value)
    return samples, report


def summarize(records, modes):
    scores, raw_scores, diagnostics = {}, {}, {}
    references = [r["references"] for r in records]
    for mode in modes:
        scores[mode], raw_scores[mode], diagnostics[mode] = {}, {}, {}
        for condition in CONDITIONS:
            predictions = [r["predictions"][mode][condition] for r in records]
            text = [p["text"] for p in predictions]
            scores[mode][condition] = score_turn_filtered(text, references, records)
            raw_scores[mode][condition] = score(text, references, records)
            diagnostics[mode][condition] = dict(samples=len(records),
                contains_document_separator=sum("|||||" in p["text"] for p in predictions),
                boundary_trimmed=sum(p["summary_extraction"]["boundary_trimmed"] for p in predictions),
                stopped_on_eos=sum(p["stopped_on_eos"] for p in predictions),
                reached_output_limit=sum(p["reached_output_limit"] for p in predictions))
    return scores, raw_scores, diagnostics


def run(args):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    samples, report = load_data(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result, status, journal = (output / name for name in ("result.json", "status.json", "predictions.jsonl"))
    if any(p.exists() for p in (result, status, journal)):
        raise FileExistsError(f"Refusing to overwrite native probe output: {output}")
    status.touch(exist_ok=False)
    total, completed = len(samples) * len(CONDITIONS) * 2, 0

    def update(state, **details):
        value = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(), completed=completed,
                     total=total, config=vars(args), **details)
        temporary = status.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(status)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        update("checking")
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        adapter = PeftConfig.from_pretrained(args.lora, local_files_only=True)
        if config.model_type != "llama" or adapter.base_model_name_or_path != f"meta-llama/Llama-2-{args.size}-hf":
            raise ValueError("Model type or adapter base does not match the requested Llama-2 size")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        torch.manual_seed(report["seed"])
        update("loading_model")
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            dtype=getattr(torch, args.dtype), device_map={"": args.device},
            use_safetensors=None if args.weight_format == "auto" else args.weight_format == "safetensors",
            attn_implementation=args.attn_implementation).eval()
        eos = model.generation_config.eos_token_id
        eos = eos if eos is not None else tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = next(iter(sorted(eos_ids)), 0)
        generation = GenerationConfig(max_new_tokens=report["max_new_tokens"], do_sample=False, num_beams=1,
            repetition_penalty=1.0, no_repeat_ngram_size=0, eos_token_id=eos, pad_token_id=pad_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=True)
        records = [dict(id=s["id"], references=s["references"], metadata=s["metadata"], predictions={}) for s in samples]
        with journal.open("x", encoding="utf-8") as handle, torch.inference_mode():
            for mode in ("base", "lora_full"):
                if mode == "lora_full":
                    model = PeftModel.from_pretrained(model, args.lora, local_files_only=True, is_trainable=False).eval()
                for condition in CONDITIONS:
                    for record, sample in zip(records, samples):
                        inputs = tokenizer(sample["prompts"][condition], add_special_tokens=True,
                                           return_tensors="pt", return_token_type_ids=False).to(args.device)
                        length = inputs.input_ids.shape[1]
                        if length + report["max_new_tokens"] > report["context_limits"][str(Path(args.model).resolve())]:
                            raise ValueError("Probe prompt exceeds the prepared context budget")
                        # No injected/reconstructed KV or custom decoding is used in this reference.
                        generated = model.generate(**inputs, generation_config=generation)
                        ids = generated[0, length:].tolist()
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        prediction = dict(text=text, token_ids=ids, input_tokens=length, output_tokens=len(ids),
                            stopped_on_eos=bool(ids and ids[-1] in eos_ids),
                            reached_output_limit=bool(ids and ids[-1] not in eos_ids and len(ids) == report["max_new_tokens"]),
                            summary_extraction=extract_summary(text))
                        record["predictions"].setdefault(mode, {})[condition] = prediction
                        handle.write(json.dumps(dict(id=record["id"], mode=mode, condition=condition,
                                                     prediction=prediction), ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
                        update("running", mode=mode, condition=condition, sample_id=record["id"])
        scores, raw_scores, diagnostics = summarize(records, ("base", "lora_full"))
        value = dict(config=vars(args), dataset_preparation=report, predictions=records, scores=scores,
            unfiltered_scores=raw_scores, generation_diagnostics=diagnostics, generation_config=generation.to_dict(),
            scorer_metadata=dict(primary_metric="rougeL", policy=TURN_POLICY, aggregation="macro mean F1",
                                 scale="fractions in [0, 1]", use_stemmer=True, reference_independent_filter=True),
            experiment=dict(engine="native Transformers generate", modes=["base", "lora_full"], batch_size=1,
                            shared_external_prefix_cache=False, minikv=False, score_based_protocol_selection=False))
        with result.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        update("complete", result=str(result), scores=scores)
        return value
    except BaseException as error:
        update("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Frozen validation JSONL; output path in prepare-only mode")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS.values()))
    parser.add_argument("--dataset-dir", default="/data/hub/multi_news")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--num-shots", type=int, default=1)
    parser.add_argument("--prefix-max-tokens", type=int, default=1024)
    parser.add_argument("--min-documents", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, help="Prepare default 512; run must match frozen data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", choices=list(MODEL_PATHS))
    parser.add_argument("--model")
    parser.add_argument("--lora")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--weight-format", choices=["bin", "safetensors", "auto"])
    parser.add_argument("--attn-implementation", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args(argv)
    if args.prepare_only and args.max_new_tokens is None:
        args.max_new_tokens = 512
    if min(args.sample_size, args.num_shots, args.prefix_max_tokens, args.min_documents) < 1 or (
            args.max_new_tokens is not None and args.max_new_tokens < 1):
        parser.error("Sample/shot/document counts and token budgets must be positive")
    if not args.prepare_only:
        if not args.size or not args.output_dir:
            parser.error("Run mode requires --size and --output-dir")
        args.model = args.model or MODEL_PATHS[args.size]
        args.lora = args.lora or f"/data/hub/multi_news-lora/llama2-{args.size}"
        args.weight_format = args.weight_format or ("safetensors" if args.size == "7b" else "bin")
    return args


if __name__ == "__main__":
    args = parse_args()
    prepare(args) if args.prepare_only else run(args)
