"""Fixed English prefix + aligned, seeded non-English Alpaca-MT test samples."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.request import urlopen

from tqdm.auto import tqdm

if __package__:
    from .alpaca_mt import language_templates
    from .common import alpaca_example, clean_examples, fingerprint, read_json
else:
    from alpaca_mt import language_templates
    from common import alpaca_example, clean_examples, fingerprint, read_json


REVISION = "66b2b0c9f8e3e77f57446f7f09decd713247a797"
SOURCE = f"https://raw.githubusercontent.com/juletx/alpaca-lora-mt/{REVISION}"
LANGUAGES = dict(pt="Portuguese", es="Spanish", ca="Catalan", eu="Basque",
                 gl="Galician", at="Asturian")
RAW_GIT_SHA = dict(en="aea0f14c1a14c7953ffc754d87cbf32a95aa3e4d",
                   pt="766228ab82977726144ffe876d77891d733c597e",
                   es="2b23c8e65faf235f60c759745251c4083e398e32",
                   ca="9399f5c721045691f07992fa31fe06c5de5b1e13",
                   eu="a274c2805d289d407c87ee9a69295dd7226f0282",
                   gl="42191026ae021b2554a7f83389fe13fd9ac30771",
                   at="8d2bd1872081294b1885c355da17179a324cbfe7")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(path, language):
    data = Path(path).read_bytes()
    git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if git_sha != RAW_GIT_SHA[language]:
        raise ValueError(f"{path}: source differs from pinned {language} file; row alignment is not verified")
    return hashlib.sha256(data).hexdigest()


def download_raw(path, language, local_files_only=False):
    path = Path(path)
    if path.is_file():
        return verify_source(path, language)
    if local_files_only:
        raise FileNotFoundError(f"Missing {path}; omit --local-files-only to download raw data")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.download")
    for attempt in range(3):
        try:
            with urlopen(f"{SOURCE}/data/alpaca_data_cleaned_{language}.json", timeout=90) as response:
                total = response.headers.get("Content-Length")
                with temporary.open("wb") as handle, tqdm(
                        total=int(total) if total else None, desc=f"Download {language}",
                        unit="B", unit_scale=True) as progress:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        progress.update(len(chunk))
            digest = verify_source(temporary, language)
            read_json(temporary)
            temporary.replace(path)
            return digest
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def read_prefix(path, english_rows):
    with Path(path).open(encoding="utf-8") as handle:
        record = next(json.loads(line) for line in handle if line.strip())
    ids = record.get("metadata", {}).get("fewshot_train_ids")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("English prefix record needs distinct metadata.fewshot_train_ids")
    indices = [int(value) for value in ids]
    if any(index < 0 or index >= len(english_rows) for index in indices):
        raise ValueError("English demonstration index is out of range")
    expected, groups = "", set()
    for index in indices:
        item = alpaca_example(english_rows[index], index)
        if item is None:
            raise ValueError(f"Invalid English demonstration {index}")
        expected += item["query"] + item["references"][0] + "\n\n"
        groups.add(item["group_key"])
    if record.get("prefix") != expected:
        raise ValueError("Prefix does not match its English source demonstrations")
    excluded = {index for index, row in enumerate(english_rows)
                if (item := alpaca_example(row, index)) is not None and item["group_key"] in groups}
    return expected, indices, excluded


def tokenizer_groups(models):
    groups = {}
    for name, tokenizer, context_limit in models:
        backend = tokenizer.backend_tokenizer.to_str()
        policy = dict(backend=backend, specials=tokenizer.special_tokens_map,
                      add_bos=getattr(tokenizer, "add_bos_token", None),
                      add_eos=getattr(tokenizer, "add_eos_token", None))
        signature = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
        group = groups.setdefault(signature, dict(tokenizer=tokenizer, models=[], limit=context_limit))
        group["models"].append(name)
        group["limit"] = min(group["limit"], context_limit)
    return list(groups.values())


def build_samples(english_rows, translated_rows, prefix_info, templates, groups, *,
                  sample_size=200, seed=42, max_new_tokens=256, language_instruction=True):
    if sample_size < 1 or max_new_tokens < 1 or not translated_rows or not groups:
        raise ValueError("Need positive sample/output sizes, languages and model tokenizers")
    prefix, demo_ids, excluded = prefix_info
    per_language, reports = {}, {}
    for language, rows in translated_rows.items():
        if len(rows) != len(english_rows):
            raise ValueError(f"{language}: row count differs from English; cannot align translations")
        # Use source positions, not IDs introduced by cleaning or deduplication.
        def formatter(row, index):
            row = dict(row, id=str(index))
            if language_instruction and isinstance(row.get("instruction"), str) and row["instruction"].strip():
                row["instruction"] += f"\n\nRespond in {LANGUAGES[language]}."
            return alpaca_example(row, index, templates[language])

        examples, stats = clean_examples(rows, formatter)
        demo_queries = {fingerprint(item["query"]) for index in excluded
                        if (item := formatter(rows[index], index)) is not None}
        candidates = [item for item in examples if int(item["id"]) not in excluded
                      and item["query_key"] not in demo_queries]
        stats["fewshot_or_equivalent_excluded"] = len(examples) - len(candidates)
        lengths = [dict() for _ in candidates]
        too_long = set()
        for group in groups:
            tokenizer = group["tokenizer"]
            for start in tqdm(range(0, len(candidates), 256), desc=f"Check {language} lengths", unit="batch"):
                batch = candidates[start:start + 256]
                encoded = tokenizer([prefix + item["query"] for item in batch],
                                    add_special_tokens=True, truncation=False,
                                    return_attention_mask=False)["input_ids"]
                for offset, ids in enumerate(encoded, start):
                    lengths[offset].update({name: len(ids) for name in group["models"]})
                    if len(ids) + max_new_tokens > group["limit"]:
                        too_long.add(offset)
        eligible = {}
        for offset, (item, token_lengths) in enumerate(zip(candidates, lengths)):
            if offset in too_long:
                continue
            index = int(item["id"])
            eligible[index] = dict(
                id=f"alpaca_mt:{language}:{index}", prefix=prefix, query=item["query"],
                references=item["references"],
                metadata=dict(dataset="alpaca_mt", language=language, prefix_language="en",
                              source_id=str(index), source_index=index, fewshot_train_ids=demo_ids,
                              seed=seed, reserved_max_new_tokens=max_new_tokens,
                              input_tokens_by_model=token_lengths,
                              language_instruction=language_instruction,
                              test_source="aligned_translations_after_english_fewshot_exclusion"))
        stats.update(overlength=len(too_long), eligible=len(eligible))
        per_language[language], reports[language] = eligible, stats
    common = sorted(set.intersection(*(set(values) for values in per_language.values())))
    if len(common) < sample_size:
        raise ValueError(f"Only {len(common)} aligned eligible tasks, cannot sample {sample_size}")
    selected = sorted(random.Random(seed).sample(common, sample_size))
    samples = [rows[index] for rows in per_language.values() for index in selected]
    report = dict(dataset="alpaca_mt", prefix_language="en", fewshot_train_ids=demo_ids,
                  prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                  seed=seed, samples_per_language=sample_size, languages=list(per_language),
                  total_samples=len(samples), common_eligible_tasks=len(common),
                  selected_source_indices=selected,
                  sampling="uniform without replacement from common eligible tasks; aligned across languages",
                  max_new_tokens=max_new_tokens, add_special_tokens=True,
                  language_instruction=language_instruction, per_language=reports,
                  adapter_training_disjointness="not_verified",
                  task_independence="translations of the same tasks, not independent task datasets")
    return samples, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-file", default="accexp/results/alpaca_mt_en_raw.json")
    parser.add_argument("--prefix-data", default="accexp/results/alpaca_mt_en_3shot_all_seed42.jsonl")
    parser.add_argument("--raw-dir", default="accexp/results/alpaca_mt_raw")
    parser.add_argument("--template-file", default="/data/hub/alpaca_mt/alpaca_mt.py")
    parser.add_argument("--models", nargs="+", default=[f"huggyllama/llama-{size}" for size in ("7b", "13b", "30b")])
    parser.add_argument("--languages", nargs="+", choices=tuple(LANGUAGES), default=list(LANGUAGES))
    parser.add_argument("--samples-per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-language-instruction", action="store_true")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Also prohibit raw JSON downloads; models are always loaded locally")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if len(set(args.languages)) != len(args.languages) or len(set(args.models)) != len(args.models):
        parser.error("Languages and models must not be repeated")
    if args.samples_per_language < 1 or args.max_new_tokens < 1:
        parser.error("Sample and output sizes must be positive")
    destination = Path(args.output)
    meta_path = Path(str(destination) + ".meta.json")
    if destination.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    english_hash = download_raw(args.english_file, "en", args.local_files_only)
    english = read_json(args.english_file)
    prefix = read_prefix(args.prefix_data, english)
    rows, hashes = {}, {"en": english_hash}
    for language in args.languages:
        path = Path(args.raw_dir) / f"alpaca_data_cleaned_{language}.json"
        hashes[language] = download_raw(path, language, args.local_files_only)
        rows[language] = read_json(path)
    from transformers import AutoConfig, AutoTokenizer

    models, configs = [], {}
    for name in args.models:
        config = AutoConfig.from_pretrained(name, local_files_only=True)
        limits = [int(value) for value in (getattr(config, "max_position_embeddings", None),
                                          getattr(config, "max_sequence_length", None)) if value]
        if not limits:
            raise ValueError(f"{name}: model configuration has no context length")
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        models.append((name, tokenizer, min(limits)))
        configs[name] = dict(context_limit=min(limits),
                             prefix_tokens=len(tokenizer.encode(prefix[0], add_special_tokens=True)))
    samples, report = build_samples(english, rows, prefix,
                                   {language: language_templates(language, True, args.template_file)
                                    for language in args.languages}, tokenizer_groups(models),
                                   sample_size=args.samples_per_language, seed=args.seed,
                                   max_new_tokens=args.max_new_tokens,
                                   language_instruction=not args.no_language_instruction)
    report.update(models=configs, source_revision=REVISION, raw_sha256=hashes,
                  prefix_data=str(Path(args.prefix_data).resolve()),
                  template_sha256=file_hash(args.template_file),
                  alignment_source=f"{SOURCE}/translate/format_translation_data.ipynb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    report["output_sha256"] = file_hash(destination)
    with meta_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(dict(output=str(destination), total_samples=len(samples),
                          samples_per_language=args.samples_per_language, models=configs,
                          prefix_sha256=report["prefix_sha256"]), ensure_ascii=False, indent=2))
    return report


def prepare_for_tp(args, destination):
    """Use the established MT protocol for the selected TP model, without downloads."""
    return main([
        "--english-file", str(args.english_file), "--prefix-data", str(args.prefix_data),
        "--raw-dir", str(args.raw_dir), "--template-file", str(args.template_file),
        "--models", str(args.model), "--languages", *args.languages,
        "--samples-per-language", str(args.samples_per_language), "--seed", str(args.seed),
        "--max-new-tokens", str(args.max_new_tokens), "--local-files-only",
        "--output", str(destination)])


if __name__ == "__main__":
    main()
