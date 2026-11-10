"""Offline, reference-independent extraction of the first Alpaca answer."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

if __package__:
    from .multilingual_score import score
else:
    from multilingual_score import score


MODES = ("base", "lora_full", "lora_minikv")
METRICS = ("exact_match", "generic_token_f1")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_boundaries(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PROMPTS" for target in node.targets):
            templates = ast.literal_eval(node.value)
            break
    else:
        raise ValueError("Template loader must define a literal PROMPTS dictionary")
    headers, preambles = set(), set()
    for variants in templates.values():
        for template in variants.values():
            before, separator, _ = template.partition("{instruction}")
            if not separator:
                raise ValueError("Template has no instruction placeholder")
            lines = before.rstrip().splitlines()
            header = lines[-1].strip()
            if not header.startswith("### ") or not header.endswith(":"):
                raise ValueError("Unexpected instruction header: " + header)
            headers.add(header)
            preamble = " ".join("\n".join(lines[:-1]).split())
            if preamble:
                preambles.add(preamble)
    return dict(headers=sorted(headers), preambles=sorted(preambles))


def extract_answer(text, boundaries):
    """Only structure determines the cut; references are deliberately absent."""
    lines = text.splitlines(keepends=True)
    offsets, outside = [], []
    offset, fence = 0, None
    for line in lines:
        offsets.append(offset)
        offset += len(line)
        marker = FENCE.match(line.rstrip("\r\n"))
        if fence is not None:
            outside.append(False)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
        elif marker:
            fence = (marker[1][0], len(marker[1]))
            outside.append(False)
        else:
            outside.append(not line.startswith(("    ", "\t")))

    for index, line in enumerate(lines):
        if not outside[index] or line.strip() not in boundaries["headers"]:
            continue
        start, reason = index, "instruction_header"
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        block_start = previous
        while block_start > 0 and lines[block_start - 1].strip():
            block_start -= 1
        if previous >= 0 and all(outside[block_start:previous + 1]):
            preamble = " ".join("".join(lines[block_start:previous + 1]).split())
            if preamble in boundaries["preambles"]:
                start, reason = block_start, "template_restart"
        cut = offsets[start]
        answer = text[:cut].rstrip()
        return dict(text=answer, truncated=True, reason=reason,
                    cut_offset=cut, header_offset=offsets[index], marker=line.strip(),
                    removed_characters=len(text) - len(answer), empty_answer=not answer.strip())
    return dict(text=text, truncated=False, reason=None, cut_offset=None,
                header_offset=None, marker=None, removed_characters=0, empty_answer=not text.strip())


def metric_delta(left, right):
    return {metric: left[metric] - right[metric] for metric in METRICS}


def process_result(source, samples, boundaries, output_dir, audit_per_group, seed):
    result = json.loads(source.read_text(encoding="utf-8"))
    records = result["predictions"]
    if len(records) != len(samples):
        raise ValueError("Incomplete result: " + str(source))
    extracted = []
    for record, sample in zip(records, samples):
        if (record["id"] != sample["id"] or record["references"] != sample["references"] or
                record["metadata"]["language"] != sample["metadata"]["language"]):
            raise ValueError("Dataset/result mismatch: " + record["id"])
        predictions = {}
        for mode in MODES:
            raw = record["predictions"][mode]
            predictions[mode] = dict(raw_text=raw["text"], **extract_answer(raw["text"], boundaries),
                                     original_output_tokens=raw["output_tokens"],
                                     original_stopped_on_eos=raw["stopped_on_eos"])
        extracted.append(dict(id=record["id"], language=sample["metadata"]["language"],
                              predictions=predictions))

    raw_scores, filtered_scores, counts = {}, {}, {}
    references = [sample["references"] for sample in samples]
    for mode in MODES:
        values = [record["predictions"][mode] for record in extracted]
        raw_scores[mode] = score([value["raw_text"] for value in values], references, samples)
        filtered_scores[mode] = score([value["text"] for value in values], references, samples)
        for metric in METRICS:
            if not math.isclose(raw_scores[mode][metric], result["scores"][mode][metric], abs_tol=1e-12):
                raise ValueError("Original scores cannot be reproduced: " + str(source))
        counts[mode] = dict(samples=len(values), truncated=sum(value["truncated"] for value in values),
                            empty_answers=sum(value["empty_answer"] for value in values),
                            reasons=dict(Counter(value["reason"] for value in values if value["truncated"])),
                            truncated_by_language=dict(Counter(
                                record["language"] for record in extracted if record["predictions"][mode]["truncated"])))

    destination = output_dir / (source.stem + ".turn_filtered.jsonl")
    with destination.open("x", encoding="utf-8") as handle:
        for record in extracted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Audit selection uses language and boundary detection, never score changes.
    groups = defaultdict(list)
    for index, record in enumerate(extracted):
        for mode in MODES:
            if record["predictions"][mode]["truncated"]:
                groups[(record["language"], mode)].append(index)
    rng, audit = random.Random(seed), []
    for (language, mode), indices in sorted(groups.items()):
        for index in sorted(rng.sample(indices, min(audit_per_group, len(indices)))):
            audit.append(dict(model=result["config"]["model"], mode=mode, language=language,
                              id=samples[index]["id"], query=samples[index]["query"],
                              references=samples[index]["references"],
                              prediction=extracted[index]["predictions"][mode]))
    with (output_dir / (source.stem + ".audit.jsonl")).open("x", encoding="utf-8") as handle:
        for record in audit:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dict(model=result["config"]["model"], samples=len(samples), raw_scores=raw_scores,
                filtered_scores=filtered_scores, counts=counts,
                filtered_minus_raw={mode: metric_delta(filtered_scores[mode], raw_scores[mode]) for mode in MODES},
                raw_minikv_minus_lora=metric_delta(raw_scores["lora_minikv"], raw_scores["lora_full"]),
                filtered_minikv_minus_lora=metric_delta(filtered_scores["lora_minikv"], filtered_scores["lora_full"]),
                extracted_predictions=str(destination.resolve()), audit_samples=len(audit))


def run(args):
    sources = [path.resolve() for path in args.results]
    if len({path.stem for path in sources}) != len(sources):
        raise ValueError("Result filenames must be distinct")
    samples = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Dataset must contain nonempty, unique sample IDs")
    boundaries = load_boundaries(args.template_loader)
    tracked = sources + [args.data.resolve(), args.template_loader.resolve(),
                         Path(__file__).resolve(), Path(score.__code__.co_filename).resolve()]
    hashes = {str(path): digest(path) for path in tracked}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(policy=dict(name="alpaca_turn_boundary_v1", reference_independent=True,
                              boundaries=boundaries, protect="fenced code, indented code, blockquotes",
                              audit_status="requires manual review; exploratory post-hoc scores",
                              limitation="Reserved headers may also be legitimate answer content; unmarked continuation remains."),
                  dataset=str(args.data.resolve()), samples=len(samples),
                  languages=dict(Counter(sample["metadata"]["language"] for sample in samples)),
                  scores_scale="fractions in [0, 1]; differences are fractions, not percentage points",
                  seed=args.seed, audit_per_language_mode=args.audit_per_group, source_sha256=hashes, models={})
    for source in sources:
        summary = process_result(source, samples, boundaries, args.output_dir, args.audit_per_group, args.seed)
        report["models"][source.stem] = summary
        print(json.dumps(dict(model=summary["model"], filtered_scores={
            mode: {metric: summary["filtered_scores"][mode][metric] for metric in METRICS} for mode in MODES},
            truncated={mode: summary["counts"][mode]["truncated"] for mode in MODES}), ensure_ascii=False), flush=True)
    if any(digest(path) != checksum for path, checksum in hashes.items()):
        raise RuntimeError("An input changed during rescoring")
    report["inputs_unchanged"] = True
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--template-loader", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.audit_per_group < 1:
        parser.error("--audit-per-group must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
