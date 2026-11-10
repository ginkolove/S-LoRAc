"""Prepare one CPU-only, rank-independent workload shared by all serving methods."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.lmsys_maf_trace import SCHEMA, load_trace


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(path, value):
    content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("refusing to overwrite a different prepared artifact: %s" % path)
    else:
        with path.open("xb") as output:
            output.write(content)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest())


def aggregate_maf(path, num_adapters, skewness, minute_start, minute_end):
    if num_adapters <= 0 or skewness <= 0 or not 1 <= minute_start <= minute_end <= 1440:
        raise ValueError("invalid MAF adapter count, skewness or minute window")
    counts = Counter()
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        minutes = [str(minute) for minute in range(minute_start, minute_end + 1)]
        if not set(minutes + ["HashOwner", "HashApp", "HashFunction"]).issubset(reader.fieldnames):
            raise ValueError("MAF file lacks requested minute or function columns")
        for row in tqdm(reader, desc="Aggregate MAF", unit="function"):
            values = [int(row[minute]) for minute in minutes]
            if min(values) < 0:
                raise ValueError("negative MAF invocation count")
            count = sum(values)
            if count:
                counts[(row["HashOwner"], row["HashApp"], row["HashFunction"])] += count
    if not counts:
        raise ValueError("MAF window has no invocations")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    loads = [0] * num_adapters
    assignments = hashlib.sha256()
    for position, (key, count) in enumerate(ordered):
        adapter = (position // skewness) % num_adapters
        loads[adapter] += count
        assignments.update((json.dumps([key, count, adapter]) + "\n").encode())
    total = sum(loads)
    probabilities = [load / total for load in loads]
    popularities = sorted(probabilities, reverse=True)
    return dict(source=str(Path(path).resolve()), source_sha256=file_hash(path),
                minute_start=minute_start, minute_end=minute_end, skewness=skewness,
                mapping="sort by descending invocation count, tie by function key; (position // skewness) % N",
                active_functions=len(ordered), total_invocations=total,
                adapter_loads=loads, adapter_probabilities=probabilities,
                distribution_summary=dict(top1_share=popularities[0],
                                          top5_share=sum(popularities[:5]),
                                          top10_share=sum(popularities[:10]),
                                          min_share=min(probabilities),
                                          nonzero_adapters=sum(load > 0 for load in loads),
                                          effective_adapters=1.0 / sum(p * p for p in probabilities)),
                assignment_sha256=assignments.hexdigest())


def poisson_arrivals(rate, duration, rng):
    if not math.isfinite(rate) or not math.isfinite(duration) or min(rate, duration) <= 0:
        raise ValueError("rate and duration must be positive and finite")
    arrivals, time = [], 0.0
    while True:
        time += float(rng.exponential(1.0 / rate))
        if time > duration:
            return arrivals
        arrivals.append(time)


def first_turns(rows):
    candidates = []
    for row_index, row in enumerate(rows):
        for branch in ("a", "b"):
            conversation = row["conversation_" + branch]
            if len(conversation) < 2:
                continue
            user, assistant = conversation[:2]
            if user["role"] != "user" or assistant["role"] != "assistant":
                continue
            if not user["content"].strip() or not assistant["content"].strip():
                continue
            candidates.append(dict(source_id="%s:%s:%s" % (row_index, row["question_id"], branch),
                                   source_model=row["model_" + branch], query=user["content"],
                                   answer=assistant["content"]))
    return candidates


def select_samples(candidates, count, tokenizer, rng, prefix_len, context_len, order=None):
    selected, rejected = [], 0
    if order is None:
        order = rng.permutation(len(candidates))
    for index in tqdm(order, desc="Select LMSYS first turns", unit="candidate"):
        item = candidates[int(index)]
        # Query is appended after the fixed prefix; do not insert an extra BOS.
        query = tokenizer.encode(item["query"], add_special_tokens=False)
        answer = tokenizer.encode(item["answer"], add_special_tokens=False)
        if not query or not answer or prefix_len + len(query) + len(answer) > context_len:
            rejected += 1
            continue
        selected.append(dict(source_id=item["source_id"], source_model=item["source_model"],
                             input_ids=query, output_len=len(answer)))
        if len(selected) == count:
            return selected, rejected
    raise ValueError("not enough eligible LMSYS first turns for sampling without replacement")


def build_trace(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    streams = np.random.SeedSequence(args.seed).spawn(2)
    arrival_rng, adapter_rng = [np.random.default_rng(stream) for stream in streams]
    sample_rng = np.random.default_rng(args.data_seed)
    arrivals = poisson_arrivals(args.request_rate, args.duration, arrival_rng)
    if not arrivals:
        raise ValueError("Poisson draw produced no requests; increase duration")
    maf = aggregate_maf(args.maf_file, args.num_adapters, args.maf_skewness,
                        args.maf_minute_start, args.maf_minute_end)
    paths = sorted(args.dataset.glob("*.parquet")) if args.dataset.is_dir() else [args.dataset]
    if not paths:
        raise ValueError("no parquet files found")
    rows = pq.read_table(paths, columns=["question_id", "model_a", "model_b",
                                       "conversation_a", "conversation_b"]).to_pylist()
    candidates = first_turns(rows)
    source_files = [dict(path=str(path.resolve()), sha256=file_hash(path)) for path in paths]
    order = sample_rng.permutation(len(candidates))
    dataset_order = dict(
        schema="lmsys-first-turn-order-v1", data_seed=args.data_seed,
        rng="numpy.default_rng(data_seed).permutation(candidate_count)",
        files=source_files, rows=len(rows), first_turn_candidates=len(candidates),
        scope="first user/assistant pair from both branches; no subsequent-turn replay",
        source_ids=[candidates[int(index)]["source_id"] for index in order])
    prepared_dir = args.output.parent / "prepared"
    order_artifact = save_artifact(prepared_dir / ("lmsys_first_turn_order_seed%s.json" % args.data_seed),
                                   dataset_order)
    maf_artifact = save_artifact(prepared_dir / ("maf_%s_minutes%s-%s_skew%s_adapters%s.json" % (
        args.maf_file.stem, args.maf_minute_start, args.maf_minute_end,
        args.maf_skewness, args.num_adapters)), maf)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    samples, rejected = select_samples(candidates, len(arrivals), tokenizer, sample_rng,
                                       args.prefix_len, args.max_context_len, order=order)
    adapters = adapter_rng.choice(args.num_adapters, size=len(arrivals), p=maf["adapter_probabilities"])
    requests = [dict(sample, req_id="lmsys-%s-%s" % (args.seed, i), req_time=arrival,
                     adapter_index=int(adapter))
                for i, (sample, arrival, adapter) in enumerate(zip(samples, arrivals, adapters))]
    query_lens = [len(request["input_ids"]) for request in requests]
    output_lens = [request["output_len"] for request in requests]
    return dict(schema=SCHEMA, config=dict(
        tokenizer=args.tokenizer, vocab_size=len(tokenizer), num_adapters=args.num_adapters,
        prefix_len=args.prefix_len, max_context_len=args.max_context_len,
        request_rate=args.request_rate, duration=args.duration, seed=args.seed, data_seed=args.data_seed,
        arrival_distribution="homogeneous_poisson", adapter_distribution="maf_block_round_robin",
        maf_skewness=args.maf_skewness, maf_minute_start=args.maf_minute_start,
        maf_minute_end=args.maf_minute_end), maf=maf,
        prepared_artifacts=dict(maf_profile=maf_artifact, dataset_order=order_artifact),
        dataset=dict(files=source_files, data_seed=args.data_seed,
                     rows=len(rows), first_turn_candidates=len(candidates),
                     scope="first user/assistant pair from both branches; no subsequent-turn replay",
                     sample_policy="fixed seed-permuted candidate order; reject empty or over-context pairs; no truncation",
                     rejected_candidates=rejected, add_special_tokens=False,
                     output_policy="generate exactly the token count of the recorded answer, ignore EOS",
                     shared_prefix="synthetic fixed prefix, not native LMSYS prefix reuse"),
        summary=dict(total_requests=len(requests), adapter_counts=dict(Counter(map(int, adapters))),
                     query_tokens=dict(min=min(query_lens), mean=float(np.mean(query_lens)), max=max(query_lens)),
                     output_tokens=dict(min=min(output_lens), mean=float(np.mean(output_lens)), max=max(output_lens))),
        requests=requests)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/data/hub/LMSYS-33k"))
    parser.add_argument("--maf-file", type=Path, default=Path(
        "/data/hub/MAF/azurefunctions2019/invocations_per_function_md.anon.d01.csv"))
    parser.add_argument("--maf-skewness", type=int, default=1)
    parser.add_argument("--maf-minute-start", type=int, default=1)
    parser.add_argument("--maf-minute-end", type=int, default=1440,
                        help="Inclusive last MAF minute; default is the full d01 day, not its first hour.")
    parser.add_argument("--tokenizer", default="huggyllama/llama-7b")
    parser.add_argument("--num-adapters", type=int, default=50)
    parser.add_argument("--prefix-len", type=int, default=10000)
    parser.add_argument("--max-context-len", type=int, default=16384)
    parser.add_argument("--request-rate", type=float, default=3)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42,
                        help="Independent fixed LMSYS permutation seed; unaffected by arrival/adapter --seed.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.seed < 0 or args.data_seed < 0 or args.num_adapters <= 0 or args.maf_skewness <= 0
            or not 0 < args.prefix_len < args.max_context_len
            or not 1 <= args.maf_minute_start <= args.maf_minute_end <= 1440):
        parser.error("invalid seed, prefix, adapter count or MAF configuration")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a frozen workload trace")
    trace = build_trace(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(trace, output, indent=2, allow_nan=False)
        output.write("\n")
    _, digest = load_trace(args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), sha256=digest,
                          maf_skewness=args.maf_skewness,
                          maf_minutes=[args.maf_minute_start, args.maf_minute_end],
                          data_seed=args.data_seed, maf_distribution=trace["maf"]["distribution_summary"],
                          prepared_artifacts=trace["prepared_artifacts"], **trace["summary"]), indent=2))
