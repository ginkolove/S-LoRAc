"""Frozen LMSYS traces with MAF or rank-stratified adapter popularity."""

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "lmsys-maf-poisson-v1"
CHAMELEON_SCHEMA = "lmsys-chameleon-poisson-v1"
CHAMELEON_RANKS = (16, 32, 64)
HF_PROFILE = "hf-llama7b"
PROFILE_SCHEMAS = {"maf": SCHEMA, "chameleon": CHAMELEON_SCHEMA, HF_PROFILE: "lmsys-hf-llama7b-poisson-v1"}
# Exact configured base, one public LoRA repository per vote, conditional on these ranks.
HF_REPO_COUNTS = {8: 147, 16: 81, 32: 32, 64: 63}


def trace_profile(trace):
    for name, schema in PROFILE_SCHEMAS.items():
        if trace.get("schema") == schema:
            return name
    raise ValueError("unsupported workload trace schema")


def hf_rank_counts(num_adapters):
    """Largest-remainder allocation, with at least one adapter per selected rank."""
    if num_adapters < len(HF_REPO_COUNTS):
        raise ValueError("hf-llama7b needs at least four adapters, one per rank")
    total = sum(HF_REPO_COUNTS.values())
    counts = {r: 1 for r in HF_REPO_COUNTS}
    for _ in range(num_adapters - len(counts)):
        rank = max(counts, key=lambda r: num_adapters * HF_REPO_COUNTS[r] - total * counts[r])
        counts[rank] += 1
    return counts


def hf_adapter_layout(counts):
    return [(rank, replica) for replica in range(max(counts.values()))
            for rank, count in counts.items() if replica < count]


def hf_profile(num_adapters):
    counts = hf_rank_counts(num_adapters)
    layout = hf_adapter_layout(counts)
    total = sum(HF_REPO_COUNTS.values())
    harmonic = {r: math.fsum(1 / j for j in range(1, n + 1)) for r, n in counts.items()}
    return dict(adapter_ranks=[r for r, _ in layout],
                adapter_probabilities=[(HF_REPO_COUNTS[r] / total) / (harmonic[r] * (i + 1))
                                       for r, i in layout],
                within_rank_zipf_alpha=1.0, source_base="huggyllama/llama-7b",
                source_snapshot="2026-11-20",
                source_repository_counts={str(r): n for r, n in HF_REPO_COUNTS.items()})


def chameleon_profile(num_adapters):
    """Round-robin adapter IDs match expand_adapter_dirs; Zipf is local to each rank."""
    groups = len(CHAMELEON_RANKS)
    if num_adapters < groups:
        raise ValueError("chameleon needs at least three adapters, one per rank")
    ranks = [CHAMELEON_RANKS[i % groups] for i in range(num_adapters)]
    probabilities = [0.] * num_adapters
    for offset in range(groups):
        indices = range(offset, num_adapters, groups)
        harmonic = math.fsum(1 / j for j in range(1, len(indices) + 1))
        for j, idx in enumerate(indices, 1):
            probabilities[idx] = 1 / (groups * harmonic * j)
    return dict(adapter_ranks=ranks, adapter_probabilities=probabilities, within_rank_zipf_alpha=1.0)


def load_trace(path):
    raw = Path(path).read_bytes()
    trace = json.loads(raw)
    profile = trace_profile(trace)
    config = trace["config"]
    for key in ("num_adapters", "prefix_len", "max_context_len", "vocab_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("invalid trace config: %s" % key)
    for key in ("request_rate", "duration"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("invalid trace config: %s" % key)
    build_profile = {"chameleon": chameleon_profile, HF_PROFILE: hf_profile}.get(profile)
    if build_profile and trace.get(profile) != build_profile(config["num_adapters"]):
        raise ValueError("invalid %s rank or popularity map" % profile)
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("invalid trace seed")
    if not trace["requests"]:
        raise ValueError("workload trace has no requests")
    previous, ids = -1.0, set()
    for request in trace["requests"]:
        arrival = request["req_time"]
        if not math.isfinite(arrival) or not 0 <= arrival <= config["duration"] or arrival <= previous:
            raise ValueError("trace arrivals must be finite, strictly increasing and within duration")
        previous = arrival
        if request["req_id"] in ids:
            raise ValueError("duplicate request id")
        ids.add(request["req_id"])
        adapter = request["adapter_index"]
        if type(adapter) is not int or not 0 <= adapter < config["num_adapters"]:
            raise ValueError("invalid adapter index")
        tokens = request["input_ids"]
        if not isinstance(tokens, list) or not tokens or any(
                type(token) is not int or not 0 <= token < config["vocab_size"] for token in tokens):
            raise ValueError("invalid query token ids")
        output = request["output_len"]
        if type(output) is not int or output <= 0:
            raise ValueError("invalid output length")
        if config["prefix_len"] + len(tokens) + output > config["max_context_len"]:
            raise ValueError("request exceeds trace context limit")
    return trace, hashlib.sha256(raw).hexdigest()


def validate_run(trace, args):
    config = trace["config"]
    profile = trace_profile(trace)
    if getattr(args, "adapter_profile", None) not in (None, profile):
        raise ValueError("frozen trace disagrees with --adapter-profile; use a fresh output directory")
    if getattr(args, "data_seed", None) is not None and config.get("data_seed") != args.data_seed:
        raise ValueError("workload trace disagrees with --data-seed; use a fresh output directory")
    for key in ("num_adapters", "prefix_len", "max_context_len", "duration"):
        if config[key] != getattr(args, key):
            raise ValueError("workload trace disagrees with --%s" % key.replace("_", "-"))
    if config["tokenizer"] != args.model_dir:
        raise ValueError("workload tokenizer must match --model-dir")
    if args.request_rates != [config["request_rate"]] or args.seeds != [config["seed"]]:
        raise ValueError("a frozen workload requires its exact single request rate and seed")
    if args.cv != 1:
        raise ValueError("LMSYS/MAF workload uses Poisson arrivals (CV=1)")
    if args.measure_saturation:
        raise ValueError("fixed-output saturation-cycle estimator does not support variable-length traces")


def bind_requests(trace, adapter_dirs):
    if len(adapter_dirs) != trace["config"]["num_adapters"]:
        raise ValueError("adapter count does not match workload")
    return [dict(request, adapter_dir=adapter_dirs[request["adapter_index"]])
            for request in trace["requests"]]
