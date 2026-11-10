"""Rank-independent request traces for LMSYS content and MAF adapter popularity."""

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "lmsys-maf-poisson-v1"


def load_trace(path):
    raw = Path(path).read_bytes()
    trace = json.loads(raw)
    if trace.get("schema") != SCHEMA:
        raise ValueError("unsupported workload trace schema")
    config = trace["config"]
    for key in ("num_adapters", "prefix_len", "max_context_len", "vocab_size"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("invalid trace config: %s" % key)
    for key in ("request_rate", "duration"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("invalid trace config: %s" % key)
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
