import argparse
import asyncio
import json
import time
from dataclasses import asdict

from slora_baseline_throughput_exp import (
    expand_adapter_dirs,
    make_requests,
    run_benchmark,
    summarize_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Prefix-S-LoRA shared-prefix-ratio throughput experiment."
    )
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--adapter-base", action="append", default=None)
    parser.add_argument("--num-adapters", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--warmup", type=float, default=40.0)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--steady-end", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-len", type=int, default=1920)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--shared-prefix-ratio", type=float, required=True)
    parser.add_argument("--shared-prefix-len", type=int, default=None)
    parser.add_argument("--max-context-len", type=int, default=2048)
    parser.add_argument("--kv-budget-multiple", type=int, default=18)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--prompt-token-id", type=int, default=100)
    parser.add_argument("--output", type=str, default="prefix_slora_shared_ratio_results.jsonl")
    parser.add_argument("--dump-per-request", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.shared_prefix_ratio < 1:
        raise ValueError("--shared-prefix-ratio must be in (0, 1)")

    prefix_len = (
        args.shared_prefix_len
        if args.shared_prefix_len is not None
        else int(args.input_len * args.shared_prefix_ratio)
    )
    query_len = args.input_len - prefix_len
    logical_len = args.input_len + args.output_len
    if prefix_len <= 0 or query_len <= 0:
        raise ValueError(
            f"invalid prefix/query split: prefix_len={prefix_len}, query_len={query_len}"
        )
    if logical_len > args.max_context_len:
        raise ValueError(
            f"logical_len={logical_len} exceeds max_context_len={args.max_context_len}"
        )

    total_kv_budget = args.kv_budget_multiple * args.max_context_len
    adapter_bases = args.adapter_base or ["dummy-lora-7b-rank-16"]
    adapter_dirs = expand_adapter_dirs(adapter_bases, args.num_adapters)
    input_ids = [args.prompt_token_id] * query_len
    requests = make_requests(
        adapter_dirs=adapter_dirs,
        input_len=query_len,
        output_len=args.output_len,
        request_rate=args.request_rate,
        duration=args.duration,
        alpha=args.alpha,
        seed=args.seed,
        input_ids=input_ids,
    )

    config = {
        "backend": "prefix_slora_fcfs_lru",
        "semantics": (
            "all full prefix KV stored on CPU; GPU prefix KV allocated in unified "
            "paging on demand; inactive GPU prefixes evicted by LRU under pressure"
        ),
        "input_len": args.input_len,
        "shared_prefix_ratio": args.shared_prefix_ratio,
        "shared_prefix_len": prefix_len,
        "query_len": query_len,
        "output_len": args.output_len,
        "logical_len": logical_len,
        "max_context_len": args.max_context_len,
        "num_adapters": args.num_adapters,
        "rank": args.rank,
        "kv_budget_multiple": args.kv_budget_multiple,
        "max_total_token_num": total_kv_budget,
        "alpha": args.alpha,
        "arrival": "poisson",
        "request_rate": args.request_rate,
        "duration": args.duration,
        "warmup": args.warmup,
        "cooldown": args.cooldown,
        "steady_end_override": args.steady_end,
        "seed": args.seed,
        "adapter_dirs": adapter_dirs,
        "prompt_mode": "query_only_token_ids",
        "prompt_token_id": args.prompt_token_id,
    }

    print(json.dumps({"config": config, "num_requests": len(requests)}, indent=2))
    if args.dry_run:
        return

    start = time.time()
    results = asyncio.run(run_benchmark(args.server, requests, debug=args.debug))
    benchmark_time = time.time() - start
    summary = summarize_results(
        requests,
        results,
        benchmark_time=benchmark_time,
        benchmark_start_time=start,
        duration=args.duration,
        warmup=args.warmup,
        cooldown=args.cooldown,
        steady_end_override=args.steady_end,
    )

    measured = summary["measured_requests"]
    measure_time = summary["measure_time"]
    steady_completed = summary["steady_completed_requests"]
    steady_time = summary["steady_time"]
    summary["shared_prefix_len"] = prefix_len
    summary["query_len"] = query_len
    summary["logical_input_len"] = args.input_len
    summary["sent_query_tokens_per_s_measured"] = measured * query_len / measure_time
    summary["logical_input_tokens_per_s_measured"] = measured * args.input_len / measure_time
    summary["logical_tokens_per_s_measured"] = (
        measured * (args.input_len + args.output_len) / measure_time
    )
    summary["sent_query_tokens_per_s_steady"] = steady_completed * query_len / steady_time
    summary["logical_input_tokens_per_s_steady"] = steady_completed * args.input_len / steady_time
    summary["logical_tokens_per_s_steady"] = (
        steady_completed * (args.input_len + args.output_len) / steady_time
    )

    record = {"config": config, "result": summary}
    print(json.dumps(record, indent=2))
    with open(args.output, "a") as f:
        f.write(json.dumps(record) + "\n")

    if args.dump_per_request is not None:
        with open(args.dump_per_request, "w") as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")


if __name__ == "__main__":
    main()
