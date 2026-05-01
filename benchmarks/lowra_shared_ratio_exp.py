import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import aiohttp
import numpy as np


@dataclass
class LowRARequest:
    req_id: str
    adapter_dir: str
    req_time: float
    prefix_len: int
    query_len: int
    output_len: int
    prompt_token_id: int


@dataclass
class RequestResult:
    req_id: str
    adapter_dir: str
    req_time: float
    prefix_len: int
    query_len: int
    output_len: int
    start_time: float
    first_token_time: Optional[float]
    end_time: float
    latency: Optional[float]
    ttft: Optional[float]
    success: bool
    status: Optional[int]
    error: Optional[str]


def zipf_probs(num_adapters: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, num_adapters + 1, dtype=np.float64)
    probs = np.power(ranks, -alpha)
    probs /= probs.sum()
    return probs


def expand_adapter_dirs(adapter_bases: List[str], num_adapters: int) -> List[str]:
    if not adapter_bases:
        raise ValueError("adapter_bases must be non-empty")
    adapter_dirs = []
    num_iter = num_adapters // len(adapter_bases) + 1
    for i in range(num_iter):
        for adapter_base in adapter_bases:
            adapter_dirs.append(f"{adapter_base}-{i}")
            if len(adapter_dirs) == num_adapters:
                return adapter_dirs
    return adapter_dirs[:num_adapters]


def make_requests(
    adapter_dirs: List[str],
    prefix_len: int,
    query_len: int,
    output_len: int,
    request_rate: float,
    duration: float,
    alpha: float,
    seed: int,
    prompt_token_id: int,
) -> List[LowRARequest]:
    rng = np.random.default_rng(seed)
    adapter_probs = zipf_probs(len(adapter_dirs), alpha)
    requests = []
    req_time = 0.0
    while True:
        req_time += float(rng.exponential(1.0 / request_rate))
        if req_time > duration:
            break
        adapter_id = int(rng.choice(len(adapter_dirs), p=adapter_probs))
        req_id = len(requests)
        requests.append(
            LowRARequest(
                req_id=f"lowra-{seed}-{req_id}",
                adapter_dir=adapter_dirs[adapter_id],
                req_time=req_time,
                prefix_len=prefix_len,
                query_len=query_len,
                output_len=output_len,
                prompt_token_id=prompt_token_id,
            )
        )
    return requests


async def send_request(
    session: aiohttp.ClientSession,
    server: str,
    request: LowRARequest,
    debug: bool = False,
) -> RequestResult:
    url = server.rstrip("/") + "/generate_stream"
    headers = {"User-Agent": "LowRA Benchmark Client"}
    start_time = time.time()
    first_token_time = None
    status = None
    error = None

    payload = {
        "req_id": request.req_id,
        "lora_dir": request.adapter_dir,
        "inputs": [request.prompt_token_id] * request.query_len,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": request.output_len,
            "return_details": False,
        },
        "lowra_lengths": {
            "shared_prefix_len": request.prefix_len,
            "query_len": request.query_len,
            "decode_len": request.output_len,
        },
    }

    try:
        async with session.post(url, headers=headers, json=payload) as response:
            status = response.status
            if status != 200:
                error = await response.text()
            else:
                async for chunk, _ in response.content.iter_chunks():
                    if chunk and first_token_time is None:
                        first_token_time = time.time()
    except Exception as exc:
        error = repr(exc)

    end_time = time.time()
    success = status == 200 and error is None and first_token_time is not None
    latency = end_time - start_time if success else None
    ttft = first_token_time - start_time if first_token_time is not None else None

    if debug:
        print(
            f"{request.req_id} adapter={request.adapter_dir} "
            f"P={request.prefix_len} Q={request.query_len} O={request.output_len} "
            f"status={status} latency={latency} ttft={ttft} error={error}"
        )

    return RequestResult(
        req_id=request.req_id,
        adapter_dir=request.adapter_dir,
        req_time=request.req_time,
        prefix_len=request.prefix_len,
        query_len=request.query_len,
        output_len=request.output_len,
        start_time=start_time,
        first_token_time=first_token_time,
        end_time=end_time,
        latency=latency,
        ttft=ttft,
        success=success,
        status=status,
        error=error,
    )


async def run_benchmark(
    server: str,
    requests: List[LowRARequest],
    debug: bool = False,
) -> List[RequestResult]:
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    connector = aiohttp.TCPConnector(limit=0)
    results = []
    tasks = []
    start = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        for request in requests:
            sleep_time = start + request.req_time - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            tasks.append(asyncio.create_task(send_request(session, server, request, debug=debug)))
        for task in asyncio.as_completed(tasks):
            results.append(await task)
    return results


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize_results(
    requests: List[LowRARequest],
    results: List[RequestResult],
    benchmark_time: float,
    warmup: float,
    cooldown: float,
) -> dict:
    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]
    measure_end = max(0.0, requests[-1].req_time - cooldown) if requests else 0.0
    measured = [
        result for result in successful
        if result.req_time >= warmup and result.req_time <= measure_end
    ]
    measure_time = max(1e-9, measure_end - warmup)

    latencies = [result.latency for result in measured if result.latency is not None]
    ttfts = [result.ttft for result in measured if result.ttft is not None]
    output_len = requests[0].output_len if requests else 0
    query_len = requests[0].query_len if requests else 0
    prefix_len = requests[0].prefix_len if requests else 0

    adapter_counts = {}
    for request in requests:
        adapter_counts[request.adapter_dir] = adapter_counts.get(request.adapter_dir, 0) + 1

    return {
        "total_requests": len(requests),
        "completed_requests": len(successful),
        "failed_requests": len(failed),
        "measured_requests": len(measured),
        "benchmark_time": benchmark_time,
        "measure_start": warmup,
        "measure_end": measure_end,
        "measure_time": measure_time,
        "completed_req_per_s_total": len(successful) / benchmark_time if benchmark_time > 0 else 0,
        "completed_req_per_s_measured": len(measured) / measure_time,
        "output_tokens_per_s_measured": len(measured) * output_len / measure_time,
        "query_tokens_per_s_measured": len(measured) * query_len / measure_time,
        "logical_tokens_per_s_measured": len(measured) * (prefix_len + query_len + output_len) / measure_time,
        "latency_mean": float(np.mean(latencies)) if latencies else None,
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
        "latency_p99": percentile(latencies, 99),
        "ttft_mean": float(np.mean(ttfts)) if ttfts else None,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "ttft_p99": percentile(ttfts, 99),
        "adapter_counts": adapter_counts,
        "errors": [
            {
                "req_id": result.req_id,
                "status": result.status,
                "error": result.error,
            }
            for result in failed[:20]
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LowRA shared-prefix-ratio throughput experiment.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--adapter-base", action="append", default=None,
                        help="Base adapter name/path. Repeated to build the adapter pool.")
    parser.add_argument("--num-adapters", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--kv-embed-dim", type=int, default=4096)
    parser.add_argument("--kv-budget-multiple", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--request-rate", type=float, required=True,
                        help="Poisson arrival rate in requests/s.")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--cooldown", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-len", type=int, default=3840)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-context-len", type=int, default=4096)
    parser.add_argument("--shared-prefix-ratio", type=float, required=True,
                        choices=[0.25, 0.5, 0.75, 0.875])
    parser.add_argument("--prompt-token-id", type=int, default=100)
    parser.add_argument("--output", type=str, default="lowra_shared_ratio_results.jsonl")
    parser.add_argument("--dump-per-request", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prefix_len = int(args.input_len * args.shared_prefix_ratio)
    query_len = args.input_len - prefix_len
    logical_len = prefix_len + query_len + args.output_len
    if logical_len > args.max_context_len:
        raise ValueError(
            f"logical_len={logical_len} exceeds max_context_len={args.max_context_len}"
        )
    total_kv_budget = args.kv_budget_multiple * args.max_context_len
    total_rank = args.num_adapters * args.rank
    static_token_equiv = prefix_len + math.ceil(prefix_len * total_rank / args.kv_embed_dim)
    dynamic_token_budget = total_kv_budget - static_token_equiv

    adapter_bases = args.adapter_base or ["dummy-lora-7b-rank-16"]
    adapter_dirs = expand_adapter_dirs(adapter_bases, args.num_adapters)
    requests = make_requests(
        adapter_dirs=adapter_dirs,
        prefix_len=prefix_len,
        query_len=query_len,
        output_len=args.output_len,
        request_rate=args.request_rate,
        duration=args.duration,
        alpha=args.alpha,
        seed=args.seed,
        prompt_token_id=args.prompt_token_id,
    )

    config = {
        "shared_prefix_ratio": args.shared_prefix_ratio,
        "prefix_len": prefix_len,
        "query_len": query_len,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "logical_len": logical_len,
        "max_context_len": args.max_context_len,
        "num_adapters": args.num_adapters,
        "rank": args.rank,
        "kv_embed_dim": args.kv_embed_dim,
        "kv_budget_multiple": args.kv_budget_multiple,
        "max_total_token_num": total_kv_budget,
        "static_token_equiv": static_token_equiv,
        "dynamic_token_budget": dynamic_token_budget,
        "alpha": args.alpha,
        "arrival": "poisson",
        "request_rate": args.request_rate,
        "duration": args.duration,
        "warmup": args.warmup,
        "cooldown": args.cooldown,
        "seed": args.seed,
        "adapter_dirs": adapter_dirs,
        "prompt_mode": "token_ids",
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
        warmup=args.warmup,
        cooldown=args.cooldown,
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
