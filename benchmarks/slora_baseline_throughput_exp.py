import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import aiohttp
import numpy as np


@dataclass
class SLoraRequest:
    req_id: str
    adapter_dir: str
    req_time: float
    input_len: int
    output_len: int
    input_ids: List[int]


@dataclass
class RequestResult:
    req_id: str
    adapter_dir: str
    req_time: float
    input_len: int
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
    input_len: int,
    output_len: int,
    request_rate: float,
    duration: float,
    alpha: float,
    seed: int,
    input_ids: List[int],
) -> List[SLoraRequest]:
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
            SLoraRequest(
                req_id=f"slora-baseline-{seed}-{req_id}",
                adapter_dir=adapter_dirs[adapter_id],
                req_time=req_time,
                input_len=input_len,
                output_len=output_len,
                input_ids=input_ids,
            )
        )
    return requests


async def send_request(
    session: aiohttp.ClientSession,
    server: str,
    request: SLoraRequest,
    debug: bool = False,
) -> RequestResult:
    url = server.rstrip("/") + "/generate_stream"
    headers = {"User-Agent": "S-LoRA Baseline Benchmark Client"}
    start_time = time.time()
    first_token_time = None
    status = None
    error = None

    payload = {
        "req_id": request.req_id,
        "lora_dir": request.adapter_dir,
        "inputs": request.input_ids,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": request.output_len,
            "return_details": False,
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
            f"I={request.input_len} O={request.output_len} "
            f"status={status} latency={latency} ttft={ttft} error={error}"
        )

    return RequestResult(
        req_id=request.req_id,
        adapter_dir=request.adapter_dir,
        req_time=request.req_time,
        input_len=request.input_len,
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
    requests: List[SLoraRequest],
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
    requests: List[SLoraRequest],
    results: List[RequestResult],
    benchmark_time: float,
    benchmark_start_time: float,
    duration: float,
    warmup: float,
    cooldown: float,
    steady_end_override: Optional[float],
) -> dict:
    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]
    measure_end = max(0.0, requests[-1].req_time - cooldown) if requests else 0.0
    measured = [
        result for result in successful
        if result.req_time >= warmup and result.req_time <= measure_end
    ]
    measure_time = max(1e-9, measure_end - warmup)
    drain_measured = [
        result for result in successful
        if result.req_time >= warmup
    ]
    drain_measure_end = max(
        (result.end_time - benchmark_start_time for result in successful),
        default=warmup,
    )
    drain_measure_time = max(1e-9, drain_measure_end - warmup)
    steady_end = steady_end_override if steady_end_override is not None else drain_measure_end
    steady_end = min(steady_end, benchmark_time, drain_measure_end)
    steady_end = max(warmup, steady_end)
    steady_time = max(1e-9, steady_end - warmup)
    steady_completed = [
        result for result in successful
        if warmup <= result.end_time - benchmark_start_time <= steady_end
    ]

    latencies = [result.latency for result in successful if result.latency is not None]
    ttfts = [result.ttft for result in successful if result.ttft is not None]
    input_len = requests[0].input_len if requests else 0
    output_len = requests[0].output_len if requests else 0

    adapter_counts = {}
    for request in requests:
        adapter_counts[request.adapter_dir] = adapter_counts.get(request.adapter_dir, 0) + 1
    completed_req_per_s_steady = len(steady_completed) / steady_time

    return {
        "total_requests": len(requests),
        "completed_requests": len(successful),
        "failed_requests": len(failed),
        "measured_requests": len(measured),
        "drain_measured_requests": len(drain_measured),
        "steady_completed_requests": len(steady_completed),
        "benchmark_time": benchmark_time,
        "measure_start": warmup,
        "measure_end": measure_end,
        "measure_time": measure_time,
        "drain_measure_end": drain_measure_end,
        "drain_measure_time": drain_measure_time,
        "steady_start": warmup,
        "steady_end": steady_end,
        "steady_time": steady_time,
        "completed_req_per_s_total": len(successful) / benchmark_time if benchmark_time > 0 else 0,
        "completed_req_per_s_measured": len(measured) / measure_time,
        "completed_req_per_s_measured_with_drain": len(drain_measured) / drain_measure_time,
        "completed_req_per_s_steady": completed_req_per_s_steady,
        "completed_req_per_s_steady_precise": f"{completed_req_per_s_steady:.12f}",
        "output_tokens_per_s_measured": len(measured) * output_len / measure_time,
        "input_tokens_per_s_measured": len(measured) * input_len / measure_time,
        "logical_tokens_per_s_measured": len(measured) * (input_len + output_len) / measure_time,
        "output_tokens_per_s_steady": len(steady_completed) * output_len / steady_time,
        "input_tokens_per_s_steady": len(steady_completed) * input_len / steady_time,
        "logical_tokens_per_s_steady": len(steady_completed) * (input_len + output_len) / steady_time,
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
    parser = argparse.ArgumentParser(description="Run baseline S-LoRA full-prefill throughput experiment.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--model-dir", type=str, default="huggyllama/llama-7b",
                        help="Tokenizer/model dir used only to build exact-length full prompts.")
    parser.add_argument("--adapter-base", action="append", default=None,
                        help="Base adapter name/path. Repeated to build the adapter pool.")
    parser.add_argument("--num-adapters", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--request-rate", type=float, required=True,
                        help="Poisson arrival rate in requests/s.")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--warmup", type=float, default=40.0)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--steady-end", type=float, default=None,
                        help="Manual steady-state completion window end time relative to benchmark start.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-len", type=int, default=1920)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--max-context-len", type=int, default=2048)
    parser.add_argument("--kv-budget-multiple", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--prompt-token-id", type=int, default=100)
    parser.add_argument("--output", type=str, default="slora_baseline_throughput_results.jsonl")
    parser.add_argument("--dump-per-request", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logical_len = args.input_len + args.output_len
    if logical_len > args.max_context_len:
        raise ValueError(
            f"logical_len={logical_len} exceeds max_context_len={args.max_context_len}"
        )
    total_kv_budget = args.kv_budget_multiple * args.max_context_len

    adapter_bases = args.adapter_base or ["dummy-lora-7b-rank-16"]
    adapter_dirs = expand_adapter_dirs(adapter_bases, args.num_adapters)
    input_ids = [args.prompt_token_id] * args.input_len
    requests = make_requests(
        adapter_dirs=adapter_dirs,
        input_len=args.input_len,
        output_len=args.output_len,
        request_rate=args.request_rate,
        duration=args.duration,
        alpha=args.alpha,
        seed=args.seed,
        input_ids=input_ids,
    )

    config = {
        "backend": "slora_baseline",
        "semantics": "full input prefill; no shared prefix cache; KV freed when request finishes",
        "input_len": args.input_len,
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
        benchmark_start_time=start,
        duration=args.duration,
        warmup=args.warmup,
        cooldown=args.cooldown,
        steady_end_override=args.steady_end,
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
