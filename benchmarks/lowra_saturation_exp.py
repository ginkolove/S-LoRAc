"""Paired full-cache / layer-serial saturation experiments over real HTTP serving.

Uses run_exp.py's open-loop asynchronous replay structure, with exact token inputs,
saved synthetic or LMSYS/MAF traces, parsed SSE completion, and completion-window statistics.
Each invocation manages one server. No CUDA profiler is enabled.
"""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

import aiohttp
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.lowra_shared_ratio_exp import expand_adapter_dirs, zipf_probs
from slora.common.prefix_modes import (
    FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, LAYER_RETAINED, LAYER_PREFETCH,
    normalize_prefix_mode,
)
from benchmarks.saturation_cycles import decode_cache_entries, summarize_cycles
from benchmarks.lmsys_maf_trace import load_trace, validate_run, bind_requests


@dataclass
class RequestResult:
    req_id: str
    adapter_dir: str
    scheduled_time: float
    start_time: float
    first_token_time: object
    end_time: float
    output_tokens: int
    stream_events: int
    finished: bool
    success: bool
    status: object
    error: object
    query_tokens: int = 0
    expected_output_tokens: int = 0
    source_id: object = None
    adapter_index: object = None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def trace_for(args, rate, seed):
    if getattr(args, "workload_trace", None):
        trace, digest = load_trace(args.workload_trace)
        validate_run(trace, args)
        if digest != args.workload_sha256:
            raise ValueError("workload trace changed after argument validation")
        if rate != trace["config"]["request_rate"] or seed != trace["config"]["seed"]:
            raise ValueError("workload rate or seed mismatch")
        dirs = expand_adapter_dirs(args.adapter_base, args.num_adapters)
        return bind_requests(trace, dirs), args.workload_trace, digest
    config = dict(adapter_bases=args.adapter_base, num_adapters=args.num_adapters,
                  prefix_len=args.prefix_len, query_len=args.query_len,
                  output_len=args.output_len, prompt_token_id=args.prompt_token_id,
                  request_rate=rate, duration=args.duration, alpha=args.alpha,
                  cv=args.cv, seed=seed)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    path = args.trace_dir / ("trace_%s.json" % digest)
    if path.exists():
        trace = json.loads(path.read_text())
        if trace["config"] != config:
            raise ValueError("trace config mismatch: %s" % path)
    else:
        rng = np.random.default_rng(seed)
        dirs = expand_adapter_dirs(args.adapter_base, args.num_adapters)
        probs = zipf_probs(args.num_adapters, args.alpha)
        requests = []
        arrival = 0.0
        while True:
            arrival += float(rng.gamma(1.0 / args.cv ** 2, args.cv ** 2 / rate))
            if arrival > args.duration:
                break
            directory = dirs[int(rng.choice(len(dirs), p=probs))]
            requests.append(dict(req_id="sat-%s-%s" % (digest, len(requests)),
                                 adapter_dir=directory, req_time=arrival))
        trace = dict(config=config, requests=requests)
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        # Concurrent modes generate identical traces; replace only a complete file.
        temporary = path.with_suffix(".%s.tmp" % os.getpid())
        write_json(temporary, trace)
        os.replace(temporary, path)
    content_hash = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    return trace["requests"], path, content_hash


def decode_sse(data):
    event = json.loads("\n".join(data))
    if event.get("finished") == -1:
        raise ValueError("server aborted request")
    if event.get("finished") not in (False, True):
        raise ValueError("invalid SSE finished flag")
    token = event.get("token")
    if not isinstance(token, dict) or token.get("id") is None:
        raise ValueError("SSE event has no token id")
    if token.get("logprob") is not None and not math.isfinite(float(token["logprob"])):
        raise ValueError("nonfinite output logprob")
    count = event.get("count_output_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("server must report cumulative count_output_tokens")
    return count, bool(event.get("finished"))


def request_payload(request, args):
    query_ids = request.get("input_ids")
    if query_ids is None:
        query_ids = [args.prompt_token_id] * args.query_len
    output_len = request.get("output_len", args.output_len)
    natural_generation = getattr(args, "generation_mode", "trace") == "eos"
    if natural_generation:
        output_len = min(args.generation_max_new_tokens,
                         args.max_context_len - args.prefix_len - len(query_ids))
        if output_len <= 0:
            raise ValueError("request has no context space left for generation")
    inputs = query_ids
    if normalize_prefix_mode(args.mode) == FULL_REPREFILL:
        inputs = [args.prompt_token_id] * args.prefix_len + query_ids
    return dict(req_id=request["req_id"], lora_dir=request["adapter_dir"],
                   inputs=inputs,
                   parameters=dict(do_sample=False, ignore_eos=not natural_generation,
                                   max_new_tokens=output_len, return_details=False),
                   lowra_lengths=dict(shared_prefix_len=args.prefix_len,
                                     query_len=len(query_ids), decode_len=output_len))


async def send_request(session, server, request, args, origin):
    start = time.monotonic()
    result = RequestResult(request["req_id"], request["adapter_dir"], request["req_time"],
                           start - origin, None, 0.0, 0, 0, False, False, None, None)
    payload = request_payload(request, args)
    result.query_tokens = payload["lowra_lengths"]["query_len"]
    result.expected_output_tokens = payload["parameters"]["max_new_tokens"]
    result.source_id = request.get("source_id")
    result.adapter_index = request.get("adapter_index")
    try:
        timeout = aiohttp.ClientTimeout(total=args.request_timeout)
        async with session.post(server + "/generate_stream", json=payload, timeout=timeout) as response:
            result.status = response.status
            if response.status != 200:
                raise ValueError("HTTP %s: %s" % (response.status, (await response.text())[:1000]))
            data = []
            while True:
                line = await response.content.readline()
                if not line:
                    if data:
                        raise ValueError("truncated SSE frame")
                    break
                line = line.decode("utf-8").strip()
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    count, finished = decode_sse(data)
                    data.clear()
                    if result.finished or count <= result.output_tokens:
                        raise ValueError("nonmonotonic token count or events after completion")
                    result.output_tokens = count
                    result.stream_events += 1
                    if result.first_token_time is None:
                        result.first_token_time = time.monotonic() - origin
                    result.finished = finished
            length_ok = (0 < result.output_tokens <= result.expected_output_tokens
                         if getattr(args, "generation_mode", "trace") == "eos"
                         else result.output_tokens == result.expected_output_tokens)
            if not result.finished or not length_ok:
                raise ValueError("incomplete generation: finished=%s tokens=%s expected=%s" %
                                 (result.finished, result.output_tokens, result.expected_output_tokens))
            result.success = True
    except asyncio.CancelledError:
        result.error = "cancelled or drain timeout"
    except Exception as exc:
        result.error = repr(exc)
    result.end_time = time.monotonic() - origin
    return result


async def replay(session, server, requests, args, results_path, origin):
    results = []
    tasks = []
    with results_path.open("w") as output, tqdm(total=len(requests), desc=results_path.stem,
                                                unit="req", mininterval=5) as progress:
        async def worker(request):
            result = await send_request(session, server, request, args, origin)
            results.append(result)
            output.write(json.dumps(asdict(result), allow_nan=False) + "\n")
            output.flush()
            progress.update(1)
        try:
            for request in requests:
                await asyncio.sleep(max(0.0, origin + request["req_time"] - time.monotonic()))
                tasks.append(asyncio.create_task(worker(request)))
            await asyncio.sleep(max(0.0, origin + args.duration - time.monotonic()))
            if tasks:
                remaining = max(0.0, origin + args.duration + args.drain_timeout - time.monotonic())
                _, pending = await asyncio.wait(tasks, timeout=remaining)
                if pending:
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending)
                    print("Drain timeout: point will be marked invalid", flush=True)
                await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return results


def distribution(values):
    if not values:
        return dict(mean=None, p50=None, p95=None, p99=None)
    return dict(mean=float(np.mean(values)),
                **{"p%s" % q: float(np.percentile(values, q)) for q in (50, 95, 99)})


def summarize(results, args):
    successful = [result for result in results if result.success]
    actual_end = max((result.end_time for result in results), default=0.0)
    steady_end = min(args.steady_end, actual_end)
    steady_time = max(0.0, steady_end - args.warmup)
    steady = [result for result in successful if args.warmup <= result.end_time <= steady_end]
    bins = []
    start = args.warmup
    while start < steady_end:
        end = min(start + args.bin_seconds, steady_end)
        # Only the final bin includes its right endpoint, including last completion.
        count = sum(start <= result.end_time < end
                    or (end == steady_end and result.end_time == end)
                    for result in successful)
        bins.append(dict(start=start, end=end, completed_requests=count, req_per_s=count / (end - start)))
        start = end
    latencies = distribution([r.end_time - r.start_time for r in successful])
    ttfts = distribution([r.first_token_time - r.start_time for r in successful])
    return dict(
        total_requests=len(results), completed_requests=len(successful),
        failed_requests=len(results) - len(successful), valid=len(successful) == len(results) and bool(results),
        benchmark_time=actual_end,
        completed_req_per_s_total=len(successful) / actual_end if actual_end > 0 else 0.0,
        output_tokens_per_s_total=sum(r.output_tokens for r in successful) / actual_end if actual_end > 0 else 0.0,
        query_tokens=distribution([r.query_tokens for r in results]),
        expected_output_tokens=distribution([r.expected_output_tokens for r in results]),
        steady_start=args.warmup, steady_end_requested=args.steady_end,
        steady_end=steady_end, steady_time=steady_time, steady_completed_requests=len(steady),
        completed_req_per_s_steady=len(steady) / steady_time if steady_time > 0 else 0.0,
        output_tokens_per_s_steady=sum(r.output_tokens for r in steady) / steady_time if steady_time > 0 else 0.0,
        steady_scope="all_successful_completions_in_[warmup,min(steady_end,actual_end)]",
        latency_ttft_scope="all_successful_requests",
        **{"latency_" + key: value for key, value in latencies.items()},
        **{"ttft_" + key: value for key, value in ttfts.items()},
        send_lag_seconds=distribution([max(0.0, r.start_time - r.scheduled_time) for r in results]),
        completion_bins=bins,
        errors=[asdict(r) for r in results if not r.success][:20],
    )


def telemetry_summary(path, origin, args, result):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    window = [row for row in rows
              if args.warmup <= row["monotonic_time"] - origin < result["steady_end"]]
    point = [row for row in rows if 0 <= row["monotonic_time"] - origin
             <= result["benchmark_time"] + args.stats_interval + 0.2]
    if not point:
        return dict(samples=0, saturation_evidence=False, error="no runtime snapshots")
    fraction = sum(row["waiting_queue_size"] > 0 for row in window) / len(window) if window else 0.0
    end_backlog = window[-1]["waiting_queue_size"] if window else 0
    last = point[-1]["per_rank"][0]
    cache = last.get("prefix_cache_stats")
    if cache:
        cache = dict(cache)
        accesses = cache["batch_unique_hits"] + cache["batch_unique_misses"]
        cache["hit_ratio"] = cache["batch_unique_hits"] / accesses if accesses else None
    summary = dict(
        reconstruction_configurations=[dict(kernel=kernel, workers=workers) for kernel, workers in sorted({
            (rank["reconstruction_kernel"], rank.get("reconstruction_workers"))
            for row in point for rank in row["per_rank"] if rank.get("reconstruction_kernel") is not None
        })],
        samples=len(window), sampling="low_frequency_after_forward_steps_not_time_weighted",
        waiting_nonempty_sample_fraction=fraction, end_of_window_waiting=end_backlog,
        backlogged_window=len(window) >= 5 and fraction >= 0.95 and end_backlog > 0,
        running_request_limit_observed=any(row["running_batch_size"] >= args.running_max_req_size for row in window),
        prefill_token_limit_observed=any(row.get("last_admission_tokens", row.get("last_prefill_tokens", row.get("last_prefill_query_tokens", 0))) + args.query_len
                                        > args.batch_max_tokens for row in window),
        prefill_token_limit_scope=("heuristic_using_max_query_length" if getattr(args, "workload_trace", None)
                                   else "heuristic_using_fixed_query_length"),
        prefix_cache_access_scope="unique_lora_per_new_batch_over_entire_point",
        prefix_cache_stats=cache,
        total_budget_tokens=args.num_token,
        static_token_equivalent=point[-1]["static_token_equivalent"],
        persistent_total_slots=last["persistent_total_slots"],
        workspace_token_equivalent=last["workspace_token_equivalent"],
        workspace_backing_bytes=last["workspace_backing_bytes"],
        workspace_peak_tokens=last.get("workspace_peak_tokens", 0),
        workspace_peak_token_equivalent=last.get("workspace_peak_token_equivalent", 0),
        workspace_peak_backing_bytes=last.get("workspace_peak_backing_bytes", 0),
        base_model_resident_bytes=last["base_model_resident_bytes"],
        peak_allocated_bytes=max(row["per_rank"][0]["peak_allocated_bytes"] for row in point),
        peak_allocated_without_base_model_bytes=max(
            row["per_rank"][0]["peak_allocated_without_base_model_bytes"] for row in point),
        peak_reserved_bytes=max(row["per_rank"][0]["peak_reserved_bytes"] for row in point),
    )
    for field in ("running_batch_size", "running_unique_loras", "waiting_queue_size"):
        summary[field] = dict(distribution([row[field] for row in window]),
                              maximum=max((row[field] for row in window), default=0))
    for field in ("persistent_used_slots", "private_kv_slots", "prefix_slots", "adapter_slots"):
        values = [row["per_rank"][0][field] for row in window]
        summary[field] = dict(distribution(values), maximum=max(values, default=0))
    for field in ("online_preemptions", "online_preempted_requests", "online_recomputed_tokens", "online_rejections"):
        summary[field] = point[-1].get(field)
    return summary


def preemption_summary(path, origin, results, benchmark_time):
    if not path.exists():
        return dict(available=False)
    identifiers = {result.req_id for result in results}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["monotonic_time"] >= origin]
    preemptions = [row for row in rows if row["event"] == "preempt" and row["req_id"] in identifiers]
    resumes = [row for row in rows if row["event"] == "resume_prefill"
               and all(req["req_id"] in identifiers for req in row["requests"])]
    counts = Counter(row["req_id"] for row in preemptions)
    prefill_seconds = sum(row["prefill_wall_seconds"] for row in resumes)
    groups = {}
    for name, affected in (("preempted", True), ("not_preempted", False)):
        selected = [result for result in results if (result.req_id in counts) == affected]
        successful = [result for result in selected if result.success]
        groups[name] = dict(
            requests=len(selected), completed_requests=len(successful),
            latency_seconds=distribution([result.end_time - result.start_time for result in successful]),
            ttft_seconds=distribution([result.first_token_time - result.start_time for result in successful
                                       if result.first_token_time is not None]))
    return dict(
        available=True, preemption_events=len(preemptions), preempted_requests=len(counts),
        preempted_request_fraction=len(counts) / len(results) if results else 0.0,
        preemptions_per_request=dict(counts),
        resumed_request_events=sum(len(row["requests"]) for row in resumes),
        recomputed_tokens=sum(req["input_tokens"] + req["generated_tokens"]
                              for row in resumes for req in row["requests"]),
        recomputed_tokens_scope="rpc_input_plus_generated_history_at_resume; "
                                "excludes_shared_prefix_in_minikv_modes",
        preemption_release_wall_seconds=sum(row["release_wall_seconds"] for row in preemptions),
        resume_prefill_batches=len(resumes),
        mixed_resume_prefill_batches=sum(row["new_requests"] > 0 for row in resumes),
        resume_batch_prefill_wall_seconds=prefill_seconds,
        resume_only_batch_prefill_wall_seconds=sum(row["prefill_wall_seconds"] for row in resumes
                                                  if row["new_requests"] == 0),
        resume_batch_prefill_fraction_of_run=prefill_seconds / benchmark_time if benchmark_time else 0.0,
        timing_scope="whole_resume_batch_init_forward_cleanup_excludes_adapter_load_and_queue_wait; "
                     "includes_new_requests_if_mixed_and_next_token_work; not_causal_throughput_loss",
        latency_comparison_scope="observational_groups_not_a_matched_causal_comparison",
        request_groups=groups)


def server_command(args, stats_path):
    command = [sys.executable, "-u", "-m", "slora.server.api_server",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--nccl_port", str(args.nccl_port), "--internal-port-start", str(args.internal_port_start),
               "--model_dir", args.model_dir, "--tokenizer_mode", "auto", "--tp", "1",
               "--max_total_token_num", str(args.num_token),
               "--max_req_input_len", str(args.prefix_len + args.query_len),
               "--max_req_total_len", str(args.max_context_len),
               "--batch_max_tokens", str(args.batch_max_tokens),
               "--running_max_req_size", str(args.running_max_req_size),
               "--model-seed", str(args.model_seed), "--swap", "--enable-lowra",
               "--lowra-shared-prefix-len", str(args.prefix_len), "--lowra-prefix-mode", args.mode,
               "--lowra-runtime-stats-path", str(stats_path),
               "--lowra-runtime-stats-interval", str(args.stats_interval)]
    command.extend(["--scheduler", getattr(args, "scheduler", "online"),
                    "--schedule-interval", str(getattr(args, "schedule_interval", 1))])
    if getattr(args, "max_active_loras", None) is not None:
        command.extend(["--max-active-loras", str(args.max_active_loras)])
    command.extend(["--lowra-reconstruction-kernel", args.reconstruction_kernel])
    if args.reconstruction_workers is not None:
        command.extend(["--lowra-reconstruction-workers", str(args.reconstruction_workers)])
    if args.mode == LAYER_RETAINED:
        command.extend(["--lowra-retain-layers", str(args.retain_layers)])
        if args.retain_top_k is not None:
            command.extend(["--lowra-retain-top-k", str(args.retain_top_k)])
    if args.pin_hot_adapters:
        command.extend(["--lowra-pin-hot-adapters", str(args.pin_hot_adapters)])
    for directory in expand_adapter_dirs(args.adapter_base, args.num_adapters):
        command.extend(["--lora-dirs", directory])
    if args.dummy:
        command.append("--dummy")
    return command


def stop_server(process):
    # The entire group belongs to this invocation, including router/detokenizer children.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


async def wait_ready(session, server, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited during initialization; inspect server.log")
        try:
            async with session.get(server + "/health", timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status == 200:
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(1)
    raise TimeoutError("server startup timed out; inspect server.log")


async def reset_runtime(session, server):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        async with session.post(server + "/lowra/reset_runtime",
                                timeout=aiohttp.ClientTimeout(total=65)) as response:
            result = await response.json()
            if response.status == 200 and result.get("ok"):
                for rank in result["stats"]["per_rank"]:
                    if rank["free_slots_after"] != rank["total_slots"]:
                        raise RuntimeError("runtime reset leaked persistent slots")
                return result
            if response.status != 409:
                raise RuntimeError("runtime reset failed: %s" % result)
        await asyncio.sleep(0.5)
    raise TimeoutError("server did not become idle for reset")


def suite_summary(points, args):
    rates = []
    for rate in args.request_rates:
        group = [point for point in points if point["request_rate"] == rate]
        valid = len(group) == len(args.seeds) and all(point["result"]["valid"] for point in group)
        values = [point["result"]["completed_req_per_s_steady"] for point in group]
        rates.append(dict(request_rate=rate, valid=valid,
                          completed_req_per_s_steady_mean=float(np.mean(values)) if values else None,
                          completed_req_per_s_steady_std=float(np.std(values)) if values else None,
                          backlogged_all_seeds=bool(group) and all(
                              point["runtime"].get("backlogged_window", False) for point in group)))
        if getattr(args, "measure_saturation", False):
            qualified = bool(group) and all(point["result"].get("saturation", {}).get("qualified")
                                            for point in group)
            observed = [point["result"].get("completed_req_per_s_saturated") for point in group]
            rates[-1].update(
                complete_cycle_capacity_qualified=qualified,
                completed_req_per_s_saturated_mean=(float(np.mean(observed))
                    if observed and all(value is not None for value in observed) else None),
                completed_req_per_s_peak_mean=(float(np.mean(observed)) if qualified else None))
    platform_pairs = []
    for previous, current in zip(rates, rates[1:]):
        low, high = previous["completed_req_per_s_steady_mean"], current["completed_req_per_s_steady_mean"]
        relative = abs(high - low) / low if low and high is not None else None
        if (previous["valid"] and current["valid"] and previous["backlogged_all_seeds"]
                and current["backlogged_all_seeds"] and relative is not None
                and relative <= args.plateau_tolerance):
            platform_pairs.append([previous["request_rate"], current["request_rate"]])
    confirmed = any(a[1] == b[0] for a, b in zip(platform_pairs, platform_pairs[1:]))
    return dict(mode=args.mode, rates=rates, plateau_pairs=platform_pairs,
                saturation_confirmed=confirmed,
                saturation_rule="two_consecutive_plateau_pairs_and_backlogged_windows_all_seeds",
                note=("Inspect batch/slot curves and measurement variance before claiming saturation."
                      if confirmed else "Saturation not confirmed; extend rates or measurement time."))


async def run_suite(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("output already contains a run; choose a new --output-dir")
    prepared_traces = {(rate, seed): trace_for(args, rate, seed)
                       for rate in args.request_rates for seed in args.seeds}
    for port in (args.port, args.nccl_port, *range(args.internal_port_start, args.internal_port_start + 4)):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("required port %s already in use; select different ports" % port)
    stats_path = output / "runtime.jsonl"
    command = server_command(args, stats_path)
    packages = {}
    for package in ("torch", "triton", "numpy", "aiohttp"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
                          "--format=csv"], capture_output=True, text=True, check=False)
    write_json(manifest_path, dict(config={key: str(value) if isinstance(value, Path) else value
                                         for key, value in vars(args).items()},
                                   prefix_source=("ordinary_lora_prefill" if args.mode == FULL_REPREFILL
                                                  else "mini_kv_reconstruction"),
                                   command=command, python=sys.executable,
                                   platform=platform.platform(), packages=packages,
                                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   thread_environment={key: os.environ.get(key) for key in
                                                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                       for name in ("slora/models/lowra/decode_attention.py",
                                                    "slora/models/lowra/prefix_attention.py",
                                                    "slora/models/peft/lora_unordered_batch_infer.py",
                                                    "slora/common/prefix_indices.py",
                                                    "slora/common/full_reprefill.py",
                                                    "slora/common/full_reconstruct.py",
                                                    "slora/common/lowra_layer_runtime.py",
                                                    "slora/common/lowra_reconstruct_kv.py",
                                                    "slora/common/lowra_reconstruct.py",
                                                    "slora/common/lowra_reconstruct_bounded.py",
                                                    "slora/common/lowra_retained.py",
                                                    "slora/common/lowra_runtime_stats.py",
                                                    "slora/common/service_cycles.py",
                                                    "slora/server/router/manager.py",
                                                    "slora/server/router/model_infer/model_rpc.py",
                                                    "slora/server/router/lowra_fifo_req_queue.py",
                                                    "slora/server/router/full_reprefill_req_queue.py",
                                                    "benchmarks/lowra_saturation_exp.py",
                                                    "benchmarks/lmsys_maf_trace.py",
                                                    "benchmarks/prepare_lmsys_maf_trace.py",
                                                    "benchmarks/saturation_cycles.py")},
                                   gpu_before=gpu.stdout, gpu_query_error=gpu.stderr))
    points = []
    server = "http://127.0.0.1:%s" % args.port
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        try:
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
                await wait_ready(session, server, process, args.startup_timeout)
                print("Ready: mode=%s server=%s" % (args.mode, server), flush=True)
                if args.preflight_requests:
                    dirs = expand_adapter_dirs(args.adapter_base, args.num_adapters)
                    origin = time.monotonic()
                    requests = [dict(req_id="preflight-%s" % i, adapter_dir=dirs[i % min(2, len(dirs))],
                                     req_time=0.0) for i in range(args.preflight_requests)]
                    if getattr(args, "workload_trace", None):
                        samples = prepared_traces[(args.request_rates[0], args.seeds[0])][0]
                        # Exercise real query/output pairs rather than unrelated maxima.
                        requests = [dict(samples[i % len(samples)], req_id="preflight-%s" % i,
                                         adapter_dir=dirs[i % min(2, len(dirs))],
                                         adapter_index=i % min(2, len(dirs)), req_time=0.0)
                                    for i in range(args.preflight_requests)]
                    checks = await asyncio.gather(*[send_request(session, server, req, args, origin)
                                                     for req in requests])
                    write_json(output / "preflight.json", [asdict(check) for check in checks])
                    if not all(check.success for check in checks):
                        raise RuntimeError("preflight failed; inspect preflight.json and server.log")
                for rate in args.request_rates:
                    for seed in args.seeds:
                        if process.poll() is not None:
                            raise RuntimeError("server exited; inspect server.log")
                        await reset_runtime(session, server)
                        cache_before = decode_cache_entries() if args.measure_saturation else set()
                        requests, trace_path, trace_hash = prepared_traces[(rate, seed)]
                        label = "rate%s_seed%s" % (format(rate, "g"), seed)
                        write_json(output / (label + "_trace.json"), dict(
                            trace_path=str(trace_path), trace_sha256=trace_hash, requests=requests))
                        print("Starting %s: %s requests, trace=%s" % (label, len(requests), trace_hash), flush=True)
                        origin = time.monotonic()
                        results = await replay(session, server, requests, args,
                                               output / (label + "_requests.jsonl"), origin)
                        result = summarize(results, args)
                        if len(results) != len(requests):
                            result["valid"] = False
                        # Ensure the final peak/cache snapshot is captured before reset.
                        await asyncio.sleep(args.stats_interval + 0.1)
                        runtime = telemetry_summary(stats_path, origin, args, result)
                        preemption = preemption_summary(
                            stats_path.with_suffix(".preemptions.jsonl"), origin, results, result["benchmark_time"])
                        if args.measure_saturation:
                            cycles_path = stats_path.with_suffix(".cycles.jsonl")
                            cycles = [json.loads(line) for line in cycles_path.read_text().splitlines()
                                      if line.strip()] if cycles_path.exists() else []
                            saturation = summarize_cycles(
                                cycles, results, origin, args.warmup, args.output_len,
                                args.saturation_warmup_cycles, args.saturation_min_cycles,
                                args.saturation_min_seconds, args.saturation_tolerance,
                                len(decode_cache_entries() - cache_before))
                            result["saturation"] = saturation
                            result["completed_req_per_s_saturated"] = saturation["saturated_completed_req_per_s"]
                            result["completed_req_per_s_peak"] = saturation["peak_completed_req_per_s"]
                        if runtime["samples"] == 0:
                            result["valid"] = False
                            result["invalid_reason"] = "no runtime snapshots in measurement window"
                        point = dict(mode=args.mode, retain_layers=args.retain_layers,
                                     retain_top_k=args.retain_top_k,
                                     pin_hot_adapters=args.pin_hot_adapters,
                                     request_rate=rate, seed=seed,
                                     trace_path=str(trace_path), trace_sha256=trace_hash,
                                     adapter_counts=dict(Counter(req["adapter_dir"] for req in requests)),
                                     benchmark_origin_monotonic=origin, result=result, runtime=runtime,
                                     preemption=preemption)
                        write_json(output / (label + ".json"), point)
                        points.append(point)
                        write_json(output / "summary.json", suite_summary(points, args))
                        print("%s: steady=%.6f req/s total=%.6f req/s valid=%s backlog=%s" %
                              (label, result["completed_req_per_s_steady"], result["completed_req_per_s_total"],
                               result["valid"], runtime.get("backlogged_window")), flush=True)
                        print("Preemption diagnostics: %s" % json.dumps(preemption), flush=True)
                        if args.measure_saturation:
                            print("Saturation cycles: %s" % json.dumps(result["saturation"]), flush=True)
                        if not result["valid"]:
                            raise RuntimeError("invalid point; stopping instead of benchmarking a broken server")
                        await reset_runtime(session, server)
        finally:
            stop_server(process)
    print("Finished: %s" % (output / "summary.json"), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=normalize_prefix_mode,
                        choices=(FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, LAYER_RETAINED,
                                 LAYER_PREFETCH), required=True)
    parser.add_argument("--retain-layers", type=int, default=0,
                        help="First K prefix layers retained per active adapter in minikv-layer_retained.")
    parser.add_argument("--reconstruction-kernel", choices=("auto", "bounded"), default="auto")
    parser.add_argument("--reconstruction-workers", type=int, default=None)
    parser.add_argument("--retain-top-k", type=int, default=None,
                        help="Retain only the K adapters with the most active requests; omitted means all.")
    parser.add_argument("--pin-hot-adapters", type=int, default=0,
                        help="Pin the first N Zipf-ranked adapters after first use until runtime reset.")
    parser.add_argument("--model-dir", default="huggyllama/llama-7b")
    parser.add_argument("--adapter-base", action="append")
    parser.add_argument("--num-adapters", type=int, default=50)
    parser.add_argument("--prefix-len", type=int, default=1440)
    parser.add_argument("--query-len", type=int, default=480)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--max-context-len", type=int, default=2048)
    parser.add_argument("--num-token", type=int, default=36864)
    parser.add_argument("--batch-max-tokens", type=int)
    parser.add_argument("--running-max-req-size", type=int, default=256)
    parser.add_argument("--scheduler", choices=("online", "slora"), default="online",
                        help="online: no output-length-based admission; slora: legacy peak reservation.")
    parser.add_argument("--schedule-interval", type=int, default=1)
    parser.add_argument("--max-active-loras", type=int, default=None)
    parser.add_argument("--generation-mode", choices=("trace", "eos"), default="trace",
                        help="trace: replay reference lengths (termination only with online); eos: real EOS generation.")
    parser.add_argument("--generation-max-new-tokens", type=int, default=512,
                        help="Common generation cap in eos mode, unrelated to reference answer lengths.")
    parser.add_argument("--request-rates", type=float, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--warmup", type=float, default=30)
    parser.add_argument("--steady-end", type=float, default=90,
                        help="Completion-window end; may exceed sending duration while queued work drains.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cv", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--prompt-token-id", type=int, default=10)
    parser.add_argument("--port", type=int)
    parser.add_argument("--nccl-port", type=int)
    parser.add_argument("--internal-port-start", type=int)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--drain-timeout", type=float, default=1800)
    parser.add_argument("--preflight-requests", type=int, default=8)
    parser.add_argument("--measure-saturation", action="store_true",
                        help="Measure complete backlogged cycles, independent of the short steady window.")
    parser.add_argument("--saturation-warmup-cycles", type=int, default=2)
    parser.add_argument("--saturation-min-cycles", type=int, default=6)
    parser.add_argument("--saturation-min-seconds", type=float, default=300)
    parser.add_argument("--saturation-tolerance", type=float, default=0.1)
    parser.add_argument("--stats-interval", type=float, default=1)
    parser.add_argument("--bin-seconds", type=float, default=10)
    parser.add_argument("--plateau-tolerance", type=float, default=0.05)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "benchmarks/results/lowra_saturation/traces")
    parser.add_argument("--workload-trace", type=Path,
                        help="Frozen LMSYS/MAF trace; per-request tokens and output lengths override fixed lengths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    if args.schedule_interval <= 0 or (args.max_active_loras is not None and args.max_active_loras <= 0):
        parser.error("schedule-interval and max-active-loras must be positive")
    if args.generation_max_new_tokens <= 0:
        parser.error("generation-max-new-tokens must be positive")
    args.output_length_admission = "hidden" if args.scheduler == "online" else "max_new_tokens_peak_reservation"
    if args.reconstruction_kernel == "bounded" and args.mode not in (LAYER_SERIAL, LAYER_PREFETCH, LAYER_RETAINED):
        parser.error("bounded reconstruction requires a layer mode")
    if args.reconstruction_workers is not None and (
            args.reconstruction_kernel != "bounded" or args.reconstruction_workers <= 0):
        parser.error("reconstruction-workers requires bounded and a positive integer")
    args.adapter_base = args.adapter_base or ["dummy-lora-7b-rank-16"]
    serial = args.mode == LAYER_SERIAL
    ports = (8002, 28767, 14000) if args.mode == FULL_RECONSTRUCT else (
        (8001, 28766, 13000) if serial else (8000, 28765, 12000))
    if args.mode == LAYER_RETAINED:
        ports = (8003, 28768, 15000)
    if args.mode == LAYER_PREFETCH:
        ports = (8004, 28769, 16000)
    if args.retain_layers < 0 or (args.retain_layers and args.mode != LAYER_RETAINED):
        parser.error("--retain-layers requires minikv-layer_retained and a nonnegative count")
    if args.retain_top_k is not None and (args.retain_top_k < 0 or args.mode != LAYER_RETAINED):
        parser.error("--retain-top-k requires minikv-layer_retained and a nonnegative count")
    if not 0 <= args.pin_hot_adapters <= args.num_adapters:
        parser.error("--pin-hot-adapters must be between zero and num-adapters")
    if args.pin_hot_adapters and (args.mode not in (FULL_RECONSTRUCT, LAYER_RETAINED)
                                 or args.retain_top_k is not None):
        parser.error("--pin-hot-adapters requires a MiniKV cache mode and no dynamic --retain-top-k")
    if args.pin_hot_adapters and args.mode == LAYER_RETAINED and args.retain_layers <= 0:
        parser.error("pinned layer retention requires positive --retain-layers")
    args.port = args.port if args.port is not None else ports[0]
    args.nccl_port = args.nccl_port if args.nccl_port is not None else ports[1]
    args.internal_port_start = args.internal_port_start if args.internal_port_start is not None else ports[2]
    args.batch_max_tokens = args.batch_max_tokens if args.batch_max_tokens is not None else args.num_token
    args.output_dir, args.trace_dir = args.output_dir.resolve(), args.trace_dir.resolve()
    args.workload_kind = "lmsys_maf_poisson" if args.workload_trace else "synthetic_zipf_gamma"
    if args.workload_trace:
        args.workload_trace = args.workload_trace.resolve()
        try:
            trace, args.workload_sha256 = load_trace(args.workload_trace)
            validate_run(trace, args)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            parser.error(str(exc))
        args.query_len = max(len(request["input_ids"]) for request in trace["requests"])
        args.output_len = max(request["output_len"] for request in trace["requests"])
        args.maf_skewness = trace["config"].get("maf_skewness")
    positive = (args.num_adapters, args.prefix_len, args.query_len, args.output_len, args.num_token,
                args.running_max_req_size, args.duration, args.cv, args.stats_interval,
                args.bin_seconds, args.startup_timeout, args.request_timeout, args.drain_timeout)
    floats = (*positive, args.warmup, args.steady_end, args.alpha, args.plateau_tolerance,
              *args.request_rates)
    if not all(math.isfinite(value) for value in floats) or min(positive) <= 0:
        parser.error("lengths, capacities, rates, and timeouts must be finite and positive")
    if args.alpha < 0 or args.prompt_token_id < 0 or args.preflight_requests < 0:
        parser.error("alpha, prompt-token-id, and preflight-requests must be nonnegative")
    if (args.saturation_warmup_cycles < 0 or args.saturation_min_cycles < 3
            or not math.isfinite(args.saturation_min_seconds) or args.saturation_min_seconds <= 0
            or not math.isfinite(args.saturation_tolerance) or not 0 < args.saturation_tolerance < 1):
        parser.error("saturation needs nonnegative warmup cycles, >=3 cycles, positive seconds and 0<tolerance<1")
    if not 0 <= args.warmup < args.steady_end:
        parser.error("completion measurement requires 0 <= warmup < steady-end")
    if not args.workload_trace and args.prefix_len + args.query_len + args.output_len > args.max_context_len:
        parser.error("prefix + query + output exceeds max-context-len")
    if args.batch_max_tokens < args.max_context_len:
        parser.error("batch-max-tokens must be at least max-context-len")
    if min(args.request_rates) <= 0 or len(set(args.request_rates)) != len(args.request_rates):
        parser.error("request-rates must be positive and unique")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be nonnegative and unique")
    if not 0 < args.plateau_tolerance < 1:
        parser.error("plateau-tolerance must be between 0 and 1")
    if not 1024 <= args.internal_port_start <= 65530:
        parser.error("internal-port-start must be between 1024 and 65530")
    if any(not 1024 <= port <= 65535 for port in (args.port, args.nccl_port)):
        parser.error("HTTP and NCCL ports must be between 1024 and 65535")
    if args.port == args.nccl_port:
        parser.error("HTTP and NCCL ports must differ")
    args.request_rates.sort()
    return args


async def main(args):
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, task.cancel)
    await run_suite(args)


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("Interrupted; owned server processes stopped", file=sys.stderr)
        sys.exit(130)
