"""
To run with real mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp.py --backend slora --suite a10g --breakdown  --mode synthetic
default to synthetic mode.
"""
import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from typing import Dict, List, Tuple

import aiohttp

from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import generate_requests, get_real_requests
sys.path.append("../bench_lora")
from slora.utils.metric import reward, attainment_func

GB = 1024 ** 3

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
vllm_packed_adapter_dir_to_url_map = {}

def get_peak_mem(server):
    url = server + "/get_peak_mem"
    response = requests.post(url)
    return response.json()["peak_mem"]


async def send_request(
    backend: str,
    server: str,
    req_id: str,
    model_dir: str,
    adapter_dir: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    debug: bool,
) -> None:
    request_start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        url = server + "/generate"
    elif backend == "vllm-packed":
        url = vllm_packed_adapter_dir_to_url_map[adapter_dir] + "/generate"
    else:
        url = server + "/generate_stream"
    
    if backend in ["slora"]:
        data = {
            'model_dir': model_dir,
            'lora_dir': adapter_dir,
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            }
        }
    elif backend in ["lightllm"]:
        data = {
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
                 # 'temperature': 0.1,
            },
        }
    elif backend in ["vllm", "vllm-packed"]:
        data = {
            'prompt': prompt,
            'max_tokens': output_len,
            'ignore_eos': True,
        }

    first_token_latency = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            # output = json.loads(output)
            # print(output)
            
            if '\"finished\": -1' not in output:
                break
            else:
                first_token_latency = None
                break
            # #     print(output)
            # #     print(json.loads(output))
            # break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    print(f"req_id {req_id} prompt_len {prompt_len} output_len {output_len} "
          f"request_latency {request_latency:.2f} s, first_token_latency {first_token_latency:.2f} s")
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency, first_token_latency))
    return (prompt_len, output_len, request_latency, first_token_latency)


async def benchmark(
    backend: str,
    server: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    debug=False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        await asyncio.sleep(start + req.req_time - time.time())
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.adapter_dir}")
        task = asyncio.create_task(send_request(backend, server,
                                                req.req_id, req.model_dir, req.adapter_dir, req.prompt,
                                                req.prompt_len, req.output_len, debug))
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency


def get_adapter_dirs(num_adapters, adapter_dirs, backend=None):
    ret = []
    num_iter = (num_adapters + len(adapter_dirs) - 1) // len(adapter_dirs)

    if backend == "vllm-packed":
        num_iter = max(1, num_adapters // len(adapter_dirs))

    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    return ret[:num_adapters]

def get_res_stats(per_req_latency, benchmark_time, backend, config_dict, warmup_time=0, warmup_num=0):
    # get throughput
    num_abort = len([i for i in per_req_latency if i[3] is None])
    per_req_latency = [i for i in per_req_latency if i[3] is not None]
    throughput = len(per_req_latency) / benchmark_time
    # print(per_req_latency)
    # if backend == "slora":
    #     peak_mem = get_peak_mem(server)
    #     print(f"GPU peak memory (GB):", [[f"{x / GB:.2f}" for x in tpg] for tpg in peak_mem])
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Aborted Request: {num_abort}")
    print(f"Throughput: {throughput:.2f} requests/s")

    strip_throughput = (len(per_req_latency) - warmup_num * 2) / (benchmark_time - warmup_time * 2)
    print(f"Throughput strip: {strip_throughput:.2f} requests/s")

    # compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _ in per_req_latency])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency, _ in per_req_latency
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency, _ in per_req_latency
    ])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")

    # compute the first token latency
    first_token_latency = [latency for _, _, _, latency in per_req_latency]
    avg_first_token_latency = np.mean(first_token_latency)
    print(f"Average first token latency: {avg_first_token_latency:.2f} s")
    print(f"90 percentile first token latency: < {np.percentile(first_token_latency, 90):.2f} s")
    print(f"50 percentile first token latency: < {np.percentile(first_token_latency, 50):.2f} s")
    abort_satisfaction = [0]*num_abort
    satisfaction = [reward(latency) for _, _, _, latency in per_req_latency] + abort_satisfaction
    avg_satisfaction = np.mean(satisfaction)
    print(f"Average satisfaction: {avg_satisfaction:.2f}")
    print(f"90 percentile satisfaction: > {np.percentile(satisfaction, 10):.2f}")
    print(f"50 percentile satisfaction: > {np.percentile(satisfaction, 50):.2f}")

    attainment = [attainment_func(latency) for _, _, _, latency in per_req_latency] + abort_satisfaction
    avg_attainment = np.mean(attainment)
    print(f"Average attainment: {avg_attainment:.2f}")

    # dump results
    if backend == "slora":
        # TODO
        # single_gpu_peak_mem = peak_mem
        single_gpu_peak_mem = 0
    else:
        single_gpu_peak_mem = 0

    result = {"total_time": benchmark_time, "gpu_peak_mem": single_gpu_peak_mem, "num_abort": num_abort,
              "throughput": throughput, "strip_throughput": strip_throughput,
              "avg_latency": avg_latency, "avg_per_token_latency": avg_per_token_latency,
              "avg_per_output_token_latency": avg_per_output_token_latency,
              "avg_first_token_latency": avg_first_token_latency,
              "avg_satisfaction": avg_satisfaction,
              "avg_attainment": avg_attainment}
    res = {"config": config_dict, "result": result}
    
    return res


def _override_config(config, args):
    num_adapters, alpha, req_rate, cv, duration, input_range, output_range = config
    if args.num_adapters_override is not None:
        num_adapters = args.num_adapters_override
    if args.alpha_override is not None:
        alpha = args.alpha_override
    if args.req_rate_override is not None:
        req_rate = args.req_rate_override
    if args.cv_override is not None:
        cv = args.cv_override
    if args.duration_override is not None:
        duration = args.duration_override
    if args.fixed_input_len is not None:
        input_range = (args.fixed_input_len, args.fixed_input_len + 1)
    if args.fixed_output_len is not None:
        output_range = (args.fixed_output_len, args.fixed_output_len + 1)
    return num_adapters, alpha, req_rate, cv, duration, input_range, output_range


def _resolve_prompt_lengths(args):
    if args.shared_prefix_ratio is None:
        return None, None
    assert args.fixed_input_len is not None, "--fixed-input-len is required with --shared-prefix-ratio"
    shared_prefix_len = int(round(args.fixed_input_len * args.shared_prefix_ratio))
    query_len = args.fixed_input_len - shared_prefix_len
    assert query_len >= 0, "shared prefix length cannot exceed logical input length"
    return shared_prefix_len, query_len


def _build_config_dict(base_config, args, logical_input_len, request_input_len, shared_prefix_len):
    config_dict = to_dict(base_config)
    config_dict.update({
        "mode": args.mode,
        "model_setting": args.model_setting,
        "base_model_override": args.base_model_override,
        "adapter_template_override": args.adapter_template_override,
        "num_adapters_effective": args.num_adapters_override if args.num_adapters_override is not None else config_dict["num_adapters"],
        "alpha_effective": args.alpha_override if args.alpha_override is not None else config_dict["alpha"],
        "req_rate_effective": args.req_rate_override if args.req_rate_override is not None else config_dict["req_rate"],
        "cv_effective": args.cv_override if args.cv_override is not None else config_dict["cv"],
        "duration_effective": args.duration_override if args.duration_override is not None else config_dict["duration"],
        "fixed_input_len": args.fixed_input_len,
        "fixed_output_len": args.fixed_output_len,
        "logical_input_len": logical_input_len,
        "request_input_len": request_input_len,
        "shared_prefix_ratio": args.shared_prefix_ratio,
        "shared_prefix_len": shared_prefix_len,
        "query_only_prefix_workload": args.query_only_prefix_workload,
        "exact_prompt_tokens": args.exact_prompt_tokens,
        "adapter_distribution": args.adapter_distribution,
    })
    return config_dict


def _has_custom_workload(args):
    return any([
        args.num_adapters_override is not None,
        args.base_model_override is not None,
        args.adapter_template_override is not None,
        args.alpha_override is not None,
        args.req_rate_override is not None,
        args.cv_override is not None,
        args.duration_override is not None,
        args.fixed_input_len is not None,
        args.fixed_output_len is not None,
        args.shared_prefix_ratio is not None,
        args.query_only_prefix_workload,
        args.exact_prompt_tokens,
        args.adapter_distribution != "power",
    ])


def _build_custom_config(args):
    input_len = args.fixed_input_len if args.fixed_input_len is not None else 8
    output_len = args.fixed_output_len if args.fixed_output_len is not None else 8
    return BenchmarkConfig(
        num_adapters=args.num_adapters_override if args.num_adapters_override is not None else 1,
        alpha=args.alpha_override if args.alpha_override is not None else 1.0,
        req_rate=args.req_rate_override if args.req_rate_override is not None else 1.0,
        cv=args.cv_override if args.cv_override is not None else 1.0,
        duration=args.duration_override if args.duration_override is not None else 60,
        input_range=[input_len, input_len + 1],
        output_range=[output_len, output_len + 1],
    )


def run_exp(model_setting, backend, server, config, output, mode, args):
    if mode == "real":
        print("*** num_adapters, cv and alpha are not used in real mode ***")
    print([(k, v) for k, v in zip(BenchmarkConfig._fields, config)])

    num_adapters, alpha, req_rate, cv, duration, input_range, output_range = _override_config(config, args)
    logical_input_len = args.fixed_input_len
    shared_prefix_len, query_len = _resolve_prompt_lengths(args)
    request_input_len = args.fixed_input_len
    if args.query_only_prefix_workload:
        assert args.shared_prefix_ratio is not None, "--shared-prefix-ratio is required with --query-only-prefix-workload"
        logical_input_len = args.fixed_input_len
        request_input_len = query_len
    elif args.shared_prefix_ratio is not None and logical_input_len is None:
        logical_input_len = None
    config_dict = _build_config_dict(config, args, logical_input_len, request_input_len, shared_prefix_len)

    if args.shared_prefix_ratio is not None:
        print(
            f"shared_prefix_ratio={args.shared_prefix_ratio:.4f} "
            f"shared_prefix_len={shared_prefix_len} request_input_len={request_input_len} "
            f"logical_input_len={logical_input_len}"
        )
    # assert duration >= 30
    if mode == "synthetic":
        base_model = args.base_model_override or BASE_MODEL[model_setting]
        adapter_dirs = [args.adapter_template_override] if args.adapter_template_override else LORA_DIR[model_setting]
        adapter_dirs = get_adapter_dirs(num_adapters, adapter_dirs)
        adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(num_adapters)]
        if num_adapters == 0:
            adapter_dirs = [(base_model, None)]
            num_adapters = 1
        requests = generate_requests(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, adapter_dirs,
                                 seed=args.seed,
                                 fixed_input_len=request_input_len,
                                 fixed_output_len=args.fixed_output_len,
                                 logical_input_len=logical_input_len,
                                 exact_prompt_tokens=args.exact_prompt_tokens,
                                 tokenizer_name=base_model,
                                 adapter_distribution=args.adapter_distribution)
        avg_prompt_len = np.mean([req.prompt_len for req in requests])
        avg_output_len = np.mean([req.output_len for req in requests])
        avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
        print(
            "avg_len:", avg_len,
            "avg_prompt_len:", avg_prompt_len,
            "avg_output_len:", avg_output_len,
            "request_input_len:", request_input_len,
        )
    else:
        # first generate your data using real_trace/clean_chat_data.py
        base_model = args.base_model_override or BASE_MODEL[model_setting]
        adapter_dirs = [args.adapter_template_override] if args.adapter_template_override else LORA_DIR[model_setting]
        adapter_dirs, requests = get_real_requests(trace_file="../../../real_trace/clean_chat_conv_20231019.json",
                                                   req_rate=req_rate, duration=duration,
                                                   base_model=base_model, adapter_dirs=adapter_dirs,
                                                   input_range=input_range, output_range=output_range,
                                                   seed=args.seed)
        # print(requests)
        avg_prompt_len = np.mean([req.prompt_len for req in requests])
        avg_output_len = np.mean([req.output_len for req in requests])
        avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
        print("num_adapters", len(adapter_dirs), "num_requests", len(requests), "avg_len:", avg_len, "avg_prompt_len:", avg_prompt_len, "avg_output_len:", avg_output_len)
        
    if args.debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)

    if backend == "vllm-packed":
        for i in range(len(adapter_dirs)):
            vllm_packed_adapter_dir_to_url_map[adapter_dirs[i][1]] = f"http://127.0.0.1:{8000 + i}"

    # benchmark
    benchmark_start_time = time.time()
    per_req_latency = asyncio.run(benchmark(backend, server, requests, args.debug))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    warmup_time = 10
    warmup_num = int(req_rate * warmup_time)
    res = get_res_stats(per_req_latency, benchmark_time, backend,
                        config_dict=config_dict, warmup_time=warmup_time, warmup_num=warmup_num)

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="slora",
                        choices=["slora", "vllm", "lightllm", "vllm-packed"])
    parser.add_argument("--suite", type=str, default="default")

    parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-copy", action="store_true")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])

    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-adapters-override", type=int, default=None)
    parser.add_argument("--base-model-override", type=str, default=None)
    parser.add_argument("--adapter-template-override", type=str, default=None)
    parser.add_argument("--alpha-override", type=float, default=None)
    parser.add_argument("--req-rate-override", type=float, default=None)
    parser.add_argument("--cv-override", type=float, default=None)
    parser.add_argument("--duration-override", type=float, default=None)
    parser.add_argument("--fixed-input-len", type=int, default=None)
    parser.add_argument("--fixed-output-len", type=int, default=None)
    parser.add_argument("--shared-prefix-ratio", type=float, default=None)
    parser.add_argument("--query-only-prefix-workload", action="store_true")
    parser.add_argument("--exact-prompt-tokens", action="store_true")
    parser.add_argument("--adapter-distribution", type=str, default="power", choices=["power", "zipf"])
    args = parser.parse_args()

    assert not args.no_lora_copy or args.no_lora_compute
    assert not (args.debug and args.breakdown)
    assert args.shared_prefix_ratio is None or 0.0 <= args.shared_prefix_ratio <= 1.0
    assert not args.query_only_prefix_workload or args.fixed_input_len is not None

    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.mode}_" + args.backend + ".jsonl"
    if args.no_lora_swap and args.no_lora_compute and args.no_lora_copy:
        args.output = "no_lora_compute_swap_copy_results.jsonl"
    elif args.no_lora_swap and args.no_lora_compute:
        args.output = "no_lora_compute_swap_results.jsonl"
    elif args.no_lora_swap:
        args.output = "no_lora_swap_results.jsonl"
    elif args.no_lora_compute:
        args.output = "no_lora_compute_results.jsonl"
    if args.debug or args.breakdown:
        args.output = "debug_" + args.output

    custom_workload = _has_custom_workload(args)
    if custom_workload:
        suites = [_build_custom_config(args)]
    else:
        suites = get_all_suites(mode=args.mode, debug=args.debug, suite=args.suite, breakdown=args.breakdown)

    if not args.append:
        os.system(f"rm {args.output}")
        results = []
    else:
        with open(args.output, "r") as f:
            lines = f.readlines()
        results = [json.loads(line)["config"] for line in lines]

    for config in tqdm(suites, desc="suites"):
        if custom_workload or to_dict(config) not in results:
            stats = run_exp(args.model_setting, args.backend, args.server, config,
                            args.output, args.mode, args)
