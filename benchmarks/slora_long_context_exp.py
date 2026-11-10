"""Check untruncated long-input prefill and decode through S-LoRA's HTTP server.

Token IDs bypass tokenizer round trips. Run against a server with LowRA disabled
to prefill the entire input, rather than injecting a cached shared prefix.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import uuid

import aiohttp
from tqdm import tqdm


def build_payload(input_len, output_len, token_id, lora_dir, request_id):
    return {
        "req_id": request_id,
        "lora_dir": lora_dir,
        "inputs": [token_id] * input_len,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": output_len,
            "return_details": False,
        },
    }


def decode_event(data, input_len, output_len, previous_count):
    event = json.loads("\n".join(data))
    if event.get("finished") == -1:
        raise ValueError("server aborted the request")
    if not isinstance(event.get("finished"), bool):
        raise ValueError("missing or invalid finished flag")
    prompt_tokens = event.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise ValueError("server did not report prompt_tokens; restart the updated server")
    if prompt_tokens != input_len:
        raise ValueError("server input length %s != requested %s; input was changed" %
                         (prompt_tokens, input_len))
    token = event.get("token")
    if not isinstance(token, dict):
        raise ValueError("missing output token")
    token_id = token.get("id")
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError("invalid output token id")
    if token.get("logprob") is not None and not math.isfinite(float(token["logprob"])):
        raise ValueError("nonfinite output logprob")
    count = event.get("count_output_tokens")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("missing or invalid cumulative output token count")
    if not previous_count < count <= output_len:
        raise ValueError("nonmonotonic or excessive output token count: %s" % count)
    if event["finished"] and count != output_len:
        raise ValueError("generation ended early: %s != %s tokens" % (count, output_len))
    return prompt_tokens, count, event["finished"]


async def send_request(session, args, request_id):
    payload = build_payload(args.input_len, args.output_len, args.prompt_token_id,
                            args.lora_dir, request_id)
    result = dict(request_id=request_id, sent_input_tokens=len(payload["inputs"]),
                  server_prompt_tokens=None, expected_output_tokens=args.output_len,
                  output_tokens=0, stream_events=0, finished=False, success=False,
                  http_status=None, ttft_s=None, latency_s=None, error=None)
    started = time.monotonic()
    completed = None
    try:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        async with session.post(args.server.rstrip("/") + "/generate_stream",
                                json=payload, timeout=timeout) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raise ValueError("HTTP %s: %s" %
                                 (response.status, (await response.text())[:1000]))
            data = []
            while True:
                raw = await response.content.readline()
                if not raw:
                    if data:
                        raise ValueError("truncated SSE frame")
                    break
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    if result["finished"]:
                        raise ValueError("received token event after completion")
                    prompt_tokens, count, finished = decode_event(
                        data, args.input_len, args.output_len, result["output_tokens"])
                    data.clear()
                    now = time.monotonic()
                    if result["ttft_s"] is None:
                        result["ttft_s"] = now - started
                    result["server_prompt_tokens"] = prompt_tokens
                    result["output_tokens"] = count
                    result["stream_events"] += 1
                    result["finished"] = finished
                    if finished:
                        completed = now
            if not result["finished"]:
                raise ValueError("stream ended without completing generation")
            result["success"] = True
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
        completed = time.monotonic()
    result["latency_s"] = (completed if completed is not None else time.monotonic()) - started
    return result


async def run(args):
    results = []
    started = time.monotonic()
    async with aiohttp.ClientSession() as session:
        with tqdm(total=args.num_requests, desc="Long-context requests", unit="req") as progress:
            for _ in range(args.num_requests):
                result = await send_request(session, args, "long-context-" + uuid.uuid4().hex)
                results.append(result)
                progress.update(1)
                if not result["success"]:
                    break
    successful = [result for result in results if result["success"]]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(server=args.server, lora_dir=args.lora_dir,
                       input_len=args.input_len, output_len=args.output_len,
                       logical_total_len=args.input_len + args.output_len,
                       prompt_token_id=args.prompt_token_id, num_requests=args.num_requests,
                       timeout_s=args.timeout, request_order="sequential"),
        "result": dict(all_passed=len(successful) == args.num_requests,
                       attempted_requests=len(results), completed_requests=len(successful),
                       failed_requests=len(results) - len(successful),
                       benchmark_time_s=time.monotonic() - started,
                       latency_mean_s=(sum(r["latency_s"] for r in successful) / len(successful)
                                       if successful else None),
                       ttft_mean_s=(sum(r["ttft_s"] for r in successful) / len(successful)
                                    if successful else None)),
        "requests": results,
        "measurement_notes": [
            "TTFT and latency are client-observed end-to-end times, not GPU kernel times.",
            "Completion uses cumulative token counts, not the number of SSE events.",
            "This validates execution beyond the trained context, not generation quality.",
            "No GPU memory or KV leak measurement is inferred from HTTP success.",
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8200")
    parser.add_argument("--lora-dir", default="dummy-lora-7b-rank-16-0")
    parser.add_argument("--input-len", type=int, default=10000)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--prompt-token-id", type=int, default=10,
                        help="Repeated input token ID; must be in the model vocabulary.")
    parser.add_argument("--num-requests", type=int, default=1,
                        help="Number of sequential requests; stops on the first failure.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/results/slora_long_context_10000.json"))
    args = parser.parse_args(argv)
    for name in ("input_len", "output_len", "num_requests"):
        if getattr(args, name) <= 0:
            parser.error(name.replace("_", "-") + " must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.prompt_token_id < 0:
        parser.error("prompt-token-id must be nonnegative")
    return args


def main():
    args = parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["result"], indent=2, allow_nan=False))
    for result in report["requests"]:
        if result["error"]:
            print(result["error"], file=sys.stderr)
    print("Results: %s" % args.output.resolve())
    return 0 if report["result"]["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
