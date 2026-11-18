"""Paired router-level GPU replay: wait one decode versus shrink and admit now."""

import argparse
import asyncio
from bisect import bisect_right
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import time

from benchmarks.lmsys_maf_trace import load_trace


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False))
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summarize(events, point, seconds):
    previous = dict(point["last_before"])
    gaps, count, ttft = [], 0, None
    for elapsed, ids in events:
        if elapsed > seconds:
            break
        count += len(ids)
        for rid in ids:
            if rid in previous:
                gaps.append(elapsed - previous[rid])
                previous[rid] = elapsed
            if rid == point["action"]["head"] and ttft is None:
                ttft = point["head_wait_s"] + elapsed
    return dict(window_s=seconds, tokens=count, tokens_s=count / seconds,
                head_ttft_s=ttft, head_censored=ttft is None,
                running_itl_count=len(gaps),
                running_itl_p95_s=sorted(gaps)[max(0, (95 * len(gaps) + 99) // 100 - 1)] if gaps else None,
                running_without_token=sum(t < 0 for t in previous.values()))


async def engine(config, args, count):
    from benchmarks.fcfs_fork_policy import ForkQueue
    from slora.server.input_params import InputParams
    from slora.server.router.manager import RouterManager
    from slora.common.prefix_modes import LAYER_ADAPTIVE
    from slora.utils.infer_utils import set_random_seed
    import numpy as np

    class BenchRouter(RouterManager):
        def _init_transport(self, *_):
            # In-process trace source and token sink. Do not measure HTTP/detokenization.
            self.last_token = {}
            self.events = None

        def _send_to_detokenization_proc(self, batch, req_ans):
            now = time.perf_counter()
            self.last_token.update(dict.fromkeys(req_ans, now))
            if self.events is not None:
                self.events.append((now - self.origin, list(req_ans)))

        def _reject_online_request(self, req, reason):
            raise RuntimeError("trace request rejected: %s: %s" % (req.request_id, reason))

    if config["tokenizer"] != "huggyllama/llama-13b":
        raise ValueError("this first experiment requires a Llama-13B trace")
    dirs = ["dummy-lora-13b-rank-16-%s" % i for i in range(config["num_adapters"])]
    p = InputParams(max_req_total_len=config["max_context_len"], max_total_token_num=args.pool,
        pool_size_lora=0, batch_max_tokens=args.pool, running_max_req_size=min(args.pool, count),
        swap=True, prefetch=False, prefetch_size=0, scheduler="online", profile=False,
        batch_num_adapters=None, enable_abort=False, dummy=True, no_lora_compute=False,
        no_lora_swap=False, no_kernel=False, no_mem_pool=False, bmm=False, no_lora=False,
        fair_weights=None, enable_lowra=True, lowra_shared_prefix_len=config["prefix_len"],
        lowra_prefix_mode=LAYER_ADAPTIVE, lowra_reconstruction_kernel="bounded",
        lowra_reconstruction_workers=1296, nccl_port=args.nccl_port)
    set_random_seed(42)
    np.random.seed(42)
    router = BenchRouter(config["tokenizer"], dirs, "HF", 1, 2, None, None, [None], p,
                         log_stats=False)
    router.req_queue = ForkQueue(p.max_total_token_num, p.batch_max_tokens, p.running_max_req_size,
        prefix_len=p.lowra_shared_prefix_len, prefix_mode=LAYER_ADAPTIVE, layer_num=p.lowra_layer_num)
    await router.wait_to_model_ready()
    return router


class Playback:
    def __init__(self, router, trace):
        self.router = router
        self.rows = trace["requests"]
        self.arrivals = [r["req_time"] for r in self.rows]
        self.cursor = 0

    def inject(self, end):
        from slora.server.io_struct import Req
        from slora.server.sampling_params import SamplingParams
        assert self.cursor <= end <= len(self.rows)
        for row in self.rows[self.cursor:end]:
            params = SamplingParams(ignore_eos=True, max_new_tokens=row["output_len"], stop_sequences=[])
            req = Req(self.router.adapter_dirs[row["adapter_index"]], row["req_id"], row["input_ids"],
                      params, {"shared_prefix_len": self.router.req_queue.prefix_len})
            self.router.req_queue.append(req)
        self.cursor = end

    async def preview(self):
        r = self.router
        r.req_queue.blocked_head = False
        if not (r.running_batch and r.has_wait_tokens >= 1 and r.req_queue.waiting_req_list):
            return None
        await r._sync_lowra_prefix_cache_state()
        r.req_queue.choose(r.running_batch.reqs, r.lora_ranks)
        return r.req_queue.opportunity

    def signature(self):
        r = self.router
        state = r.req_queue.prefix_cache_state
        worker = r.model_rpcs[0].model
        def requests(reqs):
            return [(q.request_id, q.adapter_dir, q.input_len, q.output_ids, q.preemptions) for q in reqs]
        def gpu_hash(tensor):
            return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
        private = worker.cache[r.running_batch.batch_id]
        return digest(dict(cursor=self.cursor, step=r.req_queue.decode_steps, interval=r.has_wait_tokens,
            running=requests(r.running_batch.reqs), queue=requests(r.req_queue.waiting_req_list),
            cache={k: state[k] for k in ("prefix_sizes", "retained_layers", "ready_layers", "ref_counts", "last_used")},
            resident=list(worker.infer_adapter.adapter_dirs),
            private_pages=[gpu_hash(private._free_indices_for_req(i)) for i in range(len(r.running_batch.reqs))],
            pages={k: gpu_hash(e.pool_indices) for k, e in worker.lowra_prefix_cache.entries.items()},
            allocator=gpu_hash(worker.model.mem_manager.mem_state)))


async def discover(pb, args):
    r = pb.router
    cursors, points, recent = [], [], deque(maxlen=10)
    arrivals = dict(zip((x["req_id"] for x in pb.rows), pb.arrivals))
    start = time.perf_counter()
    last_step = -args.spacing
    eligible = blocked = 0
    while pb.cursor < len(pb.rows) or r.running_batch or r.req_queue.waiting_req_list:
        if not (r.running_batch or r.req_queue.waiting_req_list) and pb.cursor < len(pb.rows):
            await asyncio.sleep(max(0, start + pb.arrivals[pb.cursor] - time.perf_counter()))
        now = time.perf_counter()
        offset = now - start
        pb.inject(bisect_right(pb.arrivals, offset))
        step = r.req_queue.decode_steps
        action = await pb.preview()
        blocked += int(r.req_queue.blocked_head)
        if action:
            eligible += 1
        if (action and len(recent) == recent.maxlen and step - last_step >= args.spacing
                and action["head"] not in r.last_token):
            last_step = step
            points.append(dict(index=len(cursors), cursor=pb.cursor, action=action,
                signature=pb.signature(), trace_time=offset, decode_step=step,
                unit_s=median(recent), head_wait_s=offset - arrivals[action["head"]],
                last_before={q.request_id: r.last_token[q.request_id] - now for q in r.running_batch.reqs}))
        cursors.append(pb.cursor)
        t0 = time.perf_counter()
        await r._step_online()
        if r.req_queue.decode_steps > step:
            recent.append(time.perf_counter() - t0)
    sampled = sorted(random.Random(42).sample(points, min(args.points, len(points))), key=lambda p:p["index"])
    print("Blocked boundaries: %d; shrink-feasible: %d; spaced candidates: %d; paired points: %d" %
          (blocked, eligible, len(points), len(sampled)), flush=True)
    return dict(cursors=cursors, points=sampled, blocked=blocked, eligible=eligible, spaced=len(points))


async def branch(pb, journal, index, arm):
    p = journal["points"][index]
    r = pb.router
    for end in journal["cursors"][:p["index"]]:
        pb.inject(end)
        await r._step_online()
    pb.inject(p["cursor"])
    if await pb.preview() != p["action"] or pb.signature() != p["signature"]:
        raise RuntimeError("fork state differs from discovery at point %d; refusing unpaired data" % index)
    r.events = []
    r.req_queue.arm = arm
    r.origin = time.perf_counter()
    await r._step_online()
    if (r.req_queue.opportunity != p["action"] or
            (arm == "wait" and r.req_queue.decode_steps != p["decode_step"] + 1) or
            (arm == "shrink" and (r.req_queue.decode_steps != p["decode_step"] or
                                  not r.events or r.events[0][1] != p["action"]["extra_ids"]))):
        raise RuntimeError("requested intervention was not executed")
    r.req_queue.arm = "shrink"
    end = r.origin + 20 * p["unit_s"]
    while time.perf_counter() < end:
        offset = time.perf_counter() - r.origin + p["trace_time"]
        pb.inject(bisect_right(pb.arrivals, offset))
        if not (r.running_batch or r.req_queue.waiting_req_list):
            if pb.cursor == len(pb.rows):
                break
            await asyncio.sleep(max(0, min(pb.arrivals[pb.cursor] - offset, end - time.perf_counter())))
        else:
            await r._step_online()
    return dict(point=index, arm=arm, signature=p["signature"], events=r.events,
                windows={str(h):summarize(r.events, p, h * p["unit_s"]) for h in (5, 10, 20)})


async def worker(args):
    trace, sha = load_trace(args.trace)
    router = await engine(trace["config"], args, len(trace["requests"]))
    pb = Playback(router, trace)
    if args.worker == "discover":
        result = await discover(pb, args)
        result.update(trace_sha256=sha, pool=args.pool, model_seed=42, config=trace["config"],
                      sample_limit=args.points, spacing=args.spacing)
        write(args.out / "journal.json", result)
    else:
        journal = read(args.out / "journal.json")
        if (sha, args.pool) != (journal["trace_sha256"], journal["pool"]):
            raise ValueError("workload or memory budget changed since discovery")
        result = await branch(pb, journal, args.point, args.worker)
        write(args.out / ("%03d-%s.json" % (args.point, args.worker)), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--points", type=int, default=30)
    parser.add_argument("--spacing", type=int, default=10, help="minimum spacing in decode steps")
    parser.add_argument("--pool", type=int, default=65000)
    parser.add_argument("--nccl-port", type=int, default=34482)
    parser.add_argument("--resume", action="store_true", help="reuse completed pairs from this experiment")
    parser.add_argument("--worker", choices=("discover", "wait", "shrink"), help=argparse.SUPPRESS)
    parser.add_argument("--point", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.points, args.spacing, args.pool) < 1:
        parser.error("points, spacing and pool must be positive")
    if args.worker:
        asyncio.run(worker(args))
        return
    # Isolate the allocator, RNGs and model state between counterfactuals.
    args.out.mkdir(parents=True, exist_ok=args.resume)
    def run(*extra):
        subprocess.run([sys.executable, "-m", "benchmarks.fcfs_fork_exp", *sys.argv[1:], *extra],
                       env=dict(os.environ, PYTHONHASHSEED="0"), check=True)
    journal_path = args.out / "journal.json"
    if not journal_path.exists():
        run("--worker", "discover")
    journal = read(journal_path)
    _, sha = load_trace(args.trace)
    if (sha, args.pool, args.points, args.spacing) != (journal["trace_sha256"], journal["pool"],
            journal["sample_limit"], journal["spacing"]):
        raise ValueError("resume requires the original workload, pool and sampling configuration")
    points = journal["points"]
    for i in range(len(points)):
        for arm in (("wait", "shrink") if i % 2 == 0 else ("shrink", "wait")):
            if args.resume and (args.out / ("%03d-%s.json" % (i, arm))).exists():
                continue
            print("Measuring pair %d/%d: %s" % (i + 1, len(points), arm), flush=True)
            run("--worker", arm, "--point", str(i))
    from benchmarks.fcfs_fork_report import report
    report(args.out)


if __name__ == "__main__":
    main()
