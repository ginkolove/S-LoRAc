"""Fixed-shape real serving regression: serial, old fused prefetch, new auto prefetch."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.lowra_layer_profile import build_server
from slora.common.lowra_layer_runtime import reconstruct_prefix_layer_bounded
from slora.common.prefix_modes import LAYER_SERIAL
from slora.server.router.model_infer.post_process import sample
from slora.utils.model_utils import get_model_config
from overlap_analysis import summarize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-len", type=int, default=4608)
    parser.add_argument("--query-len", type=int, default=480)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unique-loras", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bounded-workers", type=int, default=648)
    parser.add_argument("--serial-bounded-workers", type=int)
    parser.add_argument("--modes", nargs="+", choices=["serial_split", "serial_bounded", "prefetch_fused",
                                                       "prefetch_split", "prefetch_bounded", "prefetch_auto"],
                        default=["serial_split", "serial_bounded", "prefetch_fused", "prefetch_split", "prefetch_bounded"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.prefix_len, args.query_len, args.output_len, args.batch_size, args.unique_loras, args.iterations) <= 0:
        parser.error("dimensions and iterations must be positive")
    if args.unique_loras > args.batch_size or args.warmup < 0:
        parser.error("invalid unique LoRA count or warmup")
    return args


@torch.no_grad()
def main(args):
    torch.cuda.set_device(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    if output.exists():
        raise FileExistsError(output)
    build_args = SimpleNamespace(**vars(args), dummy=True, smoke=False, model_dir="huggyllama/llama-7b",
                                 adapter_base="dummy-lora-7b-rank-16", mode="layer_prefetch",
                                 num_adapters=args.unique_loras, rank=16, nonzero_prefix=True)
    config = get_model_config(build_args.model_dir, dummy=True)
    # Explicit performance-only context extension; the original RoPE stays unchanged.
    config["max_sequence_length"] = max(config.get("max_sequence_length", 2048),
                                        args.prefix_len + args.query_len + args.output_len)
    with patch("slora.utils.model_utils.get_model_config", return_value=config):
        server, _, requests, _ = build_server(build_args)
    runtime = server.lowra_runtime
    runtime.profiler = None
    rows = {name: [] for name in args.modes}
    reference = None
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.output_len + 1)]

    def run(name, collect_logits=False):
        runtime.mode = LAYER_SERIAL if name.startswith("serial") else "layer_prefetch"
        runtime.prefetch_reconstruction = name[9:] if name.startswith("prefetch_") else "auto"
        runtime.prefetch_workers = args.bounded_workers
        collected = []

        def capture(logits, batch):
            collected.append(logits.detach().clone())
            return sample(logits, batch)

        fresh = [dict(req, input_id=list(req["input_id"]), sampling_param=dict(req["sampling_param"])) for req in requests]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        server.exposed_add_batch("comparison", fresh, "fp16")
        starts[0].record()
        def bounded_serial(*inputs):
            workers = args.serial_bounded_workers or args.bounded_workers
            return reconstruct_prefix_layer_bounded(*inputs, workers=workers)

        with patch("slora.server.router.model_infer.model_rpc.sample", new=capture) if collect_logits else nullcontext(), \
                patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer", new=bounded_serial) if name == "serial_bounded" else nullcontext():
            server.exposed_prefill_batch("comparison")
            starts[1].record()
            for step in range(1, args.output_len):
                server.exposed_decode_batch("comparison")
                starts[step + 1].record()
        starts[-1].synchronize()
        wall = (time.perf_counter() - begin) * 1000
        row = dict(wall_ms=wall, prefill_ms=starts[0].elapsed_time(starts[1]),
                   decode_ms=starts[1].elapsed_time(starts[-1]),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   step_ms=[starts[i].elapsed_time(starts[i + 1]) for i in range(1, args.output_len)])
        server.exposed_remove_batch("comparison")
        return row, collected

    for name in rows:
        _, actual = run(name, collect_logits=True)
        if reference is None:
            reference = actual
        else:
            if len(actual) != len(reference):
                raise AssertionError("mismatched number of model outputs")
            for logits, expected in zip(actual, reference):
                torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-3)
        del actual
        for _ in range(args.warmup):
            run(name)
        print("Validated/warmed %s" % name, flush=True)
    del reference
    gc.collect()
    rng = random.Random(args.seed)
    for iteration in range(args.iterations):
        order = list(rows)
        rng.shuffle(order)
        for name in order:
            row, _ = run(name)
            rows[name].append(row)
            print(json.dumps(dict(mode=name, iteration=iteration, **row)), flush=True)
    summary = {name: {field: summarize([row[field] for row in samples])
                      for field in ("wall_ms", "prefill_ms", "decode_ms", "peak_allocated_bytes")}
               for name, samples in rows.items()}
    source = Path("slora/common/lowra_layer_runtime.py")
    report = dict(config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  scope="32-layer Llama-7B, dummy weights, nonzero Base/MiniKV, same fixed batch and logits. "
                        "Real prefill/decode/RPC/allocator paths; no HTTP, admission, or request trace.",
                  correctness_passed=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                      "slora/common/lowra_layer_runtime.py", "slora/common/lowra_reconstruct_bounded.py", __file__)},
                  summary=summary, samples=rows)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main(parse_args())
