"""Profile the serving reconstruction path without loading a base model.

Each timed iteration visits independent source tensors for all model layers,
overwriting one Unified Paging workspace. No attention, FFN or H2D is timed.
"""

import argparse
import csv
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

import slora.common.lowra_layer_runtime as runtime
from slora.common.lowra_static import LowRAStaticPrefixStore
from slora.common.mem_allocator import MemoryAllocator
from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
from slora.mprophet.model_config import get_config_json
from slora.server.router.model_infer.infer_adapter import InferAdapter


SOURCE_FILES = (
    "try/profile_minikv_rank_13b.py", "slora/common/lowra_layer_runtime.py",
    "slora/common/lowra_reconstruct_bounded.py", "slora/common/lowra_key_rope.py",
    "slora/common/lowra_static.py", "slora/common/mem_allocator.py",
    "slora/models/peft/layer_weights/lora_layer_weight.py",
    "slora/server/router/model_infer/infer_adapter.py",
)


@dataclass(frozen=True)
class Case:
    name: str
    ranks: tuple
    background_rank: int
    metadata_width: int = 0
    diagnostic_only: bool = False


def make_cases(ranks=(8, 16, 32, 64), counts=(1, 2, 4, 8, 16)):
    cases = [Case("r%d_u%d" % (r, u), (r,) * u, r) for r in ranks for u in counts]
    cases += [
        Case("mix_7r8_1r64", (8,) * 7 + (64,), 16),
        Case("mix_4r8_4r64", (8,) * 4 + (64,) * 4, 16),
        Case("mix_7r16_1r64", (16,) * 7 + (64,), 16),
        Case("mix_8_16_32_64", (8, 16, 32, 64) * 2, 16),
        Case("mix_4r16_4r32", (16,) * 4 + (32,) * 4, 16),
        Case("mix_5r16_3r32", (16,) * 5 + (32,) * 3, 16),
        Case("mix_6r16_1r32_1r64", (16,) * 6 + (32, 64), 16),
        Case("mix_6r16_2r64", (16,) * 6 + (64,) * 2, 16),
    ]
    if not any(c.name == "r8_u8" for c in cases):
        cases.append(Case("r8_u8", (8,) * 8, 8))
    cases.append(Case("r8_u8_padded64", (8,) * 8, 8, 64, True))
    return cases


def source_layout(case, pool_adapters, seed):
    if len(case.ranks) > pool_adapters:
        raise ValueError("selected adapters exceed the MiniKV source pool")
    order = list(range(pool_adapters))
    random.Random(seed).shuffle(order)
    selected = order[:len(case.ranks)]
    ranks = [case.background_rank] * pool_adapters
    for index, rank in zip(selected, case.ranks):
        ranks[index] = rank
    return ranks, selected


def pool_slots(prefix, layers, ranks):
    # Full-layer slots for active A/B weights, one layer's output cells, guards.
    return 4 * sum(ranks) + (len(ranks) * prefix + layers - 1) // layers + 128


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(values):
    return dict(mean_ms=statistics.mean(values), p50_ms=percentile(values, 0.5),
                p95_ms=percentile(values, 0.95), min_ms=min(values), max_ms=max(values),
                stdev_ms=statistics.pstdev(values), samples=len(values))


def widen_metadata(metadata, width):
    old_width = metadata.k_b_locs.shape[1]
    if width < old_width:
        raise ValueError("metadata width cannot truncate a real adapter rank")
    if width == old_width:
        return metadata
    def widen(tensor):
        result = tensor.new_zeros((tensor.shape[0], width))
        result[:, :old_width] = tensor
        return result
    return replace(metadata, k_b_locs=widen(metadata.k_b_locs), v_b_locs=widen(metadata.v_b_locs))


def reference_chunk(base, mini, hf_b, scale, cos=None, sin=None):
    delta = (mini.float() @ hf_b.float().T) * scale
    if cos is not None:
        # The production K matmul writes FP16 delta before the in-place RoPE pass.
        first, second = delta.half().float().view_as(base).chunk(2, dim=-1)
        c, s = cos.float()[:, None], sin.float()[:, None]
        delta = torch.cat((first * c - second * s, first * s + second * c), dim=-1)
    return base.float().reshape(delta.shape) + delta


class Fixture:
    def __init__(self, args, config, case):
        self.args, self.config, self.case = args, config, case
        self.prefix, self.layers = args.prefix_len, config["num_hidden_layers"]
        self.hidden, self.heads = config["hidden_size"], config["num_attention_heads"]
        self.dim = self.hidden // self.heads
        ranks, selected = source_layout(case, args.pool_adapters, args.seed)
        self.directories = ["adapter_%03d" % i for i in selected]
        all_dirs = ["adapter_%03d" % i for i in range(args.pool_adapters)]
        rank_map = dict(zip(all_dirs, ranks))
        capacity = pool_slots(self.prefix, self.layers, case.ranks)
        estimate = (4 * self.layers * self.prefix * (self.hidden + sum(ranks))
                    + 4 * self.layers * capacity * self.hidden)
        free, _ = torch.cuda.mem_get_info()
        if estimate + 1024 ** 3 > free:
            raise RuntimeError("fixture needs about %.2f GiB plus 1 GiB headroom, only %.2f GiB free" %
                               (estimate / 1024 ** 3, free / 1024 ** 3))
        self.static = LowRAStaticPrefixStore(self.prefix, all_dirs, rank_map, self.layers,
                                             self.heads, self.heads, self.dim)
        base_rng = torch.Generator(device="cuda").manual_seed(args.seed)
        mini_rng = torch.Generator(device="cuda").manual_seed(args.seed + 1)
        for layer in range(self.layers):
            for tensor in (self.static.base_k[layer], self.static.base_v[layer]):
                tensor.normal_(std=0.1, generator=base_rng)
            for tensor in (self.static.mini_k[layer], self.static.mini_v[layer]):
                tensor.normal_(std=0.1, generator=mini_rng)
        positions = torch.arange(self.prefix, device="cuda", dtype=torch.float32)[:, None]
        inv = 10000.0 ** (-torch.arange(0, self.dim, 2, device="cuda").float() / self.dim)
        angles = positions * inv[None]
        self.static.bind_key_rope(angles.cos().half(), angles.sin().half())
        self.memory = MemoryAllocator(capacity, capacity, torch.float16, self.heads, self.dim,
                                      self.layers, dynamic_layer_workspace=True)
        held = self.memory.alloc(64)
        self.memory.free(held[::2])
        self.guards = held[1::2]
        self.original_b = {}
        adapters = []
        for directory, index, rank in zip(self.directories, selected, case.ranks):
            weights = []
            for layer in range(self.layers):
                rng = torch.Generator().manual_seed(args.seed + 1009 * index + 73 * layer + 10000)
                hf = {}
                for module in ("q", "k", "v", "o"):
                    stem = "base_model.model.model.layers.%d.self_attn.%s_proj" % (layer, module)
                    hf[stem + ".lora_A.weight"] = torch.randn(rank, self.hidden, generator=rng).half() * 0.1
                    b = torch.randn(self.hidden, rank, generator=rng).half() * 0.1
                    hf[stem + ".lora_B.weight"] = b
                    if module in ("k", "v"):
                        self.original_b[(directory, layer, module)] = b
                weight = LoraLayerWeight(layer, 0, 1, {"r": rank}, config)
                weight.load_hf_weights_cpu(hf)
                weights.append(weight)
            adapters.append(SimpleNamespace(lora_dir=directory, r=rank, scaling=0.5,
                                            network_config=config, layers=weights))
        self.infer = InferAdapter.init(self.memory, None)
        self.infer.load_adapters(adapters)
        # CPU weight-loading objects are no longer needed; original K/V B is the oracle.
        del adapters, weights, weight, hf
        self.locations = self.memory.alloc_layer_workspace(len(case.ranks) * self.prefix).view(len(case.ranks), -1)
        metadata = runtime.prepare_reconstruction_metadata(self.static, self.directories,
                                                           self.infer, self.locations)
        self.metadata = widen_metadata(metadata, case.metadata_width or max(case.ranks))
        for layer in range(self.layers):
            self.memory.key_buffer[layer][self.guards] = 0.625
            self.memory.value_buffer[layer][self.guards] = -0.75
        torch.cuda.synchronize()
        self.memory_info = dict(static_bytes=4 * self.layers * self.prefix * (self.hidden + sum(ranks)),
                                source_total_rank=sum(ranks), pool_slots=capacity,
                                pool_bytes=self.memory.get_memory_size(),
                                workspace_bytes=4 * len(case.ranks) * self.prefix * self.hidden,
                                workspace_full_layer_slots=self.memory.workspace_token_equiv,
                                selected_adapter_indices=selected,
                                allocated_bytes=torch.cuda.memory_allocated())

    def layer(self, layer):
        options = {}
        block_r = getattr(self.args, "block_r", 16)
        rank_loop = getattr(self.args, "rank_loop", "max")
        if block_r != 16 or rank_loop != "max":
            options.update(block_r=block_r, rank_loop=rank_loop)
        runtime.reconstruct_prefix_layer_bounded(
            self.static, self.infer, self.metadata, layer,
            self.memory.prefix_workspace_k, self.memory.prefix_workspace_v,
            workers=self.args.workers, **options)

    def sweep(self):
        for layer in range(self.layers):
            self.layer(layer)

    def check_guards(self):
        for layer in range(self.layers):
            for buffer, value in ((self.memory.key_buffer[layer], 0.625),
                                  (self.memory.value_buffer[layer], -0.75)):
                if not bool((buffer[self.guards] == value).all().item()):
                    raise AssertionError("reconstruction overwrote protected pool slots")

    def validate(self):
        layers = sorted({0, self.layers // 2, self.layers - 1})
        stats = dict(layers=layers, elements=0, max_abs_error=0.0, abs_error_sum=0.0,
                     atol=self.args.atol, rtol=self.args.rtol, passed=False)
        for layer in tqdm(layers, desc="Validate " + self.case.name, disable=self.args.no_progress):
            for output in (self.memory.prefix_workspace_k, self.memory.prefix_workspace_v):
                output[self.locations.reshape(-1)] = float("nan")
            self.layer(layer)
            for row, directory in enumerate(self.directories):
                start, end = self.static.get_lora_rank_range_by_dir(directory)
                for module, output, base, mini in (
                    ("k", self.memory.prefix_workspace_k, self.static.base_k[layer], self.static.mini_k[layer]),
                    ("v", self.memory.prefix_workspace_v, self.static.base_v[layer], self.static.mini_v[layer]),
                ):
                    original = self.original_b[(directory, layer, module)].cuda()
                    for offset in range(0, self.prefix, self.args.validation_chunk):
                        span = slice(offset, min(offset + self.args.validation_chunk, self.prefix))
                        cos = self.static.key_rope_cos[span] if module == "k" else None
                        sin = self.static.key_rope_sin[span] if module == "k" else None
                        expected = reference_chunk(base[span], mini[span, start:end], original, 0.5, cos, sin)
                        actual = output[self.locations[row, span]].float().view_as(expected)
                        torch.testing.assert_close(actual, expected, atol=self.args.atol, rtol=self.args.rtol)
                        error = (actual - expected).abs()
                        stats["elements"] += actual.numel()
                        stats["max_abs_error"] = max(stats["max_abs_error"], error.max().item())
                        stats["abs_error_sum"] += error.sum(dtype=torch.float64).item()
                    # Check that the destination workspace did not clobber B sources.
                    b_locs = self.metadata.k_b_locs if module == "k" else self.metadata.v_b_locs
                    packed = self.memory.value_buffer[layer][b_locs[row, :end - start]]
                    torch.testing.assert_close(packed.reshape(self.hidden, end - start), original, atol=0, rtol=0)
        self.check_guards()
        stats["mean_abs_error"] = stats.pop("abs_error_sum") / stats["elements"]
        stats["passed"] = True
        return stats

    def time_sweeps(self, description):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # Materialize lazy event resources before measurement.
        start.record()
        end.record()
        end.synchronize()
        gpu_ms, wall_ms = [], []
        for _ in tqdm(range(self.args.iterations), desc=description, unit="sweep", disable=self.args.no_progress):
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            self.sweep()
            end.record()
            end.synchronize()
            wall_ms.append((time.perf_counter() - wall_start) * 1000)
            gpu_ms.append(start.elapsed_time(end))
        return dict(cuda_ms=gpu_ms, wall_ms=wall_ms, cuda=summarize(gpu_ms), wall=summarize(wall_ms))

    def breakdown(self):
        records = []
        kernel, rope = runtime.lowra_reconstruct_bounded, runtime.rotate_delta_k_add_base
        current = [0, 0]
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(3 * self.layers * self.args.diagnostic_iterations)]
        for start, end in events:
            start.record()
            end.record()
        torch.cuda.synchronize()
        available = iter(events)

        def measured(function, operation, *args, **kwargs):
            start, end = next(available)
            start.record()
            result = function(*args, **kwargs)
            end.record()
            records.append((current[0], operation, start, end))
            return result

        def traced_kernel(*args, **kwargs):
            operation = "k_matmul_scale" if current[1] == 0 else "v_matmul_scale_add"
            current[1] += 1
            return measured(kernel, operation, *args, **kwargs)

        def traced_rope(*args, **kwargs):
            return measured(rope, "k_rope_add", *args, **kwargs)

        # Scoped instrumentation only; primary sweeps call the untouched serving wrapper.
        with patch.object(runtime, "lowra_reconstruct_bounded", traced_kernel), \
                patch.object(runtime, "rotate_delta_k_add_base", traced_rope):
            for _ in range(self.args.diagnostic_iterations):
                for layer in range(self.layers):
                    current[:] = [layer, 0]
                    self.layer(layer)
        torch.cuda.synchronize()
        rows = []
        for layer in range(self.layers):
            row = dict(layer=layer)
            for operation in ("k_matmul_scale", "v_matmul_scale_add", "k_rope_add"):
                samples = [s.elapsed_time(e) for index, op, s, e in records if index == layer and op == operation]
                row[operation] = summarize(samples)
            rows.append(row)
        return dict(instrumented_separate_run=True, layers=rows,
                    operation_mean_total_ms={op: sum(row[op]["mean_ms"] for row in rows)
                                             for op in ("k_matmul_scale", "v_matmul_scale_add", "k_rope_add")})

    def close(self):
        self.memory.free_layer_workspace()
        self.infer.offload_adapters([])
        self.memory.free(self.guards)
        if self.memory.can_use_mem_size != self.memory.tot_size:
            raise AssertionError("pool allocation leak")


def write_report(output, report):
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "result.json")
    fields = ("case", "rank_loop", "block_r", "diagnostic_only", "ranks", "unique_loras", "max_rank", "metadata_width",
              "rounds", "cuda_mean_ms", "cuda_p50_ms", "cuda_p95_ms", "wall_mean_ms",
              "mean_layer_ms", "round_mean_stdev_ms", "workspace_bytes", "max_abs_error")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in report["cases"].items():
            if not result["rounds"]:
                continue
            cuda = summarize([v for r in result["rounds"] for v in r["cuda_ms"]])
            wall = summarize([v for r in result["rounds"] for v in r["wall_ms"]])
            means = [r["cuda"]["mean_ms"] for r in result["rounds"]]
            arguments = report.get("arguments", {})
            writer.writerow(dict(case=name, rank_loop=arguments.get("rank_loop", "max"),
                                 block_r=arguments.get("block_r", 16), diagnostic_only=result["diagnostic_only"],
                                 ranks=json.dumps(result["ranks"]), unique_loras=len(result["ranks"]),
                                 max_rank=max(result["ranks"]), metadata_width=result["metadata_width"],
                                 rounds=len(means), cuda_mean_ms=cuda["mean_ms"], cuda_p50_ms=cuda["p50_ms"],
                                 cuda_p95_ms=cuda["p95_ms"], wall_mean_ms=wall["mean_ms"],
                                 mean_layer_ms=cuda["mean_ms"] / report["config"]["num_hidden_layers"],
                                 round_mean_stdev_ms=statistics.pstdev(means),
                                 workspace_bytes=result["memory"]["workspace_bytes"],
                                 max_abs_error=result["validation"]["max_abs_error"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="huggyllama/llama-13b")
    parser.add_argument("--prefix-len", type=int, default=10000)
    parser.add_argument("--pool-adapters", type=int, default=50)
    parser.add_argument("--ranks", nargs="+", type=int, choices=(8, 16, 32, 64), default=[8, 16, 32, 64])
    parser.add_argument("--unique-loras", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--workers", type=int, default=1296)
    parser.add_argument("--rank-loop", choices=("max", "actual"), default="max",
                        help="max preserves the old loop; actual uses each adapter's rank")
    parser.add_argument("--block-r", type=int, choices=(16, 32), default=16,
                        help="rank-axis tile width; FP16 tl.dot requires at least 16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--diagnostic-iterations", type=int, default=3)
    parser.add_argument("--validation-chunk", type=int, default=256)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="129-token, two-layer correctness/CLI run; not a 13B performance result")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    for name in ("prefix_len", "pool_adapters", "workers", "iterations", "repeats",
                 "diagnostic_iterations", "validation_chunk"):
        if getattr(args, name) <= 0:
            parser.error("%s must be positive" % name)
    if args.warmup < 0 or any(v <= 0 for v in args.unique_loras) or args.atol <= 0 or args.rtol <= 0:
        parser.error("invalid warmup, LoRA count or validation tolerance")
    if len(set(args.ranks)) != len(args.ranks) or len(set(args.unique_loras)) != len(args.unique_loras):
        parser.error("duplicate ranks or LoRA counts")
    if not args.list_cases and args.output_dir is None:
        parser.error("--output-dir is required unless --list-cases is used")
    return args


@torch.no_grad()
def main(args):
    cases = make_cases(args.ranks, args.unique_loras)
    if args.cases:
        unknown = set(args.cases) - {case.name for case in cases}
        if unknown:
            raise ValueError("unknown cases: %s" % sorted(unknown))
        cases = [case for case in cases if case.name in args.cases]
    if args.list_cases:
        for case in cases:
            print(case.name, list(case.ranks), "metadata_width=", case.metadata_width or max(case.ranks))
        return
    if any(len(case.ranks) > args.pool_adapters for case in cases):
        raise ValueError("--pool-adapters is smaller than a selected case")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required")
    index = device.index or 0
    torch.cuda.set_device(index)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = dict(get_config_json(args.model_dir))
    if (config["hidden_size"], config["num_hidden_layers"], config["num_attention_heads"],
            config.get("num_key_value_heads", config["num_attention_heads"])) != (5120, 40, 40, 40):
        raise ValueError("this experiment requires the Llama-13B MHA configuration")
    if args.smoke:
        args.prefix_len = 129
        config["num_hidden_layers"] = 2
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("choose a fresh --output-dir")
    import triton
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    report = dict(state="running", all_passed=False, config=config,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), source_sha256=hashes,
                  reconstruction=dict(function="reconstruct_prefix_layer_bounded",
                                      launch_config=[32, 128, args.block_r, 4], rank_loop=args.rank_loop,
                                      workers=args.workers,
                                      b_packed=True, key_rope=True),
                  scope="reconstruction only; synthetic nonzero sources; no base model, attention, FFN, or H2D",
                  reference="FP32 Mini @ original HF B.T; FP16 delta before K RoPE; FP32 Base addition",
                  case_order=[], cases={})
    write_report(output, report)
    rng = random.Random(args.seed)
    try:
        for repetition in range(args.repeats):
            order = cases.copy()
            rng.shuffle(order)
            report["case_order"].append([case.name for case in order])
            for case in tqdm(order, desc="Round %d/%d" % (repetition + 1, args.repeats), disable=args.no_progress):
                print("Preparing %s round=%d (allocation, weights and validation are untimed)" %
                      (case.name, repetition + 1), flush=True)
                fixture = None
                try:
                    fixture = Fixture(args, config, case)
                    if case.name not in report["cases"]:
                        validation = fixture.validate()
                        report["cases"][case.name] = dict(ranks=case.ranks,
                            metadata_width=case.metadata_width or max(case.ranks),
                            diagnostic_only=case.diagnostic_only, memory=fixture.memory_info,
                            validation=validation, rounds=[])
                    for _ in tqdm(range(args.warmup), desc="Warmup " + case.name, disable=args.no_progress):
                        fixture.sweep()
                    torch.cuda.synchronize()
                    times = fixture.time_sweeps(case.name)
                    fixture.check_guards()
                    entry = report["cases"][case.name]
                    times["round"] = repetition + 1
                    entry["rounds"].append(times)
                    write_report(output, report)
                    if repetition == 0:
                        entry["breakdown"] = fixture.breakdown()
                    print("%s: %d-layer sweep mean=%.3f ms, p95=%.3f ms" %
                          (case.name, fixture.layers, times["cuda"]["mean_ms"], times["cuda"]["p95_ms"]), flush=True)
                    write_report(output, report)
                finally:
                    if fixture is not None:
                        fixture.close()
                    del fixture
                    gc.collect()
                    torch.cuda.empty_cache()
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise RuntimeError("production source changed while profiling")
        report.update(state="complete", all_passed=True)
    except BaseException as error:
        report.update(state="failed", error="%s: %s" % (type(error).__name__, error))
        raise
    finally:
        write_report(output, report)
    print("Results: %s" % (output / "result.json"), flush=True)
    print("Summary: %s" % (output / "summary.csv"), flush=True)


if __name__ == "__main__":
    main(parse_args())
