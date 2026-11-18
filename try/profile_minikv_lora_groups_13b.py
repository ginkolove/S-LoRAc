"""Compare different LoRA identities at fixed rank and reconstruction shape.

Each rank has one immutable GPU source pool and one reusable output workspace.
Only the selected adapter IDs change between groups; no model is loaded.
"""

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_minikv_rank_13b as baseline

torch = baseline.torch
tqdm = baseline.tqdm
ROOT = baseline.ROOT
SOURCE_FILES = baseline.SOURCE_FILES + ("try/profile_minikv_lora_groups_13b.py",)


@dataclass(frozen=True)
class Group:
    name: str
    adapter_ids: tuple  # Human-facing, one-based IDs.


def make_groups(pool_size=50, count=6, random_groups=8, seed=42):
    if count <= 0 or pool_size < 9 + count or random_groups < 0:
        raise ValueError("pool must contain IDs 1..U and 10..(9+U); random-groups must be nonnegative")
    if random_groups > math.comb(pool_size, count) - 2:
        raise ValueError("not enough distinct adapter combinations")
    groups = [Group("first", tuple(range(1, count + 1))),
              Group("second", tuple(range(10, 10 + count)))]
    seen = {group.adapter_ids for group in groups}
    rng = random.Random(seed)
    while len(groups) < random_groups + 2:
        ids = tuple(sorted(rng.sample(range(1, pool_size + 1), count)))
        if ids not in seen:
            seen.add(ids)
            groups.append(Group("random_%02d" % (len(groups) - 1), ids))
    return groups


def resident_pool_slots(prefix, layers, rank, pool_size, selected_count):
    return 4 * rank * pool_size + (prefix * selected_count + layers - 1) // layers + 128


class GroupFixture(baseline.Fixture):
    """Reuse the original profiler's reconstruction, oracle, timing and cleanup."""

    def __init__(self, args, config, rank):
        self.args, self.config, self.rank = args, config, rank
        self.prefix, self.layers = args.prefix_len, config["num_hidden_layers"]
        self.hidden, self.heads = config["hidden_size"], config["num_attention_heads"]
        self.dim = self.hidden // self.heads
        all_dirs = ["adapter_%03d" % i for i in range(1, args.pool_adapters + 1)]
        capacity = resident_pool_slots(self.prefix, self.layers, rank, args.pool_adapters, args.unique_loras)
        static_bytes = 4 * self.layers * self.prefix * (self.hidden + rank * args.pool_adapters)
        pool_bytes = 4 * self.layers * capacity * self.hidden
        free, _ = torch.cuda.mem_get_info()
        if static_bytes + pool_bytes + 1024 ** 3 > free:
            raise RuntimeError("rank %d requires about %.2f GiB plus 1 GiB headroom; %.2f GiB free" %
                               (rank, (static_bytes + pool_bytes) / 1024 ** 3, free / 1024 ** 3))
        self.static = baseline.LowRAStaticPrefixStore(
            self.prefix, all_dirs, dict.fromkeys(all_dirs, rank), self.layers, self.heads, self.heads, self.dim)
        base_rng = torch.Generator(device="cuda").manual_seed(args.seed)
        mini_rng = torch.Generator(device="cuda").manual_seed(args.seed + 1)
        for layer in tqdm(range(self.layers), desc="Sources r%d" % rank, disable=args.no_progress):
            for tensor in (self.static.base_k[layer], self.static.base_v[layer]):
                tensor.normal_(std=0.1, generator=base_rng)
            for tensor in (self.static.mini_k[layer], self.static.mini_v[layer]):
                tensor.normal_(std=0.1, generator=mini_rng)
        positions = torch.arange(self.prefix, device="cuda", dtype=torch.float32)[:, None]
        inv = 10000.0 ** (-torch.arange(0, self.dim, 2, device="cuda").float() / self.dim)
        angles = positions * inv[None]
        self.static.bind_key_rope(angles.cos().half(), angles.sin().half())

        self.memory = baseline.MemoryAllocator(capacity, capacity, torch.float16, self.heads, self.dim,
                                               self.layers, dynamic_layer_workspace=True)
        held = self.memory.alloc(64)
        self.memory.free(held[::2])
        self.guards = held[1::2]
        self.infer = baseline.InferAdapter.init(self.memory, None)
        self.original_b = {}
        # Load one adapter at a time to bound pinned CPU initialization memory.
        # All 50 stay GPU-resident throughout this rank's experiment.
        for index, directory in enumerate(tqdm(all_dirs, desc="Weights r%d" % rank, disable=args.no_progress)):
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
                weight = baseline.LoraLayerWeight(layer, 0, 1, {"r": rank}, config)
                weight.load_hf_weights_cpu(hf)
                weights.append(weight)
            adapter = SimpleNamespace(lora_dir=directory, r=rank, scaling=0.5,
                                      network_config=config, layers=weights)
            self.infer.load_adapters([adapter])
            del adapter, weights, weight, hf
        self.locations = self.memory.alloc_layer_workspace(args.unique_loras * self.prefix).view(args.unique_loras, -1)
        for layer in range(self.layers):
            self.memory.key_buffer[layer][self.guards] = 0.625
            self.memory.value_buffer[layer][self.guards] = -0.75
        torch.cuda.synchronize()
        self.memory_info = dict(static_bytes=static_bytes, pool_slots=capacity,
                                pool_bytes=self.memory.get_memory_size(),
                                resident_weight_adapters=args.pool_adapters,
                                workspace_bytes=4 * args.unique_loras * self.prefix * self.hidden,
                                workspace_full_layer_slots=self.memory.workspace_token_equiv,
                                allocated_bytes=torch.cuda.memory_allocated())

    def select(self, group):
        ids = group.adapter_ids
        if (len(ids) != self.args.unique_loras or len(set(ids)) != len(ids)
                or any(i < 1 or i > self.args.pool_adapters for i in ids)):
            raise ValueError("invalid adapter group")
        self.directories = ["adapter_%03d" % i for i in ids]
        self.case = baseline.Case("r%d_%s" % (self.rank, group.name), (self.rank,) * len(ids), self.rank)
        self.metadata = baseline.runtime.prepare_reconstruction_metadata(
            self.static, self.directories, self.infer, self.locations)


def cv_percent(values):
    mean = statistics.mean(values)
    return 100 * statistics.pstdev(values) / mean if mean else 0.0


def aggregate(report):
    group_rows, rank_rows = [], []
    for rank in report["arguments"]["ranks"]:
        cases = [(name, c) for name, c in report["cases"].items() if c["rank"] == rank and c["rounds"]]
        if not cases:
            continue
        round_medians = {}
        for _, case in cases:
            for run in case["rounds"]:
                round_medians.setdefault(run["round"], []).append(run["cuda"]["mean_ms"])
        round_medians = {r: statistics.median(values) for r, values in round_medians.items()}
        rows = []
        for name, case in cases:
            cuda = baseline.summarize([x for r in case["rounds"] for x in r["cuda_ms"]])
            wall = baseline.summarize([x for r in case["rounds"] for x in r["wall_ms"]])
            means = [r["cuda"]["mean_ms"] for r in case["rounds"]]
            row = dict(case=name, rank=rank, group=case["group"], adapter_ids=json.dumps(case["adapter_ids"]),
                       rounds=len(means), samples=cuda["samples"], cuda_mean_ms=cuda["mean_ms"],
                       cuda_p50_ms=cuda["p50_ms"], cuda_p95_ms=cuda["p95_ms"], wall_mean_ms=wall["mean_ms"],
                       mean_layer_ms=cuda["mean_ms"] / report["config"]["num_hidden_layers"],
                       within_group_round_cv_pct=cv_percent(means),
                       within_visit_cv_pct=statistics.mean(cv_percent(r["cuda_ms"]) for r in case["rounds"]),
                       round_normalized_ratio=statistics.mean(r["cuda"]["mean_ms"] / round_medians[r["round"]]
                                                              for r in case["rounds"]),
                       max_abs_error=case["validation"]["max_abs_error"])
            rows.append(row)
        group_rows.extend(rows)
        means = [r["cuda_mean_ms"] for r in rows]
        slowest, fastest = max(rows, key=lambda r: r["cuda_mean_ms"]), min(rows, key=lambda r: r["cuda_mean_ms"])
        rank_rows.append(dict(rank=rank, groups=len(rows),
                              complete=len(rows) == len(report["groups"]) and
                              all(r["rounds"] == report["arguments"]["repeats"] for r in rows),
                              fastest_group=fastest["group"], slowest_group=slowest["group"],
                              min_group_mean_ms=min(means), max_group_mean_ms=max(means),
                              max_min_ratio=max(means) / min(means),
                              between_group_cv_pct=cv_percent(means),
                              median_within_group_round_cv_pct=statistics.median(r["within_group_round_cv_pct"] for r in rows),
                              median_within_visit_cv_pct=statistics.median(r["within_visit_cv_pct"] for r in rows),
                              normalized_between_group_cv_pct=cv_percent([r["round_normalized_ratio"] for r in rows])))
    return group_rows, rank_rows


def write_report(output, report):
    group_rows, rank_rows = aggregate(report)
    report["rank_comparison"] = rank_rows
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "result.json")
    for name, rows in (("groups.csv", group_rows), ("rank_comparison.csv", rank_rows)):
        if rows:
            with (output / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="huggyllama/llama-13b")
    parser.add_argument("--prefix-len", type=int, default=10000)
    parser.add_argument("--pool-adapters", type=int, default=50)
    parser.add_argument("--unique-loras", type=int, default=6)
    parser.add_argument("--random-groups", type=int, default=8)
    parser.add_argument("--ranks", nargs="+", type=int, choices=(8, 16, 32, 64), default=[8, 16, 32, 64])
    parser.add_argument("--workers", type=int, default=1296)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-chunk", type=int, default=256)
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list-groups", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="two layers and 129 tokens; not a 13B 10K performance result")
    args = parser.parse_args(argv)
    if any(getattr(args, name) <= 0 for name in
           ("prefix_len", "workers", "iterations", "repeats", "validation_chunk", "atol", "rtol")) or args.warmup < 0:
        parser.error("invalid count or validation tolerance")
    if len(set(args.ranks)) != len(args.ranks):
        parser.error("duplicate ranks")
    try:
        make_groups(args.pool_adapters, args.unique_loras, args.random_groups, args.seed)
    except ValueError as error:
        parser.error(str(error))
    if not args.list_groups and args.output_dir is None:
        parser.error("--output-dir is required")
    return args


@torch.no_grad()
def main(args):
    groups = make_groups(args.pool_adapters, args.unique_loras, args.random_groups, args.seed)
    if args.list_groups:
        for group in groups:
            print(group.name, list(group.adapter_ids))
        return
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required")
    torch.cuda.set_device(device.index or 0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = dict(baseline.get_config_json(args.model_dir))
    if (config["hidden_size"], config["num_hidden_layers"], config["num_attention_heads"],
            config.get("num_key_value_heads", config["num_attention_heads"])) != (5120, 40, 40, 40):
        raise ValueError("this experiment requires the Llama-13B MHA configuration")
    if args.smoke:
        args.prefix_len = 129
        config["num_hidden_layers"] = 2
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("choose a fresh output directory")
    import triton
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    report = dict(state="running", all_passed=False, config=config,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  groups=[dict(name=g.name, adapter_ids=g.adapter_ids) for g in groups],
                  device=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  torch=torch.__version__, triton=triton.__version__, source_sha256=hashes,
                  scope="fixed all-adapter GPU sources per rank; reconstruction only; no model/attention/FFN/H2D",
                  reconstruction=dict(function="reconstruct_prefix_layer_bounded", workers=args.workers,
                                      launch_config=[32, 128, 16, 4], b_packed=True, key_rope=True),
                  memory_by_rank={}, execution_order=[], cases={})
    write_report(output, report)
    try:
        for rank in args.ranks:
            print("Preparing fixed rank=%d pool (%d adapters); setup is untimed" % (rank, args.pool_adapters), flush=True)
            fixture = None
            try:
                fixture = GroupFixture(args, config, rank)
                report["memory_by_rank"][str(rank)] = fixture.memory_info
                rng = random.Random(args.seed + 2027)
                for repetition in range(args.repeats):
                    order = groups.copy()
                    rng.shuffle(order)
                    report["execution_order"].append(dict(rank=rank, round=repetition + 1, groups=[g.name for g in order]))
                    for group in tqdm(order, desc="Rank %d round %d" % (rank, repetition + 1), disable=args.no_progress):
                        fixture.select(group)
                        name = fixture.case.name
                        if name not in report["cases"]:
                            report["cases"][name] = dict(rank=rank, group=group.name, adapter_ids=group.adapter_ids,
                                                         validation=fixture.validate(), rounds=[])
                        for _ in range(args.warmup):
                            fixture.sweep()
                        torch.cuda.synchronize()
                        times = fixture.time_sweeps(name)
                        fixture.check_guards()
                        times["round"] = repetition + 1
                        report["cases"][name]["rounds"].append(times)
                        write_report(output, report)
                        print("%s IDs=%s %d-layer mean=%.3f ms p95=%.3f ms" %
                              (name, list(group.adapter_ids), fixture.layers,
                               times["cuda"]["mean_ms"], times["cuda"]["p95_ms"]), flush=True)
            finally:
                if fixture is not None:
                    fixture.close()
                del fixture
                gc.collect()
                torch.cuda.empty_cache()
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h for name, h in hashes.items()):
            raise RuntimeError("profile or reconstruction source changed during the run")
        report.update(state="complete", all_passed=True)
    except BaseException as error:
        report.update(state="failed", error="%s: %s" % (type(error).__name__, error))
        raise
    finally:
        write_report(output, report)
    print("Results: %s" % output, flush=True)


if __name__ == "__main__":
    main(parse_args())
