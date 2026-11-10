"""Quantitative audit against original HF weights, not another Triton kernel."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from test_lowra_packed_reconstruction import PackedReconstructionTests
from slora.common.lowra_layer_runtime import (
    prepare_reconstruction_metadata, reconstruct_prefix_layer,
    reconstruct_prefix_layer_bounded, reconstruct_prefix_layer_fused_kv,
)
from slora.common.lowra_reconstruct_bounded import lowra_reconstruct_bounded


CASES = (
    dict(name="one_hot_exact", prefix=17, hidden=128, layers=1, ranks=[8, 16], selected=2, one_hot=True),
    dict(name="small_multilayer", prefix=129, hidden=1024, layers=3, ranks=[8, 16], selected=2),
    dict(name="mixed_rank_group_edge", prefix=257, hidden=4096, layers=2,
         ranks=[8, 16, 32, 64, 8, 16, 32, 64, 16], selected=9),
    dict(name="llama7b_experiment_shape", prefix=4608, hidden=4096, layers=1, ranks=[16] * 50, selected=20),
    dict(name="long_prefix_tail", prefix=8193, hidden=4096, layers=1, ranks=[8, 16, 32], selected=2),
    dict(name="llama13b_width", prefix=1153, hidden=5120, layers=2, ranks=[8, 16, 32], selected=3),
    dict(name="llama30b_width", prefix=513, hidden=6656, layers=2, ranks=[16, 32, 64], selected=3),
)
KERNELS = ("split", "fused_kv", "bounded_648", "bounded_1296")
ATOL, RTOL = 5e-4, 2e-3


def empty_stats():
    return dict(elements=0, failures=0, nonfinite=0, max_abs_error=0.0,
                abs_error_sum=0.0, squared_error_sum=0.0,
                different_from_rounded_reference=0, max_abs_error_vs_rounded=0.0)


def compare(actual, reference, stats):
    value = actual.float()
    error = (value - reference).abs()
    rounded = reference.half()
    stats["elements"] += actual.numel()
    stats["nonfinite"] += int((~torch.isfinite(value)).sum().item())
    stats["failures"] += int((~torch.isfinite(value) | (error > ATOL + RTOL * reference.abs())).sum().item())
    stats["max_abs_error"] = max(stats["max_abs_error"], error.max().item())
    stats["abs_error_sum"] += error.sum(dtype=torch.float64).item()
    stats["squared_error_sum"] += error.square().sum(dtype=torch.float64).item()
    stats["different_from_rounded_reference"] += int((actual != rounded).sum().item())
    stats["max_abs_error_vs_rounded"] = max(
        stats["max_abs_error_vs_rounded"], (value - rounded.float()).abs().max().item())
    torch.testing.assert_close(value, reference, atol=ATOL, rtol=RTOL)


@torch.no_grad()
def run_case(case, seed):
    ranks = {"adapter_%02d" % i: rank for i, rank in enumerate(case["ranks"])}
    static, memory, infer, original, held = PackedReconstructionTests().fixture(
        prefix=case["prefix"], hidden=case["hidden"], layers=case["layers"], ranks=ranks, seed=seed)
    guards = memory.alloc(33)
    original_scales = {
        directory: (1.0 if case.get("one_hot") else (0.25, 0.5, 1.75, -0.5, 0.0)[i % 5])
        for i, directory in enumerate(ranks)
    }
    for directory, scale in original_scales.items():
        infer.a_scaling[infer.idx_map[directory]] = scale
    for layer in range(case["layers"]):
        for buffer in (memory.key_buffer[layer], memory.value_buffer[layer]):
            buffer[held] = 0.625
            buffer[guards] = -7.25 - layer
        if case.get("one_hot"):
            for base in (static.base_k[layer], static.base_v[layer]):
                base.fill_(0.125)
            for mini in (static.mini_k[layer], static.mini_v[layer]):
                mini.zero_()
                positions = torch.arange(case["prefix"], device="cuda")
                for directory, rank in ranks.items():
                    start, _ = static.get_lora_rank_range_by_dir(directory)
                    mini[positions, start + positions % rank] = 1.0
    protected = torch.cat((held, guards, infer.a_loc))
    snapshots = [(memory.key_buffer[layer][protected].clone(), memory.value_buffer[layer][protected].clone())
                 for layer in range(case["layers"])]
    generator = torch.Generator().manual_seed(seed + 500)
    order = torch.randperm(len(ranks), generator=generator)[:case["selected"]].tolist()
    directories = [list(ranks)[index] for index in order]
    locations = memory.alloc_layer_workspace(len(directories) * case["prefix"]).view(len(directories), -1)
    metadata = prepare_reconstruction_metadata(static, directories, infer, locations)
    report = dict(**case, seed=seed, selected_directories=directories, scales=original_scales,
                  kernels={name: empty_stats() for name in KERNELS},
                  fp64_reference=dict(elements=0, max_abs_error_vs_fp32=0.0,
                                      scope="all_elements" if case["prefix"] * case["hidden"] <= 132096
                                      else "64_rows_by_64_features_per_adapter_layer_K_or_V"))
    try:
        for layer in range(case["layers"]):
            references = {}
            for row, directory in enumerate(directories):
                start, end = static.get_lora_rank_range_by_dir(directory)
                for module in ("k", "v"):
                    base = (static.base_k if module == "k" else static.base_v)[layer].view(case["prefix"], -1)
                    mini = (static.mini_k if module == "k" else static.mini_v)[layer][:, start:end]
                    key = "base_model.model.model.layers.%s.self_attn.%s_proj.lora_B.weight" % (layer, module)
                    hf_b = original[directory][layer][key]
                    scale = original_scales[directory]
                    reference = base.float() + (mini.float() @ hf_b.cuda().float().T) * scale
                    references[(row, module)] = reference
                    if report["fp64_reference"]["scope"] == "all_elements":
                        rows, features = torch.arange(case["prefix"]), torch.arange(case["hidden"])
                    else:
                        rows = torch.linspace(0, case["prefix"] - 1, min(64, case["prefix"])).long()
                        features = torch.linspace(0, case["hidden"] - 1, min(64, case["hidden"])).long()
                    rows_gpu, features_gpu = rows.cuda(), features.cuda()
                    cpu_reference = base[rows_gpu][:, features_gpu].cpu().double() + (
                        mini[rows_gpu].cpu().double() @ hf_b[features].double().T) * scale
                    sampled = reference[rows_gpu][:, features_gpu].cpu().double()
                    torch.testing.assert_close(sampled, cpu_reference, atol=5e-6, rtol=1e-5)
                    report["fp64_reference"]["elements"] += sampled.numel()
                    report["fp64_reference"]["max_abs_error_vs_fp32"] = max(
                        report["fp64_reference"]["max_abs_error_vs_fp32"],
                        (sampled - cpu_reference).abs().max().item())
                    if case.get("one_hot"):
                        columns = torch.arange(case["prefix"]) % ranks[directory]
                        analytic = (0.125 + hf_b[:, columns].T.double()).half()
                        torch.testing.assert_close(reference.half().cpu(), analytic, atol=0, rtol=0)

            for name in KERNELS:
                for buffer in (memory.prefix_workspace_k, memory.prefix_workspace_v):
                    buffer[locations.reshape(-1)] = float("nan")
                args = (static, infer, metadata, layer, memory.prefix_workspace_k, memory.prefix_workspace_v)
                if name == "split":
                    reconstruct_prefix_layer(*args)
                elif name == "fused_kv":
                    reconstruct_prefix_layer_fused_kv(*args)
                else:
                    reconstruct_prefix_layer_bounded(*args, workers=int(name.split("_")[1]))
                for row, directory in enumerate(directories):
                    for module, output in (("k", memory.prefix_workspace_k), ("v", memory.prefix_workspace_v)):
                        actual = output[locations[row]].view(case["prefix"], -1)
                        compare(actual, references[(row, module)], report["kernels"][name])
                        if case.get("one_hot"):
                            torch.testing.assert_close(actual, references[(row, module)].half(), atol=0, rtol=0)
                for check_layer, (k, v) in enumerate(snapshots):
                    torch.testing.assert_close(memory.key_buffer[check_layer][protected], k, atol=0, rtol=0)
                    torch.testing.assert_close(memory.value_buffer[check_layer][protected], v, atol=0, rtol=0)

            if layer == 0:
                row = next(index for index, directory in enumerate(directories) if original_scales[directory] != 0)
                lowra_reconstruct_bounded(
                    static.base_k[0].view(case["prefix"], -1), static.mini_k[0],
                    memory.value_buffer[0].view(-1, case["hidden"]),
                    memory.prefix_workspace_k.view(-1, case["hidden"]), locations[row:row + 1],
                    metadata.k_b_locs[row:row + 1], metadata.rank_starts[row:row + 1],
                    metadata.rank_lens[row:row + 1], metadata.scalings[row:row + 1],
                    workers=1296, launch_config=(32, 128, 16, 4), b_packed=False)
                old = memory.prefix_workspace_k[locations[row]].view(case["prefix"], -1).float()
                reference = references[(row, "k")]
                delta = (old - reference).abs()
                fraction = (delta > ATOL + RTOL * reference.abs()).float().mean().item()
                report["negative_control"] = dict(max_abs_error=delta.max().item(), mismatch_fraction=fraction)
                if fraction <= 0.5:
                    raise AssertionError("old wrong-layout negative control was not rejected")
            del references

        for stats in report["kernels"].values():
            stats["mean_abs_error"] = stats.pop("abs_error_sum") / stats["elements"]
            stats["rmse"] = (stats.pop("squared_error_sum") / stats["elements"]) ** 0.5
        report.update(all_passed=True, one_hot_bitwise_exact=bool(case.get("one_hot")),
                      source_weights_and_private_guards_unchanged=True)
    finally:
        memory.free_layer_workspace()
        infer.offload_adapters([])
        memory.free(guards)
        memory.free(held)
        if memory.can_use_mem_size != memory.tot_size:
            raise AssertionError("pool slots leaked")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    parser.add_argument("--cases", nargs="+", choices=[case["name"] for case in CASES])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; a skipped CPU-only run is not an audit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = ["slora/common/lowra_reconstruct.py", "slora/common/lowra_reconstruct_kv.py",
             "slora/common/lowra_reconstruct_bounded.py", "slora/common/lowra_layer_runtime.py",
             "test/test_lowra_packed_reconstruction.py", "try/validate_packed_reconstruction.py"]
    report = dict(device=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  torch=torch.__version__, triton=triton.__version__, seeds=args.seeds,
                  reference="Base.float() + Mini.float() @ original_HF_B.float().T * scale; TF32 disabled",
                  cpu_crosscheck="CPU float64 using original HF B; full small cases, sampled large cases",
                  weights="synthetic nonzero HF-format A/B through real S-LoRA loading; not pretrained checkpoints",
                  atol=ATOL, rtol=RTOL, all_passed=False,
                  source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}, cases=[])
    output = args.output_dir / "result.json"
    for case in CASES:
        if args.cases and case["name"] not in args.cases:
            continue
        for seed in args.seeds:
            print("Checking %s seed=%s" % (case["name"], seed), flush=True)
            result = run_case(case, seed)
            report["cases"].append(result)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(dict(case=case["name"], seed=seed, passed=True,
                                  max_abs_error=max(stats["max_abs_error"] for stats in result["kernels"].values()),
                                  cpu_fp64_max_error=result["fp64_reference"]["max_abs_error_vs_fp32"])), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    report["checked_elements"] = sum(stats["elements"] for case in report["cases"] for stats in case["kernels"].values())
    report["all_passed"] = True
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["source_unchanged_during_audit"] = all(
        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        for path, digest in report["source_sha256"].items())
    if not report["source_unchanged_during_audit"]:
        raise AssertionError("audited source changed during execution")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(all_passed=True, cases=len(report["cases"]),
                          checked_elements=report["checked_elements"], output=str(output))), flush=True)


if __name__ == "__main__":
    main()
