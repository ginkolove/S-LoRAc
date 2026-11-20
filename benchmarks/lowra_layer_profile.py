"""Fixed-batch profile of full, serial, and single-buffer FFN-overlapped prefixes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slora.common.prefix_modes import (
    FULL_RECONSTRUCT, FULL_REPREFILL, LAYER_PREFETCH, LAYER_SERIAL, LAYER_RETAINED, LAYER_ADAPTIVE, normalize_prefix_mode,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="huggyllama/llama-7b")
    parser.add_argument("--adapter-base", default="dummy-lora-7b-rank-16")
    parser.add_argument("--mode", type=lambda value: value if value == "all" else normalize_prefix_mode(value),
                        choices=("all", FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, LAYER_PREFETCH,
                                 LAYER_RETAINED, LAYER_ADAPTIVE), default="all")
    parser.add_argument("--prefix-len", type=int, default=1440)
    parser.add_argument("--query-len", type=int, default=480)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--unique-loras", type=int, default=8)
    parser.add_argument("--num-adapters", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--retain-layers", type=int, default=0)
    parser.add_argument("--retain-top-k", type=int, default=None)
    parser.add_argument("--pin-hot-adapters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dummy", action="store_true", help="Use S-LoRA dummy model/adapter weights.")
    parser.add_argument("--smoke", action="store_true", help="Explicit 3-layer, 1024-hidden dummy test model.")
    parser.add_argument("--nonzero-prefix", action="store_true", help="Nonzero BaseKV/MiniKV for numerical checks.")
    parser.add_argument("--trace-dir", type=Path, help="Export a separate CUDA Chrome trace for each mode.")
    parser.add_argument("--output", type=Path, default=Path("lowra_layer_profile.json"))
    args = parser.parse_args()
    args.num_adapters = args.num_adapters or args.unique_loras
    if not (0 < args.unique_loras <= args.batch_size and args.unique_loras <= args.num_adapters):
        parser.error("unique-loras must fit both batch-size and num-adapters")
    if min(args.prefix_len, args.query_len, args.output_len, args.rank, args.iterations) <= 0 or args.warmup < 0:
        parser.error("lengths, rank, iterations must be positive; warmup must be nonnegative")
    if args.smoke and not args.dummy:
        parser.error("--smoke requires --dummy")
    if args.retain_layers < 0 or (args.retain_layers and args.mode != LAYER_RETAINED):
        parser.error("--retain-layers requires minikv-layer_retained and a nonnegative count")
    if args.retain_top_k is not None and (args.retain_top_k < 0 or args.mode != LAYER_RETAINED):
        parser.error("--retain-top-k requires minikv-layer_retained and a nonnegative count")
    if not 0 <= args.pin_hot_adapters <= args.num_adapters:
        parser.error("--pin-hot-adapters must be between zero and num-adapters")
    if args.pin_hot_adapters and (args.mode not in (FULL_RECONSTRUCT, LAYER_RETAINED)
                                 or args.retain_top_k is not None):
        parser.error("pinned hot prefixes require a MiniKV cache mode and no dynamic top-K")
    if args.pin_hot_adapters and args.mode == LAYER_RETAINED and not args.retain_layers:
        parser.error("pinned layer retention requires positive --retain-layers")
    return args


def build_server(args):
    args.mode = normalize_prefix_mode(args.mode)
    import torch
    from slora.common.configs.config import setting
    from slora.common.lowra_layer_runtime import LowRALayerRuntime, LowRAProfileEvents
    from slora.common.lowra_static import LowRAStaticPrefixStore
    from slora.common.full_reprefill import FULL_REPREFILL_MODES, FULL_PREFIX_CACHE_MODES, FullPrefixKVCache, FullReprefillRuntime
    from slora.common.full_reconstruct import FullReconstructPrefixKVCache, FullReconstructRuntime
    from slora.common.lowra_retained import RetainedPrefixKVCache, RetainedLayerRuntime
    from slora.common.lowra_adaptive import AdaptivePrefixKVCache, AdaptiveLayerRuntime
    from slora.models.llama.model import LlamaTpPartModel
    from slora.models.peft.lora_adapter import LoraTpPartAdapter
    from slora.server.router.model_infer.infer_adapter import InferAdapter
    from slora.server.router.model_infer.model_rpc import ModelRpcServer
    from slora.utils.model_utils import get_model_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for layer profiling")
    torch.manual_seed(args.seed)
    if not torch.distributed.is_initialized():
        with tempfile.NamedTemporaryFile() as rendezvous:
            torch.distributed.init_process_group("nccl", init_method="file://" + rendezvous.name,
                                                 rank=0, world_size=1)
    config = get_model_config(args.model_dir, dummy=args.dummy)
    if args.smoke:
        config = dict(config, hidden_size=1024, num_attention_heads=8, num_hidden_layers=3,
                      intermediate_size=2048, vocab_size=256, max_sequence_length=2048)
        config.pop("num_key_value_heads", None)
    if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
        raise ValueError("S-LoRA packed adapters currently require MHA")
    logical_len = args.prefix_len + args.query_len + args.output_len
    context = config.get("max_sequence_length", config.get("max_position_embeddings", 2048))
    if logical_len > context:
        raise ValueError("prefix + query + output exceeds the model context limit")
    setting["max_req_total_len"] = logical_len
    layers = config["num_hidden_layers"]
    reprefill = args.mode in FULL_REPREFILL_MODES
    full = args.mode in FULL_PREFIX_CACHE_MODES
    workspace = args.unique_loras * args.prefix_len if not full else 0
    # Minimal backing for this fixed workload, not a same-budget throughput experiment.
    persistent = args.batch_size * (args.query_len + args.output_len) + 4 * args.unique_loras * args.rank
    prefix_budget = ((workspace + layers - 1) // layers if workspace else
                     args.unique_loras * args.prefix_len)
    if args.mode == LAYER_ADAPTIVE:
        prefix_budget = args.unique_loras * args.prefix_len + (workspace + layers - 1) // layers
    if args.mode == LAYER_RETAINED:
        retained = getattr(args, "retain_layers", 0)
        if not 0 <= retained <= layers:
            raise ValueError("retain-layers must be between zero and model layers")
        top_k = getattr(args, "retain_top_k", None)
        if getattr(args, "pin_hot_adapters", 0):
            top_k = args.pin_hot_adapters
        hot = args.unique_loras if top_k is None else min(args.unique_loras, top_k)
        scratch_count = args.unique_loras if retained < layers else args.unique_loras - hot
        prefix_budget = (scratch_count * args.prefix_len + layers - 1) // layers
        prefix_budget += hot * ((args.prefix_len * retained + layers - 1) // layers)
    with patch("slora.common.basemodel.basemodel.get_config_json", return_value=config):
        model = LlamaTpPartModel(0, 1, args.model_dir, persistent + prefix_budget,
                                mem_adapter_size=0, dummy=args.dummy,
                                dynamic_layer_workspace=not full)
    dirs = ["%s-%s" % (args.adapter_base, i) for i in range(args.num_adapters)]
    adapters = [LoraTpPartAdapter(0, 1, directory, config, swap=True, dummy=args.dummy)
                for directory in dirs]
    if any(adapter.r != args.rank for adapter in adapters):
        raise ValueError("--rank does not match the adapter configuration")
    infer_adapter = InferAdapter.init(model.mem_manager, None)
    infer_adapter.load_adapters(adapters[:args.unique_loras])
    profiler = LowRAProfileEvents()
    cache = None
    if reprefill:
        cache = FullPrefixKVCache(args.prefix_len, model.mem_manager)
        runtime = FullReprefillRuntime(args.prefix_len, model.mem_manager, cache, profiler)
    else:
        static = LowRAStaticPrefixStore(args.prefix_len, dirs, {d: args.rank for d in dirs}, layers,
                                        model.tp_k_head_num_, model.tp_v_head_num_, model.head_dim_)
        static.bind_key_rope(model._cos_cached[:args.prefix_len], model._sin_cached[:args.prefix_len])
        if args.nonzero_prefix:
            for tensor in static.base_k + static.base_v + static.mini_k + static.mini_v:
                tensor.normal_(std=0.02)
        if args.mode == FULL_RECONSTRUCT:
            cache = FullReconstructPrefixKVCache(args.prefix_len, model.mem_manager,
                                                 pinned_lora_dirs=dirs[:getattr(args, "pin_hot_adapters", 0)])
            runtime = FullReconstructRuntime(static, model.mem_manager, cache, profiler)
        elif args.mode == LAYER_ADAPTIVE:
            cache = AdaptivePrefixKVCache(args.prefix_len, model.mem_manager)
            runtime = AdaptiveLayerRuntime(static, model.mem_manager, cache, profiler,
                reconstruction_kernel=getattr(args, "reconstruction_kernel", "auto"),
                reconstruction_workers=getattr(args, "reconstruction_workers", None))
        elif args.mode == LAYER_RETAINED:
            cache = RetainedPrefixKVCache(args.prefix_len, args.retain_layers, model.mem_manager,
                                          retain_top_k=getattr(args, "retain_top_k", None),
                                          pinned_lora_dirs=dirs[:getattr(args, "pin_hot_adapters", 0)])
            runtime = RetainedLayerRuntime(static, model.mem_manager, cache, profiler,
                                          reconstruction_kernel=getattr(args, "reconstruction_kernel", "auto"),
                                          reconstruction_workers=getattr(args, "reconstruction_workers", None))
        else:
            runtime = LowRALayerRuntime(static, model.mem_manager, args.mode, profiler,
                                       reconstruction_kernel=getattr(args, "reconstruction_kernel", "auto"),
                                       reconstruction_workers=getattr(args, "reconstruction_workers", None))
    server = ModelRpcServer()
    server.model, server.infer_adapter = model, infer_adapter
    server.adapters, server.adapter_id = adapters, {d: i for i, d in enumerate(dirs)}
    server.world_size, server.cache = 1, {}
    server.lowra_runtime, server.lowra_prefix_cache = runtime, cache
    server.input_params = SimpleNamespace(no_lora=False, no_kernel=False, scheduler="slora",
                                         no_lora_compute=False, bmm=False)
    requests = [dict(request_id=str(i), adapter_dir=dirs[i % args.unique_loras],
                     input_id=[10] * args.query_len, output_len=args.output_len,
                     sampling_param=dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=1))
                for i in range(args.batch_size)]
    if reprefill:
        for request in requests:
            request["prefix_input_ids"] = [10] * args.prefix_len
    return server, profiler, requests, config


def run_round(server, profiler, requests, output_len):
    import torch
    profiler.phase = "prefill"
    start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    profiler.origin = start
    round_requests = [dict(request, input_id=list(request["input_id"]),
                           sampling_param=dict(request["sampling_param"])) for request in requests]
    server.exposed_add_batch("profile", round_requests, "fp16")
    server.exposed_prefill_batch("profile")
    prefill_end.record()
    for step in range(output_len - 1):
        profiler.phase = "decode_%s" % step
        server.exposed_decode_batch("profile")
    finish.record()
    finish.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    result = dict(prefill_ms=start.elapsed_time(prefill_end),
                  decode_ms=prefill_end.elapsed_time(finish),
                  total_cuda_ms=start.elapsed_time(finish), wall_ms=wall_ms)
    logits_ids = [list(tokens) for tokens in server.cache["profile"].all_input_ids]
    intervals = profiler.collect()
    by_phase_layer = {(event["phase"], event["layer"], event["operation"]): event
                      for event in intervals}
    overlap = 0.0
    for event in intervals:
        if event["operation"] != "reconstruct" or event["layer"] == 0:
            continue
        ffn = by_phase_layer.get((event["phase"], event["layer"] - 1, "ffn"))
        if ffn:
            overlap += max(0.0, min(event["end_ms"], ffn["end_ms"])
                           - max(event["start_ms"], ffn["start_ms"]))
    result["event_interval_overlap_ms"] = overlap
    server.exposed_remove_batch("profile")
    if server.lowra_prefix_cache is not None:
        server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
    return result, intervals, logits_ids


def tensor_storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def base_model_memory_bytes(model):
    import torch

    weights = [tensor for layer in [model.pre_post_weight, *model.trans_layers_weight]
               for tensor in vars(layer).values()
               if isinstance(tensor, torch.Tensor) and tensor.is_cuda]
    # Direct model tensors are resident buffers (RoPE), not the allocator-owned KV pool.
    buffers = [tensor for tensor in vars(model).values()
               if isinstance(tensor, torch.Tensor) and tensor.is_cuda]
    weight_bytes = tensor_storage_bytes(weights)
    resident_bytes = tensor_storage_bytes(weights + buffers)
    return dict(base_model_weight_bytes=weight_bytes,
                base_model_buffer_bytes=resident_bytes - weight_bytes,
                base_model_resident_bytes=resident_bytes)


def profile_one(args):
    import torch
    from tqdm import trange
    from slora.common.full_reprefill import FULL_REPREFILL_MODES, FULL_PREFIX_CACHE_MODES

    server, profiler, requests, config = build_server(args)
    model_memory = base_model_memory_bytes(server.model)
    for _ in trange(args.warmup, desc="%s warmup" % args.mode):
        run_round(server, profiler, requests, args.output_len)
        profiler.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    measurements, intervals, output_ids = [], [], None
    for iteration in trange(args.iterations, desc="%s profile" % args.mode):
        profiler.clear()
        timing, events, output_ids = run_round(server, profiler, requests, args.output_len)
        measurements.append(timing)
        intervals.extend(dict(event, iteration=iteration) for event in events)
    memory = server.model.mem_manager
    allocated_peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.clear()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as trace:
            run_round(server, profiler, requests, args.output_len)
        trace.export_chrome_trace(str(args.trace_dir / (args.mode + ".json")))
    grouped = defaultdict(list)
    for event in intervals:
        phase = "prefill" if event["phase"] == "prefill" else "decode"
        grouped[(phase, event["layer"], event["operation"])].append(event["milliseconds"])
    summary = [dict(phase=phase, layer=layer, operation=operation,
                    mean_ms=sum(values) / len(values), samples=len(values))
               for (phase, layer, operation), values in sorted(grouped.items())]
    times = {(row["phase"], row["layer"], row["operation"]): row["mean_ms"] for row in summary}
    ratios = []
    for phase in ("prefill", "decode"):
        for layer in range(memory.layer_num):
            reconstruct = times.get((phase, layer, "reconstruct"), 0.0)
            attention = times.get((phase, layer, "attention_kernel"), 0.0)
            compute = (times.get((phase, layer, "attention_sublayer"), 0.0)
                       + times.get((phase, layer, "ffn"), 0.0))
            if compute:
                ratios.append(dict(phase=phase, layer=layer,
                                   reconstruct_over_attention=reconstruct / attention if attention else None,
                                   reconstruct_over_layer_compute=reconstruct / compute))
    element_bytes = torch.empty((), dtype=memory.dtype).element_size()
    prefix_bytes = (args.unique_loras * args.prefix_len * memory.head_num * memory.head_dim
                    * 2 * element_bytes * (memory.layer_num if args.mode in FULL_PREFIX_CACHE_MODES else 1))
    prefix_backing_bytes = (prefix_bytes if args.mode in FULL_PREFIX_CACHE_MODES else
                            memory.peak_workspace_token_equiv * memory.layer_num
                            * memory.head_num * memory.head_dim * 2 * element_bytes)
    if args.mode == LAYER_RETAINED:
        retained = args.retain_layers
        cell_bytes = memory.head_num * memory.head_dim * 2 * element_bytes
        top_k = getattr(args, "retain_top_k", None)
        if getattr(args, "pin_hot_adapters", 0):
            top_k = args.pin_hot_adapters
        hot = args.unique_loras if top_k is None else min(args.unique_loras, top_k)
        scratch_count = args.unique_loras if retained < memory.layer_num else args.unique_loras - hot
        prefix_bytes = args.prefix_len * (hot * retained + scratch_count) * cell_bytes
        retained_slots = hot * ((args.prefix_len * retained + memory.layer_num - 1) // memory.layer_num)
        prefix_backing_bytes += retained_slots * memory.layer_num * cell_bytes
    return dict(mode=args.mode, prefix_source=("ordinary_lora_prefill" if args.mode in FULL_REPREFILL_MODES
                                             else "mini_kv_reconstruction"),
                model_config=config, smoke=args.smoke,
                config={key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()},
                torch_version=torch.__version__, triton_version=__import__("triton").__version__,
                gpu=torch.cuda.get_device_name(), measurements=measurements,
                means={key: sum(row[key] for row in measurements) / len(measurements)
                       for key in measurements[0]}, layer_summary=summary, layer_ratios=ratios, intervals=intervals,
                prefix_payload_bytes=prefix_bytes, prefix_backing_bytes=prefix_backing_bytes,
                workspace_peak_tokens=memory.peak_workspace_tokens,
                workspace_peak_token_equivalent=memory.peak_workspace_token_equiv,
                dynamic_backing_bytes=memory.get_memory_size(),
                peak_allocated_bytes=allocated_peak, peak_reserved_bytes=reserved_peak,
                **model_memory,
                peak_allocated_without_model_weights_bytes=(
                    allocated_peak - model_memory["base_model_weight_bytes"]),
                peak_allocated_without_base_model_bytes=(
                    allocated_peak - model_memory["base_model_resident_bytes"]),
                private_slots_after_cleanup=memory.tot_size - memory.can_use_mem_size,
                expected_adapter_slots=4 * args.unique_loras * args.rank,
                decode_steps=args.output_len - 1, output_ids=output_ids,
                measurement_scope="fixed workload, minimal backing; includes sampling and metadata; excludes adapter H2D")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "all":
        results = []
        for mode in (FULL_REPREFILL, LAYER_SERIAL, "layer_prefetch"):
            path = args.output.with_name(args.output.stem + "_" + mode + ".json")
            forwarded = []
            source = iter(sys.argv[1:])
            for argument in source:
                if argument in ("--mode", "--output"):
                    next(source)
                elif argument.startswith("--mode=") or argument.startswith("--output="):
                    continue
                else:
                    forwarded.append(argument)
            subprocess.run([sys.executable, __file__, *forwarded, "--mode", mode,
                            "--output", str(path)], check=True)
            results.append(json.loads(path.read_text()))
        result = dict(config={key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()}, results=results,
                      layer_output_ids_match=results[1]["output_ids"] == results[2]["output_ids"],
                      full_output_comparison="not equivalent to zero/synthetic BaseKV and MiniKV")
    else:
        result = profile_one(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Profile written to %s" % args.output)


if __name__ == "__main__":
    main()
