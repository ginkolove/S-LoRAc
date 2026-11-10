"""Measure unquantized Transformers KV storage without loading model weights."""

import argparse
import copy
import gc
import json
import math
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, DynamicCache


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def cache_geometry(config, seq_len, batch_size, dtype):
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    config = config.get_text_config(decoder=True)
    if (getattr(config, "sliding_window", None)
            or getattr(config, "attention_chunk_size", None)
            or getattr(config, "num_kv_shared_layers", 0)):
        raise ValueError("This probe requires full-context, non-shared KV layers")
    layer_types = getattr(config, "layer_types", None)
    if layer_types and any(kind != "full_attention" for kind in layer_types):
        raise ValueError("This probe does not estimate sliding or hybrid cache layers")
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        if config.hidden_size % heads:
            raise ValueError("hidden_size is not divisible by num_attention_heads")
        head_dim = config.hidden_size // heads
    layers = int(config.num_hidden_layers)
    if min(heads, kv_heads, head_dim, layers) <= 0:
        raise ValueError("Invalid cache dimensions")
    element_bytes = torch.empty((), dtype=dtype).element_size()
    per_layer = 2 * batch_size * kv_heads * seq_len * head_dim * element_bytes
    return dict(
        hidden_size=int(config.hidden_size), num_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=int(head_dim),
        shape_per_k_or_v=[batch_size, kv_heads, seq_len, int(head_dim)],
        element_bytes=element_bytes, per_layer_kv_bytes=per_layer,
        full_kv_bytes=per_layer * layers,
        full_kv_gib=per_layer * layers / 2**30,
        full_kv_gb=per_layer * layers / 10**9,
        per_layer_kv_mib=per_layer / 2**20,
    )


@torch.inference_mode()
def measure_cache(config, geometry, dtype, device, mode, min_free_gib):
    layer_count = geometry["num_layers"] if mode == "full" else 1
    expected = layer_count * geometry["per_layer_kv_bytes"]
    cuda = device.type == "cuda"
    if cuda:
        if not torch.cuda.is_available():
            return dict(status="skipped", reason="CUDA is unavailable")
        torch.cuda.set_device(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        # DynamicCache.update may copy its input K/V before the inputs are released.
        required = expected + geometry["per_layer_kv_bytes"] + 64 * 2**20
        if free < required + min_free_gib * 2**30:
            return dict(status="skipped", reason="Insufficient free VRAM with safety margin",
                        free_bytes=free, estimated_build_peak_bytes=required,
                        safety_margin_gib=min_free_gib)
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

    cache_config = copy.deepcopy(config.get_text_config(decoder=True))
    cache_config.num_hidden_layers = layer_count
    if getattr(cache_config, "layer_types", None) is not None:
        cache_config.layer_types = cache_config.layer_types[:layer_count]
    cache = DynamicCache(config=cache_config)
    shape = geometry["shape_per_k_or_v"]
    for layer in range(layer_count):
        keys = torch.zeros(shape, dtype=dtype, device=device)
        values = torch.zeros_like(keys)
        cache.update(keys, values, layer)
        del keys, values
    if cuda:
        torch.cuda.synchronize(device)

    storages = {}
    for index, layer in enumerate(cache.layers):
        assert cache.get_seq_length(index) == shape[-2]
        for tensor in (layer.keys, layer.values):
            assert list(tensor.shape) == shape
            assert tensor.dtype == dtype
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    measured_bytes = sum(storages.values())
    assert measured_bytes == expected, (measured_bytes, expected)
    result = dict(status="measured", mode=mode, layers_measured=layer_count,
                  measured_kv_storage_bytes=measured_bytes,
                  measured_kv_storage_gib=measured_bytes / 2**30,
                  full_cache_actually_allocated=layer_count == geometry["num_layers"],
                  projected_full_kv_bytes=measured_bytes * geometry["num_layers"] // layer_count,
                  storage_matches_formula=True)
    if cuda:
        result.update(
            gpu_name=torch.cuda.get_device_name(device),
            free_bytes_before=free, total_device_bytes=total,
            allocated_delta_bytes=torch.cuda.memory_allocated(device) - before_allocated,
            reserved_delta_bytes=torch.cuda.memory_reserved(device) - before_reserved,
            build_peak_allocated_delta_bytes=torch.cuda.max_memory_allocated(device) - before_allocated,
        )
    del cache, layer, tensor, storage, storages
    gc.collect()
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        result["allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated(device) - before_allocated
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["huggyllama/llama-7b",
                        "huggyllama/llama-13b", "huggyllama/llama-30b"])
    parser.add_argument("--seq-len", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("layer", "full", "estimate"), default="layer")
    parser.add_argument("--min-free-gib", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error("--min-free-gib must be finite and nonnegative")
    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        parser.error("--device must be a CUDA or CPU device")
    report = dict(
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        seq_len=args.seq_len, batch_size=args.batch_size, dtype=args.dtype,
        device=str(device), mode=args.mode, tp_size=1,
        note="Cache-storage probe only: no model forward, weights, activations, MiniKV, or paging pool. "
             "This does not validate long-context model inference. Build peak is probe-specific.",
        results=[],
    )
    for model in args.models:
        print("Checking %s (%s)" % (model, args.mode), flush=True)
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        geometry = cache_geometry(config, args.seq_len, args.batch_size, dtype)
        row = dict(model=model, geometry=geometry)
        row["measurement"] = (dict(status="not_requested") if args.mode == "estimate" else
                              measure_cache(config, geometry, dtype, device, args.mode, args.min_free_gib))
        report["results"].append(row)
        print(json.dumps(row, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Results: %s" % args.output.resolve())


if __name__ == "__main__":
    main()
