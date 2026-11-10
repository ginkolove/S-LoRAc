"""Sequential Base/Full/MiniKV accuracy with native Transformers TP=2."""

import argparse
from contextlib import ExitStack
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import peft
from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .data.prepare_alpaca_cleaned import prepare as prepare_alpaca_cleaned
    from .data.prepare_alpaca_mt import LANGUAGES, prepare_for_tp as prepare_alpaca_mt
    from .eval_minikv import (AccuracyProgress, MODES, PartialResults, capture_minikv, check_cache,
                             forward_tokens, fresh_cache, json_default, load_samples,
                             load_scorer, prepare_inputs, prepare_records, reconstruct_prefix,
                             run_mode, snapshot_cache, tensor_bytes)
else:
    from data.prepare_alpaca_cleaned import prepare as prepare_alpaca_cleaned
    from data.prepare_alpaca_mt import LANGUAGES, prepare_for_tp as prepare_alpaca_mt
    from eval_minikv import (AccuracyProgress, MODES, PartialResults, capture_minikv, check_cache,
                            forward_tokens, fresh_cache, json_default, load_samples,
                            load_scorer, prepare_inputs, prepare_records, reconstruct_prefix,
                            run_mode, snapshot_cache, tensor_bytes)


COLUMN = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
ROW = {"o_proj", "down_proj"}


def complete_file(path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing or empty checkpoint file: {path}")
    if Path(str(path) + ".aria2").exists() or Path(str(path) + ".incomplete").exists():
        raise ValueError(f"Unfinished download marker for {path}; finish or verify this file first")


def checkpoint_files(directory, weight_format, adapter=False):
    root = Path(directory)
    errors = []
    for kind in (("safetensors", "bin") if weight_format == "auto" else (weight_format,)):
        try:
            if adapter:
                files = [root / ("adapter_model.safetensors" if kind == "safetensors" else "adapter_model.bin")]
            else:
                base = "model.safetensors" if kind == "safetensors" else "pytorch_model.bin"
                index = root / (base + ".index.json")
                if index.exists():
                    complete_file(index)
                    names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
                    if not names or any(Path(name).name != name for name in names):
                        raise ValueError(f"Invalid shard index: {index}")
                    files = [root / name for name in names]
                else:
                    files = [root / base]
            for path in files:
                complete_file(path)
            return kind, files
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append(str(error))
    raise ValueError("\n".join(errors))


def validate_adapter(config):
    if (config.peft_type != "LORA" or config.bias != "none" or config.modules_to_save or
            getattr(config, "use_dora", False) or getattr(config, "fan_in_fan_out", False) or
            getattr(config, "lora_bias", False)):
        raise ValueError("TP evaluation requires standard additive LoRA without base-weight/bias changes")
    if not isinstance(config.target_modules, (set, list, tuple)) or not set(config.target_modules) <= COLUMN | ROW:
        raise ValueError("TP LoRA targets must be Llama attention/MLP linear projection names")
    if config.rank_pattern or config.layers_to_transform is not None or config.layer_replication:
        raise ValueError("TP checkpoint validation currently requires a uniform-rank adapter on all layers")


def llama_weight_shapes(config):
    h, inter = config.hidden_size, config.intermediate_size
    d = getattr(config, "head_dim", None) or h // config.num_attention_heads
    projections = dict(q_proj=(config.num_attention_heads*d, h),
                       k_proj=(config.num_key_value_heads*d, h),
                       v_proj=(config.num_key_value_heads*d, h),
                       o_proj=(h, config.num_attention_heads*d), gate_proj=(inter, h),
                       up_proj=(inter, h), down_proj=(h, inter))
    shapes = {"model.embed_tokens.weight": (config.vocab_size, h),
              "lm_head.weight": (config.vocab_size, h), "model.norm.weight": (h,)}
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[prefix + norm + ".weight"] = (h,)
        for name, shape in projections.items():
            group = "self_attn" if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            shapes[prefix + group + "." + name + ".weight"] = shape
            if getattr(config, "attention_bias" if group == "self_attn" else "mlp_bias", False):
                shapes[prefix + group + "." + name + ".bias"] = (shape[0],)
    return shapes


def validate_checkpoint_shapes(config, adapter_config, files, formats):
    expected_model = llama_weight_shapes(config)
    expected_adapter = {}
    for key, (out_size, in_size) in ((key, shape) for key, shape in expected_model.items()
                                    if key.endswith("_proj.weight")):
        if key.rsplit(".", 2)[-2] in adapter_config.target_modules:
            prefix = "base_model.model." + key.removesuffix(".weight")
            expected_adapter[prefix + ".lora_A.weight"] = (adapter_config.r, in_size)
            expected_adapter[prefix + ".lora_B.weight"] = (out_size, adapter_config.r)
    report = {}
    for kind, expected in (("model", expected_model), ("adapter", expected_adapter)):
        if formats[kind] != "safetensors" and kind == "model":
            report[kind] = dict(validation="Shapes checked by weight loader, not header preflight")
            continue
        shapes, locations = {}, {}
        if formats[kind] == "safetensors":
            for entry in files[kind]:
                with safe_open(entry["path"], framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key in shapes:
                            raise ValueError(f"Duplicate {kind} tensor across shards: {key}")
                        shapes[key] = tuple(handle.get_slice(key).get_shape())
                        locations[key] = Path(entry["path"]).name
        else:
            state = torch.load(files[kind][0]["path"], map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError("Adapter .bin must contain a tensor state dictionary")
            shapes = {key: tuple(value.shape) for key, value in state.items()}
            del state
        legacy = {}
        if kind == "model":
            d = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            # Original Llama checkpoints persisted layer-local RoPE buffers.
            legacy = {f"model.layers.{i}.self_attn.rotary_emb.inv_freq": (d//2,)
                      for i in range(config.num_hidden_layers) if
                      f"model.layers.{i}.self_attn.rotary_emb.inv_freq" in shapes}
            expected = dict(expected, **legacy)
            if config.tie_word_embeddings and "lm_head.weight" not in shapes:
                expected.pop("lm_head.weight")
            index_path = Path(files[kind][0]["path"]).parent / "model.safetensors.index.json"
            if index_path.exists() and json.loads(index_path.read_text())["weight_map"] != locations:
                raise ValueError("Safetensors shard contents do not match the model index")
        if shapes != expected:
            missing, extra = sorted(expected.keys()-shapes.keys()), sorted(shapes.keys()-expected.keys())
            bad = [(key, shapes[key], expected[key]) for key in shapes.keys() & expected.keys()
                   if shapes[key] != expected[key]]
            raise ValueError(f"{kind} checkpoint does not match configuration: missing={missing[:2]}, "
                             f"extra={extra[:2]}, wrong_shapes={bad[:2]}")
        report[kind] = dict(validation="All tensor names and shapes match configuration",
                            tensors=len(shapes), parameters=sum(math.prod(v) for k,v in shapes.items() if k not in legacy),
                            legacy_rope_buffers=len(legacy))
    return report


def shard_lora_weight(value, kind, global_shape, shard_dim, rank, world_size):
    output_size, input_size = global_shape
    if value.ndim != 2 or (kind == "A" and value.shape[1] != input_size) or (
            kind == "B" and value.shape[0] != output_size):
        raise ValueError(f"Adapter/base dimension mismatch: {tuple(value.shape)} vs {global_shape}")
    axis = 1 if shard_dim == 1 and kind == "A" else 0 if shard_dim == 0 and kind == "B" else None
    if axis is None:
        return value
    if value.shape[axis] % world_size:
        raise ValueError("LoRA projection cannot be evenly tensor-sharded")
    width = value.shape[axis] // world_size
    return value.narrow(axis, rank * width, width).contiguous()


def reduce_lora_a(module, inputs, output):
    # Row-parallel A has partial rank-space activations; B needs their sum.
    output = output.contiguous()
    dist.all_reduce(output, group=module._lowra_tp_group)
    return output


def attach_tp_lora(base, adapter_path, adapter_format):
    config = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
    validate_adapter(config)
    model = get_peft_model(base, config, autocast_adapter_dtype=True).eval()
    modules = {name: module for name, module in model.named_modules() if hasattr(module, "lora_A")}
    if not modules:
        raise ValueError("No LoRA projections were installed")
    root = Path(adapter_path)
    filename = root / ("adapter_model.safetensors" if adapter_format == "safetensors" else "adapter_model.bin")
    state = (load_file(str(filename), device="cpu") if adapter_format == "safetensors" else
             torch.load(filename, map_location="cpu", weights_only=True))
    expected = {f"{name}.lora_{kind}.weight" for name in modules for kind in ("A", "B")}
    if set(state) != expected:
        raise ValueError(f"Adapter checkpoint keys differ: missing={sorted(expected-set(state))[:3]}, "
                         f"unexpected={sorted(set(state)-expected)[:3]}")
    local_state, layouts = {}, []
    for name, module in modules.items():
        weight = module.get_base_layer().weight
        if not isinstance(weight, DTensor) or len(weight.placements) != 1 or not isinstance(weight.placements[0], Shard):
            raise ValueError(f"LoRA target is not tensor-sharded: {name}")
        shard_dim = weight.placements[0].dim % 2
        expected_dim = 0 if name.rsplit(".", 1)[-1] in COLUMN else 1
        if shard_dim != expected_dim:
            raise ValueError(f"Unexpected TP layout on {name}")
        mesh = weight.device_mesh
        for kind in ("A", "B"):
            key = f"{name}.lora_{kind}.weight"
            local_state[key] = shard_lora_weight(state[key], kind, tuple(weight.shape), shard_dim,
                                                 mesh.get_local_rank(), mesh.size())
        if shard_dim == 1:
            a = module.lora_A["default"]
            a._lowra_tp_group = mesh.get_group()
            a.register_forward_hook(reduce_lora_a)
        layouts.append(dict(name=name, shard_dim=shard_dim, global_shape=list(weight.shape),
                            local_shape=list(weight.to_local().shape)))
    loaded = set_peft_model_state_dict(model, local_state)
    missing = [name for name in loaded.missing_keys if ".lora_" in name]
    if missing or loaded.unexpected_keys:
        raise ValueError(f"Adapter loading failed: {missing}, {loaded.unexpected_keys}")
    del state, local_state
    gc.collect()
    return model.eval(), layouts


def validate_tp_model(model, world_size):
    shapes = []
    for name, module in model.named_modules():
        target = name.rsplit(".", 1)[-1]
        if target not in COLUMN | ROW:
            continue
        weight = module.weight
        dimension = 0 if target in COLUMN else 1
        if (not isinstance(weight, DTensor) or len(weight.placements) != 1 or
                not isinstance(weight.placements[0], Shard) or weight.placements[0].dim % 2 != dimension or
                weight.shape[dimension] != weight.to_local().shape[dimension] * world_size):
            raise ValueError(f"Incorrect tensor parallel projection: {name}")
        shapes.append(dict(name=name, global_shape=list(weight.shape), local_shape=list(weight.to_local().shape)))
    if not shapes:
        raise ValueError("Model has no sharded Llama projections")
    return shapes


def validate_prefix(pairs, config, world_size, prefix_length):
    heads = config.num_key_value_heads // world_size
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if len(pairs) != config.num_hidden_layers or any(
            tuple(value.shape) != (1, heads, prefix_length, dim) for pair in pairs for value in pair):
        raise ValueError("TP prefix cache has incorrect local head/layer/sequence dimensions")


def checkpoint_preflight(args):
    errors, formats, files = [], {}, {}
    for field, path, adapter in (("model", args.model, False), ("adapter", args.lora, True)):
        try:
            formats[field], values = checkpoint_files(path, getattr(args, field + "_weight_format"), adapter)
            files[field] = [dict(path=str(p), bytes=p.stat().st_size) for p in values]
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError("Checkpoint preflight failed:\n" + "\n".join(errors))
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "llama" or config.num_attention_heads % 2 or config.num_key_value_heads % 2:
        raise ValueError("TP=2 requires Llama with query and KV head counts divisible by two")
    adapter_config = PeftConfig.from_pretrained(args.lora, local_files_only=True)
    validate_adapter(adapter_config)
    headers = validate_checkpoint_shapes(config, adapter_config, files, formats)
    destination = Path(args.output)
    if any(p.exists() for p in (destination, destination.with_suffix(".partial.json"),
                                destination.with_suffix(".predictions.jsonl"))):
        raise FileExistsError("Use a new --output path; resume/overwrite is not supported")
    return config, adapter_config, formats, files, headers


def load_evaluation_data(args, config):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, local_files_only=True)
    samples = load_samples(args.data, args.max_samples)
    manifest = Path(str(args.data) + ".meta.json")
    if manifest.exists():
        expected_hash = json.loads(manifest.read_text()).get("output_sha256")
        if expected_hash and hashlib.sha256(Path(args.data).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Prepared dataset differs from its saved sampling manifest")
    limits = [int(v) for v in (getattr(config, "max_position_embeddings", None),
                               getattr(config, "max_sequence_length", None)) if v]
    if not limits:
        raise ValueError("Model configuration does not declare a context limit")
    prepared = prepare_inputs(samples, tokenizer, args.max_new_tokens, min(limits))
    if len({item["prefix_key"] for item in prepared}) != 1:
        raise ValueError("This experiment requires one fixed effective token prefix")
    if any(max(item["full_ids"]) >= config.vocab_size for item in prepared):
        raise ValueError("Tokenizer IDs exceed model vocabulary")
    return tokenizer, samples, prepared


def preflight(args):
    config, adapter_config, formats, files, headers = checkpoint_preflight(args)
    tokenizer, samples, prepared = load_evaluation_data(args, config)
    return config, adapter_config, tokenizer, samples, prepared, formats, files, headers


def prepare_runtime_data(args, rank):
    if not (args.prepare_alpaca_cleaned or args.prepare_alpaca_mt):
        return
    error = [None]
    if rank == 0:
        print(json.dumps(dict(state="preparing_data", dataset=args.data)), flush=True)
        try:
            prepare = prepare_alpaca_mt if args.prepare_alpaca_mt else prepare_alpaca_cleaned
            prepare(args, args.data)
        except Exception as exception:
            error[0] = f"{type(exception).__name__}: {exception}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"{args.dataset_name} preparation failed: {error[0]}")


def stage_prefix(model, ids, config):
    output = forward_tokens(model, ids, fresh_cache(model))
    pairs = snapshot_cache(output.past_key_values, device=None)
    del output
    validate_prefix(pairs, config, dist.get_world_size(), len(ids))
    return pairs


def run(args):
    if args.check_only:
        config, adapter_config, tokenizer, samples, prepared, formats, files, headers = preflight(args)
        print(json.dumps(dict(checkpoint_preflight="passed", formats=formats, files=files, checkpoint_headers=headers,
                              samples=len(samples), prefix_tokens=len(prepared[0]["prefix_ids"]),
                              max_new_tokens=args.max_new_tokens, tp_size=2), indent=2))
        return
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise ValueError("Launch with torchrun --standalone --nproc_per_node=2")
    config, adapter_config, formats, files, headers = checkpoint_preflight(args)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        scorer, checks, stats = load_scorer(args.scorer), [], {}
        started = time.perf_counter()
        # Keep load/preparation failures visible before the sample count is known.
        samples = []
        with torch.inference_mode(), ExitStack() as stack:
            writer = stack.enter_context(PartialResults(args.output, samples, scorer)) if rank == 0 else None
            if rank == 0:
                print(json.dumps(dict(state="loading_model", tp_size=2)), flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, config=config, dtype=getattr(torch, args.dtype),
                tp_plan="auto", device_mesh=mesh, use_safetensors=formats["model"] == "safetensors",
                attn_implementation=args.attn_implementation, local_files_only=True).eval()
            shapes = validate_tp_model(model, 2)
            dist.barrier()
            if rank == 0:
                print(json.dumps(dict(state="model_loaded", tp_size=2)), flush=True)
            prepare_runtime_data(args, rank)
            tokenizer, loaded_samples, prepared = load_evaluation_data(args, config)
            samples.extend(loaded_samples)
            fingerprint = hashlib.sha256(json.dumps([(p["sample"]["id"], p["full_ids"], p["prefix_key"])
                                                     for p in prepared]).encode()).hexdigest()
            fingerprints = [None, None]
            dist.all_gather_object(fingerprints, fingerprint)
            if len(set(fingerprints)) != 1:
                raise ValueError("TP ranks prepared different inputs")
            groups, records = prepare_records(prepared)
            items = next(iter(groups.values()))
            ids, first = items[0][1]["prefix_ids"], items[0][1]
            if writer:
                writer.write_summary()
                print(json.dumps(dict(state="data_loaded", samples=len(samples), prefix_tokens=len(ids))), flush=True)
            progress = AccuracyProgress("base", len(samples), 0) if rank == 0 else None
            if progress:
                stack.callback(progress.close)
            eos = model.generation_config.eos_token_id
            eos = tokenizer.eos_token_id if eos is None else eos
            eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
            if rank == 0:
                print("Model and data ready; starting Base, LoRA Full, then LoRA MiniKV", flush=True)
            pairs = stage_prefix(model, ids, config)
            checks.append(dict(mode="base", **check_cache(model, first, pairs,
                args.cache_check_atol, args.cache_check_rtol)))
            for mode in MODES:
                if mode == "lora_full":
                    model, layouts = attach_tp_lora(model, args.lora, formats["adapter"])
                    pairs = stage_prefix(model, ids, config)
                    checks.append(dict(mode=mode, **check_cache(model, first, pairs,
                        args.cache_check_atol, args.cache_check_rtol)))
                elif mode == "lora_minikv":
                    with model.disable_adapter():
                        bases = stage_prefix(model, ids, config)
                    with capture_minikv(model, keep_on_device=True) as minis:
                        output = forward_tokens(model, ids, fresh_cache(model))
                    del output
                    mini_bytes = tensor_bytes(minis)
                    pairs = reconstruct_prefix(model, bases, minis, keep_on_device=True)
                    validate_prefix(pairs, config, 2, len(ids))
                    del bases, minis
                if progress:
                    progress.close()
                    progress = AccuracyProgress(mode, len(samples), 0)
                    stack.callback(progress.close)
                torch.cuda.synchronize(device)
                stage_started = time.perf_counter()
                run_mode(model, items, records, pairs, mode, tokenizer, args, eos_ids,
                         progress=progress, writer=writer)
                torch.cuda.synchronize(device)
                stats[mode] = dict(generation_seconds=time.perf_counter()-stage_started,
                                   local_prefix_bytes=tensor_bytes(pairs),
                                   prefix_preparations=2 if mode == "lora_minikv" else 1)
                del pairs
                gc.collect()
                torch.cuda.empty_cache()
            stats["local_minikv_with_rope_bytes"] = mini_bytes
            stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            ranks = [None, None]
            dist.all_gather_object(ranks, stats)
            if rank == 0:
                scores = {mode: scorer([r["predictions"][mode]["text"] for r in records],
                           [s["references"] for s in samples], samples) for mode in MODES}
                result = dict(config=vars(args), versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                    peft=peft.__version__), model_config=config.to_dict(), adapter_config=adapter_config.to_dict(),
                    experiment=dict(samples=len(samples), unique_prefixes=1, tp_size=2,
                        dataset=args.dataset_name,
                        startup_order="model_then_data",
                        schedule="sequential Base / Full / MiniKV; one TP model, no replicated GPU workers",
                        elapsed_seconds=time.perf_counter()-started, prefix_tokens=len(ids),
                        data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                        decoding="greedy argmax, EOS or max_new_tokens; no logits processors",
                        minikv_source="Adapted prefix K/V LoRA A outputs; locally sharded B plus native BaseKV",
                        score_protocol=("Custom scorer: " + args.scorer if args.scorer else "English EM/token F1")
                                       + "; full generated output; no answer-boundary filtering",
                        input_fingerprint=fingerprint, ranks=ranks, tp_projection_shapes=shapes,
                        lora_tp_layouts=layouts), cache_checks=checks, checkpoint_files=files,
                    checkpoint_headers=headers, scores=scores, score_deltas={name: {
                        metric: value - scores[right][metric] for metric, value in scores[left].items()
                        if isinstance(value, (int, float)) and isinstance(scores[right].get(metric), (int, float))}
                        for name,left,right in (("lora_minus_base","lora_full","base"),
                                                ("minikv_minus_lora","lora_minikv","lora_full"))}, predictions=records)
                path = Path(args.output)
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
                    handle.write("\n")
                print(json.dumps(dict(samples=len(samples), scores=scores, output=str(path)), indent=2), flush=True)
        dist.barrier()
    except BaseException:
        # Graceful NCCL destruction can hang while a healthy peer waits for this failed rank.
        # PartialResults has already persisted the error while unwinding the evaluation context.
        try:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(1)
    else:
        dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/hub/Llama-65B")
    parser.add_argument("--lora", default="/data/hub/guanaco-lora/65b")
    parser.add_argument("--data", default="accexp/results/guanaco_local_3shot_random100_seed42.jsonl")
    parser.add_argument("--dataset-name", default="guanaco", help="Dataset label in results; does not change scoring")
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare-alpaca-cleaned", action="store_true",
                             help="Rank 0 prepares raw Alpaca-Cleaned after both TP model shards have loaded")
    preparation.add_argument("--prepare-alpaca-mt", action="store_true",
                             help="Rank 0 prepares fixed-English-prefix Alpaca-MT after loading the TP model")
    parser.add_argument("--dataset-dir", default="/data/hub/alpaca_cleaned")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--prefix-max-tokens", type=int, default=1024)
    parser.add_argument("--english-file", default="accexp/results/alpaca_mt_en_raw.json")
    parser.add_argument("--prefix-data", default="accexp/results/alpaca_mt_en_3shot_all_seed42.jsonl")
    parser.add_argument("--raw-dir", default="accexp/results/alpaca_mt_raw")
    parser.add_argument("--template-file", default="/data/hub/alpaca_mt/alpaca_mt.py")
    parser.add_argument("--languages", nargs="+", choices=tuple(LANGUAGES), default=["pt", "es", "ca"])
    parser.add_argument("--samples-per-language", type=int, default=100)
    parser.add_argument("--tokenizer", help="Default: model directory; original Llama tokenizer may be supplied separately")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-weight-format", choices=("auto", "safetensors", "bin"), default="safetensors")
    parser.add_argument("--adapter-weight-format", choices=("auto", "safetensors", "bin"), default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scorer", help="Default: same English EM/token F1 as prior Guanaco runs")
    parser.add_argument("--cache-check-atol", type=float)
    parser.add_argument("--cache-check-rtol", type=float, default=0.005)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without loading weights or initializing TP")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_samples < 0:
        parser.error("batch/output lengths must be positive and max-samples nonnegative")
    if args.cache_check_rtol < 0 or (args.cache_check_atol is not None and args.cache_check_atol < 0):
        parser.error("Cache tolerances must be nonnegative")
    if args.prepare_alpaca_cleaned:
        if args.dataset_name != "alpaca_cleaned" or args.check_only:
            parser.error("Runtime Alpaca preparation needs --dataset-name alpaca_cleaned and cannot use --check-only")
        if args.sample_size < 0 or min(args.num_shots, args.prefix_max_tokens) < 1:
            parser.error("sample-size must be nonnegative; num-shots and prefix-max-tokens must be positive")
    if args.prepare_alpaca_mt:
        if args.dataset_name != "alpaca_mt" or args.check_only:
            parser.error("Runtime MT preparation needs --dataset-name alpaca_mt and cannot use --check-only")
        if args.samples_per_language < 1 or len(set(args.languages)) != len(args.languages):
            parser.error("samples-per-language must be positive and languages must not repeat")
    return args


if __name__ == "__main__":
    run(parse_args())
