"""Real-checkpoint E2E audit; HF reference and S-LoRA run in their own environments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reference", "serving"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=[129, 513])
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-rope", action="store_true")
    return parser.parse_args()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def reference(args):
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from peft import PeftModel

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager",
        local_files_only=True).cuda().eval()
    config = model.config
    hidden, layers, heads = config.hidden_size, config.num_hidden_layers, config.num_attention_heads
    dim = hidden // heads
    if getattr(config, "num_key_value_heads", heads) != heads:
        raise ValueError("this serving integration audit currently requires MHA")
    adapter_configs = [json.loads((Path(directory) / "adapter_config.json").read_text()) for directory in args.adapters]
    for cfg in adapter_configs:
        if not set(cfg["target_modules"]).issubset({"q_proj", "k_proj", "v_proj", "o_proj"}):
            raise ValueError("S-LoRA's audited path does not implement FFN-target adapters")
    prefix_text = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                   "### Instruction:\nWhat is the capital of France?\n\n### Response:\nParis.\n\n"
                   "### Instruction:\nTranslate hello into Spanish.\n\n### Response:\nHola.\n\n")
    prefix_pool = tokenizer.encode(prefix_text * 100, add_special_tokens=True,
                                   truncation=True, max_length=max(args.prefix_lengths))
    queries = [tokenizer.encode(text, add_special_tokens=False) for text in (
        "### Instruction:\nWhat is two plus three?\n\n### Response:\n",
        "### Instruction:\nTranslate the English word cat into Spanish.\n\n### Response:\n",
        "### Instruction:\nName three planets in our solar system.\n\n### Response:\n")]
    request_adapters = [0, min(1, len(args.adapters) - 1), 0]

    def legacy_cpu(cache):
        return [(k.detach().cpu().clone(), v.detach().cpu().clone()) for k, v in cache.to_legacy_cache()]

    def from_cpu(cache):
        return DynamicCache.from_legacy_cache(tuple((k.cuda().clone(), v.cuda().clone()) for k, v in cache))

    @torch.no_grad()
    def evaluate(tokens, prefix_cache):
        cache = from_cpu(prefix_cache) if prefix_cache is not None else None
        inputs = torch.tensor([tokens], device="cuda")
        logits, generated = [], []
        for _ in range(args.output_len):
            output = model(input_ids=inputs, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            last = output.logits[:, -1].float()
            token = int(last.argmax(-1).item())
            logits.append(last[0].cpu())
            generated.append(token)
            inputs = torch.tensor([[token]], device="cuda")
        return dict(logits=torch.stack(logits), generated=generated,
                    text=tokenizer.decode(generated, skip_special_tokens=True))

    for prefix_len in args.prefix_lengths:
        print("HF reference: prefix=%s" % prefix_len, flush=True)
        prefix_ids = prefix_pool[:prefix_len]
        if len(prefix_ids) != prefix_len or prefix_len + max(map(len, queries)) + args.output_len > config.max_position_embeddings:
            raise ValueError("audit input exceeds the pretrained context or prefix source")
        if isinstance(model, PeftModel):
            with model.disable_adapter(), torch.inference_mode():
                base_output = model(input_ids=torch.tensor([prefix_ids], device="cuda"), use_cache=True)
        else:
            with torch.inference_mode():
                base_output = model(input_ids=torch.tensor([prefix_ids], device="cuda"), use_cache=True)
        base_cache = legacy_cpu(base_output.past_key_values)
        del base_output
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, args.adapters[0], adapter_name="audit0", is_trainable=False,
                                               autocast_adapter_dtype=False).eval()
            for index, directory in enumerate(args.adapters[1:], 1):
                model.load_adapter(directory, adapter_name="audit%s" % index, is_trainable=False,
                                   autocast_adapter_dtype=False)
            model.half()
        rotary = model.base_model.model.model.rotary_emb
        with torch.inference_mode():
            cos, sin = rotary(torch.empty(1, prefix_len, hidden, device="cuda", dtype=torch.float16),
                              torch.arange(prefix_len, device="cuda")[None])
        cos, sin = cos[0, :, :dim // 2].cpu(), sin[0, :, :dim // 2].cpu()
        sources = []
        for adapter_index, cfg in enumerate(adapter_configs):
            name = "audit%s" % adapter_index
            model.set_adapter(name)
            rank = cfg["r"]
            mini = [(torch.zeros(prefix_len, rank, dtype=torch.float16),
                     torch.zeros(prefix_len, rank, dtype=torch.float16)) for _ in range(layers)]
            b_weights = []
            handles = []
            for layer_id, layer in enumerate(model.base_model.model.model.layers):
                pair = []
                for module_index, module in enumerate((layer.self_attn.k_proj, layer.self_attn.v_proj)):
                    if hasattr(module, "lora_A") and name in module.lora_A:
                        def capture(_module, _inputs, output, li=layer_id, mi=module_index):
                            mini[li][mi].copy_(output[0].detach().cpu())
                        handles.append(module.lora_A[name].register_forward_hook(capture))
                        pair.append(module.lora_B[name].weight.detach().cpu().half().clone())
                    else:
                        pair.append(torch.zeros(hidden, rank, dtype=torch.float16))
                b_weights.append(pair)
            with torch.inference_mode():
                output = model(input_ids=torch.tensor([prefix_ids], device="cuda"), use_cache=True)
            full_cache = legacy_cpu(output.past_key_values)
            del output
            for handle in handles:
                handle.remove()
            rebuilt = []
            scale = float(cfg["lora_alpha"]) / rank
            for layer_id in range(layers):
                bk, bv = base_cache[layer_id]
                dk = (mini[layer_id][0].float() @ b_weights[layer_id][0].float().T * scale).half().view(prefix_len, heads, dim)
                dv = mini[layer_id][1].float() @ b_weights[layer_id][1].float().T * scale
                first, second = dk.float().chunk(2, -1)
                c, s = cos.float()[:, None], sin.float()[:, None]
                rotated = torch.cat((first * c - second * s, first * s + second * c), -1)
                k = (bk[0].transpose(0, 1).float() + rotated).half().transpose(0, 1)[None].contiguous()
                v = (bv[0].transpose(0, 1).reshape(prefix_len, hidden).float() + dv).half().view(prefix_len, heads, dim).transpose(0, 1)[None].contiguous()
                rebuilt.append((k, v))
            sources.append(dict(mini=mini, full_cache=full_cache, rebuilt_cache=rebuilt))
        cases = []
        for index, (query, adapter_index) in enumerate(zip(queries, request_adapters)):
            model.set_adapter("audit%s" % adapter_index)
            full = evaluate(query, sources[adapter_index]["full_cache"])
            rebuilt = evaluate(query, sources[adapter_index]["rebuilt_cache"])
            direct = evaluate(prefix_ids + query, None)
            cases.append(dict(request_id="r%s" % index, adapter_index=adapter_index, query=query,
                              full=full, rebuilt=rebuilt, direct=direct))
            print("HF case %s adapter=%s full=%r rebuilt=%r" % (index, adapter_index, full["text"], rebuilt["text"]), flush=True)
        bundle = dict(prefix_len=prefix_len, prefix_ids=prefix_ids, config=config.to_dict(),
                      adapter_configs=adapter_configs, base=base_cache, sources=sources, cases=cases,
                      rope_cos=cos, rope_sin=sin, output_len=args.output_len)
        torch.save(bundle, args.output_dir / ("prefix%s.pt" % prefix_len))
    save_json(args.output_dir / "reference.json", dict(model=args.model, adapters=args.adapters,
              prefix_lengths=args.prefix_lengths, output_len=args.output_len,
              torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
              base_and_adapter_dtype="float16; PEFT adapter autocast disabled",
              scope="real complete checkpoint, true LoRA-A activations; MiniKV approximation separate from runtime correctness"))


def serving(args):
    import tempfile
    from unittest.mock import patch
    import torch
    from slora.common.configs.config import setting
    from slora.common.lowra_static import LowRAStaticPrefixStore
    from slora.common.lowra_layer_runtime import LowRALayerRuntime
    from slora.common.full_reprefill import FullPrefixKVCache, FullReprefillRuntime
    from slora.models.llama.model import LlamaTpPartModel
    from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
    from slora.server.router.model_infer.infer_adapter import InferAdapter
    from slora.server.router.model_infer.model_rpc import ModelRpcServer

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    recorded = json.loads((args.output_dir / "reference.json").read_text())
    if (Path(recorded["model"]).resolve() != Path(args.model).resolve()
            or [Path(path).resolve() for path in recorded["adapters"]] != [Path(path).resolve() for path in args.adapters]
            or recorded["output_len"] != args.output_len):
        raise ValueError("serving model, adapters or output length differ from the reference")
    with tempfile.NamedTemporaryFile() as rendezvous:
        torch.distributed.init_process_group("nccl", init_method="file://" + rendezvous.name, rank=0, world_size=1)
    setting["max_req_total_len"] = max(args.prefix_lengths) + 256 + args.output_len
    print("Loading actual S-LoRA checkpoint, all layers, dummy=False", flush=True)
    model = LlamaTpPartModel(0, 1, args.model, 8192, mem_adapter_size=0, dummy=False, dynamic_layer_workspace=True)
    adapters = []
    for directory in args.adapters:
        cfg = json.loads((Path(directory) / "adapter_config.json").read_text())
        weights = torch.load(Path(directory) / "adapter_model.bin", map_location="cpu")
        root = next(iter(weights)).split("layers.")[0]
        for layer in range(model.layers_num):
            for module in ("q", "k", "v", "o"):
                for kind, shape in (("A", (cfg["r"], model.config["hidden_size"])),
                                    ("B", (model.config["hidden_size"], cfg["r"]))):
                    key = "%slayers.%s.self_attn.%s_proj.lora_%s.weight" % (root, layer, module, kind)
                    # Missing PEFT targets have exactly zero delta, not random dummy weights.
                    if key not in weights:
                        if module + "_proj" in cfg["target_modules"]:
                            raise ValueError("declared LoRA target is missing checkpoint weight: " + key)
                        weights[key] = torch.zeros(shape, dtype=torch.float16)
        layers = []
        for layer in range(model.layers_num):
            weight = LoraLayerWeight(layer, 0, 1, cfg, model.config)
            weight.load_hf_weights_cpu(weights)
            layers.append(weight)
        adapters.append(SimpleNamespace(lora_dir=directory, r=cfg["r"], scaling=cfg["lora_alpha"] / cfg["r"],
                                        network_config=model.config, layers=layers))
        del weights
    infer = InferAdapter.init(model.mem_manager, None)
    infer.load_adapters(adapters)
    server = ModelRpcServer()
    server.model, server.infer_adapter = model, infer
    server.adapters, server.adapter_id = adapters, {directory: i for i, directory in enumerate(args.adapters)}
    server.world_size, server.cache = 1, {}
    server.input_params = SimpleNamespace(no_lora=False, no_kernel=False, scheduler="slora",
                                         no_lora_compute=False, bmm=False)
    server.lowra_runtime = server.lowra_prefix_cache = None
    report = dict(model=args.model, adapters=args.adapters, key_rope=args.key_rope,
                  dummy=False, num_layers=model.layers_num, hidden_size=model.config["hidden_size"],
                  adapter_ranks=[adapter.r for adapter in adapters],
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=torch.__version__,
                  token_comparison="teacher-forced reference histories; all argmaxes must also agree",
                  source_setup="audit driver injects real HF/PEFT BaseKV and LoRA-A activations",
                  cases=[], all_passed=False)

    def difference(actual, expected, atol=0.1, rtol=0.01):
        delta = (actual - expected).abs()
        return dict(max_abs=delta.max().item(), rmse=delta.square().mean().sqrt().item(),
                    failures=int((delta > atol + rtol * expected.abs()).sum()),
                    argmax_agreement=(actual.argmax(-1) == expected.argmax(-1)).float().mean().item(),
                    finite=bool(torch.isfinite(actual).all()), atol=atol, rtol=rtol)

    for prefix_len in args.prefix_lengths:
        bundle = torch.load(args.output_dir / ("prefix%s.pt" % prefix_len), map_location="cpu")
        ranks = {adapter.lora_dir: adapter.r for adapter in adapters}
        static = LowRAStaticPrefixStore(prefix_len, args.adapters, ranks, model.layers_num,
                                        model.tp_k_head_num_, model.tp_v_head_num_, model.head_dim_)
        for layer in range(model.layers_num):
            for destination, source in ((static.base_k[layer], bundle["base"][layer][0]),
                                        (static.base_v[layer], bundle["base"][layer][1])):
                destination.copy_(source[0].transpose(0, 1))
            for index, directory in enumerate(args.adapters):
                start, end = static.get_lora_rank_range_by_dir(directory)
                static.mini_k[layer][:, start:end].copy_(bundle["sources"][index]["mini"][layer][0])
                static.mini_v[layer][:, start:end].copy_(bundle["sources"][index]["mini"][layer][1])
        if args.key_rope:
            static.bind_key_rope(bundle["rope_cos"].cuda(), bundle["rope_sin"].cuda())

        def execute(mode):
            source_key = "full" if mode in ("ordinary", "full_reprefill") else "rebuilt"
            server.lowra_runtime = server.lowra_prefix_cache = None
            if mode in ("full_reprefill", "oracle_cached"):
                cache = FullPrefixKVCache(prefix_len, model.mem_manager)
                server.lowra_prefix_cache = cache
                server.lowra_runtime = FullReprefillRuntime(prefix_len, model.mem_manager, cache)
            elif mode in ("serial", "parallel"):
                server.lowra_runtime = LowRALayerRuntime(static, model.mem_manager,
                    "layer_serial" if mode == "serial" else "layer_prefetch",
                    reconstruction_kernel="bounded", reconstruction_workers=1296)
            requests = []
            for case in bundle["cases"]:
                request = dict(request_id=case["request_id"], adapter_dir=args.adapters[case["adapter_index"]],
                               input_id=list(case["query"]), output_len=args.output_len,
                               sampling_param=dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=1))
                if mode == "ordinary":
                    request["input_id"] = bundle["prefix_ids"] + request["input_id"]
                elif mode in ("full_reprefill", "oracle_cached"):
                    request["prefix_input_ids"] = bundle["prefix_ids"]
                requests.append(request)
            def oracle_build(directories, _prefix, indices):
                for directory in directories:
                    source = bundle["sources"][args.adapters.index(directory)]["rebuilt_cache"]
                    for layer, (k, v) in enumerate(source):
                        model.mem_manager.key_buffer[layer][indices[directory]] = k[0].transpose(0, 1).cuda()
                        model.mem_manager.value_buffer[layer][indices[directory]] = v[0].transpose(0, 1).cuda()
            outputs = []
            by_id = {case["request_id"]: case for case in bundle["cases"]}
            def forced_sample(logits, batch):
                step = len(outputs)
                outputs.append(logits.detach().cpu())
                ids = torch.tensor([by_id[request["request_id"]][source_key]["generated"][step]
                                    for request in batch.requests], device="cuda")
                probs = logits.softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
                return ids, probs
            original_builder = server._prefill_full_prefixes
            if mode == "oracle_cached":
                server._prefill_full_prefixes = oracle_build
            try:
                with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=forced_sample):
                    server.exposed_add_batch("e2e", requests, "fp16")
                    server.exposed_prefill_batch("e2e")
                    for _ in range(args.output_len - 1):
                        server.exposed_decode_batch("e2e")
                    server.exposed_remove_batch("e2e")
            finally:
                server._prefill_full_prefixes = original_builder
                if server.lowra_prefix_cache is not None:
                    server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
            if model.mem_manager.can_use_mem_size != model.mem_manager.tot_size - infer.a_loc.numel():
                raise AssertionError("E2E forward leaked private, prefix or workspace slots")
            return torch.stack(outputs)

        outputs = {}
        row = dict(prefix_len=prefix_len, modes={}, runtime_checks={}, reference_checks={})
        row["reference_checks"]["hf_cache_vs_direct"] = difference(
            torch.stack([case["full"]["logits"] for case in bundle["cases"]], dim=1),
            torch.stack([case["direct"]["logits"] for case in bundle["cases"]], dim=1))
        for mode in ("ordinary", "full_reprefill", "oracle_cached", "serial", "parallel"):
            outputs[mode] = execute(mode)
            key = "full" if mode in ("ordinary", "full_reprefill") else "rebuilt"
            expected = torch.stack([case[key]["logits"] for case in bundle["cases"]], dim=1)
            row["modes"][mode] = difference(outputs[mode], expected)
            print("prefix=%s mode=%s %s" % (prefix_len, mode, json.dumps(row["modes"][mode])), flush=True)
        for mode in ("serial", "parallel"):
            row["runtime_checks"][mode + "_vs_oracle_cached"] = difference(
                outputs[mode], outputs["oracle_cached"], atol=0.05, rtol=0.005)
        row["runtime_checks"]["parallel_vs_serial"] = difference(outputs["parallel"], outputs["serial"], atol=0.01, rtol=0.001)
        row["approximation_only"] = dict(
            note="Different greedy histories are not a runtime correctness test",
            first_token_logits=difference(
                torch.stack([case["rebuilt"]["logits"][0] for case in bundle["cases"]]),
                torch.stack([case["full"]["logits"][0] for case in bundle["cases"]])))
        report["cases"].append(row)
        torch.save(outputs, args.output_dir / ("serving_prefix%s_rope%s.pt" % (prefix_len, int(args.key_rope))))
        save_json(args.output_dir / ("serving_rope%s.json" % int(args.key_rope)), report)
        del static
    report["all_passed"] = all(
        value["finite"] and value["failures"] == 0 and value["argmax_agreement"] == 1.0
        for case in report["cases"] for group in ("modes", "runtime_checks", "reference_checks")
        for value in case[group].values())
    paths = ["slora/common/lowra_reconstruct.py", "slora/common/lowra_reconstruct_kv.py",
             "slora/common/lowra_reconstruct_bounded.py", "slora/common/lowra_layer_runtime.py",
             "slora/common/lowra_static.py", "slora/models/peft/lora_unordered_batch_infer.py",
             "slora/common/lowra_key_rope.py",
             "slora/server/router/model_infer/model_rpc.py", "benchmarks/lowra_layer_profile.py",
             "try/validate_model_e2e.py"]
    report["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    save_json(args.output_dir / ("serving_rope%s.json" % int(args.key_rope)), report)
    print(json.dumps(dict(all_passed=report["all_passed"], key_rope=args.key_rope)), flush=True)
    torch.distributed.destroy_process_group()
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    args = arguments()
    if args.stage == "reference":
        reference(args)
    else:
        serving(args)
