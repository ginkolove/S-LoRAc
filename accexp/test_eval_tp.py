"""TP LoRA layout tests and optional real two-GPU numerical comparison."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from eval_minikv import (capture_minikv, forward_tokens, fresh_cache, generate,
                        reconstruct_prefix, snapshot_cache)
from eval_tp import (COLUMN, ROW, attach_tp_lora, checkpoint_files, shard_lora_weight,
                     validate_checkpoint_shapes, validate_prefix, validate_tp_model)
from test_eval_minikv import tiny_model


def make_fixture(root, dtype=torch.float32, target_modules=None):
    torch.manual_seed(42)
    model = tiny_model(attention="sdpa").to(dtype=dtype)
    model.save_pretrained(root / "model", safe_serialization=True)
    prefix, query = list(range(3, 16)), [20, 21, 22]
    expected = {}

    def collect(mode, pairs):
        expected[mode] = dict(pairs=pairs,
            logits=forward_tokens(model, query, fresh_cache(model, pairs)).logits.detach().cpu(),
            tokens=generate(model, query, pairs, 4, set())[0])

    with torch.inference_mode():
        base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        collect("base", base)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
            target_modules=sorted(COLUMN | ROW) if target_modules is None else target_modules,
            task_type="CAUSAL_LM"))
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.08)
        model.eval()
        model.save_pretrained(root / "adapter")
        with capture_minikv(model) as mini:
            output = forward_tokens(model, prefix, fresh_cache(model))
        full = snapshot_cache(output.past_key_values)
        collect("lora_full", full)
        collect("lora_minikv", reconstruct_prefix(model, base, mini))
    torch.save(dict(expected=expected, prefix=prefix, query=query, dtype=str(dtype).split(".")[-1]), root / "expected.pt")
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"t{i}": i for i in range(3, 64)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
        bos_token="[BOS]", eos_token="[EOS]", pad_token="[UNK]")
    tokenizer.save_pretrained(root / "model")
    samples = [dict(id=str(i), prefix="t3 t4 t5 ", query=query, references=["t6"], metadata={})
               for i, query in enumerate(("t20 t21", "t22 t23 t24", "t25"))]
    (root / "data.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples))


def distributed_smoke(root):
    torch.set_num_threads(1)
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank), timeout=timedelta(minutes=3))
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        reference = torch.load(root / "expected.pt", map_location="cpu", weights_only=True)
        tolerance = 0.003 if reference["dtype"] == "float16" else 3e-5
        model = AutoModelForCausalLM.from_pretrained(root / "model", dtype=getattr(torch, reference["dtype"]),
            tp_plan="auto", device_mesh=mesh, attn_implementation="sdpa", local_files_only=True).eval()
        validate_tp_model(model, 2)
        prefix, query = reference["prefix"], reference["query"]
        with torch.inference_mode():
            base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values, device=None)
            for mode in ("base", "lora_full", "lora_minikv"):
                if mode == "base":
                    pairs = base
                elif mode == "lora_full":
                    model, layouts = attach_tp_lora(model, str(root / "adapter"), "safetensors")
                    expected_dims = {0 if target in COLUMN else 1
                                     for target in model.peft_config["default"].target_modules}
                    assert {layout["shard_dim"] for layout in layouts} == expected_dims
                    with model.disable_adapter():
                        disabled = forward_tokens(model, query, fresh_cache(model, base)).logits.cpu()
                    torch.testing.assert_close(disabled, reference["expected"]["base"]["logits"], atol=tolerance, rtol=tolerance)
                    pairs = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values, device=None)
                else:
                    with capture_minikv(model, keep_on_device=True) as mini:
                        forward_tokens(model, prefix, fresh_cache(model))
                    pairs = reconstruct_prefix(model, base, mini, keep_on_device=True)
                validate_prefix(pairs, model.config, 2, len(prefix))
                expected = reference["expected"][mode]
                for pair, full_pair in zip(pairs, expected["pairs"]):
                    for local, full in zip(pair, full_pair):
                        width = full.shape[1] // 2
                        torch.testing.assert_close(local.cpu(), full[:, rank*width:(rank+1)*width], atol=tolerance, rtol=tolerance)
                logits = forward_tokens(model, query, fresh_cache(model, pairs)).logits.cpu()
                torch.testing.assert_close(logits, expected["logits"], atol=tolerance, rtol=tolerance)
                assert generate(model, query, pairs, 4, set())[0] == expected["tokens"]
                if rank == 0:
                    print(mode + ': TP=2 logits, local prefix KV and greedy tokens match single-device reference', flush=True)
            saved = [(k.clone(), v.clone()) for k, v in pairs]
            generate(model, query, pairs, 4, set())
            for before, after in zip(saved, pairs):
                for left, right in zip(before, after):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TPTests(unittest.TestCase):
    def test_column_lora_splits_b_and_replicates_a(self):
        a, b = torch.arange(24).reshape(3, 8), torch.arange(24).reshape(8, 3)
        for rank in (0, 1):
            self.assertTrue(torch.equal(shard_lora_weight(a, "A", (8, 8), 0, rank, 2), a))
            self.assertTrue(torch.equal(shard_lora_weight(b, "B", (8, 8), 0, rank, 2), b[rank*4:(rank+1)*4]))

    def test_row_lora_splits_a_and_replicates_b(self):
        a, b = torch.arange(24).reshape(3, 8), torch.arange(24).reshape(8, 3)
        for rank in (0, 1):
            self.assertTrue(torch.equal(shard_lora_weight(a, "A", (8, 8), 1, rank, 2), a[:, rank*4:(rank+1)*4]))
            self.assertTrue(torch.equal(shard_lora_weight(b, "B", (8, 8), 1, rank, 2), b))

    def test_incompatible_adapter_shape_fails(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            shard_lora_weight(torch.zeros(3, 9), "A", (8, 8), 0, 0, 2)

    def test_partial_checkpoints_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "shard.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Missing"):
                checkpoint_files(root, "safetensors")
            (root / "shard.safetensors").write_bytes(b"preallocated, not complete")
            (root / "shard.safetensors.aria2").write_bytes(b"unfinished")
            with self.assertRaisesRegex(ValueError, "Unfinished"):
                checkpoint_files(root, "safetensors")
            (root / "adapter_model.bin").write_bytes(b"complete")
            (root / "adapter_model.safetensors").write_bytes(b"incomplete")
            (root / "adapter_model.safetensors.aria2").write_bytes(b"unfinished")
            self.assertEqual(checkpoint_files(root, "auto", adapter=True)[0], "bin")

    def test_header_validation_accepts_legacy_rope_and_rejects_wrong_shapes(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = tiny_model()
            model.save_pretrained(root / "model", safe_serialization=True)
            config = model.config
            adapter = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
            get_peft_model(model, adapter).save_pretrained(root / "adapter")
            files = {kind: [dict(path=str(path))] for kind, path in (
                ("model", root / "model/model.safetensors"), ("adapter", root / "adapter/adapter_model.safetensors"))}
            state = load_file(files["model"][0]["path"])
            state["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(4)
            save_file(state, files["model"][0]["path"])
            formats = {"model": "safetensors", "adapter": "safetensors"}
            result = validate_checkpoint_shapes(config, adapter, files, formats)
            self.assertEqual(result["model"]["legacy_rope_buffers"], 1)
            key = "model.layers.0.self_attn.k_proj.weight"
            original = state[key]
            state[key] = torch.zeros(original.shape[0] + 1, original.shape[1])
            save_file(state, files["model"][0]["path"])
            with self.assertRaisesRegex(ValueError, "model checkpoint.*wrong_shapes"):
                validate_checkpoint_shapes(config, adapter, files, formats)
            state[key] = original
            save_file(state, files["model"][0]["path"])
            bad_adapter = load_file(files["adapter"][0]["path"])
            key = next(iter(bad_adapter))
            bad_adapter[key] = torch.zeros(1, 1)
            save_file(bad_adapter, files["adapter"][0]["path"])
            with self.assertRaisesRegex(ValueError, "adapter checkpoint.*wrong_shapes"):
                validate_checkpoint_shapes(config, adapter, files, formats)

    @unittest.skipUnless(os.environ.get("RUN_TP_SMOKE") == "1", "Set RUN_TP_SMOKE=1 for real two-GPU test")
    def test_two_rank_numerical_equivalence(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dtype = os.environ.get("TP_TEST_DTYPE", "float32")
            make_fixture(root, dtype=getattr(torch, dtype))
            subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                            str(Path(__file__).resolve()), "--distributed-smoke", str(root)],
                           check=True, timeout=180, env=dict(os.environ, OMP_NUM_THREADS="1"))
            output = root / "result.json"
            subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                            str(Path(__file__).with_name("eval_tp.py").resolve()),
                            "--model", str(root / "model"), "--lora", str(root / "adapter"),
                            "--data", str(root / "data.jsonl"), "--dtype", dtype,
                            "--max-new-tokens", "4", "--batch-size", "2", "--output", str(output)],
                           check=True, timeout=180, env=dict(os.environ, OMP_NUM_THREADS="1"))
            result = json.loads(output.read_text())
            self.assertEqual(result["experiment"]["tp_size"], 2)
            self.assertEqual(result["experiment"]["samples"], 3)
            self.assertEqual(len(result["experiment"]["ranks"]), 2)
            self.assertTrue(all(set(row["predictions"]) == {"base", "lora_full", "lora_minikv"}
                                for row in result["predictions"]))
            partial = json.loads(output.with_suffix(".partial.json").read_text())
            self.assertEqual(partial["state"], "complete")
            self.assertEqual(partial["completed_requests"], {mode: 3 for mode in ("base", "lora_full", "lora_minikv")})
            self.assertEqual(partial["scores"], result["scores"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--distributed-smoke":
        distributed_smoke(Path(sys.argv[2]))
    else:
        unittest.main()
