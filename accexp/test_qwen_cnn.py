"""Qwen-native CNN data, rank-64 adapter compatibility and sequential ROUGE evaluation."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model

from cnn_dailymail_score import score_turn_filtered
from data.cnn_dailymail import build_samples
from data.qwen_cnn import PROTOCOL, SYSTEM, article_user, prompt_builders
from eval_minikv import capture_minikv, forward_tokens, fresh_cache, reconstruct_prefix, snapshot_cache
from qwen_utils import check_adapter
from run_qwen_cnn import evaluator_command, main, parse_args
from test_eval_minikv import tiny_model
from test_mistral_cnn import rows, write_split
from test_qwen_gsm8k import tiny_tokenizer


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


def adapted_model():
    model = get_peft_model(tiny_model("qwen2", attention="sdpa"),
        LoraConfig(r=64, lora_alpha=128, target_modules=TARGETS, lora_dropout=0, task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)
    return model.eval()


def save_fixture(root):
    model_path, adapter_path = root / "model", root / "adapter"
    model = tiny_model("qwen2", attention="sdpa")
    model.save_pretrained(model_path)
    tokenizer = tiny_tokenizer()
    tokenizer.save_pretrained(model_path)
    adapted_model().save_pretrained(adapter_path, save_embedding_layers=False)
    # Like the real local adapter, retain the vocabulary but supply an unrelated template.
    tokenizer.chat_template = "<|start_header_id|>user<|end_header_id|>{{ messages[0]['content'] }}<|eot_id|>"
    tokenizer.save_pretrained(adapter_path)
    return model_path, adapter_path


class QwenCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_defaults_and_shared_native_evaluator(self):
        args = parse_args(["--output-dir", "out"])
        self.assertEqual(args.lora, "/data/hub/cnn_dailmail-lora/qwen2.5-7b")
        self.assertEqual((args.sample_size, args.num_shots, args.batch_size, args.dtype), (1000, 3, 8, "float32"))
        self.assertEqual((args.max_context_len, args.max_new_tokens, args.seed), (4096, 256, 42))
        command = evaluator_command(args, "samples", "result")
        self.assertIn("--minikv-independent-prefill", command)
        self.assertEqual(command[command.index("--tokenizer") + 1], args.model)
        self.assertTrue(command[command.index("--scorer") + 1].endswith("cnn_dailymail_score.py:score_turn_filtered"))
        self.assertNotIn("--lora-device", command)
        self.assertNotIn("--cache-check-atol", command)
        with self.assertRaises(SystemExit):
            parse_args(["--output-dir", "out", "--batch-size", "0"])

    def test_fixed_chat_demonstrations_sampling_lengths_and_no_gold_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            train = write_split(directory, "train", rows("train", 10))
            raw = rows("test", 8)
            for row in raw:
                row["highlights"] = "HELDOUT_SUMMARY_MUST_NOT_ENTER_PROMPT"
            raw[0]["article"] += " longer" * 1000
            test = write_split(directory, "test", raw)
            tokenizer = tiny_tokenizer()
            prefix_builder, query_builder = prompt_builders(tokenizer)
            kwargs = dict(context_limit=512, max_new_tokens=16, num_shots=3, prefix_max_tokens=300,
                sample_size=5, seed=42, prefix_builder=prefix_builder, prompt_builder=query_builder, prompt_format=PROTOCOL)
            samples, report = build_samples(train, test, tokenizer, **kwargs)
            self.assertEqual((samples, report), build_samples(train, test, tokenizer, **kwargs))
            self.assertEqual(report["test_overlength"], 1)
            self.assertEqual(len({s["prefix"] for s in samples}), 1)
            self.assertEqual(len(report["demonstrations"]), 3)
            messages = [dict(role="system", content=SYSTEM)]
            for demo in report["demonstrations"]:
                row = train.get(demo["source_index"])
                messages.extend((dict(role="user", content=article_user(row["article"])),
                                 dict(role="assistant", content=row["highlights"])))
            for sample in samples:
                article = test.get(sample["metadata"]["source_index"])["article"]
                expected = tokenizer.apply_chat_template(messages + [dict(role="user", content=article_user(article))],
                    tokenize=False, add_generation_prompt=True)
                self.assertEqual(sample["prefix"] + sample["query"], expected)
                self.assertNotIn(sample["references"][0], expected)
                self.assertFalse(sample["metadata"]["article_truncated"])
                self.assertLessEqual(sample["metadata"]["input_tokens"] + 16, 512)
                self.assertEqual(sample["metadata"]["prompt_format"], PROTOCOL)

    def test_adapter_template_mismatch_is_explicit_and_vocabulary_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, adapter = save_fixture(root)
            args = parse_args(["--model", str(model), "--lora", str(adapter), "--output-dir", str(root / "out")])
            with self.assertRaisesRegex(ValueError, "chat templates differ"):
                check_adapter(args)
            report = check_adapter(args, allow_template_mismatch=True)
            self.assertFalse(report["chat_template_equal"])
            self.assertEqual(report["rank"], 64)
            self.assertIn("deliberately unused", report["template_warning"])
            from transformers import AutoTokenizer
            other = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
            other.add_tokens(["incompatible_new_token"])
            other.save_pretrained(adapter)
            with self.assertRaisesRegex(ValueError, "vocabularies/special tokens differ"):
                check_adapter(args, allow_template_mismatch=True)

    def test_rank64_gqa_bias_and_reconstruction_are_real_low_rank_operations(self):
        model = adapted_model()
        prefix = [4, 5, 6, 7, 8]
        with torch.inference_mode(), model.disable_adapter():
            base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        with torch.inference_mode(), capture_minikv(model) as mini:
            full = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        self.assertEqual(mini[0]["mini_k"].shape, (1, len(prefix), 64))
        self.assertEqual(mini[0]["mini_v"].shape, (1, len(prefix), 64))
        actual = reconstruct_prefix(model, base, mini)
        self.assertEqual(actual[0][0].shape, (1, 2, len(prefix), 8))
        for expected, value in zip(full[0], actual[0]):
            torch.testing.assert_close(expected, value, atol=3e-6, rtol=3e-6)
        self.assertGreater(float((full[1][1] - actual[1][1]).abs().max()), 1e-7)

    def test_complete_cpu_pipeline_uses_native_generate_and_independent_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, adapter = save_fixture(root)
            write_split(root, "train", rows("train", 8))
            raw = rows("test", 3)
            for index, row in enumerate(raw):
                row["article"] += " word10" * index
            write_split(root, "test", raw)
            out = root / "out"
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"):
                main(["--model", str(model), "--lora", str(adapter), "--dataset-dir", str(root),
                    "--sample-size", "3", "--num-shots", "1", "--prefix-max-tokens", "110",
                    "--max-context-len", "128", "--max-new-tokens", "2", "--batch-size", "2",
                    "--device", "cpu", "--output-dir", str(out)])
            result = json.loads((out / "result.json").read_text())
            report = json.loads((out / "samples.jsonl.meta.json").read_text())
            samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(hashlib.sha256((out / "samples.jsonl").read_bytes()).hexdigest(), report["output_sha256"])
            self.assertEqual(result["experiment"]["generation_engine"], "transformers.generate")
            self.assertFalse(result["adapter_compatibility"]["chat_template_equal"])
            self.assertEqual(result["experiment"]["unique_prefixes"], 1)
            self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            self.assertEqual(result["scorer_metadata"]["primary_metric"], "rougeL")
            for mode in ("base", "lora_full", "lora_minikv"):
                raw = [r["predictions"][mode]["text"] for r in result["predictions"]]
                self.assertEqual(result["scores"][mode], score_turn_filtered(raw, [s["references"] for s in samples], samples))
                self.assertEqual(set(result["scores"][mode]), {"rouge1", "rouge2", "rougeL"})
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
            self.assertEqual(json.loads((out / "status.json").read_text())["state"], "complete")
            with self.assertRaises(FileExistsError):
                main(["--output-dir", str(out)])


if __name__ == "__main__":
    unittest.main()
