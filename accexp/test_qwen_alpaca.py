"""Alpaca source isolation, reference-independent scoring and real tiny Qwen integration."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model

from data.qwen_alpaca import ALPACA_PROMPTS, CAVEAT, PROTOCOL, build_samples, read_local
from eval_minikv import capture_minikv, forward_tokens, fresh_cache, prepare_inputs, reconstruct_prefix, snapshot_cache
from qwen_alpaca_score import METRICS, extract_prediction, score, score_unfiltered
from qwen_utils import check_adapter
from run_qwen_alpaca import evaluator_command, finalize, main, parse_args
from test_eval_minikv import tiny_model
from test_qwen_gsm8k import tiny_tokenizer


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


def rows(count=12):
    return [dict(instruction=f"word{4 + index} word20", input="word21" if index % 2 else "",
                 output=f"UNIQUE_REFERENCE_{index}", text="DO_NOT_USE_PRECOMBINED_GOLD_TEXT")
            for index in range(count)]


def write_rows(root, values):
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    directory = Path(root) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train-00000-of-00001.parquet"
    parquet.write_table(arrow.Table.from_pylist(values), path)
    return path


def adapted_model():
    model = get_peft_model(tiny_model("qwen2", attention="sdpa"),
        LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS, lora_dropout=0, task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.01)
    return model.eval()


class QwenAlpacaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_arguments_and_native_independent_three_stage_command(self):
        args = parse_args(["--output-dir", "unused"])
        self.assertEqual((args.model, args.dataset_dir, args.lora),
            ("/data/hub/qwen2.5-7b-instruct", "/data/hub/alpaca", "/data/hub/alpaca-lora"))
        self.assertEqual((args.sample_size, args.num_shots, args.seed, args.batch_size), (1000, 3, 42, 20))
        self.assertEqual((args.dtype, args.max_context_len, args.max_new_tokens), ("float32", 2048, 256))
        command = evaluator_command(args, "samples", "result")
        self.assertIn("--minikv-independent-prefill", command)
        self.assertIn("--no-add-special-tokens", command)
        self.assertEqual(command[command.index("--tokenizer") + 1], args.model)
        self.assertTrue(command[command.index("--scorer") + 1].endswith("qwen_alpaca_score.py:score"))
        self.assertNotIn("--cache-check-atol", command)
        self.assertNotIn("--lora-device", command)
        for flags in (["--batch-size", "0"], ["--sample-size", "-1"], ["--num-shots", "0"],
                      ["--max-context-len", "256"]):
            with self.assertRaises(SystemExit):
                parse_args(["--output-dir", "unused"] + flags)

    def test_reproducible_sampling_duplicates_fewshots_context_and_no_answer_leak(self):
        raw = rows(15)
        raw.append(dict(raw[0]))
        raw.append(dict(instruction="", input="", output="invalid"))
        raw.append(dict(instruction="word60 " * 1000, input="", output="OVERLONG_ANSWER"))
        tokenizer = tiny_tokenizer()
        options = dict(context_limit=256, max_new_tokens=16, num_shots=2,
                       prefix_max_tokens=150, sample_size=5, seed=42)
        samples, report = build_samples(raw, tokenizer, **options)
        self.assertEqual((samples, report), build_samples(raw, tokenizer, **options))
        self.assertEqual(report["split_info"]["cleaning"]["duplicate_queries"], 1)
        self.assertEqual(report["split_info"]["cleaning"]["invalid"], 1)
        self.assertEqual(report["test_overlength"], 1)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len({s["prefix"] for s in samples}), 1)
        self.assertEqual(len(report["demonstrations"]), 2)
        self.assertEqual(report["training_overlap_caveat"], CAVEAT)
        demo_ids = set(report["fewshot_train_ids"])
        expected_prefix = "".join(item["query"] + item["references"][0] + "\n\n"
                                  for item in report["demonstrations"])
        self.assertEqual(samples[0]["prefix"], expected_prefix)
        for sample in samples:
            self.assertNotIn(sample["metadata"]["source_id"], demo_ids)
            source = raw[int(sample["metadata"]["source_id"])]
            template = ALPACA_PROMPTS["prompt_input" if source["input"] else "prompt_no_input"]
            self.assertEqual(sample["query"], template.format(**source))
            self.assertEqual(sample["references"], [source["output"]])
            self.assertNotIn(source["output"], sample["prefix"] + sample["query"])
            self.assertNotIn("DO_NOT_USE_PRECOMBINED_GOLD_TEXT", sample["prefix"] + sample["query"])
            self.assertLessEqual(sample["metadata"]["input_tokens"] + 16, 256)
            self.assertFalse(sample["metadata"]["adapter_training_disjoint"])
        prepared = prepare_inputs(samples, tokenizer, 16, 256, False)
        self.assertEqual(len({item["prefix_key"] for item in prepared}), 1)
        all_samples, all_report = build_samples(raw, tokenizer, **dict(options, sample_size=0))
        self.assertEqual(len(all_samples), all_report["test_eligible"])
        with self.assertRaisesRegex(ValueError, "only .* eligible"):
            build_samples(raw, tokenizer, **dict(options, sample_size=100))

    def test_scorer_filters_structure_only_and_applies_to_every_mode(self):
        raw = "Paris.\n\n" + ALPACA_PROMPTS["prompt_no_input"].format(instruction="Next task") + "London."
        self.assertEqual(extract_prediction(raw)["text"], "Paris.")
        for mode in ("base", "lora_full", "lora_minikv"):
            self.assertEqual(score([raw], [["Paris"]], [dict(mode=mode)]), dict.fromkeys(METRICS, 1.0))
            self.assertEqual(score([raw], [["London"]], [dict(mode=mode)]), dict.fromkeys(METRICS, 0.0))
        self.assertLess(score_unfiltered([raw], [["Paris"]], [{}])["generic_token_f1"], 1)
        for text in ("A multiline\nanswer.", "```text\n### Instruction:\n```\nKeep this.",
                     "    ### Instruction:\nKeep this.", "> ### Instruction:\nKeep this."):
            self.assertEqual(extract_prediction(text)["text"], text)
            self.assertFalse(extract_prediction(text)["truncated"])
        empty_and_good = score(["", "Paris"], [["Paris"], ["Paris"]], [{}, {}])
        self.assertEqual(empty_and_good, dict.fromkeys(METRICS, 0.5))
        with self.assertRaises(ValueError):
            score(["Paris"], [], [{}])

    def test_rank8_real_minikv_shapes_rope_and_approximation(self):
        model = adapted_model()
        prefix = [4, 5, 6, 7, 8]
        with torch.inference_mode(), model.disable_adapter():
            base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        with torch.inference_mode(), capture_minikv(model) as mini:
            full = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        self.assertEqual(mini[0]["mini_k"].shape, (1, 5, 8))
        self.assertEqual(mini[0]["mini_v"].shape, (1, 5, 8))
        reconstructed = reconstruct_prefix(model, base, mini)
        self.assertEqual(reconstructed[0][0].shape, (1, 2, 5, 8))
        for actual, expected in zip(reconstructed[0], full[0]):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        self.assertGreater(float((reconstructed[1][1] - full[1][1]).abs().max()), 1e-7)

    def test_tiny_complete_pipeline_and_result_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "adapter"
            tiny_model("qwen2", attention="sdpa").save_pretrained(model_path)
            adapted_model().save_pretrained(adapter_path, save_embedding_layers=False)
            tokenizer = tiny_tokenizer()
            tokenizer.save_pretrained(model_path)
            tokenizer.save_pretrained(adapter_path)
            shard = write_rows(root, rows())
            raw_rows, sources = read_local(root)
            self.assertEqual(len(raw_rows), 12)
            self.assertEqual(sources[0]["sha256"], hashlib.sha256(shard.read_bytes()).hexdigest())
            out = root / "out"
            flags = ["--model", str(model_path), "--lora", str(adapter_path), "--dataset-dir", str(root),
                     "--num-shots", "1", "--prefix-max-tokens", "80", "--sample-size", "3",
                     "--max-context-len", "128", "--max-new-tokens", "2", "--device", "cpu",
                     "--batch-size", "2", "--output-dir", str(out)]
            args = parse_args(flags)
            compatibility = check_adapter(args)
            self.assertEqual(compatibility["rank"], 8)
            self.assertTrue(compatibility["chat_template_equal"])
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"):
                main(flags)
            result_path = out / "result.json"
            value = json.loads(result_path.read_text())
            samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
            report = json.loads((out / "samples.jsonl.meta.json").read_text())
            self.assertEqual(report["output_sha256"], hashlib.sha256((out / "samples.jsonl").read_bytes()).hexdigest())
            self.assertEqual(value["experiment"]["generation_engine"], "transformers.generate")
            self.assertEqual(value["experiment"]["unique_prefixes"], 1)
            self.assertEqual(value["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            self.assertEqual(value["config"]["prompt_protocol"], PROTOCOL)
            self.assertEqual(value["scorer_metadata"]["primary_metric"], "generic_token_f1")
            for mode in ("base", "lora_full", "lora_minikv"):
                texts = [row["predictions"][mode]["text"] for row in value["predictions"]]
                references = [sample["references"] for sample in samples]
                self.assertEqual(value["scores"][mode], score(texts, references, samples))
                self.assertEqual(value["unfiltered_scores"][mode], score_unfiltered(texts, references, samples))
                self.assertEqual(value["generation_diagnostics"][mode]["samples"], 3)
            self.assertEqual(json.loads((out / "status.json").read_text())["state"], "complete")
            with self.assertRaises(FileExistsError):
                main(flags)
            value["predictions"][0]["references"] = ["tampered reference"]
            result_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "Reference"):
                finalize(result_path, samples, report, compatibility, args)


if __name__ == "__main__":
    unittest.main()
