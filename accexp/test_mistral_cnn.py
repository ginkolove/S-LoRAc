"""CNN/DailyMail preparation, scoring and independent single-device stages."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cnn_dailymail_score import score
from data.cnn_dailymail import ParquetSplit, article_prompt, build_samples, prepare
from eval_minikv import (forward_tokens, fresh_cache, parse_args as eval_args,
                         prepare_independent_minikv_sources, prepare_lora_prefixes,
                         reconstruct_prefix, snapshot_cache)
from run_mistral_cnn import evaluator_command, parse_args
from test_eval_minikv import attach_lora, tiny_model


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(char) + 3 for char in text]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text, kwargs.get("add_special_tokens", True)) for text in texts]}


def rows(split, count):
    return [dict(id=f"{split}-{index}", article=f"Complete {split} article number {index}.",
                 highlights=f"REFERENCE_{split}_{index}") for index in range(count)]


def write_split(root, split, records):
    directory = Path(root) / "3.0.0"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), directory / f"{split}-00000-of-00001.parquet",
                   row_group_size=2)
    return ParquetSplit(root, "3.0.0", split)


class CNNDataTests(unittest.TestCase):
    def test_shared_prefix_reproducible_random_samples_and_no_test_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            train = write_split(directory, "train", rows("train", 8))
            test = write_split(directory, "test", rows("test", 12))
            kwargs = dict(context_limit=2000, max_new_tokens=64, num_shots=2,
                          prefix_max_tokens=800, sample_size=4, seed=42)
            samples, report = build_samples(train, test, CharacterTokenizer(), **kwargs)
            repeated, repeated_report = build_samples(train, test, CharacterTokenizer(), **kwargs)
            self.assertEqual(samples, repeated)
            self.assertEqual(report, repeated_report)
            self.assertEqual(len(samples), 4)
            self.assertEqual(len({sample["prefix"] for sample in samples}), 1)
            self.assertNotEqual(report["indices"], list(range(4)))
            for sample in samples:
                self.assertNotIn(sample["references"][0], sample["prefix"] + sample["query"])
                self.assertIn("REFERENCE_train_", sample["prefix"])
                self.assertTrue(sample["query"].endswith("### Summary: "))
                self.assertEqual(sample["metadata"]["input_tokens"],
                                 len(CharacterTokenizer().encode(sample["prefix"] + sample["query"])))

    def test_full_articles_are_filtered_not_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            train = write_split(directory, "train", rows("train", 5))
            raw = rows("test", 3)
            raw[0]["article"] += " longer" * 1000
            raw[1]["highlights"] += " reference" * 1000
            test = write_split(directory, "test", raw)
            samples, report = build_samples(train, test, CharacterTokenizer(), context_limit=2000,
                                            max_new_tokens=64, num_shots=1,
                                            prefix_max_tokens=800, sample_size=0)
            self.assertEqual(report["test_overlength"], 1)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["query"], article_prompt(raw[1]["article"]))
            self.assertEqual(samples[0]["references"], [raw[1]["highlights"]])

    def test_demo_overlap_excluded_and_insufficient_population_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = rows("train", 2)
            train = write_split(directory, "train", raw)
            test = write_split(directory, "test", raw + rows("test", 2))
            kwargs = dict(context_limit=2000, max_new_tokens=64, num_shots=2, prefix_max_tokens=800)
            samples, report = build_samples(train, test, CharacterTokenizer(), sample_size=0, **kwargs)
            self.assertEqual(report["test_demo_overlap"], 2)
            self.assertEqual(len(samples), 2)
            with self.assertRaisesRegex(ValueError, "only 2"):
                build_samples(train, test, CharacterTokenizer(), sample_size=3, **kwargs)

    def test_missing_shards_and_duplicate_ids_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train-00000-of-00002.parquet"
            pq.write_table(pa.Table.from_pylist(rows("train", 2)), path)
            with self.assertRaisesRegex(ValueError, "Missing or inconsistent"):
                ParquetSplit(directory, "3.0.0", "train")
        with tempfile.TemporaryDirectory() as directory:
            train = write_split(directory, "train", rows("train", 4))
            raw = rows("test", 2)
            raw[1]["id"] = raw[0]["id"]
            test = write_split(directory, "test", raw)
            with self.assertRaisesRegex(ValueError, "Duplicate test source id"):
                build_samples(train, test, CharacterTokenizer(), context_limit=2000,
                              max_new_tokens=64, num_shots=1, sample_size=0)

    def test_default_context_respects_sliding_window_and_export_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            write_split(directory, "train", rows("train", 5))
            write_split(directory, "test", rows("test", 3))
            args = parse_args(["--dataset-dir", directory, "--sample-size", "2", "--num-shots", "1"])
            destination = Path(directory) / "prepared.jsonl"
            config = SimpleNamespace(max_position_embeddings=32768, sliding_window=4096)
            with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                report = prepare(args, destination)
                self.assertEqual(report["context_limit"], 4096)
                self.assertFalse(report["prefix_may_leave_sliding_window"])
                self.assertEqual(len(destination.read_text().splitlines()), 2)
                self.assertTrue(Path(str(destination) + ".meta.json").is_file())
                with self.assertRaises(FileExistsError):
                    prepare(args, destination)


class CNNScoreTests(unittest.TestCase):
    def test_rouge_is_macro_f1_and_uses_raw_predictions(self):
        predictions = ["The red cat ate fish.", "unrelated words"]
        references = [[predictions[0]], ["different tokens"]]
        values = score(predictions, references, [{}, {}])
        for metric in ("rouge1", "rouge2", "rougeL"):
            self.assertEqual(values[metric], 0.5)
        self.assertEqual(score([""], [["nonempty reference"]], [{}])["rougeL"], 0)
        # No best-substring/oracle extraction: appended text is also scored.
        longer = score([predictions[0] + " Extra unrelated text."], [references[0]], [{}])
        self.assertLess(longer["rougeL"], 1)

    def test_scorer_validation_and_cached_results_are_independent(self):
        with self.assertRaises(ValueError):
            score([], [], [])
        with self.assertRaises(ValueError):
            score(["answer"], [[]], [{}])
        first = score(["correct answer"], [["other", "correct answer"]], [{}])
        first["rouge1"] = 100
        self.assertEqual(score(["correct answer"], [["other", "correct answer"]], [{}])["rouge1"], 1)


class SingleDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_launcher_forces_independent_single_device_evaluation(self):
        args = parse_args([])
        command = evaluator_command(args, "samples.jsonl", "result.json")
        self.assertIn("--minikv-independent-prefill", command)
        self.assertNotIn("--lora-device", command)
        self.assertNotIn("--device-map", command)
        self.assertEqual(command[command.index("--weight-format") + 1], "bin")
        with self.assertRaises(SystemExit):
            parse_args(["--lora-device", "cuda:1"])
        with self.assertRaises(SystemExit):
            eval_args(["--model", "x", "--lora", "y", "--data", "z", "--output", "out",
                       "--minikv-independent-prefill", "--lora-device", "cuda:1"])
        with self.assertRaises(SystemExit):
            eval_args(["--model", "x", "--lora", "y", "--data", "z", "--output", "out",
                       "--minikv-independent-prefill", "--device-map", "auto"])

    def test_independent_sources_are_native_base_and_real_minikv(self):
        model = tiny_model("mistral", attention="sdpa")
        prefix, query = [3, 4, 5, 6, 7], [8, 9]
        native_base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        model = attach_lora(model, targets=["q_proj", "v_proj"])
        native_full = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        groups = {"shared": [(0, dict(prefix_ids=prefix, full_ids=prefix + query, query_ids=query,
                                     sample={"id": "sample"}))]}
        stats = {"stage_seconds": {}}
        with torch.inference_mode():
            bases, minis = prepare_independent_minikv_sources(model, groups, Mock(), stats)
            reconstructed = reconstruct_prefix(model, bases["shared"], minis["shared"])
        for actual, expected in zip(bases["shared"], native_base):
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertTrue(all(layer["mini_k"] is None for layer in minis["shared"]))
        self.assertTrue(all(layer["mini_v"] is not None for layer in minis["shared"]))
        for actual, expected in zip(reconstructed[0], native_full[0]):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertGreater(float((reconstructed[1][1] - native_full[1][1]).abs().max()), 1e-7)
        self.assertIn("minikv_base_prefix_prepare", stats["stage_seconds"])
        self.assertIn("minikv_adapted_prefix_prepare", stats["stage_seconds"])

    def test_full_stage_can_prefill_without_minikv_hooks(self):
        model = attach_lora(tiny_model("mistral"), targets=["q_proj", "v_proj"])
        prefix, query = [3, 4, 5], [6]
        groups = {"shared": [(0, dict(prefix_ids=prefix, full_ids=prefix + query,
                                     query_ids=query, sample={"id": "sample"}))]}
        args = SimpleNamespace(cache_check_atol=None, cache_check_rtol=0.005)
        with torch.inference_mode(), patch("eval_minikv.capture_minikv", side_effect=AssertionError):
            full, minis = prepare_lora_prefixes(model, groups, args, Mock(), [], collect_minikv=False)
        self.assertEqual(set(full), {"shared"})
        self.assertEqual(minis, {})

    def test_cli_runs_all_three_stages_and_writes_rouge(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path, data_path = root / "model", root / "lora", root / "data"
            config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=256, sliding_window=256,
                                   bos_token_id=1, eos_token_id=2, pad_token_id=0)
            model = MistralForCausalLM(config).eval()
            model.save_pretrained(model_path, safe_serialization=False)
            vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"word{i}": i for i in range(3, 64)}}
            backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                bos_token="[BOS]", eos_token="[EOS]")
            tokenizer.save_pretrained(model_path)
            attach_lora(model, targets=["q_proj", "v_proj"]).save_pretrained(adapter_path)
            write_split(data_path, "train", rows("train", 4))
            test_rows = [dict(row, article=row["article"] + " word10" * index)
                         for index, row in enumerate(rows("test", 3))]
            write_split(data_path, "test", test_rows)
            output = root / "results"
            command = [sys.executable, str(Path(__file__).parent / "run_mistral_cnn.py"),
                       "--model", str(model_path), "--lora", str(adapter_path),
                       "--dataset-dir", str(data_path), "--output-dir", str(output),
                       "--device", "cpu", "--dtype", "float32", "--sample-size", "3", "--batch-size", "3",
                       "--num-shots", "1", "--prefix-max-tokens", "96", "--max-new-tokens", "2"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads((output / "result.json").read_text())
            self.assertTrue(result["experiment"]["minikv_independent_prefill"])
            self.assertFalse(result["experiment"]["parallel_base_lora"])
            self.assertEqual(result["experiment"]["unique_prefixes"], 1)
            self.assertEqual(result["dataset_preparation"]["selected"], 3)
            self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(set(result["scores"][mode]), {"rouge1", "rouge2", "rougeL"})
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
            journal = [json.loads(line) for line in (output / "result.predictions.jsonl").read_text().splitlines()]
            stages = [row["mode"] for row in journal]
            self.assertEqual(stages, ["base", "lora_full", "lora_minikv"])
            self.assertTrue(all(len(row["records"]) == 3 for row in journal))
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")


if __name__ == "__main__":
    unittest.main()
