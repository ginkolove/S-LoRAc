"""Few-shot readers: split hygiene, shared prefixes, budgets and evaluator wiring."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from data import alpaca_cleaned, alpaca_mt, guanaco
from data.common import (alpaca_example, clean_examples, load_dataset_samples,
                         make_fewshot)
from eval_minikv import load_samples, parse_args, prepare_inputs


class WordTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + list(range(2, 2 + len(text.split())))


def raw_rows(count, prefix="train"):
    return [dict(id=f"{prefix}-{i}", instruction=f"Question {prefix} {i}?", input="",
                 output=f"{prefix}-answer-{i}") for i in range(count)]


def examples(count, prefix="train"):
    return clean_examples(raw_rows(count, prefix), alpaca_example)[0]


class DataTests(unittest.TestCase):
    def test_shared_train_prefix_and_no_target_answer(self):
        samples, report = make_fewshot(examples(10), examples(4, "test"), WordTokenizer(),
                                       dataset="alpaca_cleaned", context_limit=2048, max_new_tokens=256)
        self.assertEqual(len({sample["prefix"] for sample in samples}), 1)
        self.assertEqual(report["num_shots"], 3)
        self.assertEqual(len(report["fewshot_train_ids"]), 3)
        for sample in samples:
            self.assertNotIn(sample["references"][0], sample["prefix"] + sample["query"])
            self.assertIn("train-answer-", sample["prefix"])
            self.assertIn("### Response:", sample["query"])

    def test_train_only_uses_all_remaining_examples(self):
        pool = examples(10)
        samples, report = make_fewshot(pool, None, WordTokenizer(), dataset="alpaca_cleaned",
                                       context_limit=2048, max_new_tokens=256, seed=42)
        repeated, repeated_report = make_fewshot(pool, None, WordTokenizer(), dataset="alpaca_cleaned",
                                                context_limit=2048, max_new_tokens=256, seed=42)
        self.assertEqual(samples, repeated)
        self.assertEqual(report, repeated_report)
        self.assertEqual(len(samples), 7)
        selected = set(report["fewshot_train_ids"])
        tested = {sample["metadata"]["source_id"] for sample in samples}
        self.assertFalse(selected & tested)
        self.assertEqual(selected | tested, {item["id"] for item in pool})
        self.assertEqual(report["fewshot_examples_excluded_from_test"], 3)
        self.assertEqual(report["split_info"]["test_source"], "dataset_remainder_after_fewshot")
        self.assertEqual(len({sample["prefix"] for sample in samples}), 1)

    def test_no_remaining_test_examples_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "No test examples remain"):
            make_fewshot(examples(3), None, WordTokenizer(), dataset="alpaca_cleaned",
                         context_limit=2048, max_new_tokens=256, num_shots=3)

    def test_seeded_group_holdout_is_disjoint(self):
        pool = examples(30)
        left, report = make_fewshot(pool, None, WordTokenizer(), dataset="alpaca_cleaned",
                                    context_limit=2048, max_new_tokens=256,
                                    holdout_fraction=0.2, seed=42)
        right, right_report = make_fewshot(pool, None, WordTokenizer(), dataset="alpaca_cleaned",
                                           context_limit=2048, max_new_tokens=256,
                                           holdout_fraction=0.2, seed=42)
        self.assertEqual(left, right)
        self.assertEqual(report, right_report)
        self.assertEqual(len(left), 6)
        self.assertTrue(set(report["fewshot_train_ids"]).isdisjoint(
            sample["metadata"]["source_id"] for sample in left))
        self.assertEqual(report["split_info"]["test_source"], "explicit_train_holdout")

    def test_budget_filters_queries_without_truncating_answers(self):
        test = examples(2, "test")
        test[0]["query"] += " long" * 1000
        test[1]["references"] = ["standard answer " * 1000]
        samples, report = make_fewshot(examples(10), test, WordTokenizer(), dataset="alpaca_cleaned",
                                       context_limit=200, max_new_tokens=32, num_shots=1,
                                       prefix_max_tokens=100)
        self.assertEqual(report["test_overlength"], 1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["references"], test[1]["references"])
        self.assertLessEqual(samples[0]["metadata"]["input_tokens"] + 32, 200)
        with self.assertRaisesRegex(ValueError, "Only 0"):
            make_fewshot(examples(10), test, WordTokenizer(), dataset="alpaca_cleaned",
                         context_limit=200, max_new_tokens=32, prefix_max_tokens=1)

    def test_test_questions_cannot_be_demonstrations(self):
        train, test = examples(10), examples(1)
        samples, report = make_fewshot(train, test, WordTokenizer(), dataset="alpaca_cleaned",
                                       context_limit=2048, max_new_tokens=256)
        self.assertEqual(report["train_overlap_excluded"], 1)
        self.assertNotIn(test[0]["id"], report["fewshot_train_ids"])

    def test_guanaco_preserves_history_and_excludes_sibling_roots(self):
        row = dict(text="### Human: First question### Assistant: Old answer"
                        "### Human: Next question### Assistant: Target answer")
        item = guanaco.format_example(row, 0)
        self.assertIn("Old answer", item["query"])
        self.assertTrue(item["query"].endswith("### Assistant: "))
        self.assertNotIn("Target answer", item["query"])
        self.assertEqual(item["references"], ["Target answer"])
        self.assertIsNone(guanaco.format_example(dict(text="### Human: Unanswered"), 0))
        sibling = dict(text="### Human: First question### Assistant: Other answer")
        train = clean_examples([row, sibling], guanaco.format_example)[0]
        test = clean_examples([row], guanaco.format_example)[0]
        with self.assertRaisesRegex(ValueError, "Only 0"):
            make_fewshot(train, test, WordTokenizer(), dataset="guanaco", context_limit=2048,
                         max_new_tokens=256, num_shots=1)

    def test_duplicate_prompt_and_source_ids(self):
        row = raw_rows(1)[0]
        cleaned, stats = clean_examples([row, dict(row, output="alternative")], alpaca_example)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(stats["duplicate_queries"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate source id"):
            clean_examples([row, dict(row, instruction="different question")], alpaca_example)

    def test_all_readers_accept_local_train_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, reader in (("alpaca_cleaned", alpaca_cleaned), ("alpaca_mt", alpaca_mt),
                                  ("guanaco", guanaco)):
                train, test = raw_rows(6), raw_rows(2, "test")
                if name == "guanaco":
                    train, test = ([dict(text=f"### Human: Train {i}### Assistant: Demo {i}") for i in range(6)],
                                   [dict(text=f"### Human: Test {i}### Assistant: Target {i}") for i in range(2)])
                train_path, test_path = root / "train.json", root / "test.json"
                train_path.write_text(json.dumps(train))
                test_path.write_text(json.dumps(test))
                args = parse_args(["--model", "unused", "--lora", "unused", "--dataset", name,
                                   "--train-file", str(train_path), "--test-file", str(test_path),
                                   "--output", "unused", "--local-files-only"])
                samples, report = load_dataset_samples(args, WordTokenizer(), 2048)
                self.assertEqual(len(samples), 2)
                self.assertEqual(report["split_info"]["test_source"], "provided_test_file")
                self.assertEqual(len({sample["prefix"] for sample in samples}), 1)

    def test_train_only_readers_need_no_holdout_option(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.json"
            train_path.write_text(json.dumps(raw_rows(8)))
            for name in ("alpaca_cleaned", "alpaca_mt"):
                args = parse_args(["--model", "unused", "--lora", "unused", "--dataset", name,
                                   "--train-file", str(train_path), "--output", "unused", "--local-files-only"])
                samples, report = load_dataset_samples(args, WordTokenizer(), 2048)
                self.assertEqual(len(samples), 5)
                self.assertEqual(report["split_info"]["test_source"], "dataset_remainder_after_fewshot")
                self.assertTrue(set(report["fewshot_train_ids"]).isdisjoint(
                    sample["metadata"]["source_id"] for sample in samples))

    def test_cli_argument_conflicts(self):
        with self.assertRaises(SystemExit):
            parse_args(["--model", "x", "--lora", "y", "--dataset", "alpaca_cleaned", "--output", "z",
                        "--test-file", "test.json", "--holdout-fraction", "0.1"])

    def test_export_and_eval_entrypoint(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from test_eval_minikv import tiny_model, attach_lora

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "lora"
            model = tiny_model()
            model.save_pretrained(model_path, safe_serialization=False)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                bos_token="[BOS]", eos_token="[EOS]")
            tokenizer.save_pretrained(model_path)
            attach_lora(model).save_pretrained(adapter_path)
            train_path = root / "train.json"
            train_path.write_text(json.dumps(raw_rows(5)))
            common = ["--model", str(model_path), "--train-file", str(train_path),
                      "--num-shots", "1", "--prefix-max-tokens", "70", "--max-new-tokens", "3",
                      "--local-files-only"]
            script_dir = Path(__file__).parent
            exported = root / "prepared.jsonl"
            process = subprocess.run([sys.executable, str(script_dir / "data/alpaca_cleaned.py"),
                                      *common, "--output", str(exported)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            samples = load_samples(exported)
            self.assertEqual(len(samples), 4)
            prepare_inputs(samples, tokenizer, 3, 128)
            output = root / "scores.json"
            process = subprocess.run([sys.executable, str(script_dir / "eval.py"), *common,
                                      "--lora", str(adapter_path), "--dataset", "alpaca_cleaned",
                                      "--output", str(output), "--device", "cpu", "--dtype", "float32",
                                      "--weight-format", "bin"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["experiment"]["unique_prefixes"], 1)
            self.assertEqual(result["dataset_preparation"]["num_shots"], 1)
            self.assertEqual(len(result["predictions"]), 4)
            self.assertEqual(result["dataset_preparation"]["split_info"]["test_source"],
                             "dataset_remainder_after_fewshot")
            for prediction in result["predictions"]:
                self.assertEqual(set(prediction["predictions"]), {"base", "lora_full", "lora_minikv"})


if __name__ == "__main__":
    unittest.main()
