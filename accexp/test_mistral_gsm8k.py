"""Offline GSM8K extraction, local data, and real tiny Mistral pipeline tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from data.gsm8k import build_samples, prepare, question_prompt, read_split
from gsm8k_score import extract_prediction, gold_number, normalize_number, score, score_unfiltered
from run_mistral_gsm8k import evaluator_command, parse_args
from test_mistral_cnn import CharacterTokenizer


def rows(split, count):
    return [dict(question=f"In {split} example {i}, what is 1 plus {i}?",
                 answer=f"Add 1 and {i}: <<1+{i}={i + 1}>>{i + 1}.\n#### {i + 1}") for i in range(count)]


def write_split(root, split, records, config="main"):
    directory = Path(root) / config
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), directory / f"{split}-00000-of-00001.parquet")


class GSMScoreTests(unittest.TestCase):
    def test_decimal_normalization_not_float_or_expression_matching(self):
        for text, expected in (("$1,200.00", "1200"), ("-0.00", "0"), ("+3.5", "3.5"),
                               ("18.", "18"), ("-2.50", "-2.5")):
            self.assertEqual(normalize_number(text), expected)
        for text in ("2+3", "1/2", "1e3", "nan", "Infinity", "12,34", "4 apples"):
            self.assertIsNone(normalize_number(text))
        self.assertEqual(score(["#### 9007199254740993"], [["9007199254740992"]], [{}])["exact_match"], 0)

    def test_gold_final_number_and_invalid_annotations_fail(self):
        self.assertEqual(gold_number("Reasoning 123 and 45.\n#### 1,200"), "1200")
        for text in ("No marker 72", "#### 72\n#### 73", "#### unknown"):
            with self.assertRaises(ValueError):
                gold_number(text)

    def test_first_answer_binding_and_no_best_correct_substring(self):
        for text in ("#### 7\n#### 3", "#### 2+1", "#### wrong\n#### 3", "#### 3/4", "#### 3e2"):
            values = score([text], [["3"]], [{}])
            self.assertEqual(values, dict(exact_match=0, flexible_exact_match=0))
        for text in ("#### 3 hours", "#### $3.00 dollars", "#### 3."):
            self.assertEqual(score([text], [["3"]], [{}]), dict(exact_match=1, flexible_exact_match=1))
        # A correct intermediate number cannot override the final declared answer.
        self.assertEqual(score(["We find 3, but conclude\n#### 7"], [["3"]], [{}])["exact_match"], 0)

    def test_multiline_reasoning_and_next_question_boundary(self):
        answer = "First compute 3 * 3 = 9.\nThen compute 9 * 60 = 540.\n#### 540"
        text = answer + "\n\nPlease answer the following question: Another problem?\nAnswer:\n#### 7"
        value = extract_prediction(text)
        self.assertEqual(value["answer_text"], answer)
        self.assertEqual(value["strict"], "540")
        self.assertTrue(value["boundary_trimmed"])
        self.assertEqual(score([text], [["540"]], [{}]), dict(exact_match=1, flexible_exact_match=1))
        # Do not accidentally borrow the answer to the following problem.
        missing = "I calculate 2.\n\nQuestion: Next problem\nAnswer:\n#### 7"
        self.assertEqual(score([missing], [["7"]], [{}]), dict(exact_match=0, flexible_exact_match=0))
        self.assertEqual(score_unfiltered([missing], [["7"]], [{}])["exact_match"], 1)

    def test_flexible_is_separate_and_same_rule_for_every_mode(self):
        text = "First 4, then the final result is 12."
        expected = dict(exact_match=0, flexible_exact_match=1)
        for mode in ("base", "lora_full", "lora_minikv"):
            self.assertEqual(score([text], [["12"]], [dict(mode=mode)]), expected)
        self.assertEqual(score([text], [["4"]], [{}]), dict(exact_match=0, flexible_exact_match=0))
        self.assertIsNone(extract_prediction("This has no numeric answer")["flexible"])
        with self.assertRaises(ValueError):
            score(["#### 3"], [["Full reasoning #### 3"]], [{}])
        with self.assertRaises(ValueError):
            score([], [], [])

    def test_extraction_cache_cannot_be_mutated_by_a_consumer(self):
        parsed = extract_prediction("#### 7")
        parsed["strict"] = "3"
        self.assertEqual(extract_prediction("#### 7")["strict"], "7")
        self.assertEqual(score(["#### 7"], [["3"]], [{}])["exact_match"], 0)


class GSMDataTests(unittest.TestCase):
    def test_seeded_test_subset_and_fixed_complete_train_prefix(self):
        train, test = rows("train", 8), rows("test", 12)
        for row in test:
            row["answer"] = "HELDOUT_REASONING_NEVER_IN_PROMPT\n#### 987654321"
        kwargs = dict(context_limit=3000, max_new_tokens=64, num_shots=3,
                      prefix_max_tokens=1200, sample_size=5, seed=42)
        samples, report = build_samples(train, test, CharacterTokenizer(), **kwargs)
        again = build_samples(train, test, CharacterTokenizer(), **kwargs)
        self.assertEqual((samples, report), again)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len({sample["prefix"] for sample in samples}), 1)
        self.assertEqual(len(report["demonstrations"]), 3)
        self.assertEqual(samples[0]["prefix"].count("####"), 3)
        for sample in samples:
            self.assertNotIn("HELDOUT_REASONING", sample["prefix"] + sample["query"])
            self.assertNotIn("987654321", sample["prefix"] + sample["query"])
            self.assertEqual(sample["references"], ["987654321"])
            self.assertEqual(sample["metadata"]["gold_solution"], test[sample["metadata"]["source_index"]]["answer"])
            self.assertEqual(sample["query"], question_prompt(test[sample["metadata"]["source_index"]]["question"]))

    def test_no_question_truncation_overlap_and_bad_gold(self):
        train, test = rows("train", 1), rows("test", 4)
        test[0]["question"] = train[0]["question"]
        test[1]["question"] = "long question " * 500
        samples, report = build_samples(train, test, CharacterTokenizer(), context_limit=1800,
                                       max_new_tokens=64, num_shots=1, prefix_max_tokens=800, sample_size=0)
        self.assertEqual(len(samples), 2)
        self.assertEqual(report["test_demo_overlap"], 1)
        self.assertEqual(report["test_overlength"], 1)
        self.assertTrue(all(not sample["metadata"]["question_truncated"] for sample in samples))
        with self.assertRaises(ValueError):
            build_samples(train, test, CharacterTokenizer(), context_limit=1800, max_new_tokens=64,
                          num_shots=1, prefix_max_tokens=800, sample_size=3)
        test[2]["answer"] = "malformed gold without final marker"
        with self.assertRaisesRegex(ValueError, "gold solution"):
            build_samples(train, test, CharacterTokenizer(), context_limit=1800, max_new_tokens=64,
                          num_shots=1, prefix_max_tokens=800, sample_size=0)

    def test_config_selection_complete_shards_and_duplicate_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            main, socratic = rows("main", 3), rows("socratic", 2)
            write_split(directory, "test", main)
            write_split(directory, "test", socratic, config="socratic")
            self.assertEqual(read_split(directory, "test")[0], main)
            self.assertEqual(read_split(directory, "test", "socratic")[0], socratic)
            with self.assertRaises(ValueError):
                read_split(Path(directory) / "main", "test", "socratic")
            (Path(directory) / "main/test-00000-of-00001.parquet").rename(
                Path(directory) / "main/test-00000-of-00002.parquet")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_split(directory, "test")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_samples(rows("train", 1), [main[0], main[0]], CharacterTokenizer(),
                          context_limit=1800, max_new_tokens=64, num_shots=1,
                          prefix_max_tokens=800, sample_size=0)

    def test_prepare_reads_model_context_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "train", rows("train", 4))
            write_split(root, "test", rows("test", 4))
            args = parse_args(["--dataset-dir", str(root), "--sample-size", "2",
                               "--num-shots", "1", "--prefix-max-tokens", "800"])
            config = SimpleNamespace(max_position_embeddings=32768, sliding_window=4096)
            with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                destination = root / "samples.jsonl"
                report = prepare(args, destination)
                self.assertEqual(report["context_limit"], 4096)
                self.assertEqual(report["selected"], 2)
                self.assertEqual(len(destination.read_text().splitlines()), 2)
                with self.assertRaises(FileExistsError):
                    prepare(args, destination)


class GSMLauncherTests(unittest.TestCase):
    def test_single_device_command_defaults_and_metric_plugin(self):
        args = parse_args([])
        command = evaluator_command(args, "samples.jsonl", "result.json")
        self.assertEqual(args.batch_size, 20)
        self.assertEqual(args.sample_size, 1000)
        self.assertEqual(args.dataset_config, "main")
        self.assertEqual(args.lora, "/data/hub/gsm8k-lora")
        self.assertIn("--minikv-independent-prefill", command)
        self.assertNotIn("--lora-device", command)
        self.assertIn(str(Path(__file__).parent / "gsm8k_score.py") + ":score", command)
        with self.assertRaises(SystemExit):
            parse_args(["--batch-size", "0"])

    def test_real_tiny_mistral_end_to_end_all_stages(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
        from test_eval_minikv import attach_lora

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "adapter"
            config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=256, sliding_window=256,
                                   bos_token_id=1, eos_token_id=2, pad_token_id=0)
            model = MistralForCausalLM(config).eval()
            model.save_pretrained(model_path, safe_serialization=False)
            vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"word{i}": i for i in range(3, 64)}}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
                                    eos_token="[EOS]").save_pretrained(model_path)
            attach_lora(model, targets=["q_proj", "v_proj"]).save_pretrained(adapter_path)
            train, test = rows("train", 4), rows("test", 3)
            for i, row in enumerate(test):
                row["question"] += " word10" * i
            write_split(root, "train", train)
            write_split(root, "test", test)
            output = root / "results"
            command = [sys.executable, str(Path(__file__).parent / "run_mistral_gsm8k.py"),
                       "--model", str(model_path), "--lora", str(adapter_path), "--dataset-dir", str(root),
                       "--output-dir", str(output), "--device", "cpu", "--dtype", "float32",
                       "--sample-size", "3", "--batch-size", "3", "--num-shots", "1",
                       "--prefix-max-tokens", "180", "--max-new-tokens", "2"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertTrue(result["experiment"]["minikv_independent_prefill"])
            self.assertFalse(result["experiment"]["parallel_base_lora"])
            self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            self.assertEqual(result["scorer_metadata"]["primary_metric"], "exact_match")
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(set(result["scores"][mode]), {"exact_match", "flexible_exact_match"})
                self.assertIn(mode, result["unfiltered_scores"])
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
                self.assertTrue(all("answer_extraction" in row["predictions"][mode] for row in result["predictions"]))
            journal = [json.loads(line) for line in (output / "result.predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["mode"] for row in journal], ["base", "lora_full", "lora_minikv"])
            self.assertTrue(all(len(row["records"]) == 3 for row in journal))


if __name__ == "__main__":
    unittest.main()
