"""Offline DROP data/metric tests and a real tiny Mistral three-stage run."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from data.drop import (OUTPUT_FORMAT, build_samples, ensure_gold_file, load_gold,
                       prepare, question_prompt, read_split, train_answer)
from drop_score import get_metrics, normalize_answer, prediction_spans, score
from run_mistral_drop import evaluator_command, parse_args
from test_mistral_cnn import CharacterTokenizer


def rows(split, count):
    return [dict(section_id=f"{split}-section-{i}", query_id=f"{split}-query-{i}",
                 passage=f"Complete {split} passage {i} with evidence.",
                 question=f"How many items in example {i}?",
                 answers_spans=dict(spans=[str(i + 1)], types=["number"])) for i in range(count)]


def raw_gold(records):
    result = {}
    for row in records:
        answer = dict(number="", spans=[], date=dict(day="", month="", year=""))
        if row["answers_spans"]["types"][0] == "number":
            answer["number"] = row["answers_spans"]["spans"][0]
        else:
            answer["spans"] = row["answers_spans"]["spans"]
        qa = dict(query_id=row["query_id"], question=row["question"], answer=answer,
                  validated_answers=[answer])
        result.setdefault(row["section_id"], dict(passage=row["passage"], qa_pairs=[]))["qa_pairs"].append(qa)
    return result


def write_split(root, split, records):
    directory = Path(root) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), directory / f"{split}-00000-of-00001.parquet")


class DROPScoreTests(unittest.TestCase):
    def test_number_normalization_and_numeric_f1_gate(self):
        self.assertEqual(get_metrics("7.0", "7"), (1, 1))
        self.assertEqual(get_metrics("7 yards", "5 yards"), (0, 0))
        self.assertEqual(get_metrics("the red red cat", "red cat"), (0, 1))
        self.assertEqual(normalize_answer("The 1,000-yard gain"), "1000.0 yard gain")
        self.assertEqual(get_metrics("", "nonempty"), (0, 0))

    def test_multispan_alignment_is_not_best_individual_span(self):
        self.assertEqual(get_metrics(["Bob", "Alice"], ["Alice", "Bob"]), (1, 1))
        self.assertEqual(get_metrics("Alice", ["Alice", "Bob"]), (0, 0.5))
        sample = dict(metadata=dict(drop_gold_answers=[["Alice", "Bob"]]))
        values = score(['["Bob", "Alice"]'], [["Alice; Bob"]], [sample])
        self.assertEqual(values, dict(exact_match=1, drop_f1=1))
        self.assertEqual(score(["Alice"], [["Alice; Bob"]], [sample]), dict(exact_match=0, drop_f1=0.5))

    def test_alternative_annotators_and_whole_output_policy(self):
        sample = dict(metadata=dict(drop_gold_answers=[["Alice"], ["Bob"]]))
        self.assertEqual(score(["Bob"], [["Alice", "Bob"]], [sample]), dict(exact_match=1, drop_f1=1))
        numeric = dict(metadata=dict(drop_gold_answers=[["3"]]))
        values = score(["3 extra words"], [["3"]], [numeric])
        self.assertEqual(values, dict(exact_match=0, drop_f1=0.5))
        self.assertEqual(prediction_spans('before ["Alice", "Bob"] after'), ('before ["Alice", "Bob"] after',))
        self.assertEqual(prediction_spans("[]"), ("[]",))

    def test_grouped_gold_required_and_macro_score(self):
        with self.assertRaises(ValueError):
            score(["answer"], [["answer"]], [{}])
        with self.assertRaises(ValueError):
            score([], [], [])
        samples = [dict(metadata=dict(drop_gold_answers=[[answer]])) for answer in ("Alice", "Bob")]
        self.assertEqual(score(["Alice", "wrong"], [["Alice"], ["Bob"]], samples),
                         dict(exact_match=0.5, drop_f1=0.5))


class DROPDataTests(unittest.TestCase):
    def test_fixed_train_prefix_seed_and_gold_not_leaked(self):
        train, validation = rows("train", 8), rows("validation", 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            gold = raw_gold(validation)
            for section in gold.values():
                for qa in section["qa_pairs"]:
                    qa["answer"]["number"] = "HELDOUT_ANSWER_NEVER"
            path.write_text(json.dumps(gold))
            kwargs = dict(context_limit=3000, max_new_tokens=64, num_shots=2,
                          prefix_max_tokens=1600, sample_size=4, seed=42)
            samples, report = build_samples(train, validation, load_gold(path), CharacterTokenizer(), **kwargs)
            again, again_report = build_samples(train, validation, load_gold(path), CharacterTokenizer(), **kwargs)
            self.assertEqual((samples, report), (again, again_report))
            self.assertEqual(len(samples), 4)
            self.assertEqual(len({sample["prefix"] for sample in samples}), 1)
            self.assertEqual(len(report["demonstrations"]), 2)
            for sample in samples:
                self.assertTrue(sample["prefix"].startswith(OUTPUT_FORMAT))
                self.assertNotIn("HELDOUT_ANSWER_NEVER", sample["prefix"] + sample["query"])
                self.assertEqual(sample["metadata"]["drop_gold_answers"][0], ["HELDOUT_ANSWER_NEVER"])
                self.assertIn("HELDOUT_ANSWER_NEVER", sample["references"])
                self.assertLessEqual(sample["metadata"]["input_tokens"] + 64, 3000)

    def test_original_groups_preserved_and_gold_mismatch_fails(self):
        validation = rows("validation", 1)
        raw = raw_gold(validation)
        qa = raw[validation[0]["section_id"]]["qa_pairs"][0]
        qa["answer"] = dict(spans=["Alice", "Bob"])
        qa["validated_answers"] = [dict(spans=["Al", "Bob"])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(json.dumps(raw))
            gold = load_gold(path)
            samples, _ = build_samples(rows("train", 4), validation, gold, CharacterTokenizer(),
                                       context_limit=2500, max_new_tokens=64, num_shots=1,
                                       prefix_max_tokens=900, sample_size=1)
            self.assertEqual(samples[0]["metadata"]["drop_gold_answers"], [["Alice", "Bob"], ["Al", "Bob"]])
            validation[0]["passage"] = "Unrelated passage"
            with self.assertRaisesRegex(ValueError, "gold does not match"):
                build_samples(rows("train", 4), validation, gold, CharacterTokenizer(),
                              context_limit=2500, max_new_tokens=64, num_shots=1,
                              prefix_max_tokens=900, sample_size=1)

    def test_no_truncation_overlap_and_repeated_query_id(self):
        train, validation = rows("train", 4), rows("validation", 4)
        validation[1]["passage"] = "a" * 4000
        validation[2]["query_id"] = validation[3]["query_id"]
        validation[2]["section_id"] = validation[3]["section_id"]
        validation[2]["passage"] = validation[3]["passage"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(json.dumps(raw_gold(validation)))
            samples, report = build_samples(train, validation, load_gold(path), CharacterTokenizer(),
                                           context_limit=2000, max_new_tokens=64, num_shots=1,
                                           prefix_max_tokens=900, sample_size=0)
            self.assertEqual(len(samples), 3)
            self.assertEqual(report["validation_overlength"], 1)
            self.assertEqual(report["repeated_source_query_ids"], 1)
            self.assertEqual(len({sample["id"] for sample in samples}), 3)
            for sample in samples:
                row = validation[sample["metadata"]["source_index"]]
                self.assertEqual(sample["query"], question_prompt(row["passage"], row["question"]))
                self.assertFalse(sample["metadata"]["passage_truncated"])
            # An identical train/validation passage is excluded, regardless of query ID.
            overlapping = rows("validation", 1)
            overlapping[0]["passage"] = train[0]["passage"]
            path.write_text(json.dumps(raw_gold(overlapping)))
            with self.assertRaises(ValueError):
                build_samples([train[0]], overlapping, load_gold(path), CharacterTokenizer(),
                              context_limit=2000, max_new_tokens=64, num_shots=1,
                              prefix_max_tokens=900, sample_size=1)

    def test_local_archive_restore_and_no_implicit_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination, archive = root / "gold.json", root / "drop.zip"
            raw = raw_gold(rows("validation", 2))
            with zipfile.ZipFile(archive, "w") as out:
                out.writestr("drop_dataset/drop_dataset_dev.json", json.dumps(raw))
            with patch("data.drop.urlopen", side_effect=AssertionError("unexpected network")):
                with self.assertRaises(FileNotFoundError):
                    ensure_gold_file(root, destination)
                self.assertEqual(ensure_gold_file(root, destination, archive), destination)
                self.assertEqual(json.loads(destination.read_text()), raw)
                self.assertEqual(ensure_gold_file(root, destination), destination)
            self.assertTrue(destination.with_suffix(".source.json").exists())

    def test_preparation_model_limits_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation = rows("train", 4), rows("validation", 4)
            write_split(root, "train", train)
            write_split(root, "validation", validation)
            gold = root / "gold.json"
            gold.write_text(json.dumps(raw_gold(validation)))
            args = parse_args(["--dataset-dir", str(root), "--gold-file", str(gold),
                               "--num-shots", "1", "--prefix-max-tokens", "900", "--sample-size", "2"])
            config = SimpleNamespace(max_position_embeddings=32768, sliding_window=4096)
            destination = root / "samples.jsonl"
            with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                report = prepare(args, destination)
                self.assertEqual(report["context_limit"], 4096)
                self.assertEqual(report["selected"], 2)
                self.assertEqual(len(destination.read_text().splitlines()), 2)
                with self.assertRaises(FileExistsError):
                    prepare(args, destination)
            data, _ = read_split(root, "validation")
            self.assertEqual(data, validation)
            (root / "data/validation-00000-of-00001.parquet").rename(root / "data/validation-00000-of-00002.parquet")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_split(root, "validation")

    def test_train_multispan_is_one_json_answer(self):
        row = rows("train", 1)[0]
        row["answers_spans"] = dict(spans=["Alice", "Bob"], types=["span", "span"])
        self.assertEqual(json.loads(train_answer(row)), ["Alice", "Bob"])


class DROPLauncherTests(unittest.TestCase):
    def test_single_device_command_and_defaults(self):
        args = parse_args([])
        command = evaluator_command(args, "samples.jsonl", "result.json")
        self.assertEqual(args.batch_size, 20)
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.lora, "/data/hub/drop-lora")
        self.assertIn("--minikv-independent-prefill", command)
        self.assertNotIn("--lora-device", command)
        self.assertIn(str(Path(__file__).parent / "drop_score.py") + ":score", command)
        with self.assertRaises(SystemExit):
            parse_args(["--batch-size", "0"])

    def test_cli_runs_three_mixed_length_batches_with_drop_metrics(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
        from test_eval_minikv import attach_lora

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "lora"
            config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=256, sliding_window=256,
                                   bos_token_id=1, eos_token_id=2, pad_token_id=0)
            model = MistralForCausalLM(config).eval()
            model.save_pretrained(model_path, safe_serialization=False)
            vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"word{i}": i for i in range(3, 64)}}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                bos_token="[BOS]", eos_token="[EOS]")
            tokenizer.save_pretrained(model_path)
            attach_lora(model, targets=["q_proj", "v_proj"]).save_pretrained(adapter_path)
            train, validation = rows("train", 4), rows("validation", 3)
            for i, row in enumerate(validation):
                row["passage"] += " word10" * i
            write_split(root, "train", train)
            write_split(root, "validation", validation)
            gold = root / "gold.json"
            gold.write_text(json.dumps(raw_gold(validation)))
            output = root / "results"
            command = [sys.executable, str(Path(__file__).parent / "run_mistral_drop.py"),
                       "--model", str(model_path), "--lora", str(adapter_path), "--dataset-dir", str(root),
                       "--gold-file", str(gold), "--output-dir", str(output), "--device", "cpu",
                       "--dtype", "float32", "--sample-size", "3", "--batch-size", "3",
                       "--num-shots", "1", "--prefix-max-tokens", "180", "--max-new-tokens", "2"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertTrue(result["experiment"]["minikv_independent_prefill"])
            self.assertFalse(result["experiment"]["parallel_base_lora"])
            self.assertEqual(result["dataset_preparation"]["evaluation_split"], "validation")
            self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(set(result["scores"][mode]), {"exact_match", "drop_f1"})
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
            journal = [json.loads(line) for line in (output / "result.predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["mode"] for row in journal], ["base", "lora_full", "lora_minikv"])
            self.assertTrue(all(len(row["records"]) == 3 for row in journal))


if __name__ == "__main__":
    unittest.main()
