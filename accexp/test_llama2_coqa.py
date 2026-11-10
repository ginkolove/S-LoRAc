"""Offline CoQA annotations, scoring, shared sampling and three-stage integration tests."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from coqa_score import extract_answer, score, score_unfiltered, turn_scores
from data.coqa import (build_samples, conversation_key, load_gold, parse_args as data_args,
                       prepare, question_prompt, read_split)
from run_llama2_coqa import evaluator_command, load_prepared, main, parse_args
from test_mistral_cnn import CharacterTokenizer


def conversations(split, count):
    return [dict(source="mctest", story=f"Complete {split} story {i}.",
                 questions=[f"{split}_{i}_QUESTION_{j}" for j in range(3)],
                 answers=dict(input_text=[f"{split}_{i}_ANSWER_{j}" for j in range(3)])) for i in range(count)]


def original_gold(rows):
    result = []
    for i, row in enumerate(rows):
        answers = [dict(turn_id=j + 1, input_text=a) for j, a in enumerate(row["answers"]["input_text"])]
        result.append(dict(id=f"story-{i}", source=row["source"], story=row["story"],
                           questions=[dict(turn_id=j + 1, input_text=q) for j, q in enumerate(row["questions"])],
                           answers=answers, additional_answers={str(j): answers for j in range(3)}))
    return dict(data=result)


def gold_map(rows):
    return {conversation_key(r["story"], r["questions"], r["answers"]["input_text"]):
            dict(id=f"story-{i}", source=r["source"], references=[[a] * 4 for a in r["answers"]["input_text"]])
            for i, r in enumerate(rows)}


def write_data(root, train, validation):
    folder = root / "data"
    folder.mkdir()
    for split, rows in (("train", train), ("validation", validation)):
        pq.write_table(pa.Table.from_pylist(rows), folder / f"{split}-00000-of-00001.parquet")
    gold = root / "gold.json"
    gold.write_text(json.dumps(original_gold(validation)))
    return gold


class CoQAScoreTests(unittest.TestCase):
    def test_official_leave_one_out_not_simple_max(self):
        self.assertEqual(turn_scores("blue", ("blue", "red", "red", "red")), (0.75, 0.75))
        self.assertEqual(turn_scores("blue", ("blue", "blue", "red", "red")), (1, 1))
        self.assertEqual(turn_scores("blue", ("blue", "red")), (0.5, 0.5))
        self.assertEqual(turn_scores("The cat!", ("cat",)), (1, 1))
        self.assertEqual(turn_scores("", ("",)), (1, 1))
        self.assertEqual(turn_scores("unknown", ("unknown",) * 4), (1, 1))

    def test_filter_only_next_turn_same_for_all_modes(self):
        raw = "blue\n\nQ: Next question?\nA: red"
        refs = ["blue"] * 4
        sample = dict(metadata=dict(coqa_gold_answers=refs))
        for mode in ("base", "lora_full", "lora_minikv"):
            self.assertEqual(score([raw], [refs], [dict(sample, mode=mode)]), dict(exact_match=1, coqa_f1=1))
        self.assertEqual(score_unfiltered([raw], [refs], [sample])["exact_match"], 0)
        # Never pick a later correct response or the best matching substring.
        self.assertEqual(score(["red\nQ: Next?\nA: blue"], [refs], [sample])["coqa_f1"], 0)
        self.assertEqual(extract_answer("first\nsecond")["answer_text"], "first\nsecond")
        self.assertEqual(extract_answer("inline Q: text")["answer_text"], "inline Q: text")
        self.assertEqual(extract_answer("Q: Next?\nA: blue")["answer_text"], "")

    def test_invalid_and_inconsistent_references_fail(self):
        for refs, sample in ((["blue"], {}), (["blue"], dict(metadata=dict(coqa_gold_answers=["red"])) )):
            with self.assertRaises(ValueError):
                score(["blue"], [refs], [sample])
        with self.assertRaises(ValueError):
            score([], [], [])


class CoQADataTests(unittest.TestCase):
    def test_history_excludes_current_and_future_answers(self):
        row = conversations("dev", 1)[0]
        text = question_prompt(row, 1)
        self.assertIn(row["story"], text)
        self.assertIn(row["questions"][0], text)
        self.assertIn(row["answers"]["input_text"][0], text)
        self.assertIn(row["questions"][1], text)
        self.assertNotIn(row["answers"]["input_text"][1], text)
        self.assertNotIn(row["questions"][2], text)
        self.assertNotIn(row["answers"]["input_text"][2], text)
        self.assertTrue(text.endswith("\nA:"))

    def test_fixed_train_prefix_seeded_turn_sampling_joint_model_limits(self):
        train, dev = conversations("train", 8), conversations("dev", 8)
        kwargs = dict(max_new_tokens=64, num_shots=3, prefix_max_tokens=1200, sample_size=7, seed=42)
        tokenizers = dict(small=CharacterTokenizer(), large=CharacterTokenizer())
        samples, report = build_samples(train, dev, gold_map(dev), tokenizers, dict(small=3000, large=2000), **kwargs)
        self.assertEqual((samples, report), build_samples(train, dev, gold_map(dev), tokenizers,
                                                        dict(small=3000, large=2000), **kwargs))
        self.assertEqual(len(samples), 7)
        self.assertEqual(len({s["prefix"] for s in samples}), 1)
        self.assertEqual(len(report["demonstrations"]), 3)
        for sample in samples:
            self.assertNotIn("dev_", sample["prefix"])
            self.assertEqual(len(sample["references"]), 4)
            self.assertEqual(sample["references"], sample["metadata"]["coqa_gold_answers"])
            self.assertLessEqual(sample["metadata"]["input_tokens_by_model"]["large"] + 64, 2000)
        dev[0]["story"] += " long" * 400
        _, filtered = build_samples(train, dev, gold_map(dev), tokenizers, dict(small=5000, large=2000), **kwargs)
        self.assertEqual(filtered["excluded"]["overlength_turns"], 3)

    def test_demo_overlap_and_gold_mismatch_are_not_silently_accepted(self):
        train = conversations("train", 1)
        dev = train + conversations("dev", 2)
        kwargs = dict(max_new_tokens=64, num_shots=1, prefix_max_tokens=800, sample_size=0)
        samples, report = build_samples(train, dev, gold_map(dev), {"m": CharacterTokenizer()}, {"m": 2000}, **kwargs)
        self.assertEqual(len(samples), 6)
        self.assertEqual(report["excluded"]["demo_overlap_turns"], 3)
        with self.assertRaisesRegex(ValueError, "do not match"):
            build_samples(train, dev, {}, {"m": CharacterTokenizer()}, {"m": 2000}, **kwargs)

    def test_shared_story_different_conversations_match_by_full_identity(self):
        dev = conversations("dev", 2)
        dev[1]["story"] = dev[0]["story"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_data(root, conversations("train", 2), dev)
            gold = load_gold(path)
            self.assertEqual(len(gold), 2)
            self.assertEqual(len(read_split(root, "validation")[0]), 2)
            raw = json.loads(path.read_text())
            raw["data"][0]["additional_answers"]["0"][0]["turn_id"] = 2
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "turn IDs"):
                load_gold(path)
            (root / "data/validation-00000-of-00001.parquet").rename(root / "data/validation-00000-of-00002.parquet")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_split(root, "validation")

    def test_preparation_hash_and_budget_are_checked_by_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = write_data(root, conversations("train", 3), conversations("dev", 2))
            path = root / "prepared.jsonl"
            args = data_args(["--models", str(root / "model"), "--dataset-dir", str(root), "--gold-file", str(gold),
                              "--sample-size", "3", "--num-shots", "1", "--prefix-max-tokens", "800", "--output", str(path)])
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=2000)), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                prepare(args, path)
                with self.assertRaises(FileExistsError):
                    prepare(args, path)
            launch = parse_args(["--size", "7b", "--model", str(root / "model"), "--data", str(path), "--output-dir", str(root / "out")])
            self.assertEqual(len(load_prepared(launch)[0]), 3)
            launch.max_new_tokens = 128
            with self.assertRaisesRegex(ValueError, "Output budget"):
                load_prepared(launch)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(launch)


class CoQALauncherTests(unittest.TestCase):
    def test_paths_formats_and_single_device(self):
        for size, batch, fmt in (("7b", 20, "safetensors"), ("13b", 8, "bin")):
            args = parse_args(["--size", size, "--data", "data.jsonl", "--output-dir", "out"])
            self.assertEqual(args.batch_size, batch)
            self.assertEqual(args.weight_format, fmt)
            self.assertEqual(args.lora, f"/data/hub/coqa-lora/llama2-{size}")
            args.max_new_tokens = 64
            command = evaluator_command(args, "out/result.json", 42)
            self.assertIn("--minikv-independent-prefill", command)
            self.assertNotIn("--lora-device", command)
        with self.assertRaises(SystemExit):
            parse_args(["--size", "7b", "--data", "data.jsonl", "--output-dir", "out", "--batch-size", "0"])

    def test_tiny_llama_all_three_stages_and_postprocessing(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from test_eval_minikv import attach_lora

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "adapter"
            model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
            model.save_pretrained(model_path)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
                                           **{f"word{i}": i for i in range(3, 64)}}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
                                    eos_token="[EOS]").save_pretrained(model_path)
            adapted = attach_lora(model, targets=["q_proj", "v_proj"])
            adapted.peft_config["default"].base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
            adapted.save_pretrained(adapter_path, save_embedding_layers=False)
            gold = write_data(root, conversations("train", 2), conversations("dev", 2))
            data = root / "samples.jsonl"
            prep = data_args(["--models", str(model_path), "--dataset-dir", str(root), "--gold-file", str(gold),
                              "--sample-size", "3", "--num-shots", "1", "--prefix-max-tokens", "180",
                              "--max-new-tokens", "2", "--output", str(data)])
            prepare(prep, data)
            output = root / "out"
            # Validate the launcher with a tiny real model in its subprocess; production shape checking stays strict.
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                    model_type="llama", hidden_size=4096, num_hidden_layers=32, num_attention_heads=32)), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"):
                main(["--size", "7b", "--model", str(model_path), "--lora", str(adapter_path), "--data", str(data),
                      "--device", "cpu", "--dtype", "float32", "--batch-size", "3", "--output-dir", str(output)])
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(set(result["scores"][mode]), {"exact_match", "coqa_f1"})
                self.assertIn(mode, result["unfiltered_scores"])
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
                self.assertTrue(all("answer_extraction" in row["predictions"][mode] for row in result["predictions"]))


if __name__ == "__main__":
    unittest.main()
