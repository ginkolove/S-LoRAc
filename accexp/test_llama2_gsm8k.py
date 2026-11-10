"""Shared Llama-2 GSM8K inputs, numeric extraction and three-stage integration."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from data.gsm8k import build_samples, question_prompt as mistral_prompt
from data.llama2_gsm8k import (PROMPT_FORMAT, demonstration_answer, parse_args as data_args,
                              prepare, question_prompt)
from gsm8k_score import score as mistral_score
from llama2_gsm8k_score import extract_prediction, score, score_unfiltered
from run_llama2_gsm8k import evaluator_command, load_prepared, main, parse_args
from test_mistral_cnn import CharacterTokenizer
from test_mistral_gsm8k import rows, write_split


class ExtraLongTokenizer(CharacterTokenizer):
    def encode(self, text, *args, **kwargs):
        result = super().encode(text, *args, **kwargs)
        return result * (10 if "In test example 0" in text else 1)


class Llama2GSMScoreTests(unittest.TestCase):
    def test_lema_prompt_and_complete_reasoning_format(self):
        self.assertEqual(question_prompt("How many?"),
            "Please solve the following math problem.\nQuestion: How many?\nAnswer: Let's think step by step.\n")
        solution = "First 2 plus 3 is <<2+3=5>>5.\nThen double: <<5*2=10>>10.\n#### 10"
        self.assertEqual(demonstration_answer(solution),
                         "First 2 plus 3 is 5.\nThen double: 10.\nThe answer is 10.")

    def test_first_declared_answer_binding_and_decimal_equality(self):
        for raw in ("Reasoning 99. The answer is $1,200.00 dollars.", "THE ANSWER IS: 1200.",
                    "The final answer is 1200.", "#### 1,200"):
            self.assertEqual(score([raw], [["1200"]], [{}]), dict(exact_match=1, flexible_exact_match=1))
        for raw in ("The answer is 7.\nThe answer is 3.", "The answer is wrong\n#### 3",
                    "The answer is 2+1", "#### 3/4", "The answer is 3e2", "#### 7\nThe answer is 3"):
            self.assertEqual(score([raw], [["3"]], [{}]), dict(exact_match=0, flexible_exact_match=0))
        self.assertEqual(score(["The answer is -0.00."], [["0"]], [{}])["exact_match"], 1)
        self.assertEqual(score(["The answer is 9007199254740993."], [["9007199254740992"]], [{}])["exact_match"], 0)
        with self.assertRaises(ValueError):
            score(["The answer is 3."], [["reasoning #### 3"]], [{}])
        with self.assertRaises(ValueError):
            extract_prediction(None)

    def test_next_question_boundary_no_oracle_and_mistral_unchanged(self):
        raw = "Compute 2.\n\n" + question_prompt("Next?") + "The answer is 7."
        for mode in ("base", "lora_full", "lora_minikv"):
            self.assertEqual(score([raw], [["7"]], [dict(mode=mode)]), dict(exact_match=0, flexible_exact_match=0))
        self.assertEqual(score_unfiltered([raw], [["7"]], [{}])["exact_match"], 1)
        self.assertTrue(extract_prediction(raw)["boundary_trimmed"])
        self.assertEqual(score(["I calculate 12."], [["12"]], [{}]), dict(exact_match=0, flexible_exact_match=1))
        parsed = extract_prediction("The answer is 7.")
        parsed["strict"] = "3"
        self.assertEqual(extract_prediction("The answer is 7.")["strict"], "7")
        self.assertEqual(mistral_score(["The answer is 3."], [["3"]], [{}]), dict(exact_match=0, flexible_exact_match=1))


class Llama2GSMDataTests(unittest.TestCase):
    def test_joint_length_filtering_and_existing_defaults_unchanged(self):
        train, test = rows("train", 6), rows("test", 6)
        kwargs = dict(context_limit=1500, max_new_tokens=64, num_shots=1,
                      prefix_max_tokens=700, sample_size=0, seed=42)
        ordinary = build_samples(train, test, CharacterTokenizer(), **kwargs)
        self.assertEqual(ordinary, build_samples(train, test, CharacterTokenizer(), additional_tokenizers=(), **kwargs))
        self.assertEqual(ordinary[0][0]["query"], mistral_prompt(test[0]["question"]))
        options = dict(additional_tokenizers=[ExtraLongTokenizer()], prompt_builder=question_prompt,
                       answer_formatter=demonstration_answer, prompt_format=PROMPT_FORMAT, **kwargs)
        shared = build_samples(train, test, CharacterTokenizer(), **options)
        self.assertEqual(shared[1]["test_overlength"], 1)
        self.assertEqual(len(shared[0]), 5)
        self.assertTrue(all(s["metadata"]["source_index"] != 0 for s in shared[0]))
        self.assertEqual(shared, build_samples(train, test, CharacterTokenizer(), **options))

    def test_shared_preparation_references_hashes_and_actual_shot_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "train", rows("train", 8))
            test = rows("test", 12)
            for row in test:
                row["answer"] = "HELDOUT_REASONING_NEVER_IN_PROMPT\n#### 987654321"
            write_split(root, "test", test)
            models = [str(root / name) for name in ("model7", "model13")]
            def arguments(path):
                return data_args(["--models", *models, "--dataset-dir", str(root), "--sample-size", "5",
                    "--prefix-max-tokens", "1200", "--max-new-tokens", "64", "--output", str(path)])
            path, repeated = root / "samples.jsonl", root / "repeated.jsonl"
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=2500)), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                report = prepare(arguments(path), path)
                self.assertEqual(report, prepare(arguments(repeated), repeated))
                self.assertEqual(path.read_bytes(), repeated.read_bytes())
                with self.assertRaises(FileExistsError):
                    prepare(arguments(path), path)
            self.assertEqual(report["num_shots"], 3)
            self.assertEqual(len(report["demonstrations"]), 3)
            self.assertEqual(set(report["context_limits"]), set(models))
            for size, model in zip(("7b", "13b"), models):
                args = parse_args(["--size", size, "--model", model, "--data", str(path), "--output-dir", str(root / size)])
                samples, _ = load_prepared(args)
                self.assertEqual(len(samples), 5)
                self.assertEqual(args.num_shots, 3)
                self.assertEqual(args.prefix_max_tokens, 1200)
                self.assertEqual(len({s["prefix"] for s in samples}), 1)
                for sample in samples:
                    self.assertEqual(sample["prefix"].count("The answer is"), 3)
                    self.assertNotIn("HELDOUT_REASONING", sample["prefix"] + sample["query"])
                    self.assertNotIn("987654321", sample["prefix"] + sample["query"])
                    self.assertEqual(sample["references"], ["987654321"])
                args.max_new_tokens = 128
                with self.assertRaisesRegex(ValueError, "Output budget"):
                    load_prepared(args)
            args.max_new_tokens = None
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(args)
            report["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["sample_ids"].reverse()
            Path(str(path) + ".meta.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "sample IDs"):
                load_prepared(args)


class Llama2GSMLauncherTests(unittest.TestCase):
    def test_paths_formats_and_single_device_command(self):
        for size, batch, fmt in (("7b", 8, "safetensors"), ("13b", 4, "bin")):
            args = parse_args(["--size", size, "--data", "samples.jsonl", "--output-dir", "out"])
            self.assertEqual(args.batch_size, batch)
            self.assertEqual(args.weight_format, fmt)
            self.assertEqual(args.lora, f"/data/hub/gsm8k-lora/llama2-{size}")
            args.max_new_tokens, args.num_shots, args.prefix_max_tokens = 256, 3, 1536
            command = evaluator_command(args, "out/result.json", 42)
            self.assertIn("--minikv-independent-prefill", command)
            self.assertIn(str(Path(__file__).parent / "llama2_gsm8k_score.py") + ":score", command)
            self.assertNotIn("--lora-device", command)
            self.assertEqual(command[command.index("--num-shots") + 1], "3")
        with self.assertRaises(SystemExit):
            parse_args(["--size", "7b", "--data", "samples.jsonl", "--output-dir", "out", "--batch-size", "0"])

    def test_tiny_llama_all_modules_three_stages_both_size_launchers(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from test_eval_minikv import attach_lora

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
            model.save_pretrained(model_path)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
                **{f"word{i}": i for i in range(3, 64)}}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
                eos_token="[EOS]").save_pretrained(model_path)
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            adapted = attach_lora(model, targets=targets)
            write_split(root, "train", rows("train", 4))
            test = rows("test", 3)
            for index, row in enumerate(test):
                row["question"] += " word10" * index
            write_split(root, "test", test)
            path = root / "samples.jsonl"
            prepare(data_args(["--models", str(model_path), "--dataset-dir", str(root), "--sample-size", "3",
                "--num-shots", "1", "--prefix-max-tokens", "180", "--max-new-tokens", "2", "--output", str(path)]), path)
            for size, shape in (("7b", (4096, 32, 32)), ("13b", (5120, 40, 40))):
                adapter_path = root / ("adapter" + size)
                adapted.peft_config["default"].base_model_name_or_path = f"meta-llama/Llama-2-{size}-hf"
                adapted.save_pretrained(adapter_path, save_embedding_layers=False)
                output = root / size
                with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                        model_type="llama", hidden_size=shape[0], num_hidden_layers=shape[1], num_attention_heads=shape[2])), \
                        patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"):
                    main(["--size", size, "--model", str(model_path), "--lora", str(adapter_path), "--data", str(path),
                          "--device", "cpu", "--dtype", "float32", "--weight-format", "safetensors",
                          "--batch-size", "2", "--output-dir", str(output)])
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
                self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                                 dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
                self.assertEqual(result["config"]["num_shots"], 1)
                self.assertEqual(result["config"]["prefix_max_tokens"], 180)
                self.assertEqual(result["config"]["prompt_protocol"], "lema_fixed_fewshot")
                self.assertEqual(result["scorer_metadata"]["primary_metric"], "exact_match")
                self.assertFalse(result["experiment"]["parallel_base_lora"])
                self.assertTrue(result["experiment"]["minikv_independent_prefill"])
                for mode in ("base", "lora_full", "lora_minikv"):
                    self.assertEqual(set(result["scores"][mode]), {"exact_match", "flexible_exact_match"})
                    self.assertIn(mode, result["unfiltered_scores"])
                    self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
                    self.assertTrue(all("answer_extraction" in r["predictions"][mode] for r in result["predictions"]))


if __name__ == "__main__":
    unittest.main()
