"""Shared CNN/DailyMail sampling, reference-independent scoring and Llama stage integration."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cnn_dailymail_score import extract_summary, score, score_turn_filtered
from data.cnn_dailymail import build_samples
from data.llama2_cnn import parse_args as data_args, prepare
from run_llama2_cnn import evaluator_command, load_prepared, main, parse_args
from test_mistral_cnn import CharacterTokenizer, rows, write_split


class ExtraLongTokenizer(CharacterTokenizer):
    def encode(self, text, *args, **kwargs):
        tokens = super().encode(text, *args, **kwargs)
        return tokens * (3 if "test article number 0" in text else 1)


class Llama2CNNTests(unittest.TestCase):
    def test_summary_filter_is_uniform_and_retains_paragraphs(self):
        summary = "Alice won the race.\nBob came second."
        raw = summary + "\n\n### Article: unrelated next article\n\n### Summary: different"
        for mode in ("base", "lora_full", "lora_minikv"):
            self.assertEqual(score_turn_filtered([raw], [[summary]], [dict(mode=mode)]),
                             dict(rouge1=1, rouge2=1, rougeL=1))
        self.assertLess(score([raw], [[summary]], [{}])["rougeL"], 1)
        wrong = "Unrelated words\n### Article: next\n### Summary: " + summary
        self.assertEqual(score_turn_filtered([wrong], [[summary]], [{}])["rougeL"], 0)
        self.assertEqual(extract_summary(summary)["summary_text"], summary)
        self.assertEqual(extract_summary("inline Article: text")["summary_text"], "inline Article: text")
        self.assertEqual(extract_summary("### Article: next")["summary_text"], "")
        with self.assertRaises(ValueError):
            extract_summary(None)

    def test_joint_model_limits_before_sampling_and_legacy_defaults_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            train = write_split(directory, "train", rows("train", 6))
            test = write_split(directory, "test", rows("test", 6))
            kwargs = dict(context_limit=1000, max_new_tokens=64, num_shots=1,
                          prefix_max_tokens=700, sample_size=0, seed=42)
            ordinary = build_samples(train, test, CharacterTokenizer(), **kwargs)
            self.assertEqual(ordinary, build_samples(train, test, CharacterTokenizer(),
                                                    additional_tokenizers=(), **kwargs))
            joint = build_samples(train, test, CharacterTokenizer(),
                                  additional_tokenizers=[ExtraLongTokenizer()], **kwargs)
            self.assertEqual(len(ordinary[0]), 6)
            self.assertEqual(len(joint[0]), 5)
            self.assertEqual(joint[1]["test_overlength"], 1)
            self.assertTrue(all(s["metadata"]["source_index"] != 0 for s in joint[0]))
            self.assertEqual(joint, build_samples(train, test, CharacterTokenizer(),
                additional_tokenizers=[ExtraLongTokenizer()], **kwargs))
            self.assertEqual(joint[0][0]["prefix"], ordinary[0][0]["prefix"])

    def test_shared_data_ids_hash_output_budget_and_untruncated_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "train", rows("train", 6))
            test = rows("test", 6)
            test[0]["highlights"] += " long reference" * 1000
            write_split(root, "test", test)
            path = root / "samples.jsonl"
            models = [str(root / name) for name in ("model7", "model13")]
            args = data_args(["--models", *models, "--dataset-dir", str(root), "--num-shots", "1",
                              "--sample-size", "0", "--prefix-max-tokens", "700", "--output", str(path)])
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=2200)), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                report = prepare(args, path)
                with self.assertRaises(FileExistsError):
                    prepare(args, path)
            self.assertEqual(set(report["context_limits"]), set(models))
            for size, model in zip(("7b", "13b"), models):
                launch = parse_args(["--size", size, "--model", model, "--data", str(path), "--output-dir", str(root / size)])
                samples, _ = load_prepared(launch)
                self.assertEqual([s["id"] for s in samples], report["sample_ids"])
                self.assertEqual(len({s["prefix"] for s in samples}), 1)
                self.assertEqual(samples[0]["references"], [test[0]["highlights"]])
                for sample in samples:
                    self.assertNotIn(sample["references"][0], sample["prefix"] + sample["query"])
                launch.max_new_tokens = 64
                with self.assertRaisesRegex(ValueError, "Output budget"):
                    load_prepared(launch)
            launch.max_new_tokens = None
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(launch)
            report["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["sample_ids"].reverse()
            Path(str(path) + ".meta.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "sample IDs"):
                load_prepared(launch)

    def test_paths_formats_and_sequential_command(self):
        for size, batch, fmt in (("7b", 8, "safetensors"), ("13b", 4, "bin")):
            args = parse_args(["--size", size, "--data", "samples.jsonl", "--output-dir", "out"])
            self.assertEqual(args.batch_size, batch)
            self.assertEqual(args.weight_format, fmt)
            self.assertEqual(args.lora, f"/data/hub/cnn_dailmail-lora/llama2-{size}")
            args.max_new_tokens = 256
            command = evaluator_command(args, "out/result.json", 42)
            self.assertIn("--minikv-independent-prefill", command)
            self.assertIn(str(Path(__file__).parent / "cnn_dailymail_score.py") + ":score_turn_filtered", command)
            self.assertNotIn("--lora-device", command)
        with self.assertRaises(SystemExit):
            parse_args(["--size", "7b", "--data", "samples.jsonl", "--output-dir", "out", "--batch-size", "0"])

    def test_wrong_adapter_base_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(model_type="llama")), \
                patch("peft.PeftConfig.from_pretrained", return_value=SimpleNamespace(base_model_name_or_path="wrong")), \
                patch("run_llama2_cnn.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "Adapter base"):
                main(["--size", "7b", "--data", "unused.jsonl", "--output-dir", directory])
            run.assert_not_called()
            self.assertEqual(json.loads((Path(directory) / "status.json").read_text())["state"], "failed")

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
            write_split(root, "train", rows("train", 3))
            test = [dict(row, article=row["article"] + " word10" * i) for i, row in enumerate(rows("test", 3))]
            write_split(root, "test", test)
            path = root / "samples.jsonl"
            prepare(data_args(["--models", str(model_path), "--dataset-dir", str(root), "--num-shots", "1",
                "--sample-size", "3", "--prefix-max-tokens", "180", "--max-new-tokens", "2", "--output", str(path)]), path)
            output = root / "out"
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                    model_type="llama", hidden_size=4096, num_hidden_layers=32, num_attention_heads=32)), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"):
                main(["--size", "7b", "--model", str(model_path), "--lora", str(adapter_path), "--data", str(path),
                      "--device", "cpu", "--dtype", "float32", "--batch-size", "2", "--output-dir", str(output)])
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            self.assertEqual(result["scorer_metadata"]["primary_metric"], "rougeL")
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(set(result["scores"][mode]), {"rouge1", "rouge2", "rougeL"})
                self.assertIn(mode, result["unfiltered_scores"])
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
                self.assertTrue(all("summary_extraction" in row["predictions"][mode] for row in result["predictions"]))


if __name__ == "__main__":
    unittest.main()
