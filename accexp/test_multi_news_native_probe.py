"""Validation-only prompt calibration without custom KV or decode paths."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multi_news_native_probe import CONDITIONS, load_data, parse_args, prepare, run
from test_llama2_multi_news import write_split
from test_mistral_cnn import CharacterTokenizer


class ProbeTests(unittest.TestCase):
    def prepare_fixture(self, root, count=3):
        write_split(root, "train", 4)
        write_split(root, "validation", count)
        path = root / "probe.jsonl"
        args = parse_args(["--prepare-only", "--models", str(root / "model"), "--dataset-dir", str(root),
            "--data", str(path), "--sample-size", "2", "--prefix-max-tokens", "700", "--max-new-tokens", "64"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=2500)), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
            samples, report = prepare(args)
        return args, samples, report

    def test_locked_validation_ids_shared_demos_and_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, samples, report = self.prepare_fixture(root)
            self.assertEqual(report["split"], "validation")
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(report["demonstrations"]), 1)
            self.assertFalse(report.get("score_based_protocol_selection", False))
            for sample in samples:
                self.assertEqual(set(sample["prompts"]), set(CONDITIONS))
                for condition in CONDITIONS:
                    text = sample["prompts"][condition]
                    self.assertNotIn(sample["references"][0], text)
                    self.assertIn("Complete validation_", text)
                    self.assertEqual("HELDOUT_train_" in text, condition.endswith("fixed_prefix"))
                    self.assertEqual(report["condition_num_shots"][condition], int(condition.endswith("fixed_prefix")))
            run_args = parse_args(["--size", "7b", "--model", str(root / "model"), "--data", args.data,
                                   "--output-dir", str(root / "out")])
            self.assertEqual(load_data(run_args), (samples, report))
            self.assertEqual(run_args.max_new_tokens, 64)
            self.assertEqual(run_args.sample_size, 2)
            run_args.max_new_tokens = 128
            with self.assertRaisesRegex(ValueError, "Output budget"):
                load_data(run_args)

    def test_refuse_test_data_checksum_corruption_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, report = self.prepare_fixture(root)
            launch = parse_args(["--size", "7b", "--model", str(root / "model"), "--data", args.data,
                                 "--output-dir", str(root / "out")])
            meta = Path(args.data + ".meta.json")
            report["split"] = "test"
            meta.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "validation"):
                load_data(launch)
            report["split"] = "validation"
            meta.write_text(json.dumps(report))
            with Path(args.data).open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(launch)
            with self.assertRaises(FileExistsError):
                prepare(args)

    def test_native_generation_all_conditions_with_tiny_real_llama(self):
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
                eos_token="[EOS]", pad_token="[UNK]").save_pretrained(model_path)
            adapted = attach_lora(model, targets=["q_proj", "v_proj"])
            adapted.peft_config["default"].base_model_name_or_path = "meta-llama/Llama-2-7b-hf"
            adapted.save_pretrained(adapter_path, save_embedding_layers=False)
            write_split(root, "train", 3)
            write_split(root, "validation", 3)
            path = root / "probe.jsonl"
            prepare(parse_args(["--prepare-only", "--models", str(model_path), "--dataset-dir", str(root),
                "--data", str(path), "--sample-size", "2", "--prefix-max-tokens", "180", "--max-new-tokens", "2"]))
            args = parse_args(["--size", "7b", "--model", str(model_path), "--lora", str(adapter_path),
                "--data", str(path), "--output-dir", str(root / "out"), "--device", "cpu", "--dtype", "float32"])
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"), \
                    patch("eval_minikv.generate_batch", side_effect=AssertionError("custom decoder must not be used")):
                result = run(args)
            self.assertFalse(result["experiment"]["minikv"])
            self.assertFalse(result["experiment"]["shared_external_prefix_cache"])
            self.assertFalse(result["experiment"]["score_based_protocol_selection"])
            self.assertEqual(result["config"]["max_new_tokens"], 2)
            self.assertEqual(result["config"]["sample_size"], 2)
            self.assertEqual(set(result["scores"]), {"base", "lora_full"})
            for mode in ("base", "lora_full"):
                self.assertEqual(set(result["scores"][mode]), set(CONDITIONS))
                for record in result["predictions"]:
                    self.assertEqual(set(record["predictions"][mode]), set(CONDITIONS))
            self.assertFalse(result["generation_config"]["do_sample"])
            self.assertEqual(result["generation_config"]["repetition_penalty"], 1.0)
            journal = (root / "out/predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(journal), 16)
            status = json.loads((root / "out/status.json").read_text())
            self.assertEqual((status["state"], status["completed"], status["total"]), ("complete", 16, 16))
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
