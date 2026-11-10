"""Joint-model DROP preparation, uniform extraction and real tiny Llama integration."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from data.drop import build_samples, load_gold
from data.llama2_drop import prepare, parse_args as data_args
from drop_score import extract_answer, score, score_turn_filtered
from run_llama2_drop import evaluator_command, load_prepared, main, parse_args
from test_mistral_cnn import CharacterTokenizer
from test_mistral_drop import rows, raw_gold, write_split


class ExtraLongTokenizer(CharacterTokenizer):
    def encode(self, text, *args, **kwargs):
        result = super().encode(text, *args, **kwargs)
        return result * (3 if "validation passage 0" in text else 1)


class Llama2DROPTests(unittest.TestCase):
    def test_uniform_boundary_filter_preserves_multispan_and_wrong_answers(self):
        raw = '[\n  "Alice",\n  "Bob"\n]\n\n### Question: Next?\n### Answer: Someone'
        samples = [dict(metadata=dict(drop_gold_answers=[["Alice", "Bob"]]))]
        for mode in ("base", "lora_full", "lora_minikv"):
            self.assertEqual(score_turn_filtered([raw], [["Alice; Bob"]], [dict(samples[0], mode=mode)]),
                             dict(exact_match=1, drop_f1=1))
        self.assertEqual(score([raw], [["Alice; Bob"]], samples)["exact_match"], 0)
        numeric = [dict(metadata=dict(drop_gold_answers=[["3"]]))]
        self.assertEqual(score_turn_filtered(["7\n### Question: Next?\n### Answer: 3"], [["3"]], numeric),
                         dict(exact_match=0, drop_f1=0))
        self.assertEqual(extract_answer("first\nsecond")["answer_text"], "first\nsecond")
        self.assertEqual(extract_answer("inline Question: text")["answer_text"], "inline Question: text")
        self.assertEqual(extract_answer("### Passage: next")["answer_text"], "")
        with self.assertRaises(ValueError):
            extract_answer(None)

    def test_joint_length_check_before_sampling_and_single_model_unchanged(self):
        train, dev = rows("train", 6), rows("validation", 6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(json.dumps(raw_gold(dev)))
            gold = load_gold(path)
            kwargs = dict(context_limit=1800, max_new_tokens=64, num_shots=1,
                          prefix_max_tokens=700, sample_size=0, seed=42)
            ordinary = build_samples(train, dev, gold, CharacterTokenizer(), **kwargs)
            self.assertEqual(ordinary, build_samples(train, dev, gold, CharacterTokenizer(),
                                                    additional_tokenizers=(), **kwargs))
            joint = build_samples(train, dev, gold, CharacterTokenizer(),
                                  additional_tokenizers=[ExtraLongTokenizer()], **kwargs)
            self.assertEqual(len(ordinary[0]), 6)
            self.assertEqual(len(joint[0]), 5)
            self.assertEqual(joint[1]["validation_overlength"], 1)
            self.assertTrue(all(s["metadata"]["source_index"] != 0 for s in joint[0]))
            self.assertEqual(joint, build_samples(train, dev, gold, CharacterTokenizer(),
                             additional_tokenizers=[ExtraLongTokenizer()], **kwargs))
            self.assertEqual(joint[0][0]["prefix"], ordinary[0][0]["prefix"])

    def test_preparation_shared_ids_hash_and_output_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev = rows("train", 6), rows("validation", 6)
            write_split(root, "train", train)
            write_split(root, "validation", dev)
            gold = root / "gold.json"
            gold.write_text(json.dumps(raw_gold(dev)))
            data = root / "samples.jsonl"
            models = [str(root / name) for name in ("model7", "model13")]
            args = data_args(["--models", *models, "--dataset-dir", str(root), "--gold-file", str(gold),
                              "--num-shots", "1", "--sample-size", "3", "--prefix-max-tokens", "700",
                              "--output", str(data)])
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=2500)), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
                report = prepare(args, data)
                with self.assertRaises(FileExistsError):
                    prepare(args, data)
            self.assertEqual(set(report["context_limits"]), set(models))
            self.assertEqual(len(report["demonstrations"]), 1)
            for size, model in zip(("7b", "13b"), models):
                launch = parse_args(["--size", size, "--model", model, "--data", str(data), "--output-dir", str(root / size)])
                samples, _ = load_prepared(launch)
                self.assertEqual([s["id"] for s in samples], report["sample_ids"])
                self.assertEqual(len({s["prefix"] for s in samples}), 1)
                self.assertNotIn("validation passage", samples[0]["prefix"])
                launch.max_new_tokens = 64
                with self.assertRaisesRegex(ValueError, "Output budget"):
                    load_prepared(launch)
            launch.max_new_tokens = None
            data.write_text(data.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(launch)
            report["output_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
            report["sample_ids"].reverse()
            Path(str(data) + ".meta.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "sample IDs"):
                load_prepared(launch)

    def test_paths_formats_and_single_device_command(self):
        for size, batch, fmt in (("7b", 8, "safetensors"), ("13b", 4, "bin")):
            args = parse_args(["--size", size, "--data", "samples.jsonl", "--output-dir", "out"])
            self.assertEqual(args.batch_size, batch)
            self.assertEqual(args.weight_format, fmt)
            self.assertEqual(args.lora, f"/data/hub/drop-lora/llama2-{size}")
            args.max_new_tokens = 256
            command = evaluator_command(args, "out/result.json", 42)
            self.assertIn("--minikv-independent-prefill", command)
            self.assertIn(str(Path(__file__).parent / "drop_score.py") + ":score_turn_filtered", command)
            self.assertNotIn("--lora-device", command)
        with self.assertRaises(SystemExit):
            parse_args(["--size", "7b", "--data", "samples.jsonl", "--output-dir", "out", "--batch-size", "0"])

    def test_tiny_llama_sequential_stages_and_result_postprocessing(self):
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
            dev = rows("validation", 3)
            write_split(root, "train", rows("train", 3))
            write_split(root, "validation", dev)
            gold = root / "gold.json"
            gold.write_text(json.dumps(raw_gold(dev)))
            data = root / "samples.jsonl"
            prepare(data_args(["--models", str(model_path), "--dataset-dir", str(root), "--gold-file", str(gold),
                "--num-shots", "1", "--sample-size", "3", "--prefix-max-tokens", "180", "--max-new-tokens", "2",
                "--output", str(data)]), data)
            output = root / "out"
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                    model_type="llama", hidden_size=4096, num_hidden_layers=32, num_attention_heads=32)), \
                    patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"):
                main(["--size", "7b", "--model", str(model_path), "--lora", str(adapter_path), "--data", str(data),
                      "--device", "cpu", "--dtype", "float32", "--batch-size", "2", "--output-dir", str(output)])
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "complete")
            self.assertEqual(result["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                             dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(set(result["scores"][mode]), {"exact_match", "drop_f1"})
                self.assertIn(mode, result["unfiltered_scores"])
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
                self.assertTrue(all("answer_extraction" in row["predictions"][mode] for row in result["predictions"]))


if __name__ == "__main__":
    unittest.main()
