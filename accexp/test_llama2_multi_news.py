"""Multi-News text pairs, shared-model sampling and independent three-stage inference."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from data.multi_news import ALPACA_PREAMBLE, DATASET, TextPairSplit, document_prompt, parse_args as data_args, prepare
from multi_news_score import extract_summary, score, score_turn_filtered
from run_llama2_multi_news import evaluator_command, load_prepared, main, parse_args
from test_mistral_cnn import CharacterTokenizer


def write_split(root, split, count):
    folder = Path(root) / "data"
    folder.mkdir(exist_ok=True)
    stem = "val" if split == "validation" else split
    documents = [f"Complete {split}_{i} first news.NEWLINE_CHARSecond paragraph. ||||| Complete second news {i}."
                 for i in range(count)]
    summaries = [f"HELDOUT_{split}_{i} summary." for i in range(count)]
    (folder / f"{stem}.src.cleaned").write_text("\n".join(documents) + "\n")
    (folder / f"{stem}.tgt").write_text("\n".join(summaries) + "\n")
    return documents, summaries


class ExtraLongTokenizer(CharacterTokenizer):
    def encode(self, text, *args, **kwargs):
        tokens = super().encode(text, *args, **kwargs)
        return tokens * (3 if "test_0 first" in text else 1)


class MultiNewsTests(unittest.TestCase):
    def test_pair_reader_preserves_utf8_offsets_newlines_separators_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "validation", 2)
            source = root / "data/val.src.cleaned"
            source.write_text("caf\u00e9 NEWLINE_CHAR report ||||| second news\nthird ||||| fourth\n")
            split = TextPairSplit(root, "validation")
            self.assertEqual(len(split), 2)
            self.assertEqual(split.get(0)["article"], "caf\u00e9 \n report ||||| second news")
            self.assertEqual(split.get(1)["article"], "third ||||| fourth")
            self.assertEqual(split.get(0)["highlights"], "HELDOUT_validation_0 summary.")
            self.assertEqual(split.get(0)["num_documents"], 2)
            self.assertEqual([start for start, _ in split.batches(1)], [0, 1])
            self.assertEqual(split.manifest()[0]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            for index in (-1, 2):
                with self.assertRaises(IndexError):
                    split.get(index)
            with self.assertRaises(ValueError):
                list(split.batches(0))

    def test_mismatched_lines_fail_and_empty_rows_do_not_shift_answer_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "test", 3)
            source, target = root / "data/test.src.cleaned", root / "data/test.tgt"
            source.write_text("\nonly one document\nfirst ||||| second\n")
            split = TextPairSplit(root, "test")
            self.assertEqual(split.get(0)["article"], "")
            self.assertEqual(split.get(1)["article"], "")
            self.assertEqual(split.get(2)["highlights"], "HELDOUT_test_2 summary.")
            self.assertEqual(TextPairSplit(root, "test", 1).get(1)["article"], "only one document")
            target.write_text("answer1\nanswer2\n")
            with self.assertRaisesRegex(ValueError, "line counts"):
                TextPairSplit(root, "test")

    def test_joint_length_filtering_before_sampling_and_untruncated_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "train", 5)
            _, answers = write_split(root, "test", 6)
            answers[1] += " long reference" * 1000
            (root / "data/test.tgt").write_text("\n".join(answers) + "\n")
            models = [str(root / "small"), str(root / "large")]
            args = data_args(["--models", *models, "--dataset-dir", str(root), "--sample-size", "0",
                "--prompt-format", "legacy", "--max-new-tokens", "64", "--prefix-max-tokens", "700", "--output", str(root / "shared.jsonl")])
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=1100)), \
                    patch("transformers.AutoTokenizer.from_pretrained", side_effect=[CharacterTokenizer(), ExtraLongTokenizer()]):
                report = prepare(args, args.output)
            samples = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
            self.assertEqual(report["test_overlength"], 1)
            self.assertEqual(report["selected"], 5)
            self.assertEqual(samples[0]["references"], [answers[1]])
            self.assertEqual(report["dataset"], DATASET)
            self.assertNotIn("cnn_dailymail", json.dumps(report))
            self.assertEqual(len({s["prefix"] for s in samples}), 1)
            self.assertEqual(len(report["demonstrations"]), 1)
            for sample in samples:
                self.assertNotIn(sample["references"][0], sample["prefix"] + sample["query"])
                self.assertIn("HELDOUT_train_", sample["prefix"])
                self.assertEqual(sample["metadata"]["num_documents"], 2)
                self.assertFalse(sample["metadata"]["document_truncated"])
                self.assertEqual(sample["metadata"]["dataset"], DATASET)
                self.assertTrue(sample["query"].endswith("### Summary: "))
                self.assertIn("|||||", sample["query"])
                self.assertNotIn("NEWLINE_CHAR", sample["query"])

    def test_preparation_determinism_hash_budget_and_shared_model_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split(root, "train", 5)
            write_split(root, "test", 8)
            models = [str(root / "small"), str(root / "large")]
            def args_for(path):
                return data_args(["--models", *models, "--dataset-dir", str(root), "--sample-size", "4",
                    "--max-new-tokens", "64", "--prefix-max-tokens", "700", "--output", str(path)])
            path, second = root / "shared.jsonl", root / "repeated.jsonl"
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(max_position_embeddings=2000)), \
                    patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **kw: CharacterTokenizer()):
                report = prepare(args_for(path), path)
                self.assertEqual(report, prepare(args_for(second), second))
                self.assertEqual(path.read_bytes(), second.read_bytes())
                with self.assertRaises(FileExistsError):
                    prepare(args_for(path), path)
            for size, model in zip(("7b", "13b"), models):
                launch = parse_args(["--size", size, "--model", model, "--data", str(path), "--output-dir", str(root / size)])
                samples, _ = load_prepared(launch)
                self.assertEqual([s["id"] for s in samples], report["sample_ids"])
                launch.max_new_tokens = 512
                with self.assertRaisesRegex(ValueError, "Output budget"):
                    load_prepared(launch)
                launch.max_new_tokens = None
                launch.require_prompt_format = "legacy"
                with self.assertRaisesRegex(ValueError, "prompt protocol"):
                    load_prepared(launch)
                launch.require_prompt_format = None
            launch.max_new_tokens = None
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(launch)
            report["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["sample_ids"].reverse()
            Path(str(path) + ".meta.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "sample IDs"):
                load_prepared(launch)

    def test_reference_independent_summary_filter_is_uniform_and_retains_multiline_text(self):
        summary = "Alice won.\nBob came second."
        raw = summary + "\n\n### Documents:\nnext news ||||| more news\n### Summary: other"
        for mode in ("base", "lora_full", "lora_minikv"):
            self.assertEqual(score_turn_filtered([raw], [[summary]], [dict(mode=mode)]), dict(rouge1=1, rouge2=1, rougeL=1))
        self.assertLess(score([raw], [[summary]], [{}])["rougeL"], 1)
        self.assertEqual(score_turn_filtered(["wrong\n### Documents:\n" + summary], [[summary]], [{}])["rougeL"], 0)
        self.assertEqual(extract_summary(summary)["summary_text"], summary)
        self.assertEqual(extract_summary("inline Documents: text")["summary_text"], "inline Documents: text")
        self.assertEqual(extract_summary("### Documents: next")["summary_text"], "")
        with self.assertRaises(ValueError):
            extract_summary(None)

    def test_paths_formats_and_sequential_single_device_command(self):
        for size, batch, fmt in (("7b", 8, "safetensors"), ("13b", 4, "bin")):
            args = parse_args(["--size", size, "--data", "samples.jsonl", "--output-dir", "out"])
            self.assertEqual(args.batch_size, batch)
            self.assertEqual(args.weight_format, fmt)
            self.assertEqual(args.lora, f"/data/hub/multi_news-lora/llama2-{size}")
            args.max_new_tokens = 512
            command = evaluator_command(args, "out/result.json", 42)
            self.assertIn("--minikv-independent-prefill", command)
            self.assertIn(str(Path(__file__).parent / "multi_news_score.py") + ":score_turn_filtered", command)
            self.assertNotIn("--lora-device", command)
        with self.assertRaises(SystemExit):
            parse_args(["--size", "7b", "--data", "samples.jsonl", "--output-dir", "out", "--batch-size", "0"])

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
            write_split(root, "train", 3)
            write_split(root, "test", 3)
            path = root / "samples.jsonl"
            prepare(data_args(["--models", str(model_path), "--dataset-dir", str(root), "--sample-size", "3",
                "--prefix-max-tokens", "180", "--max-new-tokens", "2", "--output", str(path)]), path)
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
            self.assertEqual(result["dataset_preparation"]["dataset"], DATASET)
            self.assertEqual(result["scorer_metadata"]["primary_metric"], "rougeL")
            self.assertEqual(result["config"]["num_shots"], 1)
            self.assertEqual(result["config"]["prefix_max_tokens"], 180)
            self.assertEqual(result["config"]["prompt_protocol"], "alpaca")
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(set(result["scores"][mode]), {"rouge1", "rouge2", "rougeL"})
                self.assertIn(mode, result["unfiltered_scores"])
                self.assertEqual(result["generation_diagnostics"][mode]["samples"], 3)
                self.assertTrue(all("summary_extraction" in row["predictions"][mode] for row in result["predictions"]))

    def test_alpaca_outer_protocol_and_legacy_are_explicit(self):
        self.assertEqual(document_prompt("news", "legacy"), document_prompt("news"))
        prompt = document_prompt("first ||||| second", "alpaca")
        self.assertTrue(prompt.startswith(ALPACA_PREAMBLE + "\n\n### Instruction:\n"))
        self.assertTrue(prompt.endswith("\n\n### Response:\n"))
        self.assertNotIn("### Summary:", prompt)
        self.assertNotIn("### Documents:", prompt)
        self.assertIn("first ||||| second", prompt)
        with self.assertRaises(ValueError):
            document_prompt("news", "unknown")
        text = "Actual summary.\n\n" + ALPACA_PREAMBLE + "\n\n### Instruction:\nNext task"
        self.assertEqual(extract_summary(text)["summary_text"], "Actual summary.")


if __name__ == "__main__":
    unittest.main()
