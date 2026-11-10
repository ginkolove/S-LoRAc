"""Qwen GQA/bias cache numerics, chat-template data and three-stage integration."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data.gsm8k import build_samples
from data.qwen_gsm8k import PROTOCOL, SYSTEM, demonstration_answer, prompt_builders
from eval_minikv import (capture_minikv, forward_tokens, fresh_cache, generate, generate_batch,
                        generate_batch_transformers, prepare_inputs, reconstruct_prefix, snapshot_cache)
from qwen_gsm8k_score import extract_prediction, score, score_unfiltered
from run_qwen_gsm8k import check_adapter, evaluator_command, main, parse_args
from test_eval_minikv import attach_lora, tiny_model
from test_mistral_gsm8k import rows, write_split


TEMPLATE = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + "
            "message['content'] + '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "<|im_end|>": 2, "<|im_start|>": 3,
                                   **{f"word{i}": i for i in range(4, 64)}}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"], chat_template=TEMPLATE)


class QwenGSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numeric_formats_and_first_declaration_binding(self):
        for raw in ("Compute 9.\n#### $1,200.00 dollars", r"Thus \boxed{1200}.",
                    r"The final answer is \boxed{1,200}.", "<answer>1200</answer>", "THE ANSWER IS: 1200.",
                    "The answer is:\n#### 1200", "The final answer is:\n\\[\n\\boxed{1200}\n\\]",
                    "The answer is **1200**.", r"The answer is \(1200\).",
                    "The answer is <answer>1200</answer>"):
            self.assertEqual(score([raw], [["1200"]], [{}]), dict(exact_match=1, flexible_exact_match=1))
        for raw in ("#### 7\n#### 3", r"\boxed{7} then \boxed{3}", "<answer>wrong</answer>\n#### 3",
                    r"\boxed{3/1}", "#### 3e2", "The answer is 2+1", r"\boxed{3", "<answer>3",
                    "The answer is wrong\n#### 3", "The answer is 7.\n#### 3", "The answer is <answer>3"):
            self.assertEqual(score([raw], [["3"]], [{}]), dict(exact_match=0, flexible_exact_match=0))
        self.assertEqual(score(["I computed 3."], [["3"]], [{}]), dict(exact_match=0, flexible_exact_match=1))
        self.assertEqual(score(["#### -0.00"], [["0"]], [{}])["exact_match"], 1)
        self.assertEqual(score(["#### 9007199254740993"], [["9007199254740992"]], [{}])["exact_match"], 0)
        with self.assertRaises(ValueError):
            score(["#### 3"], [["gold #### 3"]], [{}])

    def test_boundary_and_extraction_no_reference_or_mode_dependency(self):
        for boundary in ("\nQuestion: Next?\n", "<|im_start|>user\nNext?", "<|im_end|>"):
            raw = "I computed 2." + boundary + "#### 7"
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(score([raw], [["7"]], [dict(mode=mode)]), dict(exact_match=0, flexible_exact_match=0))
            self.assertTrue(extract_prediction(raw)["boundary_trimmed"])
            self.assertEqual(score_unfiltered([raw], [["7"]], [{}])["exact_match"], 1)

    def test_defaults_preserve_strict_checks_and_identical_generation_budgets(self):
        args = parse_args(["--output-dir", "unused"])
        self.assertEqual((args.sample_size, args.num_shots, args.max_new_tokens, args.seed), (1000, 3, 256, 42))
        self.assertEqual((args.batch_size, args.dtype), (20, "float32"))
        self.assertNotIn("--cache-check-atol", evaluator_command(args, "data", "result"))

    def test_native_chat_format_reproducible_sampling_no_answer_leakage(self):
        tokenizer = tiny_tokenizer()
        prefix_builder, query_builder = prompt_builders(tokenizer)
        train, test = rows("train", 8), rows("test", 12)
        for row in test:
            row["answer"] = "HELDOUT_SOLUTION_NEVER_IN_PROMPT\n#### 987654321"
        kwargs = dict(context_limit=1024, max_new_tokens=16, num_shots=3, prefix_max_tokens=900,
            sample_size=5, seed=42, prefix_builder=prefix_builder, prompt_builder=query_builder,
            answer_formatter=demonstration_answer, prompt_format=PROTOCOL)
        samples, report = build_samples(train, test, tokenizer, **kwargs)
        self.assertEqual((samples, report), build_samples(train, test, tokenizer, **kwargs))
        self.assertEqual(len(samples), 5)
        self.assertEqual(len({s["prefix"] for s in samples}), 1)
        self.assertEqual(samples[0]["prefix"].count("####"), 4)  # System format instruction plus three solutions.
        for sample in samples:
            messages = [dict(role="system", content=SYSTEM)]
            for demo in report["demonstrations"]:
                row = train[demo["source_index"]]
                messages += [dict(role="user", content=row["question"]),
                             dict(role="assistant", content=demonstration_answer(row["answer"]))]
            messages.append(dict(role="user", content=test[sample["metadata"]["source_index"]]["question"]))
            self.assertEqual(sample["prefix"] + sample["query"],
                             tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            self.assertNotIn("HELDOUT_SOLUTION", sample["prefix"] + sample["query"])
            self.assertNotIn("987654321", sample["prefix"] + sample["query"])
            self.assertEqual(sample["references"], ["987654321"])
        prepared = prepare_inputs(samples, tokenizer, 16, 1024, False)
        self.assertEqual(len({s["prefix_key"] for s in prepared}), 1)

    def exercise_batched(self, device="cpu", dtype=torch.float32):
        model = tiny_model("qwen2", attention="sdpa", device=device, dtype=dtype)
        prefix = list(range(4, 17))
        queries = [[20, 21], [22, 23, 24, 25, 26], [27]]
        base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        model = attach_lora(model, targets=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"])
        with capture_minikv(model) as mini:
            full = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        reconstructed = reconstruct_prefix(model, base, mini)
        self.assertEqual(reconstructed[0][0].shape, (1, 2, len(prefix), 8))
        for mode, pairs in (("base", base), ("lora_full", full), ("lora_minikv", reconstructed)):
            context = model.disable_adapter() if mode == "base" else torch.inference_mode()
            with context, self.subTest(mode=mode):
                expected = [generate(model, query, pairs, 6, set())[0] for query in queries]
                originals = tuple((k.clone(), v.clone()) for k, v in pairs)
                with patch.object(model, "generate", wraps=model.generate) as native:
                    actual = generate_batch(model, queries, pairs, 6, set())
                native.assert_called_once()
                kwargs = native.call_args.kwargs
                self.assertEqual(kwargs["input_ids"].shape, (3, 5))
                self.assertEqual(kwargs["attention_mask"].shape, (3, len(prefix) + 5))
                self.assertEqual(kwargs["cache_position"].tolist(), list(range(len(prefix), len(prefix) + 5)))
                self.assertFalse(kwargs["use_model_defaults"])
                self.assertEqual(kwargs["generation_config"].repetition_penalty, 1.0)
                self.assertEqual(actual, expected)
                self.assertEqual(actual, generate_batch(model, queries, pairs, 6, set()))
                for before, after in zip(originals, pairs):
                    for a, b in zip(before, after):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_native_matches_uncached_generate_and_stops_at_each_eos(self):
        model = tiny_model("qwen2", attention="sdpa")
        prefix, queries = [4, 5, 6], [[20, 21], [22, 23, 24, 25, 26], [27]]
        pairs = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        singles = [model.generate(torch.tensor([prefix + query]), max_new_tokens=6,
                                 do_sample=False, eos_token_id=[], pad_token_id=0,
                                 repetition_penalty=1.0, use_model_defaults=False)[0, len(prefix + query):].tolist()
                   for query in queries]
        self.assertEqual(generate_batch_transformers(model, queries, pairs, 6, set()), singles)
        eos = {singles[0][0], singles[1][1]}
        expected = [row[:next((i + 1 for i, token in enumerate(row) if token in eos), len(row))] for row in singles]
        self.assertEqual(generate_batch_transformers(model, queries, pairs, 6, eos), expected)
        model.generation_config.do_sample = True
        model.generation_config.repetition_penalty = 2.0
        self.assertEqual(generate_batch_transformers(model, queries, pairs, 6, set()), singles)

    def test_mixed_length_batch_matches_individual_all_modes(self):
        self.exercise_batched()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_bfloat16_gqa_mixed_batches(self):
        self.exercise_batched("cuda:0", torch.bfloat16)

    def test_launcher_and_complete_tiny_qwen_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "adapter"
            model = tiny_model("qwen2", attention="sdpa")
            model.save_pretrained(model_path)
            tokenizer = tiny_tokenizer()
            tokenizer.save_pretrained(model_path)
            adapted = attach_lora(model, targets=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"])
            adapted.save_pretrained(adapter_path, save_embedding_layers=False)
            tokenizer.save_pretrained(adapter_path)
            write_split(root, "train", rows("train", 8))
            write_split(root, "test", rows("test", 3))
            args_list = ["--model", str(model_path), "--lora", str(adapter_path), "--dataset-dir", str(root),
                "--num-shots", "1", "--prefix-max-tokens", "110", "--sample-size", "3",
                "--max-context-len", "128", "--max-new-tokens", "2", "--device", "cpu", "--dtype", "float32",
                "--batch-size", "2", "--output-dir", str(root / "out")]
            args = parse_args(args_list)
            compatibility = check_adapter(args)
            self.assertEqual(compatibility["checked_tensors"], 28)
            command = evaluator_command(args, root / "data.jsonl", root / "result.json")
            self.assertIn("--minikv-independent-prefill", command)
            self.assertIn("--no-add-special-tokens", command)
            self.assertNotIn("--lora-device", command)
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"):
                main(args_list)
            value = json.loads((root / "out/result.json").read_text())
            self.assertEqual(value["experiment"]["samples"], 3)
            self.assertEqual(value["experiment"]["unique_prefixes"], 1)
            self.assertEqual(value["experiment"]["generation_engine"], "transformers.generate")
            self.assertEqual(value["experiment"]["workers"]["single_worker"]["prefix_prepare_forwards"],
                dict(base=1, lora_full=1, minikv_base=1, minikv_adapted=1))
            self.assertEqual(value["config"]["prompt_protocol"], PROTOCOL)
            self.assertEqual(set(value["scores"]), {"base", "lora_full", "lora_minikv"})
            self.assertEqual(json.loads((root / "out/status.json").read_text())["state"], "complete")
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(value["generation_diagnostics"][mode]["samples"], 3)
            altered = adapted.base_model.model.model.layers[0].self_attn.k_proj.lora_A["default"]
            self.assertEqual(altered.weight.shape[0], 4)
            bad = json.loads((adapter_path / "adapter_config.json").read_text())
            bad["r"] = 8
            (adapter_path / "adapter_config.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                check_adapter(args)


if __name__ == "__main__":
    unittest.main()
