"""Fixed English prefix, translation alignment, length filtering and scoring."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from data.alpaca_mt import language_templates
from data.common import ALPACA_PROMPTS, alpaca_example
from data.prepare_alpaca_mt import build_samples, read_prefix, verify_source
from eval_minikv import PartialResults, load_scorer
from multilingual_score import normalize_answer, score
from run_alpaca_mt import validate_data


class WordTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[1] + list(range(len(text.split()))) for text in texts]}


def fixtures(count=20):
    english = [dict(instruction=f"English question {i}", input="", output=f"English answer {i}")
               for i in range(count)]
    rows = {language: [dict(instruction=f"{language} question {i}", input="",
                           output=f"{language} answer {i}") for i in range(count)]
            for language in ("es", "pt")}
    demo = alpaca_example(english[0], 0)
    prefix = demo["query"] + demo["references"][0] + "\n\n"
    return english, rows, (prefix, [0], {0})


def prepare(english, rows, prefix, **kwargs):
    return build_samples(english, rows, prefix,
                         {language: ALPACA_PROMPTS for language in rows},
                         [dict(tokenizer=WordTokenizer(), models=["tiny"], limit=200)],
                         sample_size=5, max_new_tokens=10, **kwargs)


class MultilingualTests(unittest.TestCase):
    def test_seeded_aligned_sampling_with_one_english_prefix(self):
        english, rows, prefix = fixtures()
        left, report = prepare(english, rows, prefix, seed=42)
        right, repeated = prepare(english, rows, prefix, seed=42)
        other, _ = prepare(english, rows, prefix, seed=43)
        self.assertEqual(left, right)
        self.assertEqual(report, repeated)
        self.assertNotEqual([row["id"] for row in left], [row["id"] for row in other])
        self.assertEqual(len(left), 10)
        self.assertEqual({row["prefix"] for row in left}, {prefix[0]})
        self.assertNotIn(0, report["selected_source_indices"])
        self.assertEqual({row["metadata"]["source_index"] for row in left[:5]},
                         {row["metadata"]["source_index"] for row in left[5:]})
        for row in left:
            language = row["metadata"]["language"]
            index = row["metadata"]["source_index"]
            self.assertEqual(row["references"], [rows[language][index]["output"]])
            self.assertNotIn(row["references"][0], row["prefix"] + row["query"])
            self.assertIn("Respond in", row["query"])
            self.assertLessEqual(row["metadata"]["input_tokens_by_model"]["tiny"] + 10, 200)

    def test_filter_before_sampling_and_exclude_translated_demo_duplicates(self):
        english, rows, prefix = fixtures()
        rows["es"][1]["instruction"] += " long" * 300
        rows["pt"][2]["instruction"] = rows["pt"][0]["instruction"]
        _, report = prepare(english, rows, prefix)
        self.assertEqual(report["common_eligible_tasks"], 17)
        self.assertEqual(report["per_language"]["es"]["overlength"], 1)
        self.assertTrue({0, 1, 2}.isdisjoint(report["selected_source_indices"]))

    def test_language_instruction_does_not_make_empty_questions_valid(self):
        english, rows, prefix = fixtures()
        rows["es"][1]["instruction"] = " "
        _, report = prepare(english, rows, prefix)
        self.assertEqual(report["per_language"]["es"]["invalid"], 1)
        self.assertEqual(report["common_eligible_tasks"], 18)

    def test_overlength_on_any_model_excludes_task_in_all_languages(self):
        english, rows, prefix = fixtures()

        class LongerTokenizer(WordTokenizer):
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[1] * (300 if "question 2\n" in text else 60) for text in texts]}

        samples, report = build_samples(
            english, rows, prefix, {language: ALPACA_PROMPTS for language in rows},
            [dict(tokenizer=WordTokenizer(), models=["a"], limit=200),
             dict(tokenizer=LongerTokenizer(), models=["b"], limit=200)], sample_size=5, max_new_tokens=10)
        self.assertNotIn(2, report["selected_source_indices"])
        self.assertTrue(all(set(row["metadata"]["input_tokens_by_model"]) == {"a", "b"} for row in samples))

    def test_invalid_alignment_and_insufficient_population_fail(self):
        english, rows, prefix = fixtures()
        rows["es"].pop()
        with self.assertRaisesRegex(ValueError, "row count"):
            prepare(english, rows, prefix)
        english, rows, prefix = fixtures(4)
        with self.assertRaisesRegex(ValueError, "Only 3"):
            prepare(english, rows, prefix)

    def test_prefix_provenance_and_pinned_source_checks(self):
        english, _, prefix = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.jsonl"
            record = dict(prefix=prefix[0], metadata=dict(fewshot_train_ids=["0"]))
            path.write_text(json.dumps(record) + "\n")
            self.assertEqual(read_prefix(path, english), prefix)
            record["prefix"] += "wrong"
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_prefix(path, english)
            with self.assertRaisesRegex(ValueError, "source differs"):
                verify_source(path, "en")

    def test_local_template_loader_does_not_execute_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loader.py"
            path.write_text("raise RuntimeError('do not execute')\nPROMPTS = " +
                            repr({"es": ALPACA_PROMPTS}))
            self.assertEqual(language_templates("es", True, path), ALPACA_PROMPTS)

    def test_unicode_scoring_keeps_articles_and_accents(self):
        self.assertEqual(normalize_answer("  A\u00a0CASA\u3002  "), "a casa")
        self.assertEqual(normalize_answer("caf\u0065\u0301"), "caf\u00e9")
        self.assertNotEqual(normalize_answer("si"), normalize_answer("s\u00ed"))
        result = score(["a casa", "a casa", "no"], [["casa"], ["a casa"], ["yes"]],
                       [dict(metadata=dict(language=language)) for language in ("es", "pt", "pt")])
        self.assertAlmostEqual(result["generic_token_f1"], 5 / 9)
        self.assertAlmostEqual(result["macro_generic_token_f1"], 7 / 12)
        self.assertEqual(result["by_language"]["pt"]["samples"], 2)
        with self.assertRaises(ValueError):
            score(["x"], [["x"]], [{}])

    def test_custom_scorer_persists_partial_per_language_results(self):
        english, rows, prefix = fixtures()
        samples, _ = prepare(english, rows, prefix)
        scorer = load_scorer(str(Path(__file__).with_name("multilingual_score.py")) + ":score")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            with PartialResults(path, samples, scorer) as writer:
                records = [(i, dict(id=sample["id"], predictions={"base": dict(text=sample["references"][0])}))
                           for i, sample in enumerate(samples)]
                writer.commit("base", records)
                summary = json.loads(writer.summary_path.read_text())
                self.assertEqual(summary["scores"]["base"]["generic_token_f1"], 1)
                self.assertEqual(set(summary["scores"]["base"]["by_language"]), {"pt", "es"})

    def test_suite_validates_data_hash_counts_and_alignment(self):
        english, rows, prefix = fixtures()
        samples, report = prepare(english, rows, prefix)
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in samples))
            report["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            Path(str(path) + ".meta.json").write_text(json.dumps(report))
            self.assertEqual(validate_data(path)[0], samples)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                validate_data(path)

    def test_three_mode_evaluation_with_multilingual_scorer(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from eval_minikv import parse_args, run
        from test_eval_minikv import attach_lora, tiny_model

        torch.set_num_threads(1)
        english, rows, prefix = fixtures()
        samples, _ = prepare(english, rows, prefix)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "adapter"
            model = tiny_model()
            model.save_pretrained(model_path, safe_serialization=False)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                bos_token="[BOS]", eos_token="[EOS]")
            tokenizer.save_pretrained(model_path)
            attach_lora(model).save_pretrained(adapter_path)
            data, output = root / "data.jsonl", root / "scores.json"
            data.write_text("".join(json.dumps(row) + "\n" for row in samples))
            args = parse_args(["--model", str(model_path), "--lora", str(adapter_path),
                               "--data", str(data), "--output", str(output), "--device", "cpu",
                               "--dtype", "float32", "--max-new-tokens", "3", "--batch-size", "2",
                               "--weight-format", "bin", "--local-files-only", "--scorer",
                               str(Path(__file__).with_name("multilingual_score.py")) + ":score"])
            result = run(args)
            self.assertEqual(result["experiment"]["unique_prefixes"], 1)
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(result["scores"][mode]["by_language"]["es"]["samples"], 5)
                self.assertEqual(result["scores"][mode]["by_language"]["pt"]["samples"], 5)
            partial = json.loads(output.with_suffix(".partial.json").read_text())
            self.assertEqual(partial["state"], "complete")
            self.assertEqual(partial["scores"], result["scores"])

    def test_launcher_automatically_filters_all_models_and_preserves_raw_scores(self):
        import hashlib
        from run_alpaca_mt import main

        english, rows, prefix = fixtures()
        samples, report = prepare(english, rows, prefix)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, output, template = root / "samples.jsonl", root / "run", root / "loader.py"
            data.write_text("".join(json.dumps(row) + "\n" for row in samples))
            template.write_text("PROMPTS = " + repr({"en": ALPACA_PROMPTS}))
            report.update(output_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                          template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                          models={f"huggyllama/llama-{size}": dict(context_limit=2048) for size in ("7b", "13b")})
            Path(str(data) + ".meta.json").write_text(json.dumps(report))
            modes = ("base", "lora_full", "lora_minikv")

            def launch(command, **kwargs):
                model = command[command.index("--model") + 1]
                size = model.rsplit("-", 1)[-1]
                self.assertEqual(command[command.index("--lora") + 1], f"/data/hub/alpaca_mt-lora/{size}")
                self.assertTrue(command[command.index("--scorer") + 1].endswith("multilingual_score.py:score"))
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0,1")
                texts = [row["references"][0] + "\n\n### Instruction:\nAn unrelated next task." for row in samples]
                expected = score(texts,
                                 [row["references"] for row in samples], samples)
                result = dict(config=dict(model=model), experiment=dict(samples=len(samples)),
                              predictions=[dict(id=row["id"], references=row["references"], metadata=row["metadata"],
                                                predictions={mode: dict(text=text, output_tokens=20, stopped_on_eos=False)
                                                             for mode in modes})
                                           for row, text in zip(samples, texts)],
                              scores={mode: expected for mode in modes}, score_deltas={})
                Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
                return Mock(pid=123, wait=Mock(return_value=0))

            with patch("run_alpaca_mt.free_gpu_memory", return_value=[80 * 1024] * 2), \
                    patch("run_alpaca_mt.subprocess.Popen", side_effect=launch):
                main(["--data", str(data), "--output-dir", str(output), "--sizes", "7b", "13b",
                      "--template-loader", str(template)])
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["filtering_policy"], "alpaca_turn_boundary_v1")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(set(summary), {"7b", "13b"})
            for size in summary:
                model = summary[size]
                raw = json.loads((output / f"alpaca_mt_{size}_non_en_seed42.json").read_text())
                self.assertEqual(model["scores"], raw["scores"])
                self.assertEqual(model["minikv_minus_lora_by_language"]["es"]["generic_token_f1"], 0)
                self.assertEqual(model["filtered_minikv_minus_lora_by_language"]["es"]["generic_token_f1"], 0)
                self.assertEqual(model["filtered_score_deltas"]["minikv_minus_lora"]["exact_match"], 0)
                for mode in modes:
                    self.assertLess(model["scores"][mode]["generic_token_f1"], 1)
                    self.assertEqual(model["filtered_scores"][mode]["generic_token_f1"], 1)
                    self.assertEqual(model["filter_counts"][mode]["truncated"], len(samples))
                filtered = [json.loads(line) for line in Path(model["filtered_predictions"]).read_text().splitlines()]
                self.assertEqual(len(filtered), len(samples))
                self.assertIn("### Instruction:", filtered[0]["predictions"]["base"]["raw_text"])
                self.assertEqual(filtered[0]["predictions"]["base"]["text"], samples[0]["references"][0])

            failed_output = root / "failed_filter"
            with patch("run_alpaca_mt.free_gpu_memory", return_value=[80 * 1024] * 2), \
                    patch("run_alpaca_mt.subprocess.Popen", side_effect=launch), \
                    patch("run_alpaca_mt.rescore_turn_boundaries", side_effect=RuntimeError("filter failed")):
                with self.assertRaisesRegex(RuntimeError, "filter failed"):
                    main(["--data", str(data), "--output-dir", str(failed_output), "--sizes", "7b",
                          "--template-loader", str(template)])
            self.assertEqual(json.loads((failed_output / "status.json").read_text())["state"], "failed")
            self.assertTrue((failed_output / "alpaca_mt_7b_non_en_seed42.json").is_file())


if __name__ == "__main__":
    unittest.main()
