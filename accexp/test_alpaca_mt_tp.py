"""65B MT launcher, fixed-prefix protocol and optional tiny Q/V-only TP tests."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from safetensors.torch import load_file
import torch

from data.common import ALPACA_PROMPTS, alpaca_example
from data.prepare_alpaca_mt import prepare_for_tp
from eval_tp import parse_args as tp_args, preflight, prepare_runtime_data
from multilingual_score import score
from run_alpaca_cleaned_65b import launcher_environment
from run_alpaca_mt_65b import evaluation_command, parse_args, run, validate_prepared_data
from test_eval_tp import make_fixture


def fixture(root):
    torch.set_num_threads(1)
    make_fixture(root, target_modules=["q_proj", "v_proj"])
    torch.save(load_file(root / "adapter/adapter_model.safetensors"), root / "adapter/adapter_model.bin")
    raw = root / "raw"
    raw.mkdir()
    hashes, english = {}, None
    for language in ("en", "pt", "es", "ca"):
        rows = [dict(instruction=f"{language} t{20+i}", input="", output=f"t{40+i}") for i in range(12)]
        payload = json.dumps(rows).encode()
        (raw / f"alpaca_data_cleaned_{language}.json").write_bytes(payload)
        hashes[language] = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if language == "en":
            english = rows
    demo = alpaca_example(english[0], 0)
    (root / "prefix.jsonl").write_text(json.dumps(dict(
        prefix=demo["query"] + demo["references"][0] + "\n\n",
        metadata=dict(fewshot_train_ids=["0"]))) + "\n")
    (raw / "alpaca_mt.py").write_text("PROMPTS = " + repr(
        {language: ALPACA_PROMPTS for language in ("en", "pt", "es", "ca")}))
    argv = ["--model", str(root / "model"), "--lora", str(root / "adapter"),
            "--dataset-dir", str(raw), "--english-file", str(raw / "alpaca_data_cleaned_en.json"),
            "--raw-dir", str(raw), "--prefix-data", str(root / "prefix.jsonl"),
            "--samples-per-language", "2", "--max-new-tokens", "4", "--batch-size", "2",
            "--dtype", "float32", "--min-free-gpu-gib", "0", "--output-dir", str(root / "run")]
    return argv, hashes


def prepare_fixture(args, hashes, data):
    with patch.dict("data.prepare_alpaca_mt.RAW_GIT_SHA", hashes):
        return prepare_for_tp(args, data)


class MultilingualTPTests(unittest.TestCase):
    def test_local_preparation_binary_preflight_and_aligned_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, hashes = fixture(root)
            args = parse_args(argv)
            data = root / "prepared.jsonl"
            report = prepare_fixture(args, hashes, data)
            rows, checked = validate_prepared_data(args, data)
            self.assertEqual(report, checked)
            self.assertEqual(report["languages"], ["pt", "es", "ca"])
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({row["prefix"] for row in rows}), 1)
            self.assertTrue(all(0 not in report["selected_source_indices"] for _ in rows))
            self.assertEqual({row["metadata"]["language"] for row in rows}, {"pt", "es", "ca"})
            for language in args.languages:
                selected = [row for row in rows if row["metadata"]["language"] == language]
                self.assertEqual(len(selected), 2)
                self.assertEqual([row["metadata"]["source_index"] for row in selected],
                                 report["selected_source_indices"])
                self.assertTrue(all("Respond in" in row["query"] for row in selected))
            repeated = prepare_fixture(args, hashes, root / "again.jsonl")
            self.assertEqual(repeated["output_sha256"], report["output_sha256"])
            command = evaluation_command(args, data, root / "result.json", check_only=True)
            config, adapter, _, samples, prepared, formats, _, _ = preflight(
                tp_args(command[command.index("--model"):]))
            self.assertEqual(set(adapter.target_modules), {"q_proj", "v_proj"})
            self.assertEqual(formats["adapter"], "bin")
            self.assertEqual(len(samples), 6)
            self.assertTrue(all(len(row["full_ids"]) + args.max_new_tokens <= config.max_position_embeddings
                                for row in prepared))
            args.languages = ["pt", "es"]
            with self.assertRaisesRegex(ValueError, "settings differ"):
                validate_prepared_data(args, data)

    def test_command_and_runtime_preparation_keep_cleaned_separate(self):
        args = parse_args(["--samples-per-language", "100", "--batch-size", "6", "--output-dir", "unused"])
        command = evaluation_command(args, "data.jsonl", "result.json")
        worker = tp_args(command[command.index("--model"):])
        self.assertEqual(command[:6], [sys.executable, "-u", "-m", "torch.distributed.run",
                                      "--standalone", "--nproc_per_node=2"])
        self.assertTrue(worker.prepare_alpaca_mt)
        self.assertFalse(worker.prepare_alpaca_cleaned)
        self.assertEqual(worker.languages, ["pt", "es", "ca"])
        self.assertEqual(worker.samples_per_language, 100)
        self.assertEqual(worker.batch_size, 6)
        self.assertTrue(worker.scorer.endswith("multilingual_score.py:score"))
        with patch("eval_tp.prepare_alpaca_mt") as prepare_mt, \
                patch("eval_tp.prepare_alpaca_cleaned") as prepare_cleaned, \
                patch("eval_tp.dist.broadcast_object_list"):
            prepare_runtime_data(worker, 1)
            prepare_mt.assert_not_called()
            prepare_runtime_data(worker, 0)
            prepare_mt.assert_called_once_with(worker, worker.data)
            prepare_cleaned.assert_not_called()
        args.data = Path("existing.jsonl")
        self.assertNotIn("--prepare-alpaca-mt", evaluation_command(args, args.data, "result.json"))

    def test_prepare_only_does_not_touch_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, hashes = fixture(root)
            args = parse_args(argv + ["--prepare-only"])
            with patch.dict("data.prepare_alpaca_mt.RAW_GIT_SHA", hashes), \
                    patch("run_alpaca_mt_65b.check_gpu_memory") as gpu, \
                    patch("run_alpaca_mt_65b.subprocess.Popen") as start, \
                    patch("run_alpaca_mt_65b.subprocess.run") as check:
                run(args)
            gpu.assert_not_called()
            start.assert_not_called()
            self.assertIn("--check-only", check.call_args.args[0])
            self.assertEqual(json.loads((root / "run/status.json").read_text())["state"], "prepared")

    def test_busy_gpu_does_not_prepare_data_or_start_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--output-dir", directory])
            with patch("run_alpaca_mt_65b.check_gpu_memory", side_effect=RuntimeError("GPU busy")), \
                    patch("run_alpaca_mt_65b.prepare_for_tp") as prepare, \
                    patch("run_alpaca_mt_65b.subprocess.Popen") as start:
                with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                    run(args)
            prepare.assert_not_called()
            start.assert_not_called()
            self.assertEqual(json.loads((Path(directory) / "status.json").read_text())["state"], "failed")

    def test_filtering_applies_to_all_modes_and_preserves_raw_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, hashes = fixture(root)
            args = parse_args(argv)
            data = root / "prepared.jsonl"
            report = prepare_fixture(args, hashes, data)
            rows, _ = validate_prepared_data(args, data)
            args.data = data
            modes = ("base", "lora_full", "lora_minikv")

            def launch(command, **kwargs):
                texts = [row["references"][0] + "\n\n### Instruction:\nNext task." for row in rows]
                raw = score(texts, [row["references"] for row in rows], rows)
                result = dict(config=dict(model=args.model), experiment=dict(
                    samples=len(rows), data_sha256=report["output_sha256"]),
                    scores={mode: raw for mode in modes}, score_deltas={},
                    predictions=[dict(id=row["id"], references=row["references"], metadata=row["metadata"],
                        predictions={mode: dict(text=text, output_tokens=20, stopped_on_eos=False) for mode in modes})
                        for row, text in zip(rows, texts)])
                Path(command[command.index("--output")+1]).write_text(json.dumps(result))
                process = MagicMock(pid=123)
                process.wait.return_value = 0
                process.__enter__.return_value = process
                return process

            with patch("run_alpaca_mt_65b.check_gpu_memory", return_value=[]), \
                    patch("run_alpaca_mt_65b.subprocess.Popen", side_effect=launch):
                run(args)
            summary = json.loads((root / "run/summary.json").read_text())
            raw = json.loads((root / "run/result.json").read_text())
            for mode in modes:
                self.assertEqual(summary["scores"][mode], raw["scores"][mode])
                self.assertLess(summary["scores"][mode]["generic_token_f1"], 1)
                self.assertEqual(summary["filtered_scores"][mode]["generic_token_f1"], 1)
                self.assertEqual(summary["filter_counts"][mode]["truncated"], 6)
            self.assertEqual(json.loads((root / "run/status.json").read_text())["state"], "complete")

    @unittest.skipUnless(os.environ.get("RUN_TP_SMOKE") == "1", "Set RUN_TP_SMOKE=1 for real Q/V-only TP=2 test")
    def test_qv_binary_tp_all_modes_multilingual_scores_and_filtering(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, hashes = fixture(root)
            args = parse_args(argv)
            data = root / "prepared.jsonl"
            prepare_fixture(args, hashes, data)
            subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                str(Path(__file__).with_name("test_eval_tp.py")), "--distributed-smoke", str(root)],
                check=True, timeout=180, env=launcher_environment())
            command = [sys.executable, "-u", str(Path(__file__).with_name("run_alpaca_mt_65b.py")),
                       *argv, "--data", str(data)]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=240, env=launcher_environment())
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertLess(completed.stdout.index('"state": "model_loaded"'),
                            completed.stdout.index('"state": "data_loaded"'))
            result = json.loads((root / "run/result.json").read_text())
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(result["experiment"]["startup_order"], "model_then_data")
            self.assertEqual(result["experiment"]["tp_size"], 2)
            self.assertEqual(result["checkpoint_headers"]["adapter"]["tensors"], 8)
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(result["scores"][mode]["by_language"]["pt"]["samples"], 2)
                self.assertEqual(set(summary["filtered_scores"][mode]["by_language"]), {"pt", "es", "ca"})
            self.assertEqual(json.loads((root / "run/status.json").read_text())["state"], "complete")


if __name__ == "__main__":
    unittest.main()
