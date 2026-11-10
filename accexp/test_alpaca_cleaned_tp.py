"""Alpaca-Cleaned sampling, binary adapter preflight and TP launcher checks."""

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from peft import PeftConfig
from safetensors.torch import load_file
import torch
from transformers import AutoConfig

from data.prepare_alpaca_cleaned import prepare, select_samples
from eval_minikv import prepare_inputs
from eval_tp import (load_evaluation_data, preflight, prepare_runtime_data,
                     parse_args as tp_args, run as run_tp, validate_checkpoint_shapes)
from run_alpaca_cleaned_65b import (check_gpu_memory, evaluation_command, launcher_environment,
                                   parse_args, run, validate_gpu_memory)
from test_eval_tp import make_fixture


def fixture(root):
    torch.set_num_threads(1)
    make_fixture(root)
    state = load_file(root / "adapter/adapter_model.safetensors")
    torch.save(state, root / "adapter/adapter_model.bin")
    (root / "adapter/adapter_model.safetensors").unlink()
    dataset = root / "raw"
    dataset.mkdir()
    rows = [dict(instruction=f"t{20+i}", input="", output=f"t{40+i}") for i in range(12)]
    rows += [dict(instruction="t20 " * 300, input="", output="t6"), rows[0]]
    (dataset / "alpaca_data_cleaned.json").write_text(json.dumps(rows))
    return ["--model", str(root / "model"), "--lora", str(root / "adapter"),
            "--dataset-dir", str(dataset), "--sample-size", "3", "--num-shots", "1",
            "--prefix-max-tokens", "64", "--max-new-tokens", "4", "--dtype", "float32",
            "--min-free-gpu-gib", "0",
            "--output-dir", str(root / "run")]


class AlpacaTPTests(unittest.TestCase):
    def test_sampling_matches_previous_cleaned_protocol(self):
        samples = [dict(id=str(i)) for i in range(50)]
        selected, indices = select_samples(samples, 10, 42)
        expected = sorted(random.Random(42).sample(range(50), 10))
        self.assertEqual(indices, expected)
        self.assertEqual(selected, [samples[i] for i in expected])
        self.assertEqual(select_samples(samples, 0, 42)[0], samples)
        for count in (-1, 51):
            with self.assertRaises(ValueError):
                select_samples(samples, count, 42)

    def test_prepare_and_bin_preflight_without_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(fixture(root))
            destination = root / "selected.jsonl"
            report = prepare(args, destination)
            _, _, tokenizer, samples, prepared, formats, files, _ = preflight(tp_args(
                ["--model", args.model, "--lora", args.lora, "--data", str(destination),
                 "--dataset-name", "alpaca_cleaned", "--max-new-tokens", "4",
                 "--output", str(root / "result.json"), "--check-only"]))
            self.assertEqual(formats["adapter"], "bin")
            self.assertEqual(len(samples), 3)
            self.assertEqual(len({row["prefix"] for row in samples}), 1)
            self.assertEqual(report["test_overlength"], 1)
            self.assertEqual(report["split_info"]["train_stats"]["duplicate_queries"], 1)
            self.assertTrue(set(report["fewshot_train_ids"]).isdisjoint(
                row["metadata"]["source_id"] for row in samples))
            self.assertTrue(all(len(row["full_ids"]) + 4 <= 128 for row in prepared))
            self.assertEqual(prepare_inputs(samples, tokenizer, 4, 128), prepared)
            duplicate = root / "selected_again.jsonl"
            self.assertEqual(prepare(args, duplicate)["output_sha256"], report["output_sha256"])
            with self.assertRaises(FileExistsError):
                prepare(args, destination)
            state = torch.load(root / "adapter/adapter_model.bin", weights_only=True)
            key = next(iter(state))
            state[key] = torch.zeros(1, 1)
            torch.save(state, root / "adapter/adapter_model.bin")
            with self.assertRaisesRegex(ValueError, "adapter checkpoint.*wrong_shapes"):
                validate_checkpoint_shapes(AutoConfig.from_pretrained(args.model),
                    PeftConfig.from_pretrained(args.lora), files, formats)

    def test_prepare_only_never_starts_distributed_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(fixture(root) + ["--prepare-only"])
            with patch("run_alpaca_cleaned_65b.subprocess.run") as check, \
                    patch("run_alpaca_cleaned_65b.subprocess.Popen") as start, \
                    patch("run_alpaca_cleaned_65b.check_gpu_memory") as gpu:
                run(args)
            check.assert_called_once()
            self.assertIn("--check-only", check.call_args.args[0])
            start.assert_not_called()
            gpu.assert_not_called()
            status = json.loads((root / "run/status.json").read_text())
            self.assertEqual(status["state"], "prepared")
            self.assertEqual(status["samples"], 3)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_command_uses_tp2_and_signal_safe_environment(self):
        args = parse_args(["--output-dir", "unused"])
        command = evaluation_command(args, "data.jsonl", "result.json")
        self.assertEqual(command[:6], [sys.executable, "-u", "-m", "torch.distributed.run",
                                      "--standalone", "--nproc_per_node=2"])
        self.assertEqual(command[command.index("--dataset-name")+1], "alpaca_cleaned")
        self.assertEqual(command[command.index("--batch-size")+1], "3")
        self.assertIn("--prepare-alpaca-cleaned", command)
        self.assertEqual(command[command.index("--sample-size")+1], "1000")
        check = evaluation_command(args, "data", "result", True)
        self.assertNotIn("torch.distributed.run", check)
        self.assertNotIn("--prepare-alpaca-cleaned", check)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="1,0", TORCHELASTIC_SIGNALS_TO_HANDLE="SIGHUP"):
            environment = launcher_environment()
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1,0")
        self.assertNotIn("SIGHUP", environment["TORCHELASTIC_SIGNALS_TO_HANDLE"].split(","))
        self.assertIn("SIGTERM", environment["TORCHELASTIC_SIGNALS_TO_HANDLE"].split(","))
        self.assertEqual(command[command.index("--monitor-interval")+1], "1")

    def test_gpu_memory_guard_rejects_one_busy_gpu(self):
        devices = [dict(device=0, name="A100", free_bytes=11*2**30),
                   dict(device=1, name="A100", free_bytes=79*2**30)]
        with self.assertRaisesRegex(RuntimeError, "cuda:0.*11.00 GiB free"):
            validate_gpu_memory(devices, 76)
        devices[0]["free_bytes"] = 79*2**30
        self.assertEqual(validate_gpu_memory(devices, 76), devices)
        with self.assertRaisesRegex(RuntimeError, "exactly two"):
            validate_gpu_memory(devices[:1], 0)
        with patch("run_alpaca_cleaned_65b.subprocess.run") as probe:
            probe.return_value.returncode = 0
            probe.return_value.stdout = json.dumps(devices)
            self.assertEqual(check_gpu_memory(76), devices)
            self.assertIn("torch.cuda.mem_get_info", probe.call_args.args[0][-1])

    def test_gpu_guard_failure_does_not_start_model_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(["--output-dir", str(root)])
            sampling = dict(selected=3, prefix_tokens=26, output_sha256="test")
            with patch("run_alpaca_cleaned_65b.prepare", return_value=sampling) as prepare_data, \
                    patch("run_alpaca_cleaned_65b.subprocess.run") as check, \
                    patch("run_alpaca_cleaned_65b.check_gpu_memory", side_effect=RuntimeError("GPU busy")), \
                    patch("run_alpaca_cleaned_65b.subprocess.Popen") as start:
                with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                    run(args)
            start.assert_not_called()
            prepare_data.assert_not_called()
            check.assert_not_called()
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["error"], "GPU busy")

    def test_worker_loads_model_before_preparing_and_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher_args = parse_args(fixture(root))
            destination = root / "selected.jsonl"
            command = evaluation_command(launcher_args, destination, root / "result.json")
            args = tp_args(command[command.index("--model"):])
            events = []
            model = MagicMock()
            model.eval.return_value = model
            model.generation_config.eos_token_id = 2

            def load_model(*unused, **kwargs):
                self.assertFalse(destination.exists())
                events.append("model")
                return model

            def prepare_data(arguments, path):
                events.append("prepare")
                return prepare(arguments, path)

            def read_data(arguments, config):
                events.append("read")
                return load_evaluation_data(arguments, config)

            def gather(values, value):
                values[:] = [value, value]

            with patch.dict(os.environ, WORLD_SIZE="2", RANK="0", LOCAL_RANK="0"), \
                    patch("eval_tp.torch.cuda.set_device"), \
                    patch("eval_tp.dist.init_process_group"), \
                    patch("eval_tp.init_device_mesh"), \
                    patch("eval_tp.dist.barrier", side_effect=lambda: events.append("barrier")), \
                    patch("eval_tp.AutoModelForCausalLM.from_pretrained", side_effect=load_model), \
                    patch("eval_tp.validate_tp_model", return_value=[]), \
                    patch("eval_tp.prepare_alpaca_cleaned", side_effect=prepare_data), \
                    patch("eval_tp.dist.broadcast_object_list"), \
                    patch("eval_tp.load_evaluation_data", side_effect=read_data), \
                    patch("eval_tp.dist.all_gather_object", side_effect=gather), \
                    patch("eval_tp.stage_prefix", side_effect=RuntimeError("Stop after data loading")), \
                    patch("eval_tp.os._exit", side_effect=SystemExit(1)), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_tp(args)
            self.assertEqual(events, ["model", "barrier", "prepare", "read"])
            partial = json.loads((root / "result.partial.json").read_text())
            self.assertEqual(partial["total_samples"], 3)
            self.assertEqual(partial["state"], "failed")

    def test_runtime_preparation_is_rank_zero_only_and_propagates_errors(self):
        args = tp_args(["--output", "unused", "--dataset-name", "alpaca_cleaned",
                        "--prepare-alpaca-cleaned"])
        with patch("eval_tp.prepare_alpaca_cleaned") as prepare_data, \
                patch("eval_tp.dist.broadcast_object_list") as broadcast:
            prepare_runtime_data(args, 1)
        prepare_data.assert_not_called()
        broadcast.assert_called_once()
        with patch("eval_tp.prepare_alpaca_cleaned", side_effect=ValueError("Bad data")), \
                patch("eval_tp.dist.broadcast_object_list") as broadcast:
            with self.assertRaisesRegex(RuntimeError, "ValueError: Bad data"):
                prepare_runtime_data(args, 0)
        self.assertEqual(broadcast.call_args.args[0], ["ValueError: Bad data"])

    @unittest.skipUnless(os.environ.get("RUN_TP_SMOKE") == "1", "Set RUN_TP_SMOKE=1 for tiny real TP=2 test")
    def test_tiny_two_gpu_binary_adapter_all_modes(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = fixture(root)
            command = [sys.executable, "-u", str(Path(__file__).with_name("run_alpaca_cleaned_65b.py"))] + arguments
            completed = subprocess.run(command, capture_output=True, text=True, timeout=240,
                                       env=launcher_environment())
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            phases = [completed.stdout.index(f'"state": "{phase}"') for phase in
                      ("loading_model", "model_loaded", "preparing_data", "data_loaded")]
            self.assertEqual(phases, sorted(phases))
            result = json.loads((root / "run/result.json").read_text())
            status = json.loads((root / "run/status.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(result["experiment"]["dataset"], "alpaca_cleaned")
            self.assertEqual(result["experiment"]["tp_size"], 2)
            self.assertEqual(result["experiment"]["samples"], 3)
            self.assertEqual(result["experiment"]["startup_order"], "model_then_data")
            standalone = prepare(parse_args(arguments), root / "standalone.jsonl")
            self.assertEqual(result["experiment"]["data_sha256"], standalone["output_sha256"])
            self.assertTrue(result["checkpoint_files"]["adapter"][0]["path"].endswith(".bin"))
            modes = {"base", "lora_full", "lora_minikv"}
            self.assertEqual(set(result["scores"]), modes)
            self.assertTrue(all(set(row["predictions"]) == modes for row in result["predictions"]))


    @unittest.skipUnless(os.environ.get("RUN_TP_SMOKE") == "1", "Set RUN_TP_SMOKE=1 for injected TP failure test")
    def test_failed_rank_exits_without_waiting_for_nccl_watchdog(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                       "--monitor-interval", "1", str(Path(__file__).resolve()), "--injected-load-failure", str(root)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                                    env=launcher_environment())
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Injected rank-0 OOM", result.stderr)
            self.assertNotIn("Watchdog caught collective operation timeout", result.stderr)
            partial = json.loads((root / "failure.partial.json").read_text())
            self.assertEqual(partial["state"], "failed")
            self.assertIn("Injected rank-0 OOM", partial["error"])
            self.assertFalse((root / "failure_data.jsonl").exists())


def injected_load_failure(root):
    args = tp_args(["--model", str(root / "model"), "--lora", str(root / "adapter"),
                    "--data", str(root / "failure_data.jsonl"), "--max-new-tokens", "4",
                    "--dataset-name", "alpaca_cleaned", "--prepare-alpaca-cleaned",
                    "--dataset-dir", str(root / "raw"), "--sample-size", "3", "--num-shots", "1",
                    "--prefix-max-tokens", "64",
                    "--output", str(root / "failure.json")])
    if int(os.environ["RANK"]) == 0:
        with patch("eval_tp.AutoModelForCausalLM.from_pretrained",
                   side_effect=torch.OutOfMemoryError("Injected rank-0 OOM")):
            run_tp(args)
    else:
        run_tp(args)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--injected-load-failure":
        injected_load_failure(Path(sys.argv[2]))
    else:
        unittest.main()
