"""CPU-only checks for preemption attribution and the paired GPU1 launcher."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

from benchmarks.lowra_saturation_exp import RequestResult, preemption_summary


ROOT = Path(__file__).resolve().parents[1]


class PreemptionMetricsTests(unittest.TestCase):
    def test_repeated_preemptions_are_not_distinct_requests_or_pure_cost(self):
        results = [RequestResult(name, "adapter", 0, 0, 1, end, 4, 4, True, True, 200, None)
                   for name, end in (("a", 8), ("b", 4))]
        rows = [dict(event="preempt", monotonic_time=stamp, req_id=name, release_wall_seconds=0.1)
                for stamp, name in ((9, "a"), (11, "a"), (13, "a"), (14, "preflight"))]
        for stamp, new_count, seconds, generated in ((12, 1, 0.5, 1), (14, 0, 0.2, 3)):
            rows.append(dict(event="resume_prefill", monotonic_time=stamp,
                             requests=[dict(req_id="a", input_tokens=5, generated_tokens=generated)],
                             new_requests=new_count, prefill_wall_seconds=seconds))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preemptions.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            summary = preemption_summary(path, 10, results, 8)
        self.assertEqual(summary["preemption_events"], 2)
        self.assertEqual(summary["preempted_requests"], 1)
        self.assertEqual(summary["preemptions_per_request"], {"a": 2})
        self.assertEqual(summary["preempted_request_fraction"], 0.5)
        self.assertEqual(summary["recomputed_tokens"], 14)
        self.assertEqual(summary["resumed_request_events"], 2)
        self.assertEqual(summary["mixed_resume_prefill_batches"], 1)
        self.assertAlmostEqual(summary["resume_batch_prefill_wall_seconds"], 0.7)
        self.assertAlmostEqual(summary["resume_only_batch_prefill_wall_seconds"], 0.2)
        self.assertAlmostEqual(summary["resume_batch_prefill_fraction_of_run"], 0.7 / 8)
        self.assertIn("not_causal_throughput_loss", summary["timing_scope"])
        self.assertEqual(summary["request_groups"]["preempted"]["latency_seconds"]["mean"], 8)
        self.assertEqual(summary["request_groups"]["not_preempted"]["latency_seconds"]["mean"], 4)

    def test_missing_log_is_not_zero_preemptions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "absent.jsonl"
            self.assertEqual(preemption_summary(path, 0, [], 1), dict(available=False))
            path.write_text("")
            summary = preemption_summary(path, 0, [], 1)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["preempted_requests"], 0)
        self.assertEqual(summary["preemption_events"], 0)
        self.assertIsNone(summary["request_groups"]["preempted"]["latency_seconds"]["mean"])

    def test_launcher_pairs_trace_and_runs_full_then_serial_on_gpu1(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = subprocess.check_output(
                ["bash", str(ROOT / "benchmarks/run_lmsys_preemption.sh"), temporary, "--dry-run"],
                cwd=ROOT, text=True)
            commands = [shlex.split(line) for line in output.splitlines() if line.startswith("/")]
            trace = next(cmd for cmd in commands if "benchmarks/prepare_lmsys_maf_trace.py" in cmd)
            runs = [cmd for cmd in commands if "benchmarks/lowra_saturation_exp.py" in cmd]
            value = lambda cmd, flag: cmd[cmd.index(flag) + 1]
            self.assertEqual(value(trace, "--request-rate"), "3")
            self.assertEqual(value(trace, "--duration"), "180")
            self.assertEqual(value(trace, "--maf-minute-end"), "1440")
            self.assertEqual(value(trace, "--data-seed"), "42")
            self.assertEqual([value(cmd, "--mode") for cmd in runs],
                             ["minikv-full_reconstruct", "minikv-layer_serial"])
            for cmd in runs:
                for flag, expected in (("--request-rates", "3"), ("--duration", "180"),
                                       ("--prefix-len", "10000"), ("--num-token", "125000"),
                                       ("--scheduler", "online"), ("--schedule-interval", "1"),
                                       ("--num-adapters", "50"), ("--seeds", "42"),
                                       ("--warmup", "0"), ("--steady-end", "1000000000")):
                    self.assertEqual(value(cmd, flag), expected)
                self.assertEqual(value(cmd, "--workload-trace"), value(trace, "--output"))
            self.assertEqual(value(runs[1], "--reconstruction-kernel"), "bounded")
            self.assertFalse(list(Path(temporary).iterdir()))
        self.assertIn("CUDA_VISIBLE_DEVICES=1", output)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0", output)

    def test_13b_uses_its_own_model_adapter_budget_and_gpu0_ports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = subprocess.check_output(
                ["bash", str(ROOT / "benchmarks/run_lmsys_preemption.sh"), temporary,
                 "--model-size", "13b", "--gpu", "0", "--dry-run"], cwd=ROOT, text=True)
        commands = [shlex.split(line) for line in output.splitlines() if line.startswith("/")]
        trace = next(cmd for cmd in commands if "benchmarks/prepare_lmsys_maf_trace.py" in cmd)
        self.assertEqual(trace[trace.index("--tokenizer") + 1], "huggyllama/llama-13b")
        runs = [cmd for cmd in commands if "benchmarks/lowra_saturation_exp.py" in cmd]
        self.assertEqual(len(runs), 2)
        for cmd in runs:
            for flag, expected in (("--model-dir", "huggyllama/llama-13b"),
                                   ("--adapter-base", "dummy-lora-13b-rank-16"),
                                   ("--num-token", "60000"), ("--port", "9400"),
                                   ("--nccl-port", "34400"), ("--internal-port-start", "50000"),
                                   ("--request-rates", "3"), ("--duration", "180")):
                self.assertEqual(cmd[cmd.index(flag) + 1], expected)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", output)

    def test_invalid_model_and_gpu_fail_before_any_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            for extra in (["--gpu", "other"], ["--model-size", "65b"], ["--gpu"],
                          ["--prefix-len"], ["--prefix-len", "-1"], ["--prefix-len", "0"],
                          ["--prefix-len", "16384"], ["--prefix-len", "4000.5"],
                          ["--prefix-len", "04000"], ["--request-rate"], ["--duration"],
                          ["--request-rate", "0"], ["--request-rate", "-1"],
                          ["--request-rate", "nan"], ["--duration", "0.0"],
                          ["--duration", "-240"], ["--duration", "inf"]):
                result = subprocess.run(
                    ["bash", str(ROOT / "benchmarks/run_lmsys_preemption.sh"), temporary] + extra,
                    cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
            self.assertFalse(list(Path(temporary).iterdir()))

    def test_rate_and_duration_override_match_trace_and_both_13b_runs(self):
        for rate, duration in (("5", "240"), ("2.5", "60.5")):
            with tempfile.TemporaryDirectory() as temporary:
                output = subprocess.check_output(
                    ["bash", str(ROOT / "benchmarks/run_lmsys_preemption.sh"), temporary,
                     "--model-size", "13b", "--prefix-len", "10000", "--gpu", "0",
                     "--request-rate", rate, "--duration", duration, "--dry-run"],
                    cwd=ROOT, text=True)
                self.assertFalse(list(Path(temporary).iterdir()))
            commands = [shlex.split(line) for line in output.splitlines() if line.startswith("/")]
            trace = next(cmd for cmd in commands if "benchmarks/prepare_lmsys_maf_trace.py" in cmd)
            runs = [cmd for cmd in commands if "benchmarks/lowra_saturation_exp.py" in cmd]
            value = lambda cmd, flag: cmd[cmd.index(flag) + 1]
            self.assertEqual(value(trace, "--request-rate"), rate)
            self.assertEqual(value(trace, "--duration"), duration)
            self.assertEqual([value(cmd, "--mode") for cmd in runs],
                             ["minikv-full_reconstruct", "minikv-layer_serial"])
            for cmd in runs:
                for flag, expected in (("--request-rates", rate), ("--duration", duration),
                                       ("--prefix-len", "10000"), ("--num-token", "60000"),
                                       ("--adapter-base", "dummy-lora-13b-rank-16"),
                                       ("--scheduler", "online"), ("--schedule-interval", "1")):
                    self.assertEqual(value(cmd, flag), expected)
                self.assertEqual(value(cmd, "--workload-trace"), value(trace, "--output"))

    def test_30b_uses_4k_prefix_and_matching_trace_on_gpu1(self):
        with tempfile.TemporaryDirectory() as temporary:
            for extra in ([], ["--prefix-len", "4000"]):
                output = subprocess.check_output(
                    ["bash", str(ROOT / "benchmarks/run_lmsys_preemption.sh"), temporary,
                     "--model-size", "30b", "--gpu", "1", "--dry-run"] + extra,
                    cwd=ROOT, text=True)
                commands = [shlex.split(line) for line in output.splitlines() if line.startswith("/")]
                trace = next(cmd for cmd in commands if "benchmarks/prepare_lmsys_maf_trace.py" in cmd)
                value = lambda cmd, flag: cmd[cmd.index(flag) + 1]
                self.assertEqual(value(trace, "--tokenizer"), "huggyllama/llama-30b")
                self.assertEqual(value(trace, "--prefix-len"), "4000")
                runs = [cmd for cmd in commands if "benchmarks/lowra_saturation_exp.py" in cmd]
                self.assertEqual([value(cmd, "--mode") for cmd in runs],
                                 ["minikv-full_reconstruct", "minikv-layer_serial"])
                for cmd in runs:
                    for flag, expected in (("--model-dir", "huggyllama/llama-30b"),
                                           ("--adapter-base", "dummy-lora-30b-rank-16"),
                                           ("--num-token", "10240"), ("--prefix-len", "4000"),
                                           ("--port", "9401"), ("--nccl-port", "34401"),
                                           ("--internal-port-start", "50016"),
                                           ("--request-rates", "3"), ("--duration", "180")):
                        self.assertEqual(value(cmd, flag), expected)
                    self.assertEqual(value(cmd, "--workload-trace"), value(trace, "--output"))
                self.assertIn("CUDA_VISIBLE_DEVICES=1", output)
            self.assertFalse(list(Path(temporary).iterdir()))

    def test_prefix_override_applies_to_trace_and_both_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = subprocess.check_output(
                ["bash", str(ROOT / "benchmarks/run_lmsys_preemption.sh"), temporary,
                 "--prefix-len", "5000", "--dry-run"], cwd=ROOT, text=True)
        commands = [shlex.split(line) for line in output.splitlines() if line.startswith("/")]
        prefixes = [cmd[cmd.index("--prefix-len") + 1] for cmd in commands if "--prefix-len" in cmd]
        self.assertEqual(prefixes, ["5000", "5000", "5000"])


class PreemptionLauncherLifecycleTests(unittest.TestCase):
    def run_mocked(self, busy):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "benchmarks").mkdir()
            launcher = root / "benchmarks/run_lmsys_preemption.sh"
            shutil.copyfile(ROOT / "benchmarks/run_lmsys_preemption.sh", launcher)
            binary = root / "bin"
            binary.mkdir()
            program = binary / "fake-python"
            program.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['EVENTS'], 'a') as output:
    output.write(json.dumps(dict(kind='command', args=sys.argv[1:],
                                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))) + '\\n')
""")
            query = binary / "nvidia-smi"
            query.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
state = pathlib.Path(os.environ['QUERY_STATE'])
count = int(state.read_text()) + 1 if state.exists() else 1
state.write_text(str(count))
with open(os.environ['EVENTS'], 'a') as output:
    output.write(json.dumps(dict(kind='query', args=sys.argv[1:], count=count)) + '\\n')
busy = os.environ['BUSY']
if busy == 'initial' or (busy == 'once' and count == 3) or (busy == 'persistent' and count >= 3):
    print('12345')
""")
            sleep = binary / "sleep"
            sleep.write_text("""#!/usr/bin/env python3
import json, os
with open(os.environ['EVENTS'], 'a') as output:
    output.write(json.dumps(dict(kind='sleep')) + '\\n')
""")
            for path in (program, query, sleep):
                path.chmod(0o755)
            events = root / "events.jsonl"
            environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                               SLORA_PYTHON=str(program), SLORA_TRACE_PYTHON=str(program),
                               QUERY_STATE=str(root / "query-count"), EVENTS=str(events), BUSY=busy)
            result = subprocess.run(["bash", str(launcher), str(root / "out")], env=environment,
                                    capture_output=True, text=True, timeout=30)
            return result, [json.loads(line) for line in events.read_text().splitlines()]

    def test_delayed_cuda_cleanup_does_not_skip_serial_run(self):
        result, events = self.run_mocked("once")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [event for event in events if "benchmarks/lowra_saturation_exp.py" in event.get("args", [])]
        self.assertEqual([event["args"][event["args"].index("--mode") + 1] for event in commands],
                         ["minikv-full_reconstruct", "minikv-layer_serial"])
        self.assertTrue(all(event["gpu"] == "1" for event in commands))
        self.assertEqual(sum(event["kind"] == "sleep" for event in events), 1)
        self.assertIn("CUDA cleanup", result.stdout)

    def test_initial_busy_gpu_still_fails_without_waiting(self):
        result, events = self.run_mocked("initial")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(event["kind"] in ("command", "sleep") for event in events))
        self.assertIn("12345", result.stderr)

    def test_cleanup_wait_is_bounded_and_does_not_launch_on_busy_gpu(self):
        result, events = self.run_mocked("persistent")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(event["kind"] == "sleep" for event in events), 60)
        self.assertEqual(sum("benchmarks/lowra_saturation_exp.py" in event.get("args", [])
                             for event in events), 1)
        self.assertIn("No processes were killed", result.stderr)


if __name__ == "__main__":
    unittest.main()
