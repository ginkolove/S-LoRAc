"""CPU-only checks: no real GPU queries or serving processes are started."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "benchmarks/run_online_lmsys_after_idle.sh"


class OnlineLauncherTests(unittest.TestCase):
    def test_dry_run_preserves_models_pools_and_online_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            result = subprocess.run(["bash", str(LAUNCHER), str(output), "--dry-run"],
                                    capture_output=True, text=True, check=True)
            commands = [shlex.split(line) for line in result.stdout.splitlines()
                        if "benchmarks/lowra_saturation_exp.py --mode" in line]
            self.assertEqual(len(commands), 4)
            for command in commands:
                for option in ("--max-context-len", "--batch-max-tokens", "--running-max-req-size"):
                    self.assertNotIn(option, command)
                value = lambda flag: command[command.index(flag) + 1]
                self.assertEqual(value("--scheduler"), "online")
                self.assertEqual(value("--schedule-interval"), "1")
                self.assertEqual(value("--generation-mode"), "trace")
                self.assertEqual(value("--request-rates"), "5")
                self.assertEqual(value("--duration"), "60")
                self.assertEqual(value("--prefix-len"), "10000")
                self.assertEqual(value("--seeds"), "42")
                self.assertEqual(value("--num-adapters"), "50")
                size = "13b" if value("--model-dir").endswith("13b") else "7b"
                self.assertEqual(value("--num-token"), "65000" if size == "13b" else "125000")
                self.assertIn("llama%s" % size, value("--workload-trace"))
                self.assertEqual(value("--adapter-base"), "dummy-lora-%s-rank-16" % size)
            self.assertFalse(output.exists())

    def test_trace_python_override_does_not_change_serving_python(self):
        environment = dict(os.environ, SLORA_PYTHON="/fake/serving/python",
                           SLORA_TRACE_PYTHON="/fake/trace/python")
        result = subprocess.run(["bash", str(LAUNCHER), "/tmp/online-dry-run", "--dry-run"],
                                env=environment, capture_output=True, text=True, check=True)
        preparations = [shlex.split(line) for line in result.stdout.splitlines()
                        if "benchmarks/prepare_lmsys_maf_trace.py" in line]
        serving = [shlex.split(line) for line in result.stdout.splitlines()
                   if "benchmarks/lowra_saturation_exp.py" in line]
        self.assertEqual(len(preparations), 2)
        self.assertEqual(len(serving), 4)
        self.assertTrue(all(command[0] == "/fake/trace/python" for command in preparations))
        self.assertTrue(all(command[0] == "/fake/serving/python" for command in serving))

    def run_mocked_launcher(self, query_fails=False, preparation_fails=False,
                            no_wait=False, gpu_busy_once=True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "benchmarks/results").mkdir(parents=True)
            launcher = root / "benchmarks" / LAUNCHER.name
            shutil.copyfile(LAUNCHER, launcher)
            shutil.copyfile(LAUNCHER.parent / "llama_memory_budget.sh",
                            root / "benchmarks/llama_memory_budget.sh")
            binary = root / "bin"
            binary.mkdir()
            python = binary / "fake-python"
            python.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['EVENTS'], 'a') as output:
    output.write(json.dumps(dict(kind='command', args=sys.argv[1:],
                                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))) + '\\n')
if os.environ.get('PREPARATION_FAIL') == '1' and 'benchmarks/prepare_lmsys_maf_trace.py' in sys.argv:
    print('Mock trace preparation error', file=sys.stderr)
    sys.exit(1)
""")
            query = binary / "nvidia-smi"
            query.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
state = pathlib.Path(os.environ['QUERY_STATE'])
count = int(state.read_text()) + 1 if state.exists() else 1
state.write_text(str(count))
with open(os.environ['EVENTS'], 'a') as output:
    output.write(json.dumps(dict(kind='query', count=count, args=sys.argv[1:])) + '\\n')
if os.environ.get('QUERY_FAIL') == '1':
    sys.exit(1)
if count == 1 and os.environ.get('GPU_BUSY_ONCE') == '1':
    print('12345')
""")
            sleep = binary / "sleep"
            sleep.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['EVENTS'], 'a') as output:
    output.write(json.dumps(dict(kind='sleep', args=sys.argv[1:])) + '\\n')
""")
            for path in (python, query, sleep):
                path.chmod(0o755)
            events = root / "events.jsonl"
            environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                               SLORA_PYTHON=str(python), SLORA_TRACE_PYTHON=str(python), EVENTS=str(events),
                               QUERY_STATE=str(root / "query-count"),
                               QUERY_FAIL="1" if query_fails else "0",
                               PREPARATION_FAIL="1" if preparation_fails else "0",
                               GPU_BUSY_ONCE="1" if gpu_busy_once else "0")
            command = ["bash", str(launcher), str(root / "out")]
            if no_wait:
                command.append("--no-wait")
            result = subprocess.run(command,
                                    env=environment, capture_output=True, text=True, timeout=15)
            return result, [json.loads(line) for line in events.read_text().splitlines()]

    def test_waits_for_both_gpus_before_starting_both_model_suites(self):
        result, events = self.run_mocked_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        queries = [event for event in events if event["kind"] == "query"]
        self.assertEqual(len(queries), 3)
        self.assertTrue(all(event["args"][:2] == ["-i", "0,1"] for event in queries))
        last_query = max(i for i, event in enumerate(events) if event["kind"] == "query")
        benches = [(i, event) for i, event in enumerate(events)
                   if event["kind"] == "command" and "benchmarks/lowra_saturation_exp.py" in event["args"]]
        self.assertEqual(len(benches), 4)
        self.assertTrue(all(index > last_query for index, _ in benches))
        for gpu, size in (("0", "7b"), ("1", "13b")):
            commands = [event["args"] for _, event in benches if event["gpu"] == gpu]
            self.assertEqual(len(commands), 2)
            self.assertTrue(all("huggyllama/llama-" + size in command for command in commands))
            self.assertEqual([command[command.index("--mode") + 1] for command in commands],
                             ["minikv-full_reconstruct", "minikv-layer_serial"])

    def test_gpu_query_failure_never_launches_a_benchmark(self):
        result, events = self.run_mocked_launcher(query_fails=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to start", result.stderr)
        self.assertFalse(any("benchmarks/lowra_saturation_exp.py" in event.get("args", [])
                             for event in events))

    def test_trace_preparation_failure_is_visible_and_never_launches_a_benchmark(self):
        result, events = self.run_mocked_launcher(preparation_fails=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Trace preparation failed for 7b", result.stderr)
        self.assertIn("Mock trace preparation error", result.stderr)
        self.assertFalse(any(event["kind"] == "query" for event in events))
        self.assertFalse(any("benchmarks/lowra_saturation_exp.py" in event.get("args", [])
                             for event in events))

    def test_no_wait_starts_both_suites_after_one_check_without_sleep(self):
        result, events = self.run_mocked_launcher(no_wait=True, gpu_busy_once=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(event["kind"] == "query" for event in events), 1)
        self.assertFalse(any(event["kind"] == "sleep" for event in events))
        benches = [event for event in events
                   if "benchmarks/lowra_saturation_exp.py" in event.get("args", [])]
        self.assertEqual(len(benches), 4)
        self.assertEqual({event["gpu"] for event in benches}, {"0", "1"})

    def test_no_wait_busy_gpu_exits_without_sleep_or_benchmark(self):
        result, events = self.run_mocked_launcher(no_wait=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--no-wait refuses", result.stderr)
        self.assertEqual(sum(event["kind"] == "query" for event in events), 1)
        self.assertFalse(any(event["kind"] == "sleep" for event in events))
        self.assertFalse(any("benchmarks/lowra_saturation_exp.py" in event.get("args", [])
                             for event in events))


if __name__ == "__main__":
    unittest.main()
