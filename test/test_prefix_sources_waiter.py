import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


WAITER = Path(__file__).resolve().parents[1] / "benchmarks/run_prefix_sources_after_idle.sh"


class PrefixSourceWaiterTests(unittest.TestCase):
    def replay(self, query, fail_gpu1=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "benchmarks"
            scripts.mkdir()
            shutil.copy(WAITER, scripts)
            out = root / "out"
            events = root / "events"
            for path, body in [
                (root / "nvidia-smi", 'echo "query $*" >> "$EVENTS"\n' + query),
                (root / "sleep", 'echo "sleep $1" >> "$EVENTS"'),
                (scripts / "run_lmsys_preemption.sh", 'echo "suite $*" >> "$EVENTS"\n'
                 '[[ "$3" != 1 || "$FAIL" == 0 ]]'),
                (root / "compare", 'echo "compare $*" >> "$EVENTS"'),
            ]:
                path.write_text("#!/usr/bin/env bash\n" + body + "\n")
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + ":" + os.environ["PATH"], EVENTS=str(events),
                       SLORA_PYTHON=str(root / "compare"), FAIL=str(int(fail_gpu1)))
            result = subprocess.run(["bash", str(scripts / WAITER.name), str(out)], env=env,
                                    capture_output=True, text=True, timeout=5)
            return result.returncode, events.read_text().splitlines()

    def test_busy_then_idle(self):
        code, events = self.replay('if [[ ! -e "$EVENTS.busy" ]]; then touch "$EVENTS.busy"; echo 123; fi')
        self.assertEqual(code, 0)
        self.assertTrue(events[0].startswith("query -i 0,1 "))
        self.assertEqual(events[1], "sleep 2")
        self.assertEqual(events[2], events[0])
        suites = sorted(events[3:5])
        self.assertIn("--gpu 0 --prefix-source-group compute", suites[0])
        self.assertIn("--gpu 1 --prefix-source-group cpu --cpu-prefix-pool-gb 100", suites[1])
        self.assertTrue(events[-1].startswith("compare "))

    def test_idle_starts_without_sleep_and_failure_skips_comparison(self):
        code, events = self.replay("exit 0", fail_gpu1=True)
        self.assertNotEqual(code, 0)
        self.assertEqual([e.split()[0] for e in events], ["query", "suite", "suite"])

    def test_query_failure_does_not_launch(self):
        code, events = self.replay("exit 1")
        self.assertNotEqual(code, 0)
        self.assertEqual(len(events), 1)

    def test_one_queued_job_waits_for_old_lock_then_launches_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "benchmarks"
            results = scripts / "results"
            results.mkdir(parents=True)
            shutil.copy(WAITER, scripts)
            gpu_query = root / "nvidia-smi"
            gpu_query.write_text("#!/bin/sh\nexit 0\n")
            gpu_query.chmod(0o755)
            (scripts / "run_lmsys_preemption.sh").write_text('echo "$@"\n')
            env = dict(os.environ, PATH=str(root) + ":" + os.environ["PATH"], SLORA_PYTHON="true")
            def cmd(out):
                return ["bash", str(scripts / WAITER.name), str(out), "--queue",
                        "--reprefill-num-token", "62000"]
            with (results / "prefix_sources_after_idle.lock").open("a") as old:
                fcntl.flock(old, fcntl.LOCK_EX)
                with subprocess.Popen(cmd(root / "first"), env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as job:
                    try:
                        deadline = time.monotonic() + 5
                        while not (root / "first/waiter.pid").exists():
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(0.01)
                        second = subprocess.run(cmd(root / "second"), env=env, capture_output=True,
                                                text=True, timeout=5)
                        self.assertNotEqual(second.returncode, 0)
                        self.assertIn("not scheduling a duplicate", second.stderr)
                        self.assertFalse((root / "first/gpu0.log").exists())
                    finally:
                        fcntl.flock(old, fcntl.LOCK_UN)
                    output, _ = job.communicate(timeout=5)
                    self.assertEqual(job.returncode, 0, output)
            self.assertIn("--reprefill-num-token 62000", (root / "first/gpu0.log").read_text())
            self.assertNotIn("--reprefill-num-token", (root / "first/gpu1.log").read_text())
