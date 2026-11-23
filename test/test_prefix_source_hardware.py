"""Check the real launchers' defaults without hardware or downloaded assets."""
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def plan(launcher, *options):
    output = subprocess.check_output([*launcher, "/tmp/oreo-hardware-plan", *options, "--dry-run"],
                                     cwd=ROOT, text=True)
    return [shlex.split(row) for row in output.splitlines() if row.startswith("/") and " -u " in row]


class PrefixSourceHardware(unittest.TestCase):
    entry = [sys.executable, "benchmarks/run_prefix_sources.py"]

    def test_four_methods_use_requested_physical_cards_and_shared_trace(self):
        for size, first, second, lengths in (("13b", 2, 3, (5000, 8000, 10000)),
                                             ("30b", 4, 5, (1000, 2000, 3000))):
            rows = plan(self.entry, "--hardware", "h20", "--model-size", size, "--four-methods",
                        "--gpus", str(first), str(second))
            runs = [row for row in rows if row[2] == "benchmarks/lowra_saturation_exp.py"]
            cases = [(first, "active-reprefill"), (first, "minikv-full_reconstruct"),
                     (first, "cpu-read"), (second, "cpu-read-fixed1-r64")]
            self.assertEqual(len(runs), len(lengths) * len(cases))
            for row, (prefix, gpu, mode) in zip(runs, ((p, g, m) for p in lengths for g, m in cases)):
                self.assertEqual(row[row.index("--adapter-profile") + 1], "hf-llama7b")
                self.assertEqual([row[i + 1] for i, flag in enumerate(row) if flag == "--adapter-base"],
                                 [f"dummy-lora-{size}-rank-{r}" for r in (8, 16, 32, 64)])
                self.assertEqual(row[row.index("--mode") + 1], mode)
                self.assertEqual(row[row.index("--port") + 1], str(24000 + 1000 * gpu))
                self.assertEqual(row[row.index("--workload-trace") + 1],
                                 f"/tmp/oreo-hardware-plan/{size}_p{prefix}/trace.json")

    def test_a100_default_matches_original_launcher(self):
        self.assertEqual(plan(self.entry, "--hardware", "a100"),
                         plan(["bash", "benchmarks/run_prefix_sources_300s.sh"]))

    def test_h20_changes_only_machine_settings_for_matched_duration(self):
        h20 = plan(self.entry, "--hardware", "h20", "--adapter-profile", "chameleon")
        a100 = plan(self.entry, "--hardware", "a100", "--duration", "120")
        self.assertEqual(len(h20), 22 + 55)
        self.assertEqual(len(h20), len(a100))
        for original, actual in zip(a100, h20):
            self.assertEqual(actual[actual.index("--duration") + 1], "120")
            adjusted = original.copy()
            trace = actual[2].endswith("prepare_lmsys_maf_trace.py")
            model = "--tokenizer" if trace else "--model-dir"
            i = actual.index(model) + 1
            self.assertEqual(Path(actual[i]).name, Path(original[i]).name)
            self.assertIn("/.h20/", actual[i])
            expected_python = "/opt/conda/bin/python" if trace else "/opt/conda/envs/slora/bin/python"
            self.assertEqual(actual[0], expected_python)
            adjusted[0] = actual[0]
            machine_options = (model, "--dataset") if trace else (
                model, "--port", "--nccl-port", "--internal-port-start")
            for opt in machine_options:
                i = original.index(opt) + 1
                adjusted[i] = actual[i]
            self.assertEqual(adjusted, actual)
