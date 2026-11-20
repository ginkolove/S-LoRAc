"""CPU-only tests for real-content, rank-independent replay."""

import asyncio
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.lmsys_maf_trace import CHAMELEON_SCHEMA, SCHEMA, bind_requests, chameleon_profile, load_trace
from benchmarks.prepare_lmsys_maf_trace import (
    aggregate_maf, first_turns, poisson_arrivals, select_samples, save_artifact,
    parse_args as prepare_args,
)
from benchmarks.lowra_saturation_exp import parse_args, request_payload, send_request, server_command, trace_for


def fixture():
    return dict(schema=SCHEMA, config=dict(tokenizer="huggyllama/llama-7b", vocab_size=32000,
                num_adapters=2, prefix_len=10, max_context_len=20, duration=10,
                request_rate=3, seed=42, data_seed=42), requests=[
                    dict(req_id="r0", req_time=0.3, adapter_index=0, source_id="row:a",
                         input_ids=[11] * 7, output_len=2),
                    dict(req_id="r1", req_time=0.8, adapter_index=1, source_id="row:b",
                         input_ids=[12], output_len=8)])


def parse_fixture(path, rank=8, mode="minikv-layer_serial", extra=()):
    argv = ["benchmark", "--mode", mode, "--workload-trace", str(path),
            "--adapter-base", "dummy-lora-7b-rank-%s" % rank, "--num-adapters", "2",
            "--prefix-len", "10", "--max-context-len", "20", "--request-rates", "3",
            "--duration", "10", "--seeds", "42", "--warmup", "0", "--steady-end", "1000",
            "--output-dir", str(path.parent / ("rank%s" % rank))] + list(extra)
    with patch("sys.argv", argv):
        return parse_args()


class TraceTests(unittest.TestCase):
    def test_chameleon_distribution(self):
        for n in (3, 50, 51, 52):
            profile = chameleon_profile(n)
            probs = profile["adapter_probabilities"]
            ranks = profile["adapter_ranks"]
            self.assertEqual(ranks, ([16, 32, 64] * n)[:n])
            for r in (16, 32, 64):
                group = [p for p, rank in zip(probs, ranks) if rank == r]
                self.assertAlmostEqual(sum(group), 1 / 3)
                for j, p in enumerate(group, 1):
                    self.assertAlmostEqual(p * j, group[0])
        with self.assertRaises(ValueError):
            chameleon_profile(2)

    def test_five_modes_bind_the_same_mixed_rank_trace_and_pin_by_prior(self):
        launcher = Path(__file__).resolve().parents[1] / "benchmarks/run_lmsys_preemption.sh"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            data = fixture()
            data.update(schema=CHAMELEON_SCHEMA, chameleon=chameleon_profile(50))
            data["config"].update(num_adapters=50)
            data["requests"][1]["adapter_index"] = 49
            path.write_text(json.dumps(data))
            lines = subprocess.check_output(
                ["bash", str(launcher), tmp, "--prefix-sources", "--dry-run", "--prefix-len", "10",
                 "--request-rate", "3", "--duration", "10"], text=True).splitlines()
            commands = [shlex.split(line)[2:] for line in lines if " -u benchmarks/lowra_saturation_exp.py " in line]
            self.assertEqual(len(commands), 5)
            reference = None
            for command in commands:
                with patch("sys.argv", command):
                    args = parse_args()
                bound, _, digest = trace_for(args, 3, 42)
                self.assertEqual([r["adapter_dir"] for r in bound],
                                 ["dummy-lora-7b-rank-16-0", "dummy-lora-7b-rank-32-16"])
                server = server_command(args, Path(tmp) / "stats")
                dirs = [server[i+1] for i, option in enumerate(server) if option == "--lora-dirs"]
                self.assertEqual(dirs, ["dummy-lora-7b-rank-%d-%d" % (rank, i//3)
                                        for i, rank in enumerate(data["chameleon"]["adapter_ranks"])])
                if reference is None:
                    reference = (digest, bound)
                self.assertEqual(reference, (digest, bound))
                if args.mode == "cpu-read-fixed2":
                    self.assertEqual(args.fixed_hot_indices, [2, 0])  # ID 2 got no requests
                    top1 = ["cpu-read-fixed1-r64" if arg == args.mode else arg for arg in command]
                    with patch("sys.argv", top1):
                        hot = parse_args()
                    self.assertEqual(hot.fixed_hot_indices, [2])
                    cmd = server_command(hot, Path(tmp) / "stats")
                    self.assertEqual(cmd[cmd.index("--cpu-fixed-hot-indices")+1], "2")
            with patch("sys.argv", command + ["--adapter-profile", "maf"]), self.assertRaises(SystemExit):
                parse_args()
            wrong_rank = ["dummy-lora-7b-rank-8" if part == "dummy-lora-7b-rank-64" else part
                          for part in command]
            with patch("sys.argv", wrong_rank), self.assertRaises(SystemExit):
                parse_args()
            data["chameleon"]["adapter_ranks"][2] = 16
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "rank or popularity"):
                load_trace(path)

    def test_launcher_forwards_independent_data_seed(self):
        launcher = Path(__file__).resolve().parents[1] / "benchmarks/run_lmsys_preemption.sh"
        result = subprocess.run(
            ["bash", str(launcher), "/tmp/lmsys-dry-run", "--data-seed", "17", "--dry-run"],
            check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.count("--data-seed 17"), 3)
        self.assertIn("--seed 42 --data-seed 17", result.stdout)
        self.assertIn("--seeds 42", result.stdout)

    def test_rejects_reusing_trace_with_different_data_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(fixture()))
            self.assertEqual(parse_fixture(path, extra=["--data-seed", "42"]).data_seed, 42)
            with self.assertRaises(SystemExit):
                parse_fixture(path, extra=["--data-seed", "17"])

    def test_launcher_uses_matched_budget_and_preserves_explicit_overrides(self):
        launcher = Path(__file__).resolve().parents[1] / "benchmarks/run_lmsys_rank_compare.sh"
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.json"
            trace.write_text(json.dumps(fixture()))
            for extra, model, tokens, duration in (([], "7b", "125000", "10"),
                    (["13b"], "13b", "65000", "10"),
                    (["13b", "60000", "30"], "13b", "60000", "30")):
                with self.subTest(model=model, duration=duration):
                    output = subprocess.run(
                        ["bash", str(launcher), "0", "16", str(trace), str(Path(tmp) / "results")] + extra,
                        env=dict(os.environ, SLORA_PYTHON="/bin/echo"),
                        check=True, capture_output=True, text=True).stdout
                    commands = [line.split() for line in output.splitlines()
                                if "benchmarks/lowra_saturation_exp.py" in line]
                    self.assertEqual(len(commands), 2)
                    for command in commands:
                        self.assertEqual(command[command.index("--duration") + 1], duration)
                        self.assertEqual(command[command.index("--num-token") + 1], tokens)
                        self.assertEqual(command[command.index("--model-dir") + 1], "huggyllama/llama-" + model)
                        self.assertEqual(command[command.index("--adapter-base") + 1],
                                         "dummy-lora-%s-rank-16" % model)

    def test_default_profile_uses_all_d01_and_data_seed_is_independent(self):
        args = prepare_args(["--output", "/tmp/unused-trace.json", "--seed", "7"])
        self.assertEqual((args.maf_minute_start, args.maf_minute_end), (1, 1440))
        self.assertEqual((args.maf_skewness, args.seed, args.data_seed), (1, 7, 42))
        self.assertIsNone(args.max_context_len)

    def test_full_day_includes_late_minute_invocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "maf.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["HashOwner", "HashApp", "HashFunction"] + list(range(1, 1441)))
                early, late = [0] * 1440, [0] * 1440
                early[0], late[60], late[1439] = 10, 20, 30
                writer.writerow(["o", "a", "early"] + early)
                writer.writerow(["o", "a", "late"] + late)
            hour = aggregate_maf(path, 2, 1, 1, 60)
            day = aggregate_maf(path, 2, 1, 1, 1440)
            self.assertEqual(hour["adapter_loads"], [10, 0])
            self.assertEqual(day["adapter_loads"], [50, 10])
            self.assertEqual(day["total_invocations"], 60)
            self.assertEqual(day["distribution_summary"]["nonzero_adapters"], 2)
            self.assertAlmostEqual(sum(day["adapter_probabilities"]), 1)

    def test_prepared_artifacts_are_reusable_but_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prepared" / "order.json"
            first = save_artifact(path, dict(source_ids=["b", "a"], data_seed=42))
            self.assertEqual(first, save_artifact(path, dict(source_ids=["b", "a"], data_seed=42)))
            with self.assertRaises(FileExistsError):
                save_artifact(path, dict(source_ids=["a", "b"], data_seed=42))

    def test_fixed_candidate_order_is_shared_across_short_and_long_traces(self):
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [1] * len(text))
        candidates = [dict(source_id=str(i), source_model="a", query="q", answer="a")
                      for i in range(100)]
        order = np.random.default_rng(42).permutation(len(candidates))
        short, _ = select_samples(candidates, 3, tokenizer, None, 10, 20, order=order)
        long, _ = select_samples(candidates, 12, tokenizer, None, 10, 20, order=order)
        self.assertEqual(short, long[:3])
        self.assertEqual(len(set(int(index) for index in order)), 100)
        self.assertEqual([int(sample["source_id"]) for sample in short], list(order[:3]))

    def test_maf_aggregation_sorts_then_block_round_robins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "maf.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["HashOwner", "HashApp", "HashFunction", "1", "2"])
                for key, count in (("d", 10), ("a", 40), ("c", 20), ("b", 30), ("z", 0)):
                    writer.writerow(["owner", "app", key, count, 0])
            one = aggregate_maf(path, 2, 1, 1, 2)
            two = aggregate_maf(path, 2, 2, 1, 2)
            self.assertEqual(one["adapter_loads"], [60, 40])
            self.assertEqual(two["adapter_loads"], [70, 30])
            self.assertEqual(one["active_functions"], 4)

    def test_independent_poisson_stream_does_not_fix_request_count(self):
        def generate():
            streams = np.random.SeedSequence(42).spawn(3)
            return poisson_arrivals(3, 10, np.random.default_rng(streams[0]))
        self.assertEqual(generate(), generate())
        self.assertTrue(all(0 < value < 10 for value in generate()))
        self.assertEqual(generate(), sorted(set(generate())))

    def test_first_turns_do_not_replay_followups_as_independent_requests(self):
        conversation = [dict(role="user", content="q"), dict(role="assistant", content="a"),
                        dict(role="user", content="followup"), dict(role="assistant", content="second")]
        rows = [dict(question_id="id", model_a="a", model_b="b",
                     conversation_a=conversation, conversation_b=conversation)]
        candidates = first_turns(rows)
        self.assertEqual(len(candidates), 2)
        self.assertEqual([item["query"] for item in candidates], ["q", "q"])
        self.assertNotEqual(candidates[0]["source_id"], candidates[1]["source_id"])

    def test_explicit_context_fails_without_resampling_and_legacy_filter_is_opt_in(self):
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(range(len(text))))
        rng = SimpleNamespace(permutation=lambda count: range(count))
        candidates = [dict(source_id="long", source_model="a", query="x" * 12, answer="abc"),
                      dict(source_id="short", source_model="a", query="hi", answer="abc")]
        with self.assertRaisesRegex(ValueError, "sample long needs 25"):
            select_samples(candidates, 1, tokenizer, rng, 10, 20)
        samples, rejected = select_samples(candidates, 1, tokenizer, rng, 10, 20,
                                           filter_over_context=True)
        self.assertEqual(rejected, 1)
        self.assertEqual(samples[0]["input_ids"], [0, 1])
        self.assertEqual(samples[0]["output_len"], 3)

    def test_auto_context_keeps_long_samples_identical_across_prefix_lengths(self):
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [1] * len(text))
        candidates = [dict(source_id="long", source_model="a", query="x" * 2000, answer="a" * 500),
                      dict(source_id="short", source_model="a", query="hi", answer="abc")]
        small, _ = select_samples(candidates, 2, tokenizer, None, 10000, None, order=[0, 1])
        large, rejected = select_samples(candidates, 2, tokenizer, None, 15000, None, order=[0, 1])
        self.assertEqual(small, large)
        self.assertEqual(rejected, 0)
        self.assertEqual(15000 + len(large[0]["input_ids"]) + large[0]["output_len"], 17500)

    def test_benchmark_inherits_context_and_removes_fixed_batch_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            data = fixture()
            data["config"].update(prefix_len=15000, max_context_len=18000)
            data["requests"][0].update(input_ids=[11] * 2000, output_len=1000)
            path.write_text(json.dumps(data))
            argv = ["bench", "--mode", "minikv-layer_serial", "--workload-trace", str(path),
                    "--prefix-len", "15000", "--num-adapters", "2", "--request-rates", "3",
                    "--duration", "10", "--seeds", "42", "--warmup", "0", "--steady-end", "1000",
                    "--num-token", "12000", "--output-dir", tmp]
            with patch("sys.argv", argv):
                args = parse_args()
            self.assertEqual(args.max_context_len, 18000)
            self.assertEqual(args.batch_max_tokens, 12000)
            self.assertEqual(args.running_max_req_size, 2)
            command = server_command(args, Path(tmp) / "stats.jsonl")
            self.assertEqual(command[command.index("--max_req_input_len") + 1], "17000")
            self.assertEqual(command[command.index("--max_req_total_len") + 1], "18000")
            with patch("sys.argv", argv + ["--batch-max-tokens", "2500", "--running-max-req-size", "1"]):
                capped = parse_args()
            self.assertEqual((capped.batch_max_tokens, capped.running_max_req_size), (2500, 1))

    def test_synthetic_context_and_running_limit_are_auto_but_explicit_values_survive(self):
        argv = ["bench", "--mode", "minikv-layer_serial", "--prefix-len", "15000",
                "--query-len", "2000", "--output-len", "1000", "--output-dir", "/tmp/unused"]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual(args.max_context_len, 18000)
        self.assertEqual(args.running_max_req_size, args.num_token)
        with patch("sys.argv", argv + ["--max-context-len", "32768", "--running-max-req-size", "256"]):
            args = parse_args()
        self.assertEqual((args.max_context_len, args.running_max_req_size), (32768, 256))

    def test_all_modes_and_ranks_share_ids_arrivals_tokens_and_adapter_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(fixture()))
            reference = None
            for rank in (8, 16):
                for mode in ("minikv-full_reconstruct", "minikv-layer_serial"):
                    args = parse_fixture(path, rank, mode)
                    requests, _, digest = trace_for(args, 3, 42)
                    invariant = ([{key: value for key, value in request.items() if key != "adapter_dir"}
                                  for request in requests], digest)
                    if reference is None:
                        reference = invariant
                    self.assertEqual(invariant, reference)
                    self.assertIn("rank-%s" % rank, requests[0]["adapter_dir"])
                    # Separate per-request maxima need not fit as a single request.
                    self.assertEqual((args.query_len, args.output_len), (7, 8))
                    command = server_command(args, Path(tmp) / "stats.jsonl")
                    self.assertEqual(command[command.index("--max_req_input_len") + 1], "17")
                    self.assertEqual(request_payload(requests[1], args)["parameters"]["max_new_tokens"], 8)
            path.write_text(json.dumps(dict(fixture(), note="modified")))
            with self.assertRaisesRegex(ValueError, "changed"):
                trace_for(args, 3, 42)

    def test_mismatch_and_invalid_tokens_fail_before_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(fixture()))
            for extra in (["--duration", "11"], ["--request-rates", "8"], ["--seeds", "0"],
                          ["--measure-saturation"], ["--cv", "2"], ["--prefix-len", "11"]):
                with self.assertRaises(SystemExit):
                    parse_fixture(path, extra=extra)
            for field, value in (("input_ids", [-1]), ("output_len", 0), ("adapter_index", 2),
                                 ("req_time", float("nan")), ("output_len", 100)):
                data = fixture()
                data["requests"][0][field] = value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_trace(path)

    def test_sse_completion_uses_each_requests_own_output_length(self):
        class Content:
            def __init__(self, count):
                event = dict(token=dict(id=1, logprob=-1), finished=True, count_output_tokens=count)
                self.lines = iter([("data: " + json.dumps(event) + "\n").encode(), b"\n", b""])

            async def readline(self):
                return next(self.lines)

        class Response:
            status = 200

            def __init__(self, count):
                self.content = Content(count)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(fixture()))
            args = parse_fixture(path)
            request = bind_requests(fixture(), ["a", "b"])[0]
            session = SimpleNamespace(post=lambda *a, **kw: Response(2))
            result = asyncio.run(send_request(session, "http://test", request, args, time.monotonic()))
            self.assertTrue(result.success, result.error)
            self.assertEqual((result.query_tokens, result.expected_output_tokens), (7, 2))
            self.assertEqual(asdict(result)["source_id"], "row:a")
            session.post = lambda *a, **kw: Response(8)
            result = asyncio.run(send_request(session, "http://test", request, args, time.monotonic()))
            self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
