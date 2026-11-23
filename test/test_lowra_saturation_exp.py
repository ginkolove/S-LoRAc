"""CPU-only checks for paired traces, SSE validation and completion statistics."""

import asyncio
from collections import Counter
from dataclasses import replace
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from benchmarks.lowra_saturation_exp import RequestResult, decode_sse, summarize, trace_for, request_payload
from benchmarks.lowra_saturation_exp import output_window_metrics, send_request, suite_summary
from slora.common.prefix_modes import FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, normalize_prefix_mode


class SaturationTests(unittest.TestCase):
    def test_online_flags_and_real_generation_do_not_use_reference_answer_length(self):
        from benchmarks.lowra_saturation_exp import parse_args, server_command
        with patch("sys.argv", ["bench", "--mode", LAYER_SERIAL, "--output-dir", "/tmp/online-test",
                               "--generation-mode", "eos", "--generation-max-new-tokens", "64",
                               "--max-active-loras", "3"]):
            args = parse_args()
        command = server_command(args, Path("/tmp/runtime.jsonl"))
        self.assertEqual(command[command.index("--scheduler") + 1], "online")
        self.assertEqual(command[command.index("--schedule-interval") + 1], "1")
        self.assertEqual(command[command.index("--max-active-loras") + 1], "3")
        self.assertEqual(args.output_length_admission, "hidden")
        for answer_length in (1, 500):
            payload = request_payload(dict(req_id="r", adapter_dir="a", input_ids=[10] * 8,
                                           output_len=answer_length), args)
            self.assertEqual(payload["parameters"]["max_new_tokens"], 64)
            self.assertFalse(payload["parameters"]["ignore_eos"])

    def test_overlap_launcher_passes_matched_kernels_through_to_server(self):
        from benchmarks.run_decode_saturation import parse_args as pair_args, build_commands
        from benchmarks.lowra_saturation_exp import parse_args, server_command

        with tempfile.TemporaryDirectory() as temporary:
            args = pair_args(["--comparison", "overlap", "--output-dir", temporary])
            commands = build_commands(args)
            self.assertEqual(list(commands), ["serial_bounded", "parallel_bounded"])
            traces = []
            modes = []
            for command in commands.values():
                with patch("sys.argv", command[2:]):
                    child = parse_args()
                modes.append(child.mode)
                self.assertEqual(child.output_len, 512)
                self.assertEqual((child.duration, child.prefix_len, child.query_len), (120, 4608, 480))
                self.assertEqual((child.reconstruction_kernel, child.reconstruction_workers), ("bounded", 1296))
                self.assertEqual(child.num_token, 36864)
                server = server_command(child, Path(temporary) / "stats.jsonl")
                self.assertEqual(server[server.index("--lowra-reconstruction-kernel") + 1], "bounded")
                self.assertEqual(server[server.index("--lowra-reconstruction-workers") + 1], "1296")
                traces.append(trace_for(child, 8, 42))
            self.assertEqual(modes, [LAYER_SERIAL, "layer_prefetch"])
            self.assertEqual(traces[0], traces[1])
            self.assertEqual(len(traces[0][0]), 936)
            legacy = build_commands(pair_args(["--output-dir", temporary]))
            self.assertEqual(list(legacy), ["serial_all", "dynamic_top1"])
            self.assertTrue(all("--reconstruction-kernel" not in cmd for cmd in legacy.values()))

    def test_invalid_kernel_mode_or_worker_options_fail_early(self):
        from benchmarks.lowra_saturation_exp import parse_args

        for mode, extra in ((FULL_REPREFILL, ["--reconstruction-kernel", "bounded"]),
                            (LAYER_SERIAL, ["--reconstruction-workers", "1296"]),
                            (LAYER_SERIAL, ["--reconstruction-kernel", "bounded", "--reconstruction-workers", "0"])):
            with patch("sys.argv", ["benchmark", "--mode", mode, "--output-dir", "/tmp/unused"] + extra), \
                    self.assertRaises(SystemExit):
                parse_args()

    def test_prefetch_top1_pair_uses_bounded_for_both_modes_and_the_same_trace(self):
        from benchmarks.run_decode_saturation import parse_args as pair_args, build_commands
        from benchmarks.lowra_saturation_exp import parse_args, server_command

        with tempfile.TemporaryDirectory() as temporary:
            pair = pair_args(["--comparison", "prefetch-top1", "--output-dir", temporary])
            commands = build_commands(pair)
            self.assertEqual(list(commands), ["parallel_bounded", "dynamic_top1_bounded"])
            traces = []
            for name, command in commands.items():
                with patch("sys.argv", command[2:]):
                    args = parse_args()
                self.assertEqual((args.reconstruction_kernel, args.reconstruction_workers), ("bounded", 1296))
                self.assertEqual((args.output_len, args.prefix_len, args.query_len), (512, 4608, 480))
                self.assertEqual(args.num_token, 36864)
                self.assertEqual(args.pin_hot_adapters, 0)
                if name == "dynamic_top1_bounded":
                    self.assertEqual((args.retain_top_k, args.retain_layers), (1, 32))
                server = server_command(args, Path(temporary) / "stats.jsonl")
                self.assertEqual(server[server.index("--lowra-reconstruction-kernel") + 1], "bounded")
                self.assertEqual(server[server.index("--lowra-reconstruction-workers") + 1], "1296")
                traces.append(trace_for(args, 8, 42))
            self.assertEqual(traces[0], traces[1])
            self.assertEqual(len(traces[0][0]), 936)

    def test_three_way_comparison_preserves_original_fields(self):
        from benchmarks.compare_lowra_saturation import compare
        fields = ("model_dir adapter_base num_adapters prefix_len query_len output_len num_token "
                  "batch_max_tokens running_max_req_size model_seed dummy duration warmup steady_end "
                  "alpha cv prompt_token_id stats_interval preflight_requests")
        config = dict.fromkeys(fields.split(), 1)
        config.update(request_rates=[8], seeds=[42])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = []
            for mode, throughput in (("full_reprefill", 4), ("full_reconstruct", 8), ("layer_serial", 6)):
                directory = root / mode
                directory.mkdir()
                directories.append(directory)
                (directory / "manifest.json").write_text(json.dumps(dict(config=dict(config, mode=mode))))
                point = dict(trace_sha256="paired", result=dict(valid=True, completed_req_per_s_steady=throughput),
                             runtime=dict(running_batch_size=dict(mean=10), backlogged_window=True,
                                          peak_allocated_without_base_model_bytes=2 ** 30))
                point["result"].update(output_window_metrics({0: 10 * throughput}, 10., True))
                (directory / "rate8_seed42.json").write_text(json.dumps(point))
            original = compare(directories[0], directories[2])[0]
            self.assertEqual(original["full_req_per_s"], 4)
            output = root / "comparison.json"
            script = Path(__file__).resolve().parents[1] / "benchmarks/compare_lowra_saturation.py"
            subprocess.run([sys.executable, str(script), "--full-dir", str(directories[0]),
                            "--reconstruct-dir", str(directories[1]), "--serial-dir", str(directories[2]),
                            "--output", str(output)], check=True, capture_output=True)
            combined = json.loads(output.read_text())[0]
            for key, value in original.items():
                self.assertEqual(combined[key], value)
            self.assertEqual(combined["reconstruct_req_per_s"], 8)
            self.assertEqual(combined["reconstruct_peak_output_window_10s"]["output_tokens_per_s"], 8)
            self.assertEqual(combined["serial_over_reconstruct"], 0.75)
            manifest = json.loads((directories[2] / "manifest.json").read_text())
            manifest["config"]["scheduler"] = "online"
            (directories[2] / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "scheduler"):
                compare(directories[0], directories[2])

    def test_new_mode_has_separate_ports_and_old_defaults_are_unchanged(self):
        from benchmarks.lowra_saturation_exp import parse_args
        for mode, ports in (("full_reprefill", (8000, 28765, 12000)),
                            ("layer_serial", (8001, 28766, 13000)),
                            ("full_reconstruct", (8002, 28767, 14000)),
                            (FULL_REPREFILL, (8000, 28765, 12000)),
                            (LAYER_SERIAL, (8001, 28766, 13000)),
                            (FULL_RECONSTRUCT, (8002, 28767, 14000)),
                            ("layer_prefetch", (8004, 28769, 16000))):
            with patch("sys.argv", ["benchmark", "--mode", mode, "--output-dir", "/tmp/unused"]):
                args = parse_args()
            self.assertEqual((args.port, args.nccl_port, args.internal_port_start), ports)
            self.assertEqual(args.mode, normalize_prefix_mode(mode))

    def test_canonical_and_legacy_modes_have_the_same_payload(self):
        request = dict(req_id="r", adapter_dir="a")
        args = SimpleNamespace(prefix_len=17, query_len=5, output_len=4, prompt_token_id=10)
        for legacy, canonical in (("full", FULL_REPREFILL), ("full_reprefill", FULL_REPREFILL),
                                  ("full_reconstruct", FULL_RECONSTRUCT), ("layer_serial", LAYER_SERIAL)):
            args.mode = legacy
            old = request_payload(request, args)
            args.mode = canonical
            self.assertEqual(request_payload(request, args), old)

    def test_full_payload_contains_actual_prefix_and_serial_stays_query_only(self):
        args = SimpleNamespace(mode="full", prefix_len=17, query_len=5,
                               output_len=4, prompt_token_id=10)
        request = dict(req_id="r", adapter_dir="a")
        full = request_payload(request, args)
        self.assertEqual(full["inputs"], [10] * 22)
        args.mode = "layer_serial"
        serial = request_payload(request, args)
        self.assertEqual(serial["inputs"], [10] * 5)
        self.assertEqual(full["lowra_lengths"], serial["lowra_lengths"])
        args.mode = "full_reconstruct"
        reconstructed = request_payload(request, args)
        self.assertEqual(reconstructed["inputs"], serial["inputs"])
        self.assertEqual(reconstructed["lowra_lengths"], serial["lowra_lengths"])
        args.mode = "layer_prefetch"
        self.assertEqual(request_payload(request, args), serial)

    def test_rendezvous_port_survives_config_serialization(self):
        from slora.server.input_params import InputParams
        kwargs = dict(max_req_total_len=2048, max_total_token_num=36864,
                      pool_size_lora=0, batch_max_tokens=36864, running_max_req_size=256,
                      swap=True, prefetch=False, prefetch_size=0, scheduler="slora", profile=False,
                      batch_num_adapters=None, enable_abort=False, dummy=True,
                      no_lora_compute=False, no_lora_swap=False, no_kernel=False,
                      no_mem_pool=False, bmm=False, no_lora=False, fair_weights=[])
        full = InputParams(**kwargs)
        serial = pickle.loads(pickle.dumps(InputParams(**kwargs, nccl_port=28766)))
        self.assertEqual(full.nccl_port, 28765)
        self.assertEqual(serial.nccl_port, 28766)
        kwargs["enable_lowra"] = True
        for mode in (LAYER_SERIAL, "layer_prefetch", "minikv-layer_retained"):
            params = pickle.loads(pickle.dumps(InputParams(**kwargs, lowra_prefix_mode=mode,
                lowra_reconstruction_kernel="bounded", lowra_reconstruction_workers=1296)))
            self.assertEqual(params.lowra_reconstruction_kernel, "bounded")
            self.assertEqual(params.lowra_reconstruction_workers, 1296)

    def test_trace_is_shared_across_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(adapter_base=["dummy-lora-7b-rank-16"], num_adapters=50,
                                   prefix_len=1440, query_len=480, output_len=64, prompt_token_id=10,
                                   duration=5.0, alpha=1.0, cv=1.0, trace_dir=Path(directory), mode="full")
            requests, path, digest = trace_for(args, 8.0, 42)
            args.mode = "layer_serial"
            other, other_path, other_digest = trace_for(args, 8.0, 42)
            self.assertEqual((requests, path, digest), (other, other_path, other_digest))
            args.mode = "full_reconstruct"
            self.assertEqual((requests, path, digest), trace_for(args, 8.0, 42))
            args.mode = "layer_prefetch"
            self.assertEqual((requests, path, digest), trace_for(args, 8.0, 42))
            self.assertTrue(all(0 < req["req_time"] <= args.duration for req in requests))
            self.assertTrue(all(int(req["adapter_dir"].rsplit("-", 1)[1]) < 50 for req in requests))

    def test_steady_uses_completion_not_arrival_and_latency_uses_all_successes(self):
        args = SimpleNamespace(warmup=30.0, steady_end=90.0, bin_seconds=10.0)
        first = RequestResult("early-arrival", "adapter", 1.0, 1.0, 31.0, 40.0,
                              64, 64, True, True, 200, None)
        second = replace(first, req_id="late-completion", scheduled_time=35.0,
                         start_time=35.0, first_token_time=95.0, end_time=100.0)
        result = summarize([first, second], args)
        self.assertEqual(result["steady_completed_requests"], 1)
        self.assertAlmostEqual(result["completed_req_per_s_steady"], 1 / 60)
        self.assertAlmostEqual(result["latency_mean"], (39 + 65) / 2)
        self.assertAlmostEqual(result["ttft_mean"], (30 + 60) / 2)
        self.assertEqual(sum(row["completed_requests"] for row in result["completion_bins"]), 1)

    def test_steady_end_clamps_to_actual_end(self):
        args = SimpleNamespace(warmup=50.0, steady_end=110.0, bin_seconds=10.0)
        requests = [RequestResult(str(i), "adapter", 1.0, 1.0, 55.0, end,
                                  64, 64, True, True, 200, None)
                    for i, end in enumerate((60.0, 87.0))]
        result = summarize(requests, args)
        self.assertEqual(result["steady_end"], 87.0)
        self.assertEqual(result["steady_time"], 37.0)
        self.assertEqual(result["steady_completed_requests"], 2)
        self.assertAlmostEqual(result["completed_req_per_s_steady"], 2 / 37)
        self.assertEqual(sum(row["completed_requests"] for row in result["completion_bins"]), 2)

    def test_sending_duration_and_completion_window_are_independent(self):
        from benchmarks.lowra_saturation_exp import parse_args
        for mode, extra in ((FULL_RECONSTRUCT, ["--pin-hot-adapters", "1"]),
                            ("minikv-layer_retained", ["--pin-hot-adapters", "1", "--retain-layers", "32"]),
                            (LAYER_SERIAL, [])):
            with patch("sys.argv", ["benchmark", "--mode", mode, "--duration", "120",
                                    "--warmup", "70", "--steady-end", "280", "--seeds", "42",
                                    "--output-dir", "/tmp/unused-completion-window"] + extra):
                args = parse_args()
            self.assertEqual(args.duration, 120)
            self.assertEqual(args.steady_end, 280)
            requests = [RequestResult(str(i), "adapter", 1, 1, 75, end,
                                      64, 64, True, True, 200, None)
                        for i, end in enumerate((80, 150, 270, 300))]
            result = summarize(requests, args)
            self.assertEqual(result["steady_end"], 280)
            self.assertEqual(result["steady_time"], 210)
            self.assertEqual(result["steady_completed_requests"], 3)
            self.assertAlmostEqual(result["completed_req_per_s_steady"], 3 / 210)
            self.assertAlmostEqual(result["completed_req_per_s_total"], 4 / 300)
            self.assertAlmostEqual(result["latency_mean"], (80 + 150 + 270 + 300) / 4 - 1)

            truncated = summarize(requests[:2], args)
            self.assertEqual(truncated["steady_end"], 150)
            self.assertEqual(truncated["steady_time"], 80)
            self.assertEqual(truncated["steady_completed_requests"], 2)

    def test_sse_cumulative_count_allows_coalesced_tokens(self):
        event = dict(token=dict(id=10, logprob=-0.1), count_output_tokens=64, finished=True)
        self.assertEqual(decode_sse([json.dumps(event)]), (64, True))
        event["finished"] = -1
        with self.assertRaises(ValueError):
            decode_sse([json.dumps(event)])

    def test_output_windows_count_sse_deltas_at_receipt_across_boundaries(self):
        session = MagicMock()
        response = session.post.return_value.__aenter__.return_value
        response.status = 200
        frames = []
        for count in (3, 5, 9):
            event = dict(token=dict(id=1), count_output_tokens=count, finished=count == 9)
            frames.extend([("data: " + json.dumps(event)).encode(), b"\n"])
        response.content.readline = AsyncMock(side_effect=[*frames, b""])
        args = SimpleNamespace(mode=LAYER_SERIAL, prompt_token_id=1, query_len=1, output_len=9,
                               prefix_len=10, request_timeout=30)
        req = dict(req_id="req", adapter_dir="adapter", req_time=0.)
        counts = Counter()
        with patch("benchmarks.lowra_saturation_exp.time") as clock:
            clock.monotonic.side_effect = [100., 109.9, 110., 120.05, 120.1]
            result = asyncio.run(send_request(session, "http://test", req, args, 100., counts))
        self.assertTrue(result.success, result.error)
        self.assertEqual(counts, {0: 3, 1: 2, 2: 4})
        report = output_window_metrics(counts, result.end_time, result.success)
        windows = report["output_windows_10s"]
        self.assertEqual(sum(w["output_tokens"] for w in windows), result.output_tokens)
        self.assertAlmostEqual(windows[-1]["output_tokens_per_s"], 40.)
        self.assertFalse(windows[-1]["complete"])
        self.assertEqual(report["peak_output_window_10s"], windows[0])

    def test_output_peak_handles_idle_windows_ties_short_runs_and_final_endpoint(self):
        report = output_window_metrics({1: 8, 2: 8}, 30., True)
        self.assertEqual([w["output_tokens"] for w in report["output_windows_10s"]], [0, 8, 8])
        self.assertEqual(report["peak_output_window_10s"]["start"], 10)
        self.assertEqual(output_window_metrics({1: 4, 2: 2}, 20., True)["peak_output_window_10s"]["output_tokens"], 6)
        self.assertIsNone(output_window_metrics({0: 5}, 1., True)["peak_output_window_10s"])
        self.assertIsNone(output_window_metrics({}, 0., True)["peak_output_window_10s"])
        self.assertIsNone(output_window_metrics({0: 5}, 20., False)["peak_output_window_10s"])

    def test_suite_peak_identifies_rate_and_seed_of_best_valid_point(self):
        args = SimpleNamespace(mode=LAYER_SERIAL, request_rates=[3, 4], seeds=[41, 42], plateau_tolerance=.05)
        points = []
        for rate, seed, count, valid in [(3, 41, 48, True), (3, 42, 95, True), (4, 41, 110, False)]:
            metrics = output_window_metrics({1: count}, 25., valid)
            result = dict(metrics, valid=valid, completed_req_per_s_steady=1.)
            points.append(dict(result=result, runtime={}, request_rate=rate, seed=seed))
        peak = suite_summary(points, args)["peak_output_window_10s"]
        self.assertEqual((peak["request_rate"], peak["seed"], peak["start"], peak["output_tokens_per_s"]),
                         (3, 42, 10, 9.5))

    def test_missing_counter_and_nan_are_rejected(self):
        event = dict(token=dict(id=10, logprob=-0.1), finished=True)
        with self.assertRaises(ValueError):
            decode_sse([json.dumps(event)])
        event["count_output_tokens"] = 64
        event["token"]["logprob"] = float("nan")
        with self.assertRaises(ValueError):
            decode_sse([json.dumps(event)])


if __name__ == "__main__":
    unittest.main()
