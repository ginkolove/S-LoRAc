"""Coverage for lifetime board peaks, per-method pool calibration and completed work."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import h20_memory_guard as guard
import h20_peak_report
import run_h20_memory_cap as cap
from lmsys_maf_trace import hf_profile
from slora.common import benchmark_gpu
from slora.common.startup_memory_profile import extend_decode_history, fit_lengths, probe_shape
from slora.server.router.online_req_queue import OnlineReqQueue
from slora.common.mem_allocator import MemoryAllocator
from benchmarks.lowra_saturation_exp import output_window_metrics
import torch


def workload():
    return dict(schema="lmsys-hf-llama7b-poisson-v1", config=dict(num_adapters=4, prefix_len=32, duration=120),
                **{cap.HF_PROFILE: hf_profile(4)}, requests=[
                    dict(req_id="a", req_time=.1, source_id="s1", input_ids=[5], adapter_index=0, output_len=20),
                    dict(req_id="b", req_time=.7, source_id="s2", input_ids=[2, 3], adapter_index=3, output_len=48)])


class GuardTests(unittest.TestCase):
    def test_sample_includes_startup_and_teardown_and_latches_a_short_observed_breach(self):
        device = MagicMock(uuid="GPU-test", cap_bytes=guard.CAP_BYTES)
        device.used.side_effect = (300, guard.CAP_BYTES + 1, 300)
        with tempfile.TemporaryDirectory() as tmp:
            with guard.DeviceSampler(device, Path(tmp), 60) as monitor:
                monitor._sample()
            result = monitor.summary()
            self.assertTrue(result["over_limit"])
            self.assertEqual((result["sampled_peak_bytes"], result["final_used_bytes"], result["samples"]),
                             (guard.CAP_BYTES + 1, 300, 3))
            self.assertEqual(len((Path(tmp) / "device_memory.csv").read_text().splitlines()), 4)

    def test_device_capacity_accepts_startup_and_records_peaks_above_80_gb(self):
        nv = MagicMock()
        nv.nvmlDeviceGetName.return_value = "NVIDIA H20"
        nv.nvmlDeviceGetUUID.return_value = "GPU-test"
        nv.nvmlDeviceGetMigMode.return_value = (0, 0)
        info = SimpleNamespace(total=102 * cap.GB, used=cap.GB)
        nv.nvmlDeviceGetMemoryInfo.return_value = info
        device = guard.PhysicalGPU(nv, 0, use_device_capacity=True)
        self.assertEqual(guard.PhysicalGPU(nv, 0).cap_bytes, guard.CAP_BYTES)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            quota = 99 * cap.GB
            guard.save_json(directory / "mps_client.json", dict(gpu_uuid=device.uuid, quota_bytes=quota,
                                                               initial_cuda_free_bytes=98 * cap.GB))
            guard.verify_startup_budget(device, directory, quota)
            with guard.DeviceSampler(device, directory, 60) as monitor:
                info.used = 92 * cap.GB
                monitor._sample()
                info.used = cap.GB
            record = monitor.summary()
            self.assertFalse(record["over_limit"])
            self.assertEqual((record["cap_bytes"], record["sampled_peak_bytes"]), (info.total, 92 * cap.GB))

    def test_live_peak_survives_a_drop_and_oom_is_labelled_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            point = Path(tmp)
            method = "cpu-read-fixed1-r64"
            attempt = point / method / "formal_65000"
            attempt.mkdir(parents=True)
            guard.save_json(point / "experiment.json", dict(fixed_pool_tokens=65000))
            (attempt / "device_memory.csv").write_text("elapsed_seconds,nvml_device_used_bytes\n"
                                                        "0.0,10\n1.0,90000000000\n2.0,0\n3.0,95000000001")
            self.assertEqual(h20_peak_report.method_peak(point, method), (65000, 90 * cap.GB, "sampled so far"))
            guard.save_json(attempt / "memory.json", dict(cuda_oom=True, sampled_peak_bytes=91 * cap.GB))
            self.assertEqual(h20_peak_report.method_peak(point, method), (65000, 91 * cap.GB, "OOM"))

    def test_only_router_gets_gpu_and_restores_parent_on_spawn_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "CUDA_VISIBLE_DEVICES": "", "SLORA_BENCHMARK_MPS_GPU": "GPU-test",
                "SLORA_BENCHMARK_MPS_REPORT": str(Path(tmp) / "mps_client.json")}), \
                patch.object(benchmark_gpu.mp, "get_start_method", return_value="spawn"), \
                patch.object(benchmark_gpu.mp, "Process") as create:
            child = create.return_value
            child.pid = 111
            def start():
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-test")
            child.start.side_effect = start
            benchmark_gpu.spawn_router(None, (SimpleNamespace(tp=1),))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual((Path(tmp) / "router.pid").read_text(), "111")
            child.start.side_effect = OSError("no child")
            with self.assertRaises(OSError):
                benchmark_gpu.spawn_router(None, (SimpleNamespace(tp=1),))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            with self.assertRaises(ValueError):
                benchmark_gpu.spawn_router(None, (SimpleNamespace(tp=2),))

    def test_incremental_cuda_oom_including_final_exit_and_split_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("general runtime error\nCUDA out of ")
            found, offset = guard.read_cuda_oom(log, 0)
            self.assertFalse(found)
            with log.open("a") as stream:
                stream.write("memory. Tried to allocate")
            self.assertTrue(guard.read_cuda_oom(log, offset)[0])

    def test_worker_can_load_weights_only_if_its_free_quota_fits_on_board(self):
        device = MagicMock(uuid="gpu", cap_bytes=guard.CAP_BYTES)
        quota = 75 * cap.GB
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            guard.save_json(dest / "mps_client.json", dict(gpu_uuid="gpu", quota_bytes=quota,
                                                           initial_cuda_free_bytes=quota - cap.GB))
            device.used.return_value = 7 * cap.GB
            with self.assertRaisesRegex(RuntimeError, "startup budget"):
                guard.verify_startup_budget(device, dest, quota)
            self.assertFalse((dest / "startup_budget.json").exists())
            device.used.return_value = 2 * cap.GB
            guard.verify_startup_budget(device, dest, quota)
            self.assertEqual(json.loads((dest / "startup_budget.json").read_text())
                             ["board_used_plus_client_free_bytes"], 76 * cap.GB)

    def test_private_worker_must_be_visible_and_foreign_process_fails_closed(self):
        device = MagicMock(uuid="gpu")
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "router.pid"
            pidfile.write_text("11")
            with self.assertRaisesRegex(RuntimeError, "cannot see"):
                guard.check_gpu_owners(device, set(), None, {}, pidfile, require_client=True)
            with patch.object(guard.subprocess, "check_output", side_effect=["10\n", "12\n"]):
                with self.assertRaisesRegex(RuntimeError, "Unexpected CUDA clients"):
                    guard.check_gpu_owners(device, {100, 102}, None, {}, pidfile, require_client=True)
            with patch.object(guard.subprocess, "check_output", side_effect=["10\n", ""]):
                with self.assertRaisesRegex(RuntimeError, "worker is missing"):
                    guard.check_gpu_owners(device, {100}, None, {}, pidfile, require_client=True)

    def test_only_live_other_board_router_with_zero_bytes_in_every_list_is_excluded(self):
        nv = MagicMock()
        nv.nvmlDeviceGetUUID.return_value = "GPU-local"
        nv.nvmlDeviceGetName.return_value = "NVIDIA H20"
        nv.nvmlDeviceGetMigMode.return_value = (0, 0)
        device = guard.PhysicalGPU(nv, 0)
        peer, = guard.nvml_host_pids({os.getpid()})
        def entry(pid, size):
            return SimpleNamespace(pid=pid, usedGpuMemory=size)
        nv.nvmlDeviceGetComputeRunningProcesses.return_value = [entry(peer, 0), entry(peer + 1, 0)]
        nv.nvmlDeviceGetGraphicsRunningProcesses.return_value = []
        with tempfile.TemporaryDirectory() as tmp:
            pipe = Path(tmp) / "slora-mps-test"
            pipe.mkdir()
            with patch.object(guard.tempfile, "gettempdir", return_value=tmp), patch.dict(os.environ, {
                    "CUDA_MPS_PIPE_DIRECTORY": str(pipe), "CUDA_VISIBLE_DEVICES": "GPU-other",
                    "SLORA_BENCHMARK_MPS_GPU": "GPU-other"}):
                with benchmark_gpu.register_mps_router():
                    self.assertEqual(device.processes(), {peer + 1})
                    self.assertEqual(guard.other_board_routers("GPU-other"), set())
                    for size in (1, None):
                        nv.nvmlDeviceGetMPSComputeRunningProcesses.return_value = [entry(peer, size)]
                        self.assertEqual(device.processes(), {peer, peer + 1})
                nv.nvmlDeviceGetMPSComputeRunningProcesses.return_value = []
                self.assertEqual(device.processes(), {peer, peer + 1})


class StartupProfileTests(unittest.TestCase):
    def test_impress_replays_cpu_read_workload_with_full_slot_budget(self):
        dest = Path("13b_p10000")
        cpu = ["runner", "--mode", "cpu-read", "--output-dir", str(dest / "cpu-read"),
               "--workload-trace", "frozen.json", "--cpu-prefix-pool-gb", "100", "--num-token", "65000"]
        with patch.object(cap, "point_commands", return_value=[("cpu-read", cpu)]):
            commands = cap.commands_for(dest, "13b", 10000, {}, ["cpu-read", "impress"])
            custom = cap.commands_for(dest, "13b", 10000, {}, ["cpu-read", "impress"], impress_read_percent=60)
        self.assertEqual(custom["cpu-read"], commands["cpu-read"])
        self.assertEqual(cap.option(custom["impress"], "--impress-read-percent"), "60")
        self.assertIn("impress", cap.ALL_METHODS)
        self.assertEqual(commands["impress"], [str(v).replace("cpu-read", "impress") for v in commands["cpu-read"]])
        model = dict(num_hidden_layers=40, hidden_size=5120, num_attention_heads=40)
        self.assertEqual(cap.pool_geometry("impress", model, workload()), cap.pool_geometry("cpu-read", model, workload()))

    def test_fixed_hot_cpu_replay_uses_requested_slots_without_calibration_or_oom_backoff(self):
        trace = workload()
        command = ["runner", "--mode", "cpu-read-fixed1-r64"]
        with tempfile.TemporaryDirectory() as tmp, patch.object(cap, "pool_geometry", return_value=(819200, 21561)), \
                patch.object(cap, "run_attempt", side_effect=[dict(sampled_board_peak_bytes=71 * cap.GB), None]) as run:
            dest = Path(tmp)
            args = (dest, command, "source", {}, "gpu", 77 * cap.GB, ("trace", trace, "sha"), None, {})
            result = cap.measure_method(*args, fixed_slots=45532)
            self.assertNotIn("startup_profile", result)
            self.assertEqual(result["experiment_sampled_peak_bytes"], 71 * cap.GB)
            with self.assertRaisesRegex(cap.PoolSizeOOM, "45532"):
                cap.measure_method(*args, fixed_slots=45532)
            with self.assertRaisesRegex(ValueError, "at least 21561"):
                cap.measure_method(*args, fixed_slots=21000)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.args, (dest / "formal_45532", command, "source", {}, "gpu", 77 * cap.GB,
                                             45532, ("trace", trace, "sha")))
                self.assertIsNone(call.kwargs["reuse_manifest"])

    def test_reuse_runs_only_formal_at_saved_capacity_and_keeps_current_peak(self):
        trace = workload()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            (old / "run").mkdir(parents=True)
            raw = dict(mode="cpu-read", trace_sha256="sha", pool_tokens=4000, dynamic_pool_slots=4000,
                       static_token_equivalents=0, slot_bytes=10**6, phases={k: {} for k in cap.PROFILE_PHASES})
            profile = dict(raw, directory=str(old), single_request_floor=1000, headroom_bytes=2*cap.GB,
                           sampled_board_peak_bytes=74*cap.GB, startup_sampled_peak_bytes=78*cap.GB,
                           client_quota_bytes=77*cap.GB, torch_peak_reserved_bytes=73*cap.GB, other_client_bytes=0)
            memory = dict(verified_before_model_load=True, returncode=0, error=None, cuda_oom=False,
                          over_limit=False, monitor_error=None, cap_bytes=cap.CAP_BYTES,
                          sampled_peak_bytes=profile["sampled_board_peak_bytes"], client_quota_bytes=77*cap.GB)
            for path, value in ((old/"startup_profile.json", profile), (old/"forward_profile.json", raw),
                                (old/"memory.json", memory), (old/"run/manifest.json", {})):
                guard.save_json(path, value)
            path = old / "startup_profile.json"
            with self.assertRaisesRegex(ValueError, "method and frozen trace"):
                cap.read_verified_profile(path, "cpu-read-fixed1-r64", "sha")
            with self.assertRaisesRegex(ValueError, "method and frozen trace"):
                cap.read_verified_profile(path, "cpu-read", "other")
            prior = cap.read_verified_profile(path, "cpu-read", "sha")
            with patch.object(cap, "pool_geometry", return_value=(10**6, 1000)), patch.object(
                    cap, "run_attempt", return_value=dict(sampled_board_peak_bytes=70*cap.GB)) as replay:
                result = cap.measure_method(root/"new", ["runner","--mode","cpu-read"], "", {}, "", 77*cap.GB,
                                            ("",trace,"sha"), None, {}, reused=prior)
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(replay.call_args.args[0].name, "formal_4000")
            self.assertEqual(replay.call_args.kwargs["reuse_manifest"], old/"run/manifest.json")
            self.assertEqual(result["experiment_sampled_peak_bytes"], 70*cap.GB)
            self.assertEqual(json.loads((root/"new/startup_profile.json").read_text())["reused_from"], str(path))
            guard.save_json(old/"memory.json", dict(memory, cuda_oom=True))
            with self.assertRaisesRegex(ValueError, "successful 80 GB profile"):
                cap.read_verified_profile(path, "cpu-read", "sha")

    def test_reuse_checks_inference_options_and_sources_before_starting_the_server(self):
        from benchmarks.lowra_saturation_exp import verify_reused_configuration

        command = ["python", "--port", "1", "--nccl_port", "2", "--internal-port-start", "3",
                   "--lowra-runtime-stats-path", "old", "--running_max_req_size", "351"]
        old = dict(command=command, source_sha256={"slora/model.py":"a"}, packages={"torch":"2", "triton":"3"})
        new = dict(old, command=list(command))
        new["command"][2], new["command"][8] = "4000", "new"
        verify_reused_configuration(old, new)
        with self.assertRaisesRegex(ValueError, "Server configuration"):
            verify_reused_configuration(old, dict(new, command=[*new["command"][:-1], "350"]))
        with self.assertRaisesRegex(ValueError, "Inference implementation"):
            verify_reused_configuration(old, dict(new, source_sha256={"slora/model.py":"b"}))

    def test_trace_shape_bounds_and_pool_floor_use_actual_rank(self):
        original = workload()
        self.assertEqual(cap.profile_spec(original, "sha"),
                         dict(query_lengths=[2, 1], history_lengths=[49, 20], adapter_indices=[0, 3],
                              trace_sha256="sha"))
        cfg = dict(num_hidden_layers=4, hidden_size=512, num_attention_heads=8)
        full = cap.pool_geometry(cap.RECONSTRUCTION[0], cfg, original)
        serial = cap.pool_geometry(cap.RECONSTRUCTION[1], cfg, original)
        self.assertEqual(full[0], 4 * 4 * 512)
        self.assertEqual(full[1] - serial[1], 32 - 32 // 4)
        self.assertEqual(cap.pool_geometry(cap.V3, cfg, original), serial)
        self.assertEqual(cap.pool_geometry("minikv-serial-le16-full-gt16", cfg, original), full)

    def test_candidate_and_formal_oom_both_require_new_profile_before_timed_replay(self):
        trace = workload()
        cmd = ["runner", "--num-token", "4000", "--mode", cap.V3]
        pilot = dict(sampled_board_peak_bytes=64 * cap.GB, slot_bytes=10**6, headroom_bytes=9 * cap.GB,
                     dynamic_pool_slots=999, static_token_equivalents=1, directory="profile")
        formal = {"completed_requests": 2, "sampled_board_peak_bytes": 70 * cap.GB}
        with tempfile.TemporaryDirectory() as tmp, patch.object(cap, "pool_geometry", return_value=(10**6, 1000)), \
                patch.object(cap, "run_attempt", side_effect=[pilot, None, pilot, None, pilot, formal]) as attempt:
            result = cap.measure_method(Path(tmp), cmd, "source", {}, "device", 77 * cap.GB,
                                        ("trace", trace, "real"), "spec", {})
            calls = [c.args for c in attempt.call_args_list]
            self.assertEqual(result["completed_requests"], 2)
            self.assertEqual(result["experiment_sampled_peak_bytes"], 70 * cap.GB)
            self.assertEqual([c[6] for c in calls], [1000, 9000, 7000, 7000, 5500, 5500])
            self.assertEqual([len(c) for c in calls], [9, 9, 9, 8, 9, 8])
            with self.assertRaisesRegex(RuntimeError, "one request"):
                cap.smaller_pool(1000, 1000, 10**6)

    def test_peak_includes_discarded_candidates_and_cold_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, peak in (("profile_10", 20), ("profile_12_retry1", 70), ("formal_10", 60)):
                path = root / name
                path.mkdir()
                guard.save_json(path / "memory.json", dict(sampled_peak_bytes=peak))
            self.assertEqual(cap.sampled_attempt_peak(root, "profile", 45), 70)
            self.assertEqual(cap.sampled_attempt_peak(root, "formal", 45), 60)

    def test_profile_only_shrinks_for_margin_and_never_launches_traffic(self):
        result = dict(slot_bytes=10**6, dynamic_pool_slots=999, static_token_equivalents=1,
                      sampled_board_peak_bytes=70 * cap.GB, directory="profile", headroom_bytes=3 * cap.GB)
        with tempfile.TemporaryDirectory() as tmp, patch.object(cap, "pool_geometry", return_value=(10**6, 1000)), \
                patch.object(cap, "run_attempt", side_effect=[result, dict(result, headroom_bytes=0), result]) as run:
            summary = cap.measure_method(Path(tmp), ["runner", "--mode", cap.V3], "", {}, "", 77 * cap.GB,
                                         ("", workload(), ""), "spec", {}, True)
            self.assertTrue(summary["profile_only"])
            self.assertEqual([c.args[6] for c in run.call_args_list], [1000, 3000, 1932])
            self.assertTrue(all(c.args[-1] == "spec" for c in run.call_args_list))

    def test_formal_replay_stops_on_first_foreign_gpu_process(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                cap, "guarded_replay", side_effect=guard.ForeignGPUProcess("unexpected PID 1234")) as replay:
            device = MagicMock()
            dest = Path(tmp) / "formal_45532"
            with self.assertRaisesRegex(guard.ForeignGPUProcess, "1234"):
                cap.run_attempt(dest, ["runner", "--mode", "cpu-read-fixed1-r64"], "", {}, device, cap.GB,
                                45532, ("trace", workload(), "sha"))
            replay.assert_called_once()
            self.assertEqual(replay.call_args.args[-1], dest)
            device.processes.assert_not_called()

    def test_transient_foreign_context_can_retry_a_startup_probe(self):
        def replay(cmd, source, env, device, quota, directory):
            self.assertEqual(cap.option(cmd, "--output-dir"), str(directory / "run"))
            self.assertEqual(cap.option(cmd, "--running-max-req-size"), "2")
            directory.mkdir()
            if directory.name == "profile_1000":
                guard.save_json(directory / "memory.json", dict(over_limit=False, monitor_error=None))
                raise guard.ForeignGPUProcess("different worker")
            phases = {name: dict(peak_reserved_bytes=200, other_client_bytes=10) for name in cap.PROFILE_PHASES}
            guard.save_json(directory / "forward_profile.json", dict(mode=cap.V3, trace_sha256="sha",
                            pool_tokens=1000, dynamic_pool_slots=800, static_token_equivalents=200, phases=phases))
            return dict(cap_bytes=guard.CAP_BYTES, over_limit=False, monitor_error=None, cuda_oom=False, returncode=0,
                        verified_before_model_load=True, sampled_peak_bytes=300)

        with tempfile.TemporaryDirectory() as tmp, patch.object(cap, "guarded_replay", side_effect=replay) as run:
            device = MagicMock()
            device.processes.return_value = set()
            result = cap.run_attempt(Path(tmp) / "profile_1000", ["runner", "--mode", cap.V3], "", {}, device,
                                     cap.GB, 1000, ("trace", workload(), "sha"), spec_path=Path(tmp) / "probe.json")
            self.assertEqual(run.call_count, 2)
            self.assertEqual(Path(result["directory"]).name, "profile_1000_retry1")

    def test_shape_uses_actual_prefix_build_and_slot_limits_then_fills_private_history(self):
        q = OnlineReqQueue(115, 54, 5, prefix_len=20, prefix_mode="active-reprefill", layer_num=10)
        dirs, prompts, contexts = probe_shape(q, {"a": 2, "b": 2}, ["a", "b", "a", "b", "a"],
                                              [12, 9, 7, 4, 2], [32, 21, 15, 10, 5])
        self.assertEqual(len(dirs), 5)
        self.assertEqual(sum(prompts), 54 - 2 * 20)
        self.assertEqual(sum(contexts) + len(dirs) + 16 + 40, 115)
        self.assertTrue(all(p <= h for p, h in zip(prompts, contexts)))
        self.assertEqual(fit_lengths([1, 1], [4, 8], 3), [2, 1])

    def test_deep_decode_has_unique_valid_locations_and_can_free_normally(self):
        pool = MemoryAllocator(32, 32, torch.float32, 1, 2, 2, device="cpu")
        resident = pool.alloc(5)
        table = torch.full((2, 20), -1, dtype=torch.long)
        table[0, :3], table[1, 1:3] = resident[:3], resident[3:]
        batch = SimpleNamespace(mem_manager=pool, nopad_b_loc=table, nopad_b_seq_len=torch.tensor([4, 3]))
        extend_decode_history(batch, [3, 2], [6, 5])
        self.assertEqual((batch.nopad_total_token_num, batch.nopad_max_len_in_batch), (13, 7))
        self.assertEqual(batch.nopad_b_start_loc.tolist(), [0, 7])
        self.assertEqual(table[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(table[1, 1:3].tolist(), [3, 4])
        active = torch.cat((table[0, :6], table[1, 1:6]))
        self.assertEqual(active.unique().numel(), 11)
        self.assertTrue(torch.all(pool.key_buffer[0][table[0, 3:6]] == 0))
        pool.free(active)
        self.assertEqual(pool.can_use_mem_size, 32)

    def test_validation_checks_actual_outputs_reset_trace_pool_and_quota(self):
        trace = workload()
        cmd = [item for r in (8, 16, 32, 64) for item in ("--adapter-base", f"r{r}")]
        memory = dict(gpu_uuid="gpu", client_quota_bytes=100, sampled_peak_bytes=95, verified_before_model_load=True)
        result = {k: 1 for k in ("benchmark_time", "completed_req_per_s_total", "output_tokens_per_s_total",
                                  "ttft_mean", "ttft_p95", "latency_mean", "latency_p95")}
        result.update(valid=True, total_requests=2, completed_requests=2)
        point = dict(result=result, reset={"status": "confirmed"}, mode=cap.V3, trace_sha256="sha",
                     runtime=dict(total_budget_tokens=32, static_token_equivalent=5, persistent_total_slots=25,
                                  workspace_token_equivalent=2, peak_reserved_bytes=80, peak_allocated_bytes=75))
        rows = [dict(req_id=r["req_id"], success=True, finished=True, scheduled_time=r["req_time"],
                     adapter_index=r["adapter_index"], adapter_dir=d, source_id=r["source_id"],
                     query_tokens=len(r["input_ids"]), expected_output_tokens=r["output_len"],
                     output_tokens=r["output_len"]) for r, d in zip(trace["requests"], ("r8-0", "r64-0"))]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            run = dest / "run"
            run.mkdir()
            guard.save_json(dest / "mps_client.json", dict(gpu_uuid="gpu", quota_bytes=100))
            guard.save_json(run / (cap.POINT_FILE + ".json"), point)
            log = run / (cap.POINT_FILE + "_requests.jsonl")
            log.write_text("".join(json.dumps(r) + "\n" for r in rows))
            args = (dest, cap.V3, 32, trace, "sha", cmd, memory)
            report = cap.validate_replay(*args)
            self.assertEqual((report["output_tokens"], report["dynamic_pool_slots"], report["static_token_equivalents"]),
                             (68, 27, 5))
            self.assertIsNone(report["peak_output_window_10s"])
            result["benchmark_time"] = 21.
            result.update(output_window_metrics({0: 15, 1: 50, 2: 3}, result["benchmark_time"], True))
            guard.save_json(run / (cap.POINT_FILE + ".json"), point)
            self.assertEqual(cap.validate_replay(*args)["peak_output_window_10s"]["output_tokens_per_s"], 5.)
            result["output_windows_10s"][0]["output_tokens"] -= 1
            guard.save_json(run / (cap.POINT_FILE + ".json"), point)
            with self.assertRaisesRegex(ValueError, "Windowed output token counts"):
                cap.validate_replay(*args)
            rows[-1]["output_tokens"] -= 1
            log.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "differs"):
                cap.validate_replay(*args)
            point["reset"]["status"] = "pending"
            guard.save_json(run / (cap.POINT_FILE + ".json"), point)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                cap.validate_replay(*args)


if __name__ == "__main__":
    unittest.main()
