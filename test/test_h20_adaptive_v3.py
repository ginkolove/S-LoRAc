"""Regressions for isolated V3 payback, source reuse and paired H20 replay."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from benchmarks.h20_adaptive_v3 import AdaptiveV3ReqQueue, POLICY
from benchmarks.h20_adaptive_v3_runtime import AdaptiveV3LayerRuntime, AdaptiveV3PrefixKVCache
from slora.common.lowra_layer_runtime import prepare_reconstruction_metadata
from slora.common.mem_allocator import MemoryAllocator
from slora.common.prefix_modes import LAYER_ADAPTIVE
from test_h20_adaptive_v2 import ElasticRPC, LOCAL_TRACES, retained
from test_online_scheduler import make_router, request


ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(ROOT / "benchmarks"), *sys.path]):
    from run_h20_adaptive_v3 import paired_summary, stage_v3_source


def calibrated_queue(capacity=1000, window=8, prefix=80, layers=8):
    queue = AdaptiveV3ReqQueue(capacity, 1000, 100, prefix, LAYER_ADAPTIVE, layer_num=layers,
                               space_plan_config=dict(window=window))
    queue.cost.coef["reconstruct"] = np.array([1., 0., 0.])
    for category in ("decode", "reconstruct"):
        queue.cost.samples[category].extend([None] * 3)
    return queue


class PaybackTests(unittest.TestCase):
    def test_short_epoch_defers_and_later_pays_back_without_new_arrivals(self):
        queue = calibrated_queue()
        state = dict(v3_apply_samples=[dict(seq=1, changed_dirs=2, host_ms=24., device_ms=20.)])
        queue.update_prefix_cache_state(state)
        queue.update_prefix_cache_state(state)
        self.assertEqual(list(queue.apply_unit_samples), [12.])
        running = [request("one", query=4)]
        self.assertEqual(queue.choose(running, {"a": 8}).layers, {"a": 0})
        self.assertEqual(queue.last_record["reuse_steps"], 2)
        queue.decode_steps = 1
        queue.choose(running, {"a": 8})
        self.assertFalse(queue.last_record["heavy"])
        queue.decode_steps = 8
        self.assertEqual(queue.choose(running, {"a": 8}).layers, {"a": 8})
        self.assertIn("reuse_recheck", queue.last_record["trigger_reasons"])
        self.assertEqual(queue.last_record["predicted_net_saving_ms"], 44.)

    def test_private_growth_limits_reuse_of_a_promotion(self):
        queue = calibrated_queue(116)
        # Eight layers fit for exactly one decode; six leave room for a paid reuse.
        choice = queue.choose([request("one", query=4)], {"a": 8})
        self.assertEqual(choice.layers, {"a": 6})
        self.assertEqual(queue.last_record["reuse_steps"], 2)
        self.assertEqual(queue.last_record["predicted_net_saving_ms"], 6.)

    def test_first_prefill_only_materializes_selected_adapters(self):
        queue = calibrated_queue()
        queue.cost.coef["reconstruct"] = np.array([0., 125., 0.])
        running, selected = [request("old", "a", query=4)], [request("new", "b", query=4)]
        ranks = {"a": 8, "b": 8}
        for layers, expected in (({"a": 0, "b": 0}, 24.), ({"a": 8, "b": 0}, 24.),
                                 ({"a": 8, "b": 8}, 16.)):
            choice = queue._evaluate(running, selected, ranks, layers)
            self.assertEqual(queue._initial_reconstruction(choice, {"b"}, ranks), expected)

    def test_forced_shrink_cold_head_and_hard_adapter_limit(self):
        queue = calibrated_queue(140)
        queue.apply_ms_per_dir = 1e6
        retained(queue, {"a": 8})
        queue.waiting_req_list = [request("cold", "b", query=4), request("hot", "a", query=4)]
        running = [request("active", query=4)]
        ranks = {"a": 8, "b": 8}
        choice = queue.choose(running, ranks)
        self.assertEqual(choice.selected, queue.waiting_req_list)
        self.assertTrue(choice.plan.fits)
        self.assertLess(choice.layers["a"], 8)
        self.assertIn("memory_pressure", queue.last_record["trigger_reasons"])
        queue.max_active_loras = 1
        self.assertFalse(queue.choose(running, ranks).selected)
        queue.max_total_tokens = 3
        self.assertFalse(queue.choose(running, ranks).plan.fits)

    def test_no_lookahead_and_nonintegral_slot_boundary(self):
        class HiddenOutput:
            adapter_dir, input_len, aborted = "a", 3, False
            output_ids = ()

            def __init__(self, i):
                self.request_id = str(i)

            @property
            def max_output_len(self):
                raise AssertionError("used hidden output length")

        queue = calibrated_queue(40, prefix=7, layers=5)
        queue.waiting_req_list = [HiddenOutput(i) for i in range(4)]
        for capacity in range(34, 59):
            queue.max_total_tokens = capacity
            choice = queue.choose([], {"a": 8})
            limit = max(0, min(4, (capacity - 32 - 2) // 4))
            self.assertEqual(choice.selected, queue.waiting_req_list[:limit])
            self.assertTrue(choice.plan.fits)

    def test_distinct_group_history_resets_on_idle(self):
        queue = calibrated_queue()
        self.assertEqual(queue._group_horizon({"a"}, True), 2)
        queue.decode_steps = 6
        self.assertEqual(queue._group_horizon({"a", "b"}, False), 3)
        queue.decode_steps = 50
        self.assertEqual(queue._group_horizon({"a", "b"}, False), 8)
        self.assertEqual(queue._group_horizon({"a", "b"}, True), 2)


class RuntimeTests(unittest.TestCase):
    def test_pending_device_timing_is_consumed_once_after_completion(self):
        pool = MemoryAllocator(10, 10, torch.float32, 1, 1, 4, dynamic_layer_workspace=True, device="cpu")
        cache = AdaptiveV3PrefixKVCache(5, pool)
        begin, end = Mock(), Mock()
        end.query.side_effect = (False, True)
        begin.elapsed_time.return_value = 12.
        cache._pending_applies.append((1, 3, 5., begin, end))
        queue = calibrated_queue()
        queue.update_prefix_cache_state(cache.state_dict())
        self.assertFalse(queue.apply_unit_samples)
        begin.elapsed_time.assert_not_called()
        queue.update_prefix_cache_state(cache.state_dict())
        queue.update_prefix_cache_state(cache.state_dict())
        self.assertEqual(list(queue.apply_unit_samples), [4.])
        begin.elapsed_time.assert_called_once_with(end)
        end.synchronize.assert_not_called()

    def test_shared_sources_use_current_destinations_and_invalidate_on_repack(self):
        pool = MemoryAllocator(100, 100, torch.float32, 1, 2, 4, dynamic_layer_workspace=True, device="cpu")
        cache = AdaptiveV3PrefixKVCache(5, pool)
        ranges = {"a": (0, 2), "b": (2, 4)}
        static = SimpleNamespace(layer_num=4, prefix_len=5, base_k=[torch.zeros(1)],
                                 get_lora_rank_range_by_dir=ranges.__getitem__)
        adapter = SimpleNamespace(idx_map={"a": 0, "b": 1}, a_loc=torch.arange(16).roll(3),
                                  a_start=torch.tensor([0, 8]), a_len=torch.tensor([8, 8]),
                                  a_scaling=torch.tensor([.25, .5]))
        runtime = AdaptiveV3LayerRuntime(static, pool, cache, reconstruction_kernel="bounded",
                                         reconstruction_workers=1)
        dirs = ["a", "b", "a"]
        cache.acquire_many(dirs)
        cache.apply_plan({"a": 2, "b": 0})
        cache.entries["a"].ready_layers = 2
        for i, expected_builds in enumerate((2, 2, 4, 6, 8, 9)):
            if i == 1:
                private = pool.alloc(4)  # Move the transient destinations.
            elif i == 2:
                adapter.a_loc.copy_(adapter.a_loc.roll(1))
            elif i == 3:
                adapter.a_scaling = adapter.a_scaling.clone()
            elif i == 4:
                adapter.idx_map = {"a": 1, "b": 0}
            elif i == 5:
                dirs = ["b", "a"]
            frame = runtime.prepare(dirs, adapter)
            try:
                for layer, (group, actual) in enumerate(zip(frame.missing, frame.metadata)):
                    ids = [frame.unique.index(d) for d in group]
                    destination = frame.tables[layer][ids]
                    expected = prepare_reconstruction_metadata(static, group, adapter, destination)
                    for name, tensor in vars(expected).items():
                        self.assertTrue(torch.equal(tensor, getattr(actual, name)), (layer, name))
            finally:
                frame.finish()
            self.assertEqual(cache.source_metadata_builds, expected_builds)
            self.assertEqual(cache.source_metadata_reuses, 4 * (i + 1) - expected_builds)
            self.assertLessEqual(len(runtime._metadata_cache), static.layer_num)
        pool.free(private)
        sample, = cache.state_dict()["v3_apply_samples"]
        self.assertEqual((sample["seq"], sample["changed_dirs"]), (1, 1))
        self.assertGreater(sample["host_ms"], 0)
        cache.reset_access_stats()
        self.assertEqual(cache.access_stats()["source_metadata_builds"], 0)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuous_batch_drains_with_pressure(self):
        queue = calibrated_queue(95, prefix=20, layers=4)
        ranks = {"a": 1, "b": 1, "c": 2}
        rpc = ElasticRPC(95, ranks)
        router = make_router(queue, rpc, ranks)
        arrivals = {0: [request("a1", "a", query=4, output=14), request("b1", "b", query=4, output=8)],
                    3: [request("c1", "c", query=4, output=15)], 7: [request("a2", "a", query=4, output=12)]}
        admitted = []
        for step in range(100):
            for req in arrivals.get(step, []):
                queue.append(req)
            await router._step_online()
            admitted.extend(queue.last_record["selected"])
            if step >= 7 and not router.running_batch and not queue.waiting_req_list:
                break
        self.assertEqual(admitted, ["a1", "b1", "c1", "a2"])
        self.assertFalse(rpc.batches)
        self.assertFalse(queue.waiting_req_list)
        self.assertIsNone(router.running_batch)
        self.assertEqual(router.online_rejections, 0)


class ReplayTests(unittest.TestCase):
    @unittest.skipUnless(LOCAL_TRACES, "requires the two frozen H20 traces")
    def test_staged_serving_command_and_baseline_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            stage_v3_source(source)
            self.assertEqual((source / "slora/server/router/h20_adaptive_v2.py").read_bytes(),
                             (ROOT / "benchmarks/h20_adaptive_v2.py").read_bytes())
            plan = subprocess.check_output([sys.executable, "benchmarks/run_h20_adaptive_v3.py", str(root / "out"),
                "--gpu", "1", "--point", "13b:8000", "13b:10000", "--compare-v2", "v3-first", "--dry-run"],
                cwd=ROOT, text=True)
            commands = [shlex.split(line) for line in plan.splitlines() if line.startswith("/")]
            self.assertEqual(len(commands), 4)
            for v3_cmd, v2_cmd in zip(commands[::2], commands[1::2]):
                for flag in ("--workload-trace", "--model-dir", "--num-token", "--seeds", "--port"):
                    self.assertEqual(v3_cmd[v3_cmd.index(flag) + 1], v2_cmd[v2_cmd.index(flag) + 1])
                self.assertEqual(v3_cmd[v3_cmd.index("--space-plan-window") + 1], "8")
                self.assertEqual(v2_cmd[v2_cmd.index("--space-plan-window") + 1], "1")
                self.assertEqual(v3_cmd[v3_cmd.index("--mode") + 1], POLICY)
            self.assertFalse((root / "out").exists())
            code = '''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from benchmarks.lowra_saturation_exp import parse_args, server_command
from slora.server.router.manager import get_scheduler
from slora.server.router.h20_adaptive_v3 import AdaptiveV3ReqQueue
from slora.server.router.h20_adaptive_v2 import AdaptiveV2ReqQueue
from slora.server.router.adaptive_req_queue import AdaptiveReqQueue
from slora.server.router.online_req_queue import OnlineReqQueue
sys.argv = ["probe", *json.loads(sys.argv[1])]
args = parse_args()
assert args.rank_loop == "actual" and args.block_r == 16 and args.num_token == 65000
server = server_command(args, Path("probe.jsonl"))
assert server[server.index("--lowra-prefix-mode")+1] == "minikv-adaptive-v3"
assert server[server.index("--space-plan-window")+1] == "8"
p = SimpleNamespace(scheduler="online", enable_lowra=True, space_plan_config={}, max_total_token_num=1000,
    batch_max_tokens=1000, running_max_req_size=100, lowra_shared_prefix_len=80, lowra_layer_num=8,
    lowra_retain_layers=0, lowra_retain_top_k=None, lowra_pin_hot_adapters=0,
    max_active_loras=None, lowra_workspace_loras=None)
for mode, cls in (("minikv-adaptive-v3", AdaptiveV3ReqQueue), ("minikv-adaptive-v2", AdaptiveV2ReqQueue),
                  ("minikv-adaptive", AdaptiveReqQueue), ("minikv-layer_serial", OnlineReqQueue)):
    p.lowra_prefix_mode = mode
    assert type(get_scheduler(p, [])) is cls
'''
            uvloop = ROOT / "benchmarks/results/.h20/uvloop-0.21.0-py39"
            probe = subprocess.run([commands[0][0], "-c", code, json.dumps(commands[0][3:])],
                                   cwd=source, env=dict(os.environ, HF_HUB_OFFLINE="1",
                                   PYTHONPATH=os.pathsep.join((str(uvloop), str(source)))),
                                   capture_output=True, text=True)
            self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_summary_checks_matched_requests_and_tpot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(req_id="1", scheduled_time=0, source_id="s", adapter_dir="a", query_tokens=4,
                       output_tokens=3, expected_output_tokens=3, success=True, end_time=3., first_token_time=1.)
            result = dict(valid=True, total_requests=1, completed_requests=1, benchmark_time=3.,
                          ttft_mean=1., ttft_p95=1., latency_mean=3., latency_p95=3.)
            for mode, throughput in (("minikv-adaptive-v2", 1.), (POLICY, 1.1)):
                directory = root / mode
                directory.mkdir()
                point = dict(mode=mode, result=dict(result, completed_req_per_s_total=throughput),
                             reset=dict(status="confirmed"), trace_sha256="same")
                (directory / "rate3_seed42.json").write_text(json.dumps(point))
                (directory / "rate3_seed42_requests.jsonl").write_text(json.dumps(row))
            summary = paired_summary(root)
            self.assertAlmostEqual(summary["v3_throughput_change_pct"], 10.)
            self.assertEqual(summary["methods"][POLICY]["mean_request_tpot_ms"], 1000.)
            (directory / "rate3_seed42_requests.jsonl").write_text(json.dumps(dict(row, adapter_dir="other")))
            with self.assertRaisesRegex(ValueError, "Different traces"):
                paired_summary(root)
