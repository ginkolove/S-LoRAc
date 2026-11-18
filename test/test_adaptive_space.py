"""Elastic storage, bounded online planning, and opt-in real GPU regressions."""

import asyncio
import gc
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from slora.common.mem_allocator import MemoryAllocator
from slora.common.lowra_adaptive import AdaptivePrefixKVCache, AdaptiveLayerRuntime
from slora.common.prefix_modes import LAYER_ADAPTIVE, LAYER_SERIAL
from slora.common.space_plan_config import SpacePlanConfig
from slora.server.router.adaptive_req_queue import AdaptiveReqQueue, MeasuredCostModel
from test_online_scheduler import request, make_router, drain


class ElasticStorageTests(unittest.TestCase):
    def pool(self):
        return MemoryAllocator(80, 80, torch.float32, 1, 2, 5,
                               dynamic_layer_workspace=True, device="cpu")

    def test_resize_preserves_cells_and_private_state_and_rounding(self):
        memory = self.pool()
        private = memory.alloc(8)
        for layer in range(5):
            memory.key_buffer[layer][private] = layer + 100
            memory.value_buffer[layer][private] = layer + 200
        cache = AdaptivePrefixKVCache(7, memory)
        cache.acquire_many(["a", "a", "b"])
        cache.apply_plan({"a": 2, "b": 4})
        a = cache.entries["a"]
        original = a.prefix_indices.clone()
        memory.prefix_workspace_k[original] = 42
        a.ready_layers = 2
        cache.apply_plan({"a": 5, "b": 1})
        self.assertEqual(a.ready_layers, 2)
        torch.testing.assert_close(a.prefix_indices[:2], original)
        self.assertTrue(torch.all(memory.prefix_workspace_k[a.prefix_indices[:2]] == 42))
        self.assertEqual(a.allocated_slots, 7)
        self.assertEqual(cache.entries["b"].allocated_slots, 2)
        cache.apply_plan({"a": 1, "b": 0})
        self.assertEqual(a.ready_layers, 1)
        self.assertEqual(a.allocated_slots, 2)
        for layer in range(5):
            self.assertTrue(torch.all(memory.key_buffer[layer][private] == layer + 100))
            self.assertTrue(torch.all(memory.value_buffer[layer][private] == layer + 200))
        cache.release_many(["a", "a", "b"])
        self.assertEqual(set(cache.entries), {"a"})
        cache.evict_many(["a"])
        memory.free(private)
        self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def test_capacity_checks_are_atomic_and_workspace_blocks_resize(self):
        memory = self.pool()
        cache = AdaptivePrefixKVCache(100, memory)
        cache.acquire_many(["a"])
        with self.assertRaises(RuntimeError):
            cache.apply_plan({"a": 5})
        self.assertEqual(memory.can_use_mem_size, 80)
        memory.alloc_layer_workspace(2)
        with self.assertRaises(RuntimeError):
            cache.apply_plan({"a": 1})
        memory.free_layer_workspace()
        with self.assertRaises(ValueError):
            cache.apply_plan({"a": 6})

    def test_mixed_layers_ready_flags_unique_reconstruction_and_reuse(self):
        memory = self.pool()
        cache = AdaptivePrefixKVCache(7, memory)
        static = SimpleNamespace(layer_num=5, prefix_len=7, base_k=[torch.zeros(1)])
        runtime = AdaptiveLayerRuntime(static, memory, cache)
        dirs = ["a", "a", "b", "c"]
        cache.acquire_many(dirs)
        calls = []

        def metadata(static, directories, adapter, locations):
            return SimpleNamespace(prefix_locs=locations, dirs=directories)

        def reconstruct(static, adapter, meta, layer, keys, values):
            for d, indices in zip(meta.dirs, meta.prefix_locs):
                calls.append((d, layer))
                keys[indices] = ord(d) * 10 + layer
                values[indices] = -ord(d) * 10 - layer

        with patch("slora.common.lowra_retained.prepare_reconstruction_metadata", side_effect=metadata), \
                patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer", side_effect=reconstruct):
            cache.apply_plan({"a": 5, "b": 2, "c": 0})
            for step in range(4):
                if step == 2:
                    cache.apply_plan({"a": 1, "b": 5, "c": 3})
                frame = runtime.prepare(dirs, object())
                for layer in range(5):
                    frame.before_layer(layer)
                    keys, values = frame.buffers(layer)
                    for d, indices in zip(dirs, frame.prefix_locations(layer)):
                        self.assertTrue(torch.all(keys[indices] == ord(d) * 10 + layer))
                        self.assertTrue(torch.all(values[indices] == -ord(d) * 10 - layer))
                frame.finish()
                self.assertEqual(memory.workspace_token_equiv, 0)
        self.assertEqual(calls.count(("a", 0)), 1)
        self.assertEqual(calls.count(("b", 0)), 1)
        self.assertEqual(calls.count(("b", 4)), 3)
        self.assertEqual(calls.count(("c", 4)), 4)
        cache.release_many(dirs)
        cache.evict_many(list(cache.entries))
        self.assertEqual(memory.can_use_mem_size, 80)


class PlannerTests(unittest.TestCase):
    def queue(self, capacity=200, **config):
        return AdaptiveReqQueue(capacity, 1000, 100, 20, LAYER_ADAPTIVE,
                                layer_num=4, space_plan_config=config)

    def test_reference_length_never_used_and_ten_step_reserve(self):
        class UnknownOutput:
            request_id = "x"
            adapter_dir = "a"
            input_len = 5
            output_ids = []
            aborted = False
            arrival_time = 0

            @property
            def max_output_len(self):
                raise AssertionError("future output length accessed")

        queue = self.queue()
        queue.waiting_req_list = [UnknownOutput()]
        result = queue.choose([], {"a": 1})
        self.assertEqual(result.window, 10)
        self.assertEqual(result.plan.required_slots, 5 + 1 + 9 + 4 + 5)
        self.assertTrue(result.plan.fits)
        self.assertEqual(result.layers, {"a": 0})

    def test_short_window_fallback_does_not_reject_serviceable_request(self):
        queue = self.queue(15)
        queue.waiting_req_list = [request("x", query=5)]
        choice = queue.choose([], {"a": 1})
        self.assertEqual(choice.window, 1)
        self.assertTrue(choice.plan.fits)
        self.assertEqual(len(choice.selected), 1)

    def test_pressure_demotes_active_prefix_before_preemption(self):
        queue = self.queue(30)
        queue.update_prefix_cache_state(dict(retained_layers={"a": 4}, ready_layers={"a": 4},
            prefix_sizes={"a": 20}, cached_lora_dirs={"a"}, inactive_lora_dirs=set()))
        running = request("x", query=20)
        self.assertFalse(queue.plan([running], [], {"a": 1}).fits)
        choice = queue.choose([running], {"a": 1}, allow_admission=False)
        self.assertTrue(choice.plan.fits)
        self.assertEqual(choice.layers, {"a": 0})
        self.assertEqual(choice.window, 1)

    def test_idle_cache_lru_is_only_reclaimed_when_needed(self):
        queue = self.queue(35)
        queue.update_prefix_cache_state(dict(prefix_sizes={"old": 20, "recent": 10},
            inactive_lora_dirs={"old", "recent"}, last_used={"old": 1, "recent": 2}))
        plan = queue._layout([], [request("x")], {"a": 1}, {"a": 0}, 10)
        self.assertTrue(plan.fits)
        self.assertEqual(plan.evictions, ["old"])

    def test_same_lora_multiplicity_does_not_multiply_reconstruction(self):
        queue = self.queue()
        queue.cost.coef["decode"] = np.array([10., 0, 0, 0])
        queue.cost.coef["reconstruct"] = np.array([2., 0, 0])
        first = queue._evaluate([request("1")], [], {"a": 16}, {"a": 0}, 1)
        second = queue._evaluate([request("1"), request("2")], [], {"a": 16}, {"a": 0}, 1)
        self.assertEqual(first.step_ms, second.step_ms)

    def test_heavy_planning_interval_and_immediate_safety(self):
        queue = self.queue(max_wait_ms=0)
        req = request("x")
        queue.choose([req], {"a": 1}, False)
        self.assertTrue(queue.last_record["heavy"])
        queue.decode_steps = 1
        queue.choose([req], {"a": 1}, False)
        self.assertFalse(queue.last_record["heavy"])
        queue.decode_steps = 10
        queue.choose([req], {"a": 1}, False)
        self.assertTrue(queue.last_record["heavy"])

    def test_profile_measurements_fit_nonnegative_finite_costs(self):
        model = MeasuredCostModel()
        rows = [dict(seq=i, batch=i, private_tokens=i * 100, rank_sum=i * 16,
                     prefill=False, prefix_len=1000, core_ms=1 + i * 2,
                     reconstruction=[dict(unique=i, rank_sum=16 * i, ms=0.2 * i)])
                for i in range(1, 20)]
        model.update(rows)
        self.assertTrue(model.calibrated)
        self.assertTrue(all(np.all(c >= 0) for c in model.coef.values()))
        self.assertAlmostEqual(model.core_ms(10, 1000, 160, 1000), 21, delta=2)
        self.assertGreater(model.reconstruction_ms([64], 1000), model.reconstruction_ms([8], 1000))
        count = len(model.samples["decode"])
        model.update(rows)
        self.assertEqual(len(model.samples["decode"]), count)

    def test_interval_cost_matches_layer_by_layer_with_mixed_ranks(self):
        model = MeasuredCostModel()
        model.coef["reconstruct"] = np.array([0.2, 0.5, 0.8])
        intervals = [(0, 40, 8), (4, 24, 64), (16, 40, 16), (40, 40, 32)]
        expected = sum(model.reconstruction_ms(
            [rank for first, end, rank in intervals if first <= layer < end], 10000)
            for layer in range(40))
        self.assertAlmostEqual(model.layer_intervals_ms(intervals, 10000), expected)

    def test_measured_search_retains_costly_prefix_without_topk_rule(self):
        # Two active adapters cannot both fully reside. Higher measured
        # reconstruction cost, rather than request multiplicity, wins here.
        queue = self.queue(120, window=2, budget_ms=100, max_wait_ms=0)
        queue.cost.coef["decode"] = np.array([10., 0, 0, 0])
        queue.cost.coef["reconstruct"] = np.array([0., 0., 1000.])
        queue.cost.error = 0.
        for name in ("decode", "reconstruct"):
            queue.cost.samples[name].extend([None] * 3)
        queue.update_prefix_cache_state(dict(retained_layers={"a": 0, "b": 0},
            ready_layers={"a": 0, "b": 0}, prefix_sizes={"a": 0, "b": 0}))
        # 72 weights + 10 private + 2 growth + 40 fully retained = 124 > 120.
        running = [request("1", "a", query=5), request("2", "b", query=5)]
        result = queue.choose(running, {"a": 2, "b": 16}, allow_admission=False)
        self.assertTrue(result.plan.fits)
        self.assertGreater(result.layers["b"], result.layers["a"])
        self.assertGreater(result.layers["b"], 0)


class AdaptiveLauncherTests(unittest.TestCase):
    def test_three_model_commands_opt_in_without_changing_trace(self):
        root = Path(__file__).resolve().parents[1]
        for size, prefix, tokens, gpu in (("7b", "12000", "125000", "0"),
                                         ("13b", "10000", "65000", "0"),
                                         ("30b", "2000", "8800", "1")):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as out:
                output = subprocess.check_output([
                    "bash", str(root / "benchmarks/run_lmsys_preemption.sh"), out,
                    "--adaptive-only", "--model-size", size, "--prefix-len", prefix,
                    "--gpu", gpu, "--request-rate", "2", "--duration", "120",
                    "--space-plan-interval", "10", "--space-plan-window", "10", "--dry-run"],
                    cwd=root, text=True)
                commands = [shlex.split(line) for line in output.splitlines() if line.startswith("/")]
                trace = next(c for c in commands if "benchmarks/prepare_lmsys_maf_trace.py" in c)
                runs = [c for c in commands if "benchmarks/lowra_saturation_exp.py" in c]
                value = lambda command, flag: command[command.index(flag) + 1]
                self.assertEqual(len(runs), 1)
                run = runs[0]
                for flag, expected in (("--mode", LAYER_ADAPTIVE), ("--num-token", tokens),
                    ("--prefix-len", prefix), ("--request-rates", "2"), ("--duration", "120"),
                    ("--schedule-interval", "1"), ("--space-plan-interval", "10"),
                    ("--space-plan-window", "10"), ("--reconstruction-kernel", "bounded")):
                    self.assertEqual(value(run, flag), expected)
                self.assertEqual(value(trace, "--maf-minute-end"), "1440")
                self.assertEqual(value(trace, "--data-seed"), "42")
                self.assertEqual(value(trace, "--output"), value(run, "--workload-trace"))
                self.assertIn("CUDA_VISIBLE_DEVICES=" + gpu, output)
                self.assertFalse(list(Path(out).iterdir()))


@unittest.skipUnless(os.environ.get("RUN_ADAPTIVE_CUDA_TESTS") == "1" and torch.cuda.is_available(),
                     "opt-in GPU correctness test")
class AdaptiveGPUTests(unittest.TestCase):
    def args(self, mode):
        return SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
            adapter_base="dummy-lora-7b-rank-16", mode=mode, prefix_len=31, query_len=5,
            output_len=20, batch_size=2, unique_loras=2, num_adapters=2, rank=16,
            nonzero_prefix=True, retain_layers=0, reconstruction_kernel="bounded",
            reconstruction_workers=1296)

    def test_real_logits_match_serial_through_promotions_and_demotions(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.server.router.model_infer.post_process import sample
        reference = None
        for mode in (LAYER_SERIAL, LAYER_ADAPTIVE):
            server, _, requests, _ = build_server(self.args(mode))
            server.lowra_runtime.profiler = None
            collected = []

            def capture(logits, batch):
                collected.append(logits.detach().clone())
                return sample(logits, batch)

            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture):
                server.exposed_add_batch("x", requests, "fp16")
                server.exposed_prefill_batch("x")
                dirs = [r["adapter_dir"] for r in requests]
                for step, counts in enumerate([(3, 1), (3, 1), (1, 3), (0, 2), (3, 3), (0, 0)]):
                    if mode == LAYER_ADAPTIVE:
                        server.exposed_apply_space_plan(dict(zip(dirs, counts)), [])
                    server.exposed_decode_batch("x")
                server.exposed_remove_batch("x")
            if reference is None:
                reference = collected
            else:
                for actual, expected in zip(collected, reference):
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
                self.assertGreater(server.lowra_prefix_cache.promotions, 0)
                self.assertGreater(server.lowra_prefix_cache.demotions, 0)
                server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
            server.exposed_offload_adapters([])
            self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)
            del server
            gc.collect()
            torch.cuda.empty_cache()

    def test_online_pressure_merge_filter_drain(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.server.router.model_infer.model_rpc import ModelRpcClient
        server, _, _, _ = build_server(self.args(LAYER_ADAPTIVE))
        server.lowra_runtime.profiler = None
        dirs = list(server.adapter_id)
        server.exposed_offload_adapters([])
        queue = AdaptiveReqQueue(server.model.mem_manager.tot_size, 1000, 30, 31,
            LAYER_ADAPTIVE, layer_num=3, space_plan_config=dict(interval=2, window=2, budget_ms=20))
        reqs = [request(str(i), dirs[i % 2], query=3 + i, output=8 + i) for i in range(5)]
        queue.waiting_req_list = reqs[:]
        router = make_router(queue, ModelRpcClient(server, 1), {d: 16 for d in dirs})
        router.input_params.enable_lowra = True
        router.input_params.lowra_prefix_mode = LAYER_ADAPTIVE
        asyncio.run(drain(router))
        self.assertEqual(router.online_rejections, 0)
        self.assertTrue(all(len(r.output_ids) == r.max_output_len for r in reqs))
        self.assertGreater(queue.decode_steps, 0)
        self.assertGreater(queue.cost.last_seq, 0)
        server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
        self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)


if __name__ == "__main__":
    unittest.main()
