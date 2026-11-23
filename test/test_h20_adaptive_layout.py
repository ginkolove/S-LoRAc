"""Exercise the same staged H20 planner that the replay runner imports."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks import h20_adaptive_layout
from slora.common.prefix_modes import LAYER_ADAPTIVE
from test_online_scheduler import request


class H20HeadPressureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bench = Path(__file__).resolve().parents[1] / "benchmarks"
        with patch.object(sys, "path", [str(bench), *sys.path]):
            from run_h20_minikv_actual import stage_source
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"
            stage_source(src, 16, adaptive=True)
            spec = importlib.util.spec_from_file_location(
                "h20_staged_planner", src / "slora/server/router/adaptive_req_queue.py")
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {spec.name: module,
                            "slora.common.h20_adaptive_layout": h20_adaptive_layout}):
                spec.loader.exec_module(module)
        cls.queue_class = module.AdaptiveReqQueue

    def scenario(self, reconstruct_ms=0.05, capacity=140, head_query=4, max_wait_ms=0):
        queue = self.queue_class(capacity, 1000, 100, 80, LAYER_ADAPTIVE, layer_num=8,
                                 space_plan_config=dict(max_wait_ms=max_wait_ms))
        queue.decode_steps, queue.last_heavy_step = 3, 0
        queue.update_prefix_cache_state(dict(retained_layers={"a": 8}, ready_layers={"a": 8},
            prefix_sizes={"a": 80}, inactive_lora_dirs=set()))
        queue.cost.coef.update(decode=np.array([50., 0, 0, 0]), prefill=np.array([1., 0, 0, 0]),
                               reconstruct=np.array([reconstruct_ms, 0, 0]))
        for name in ("decode", "reconstruct"):
            queue.cost.samples[name].extend([None] * 3)
        queue.cost.error = 0
        running = [request("active", "a", query=4)]
        head = request("head", "b", query=head_query)
        head.arrival_time = 1.0
        queue.waiting_req_list = [head]
        return queue, running, {"a": 8, "b": 8}

    def test_mid_interval_demotes_only_needed_layers_and_admits(self):
        queue, running, ranks = self.scenario()
        choice = queue.choose(running, ranks, now=1.1)
        self.assertFalse(queue.last_record["heavy"])
        self.assertEqual(queue.last_heavy_step, 0)
        self.assertEqual(choice.layers, {"a": 3, "b": 0})
        self.assertEqual(choice.selected, queue.waiting_req_list)
        self.assertTrue(choice.plan.fits)
        self.assertFalse(queue._layout(running, choice.selected, ranks, {"a": 4}, choice.window).fits)
        self.assertTrue(queue.last_record["head_pressure"]["admitted"])

    def test_can_keep_expensive_ready_layers_or_prioritize_aged_head(self):
        for max_wait, admitted in ((0, False), (3000, True)):
            queue, running, ranks = self.scenario(reconstruct_ms=200, max_wait_ms=max_wait)
            choice = queue.choose(running, ranks, now=5)
            record = queue.last_record["head_pressure"]
            self.assertGreater(record["keep_score"], record["release_score"])
            self.assertEqual(bool(choice.selected), admitted)
            self.assertEqual(record["admitted"], admitted)
            if not admitted:
                self.assertEqual(choice.layers, {"a": 8})

    def test_uses_common_shorter_horizon_when_both_actions_fit_now(self):
        queue, running, ranks = self.scenario(capacity=116, head_query=24)
        choice = queue.choose(running, ranks, now=1.1)
        self.assertTrue(choice.plan.fits)
        self.assertEqual(queue.last_record["head_pressure"]["window"], 1)
        self.assertTrue(choice.selected)

    def test_hard_limit_disabled_admission_and_idle_eviction_do_not_trigger(self):
        for reason in ("request_limit", "no_admission", "idle_evictable"):
            queue, running, ranks = self.scenario()
            if reason == "request_limit":
                queue.running_max_req_size = 1
            if reason == "idle_evictable":
                queue.max_total_tokens = 180
                queue.prefix_cache_state["prefix_sizes"]["idle"] = 80
                queue.prefix_cache_state["inactive_lora_dirs"].add("idle")
            queue.choose(running, ranks, allow_admission=reason != "no_admission", now=1.1)
            self.assertNotIn("head_pressure", queue.last_record)

    def test_rank_tiling_applies_only_to_reconstruction(self):
        queue, running, ranks = self.scenario()
        queue.cost.coef["reconstruct"] = np.array([0., 0., 1.])
        def cost(rank):
            return queue.cost.layer_intervals_ms([(0, 8, rank)], 80)
        self.assertEqual(cost(8), cost(16))
        self.assertLess(cost(16), cost(32))
        self.assertLess(cost(32), cost(64))
        slots = queue.plan(running, [], ranks).required_slots
        self.assertEqual(queue.plan(running, [], {"a": 16}).required_slots - slots, 32)
