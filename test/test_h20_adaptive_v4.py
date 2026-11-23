"""V4 admission, grouped slot accounting and real-time throughput feedback."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.h20_adaptive_v4 import AdaptiveV4ReqQueue
from slora.common.prefix_modes import LAYER_ADAPTIVE


class Request:
    aborted = False

    def __init__(self, name, adapter=None, length=1):
        self.request_id = name
        self.adapter_dir = adapter or name
        self.input_len = length
        self.output_ids = []

    @property
    def max_output_len(self):
        raise AssertionError("V4 consulted a future output length")


RANKS = {"a": 8, "b": 8, "h": 32, None: 0}


def queue(capacity, **limits):
    return AdaptiveV4ReqQueue(capacity, 1000, 100, 80, LAYER_ADAPTIVE, layer_num=8,
                              space_plan_config={"feedback_seconds": 1., "hysteresis": .05}, **limits)


def apply(q, choice, inactive=()):
    q.update_prefix_cache_state(dict(retained_layers=choice.layers.copy(),
                                     prefix_sizes={d: (k * q.prefix_len + q.layer_num - 1) // q.layer_num
                                                   for d, k in choice.layers.items()},
                                     inactive_lora_dirs=inactive))


def begin(capacity=358, **limits):
    q = queue(capacity, **limits)
    a, b, h = Request("a"), Request("b"), Request("h")
    q.waiting_req_list = [a, b]
    cold = q.choose([], RANKS, now=0.)
    assert cold.layers == {"a": 8, "b": 8}
    apply(q, cold)
    q.waiting_req_list = [h]
    q.observe_output(2)  # Actual first tokens seed the optional two-slot headroom.
    q.observe_output(98)
    return q, [a, b], h


class RankFeedbackTests(unittest.TestCase):
    def test_full_first_then_minimum_equal_rank_demotion_and_fcfs(self):
        q, running, h = begin()
        # No measured old window yet. The old batch can continue serving.
        self.assertFalse(q.choose(running, RANKS, now=.2).selected)
        chosen = q.choose(running, RANKS, now=1.)
        self.assertEqual(chosen.selected, [h])
        self.assertEqual(chosen.layers, {"a": 3, "b": 3, "h": 8})
        self.assertEqual(chosen.plan.required_slots, 356)
        self.assertEqual(q.trial.baseline, 100.)
        self.assertEqual(q.last_record["trigger_reasons"], ["admission_pressure"])

    def test_only_move_on_to_next_rank_when_first_reaches_zero(self):
        q, running, h = begin(285)
        choice = q.choose(running, RANKS, now=1.)
        self.assertEqual(choice.selected, [h])
        self.assertEqual(choice.layers, {"a": 0, "b": 0, "h": 5})

    def test_partial_layers_use_actual_rounded_slots_including_shared_scratch(self):
        for capacity, depth, slots in [(366, 3, 364), (444, 6, 426)]:
            with self.subTest(capacity=capacity):
                q = queue(capacity)
                q.prefix_len = 83  # An uneven split across eight layers.
                running = [Request("a"), Request("b")]
                q.waiting_req_list = running
                apply(q, q.choose([], RANKS, now=0.))
                q.waiting_req_list = [Request("h")]
                q.observe_output(2)
                choice = q.choose(running, RANKS, now=1.)
                self.assertEqual(choice.layers, {"a": depth, "b": depth, "h": 8})
                self.assertEqual(choice.plan.required_slots, slots)

    def test_idle_eviction_admits_without_reducing_any_active_prefix(self):
        q = queue(235)
        q.update_prefix_cache_state(dict(retained_layers={"a": 8, "h": 8},
                                         prefix_sizes={"a": 80, "h": 80}, inactive_lora_dirs=["h"]))
        q.waiting_req_list = [Request("b")]
        choice = q.choose([Request("a")], RANKS, now=0.)
        self.assertEqual(choice.selected, q.waiting_req_list)
        self.assertEqual(choice.plan.evictions, ["h"])
        self.assertEqual(choice.layers, {"a": 8, "b": 8})
        self.assertIsNone(q.trial)

    def test_exact_physical_fit_admits_full_even_with_large_promotion_reserve(self):
        q, running, h = begin(436)  # 3 full prefixes + 192 weight slots + 4 private slots.
        q.observe_preemption(50)
        choice = q.choose(running, RANKS, now=.1)
        self.assertEqual(choice.selected, [h])
        self.assertEqual(choice.plan.required_slots, q.max_total_tokens)
        self.assertEqual(choice.layers, {"a": 8, "b": 8, "h": 8})
        self.assertIsNone(q.trial)

    def test_hard_limits_do_not_cause_shrink_or_skip_queue_head(self):
        q, running, h = begin(max_active_loras=2)
        q.waiting_req_list.append(Request("reuse", "a"))
        choice = q.choose(running, RANKS, now=1.)
        self.assertEqual(choice.selected, [])
        self.assertEqual(choice.layers, {"a": 8, "b": 8})
        self.assertIsNone(q.trial)

    def test_two_bad_windows_defer_rollback_until_accepted_work_fits(self):
        q, running, h = begin()
        changed = q.choose(running, RANKS, now=1.)
        apply(q, changed)
        q.waiting_req_list = []
        live = running + [h]
        q.observe_output(18)
        q.choose(live, RANKS, now=2.)
        self.assertEqual(q.trial.low_windows, 1)
        q.observe_output(19)
        waiting = Request("queued", "a")
        q.waiting_req_list = [waiting]
        deferred = q.choose(live, RANKS, now=3.)
        self.assertEqual(deferred.layers, changed.layers)
        self.assertFalse(deferred.selected)
        self.assertEqual(q.last_record["output_tokens_s"], 19.)
        self.assertIn("rollback_deferred", q.last_record["trigger_reasons"])
        apply(q, changed, inactive=["h"])
        restored = q.choose(running, RANKS, now=3.1)
        self.assertEqual(restored.layers, {"a": 8, "b": 8})
        self.assertIn("rollback", q.last_record["trigger_reasons"])
        self.assertIsNone(q.rollback)
        apply(q, restored)
        self.assertEqual(q.choose(running, RANKS, now=3.2).selected, [waiting])

    def test_small_drop_accepts_then_completion_opens_promotion_trial(self):
        q, running, h = begin()
        changed = q.choose(running, RANKS, now=1.)
        apply(q, changed)
        q.waiting_req_list = []
        q.observe_output(97)
        q.choose(running + [h], RANKS, now=2.)
        self.assertIn("trial_accepted", q.last_record["trigger_reasons"])
        apply(q, changed, inactive=["h"])
        promoted = q.choose(running, RANKS, now=2.1)
        self.assertEqual(promoted.layers, {"a": 8, "b": 8})
        self.assertIn("capacity_released", q.last_record["trigger_reasons"])
        apply(q, promoted)
        for t in (3.1, 4.1, 4.2):
            q.observe_output(3)
            after = q.choose(running, RANKS, now=t)
        self.assertEqual(after.layers, {"a": 3, "b": 3})

    def test_running_growth_forces_feasible_layout_without_arrivals_or_rate(self):
        q, running, _ = begin()
        q.waiting_req_list = []
        running[0].output_ids = [0] * 162
        choice = q.choose(running, RANKS, now=.1)
        self.assertTrue(choice.plan.fits)
        self.assertLess(choice.layers["a"], 8)
        self.assertEqual(choice.layers["a"], choice.layers["b"])
        self.assertIn("running_pressure", q.last_record["trigger_reasons"])
        self.assertIsNone(q.trial)

    def test_structural_serial_fallback_and_idle_head_resume(self):
        q = queue(113)  # 80 + 4*8 + 2 needs 114; the router also supplies None: 0.
        a = Request("a")
        q.waiting_req_list = [a]
        self.assertEqual(q.choose([], RANKS, now=0.).layers, {"a": 0})
        self.assertTrue(q.last_record["serial_only"])
        self.assertEqual(q.pending.selected, [a])
        q.reset_feedback()
        q.max_total_tokens = 114
        self.assertEqual(q.choose([], RANKS, now=1.).layers, {"a": 8})
        self.assertFalse(q.last_record["serial_only"])
        q.reset_feedback()
        q.waiting_req_list = [Request("long", "a", 61)]
        smaller = q.choose([], RANKS, now=2.)
        self.assertTrue(smaller.plan.fits)
        self.assertEqual(smaller.selected, q.waiting_req_list)
        self.assertLess(smaller.layers["a"], 8)

    def test_new_prefix_inherits_active_group_depth(self):
        q = queue(800)
        running = [Request("a")]
        q.update_prefix_cache_state({"retained_layers": {"a": 3}})
        q.waiting_req_list = [Request("b"), Request("b2", "b")]
        chosen = q.choose(running, RANKS, now=0.)
        self.assertEqual(chosen.layers, {"a": 3, "b": 3})
        self.assertEqual(chosen.selected, q.waiting_req_list)

    def test_real_preemption_raises_reserve_and_reset_discards_trial_stats(self):
        q, running, _ = begin()
        q.observe_preemption(11)
        self.assertEqual(q.reserve, 11)
        q.choose(running, RANKS, now=1.)
        q.reset_feedback()
        self.assertEqual(q.reserve, 0)
        self.assertIsNone(q.last_rate)
        self.assertIsNone(q.pending)
        self.assertIsNone(q.trial)
        self.assertFalse(q.members)


class Staging(unittest.TestCase):
    def test_v4_snapshot_connects_real_token_accounting_and_worker(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(sys, "path", [str(root / "benchmarks"), *sys.path]):
            from run_h20_memory_cap import stage_v4_source
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            stage_v4_source(source)
            policy = source / "slora/server/router/h20_adaptive_v4.py"
            self.assertEqual(policy.read_bytes(), (root / "benchmarks/h20_adaptive_v4.py").read_bytes())
            manager = policy.with_name("manager.py").read_text()
            self.assertIn("LAYER_ADAPTIVE_V4: AdaptiveV4ReqQueue", manager)
            self.assertIn("self.req_queue.observe_output(", manager)
            self.assertIn("self.req_queue.observe_preemption(", manager)


if __name__ == "__main__":
    unittest.main()
