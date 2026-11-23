"""FCFS admission, local optimum, event triggers and isolated H20 integration."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.h20_adaptive_v2 import (
    AdaptiveV2ReqQueue, POLICY, POLICY_REVISION, local_layout_candidates, prefix_slots,
)
from slora.common.prefix_modes import LAYER_ADAPTIVE
from test_online_scheduler import request, FakeRPC, make_router


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "benchmarks/results/h20_hf_llama7b_rankmix_120s_traces"
LOCAL_TRACES = all((TRACE_ROOT / f"13b_p{p}/trace.json").is_file() for p in (8000, 10000))


def calibrated_queue(capacity=140, prefix=80, layers=8, **config):
    queue = AdaptiveV2ReqQueue(capacity, 1000, 100, prefix, LAYER_ADAPTIVE,
                              layer_num=layers, space_plan_config=config)
    queue.cost.coef.update(decode=np.array([50., 0, 0, 0]),
                           prefill=np.array([1., 0, 0, 0]),
                           reconstruct=np.array([.05, 0, 0]))
    for name in ("decode", "reconstruct"):
        queue.cost.samples[name].extend([None] * 3)
    return queue


def retained(queue, targets, ready=None, inactive=()):
    queue.update_prefix_cache_state(dict(
        retained_layers=dict(targets), ready_layers=dict(targets if ready is None else ready),
        prefix_sizes={d: (k * queue.prefix_len + queue.layer_num - 1) // queue.layer_num
                      for d, k in targets.items()},
        inactive_lora_dirs=set(inactive)))


class LocalLayoutTests(unittest.TestCase):
    def test_model_optimum_matches_exhaustive_layer_assignments(self):
        rng = random.Random(42)
        checks = 0
        for layers in (3, 5, 8):
            for n in range(1, 5):
                vectors = list(itertools.product(range(layers + 1), repeat=n))
                for _ in range(15):
                    weights = [rng.choice((0., .25, 1., 2., 4., 8.)) for _ in range(n)]
                    launch = rng.choice((0., .5, 2., 10., 100.))
                    def benefit(k):
                        return launch * min(k) + sum(x * w for x, w in zip(k, weights))
                    exact = [-float("inf")] * (n * layers + 1)
                    for k in vectors:
                        # Independent memory accounting: each non-full adapter
                        # needs a scratch row, including a zero-retention one.
                        cost = sum(k) + sum(x < layers for x in k)
                        exact[cost] = max(exact[cost], benefit(k))
                    for capacity in range(1, len(exact)):
                        exact[capacity] = max(exact[capacity], exact[capacity - 1])
                    for capacity in range(n, n * layers + 1):
                        candidates = list(local_layout_candidates(weights, layers, 10 * layers,
                                                                  10 * capacity))
                        self.assertTrue(candidates)
                        self.assertTrue(all(prefix_slots(k, 10 * layers, layers) <= 10 * capacity
                                            for k in candidates))
                        self.assertAlmostEqual(max(map(benefit, candidates)), exact[capacity])
                        checks += 1
        self.assertEqual(checks, 2130)

    def test_nonintegral_prefix_rounding_is_allocator_safe(self):
        for budget in range(5, 50):
            candidates = list(local_layout_candidates([4., 2., 1.], 5, 7, budget))
            self.assertTrue(candidates)
            self.assertTrue(all(prefix_slots(k, 7, 5) <= budget for k in candidates))
        self.assertEqual(prefix_slots((7, 0), 80, 8), prefix_slots((8, 0), 80, 8))

    def test_fixed_batch_common_floor_matches_queue_oracle(self):
        queue = calibrated_queue(190)
        queue.cost.coef["reconstruct"] = np.array([10., 0, 0])
        running = [request("a", "a", query=4), request("b", "b", query=4)]
        ranks = {"a": 8, "b": 8}
        choice = queue.choose(running, ranks, False)
        oracle = min((queue._evaluate(running, [], ranks, dict(zip(ranks, k)))
                      for k in itertools.product(range(9), repeat=2)),
                     key=lambda c: c.step_ms if c.plan.fits else float("inf"))
        self.assertEqual(choice.layers, {"a": 4, "b": 4})
        self.assertAlmostEqual(choice.step_ms, oracle.step_ms)
        self.assertTrue(queue.last_record["solver_exact_model"])

    def test_idle_cache_can_be_reclaimed_for_active_retention(self):
        queue = calibrated_queue(160)
        retained(queue, {"a": 0, "old": 8}, inactive=("old",))
        choice = queue.choose([request("a", query=4)], {"a": 8}, False)
        self.assertEqual(choice.layers, {"a": 8})
        self.assertEqual(choice.plan.evictions, ["old"])
        self.assertTrue(choice.plan.fits)

    def test_equal_cost_keeps_ready_layout(self):
        queue = calibrated_queue(132)
        queue.cost.coef["reconstruct"] = np.array([0., 125., 0.])
        retained(queue, {"a": 2, "b": 2})
        choice = queue.choose([request("a", "a", query=4), request("b", "b", query=4)],
                              {"a": 8, "b": 8}, False)
        self.assertEqual(choice.layers, {"a": 2, "b": 2})

    def test_first_forward_groups_build_and_stream_once(self):
        queue = calibrated_queue(1000)
        queue.cost.coef["reconstruct"] = np.array([1., 0., 125.])
        retained(queue, {"a": 0, "b": 0})
        running = [request("a", "a", query=4), request("b", "b", query=4)]
        ranks = {"a": 16, "b": 32}
        promoted = queue._evaluate(running, [], ranks, {"a": 8, "b": 0})
        serial = queue._evaluate(running, [], ranks, {"a": 0, "b": 0})
        self.assertEqual(promoted.first_forward_reconstruction_ms, 32.)
        self.assertEqual(promoted.first_forward_reconstruction_ms, serial.first_forward_reconstruction_ms)
        self.assertEqual(promoted.reconstruction_ms, 24.)

    def test_rank_tiling_and_shared_lora_reconstruction(self):
        queue = calibrated_queue(1000)
        queue.cost.coef["reconstruct"] = np.array([0., 0., 100.])
        self.assertEqual(queue.cost.reconstruction_ms([8], 80), queue.cost.reconstruction_ms([16], 80))
        self.assertLess(queue.cost.reconstruction_ms([16], 80), queue.cost.reconstruction_ms([32], 80))
        one = [request("one", query=4)]
        two = one + [request("two", query=4)]
        self.assertEqual(queue._evaluate(one, [], {"a": 8}, {"a": 0}).reconstruction_ms,
                         queue._evaluate(two, [], {"a": 8}, {"a": 0}).reconstruction_ms)


class FCFSAdmissionTests(unittest.TestCase):
    def test_admits_even_when_prefill_or_reconstruction_is_expensive(self):
        for capacity in (140, 1000):
            for hysteresis in (0., .9):
                with self.subTest(capacity=capacity, hysteresis=hysteresis):
                    queue = calibrated_queue(capacity, hysteresis=hysteresis, max_wait_ms=0)
                    queue.cost.coef["prefill"] = np.array([1e9, 0, 0, 0])
                    queue.cost.coef["reconstruct"] = np.array([200., 0, 0])
                    retained(queue, {"a": 8})
                    queue.waiting_req_list = [request("head", "b", query=4)]
                    choice = queue.choose([request("active", query=4)], {"a": 8, "b": 8})
                    self.assertEqual(choice.selected, queue.waiting_req_list)
                    self.assertTrue(choice.plan.fits)

    def test_legacy_search_horizon_and_slo_flags_never_limit_fcfs_admission(self):
        queue = calibrated_queue(10000, candidates=1, window=1000, group=1, beam=1,
                                 budget_ms=.000001, max_wait_ms=1, max_step_ms=.001)
        queue.waiting_req_list = [request(str(i), query=2) for i in range(70)]
        choice = queue.choose([], {"a": 8})
        self.assertEqual(choice.selected, queue.waiting_req_list)
        self.assertEqual(choice.window, 1)
        self.assertEqual(queue.last_record["waiting_considered"], 70)

    def test_unknown_output_length_and_abort_are_not_observed(self):
        class UnknownOutput:
            request_id, adapter_dir = "head", "a"
            input_len, output_ids, aborted = 4, [], False

            @property
            def max_output_len(self):
                raise AssertionError("future output length accessed")

            @property
            def arrival_time(self):
                raise AssertionError("wait age used to gate FCFS")

        queue = calibrated_queue(47, window=1000)
        aborted = request("aborted", query=10000)
        aborted.aborted = True
        queue.waiting_req_list = [aborted, UnknownOutput()]
        choice = queue.choose([], {"a": 8})
        self.assertEqual([r.request_id for r in choice.selected], ["head"])
        self.assertEqual(choice.plan.required_slots, 47)

    def test_capacity_boundary_matches_linear_oracle_with_idle_reactivation(self):
        for capacity in (100, 170, 250, 350):
            queue = calibrated_queue(capacity)
            retained(queue, {"a": 8, "idle": 6}, inactive=("idle",))
            running = [request("active", query=4)]
            queue.waiting_req_list = [request(str(i), "idle" if i % 3 == 0 else "a", query=3)
                                     for i in range(80)]
            ranks = {"a": 8, "idle": 8}
            feasible = []
            for count in range(81):
                selected = queue.waiting_req_list[:count]
                serial = dict.fromkeys(queue._targets(running, selected), 0)
                if queue._layout(running, selected, ranks, serial, 1).fits:
                    feasible.append(count)
            choice = queue.choose(running, ranks)
            self.assertEqual(len(choice.selected), max(feasible))
            self.assertEqual(choice.selected, queue.waiting_req_list[:max(feasible)])
            self.assertTrue(choice.plan.fits)

    def test_fcfs_does_not_skip_head_and_respects_hard_limits(self):
        for limit in ("prefill", "requests", "adapters", "workspace"):
            queue = calibrated_queue(1000)
            ranks = {"a": 8, "b": 8}
            if limit == "prefill":
                queue.batch_max_tokens = 3
            elif limit == "requests":
                queue.running_max_req_size = 1
            elif limit == "adapters":
                queue.max_active_loras = 1
            else:
                queue.workspace_lora_limit = 1
            queue.waiting_req_list = [request("head", "b", query=4), request("short", query=1)]
            choice = queue.choose([request("active", query=4)], ranks)
            self.assertFalse(choice.selected)

    def test_router_admission_gate_is_respected(self):
        queue = calibrated_queue(1000)
        queue.waiting_req_list = [request("head", query=1)]
        self.assertFalse(queue.choose([request("active")], {"a": 8}, False).selected)


class LocalEventTests(unittest.TestCase):
    def test_no_periodic_search_and_completion_with_same_lora_is_an_event(self):
        queue = calibrated_queue(1000)
        first, second = request("one", query=4), request("two", query=4)
        choice = queue.choose([first, second], {"a": 8}, False)
        retained(queue, choice.layers)
        queue.decode_steps = 100
        first.output_ids.append(11)
        queue.choose([first, second], {"a": 8}, False)
        self.assertFalse(queue.last_record["heavy"])
        self.assertEqual(queue.last_record["search_evaluations"], 0)
        queue.choose([second], {"a": 8}, False)
        self.assertIn("batch_changed", queue.last_record["trigger_reasons"])

    def test_private_growth_releases_cache_before_preemption(self):
        queue = calibrated_queue(120)
        running = request("active", query=4)
        choice = queue.choose([running], {"a": 8}, False)
        self.assertEqual(choice.layers, {"a": 8})
        retained(queue, choice.layers)
        running.output_ids.extend([11] * 30)
        choice = queue.choose([running], {"a": 8}, False)
        self.assertEqual(queue.last_record["trigger_reasons"], ["memory_pressure"])
        self.assertTrue(choice.plan.fits)
        self.assertEqual(choice.layers, {"a": 4})
        running.output_ids.extend([11] * 100)
        self.assertFalse(queue.choose([running], {"a": 8}, False).plan.fits)

    def test_calibration_triggers_optimization_without_batch_change(self):
        queue = calibrated_queue(120)
        queue.cost.samples["decode"].clear()
        running = [request("active", query=4)]
        cold = queue.choose(running, {"a": 8}, False)
        self.assertEqual(cold.layers, {"a": 0})
        retained(queue, cold.layers)
        queue.cost.samples["decode"].extend([None] * 3)
        warm = queue.choose(running, {"a": 8}, False)
        self.assertEqual(warm.layers, {"a": 8})
        self.assertEqual(queue.last_record["trigger_reasons"], ["model_calibrated"])

    def test_admission_after_idle_reset_even_with_reused_request_id(self):
        queue = calibrated_queue(120)
        queue.waiting_req_list = [request("same-id", query=4)]
        first = queue.choose([], {"a": 8})
        self.assertEqual(first.layers, {"a": 8})
        retained(queue, {})
        queue.waiting_req_list = [request("same-id", query=4)]
        again = queue.choose([], {"a": 8})
        self.assertIn("batch_changed", queue.last_record["trigger_reasons"])
        self.assertEqual(again.layers, {"a": 8})


class ElasticRPC(FakeRPC):
    """CPU lifecycle model of ready prefix storage and transient workspace."""
    def __init__(self, capacity, ranks, prefix=20, layers=4):
        super().__init__(capacity)
        self.ranks, self.prefix, self.layers = ranks, prefix, layers
        self.targets, self.ready, self.streaming = {}, {}, set()

    def active(self):
        return {e["req"]["adapter_dir"] for batch in self.batches.values() for e in batch.values()}

    def check(self):
        used = sum(4 * self.ranks[d] for d in self.adapters)
        used += sum(e["kv"] for batch in self.batches.values() for e in batch.values())
        used += sum((k * self.prefix + self.layers - 1) // self.layers for k in self.targets.values())
        used += (len(self.streaming) * self.prefix + self.layers - 1) // self.layers
        if used > self.capacity:
            raise AssertionError(f"allocator overflow: {used} > {self.capacity}")

    async def lowra_prefix_cache_state(self):
        return dict(retained_layers=dict(self.targets), ready_layers=dict(self.ready),
                    prefix_sizes={d: (k * self.prefix + self.layers - 1) // self.layers
                                  for d, k in self.targets.items()},
                    inactive_lora_dirs=set(self.targets) - self.active())

    async def apply_space_plan(self, layers, evictions):
        await self.evict_lowra_prefixes(evictions)
        for d, k in layers.items():
            self.targets[d], self.ready[d] = k, min(k, self.ready.get(d, 0))
        self.check()

    async def evict_lowra_prefixes(self, dirs):
        assert not (set(dirs) & self.active())
        for d in dirs:
            self.targets.pop(d, None)
            self.ready.pop(d, None)

    async def forward(self, method, batch_id):
        dirs = {e["req"]["adapter_dir"] for e in self.batches[batch_id].values()}
        self.streaming = {d for d in dirs if self.targets.get(d, 0) < self.layers}
        result = await method(batch_id)
        for d in dirs:
            self.ready[d] = self.targets.get(d, 0)
        self.streaming.clear()
        return result

    async def prefill_batch(self, batch_id):
        return await self.forward(super().prefill_batch, batch_id)

    async def decode_batch(self, batch_id):
        return await self.forward(super().decode_batch, batch_id)


class LocalLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuous_arrival_completion_and_cache_pressure_drain(self):
        queue = calibrated_queue(95, prefix=20, layers=4)
        ranks = {"a": 1, "b": 1, "c": 2}
        rpc = ElasticRPC(95, ranks)
        router = make_router(queue, rpc, ranks)
        arrivals = {0: [request("a1", "a", query=4, output=14),
                        request("b1", "b", query=4, output=8)],
                    3: [request("c1", "c", query=4, output=15)],
                    7: [request("a2", "a", query=4, output=12)]}
        admitted, reasons = [], set()
        for step in range(200):
            for req in arrivals.get(step, []):
                queue.append(req)
            await router._step_online()
            if queue.last_record:
                admitted.extend(queue.last_record["selected"])
                reasons.update(queue.last_record["trigger_reasons"])
            if step >= 7 and not router.running_batch and not queue.waiting_req_list:
                break
        else:
            self.fail("continuous batching failed to drain")
        self.assertEqual(admitted, ["a1", "b1", "c1", "a2"])
        self.assertIn("batch_changed", reasons)
        self.assertIn("memory_pressure", reasons)
        self.assertEqual(router.online_preemptions, 0)
        self.assertEqual(router.online_rejections, 0)
        self.assertFalse(rpc.batches)
        self.assertFalse(rpc.adapters)


class V2StagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "path", [str(ROOT / "benchmarks"), *sys.path]):
            from run_h20_adaptive_v2 import stage_v2_source
        cls.tmp = tempfile.TemporaryDirectory(prefix="h20-v2-test-")
        cls.source = Path(cls.tmp.name) / "source"
        originals = [ROOT / p for p in ("slora/common/prefix_modes.py",
            "slora/server/router/adaptive_req_queue.py", "benchmarks/h20_adaptive.patch",
            "benchmarks/h20_adaptive_layout.py", "benchmarks/run_h20_rankmix_suite.py")]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
        stage_v2_source(cls.source)
        assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_staged_mode_registration_and_original_policy(self):
        code = '''
from slora.common.prefix_modes import PREFIX_MODES, normalize_prefix_mode, LAYER_ADAPTIVE
from slora.server.router.adaptive_req_queue import AdaptiveReqQueue
from slora.server.router.h20_adaptive_v2 import AdaptiveV2ReqQueue
assert LAYER_ADAPTIVE == "minikv-adaptive"
for mode in ("minikv-adaptive", "minikv-adaptive-v2", "minikv-layer_serial", "minikv-full_reconstruct"):
    assert mode in PREFIX_MODES and normalize_prefix_mode(mode) == mode
assert AdaptiveReqQueue.choose is not AdaptiveV2ReqQueue.choose
assert issubclass(AdaptiveV2ReqQueue, AdaptiveReqQueue)
q = AdaptiveV2ReqQueue(100, 100, 10, 20, "minikv-adaptive-v2", layer_num=4)
assert q.prefix_mode == "minikv-adaptive-v2"
'''
        subprocess.run([sys.executable, "-c", code], cwd=self.source,
                       env=dict(os.environ, PYTHONPATH=str(self.source)), check=True,
                       capture_output=True, text=True)
        # The integration patch only selects v2 for its own name.
        text = (self.source / "slora/server/router/manager.py").read_text()
        self.assertIn("AdaptiveV2ReqQueue if input_params.lowra_prefix_mode == LAYER_ADAPTIVE_V2", text)
        self.assertIn("else AdaptiveReqQueue if adaptive else OnlineReqQueue", text)

    @unittest.skipUnless(LOCAL_TRACES, "local frozen H20 traces")
    def test_requested_gpu_plans_use_same_trace_budget_and_separate_ports(self):
        for gpu, prefix in ((0, 8000), (1, 10000)):
            result = subprocess.run([sys.executable, "benchmarks/run_h20_adaptive_v2.py",
                str(Path(self.tmp.name) / "plan"), "--gpu", str(gpu), "--point", f"13b:{prefix}",
                "--dry-run"], cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertIn(f"--prefix-len {prefix}", result.stdout)
            self.assertIn("--num-token 65000", result.stdout)
            self.assertIn(f"--port {24000 + 1000 * gpu}", result.stdout)
            self.assertIn(f"h20_hf_llama7b_rankmix_120s_traces/13b_p{prefix}/trace.json", result.stdout)
            self.assertIn("--mode minikv-adaptive-v2", result.stdout)
            self.assertIn("--duration 120", result.stdout)
            self.assertIn("--space-plan-window 1", result.stdout)
            self.assertIn("--space-plan-hysteresis 0.0", result.stdout)
            self.assertIn(POLICY_REVISION, result.stdout)
        self.assertFalse((Path(self.tmp.name) / "plan").exists())

    @unittest.skipUnless(LOCAL_TRACES and Path("/opt/conda/envs/slora/bin/python").is_file(),
                         "local H20 serving environment and frozen traces")
    def test_serving_environment_parses_trace_and_selects_only_new_policy(self):
        plan = subprocess.check_output([sys.executable, "benchmarks/run_h20_adaptive_v2.py",
            str(Path(self.tmp.name) / "serve-plan"), "--gpu", "0", "--point", "13b:8000",
            "--space-plan-window", "7", "--dry-run"], cwd=ROOT, text=True)
        command = shlex.split(plan.splitlines()[1])
        code = '''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from benchmarks.lowra_saturation_exp import parse_args, server_command
from slora.server.router.manager import get_scheduler
from slora.server.router.adaptive_req_queue import AdaptiveReqQueue
from slora.server.router.h20_adaptive_v2 import AdaptiveV2ReqQueue
from slora.server.router.online_req_queue import OnlineReqQueue
sys.argv = ["probe", *json.loads(sys.argv[1])]
args = parse_args()
assert args.mode == "minikv-adaptive-v2" and args.num_token == 65000
assert args.rank_loop == "actual" and args.block_r == 16
assert args.space_plan_window == 7
server = server_command(args, Path("/tmp/h20-v2-probe-runtime.jsonl"))
assert server[server.index("--lowra-prefix-mode") + 1] == args.mode
assert server[server.index("--space-plan-window") + 1] == "7"
params = SimpleNamespace(scheduler="online", lowra_prefix_mode=args.mode,
    space_plan_config={}, max_total_token_num=65000, batch_max_tokens=65000,
    running_max_req_size=351, lowra_shared_prefix_len=8000, enable_lowra=True,
    lowra_layer_num=40, lowra_retain_layers=0, lowra_retain_top_k=None,
    lowra_pin_hot_adapters=0, max_active_loras=None, lowra_workspace_loras=None)
for mode, expected in (("minikv-adaptive-v2", AdaptiveV2ReqQueue),
                       ("minikv-adaptive", AdaptiveReqQueue),
                       ("minikv-layer_serial", OnlineReqQueue),
                       ("minikv-full_reconstruct", OnlineReqQueue)):
    params.lowra_prefix_mode = mode
    assert type(get_scheduler(params, [])) is expected
print("serving parse, trace validation, scheduler selection and planner flags passed")
'''
        uvloop = ROOT / "benchmarks/results/.h20/uvloop-0.21.0-py39"
        result = subprocess.run([command[0], "-c", code, json.dumps(command[3:])],
            cwd=self.source, env=dict(os.environ, HF_HUB_OFFLINE="1",
                PYTHONPATH=os.pathsep.join((str(uvloop), str(self.source)))),
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
