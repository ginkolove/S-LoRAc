import asyncio
from pathlib import Path
import tempfile
import time
import unittest

from benchmarks.fcfs_fork_exp import Playback, branch, read, summarize, write
from benchmarks.fcfs_fork_policy import ForkQueue
from benchmarks.fcfs_fork_report import fit, report
from slora.common.prefix_modes import LAYER_ADAPTIVE
from test_online_scheduler import FakeRPC, make_router, request


def queue(cap=44):
    q = ForkQueue(cap, 100, 20, 20, LAYER_ADAPTIVE, layer_num=4)
    q.update_prefix_cache_state(dict(retained_layers={"a": 4}, ready_layers={"a": 4},
                                    prefix_sizes={"a": 20}))
    return q


class ForkTests(unittest.TestCase):
    def test_actions_are_exact_fcfs_and_account_for_scratch(self):
        q = queue()
        running = [request("r", query=6)]
        q.waiting_req_list = [request("h", "b", query=7), request("tail")]
        ranks = {"a": 1, "b": 1}
        q.arm = "wait"
        self.assertEqual(q.choose(running, ranks).selected, [])
        fact = q.opportunity
        q.arm = "shrink"
        plan = q.choose(running, ranks)
        self.assertEqual(fact, q.opportunity)
        self.assertEqual([r.request_id for r in plan.selected], ["h"])
        self.assertEqual(plan.layers, {"a": 2, "b": 0})
        self.assertEqual(fact["new_groups"], ["b"])
        self.assertEqual(fact["lost_ready"], {"a": 2})
        self.assertEqual(plan.plan.required_slots, 42)
        self.assertEqual(q.prefix_cache_state["retained_layers"], {"a": 4})

    def test_no_fake_pressure_or_bypass(self):
        q = queue()
        q.waiting_req_list = [request("large", "b", query=100), request("fits", "a", query=1)]
        self.assertFalse(q.choose([request("r")], {"a": 1, "b": 1}).selected)
        self.assertIsNone(q.opportunity)
        q.waiting_req_list.pop(0)
        self.assertTrue(q.choose([request("r")], {"a": 1}).selected)
        self.assertIsNone(q.opportunity)

    def test_idle_lru_before_active_demotion(self):
        q = queue(53)
        q.prefix_cache_state.update(prefix_sizes={"a":20, "idle":20}, inactive_lora_dirs=["idle"])
        q.waiting_req_list = [request("h", "b")]
        choice = q.choose([request("r")], {"a":1, "b":1})
        self.assertEqual(choice.plan.evictions, ["idle"])
        self.assertIsNone(q.opportunity)
        self.assertEqual(choice.layers["a"], 4)

    def test_window_boundary_and_censoring(self):
        p = dict(last_before={"old":-.2}, head_wait_s=3, action=dict(head="h"))
        events = [(0.2, ["old"]), (1., ["h", "old"]), (1.01, ["other"])]
        a = summarize(events, p, 0.5)
        self.assertTrue(a["head_censored"])
        self.assertEqual(a["running_itl_p95_s"], .4)
        b = summarize(events, p, 1.)
        self.assertEqual((b["tokens"], b["head_ttft_s"]), (3, 4.))

    def test_no_threshold_without_crossover(self):
        sample = [dict(rho=.1, delta_tokens=-5), dict(rho=.2, delta_tokens=8)]
        self.assertAlmostEqual(fit(sample), .15)
        self.assertIsNone(fit(sample[1:]))
        self.assertIsNone(fit([dict(rho=.1, delta_tokens=5), dict(rho=.2, delta_tokens=-8)]))

    def test_pair_through_real_router_with_fake_model(self):
        class RPC(FakeRPC):
            layers = None

            async def lowra_prefix_cache_state(self):
                layers = self.layers or {}
                return dict(retained_layers=layers, ready_layers=layers,
                            prefix_sizes={d:k*5 for d,k in layers.items()})

            async def apply_space_plan(self, layers, evictions):
                self.layers = dict(layers)

        class FakePlayback(Playback):
            def signature(self):
                return repr([(r.request_id, r.output_ids) for r in self.router.running_batch.reqs])

        rows = [dict(req_id="old", req_time=0., input_ids=[4]*6, output_len=8, adapter_index=0),
                dict(req_id="h", req_time=.01, input_ids=[4]*7, output_len=3, adapter_index=1)]
        def setup():
            r = make_router(queue(), RPC(44), {"a":1, "b":1})
            r.adapter_dirs = ["a", "b"]
            r.events, r.last_token = None, {}
            def emit(batch, ans):
                now = time.perf_counter()
                r.last_token.update(dict.fromkeys(ans, now))
                if r.events is not None:
                    r.events.append((now-r.origin, list(ans)))
            r._send_to_detokenization_proc = emit
            return FakePlayback(r, dict(requests=rows))

        async def run():
            pb = setup()
            pb.inject(1)
            await pb.router._step_online()
            await pb.router._step_online()
            pb.inject(2)
            p = dict(index=2, cursor=2, action=await pb.preview(), signature=pb.signature(),
                     head_wait_s=.01, trace_time=.02, unit_s=.002, decode_step=1,
                     last_before={"old":-.001})
            journal = dict(points=[p], cursors=[1,1], eligible=1, blocked=1, config={"duration": 1})
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                write(out / "journal.json", journal)
                for arm, first in (("wait", ["old"]), ("shrink", ["h"])):
                    result = await branch(setup(), journal, 0, arm)
                    self.assertEqual(result["events"][0][1], first)
                    write(out / ("000-%s.json" % arm), result)
                report(out)
                self.assertEqual(read(out / "summary.json")["pairs"], 1)
                self.assertTrue((out / "gain.svg").is_file())
                p["signature"] = "mismatch"
                with self.assertRaisesRegex(RuntimeError, "refusing unpaired"):
                    await branch(setup(), journal, 0, "wait")
        asyncio.run(run())
