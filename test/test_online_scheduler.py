"""Unknown-output admission and preempt/recompute lifecycle regressions."""

from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest

from slora.server.io_struct import Req, Batch, BatchTokenIdOut, BatchAbortReq
from slora.server.sampling_params import SamplingParams
from slora.server.router.online_req_queue import OnlineReqQueue
from slora.common.service_cycles import ServiceCycleRecorder
from slora.common.prefix_modes import FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, LAYER_RETAINED


def request(identifier, adapter="a", query=5, output=20):
    params = SamplingParams(max_new_tokens=output, ignore_eos=True)
    params.stop_sequences = []
    return Req(adapter, identifier, [10] * query, params)


class OnlineAdmissionTests(unittest.TestCase):
    def test_prefill_token_cap_is_not_a_decode_batch_size_cap(self):
        queue = OnlineReqQueue(100, 2, 10)
        running = [request(str(i), query=1) for i in range(3)]
        self.assertTrue(queue.plan(running, [], {"a": 1}).fits)
        self.assertTrue(queue.plan(running, [request("new", query=2)], {"a": 1}).fits)
        self.assertFalse(queue.plan(running, [request("large", query=3)], {"a": 1}).fits)
        queue.running_max_req_size = 3
        self.assertFalse(queue.plan(running, [request("new", query=1)], {"a": 1}).fits)

    def test_output_length_is_never_read(self):
        class HiddenLength:
            def __init__(self, identifier):
                self.request_id, self.adapter_dir = identifier, "a"
                self.input_len, self.output_ids, self.aborted = 5, [], False

            @property
            def max_output_len(self):
                raise AssertionError("output length leaked into admission")

        for mode, prefix in ((FULL_RECONSTRUCT, 20), (LAYER_SERIAL, 20), (FULL_REPREFILL, 0)):
            queue = OnlineReqQueue(50, 100, 10, prefix, mode, layer_num=4)
            queue.waiting_req_list = [HiddenLength(str(i)) for i in range(5)]
            batch = queue.generate_new_batch(None, {"a": 1})
            self.assertIsNotNone(batch)
            self.assertTrue(queue.plan(batch.reqs, [], {"a": 1}).fits)

    def test_same_admission_for_arbitrary_caps(self):
        decisions = []
        for cap in (2, 20, 20000):
            queue = OnlineReqQueue(42, 100, 20, 20, FULL_RECONSTRUCT)
            queue.waiting_req_list = [request(str(i), output=cap) for i in range(8)]
            batch = queue.generate_new_batch(None, {"a": 1})
            decisions.append([req.request_id for req in batch.reqs])
        self.assertEqual(decisions, [["0", "1", "2"]] * 3)

    def test_resume_accounts_for_known_generated_history(self):
        queue = OnlineReqQueue(25, 100, 10)
        req = request("r", output=1000)
        req.output_ids = [11] * 15
        self.assertEqual(queue.plan([], [req], {"a": 1}).required_slots, 25)
        req.output_ids.append(11)
        self.assertFalse(queue.plan([], [req], {"a": 1}).fits)

    def test_inactive_lru_then_memory_block_no_bypass(self):
        queue = OnlineReqQueue(60, 100, 10, 20, FULL_RECONSTRUCT)
        queue.update_prefix_cache_state(dict(cached_lora_dirs={"a", "old", "recent"},
            inactive_lora_dirs={"old", "recent"}, prefix_sizes={"a": 20, "old": 20, "recent": 20},
            last_used={"old": 1, "recent": 2}))
        active = request("running")
        active.output_ids = [11]
        queue.waiting_req_list = [request("b", "b"), request("large", "c", query=40), request("small")]
        batch = queue.generate_new_batch(Batch("running", [active]), {"a": 1, "b": 1, "c": 1})
        self.assertEqual([req.request_id for req in batch.reqs], ["b"])
        self.assertEqual(batch.lowra_evict_prefix_lora_dirs, ["old", "recent"])
        self.assertEqual(queue.prefix_cache_state["cached_lora_dirs"], {"a", "old", "recent"})

    def test_adapter_cap_skips_new_adapter_only(self):
        queue = OnlineReqQueue(100, 100, 10, max_active_loras=2)
        running = Batch("r", [request("a"), request("b", "b")])
        queue.waiting_req_list = [request("c", "c"), request("a2"), request("b2", "b")]
        batch = queue.generate_new_batch(running, {"a": 1, "b": 1, "c": 1})
        self.assertEqual([req.request_id for req in batch.reqs], ["a2", "b2"])
        self.assertEqual([req.request_id for req in queue.waiting_req_list], ["c"])

    def test_layer_workspace_and_per_adapter_rounding(self):
        queue = OnlineReqQueue(100, 100, 10, 17, LAYER_RETAINED, layer_num=4, retain_layers=2)
        requests = [request("a"), request("a2"), request("b", "b")]
        # Private 18; adapters 8; retained ceil(34/4)*2=18; scratch ceil(34/4)=9.
        self.assertEqual(queue.plan([], requests, {"a": 1, "b": 1}).required_slots, 53)

    def test_dynamic_top_k_covers_old_allocations_until_rebalance(self):
        queue = OnlineReqQueue(100, 100, 10, 20, LAYER_RETAINED, layer_num=4,
                               retain_layers=4, retain_top_k=1)
        queue.update_prefix_cache_state(dict(prefix_sizes={"old": 20}, cached_lora_dirs={"old"}))
        running = [request("r", "old")]
        selected = [request("n1", "new"), request("n2", "new")]
        plan = queue.plan(running, selected, {"old": 1, "new": 1})
        self.assertEqual(plan.required_slots, 17 + 8 + 20 + 5)

    def test_pinned_idle_prefix_is_not_evicted(self):
        queue = OnlineReqQueue(45, 100, 10, 20, FULL_RECONSTRUCT, pinned_lora_dirs=["pinned"])
        queue.update_prefix_cache_state(dict(prefix_sizes={"pinned": 20},
            cached_lora_dirs={"pinned"}, inactive_lora_dirs={"pinned"}))
        plan = queue.plan([], [request("r")], {"a": 1})
        self.assertFalse(plan.fits)
        self.assertEqual(plan.evictions, [])

    def test_rpc_prompt_and_output_ownership(self):
        req = request("r")
        req.output_ids = [11, 11, 12]
        rpc = req.to_rpc_obj()
        rpc["input_id"].append(13)
        rpc["generated_ids"].append(13)
        self.assertEqual(req.prompt_ids, [10] * 5)
        self.assertEqual(req.output_ids, [11, 11, 12])


class FakeRPC:
    """Enforce real allocation timing without using a GPU or output length oracle."""
    def __init__(self, capacity):
        self.capacity, self.batches, self.adapters = capacity, {}, set()
        self.resumes = []

    def check(self):
        used = 4 * len(self.adapters) + sum(
            entry["kv"] for batch in self.batches.values() for entry in batch.values())
        if used > self.capacity:
            raise AssertionError("allocator overflow: %s > %s" % (used, self.capacity))

    async def lowra_prefix_cache_state(self):
        return None

    async def load_adapters(self, dirs):
        self.adapters.update(dirs)
        self.check()

    async def offload_adapters(self, dirs=None):
        self.adapters.intersection_update(dirs or [])

    async def init_batch(self, batch_id, requests):
        self.batches[batch_id] = {}
        for req in requests:
            self.batches[batch_id][req["request_id"]] = dict(req=deepcopy(req), kv=0,
                emitted=len(req.get("generated_ids", ())))
            if req.get("generated_ids"):
                self.resumes.append(deepcopy(req))

    async def prefill_batch(self, batch_id):
        result = {}
        for identifier, entry in self.batches[batch_id].items():
            entry["kv"] = len(entry["req"]["input_id"])
            entry["emitted"] += 1
            result[identifier] = (entry["emitted"] + 100, {"id": entry["emitted"] + 100})
        self.check()
        return result

    async def decode_batch(self, batch_id):
        result = {}
        for identifier, entry in self.batches[batch_id].items():
            entry["kv"] += 1
            entry["emitted"] += 1
            result[identifier] = (entry["emitted"] + 100, {"id": entry["emitted"] + 100})
        self.check()
        return result

    async def filter_batch(self, batch_id, ids):
        self.batches[batch_id] = {key: value for key, value in self.batches[batch_id].items() if key in ids}

    async def remove_batch(self, batch_id):
        del self.batches[batch_id]

    async def merge_batch(self, first, second):
        self.batches[first].update(self.batches.pop(second))


def make_router(queue, rpc, ranks):
    from slora.server.router.manager import RouterManager
    from slora.server.router.stats import Stats
    router = RouterManager.__new__(RouterManager)
    router.input_params = SimpleNamespace(scheduler="online", schedule_interval=1,
        no_lora=False, enable_lowra=False, lowra_prefix_mode=FULL_REPREFILL)
    router.req_queue, router.model_rpcs, router.lora_ranks = queue, [rpc], ranks
    router.world_size, router.eos_id, router.has_wait_tokens = 1, 2, 0
    router.running_batch = None
    router.stats_tool = Stats(False, 10)
    router.messages = []
    router.send_to_detokenization = SimpleNamespace(send_pyobj=router.messages.append)
    router.service_cycles = ServiceCycleRecorder(io.StringIO())
    router.online_preemptions = router.online_recomputed_tokens = router.online_rejections = 0
    router.online_preempted_request_ids = set()
    router.preemption_stats_file = io.StringIO()
    return router


async def drain(router, steps=2000):
    for _ in range(steps):
        await router._step()
        if router.running_batch is None and not router.req_queue.waiting_req_list:
            return
    raise AssertionError("scheduler failed to drain")


class OnlineLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_pressure_recompute_preserves_every_output_and_frees_every_slot(self):
        queue, rpc = OnlineReqQueue(32, 100, 10), FakeRPC(32)
        reqs = [request("a1", output=20), request("a2", output=15), request("b", "b", output=18)]
        queue.waiting_req_list = reqs[:]
        router = make_router(queue, rpc, {"a": 1, "b": 1})
        await drain(router)
        self.assertGreater(router.online_preemptions, 0)
        self.assertGreater(len(rpc.resumes), 0)
        self.assertEqual(router.online_rejections, 0)
        events = [json.loads(line) for line in router.preemption_stats_file.getvalue().splitlines()]
        preemptions = [row for row in events if row["event"] == "preempt"]
        resumes = [row for row in events if row["event"] == "resume_prefill"]
        self.assertEqual(len(preemptions), router.online_preemptions)
        self.assertEqual(router.online_preemptions, sum(req.preemptions for req in reqs))
        self.assertEqual(router.online_preempted_request_ids,
                         {req.request_id for req in reqs if req.preemptions})
        self.assertEqual(router.online_recomputed_tokens, sum(
            req["input_tokens"] + req["generated_tokens"] for row in resumes for req in row["requests"]))
        self.assertEqual(sum(len(row["requests"]) for row in resumes), len(rpc.resumes))
        self.assertTrue(all(row["prefill_wall_seconds"] >= 0 for row in resumes))
        for req in reqs:
            emitted = [token for message in router.messages if isinstance(message, BatchTokenIdOut)
                       for identifier, token, _, _, _ in message.reqs_infs if identifier == req.request_id]
            self.assertEqual(emitted, list(range(101, 101 + req.max_output_len)))
            self.assertEqual(req.output_ids, emitted)
            self.assertEqual(req.prompt_ids, [10] * 5)
        self.assertEqual(rpc.batches, {})
        self.assertEqual(rpc.adapters, set())

    async def test_impossible_prompt_rejects_instead_of_hanging(self):
        queue, rpc = OnlineReqQueue(20, 100, 10), FakeRPC(20)
        queue.waiting_req_list = [request("large", query=30), request("ok", output=3)]
        router = make_router(queue, rpc, {"a": 1})
        await drain(router)
        self.assertEqual(router.online_rejections, 1)
        self.assertTrue(any(isinstance(msg, BatchAbortReq) and msg.reqs == ["large"] for msg in router.messages))

    async def test_single_request_outgrows_pool_terminates_cleanly(self):
        queue, rpc = OnlineReqQueue(16, 100, 10), FakeRPC(16)
        queue.waiting_req_list = [request("too-long", output=100)]
        router = make_router(queue, rpc, {"a": 1})
        await drain(router)
        self.assertEqual(router.online_rejections, 1)
        self.assertEqual(rpc.batches, {})

    async def test_eos_finishes_before_cap_and_releases_memory(self):
        queue, rpc = OnlineReqQueue(20, 100, 10), FakeRPC(20)
        req = request("eos", output=10000)
        req.sample_params.ignore_eos = False
        queue.append(req)
        router = make_router(queue, rpc, {"a": 1})
        router.eos_id = 103
        await drain(router)
        self.assertEqual(req.output_ids, [101, 102, 103])
        self.assertFalse(req.aborted)
        self.assertEqual(router.online_rejections, 0)
        self.assertEqual(rpc.batches, {})

    async def test_stop_sequence_survives_preemption(self):
        queue, rpc = OnlineReqQueue(22, 100, 10), FakeRPC(22)
        first, second = request("first", output=8), request("second", output=100)
        second.sample_params.stop_sequences = [[105, 106]]
        queue.waiting_req_list = [first, second]
        router = make_router(queue, rpc, {"a": 1})
        await drain(router)
        self.assertGreater(router.online_preemptions, 0)
        self.assertEqual(second.output_ids, list(range(101, 107)))
        self.assertFalse(second.aborted)

    async def test_abort_preempted_request_does_not_resume(self):
        queue, rpc = OnlineReqQueue(20, 100, 10), FakeRPC(20)
        reqs = [request("a1"), request("a2")]
        queue.waiting_req_list = reqs[:]
        router = make_router(queue, rpc, {"a": 1})
        for _ in range(20):
            await router._step()
            if router.online_preemptions:
                break
        victim = next(req for req in queue.waiting_req_list if req.preemptions)
        await router.abort(victim.request_id)
        await drain(router)
        self.assertFalse(any(req["request_id"] == victim.request_id for req in rpc.resumes))


if __name__ == "__main__":
    unittest.main()
