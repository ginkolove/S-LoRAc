"""FIFO admission, cache ownership, and full-prefill numerical equivalence."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from slora.common.full_reprefill import FullPrefixKVCache, FullReprefillRuntime
from slora.server.router.full_reprefill_req_queue import FullReprefillReqQueue


def request(directory, identifier, query=5, output=4):
    return SimpleNamespace(adapter_dir=directory, request_id=identifier, input_len=query,
                           max_output_len=output, output_ids=[], aborted=False)


class CPUPool:
    def __init__(self, size):
        self.size = size
        self.used = set()

    @property
    def can_use_mem_size(self):
        return self.size - len(self.used)

    def alloc(self, count):
        slots = [i for i in range(self.size) if i not in self.used][:count]
        if len(slots) != count:
            raise RuntimeError("pool exhausted")
        self.used.update(slots)
        return torch.tensor(slots)

    def free(self, indices):
        for index in indices.tolist():
            self.used.remove(index)


class FullSchedulerTests(unittest.TestCase):
    def test_unique_prefix_and_future_decode_reservation(self):
        queue = FullReprefillReqQueue(44, 100, 10, 20)
        requests = [request("a", str(i)) for i in range(3)]
        for req in requests:
            queue.append(req)
        batch = queue.generate_new_batch(None, {"a": 1})
        # Two requests: 20 prefix + 4 adapter + 2*(5 query + 4 output) = 42.
        self.assertEqual(batch.reqs, requests[:2])
        self.assertEqual(batch.prefix_prefill_tokens, 20)
        self.assertEqual(batch.prefill_token_num, 30)
        self.assertEqual(queue.waiting_req_list, requests[2:])

    def test_lru_eviction_before_admission_and_active_pin(self):
        queue = FullReprefillReqQueue(66, 100, 10, 20)
        queue.update_prefix_cache_state(dict(cached_lora_dirs={"active", "old", "recent"},
                                            inactive_lora_dirs={"old", "recent"},
                                            last_used={"old": 1, "recent": 2, "active": 0}))
        running_req = request("active", "running")
        running_req.output_ids = [1]
        incoming = request("new", "new")
        queue.append(incoming)
        batch = queue.generate_new_batch(SimpleNamespace(reqs=[running_req]),
                                         {d: 1 for d in ("active", "new")})
        self.assertEqual(batch.reqs, [incoming])
        self.assertEqual(batch.lowra_evict_prefix_lora_dirs, ["old", "recent"])

    def test_fifo_no_bypass_and_rejected_candidate_has_no_eviction_effect(self):
        queue = FullReprefillReqQueue(40, 100, 10, 20)
        queue.update_prefix_cache_state(dict(cached_lora_dirs={"hot", "old"},
                                            inactive_lora_dirs={"hot", "old"},
                                            last_used={"old": 1, "hot": 2}))
        first, second = request("cold", "first", output=20), request("hot", "second")
        queue.append(first)
        queue.append(second)
        self.assertIsNone(queue.generate_new_batch(None, {"cold": 1, "hot": 1}))
        self.assertEqual(queue.waiting_req_list, [first, second])
        self.assertEqual(queue.prefix_cache_state["cached_lora_dirs"], {"hot", "old"})

    def test_prefill_limit_counts_unique_misses_not_hits(self):
        queue = FullReprefillReqQueue(200, 30, 10, 20)
        queue.update_prefix_cache_state(dict(cached_lora_dirs={"hit"}, inactive_lora_dirs={"hit"}))
        requests = [request("miss", "1"), request("miss", "2"), request("hit", "3")]
        for req in requests:
            queue.append(req)
        batch = queue.generate_new_batch(None, {"miss": 1, "hit": 1})
        self.assertEqual(batch.reqs, requests[:2])
        self.assertEqual(batch.prefill_token_num, 30)


class FullCacheTests(unittest.TestCase):
    def setUp(self):
        self.pool = CPUPool(15)
        self.cache = FullPrefixKVCache(5, self.pool)
        self.builds = []

    @staticmethod
    def requests(*directories):
        return [dict(adapter_dir=d, prefix_input_ids=[1, 2, 3, 4, 5]) for d in directories]

    def build(self, directories, tokens, indices):
        self.builds.append(list(directories))

    def test_unique_build_sharing_and_cache_owned_release(self):
        indices = self.cache.acquire_many(self.requests("a", "a", "b"), self.build)
        self.assertEqual(self.builds, [["a", "b"]])
        self.assertEqual(len(self.pool.used), 10)
        self.assertEqual(self.cache.entries["a"].ref_count, 2)
        self.cache.release_many(["a", "a", "b"])
        self.assertEqual(len(self.pool.used), 10)
        again = self.cache.acquire_many(self.requests("a"), self.build)
        self.assertEqual(self.builds, [["a", "b"]])
        torch.testing.assert_close(indices["a"], again["a"])
        self.cache.release_many(["a"])
        self.assertTrue(self.cache.evict_one())
        self.assertNotIn("b", self.cache.entries)
        self.assertIn("a", self.cache.entries)

    def test_failed_build_rolls_back_slots_and_hit_references(self):
        self.cache.acquire_many(self.requests("a"), self.build)
        self.cache.release_many(["a"])
        def fail(*args):
            raise RuntimeError("build failed")
        with self.assertRaisesRegex(RuntimeError, "build failed"):
            self.cache.acquire_many(self.requests("a", "b"), fail)
        self.assertEqual(len(self.pool.used), 5)
        self.assertEqual(self.cache.entries["a"].ref_count, 0)
        self.assertNotIn("b", self.cache.entries)

    def test_active_protection_and_prefix_identity(self):
        self.cache.acquire_many(self.requests("a"), self.build)
        self.assertFalse(self.cache.evict("a"))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.cache.acquire_many([dict(adapter_dir="b", prefix_input_ids=[9] * 5)], self.build)
        self.assertEqual(len(self.pool.used), 5)
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            self.cache.release_many(["a", "a"])
        self.assertEqual(self.cache.entries["a"].ref_count, 1)


class FullHTTPTests(unittest.IsolatedAsyncioTestCase):
    def manager(self, mode="full"):
        from slora.server.httpserver.manager import HttpServerManager
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.enable_lowra, manager.full_reprefill = True, mode == "full"
        manager.lowra_shared_prefix_len, manager.shared_prefix_ids = 5, None
        manager.max_req_input_len, manager.max_req_total_len, manager.total_token_num = 8, 12, 100
        manager.tokenizer = None
        manager.req_id_to_out_inf = {}
        manager.sent = []
        def send(value):
            manager.sent.append(value)
            def complete():
                _, _, _, event = manager.req_id_to_out_inf[value[3]]
                manager.req_id_to_out_inf[value[3]] = ("ok", {"id": 1}, True, event)
                event.set()
            asyncio.get_running_loop().call_soon(complete)
        manager.send_to_router = SimpleNamespace(send_pyobj=send)
        return manager

    async def test_full_input_is_split_and_invalid_request_does_not_bind_prefix(self):
        manager = self.manager()
        params = SimpleNamespace(max_new_tokens=3, stop_sentences_to_token_ids=lambda tokenizer: None)
        with self.assertRaises(ValueError):
            async for _ in manager.generate("a", [9] * 10, params, "invalid"):
                pass
        self.assertIsNone(manager.shared_prefix_ids)
        async for _, metadata, _ in manager.generate("a", [1, 2, 3, 4, 5, 6, 7, 8], params, "valid"):
            self.assertEqual(metadata["prompt_tokens"], 8)
        self.assertEqual(manager.sent[0][1], [6, 7, 8])
        self.assertEqual(manager.sent[0][4]["prefix_input_ids"], [1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "same shared prefix"):
            async for _ in manager.generate("a", [9] * 8, params, "changed"):
                pass

    async def test_full_rejects_query_only_and_serial_preserves_query_only(self):
        params = SimpleNamespace(max_new_tokens=3, stop_sentences_to_token_ids=lambda tokenizer: None)
        lengths = dict(shared_prefix_len=5, query_len=3, decode_len=3)
        with self.assertRaisesRegex(ValueError, "full reprefill"):
            async for _ in self.manager().generate("a", [6, 7, 8], params, "r", lengths):
                pass
        manager = self.manager("layer_serial")
        async for _ in manager.generate("a", [6, 7, 8], params, "r", lengths):
            pass
        self.assertEqual(manager.sent[0][1], [6, 7, 8])

    async def test_online_http_does_not_reserve_the_output_cap_in_the_pool(self):
        params = SimpleNamespace(max_new_tokens=3, stop_sentences_to_token_ids=lambda tokenizer: None)
        manager = self.manager()
        manager.total_token_num = 10
        with self.assertRaisesRegex(ValueError, "total len"):
            async for _ in manager.generate("a", [1] * 8, params, "legacy"):
                pass
        manager.scheduler = "online"
        async for _ in manager.generate("a", [1] * 8, params, "online"):
            pass
        self.assertEqual(manager.sent[-1][3], "online")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FullNumericalTests(unittest.TestCase):
    def test_cached_prefill_and_decode_match_ordinary_full_prompt(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.server.router.model_infer.post_process import sample

        args = SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                               adapter_base="dummy-lora-7b-rank-16", mode="full", prefix_len=129,
                               query_len=5, output_len=4, batch_size=3, unique_loras=2,
                               num_adapters=2, rank=16, nonzero_prefix=False)
        server, profiler, requests, _ = build_server(args)
        self.assertIsInstance(server.lowra_runtime, FullReprefillRuntime)
        self.assertFalse(hasattr(server.lowra_runtime, "static_store"))
        prompts = [[11, 12, 13], [14] * 5, [15] * 7]
        for req, prompt in zip(requests, prompts):
            req["input_id"] = list(prompt)
        cached = []
        def capture(logits, batch):
            cached.append(logits.detach().clone())
            return sample(logits, batch)
        with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture):
            server.exposed_add_batch("cached", requests, "fp16")
            cache = server.lowra_prefix_cache
            self.assertEqual(cache.prefix_prefill_builds, 2)
            frame = server.lowra_runtime.prepare([r["adapter_dir"] for r in requests])
            self.assertTrue(torch.equal(frame.prefix_b_loc[0], frame.prefix_b_loc[2]))
            for layer in range(server.model.layers_num):
                for entry in cache.entries.values():
                    self.assertTrue(torch.isfinite(server.model.mem_manager.key_buffer[layer][entry.prefix_indices]).all())
                    self.assertGreater(server.model.mem_manager.value_buffer[layer][entry.prefix_indices].abs().sum(), 0)
            server.exposed_prefill_batch("cached")
            for _ in range(3):
                server.exposed_decode_batch("cached")
        server.exposed_remove_batch("cached")
        # Offloading weights does not evict the derived prefix cache.
        indices = cache.entries[requests[0]["adapter_dir"]].prefix_indices.clone()
        server.infer_adapter.offload_adapters([])
        server.infer_adapter.load_adapters(server.adapters[:2])
        hit = dict(requests[0], request_id="hit", input_id=[10] * 5,
                   sampling_param=dict(requests[0]["sampling_param"]))
        server.exposed_add_batch("hit", [hit], "fp16")
        server.exposed_prefill_batch("hit")
        server.exposed_remove_batch("hit")
        self.assertEqual(cache.prefix_prefill_builds, 2)
        torch.testing.assert_close(cache.entries[hit["adapter_dir"]].prefix_indices, indices)
        cache.evict_many(list(cache.entries))
        server.lowra_runtime, server.lowra_prefix_cache = None, None
        for row, req in enumerate(requests):
            ordinary = []
            def capture_reference(logits, batch):
                ordinary.append(logits.detach().clone())
                return sample(logits, batch)
            ref = dict(req, request_id="reference", input_id=[10] * 129 + prompts[row],
                       sampling_param=dict(req["sampling_param"]))
            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture_reference):
                server.exposed_add_batch("reference", [ref], "fp16")
                server.exposed_prefill_batch("reference")
                for _ in range(3):
                    server.exposed_decode_batch("reference")
            for actual, expected in zip(cached, ordinary):
                torch.testing.assert_close(actual[row:row + 1], expected, atol=2e-4, rtol=2e-3)
            server.exposed_remove_batch("reference")
        self.assertEqual(server.model.mem_manager.tot_size - server.model.mem_manager.can_use_mem_size, 128)
        server.infer_adapter.offload_adapters([])
        self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)


if __name__ == "__main__":
    unittest.main()
