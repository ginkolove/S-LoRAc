"""Full reconstruction: LRU ownership, nonzero math, and serving lifecycle."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from slora.common.full_reconstruct import FullReconstructPrefixKVCache, FullReconstructRuntime
from slora.common.mem_allocator import MemoryAllocator
from slora.server.router.full_reprefill_req_queue import FullReprefillReqQueue
from test_full_reprefill import CPUPool, request


class FullReconstructCacheTests(unittest.TestCase):
    @staticmethod
    def requests(*directories):
        return [dict(adapter_dir=directory) for directory in directories]

    def test_unique_builds_sharing_and_hits_do_not_reconstruct(self):
        pool = CPUPool(20)
        cache = FullReconstructPrefixKVCache(5, pool)
        builds = []
        def build(directories, tokens, indices):
            self.assertIsNone(tokens)
            builds.append(list(directories))
        first = cache.acquire_many(self.requests("a", "a", "b"), build)
        self.assertEqual(builds, [["a", "b"]])
        self.assertEqual(len(pool.used), 10)
        self.assertEqual(cache.entries["a"].ref_count, 2)
        cache.release_many(["a", "a", "b"])
        second = cache.acquire_many(self.requests("a", "a"), build)
        self.assertIs(first["a"], second["a"])
        self.assertEqual(builds, [["a", "b"]])
        self.assertEqual(cache.prefix_reconstruct_builds, 2)
        self.assertEqual(cache.prefix_reconstruct_tokens, 10)
        self.assertEqual(cache.prefix_prefill_builds, 0)
        self.assertEqual(cache.batch_unique_hits, 1)
        cache.release_many(["a", "a"])
        self.assertEqual(len(pool.used), 10)

    def test_lru_protects_hits_before_allocating_misses(self):
        pool = CPUPool(10)
        cache = FullReconstructPrefixKVCache(5, pool)
        builds = []
        def build(directories, tokens, indices):
            builds.append(list(directories))
        cache.acquire_many(self.requests("hot", "victim"), build)
        cache.release_many(["hot", "victim"])
        hot = cache.entries["hot"].prefix_indices
        cache.acquire_many(self.requests("cold", "hot", "hot"), build)
        self.assertIs(cache.entries["hot"].prefix_indices, hot)
        self.assertEqual(cache.entries["hot"].ref_count, 2)
        self.assertNotIn("victim", cache.entries)
        self.assertEqual(builds, [["hot", "victim"], ["cold"]])
        self.assertFalse(cache.evict("hot"))
        cache.release_many(["cold", "hot", "hot"])
        cache.acquire_many(self.requests("hot"), build)
        cache.release_many(["hot"])
        self.assertTrue(cache.evict_one())
        self.assertNotIn("cold", cache.entries)

    def test_failed_reconstruction_rolls_back_slots_and_references(self):
        pool = CPUPool(15)
        cache = FullReconstructPrefixKVCache(5, pool)
        cache.acquire_many(self.requests("hit"), lambda *args: None)
        cache.release_many(["hit"])
        def fail(*args):
            raise RuntimeError("reconstruction failed")
        with self.assertRaisesRegex(RuntimeError, "reconstruction failed"):
            cache.acquire_many(self.requests("hit", "new", "new"), fail)
        self.assertEqual(len(pool.used), 5)
        self.assertEqual(cache.entries["hit"].ref_count, 0)
        self.assertNotIn("new", cache.entries)
        self.assertEqual(cache.prefix_reconstruct_builds, 1)

    def test_all_layers_are_built_before_frame_without_workspace(self):
        memory = MemoryAllocator(20, 20, torch.float16, 1, 16, 3, device="cpu")
        static = SimpleNamespace(prefix_len=5, layer_num=3)
        cache = FullReconstructPrefixKVCache(5, memory)
        runtime = FullReconstructRuntime(static, memory, cache)
        with patch("slora.common.full_reconstruct.prepare_reconstruction_metadata") as metadata, \
                patch("slora.common.full_reconstruct.reconstruct_prefix_layer") as reconstruct:
            runtime.acquire_many(self.requests("a", "a", "b"), object())
            metadata.assert_called_once()
            self.assertEqual(metadata.call_args.args[1], ["a", "b"])
            self.assertEqual(metadata.call_args.args[3].shape, (2, 5))
            self.assertEqual(reconstruct.call_count, 3)
            for layer, call in enumerate(reconstruct.call_args_list):
                self.assertEqual(call.args[3], layer)
                self.assertIs(call.args[4], memory.key_buffer[layer])
                self.assertIs(call.args[5], memory.value_buffer[layer])
            frame = runtime.prepare(["a", "a", "b"])
            self.assertTrue(torch.equal(frame.prefix_b_loc[0], frame.prefix_b_loc[1]))
            for layer in range(3):
                frame.before_layer(layer)
                frame.after_attention(layer)
            frame.finish()
            self.assertEqual(reconstruct.call_count, 3)
        self.assertEqual(memory.workspace_token_equiv, 0)
        self.assertEqual(memory.tot_size - memory.can_use_mem_size, 10)
        cache.release_many(["a", "a", "b"])
        cache.evict_many(list(cache.entries))
        self.assertEqual(memory.can_use_mem_size, 20)

    def test_reconstruction_uses_exact_full_reprefill_scheduler(self):
        from slora.server.router.manager import get_scheduler
        queues = []
        for mode in ("full_reprefill", "full_reconstruct"):
            params = SimpleNamespace(enable_lowra=True, lowra_prefix_mode=mode,
                                     max_total_token_num=66, batch_max_tokens=100,
                                     running_max_req_size=10, lowra_shared_prefix_len=20)
            queue = get_scheduler(params, ["a", "b"])
            self.assertIs(type(queue), FullReprefillReqQueue)
            queue.update_prefix_cache_state(dict(cached_lora_dirs={"a", "old"},
                                                inactive_lora_dirs={"a", "old"},
                                                last_used={"old": 1, "a": 2}))
            for i, directory in enumerate(("b", "a", "a")):
                queue.append(request(directory, str(i)))
            queues.append(queue)
        batches = [queue.generate_new_batch(None, {"a": 1, "b": 1}) for queue in queues]
        self.assertEqual([r.request_id for r in batches[0].reqs],
                         [r.request_id for r in batches[1].reqs])
        self.assertEqual(batches[0].lowra_evict_prefix_lora_dirs, batches[1].lowra_evict_prefix_lora_dirs)
        self.assertEqual(batches[0].prefill_token_num, batches[1].prefill_token_num)


class FullReconstructHTTPTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        from test_full_reprefill import FullHTTPTests
        manager = FullHTTPTests().manager("layer_serial")
        manager.full_reconstruct = True
        return manager

    async def test_query_only_default_and_explicit_lengths(self):
        params = SimpleNamespace(max_new_tokens=3, stop_sentences_to_token_ids=lambda tokenizer: None)
        for lengths in (None, dict(shared_prefix_len=5, query_len=3, decode_len=3)):
            manager = self.manager()
            async for _ in manager.generate("a", [6, 7, 8], params, "r", lengths):
                pass
            self.assertEqual(manager.sent[0][1], [6, 7, 8])
            self.assertEqual(manager.sent[0][4]["query_len"], 3)
            self.assertNotIn("prefix_input_ids", manager.sent[0][4])

    async def test_logical_input_limit_and_decode_length_validation(self):
        manager = self.manager()
        params = SimpleNamespace(max_new_tokens=3, stop_sentences_to_token_ids=lambda tokenizer: None)
        with self.assertRaisesRegex(ValueError, "input prompt token len 9"):
            async for _ in manager.generate("a", [6] * 4, params, "long"):
                pass
        with self.assertRaisesRegex(ValueError, "decode_len"):
            async for _ in manager.generate("a", [6] * 3, params, "decode",
                                           dict(shared_prefix_len=5, query_len=3, decode_len=4)):
                pass
        with self.assertRaisesRegex(ValueError, "requires a LoRA"):
            async for _ in manager.generate(None, [6] * 3, params, "base"):
                pass
        self.assertEqual(manager.sent, [])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FullReconstructNumericalTests(unittest.TestCase):
    @staticmethod
    def args(mode):
        return SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                               adapter_base="dummy-lora-7b-rank-16", mode=mode, prefix_len=129,
                               query_len=5, output_len=4, batch_size=3, unique_loras=2,
                               num_adapters=3, rank=16, nonzero_prefix=True)

    def test_nonzero_math_prefill_decode_and_cache_reuse(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.common.lowra_layer_runtime import prepare_reconstruction_metadata, reconstruct_prefix_layer
        from slora.server.router.model_infer.post_process import sample

        servers, captured = [], []
        for mode in ("full_reconstruct", "layer_serial"):
            server, _, requests, _ = build_server(self.args(mode))
            servers.append(server)
            for req, tokens in zip(requests, ([11, 12, 13], [14] * 5, [15] * 7)):
                req["input_id"] = list(tokens)
            logits_by_step = []
            def capture(logits, batch):
                logits_by_step.append(logits.detach().clone())
                return sample(logits, batch)
            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture), \
                    patch("slora.common.full_reconstruct.reconstruct_prefix_layer",
                          wraps=reconstruct_prefix_layer) as reconstruct:
                server.exposed_add_batch("numeric", requests, "fp16")
                if mode == "full_reconstruct":
                    cache, static = server.lowra_prefix_cache, server.lowra_runtime.static_store
                    directories = list(cache.entries)
                    locations = torch.stack([cache.entries[d].prefix_indices for d in directories])
                    metadata = prepare_reconstruction_metadata(static, directories, server.infer_adapter, locations)
                    for layer in range(server.model.layers_num):
                        b = server.model.mem_manager.value_buffer[layer].view(-1, static.kv_embed_dim_v)
                        for row, directory in enumerate(directories):
                            start, end = static.get_lora_rank_range_by_dir(directory)
                            for base, mini, b_locs, output in (
                                (static.base_k[layer], static.mini_k[layer], metadata.k_b_locs,
                                 server.model.mem_manager.key_buffer[layer]),
                                (static.base_v[layer], static.mini_v[layer], metadata.v_b_locs,
                                 server.model.mem_manager.value_buffer[layer]),
                            ):
                                delta = (mini[:, start:end].float()
                                         @ b[b_locs[row, :end-start]].reshape(-1, end-start).float().T
                                         * metadata.scalings[row])
                                if output is server.model.mem_manager.key_buffer[layer]:
                                    first, second = delta.half().float().view(129, static.tp_k_head_num, static.head_dim).chunk(2, -1)
                                    cos = static.key_rope_cos.float()[:, None]
                                    sin = static.key_rope_sin.float()[:, None]
                                    delta = torch.cat((first * cos - second * sin,
                                                       first * sin + second * cos), -1).view_as(delta)
                                expected = base.view(129, -1).float() + delta
                                actual = output[locations[row]].view(129, -1)
                                torch.testing.assert_close(actual.float(), expected, atol=2e-4, rtol=2e-3)
                    self.assertEqual(reconstruct.call_count, server.model.layers_num)
                server.exposed_prefill_batch("numeric")
                for _ in range(3):
                    server.exposed_decode_batch("numeric")
                if mode == "full_reconstruct":
                    self.assertEqual(reconstruct.call_count, server.model.layers_num)
                    self.assertEqual(cache.prefix_reconstruct_builds, 2)
                    self.assertEqual(cache.prefix_prefill_builds, 0)
                server.exposed_remove_batch("numeric")
                if mode == "full_reconstruct":
                    directory = requests[0]["adapter_dir"]
                    indices = cache.entries[directory].prefix_indices.clone()
                    server.infer_adapter.offload_adapters([])
                    server.infer_adapter.load_adapters(server.adapters[:2])
                    hit = dict(requests[0], request_id="hit", sampling_param=dict(requests[0]["sampling_param"]))
                    server.exposed_add_batch("hit", [hit], "fp16")
                    server.exposed_prefill_batch("hit")
                    server.exposed_remove_batch("hit")
                    self.assertEqual(cache.prefix_reconstruct_builds, 2)
                    self.assertEqual(reconstruct.call_count, server.model.layers_num)
                    torch.testing.assert_close(indices, cache.entries[directory].prefix_indices)
                    cache.evict(directory)
                    server.exposed_add_batch("miss", [hit], "fp16")
                    self.assertEqual(cache.prefix_reconstruct_builds, 3)
                    server.exposed_prefill_batch("miss")
                    self.assertEqual(reconstruct.call_count, 2 * server.model.layers_num)
                    server.exposed_remove_batch("miss")
            captured.append(logits_by_step[:4])
            if server.lowra_prefix_cache:
                server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
            server.infer_adapter.offload_adapters([])
            self.assertEqual(server.model.mem_manager.workspace_token_equiv, 0)
            self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)
        for actual, expected in zip(*captured):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)

    def test_filter_merge_decode_and_idle_reset_keep_prefix_ownership(self):
        from benchmarks.lowra_layer_profile import build_server
        server, _, requests, _ = build_server(self.args("full_reconstruct"))
        server.exposed_add_batch("a", requests, "fp16")
        server.exposed_prefill_batch("a")
        server.exposed_filter_batch("a", ["1", "2"])
        cache = server.lowra_prefix_cache
        hit = dict(requests[0], request_id="extra", sampling_param=dict(requests[0]["sampling_param"]))
        server.exposed_add_batch("b", [hit], "fp16")
        server.exposed_prefill_batch("b")
        server.exposed_merge_batch("a", "b")
        server.exposed_decode_batch("a")
        self.assertEqual(cache.prefix_reconstruct_builds, 2)
        self.assertEqual(sum(entry.ref_count for entry in cache.entries.values()), 3)
        server.exposed_remove_batch("a")
        self.assertTrue(all(entry.ref_count == 0 for entry in cache.entries.values()))
        pool = server.model.mem_manager
        self.assertEqual(pool.tot_size - pool.can_use_mem_size, 2 * 129 + 2 * 4 * 16)
        server.exposed_reset_lowra_runtime()
        self.assertEqual(cache.entries, {})
        self.assertEqual(cache.prefix_reconstruct_builds, 0)
        self.assertEqual(pool.can_use_mem_size, pool.tot_size)


if __name__ == "__main__":
    unittest.main()
