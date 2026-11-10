"""Cross-batch retained-prefix reuse, admission pressure and shared index tables."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from slora.common.mem_allocator import MemoryAllocator
from slora.common.lowra_retained import RetainedPrefixKVCache, RetainedLayerRuntime
from slora.common.prefix_indices import PrefixIndexCache
from slora.common.prefix_modes import LAYER_RETAINED
from slora.server.router.lowra_fifo_req_queue import LowRAFIFOReqQueue


def pool(size=40):
    return MemoryAllocator(size, size, torch.float16, 1, 16, 4,
                           dynamic_layer_workspace=True, device="cpu")


def request(directory, ident=None, output=10):
    return SimpleNamespace(adapter_dir=directory, request_id=ident or directory,
                           input_len=3, max_output_len=output, output_ids=[], aborted=False)


class RetainedLRUTests(unittest.TestCase):
    def test_idle_hit_lru_and_active_protection(self):
        memory = pool(12)
        cache = RetainedPrefixKVCache(7, 2, memory)
        for directory in ("a", "b", "c"):
            cache.acquire_many([directory])
            cache.entries[directory].ready = True
            memory.prefix_workspace_k[cache.entries[directory].prefix_indices] = ord(directory)
            cache.release_many([directory])
        old_a = cache.entries["a"].prefix_indices
        cache.acquire_many(["a", "a"])
        cache.acquire_many(["d"])
        self.assertNotIn("b", cache.entries)
        self.assertIn("c", cache.entries)
        self.assertIs(cache.entries["a"].prefix_indices, old_a)
        self.assertTrue(torch.all(memory.prefix_workspace_k[old_a] == ord("a")))
        self.assertFalse(cache.evict("a"))
        self.assertEqual(cache.entries["a"].ref_count, 2)
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(cache.batch_unique_hits, 1)
        cache.release_many(["a", "a", "d"])
        self.assertNotIn("d", cache.entries)  # An incomplete entry is not reusable.
        cache.evict_many(list(cache.entries))
        self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def test_batch_acquisition_protects_all_hits_before_allocating_misses(self):
        memory = pool(8)
        cache = RetainedPrefixKVCache(7, 2, memory)
        for directory in ("hit", "victim"):
            cache.acquire_many([directory])
            cache.entries[directory].ready = True
            cache.release_many([directory])
        old = cache.entries["hit"]
        cache.acquire_many(["new", "hit"])
        self.assertIs(cache.entries["hit"], old)
        self.assertEqual(old.ref_count, 1)
        self.assertNotIn("victim", cache.entries)

    def test_zero_layers_leave_no_idle_prefix(self):
        cache = RetainedPrefixKVCache(7, 0, pool())
        cache.acquire_many(["a", "a"])
        cache.release_many(["a", "a"])
        self.assertFalse(cache.entries)
        self.assertEqual(cache.mem_manager.can_use_mem_size, 40)

    def test_fifo_evicts_lru_for_decode_reservation_and_does_not_commit_rejection(self):
        # New a: private peak 13 + weights 4 + retained 4 + scratch 2 = 23.
        # Two idle entries add 8; budget 28 admits a after one LRU eviction.
        queue = LowRAFIFOReqQueue(28, 100, 20, 7, LAYER_RETAINED, layer_num=4, retain_layers=2)
        state = dict(cached_lora_dirs={"old", "recent"}, inactive_lora_dirs={"old", "recent"},
                     prefix_sizes={"old": 4, "recent": 4}, last_used={"old": 1, "recent": 2})
        queue.update_prefix_cache_state(state)
        queue.append(request("a"))
        queue.append(request("too_big", output=100))
        batch = queue.generate_new_batch(None, {"a": 1, "too_big": 1})
        self.assertEqual([req.adapter_dir for req in batch.reqs], ["a"])
        self.assertEqual(batch.lowra_evict_prefix_lora_dirs, ["old"])
        self.assertEqual(state["cached_lora_dirs"], {"old", "recent"})
        self.assertEqual(queue.waiting_req_list[0].adapter_dir, "too_big")

    def test_fifo_preserves_idle_when_it_fits_and_active_or_pinned_under_pressure(self):
        for budget, active, pinned, admitted, victims in (
                (32, False, False, True, []),
                (28, False, False, True, ["old"]),
                (28, True, False, True, ["recent"]),
                (24, True, False, False, []),
                (24, False, True, False, [])):
            queue = LowRAFIFOReqQueue(budget, 100, 20, 7, LAYER_RETAINED, layer_num=4,
                                     retain_layers=2, pinned_lora_dirs=["old", "a"] if pinned else ())
            queue.update_prefix_cache_state(dict(
                cached_lora_dirs={"old", "recent"}, inactive_lora_dirs={"recent"} if active else {"old", "recent"},
                prefix_sizes={"old": 4, "recent": 4}, last_used={"old": 1, "recent": 2}))
            queue.append(request("a"))
            batch = queue.generate_new_batch(None, {"a": 1})
            self.assertEqual(batch is not None, admitted)
            if batch:
                self.assertEqual(batch.lowra_evict_prefix_lora_dirs, victims)

    def test_idle_reuse_skips_retained_reconstruction_and_reuses_unique_indices(self):
        for retained in (0, 2, 4):
            memory = pool()
            cache = RetainedPrefixKVCache(7, retained, memory)
            static = SimpleNamespace(layer_num=4, prefix_len=7, base_k=[torch.zeros(1)])
            runtime = RetainedLayerRuntime(static, memory, cache)

            def metadata(static, dirs, adapter, locs):
                return SimpleNamespace(prefix_locs=locs)

            def reconstruct(static, adapter, meta, layer, keys, values):
                keys[meta.prefix_locs] = layer + 1
                values[meta.prefix_locs] = -layer - 1

            with patch("slora.common.lowra_retained.prepare_reconstruction_metadata", side_effect=metadata), \
                    patch("slora.common.lowra_retained.reconstruct_prefix_layer", side_effect=reconstruct) as kernel:
                previous = None
                for round_id in range(2):
                    cache.acquire_many(["a", "a", "b"])
                    frame = runtime.prepare(["a", "a", "b"], object())
                    self.assertEqual(frame.prefix_index_table(0).shape, (2, 7))
                    self.assertEqual(frame.prefix_request_ids.tolist(), [0, 0, 1])
                    if retained:
                        if previous is not None:
                            self.assertIs(frame.retained_requests, previous)
                        previous = frame.retained_requests
                    for layer in range(4):
                        frame.before_layer(layer)
                    frame.finish()
                    cache.release_many(["a", "a", "b"])
                    self.assertEqual(memory.workspace_token_equiv, 0)
                    self.assertEqual(len(cache.entries), 2 if retained else 0)
                self.assertEqual(kernel.call_count, 8 - retained)

    def test_prepare_reclaims_idle_for_workspace_and_private_allocation(self):
        memory = pool(12)
        cache = RetainedPrefixKVCache(7, 2, memory)
        cache.acquire_many(["idle", "active"])
        for entry in cache.entries.values():
            entry.ready = True
        cache.release_many(["idle"])
        static = SimpleNamespace(layer_num=4, prefix_len=7, base_k=[torch.zeros(1)])
        runtime = RetainedLayerRuntime(static, memory, cache)
        with patch("slora.common.lowra_retained.prepare_reconstruction_metadata", return_value=None):
            frame = runtime.prepare(["active"], object(), reserve_tokens=5)
        self.assertNotIn("idle", cache.entries)
        self.assertEqual(memory.workspace_token_equiv, 2)
        private = memory.alloc(5)
        memory.free(private)
        frame.finish()

    def test_top_k_promotion_reclaims_idle_without_counting_it_as_active_hot(self):
        memory = pool(16)
        cache = RetainedPrefixKVCache(7, 4, memory, retain_top_k=1)
        cache.acquire_many(["a", "a", "b"])
        cache.rebalance()
        cache.entries["a"].ready = True
        cache.release_many(["a", "a"])
        self.assertEqual(cache._selected_hot(), {"b"})
        cache.rebalance()
        self.assertEqual(cache.state_dict()["inactive_lora_dirs"], {"a"})
        self.assertEqual(memory.can_use_mem_size, 2)
        cache.ensure_available(5)
        self.assertNotIn("a", cache.entries)
        self.assertIn("b", cache.entries)
        self.assertEqual(cache.evictions, 1)


class PrefixIndexTests(unittest.TestCase):
    def test_shared_mapping_reuse_and_invalidation(self):
        cache = PrefixIndexCache()
        dirs, ids = cache.request_mapping(["a", "b", "a"], "cpu")
        self.assertEqual(dirs, ["a", "b"])
        self.assertEqual(ids.tolist(), [0, 1, 0])
        self.assertIs(cache.request_mapping(["a", "b", "a"], "cpu")[1], ids)
        self.assertEqual(cache.request_mapping(["b", "a"], "cpu")[1].tolist(), [0, 1])
        a, b = torch.arange(14).view(2, 7), torch.arange(14, 28).view(2, 7)
        table = cache.stack("retained", [a, b], dim=1)
        self.assertEqual(table.shape, (2, 2, 7))
        self.assertIs(cache.stack("retained", [a, b], dim=1), table)
        b.add_(10)
        self.assertIsNot(cache.stack("retained", [a, b], dim=1), table)
        replacement = b.clone()
        self.assertTrue(torch.equal(cache.stack("retained", [a, replacement], dim=1)[:, 1], b))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PrefixIndirectionCUDATests(unittest.TestCase):
    def test_prefill_and_decode_match_torch_with_shared_scattered_indices(self):
        from slora.models.lowra.prefix_attention import context_attention_fwd_with_prefix
        from slora.models.lowra.decode_attention import token_attention_fwd_with_prefix

        torch.manual_seed(42)
        for prefix, lengths_list in ((17, [3, 5, 2]), (257, [17, 3, 65]), (10000, [5, 3, 7]),
                                     (12000, [3, 5, 2])):
            heads, kv_heads, dim = 4, 2, 128
            starts_list = [0, lengths_list[0], sum(lengths_list[:2])]
            total, maximum = sum(lengths_list), max(lengths_list)
            starts = torch.tensor(starts_list, device="cuda", dtype=torch.int32)
            lengths = torch.tensor(lengths_list, device="cuda", dtype=torch.int32)
            request_ids = torch.tensor([1, 0, 1], device="cuda")
            locs = torch.randperm(2 * prefix + 9, device="cuda")[:2 * prefix].view(2, prefix)
            pk = torch.randn(2 * prefix + 9, kv_heads, dim, device="cuda", dtype=torch.float16) * 0.2
            pv = torch.randn_like(pk) * 0.2
            q = torch.randn(total, heads, dim, device="cuda", dtype=torch.float16) * 0.2
            k = torch.randn(total, kv_heads, dim, device="cuda", dtype=torch.float16) * 0.2
            v = torch.randn_like(k) * 0.2
            actual, legacy = torch.empty_like(q), torch.empty_like(q)
            context_attention_fwd_with_prefix(q, k, v, actual, starts, lengths, maximum,
                                              locs, prefix, pk, pv, prefix_request_ids=request_ids)
            context_attention_fwd_with_prefix(q, k, v, legacy, starts, lengths, maximum,
                                              locs[request_ids], prefix, pk, pv)
            torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
            private_locs = torch.zeros(3, maximum, device="cuda", dtype=torch.long)
            last = []
            for row, (start, length) in enumerate(zip(starts_list, lengths_list)):
                full_k = torch.cat((pk[locs[request_ids[row]]], k[start:start + length])).float().repeat_interleave(2, 1)
                full_v = torch.cat((pv[locs[request_ids[row]]], v[start:start + length])).float().repeat_interleave(2, 1)
                scores = torch.einsum("qhd,khd->hqk", q[start:start + length].float(), full_k) * dim ** -0.5
                visible = torch.arange(prefix + length, device="cuda")[None] <= (
                    prefix + torch.arange(length, device="cuda")[:, None])
                expected = torch.einsum("hqk,khd->qhd", scores.masked_fill(~visible[None], -float("inf")).softmax(-1), full_v)
                torch.testing.assert_close(actual[start:start + length].float(), expected, atol=0.002, rtol=0.01)
                private_locs[row, maximum - length:] = torch.arange(start, start + length, device="cuda")
                last.append(start + length - 1)
            query = q[last]
            decoded = token_attention_fwd_with_prefix(query, pk, pv, k, v, locs, private_locs,
                                                      lengths, maximum, prefix_request_ids=request_ids)
            direct = token_attention_fwd_with_prefix(query, pk, pv, k, v, locs[request_ids], private_locs,
                                                     lengths, maximum)
            torch.testing.assert_close(decoded, direct, atol=0, rtol=0)
            torch.testing.assert_close(decoded, actual[last], atol=0.002, rtol=0.01)

    def test_idle_retained_prefix_survives_adapter_reload_without_reconstruction(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.server.router.model_infer.post_process import sample

        args = SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                               adapter_base="dummy-lora-7b-rank-16", mode=LAYER_RETAINED, prefix_len=17,
                               query_len=8, output_len=5, batch_size=3, unique_loras=2, num_adapters=2,
                               rank=16, nonzero_prefix=True, retain_layers=2,
                               reconstruction_kernel="bounded", reconstruction_workers=1296)
        server, _, requests, _ = build_server(args)
        cache = server.lowra_prefix_cache
        outputs = []

        def capture(logits, batch):
            outputs[-1].append(logits.detach().clone())
            return sample(logits, batch)

        with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture):
            for round_id in range(2):
                outputs.append([])
                if round_id:
                    server.infer_adapter.load_adapters(server.adapters)
                server.exposed_add_batch("test", requests, "fp16")
                server.exposed_prefill_batch("test")
                server.exposed_decode_batch("test")
                server.exposed_remove_batch("test")
                server.infer_adapter.offload_adapters([])
                self.assertEqual(cache.retained_layer_builds, 4)
                self.assertEqual(len(cache.state_dict()["inactive_lora_dirs"]), 2)
                self.assertEqual(server.model.mem_manager.workspace_token_equiv, 0)
        for actual, expected in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
        cache.evict_many(list(cache.entries))
        self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)


if __name__ == "__main__":
    unittest.main()
