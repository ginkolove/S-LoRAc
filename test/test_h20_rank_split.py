"""Check both H20 cutoffs, including mixed batches and the full-cache lifetime."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from benchmarks.h20_rank_split_runtime import RankSplitCache, RankSplitRuntime
from slora.common.mem_allocator import MemoryAllocator


class RankSplitTests(unittest.TestCase):
    def test_mixed_and_homogeneous_batches(self):
        for cutoff in (16, 32):
            with self.subTest(cutoff=cutoff):
                self.check_batches(cutoff)

    def check_batches(self, cutoff):
        ranks = {"a": 8, "b": 16, "c": 32, "d": 64}
        layers, prefix = 3, 5
        memory = MemoryAllocator(40, 40, torch.float32, 1, 1, layers,
                                 dynamic_layer_workspace=True, device="cpu")
        static = SimpleNamespace(lora_ranks=ranks, layer_num=layers, prefix_len=prefix,
                                 base_k=memory.key_buffer)
        cache = RankSplitCache(static, memory, cutoff)
        runtime = RankSplitRuntime(static, memory, cache, "rank_split", 1296)
        dirs = ["d", "b", "a", "c", "d", "b", "c"]
        full = {d for d in dirs if ranks[d] > cutoff}
        low = set(dirs) - full
        full_calls, serial_calls = [], []

        def writer(calls):
            def reconstruct(static, adapter, metadata, layer, keys, values, **kwargs):
                names, locs = metadata
                calls.append((layer, tuple(names)))
                for name, loc in zip(names, locs):
                    expected = ranks[name] + layer * 100 + torch.arange(prefix) / 10
                    keys[loc, 0, 0], values[loc, 0, 0] = expected, -expected
            return reconstruct

        with patch("slora.common.full_reconstruct.prepare_reconstruction_metadata",
                   side_effect=lambda static, ds, adapter, locs: (ds, locs)), \
                patch("slora.common.full_reconstruct.reconstruct_prefix_layer", writer(full_calls)), \
                patch.object(runtime.serial, "_prepare_metadata",
                             side_effect=lambda ds, adapter, locs: (ds, locs)), \
                patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer_bounded",
                      writer(serial_calls)):
            requests = [dict(adapter_dir=d) for d in dirs]
            runtime.acquire_many(requests, None)
            self.assertEqual(set(cache.entries), full)
            self.assertEqual(cache.entries["d"].ref_count, 2)
            for batch in (dirs, ["c", "d", "a", "a"], ["d", "d"], ["b", "a", "b"]):
                serial_calls.clear()
                frame = runtime.prepare(batch, None)
                expected_low = set(batch) & low
                self.assertEqual(memory.layer_workspace_tokens, len(expected_low) * prefix)
                for layer in range(layers):
                    frame.before_layer(layer)
                    frame.before_attention(layer)
                    k, v = frame.buffers(layer)
                    for directory, loc in zip(batch, frame.prefix_locations(layer)):
                        expected = ranks[directory] + layer * 100 + torch.arange(prefix) / 10
                        torch.testing.assert_close(k[loc, 0, 0], expected)
                        torch.testing.assert_close(v[loc, 0, 0], -expected)
                    frame.after_attention(layer)
                frame.finish()
                frame.finish()
                self.assertEqual(memory.tot_size - memory.can_use_mem_size, len(full) * prefix)
                self.assertEqual([set(names) for _, names in serial_calls],
                                 [expected_low] * (layers if expected_low else 0))
            cache.release_many(dirs)
            runtime.acquire_many(requests, None)
            self.assertEqual([set(names) for _, names in full_calls], [full] * layers)
            self.assertEqual(cache.access_stats()["idle_unpinned_hits"], len(full))
            cache.release_many(dirs)
            cache.evict_many(full)
            self.assertEqual(memory.tot_size, memory.can_use_mem_size)
