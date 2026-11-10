"""Fixed hot-prefix protection across idle gaps, pressure and real attention."""

from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from slora.common.full_reconstruct import FullReconstructPrefixKVCache
from slora.common.lowra_retained import RetainedPrefixKVCache, RetainedLayerRuntime
from slora.common.mem_allocator import MemoryAllocator
from slora.common.prefix_modes import FULL_RECONSTRUCT, LAYER_RETAINED
from slora.server.router.full_reprefill_req_queue import FullReprefillReqQueue
from slora.server.router.lowra_fifo_req_queue import LowRAFIFOReqQueue
from test_full_reprefill import CPUPool, request


class PinnedAccountingTests(unittest.TestCase):
    def test_full_cache_cannot_evict_idle_pin_but_cold_lru_unchanged(self):
        pool = CPUPool(10)
        cache = FullReconstructPrefixKVCache(5, pool, pinned_lora_dirs=["hot"])
        cache.acquire_many([dict(adapter_dir=d) for d in ("hot", "cold")], lambda *args: None)
        hot = cache.entries["hot"].prefix_indices
        cache.release_many(["hot", "cold"])
        self.assertFalse(cache.evict("hot"))
        cache.acquire_many([dict(adapter_dir="new")], lambda *args: None)
        self.assertNotIn("cold", cache.entries)
        self.assertIs(cache.entries["hot"].prefix_indices, hot)
        cache.release_many(["new"])
        cache.acquire_many([dict(adapter_dir="hot")], lambda *args: self.fail("pin rebuilt"))
        self.assertEqual(cache.prefix_build_counts["hot"], 1)
        cache.release_many(["hot"])
        self.assertTrue(cache.evict_one())
        self.assertFalse(cache.evict_one())
        self.assertEqual(set(cache.entries), {"hot"})
        self.assertEqual(cache.clear_metadata_if_inactive(), 1)

    def test_full_admission_counts_idle_pin_and_never_plans_its_eviction(self):
        for budget in (50, 54):
            queue = FullReprefillReqQueue(budget, 100, 10, 20)
            queue.update_prefix_cache_state(dict(cached_lora_dirs={"hot"},
                                                inactive_lora_dirs={"hot"}, pinned_lora_dirs=["hot"]))
            queue.append(request("cold", "1"))
            batch = queue.generate_new_batch(None, {"cold": 1})
            if budget == 50:
                self.assertIsNone(batch)
            else:
                self.assertEqual(len(batch.reqs), 1)
                self.assertEqual(batch.lowra_evict_prefix_lora_dirs, [])

    def test_layer_pin_survives_idle_and_cold_scratch_without_rebuild(self):
        for layers in (2, 4):
            pool = MemoryAllocator(40, 40, torch.float16, 1, 16, 4,
                                   dynamic_layer_workspace=True, device="cpu")
            static = SimpleNamespace(layer_num=4, prefix_len=7, base_k=[torch.zeros(1)])
            cache = RetainedPrefixKVCache(7, layers, pool, pinned_lora_dirs=["hot"])
            runtime = RetainedLayerRuntime(static, pool, cache)

            def metadata(static, dirs, adapter, locations):
                return SimpleNamespace(prefix_locs=locations, dirs=dirs)

            def rebuild(static, adapter, meta, layer, keys, values):
                for directory, locations in zip(meta.dirs, meta.prefix_locs):
                    keys[locations] = (100 if directory == "hot" else 200) + layer
                    values[locations] = -keys[locations]

            with patch("slora.common.lowra_retained.prepare_reconstruction_metadata", side_effect=metadata), \
                    patch("slora.common.lowra_retained.reconstruct_prefix_layer", side_effect=rebuild):
                for dirs in (["hot", "cold"], ["cold", "other"], ["hot"]):
                    cache.acquire_many(dirs)
                    frame = runtime.prepare(dirs, object())
                    for layer in range(4):
                        frame.before_layer(layer)
                        for directory, loc in zip(dirs, frame.prefix_locations(layer)):
                            expected = (100 if directory == "hot" else 200) + layer
                            self.assertTrue(torch.all(pool.prefix_workspace_k[loc] == expected))
                    frame.finish()
                    cache.release_many(dirs)
                    self.assertEqual(set(cache.entries), {"hot"})
                    self.assertEqual(cache.entries["hot"].ref_count, 0)
                    expected = torch.arange(layers)[:, None] + 100
                    self.assertTrue(torch.all(pool.prefix_workspace_k[
                        cache.entries["hot"].prefix_indices][:, :, 0, 0] == expected))
            self.assertEqual(cache.retained_build_counts, {"hot": 1})
            self.assertEqual(cache.retained_layer_builds, layers)
            self.assertEqual(cache.clear_metadata_if_inactive(), 1)
            pool.free_all()
            self.assertEqual(pool.can_use_mem_size, 40)

    def test_layer_admission_counts_idle_pin_without_startup_reservation(self):
        for built in (False, True):
            for budget in (18, 19):
                queue = LowRAFIFOReqQueue(budget, 100, 10, 7, LAYER_RETAINED,
                                         layer_num=4, retain_layers=4, pinned_lora_dirs=["hot"])
                queue.update_prefix_cache_state(dict(cached_lora_dirs={"hot"} if built else set()))
                queue.append(SimpleNamespace(adapter_dir="cold", input_len=3, max_output_len=2,
                                             output_ids=[], aborted=False, request_id="cold"))
                batch = queue.generate_new_batch(None, {"cold": 1})
                # Private 5 + adapter 4 + cold scratch 2 + built idle hot prefix 7.
                self.assertEqual(batch is not None, not built or budget == 19)

    def test_cli_forwards_fixed_pinning_and_rejects_dynamic_mix(self):
        from benchmarks.lowra_saturation_exp import parse_args, server_command
        for mode in (FULL_RECONSTRUCT, LAYER_RETAINED):
            argv = ["bench", "--mode", mode, "--pin-hot-adapters", "1", "--output-dir", "/tmp/pin-cli"]
            if mode == LAYER_RETAINED:
                argv += ["--retain-layers", "32"]
            with patch("sys.argv", argv):
                args = parse_args()
            command = server_command(args, Path("/tmp/pin-stats.jsonl"))
            self.assertEqual(command[command.index("--lowra-pin-hot-adapters") + 1], "1")
        with patch("sys.argv", argv + ["--retain-top-k", "1"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PinnedServingTests(unittest.TestCase):
    def test_nonzero_logits_and_once_only_build_across_idle_gaps(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.server.router.model_infer.post_process import sample
        reference = None
        for mode, layers, pins in (("layer_serial", 0, 0), (FULL_RECONSTRUCT, 0, 1),
                                  (LAYER_RETAINED, 3, 1), (LAYER_RETAINED, 1, 1)):
            args = SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                                   adapter_base="dummy-lora-7b-rank-16", mode=mode, prefix_len=17,
                                   query_len=8, output_len=5, batch_size=3, unique_loras=2,
                                   num_adapters=2, rank=16, nonzero_prefix=True, retain_layers=layers,
                                   retain_top_k=None, pin_hot_adapters=pins)
            server, _, requests, _ = build_server(args)
            pool, cache = server.model.mem_manager, server.lowra_prefix_cache
            hot_dir = requests[0]["adapter_dir"]
            collected = []

            def capture(logits, batch):
                collected.append(logits.detach().clone())
                return sample(logits, batch)

            def hot_keys():
                indices = cache.entries[hot_dir].prefix_indices
                if mode == FULL_RECONSTRUCT:
                    return torch.stack([key[indices] for key in pool.key_buffer])
                return pool.prefix_workspace_k[indices]

            expected_keys = None
            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture):
                for step, indices in enumerate(([0, 1, 2], [1], [0])):
                    reqs = [dict(requests[i], request_id="%s-%s" % (step, j),
                                 input_id=[10] * (3 + i)) for j, i in enumerate(indices)]
                    server.exposed_add_batch("batch", reqs, "fp16")
                    server.exposed_prefill_batch("batch")
                    server.exposed_decode_batch("batch")
                    server.exposed_decode_batch("batch")
                    server.exposed_remove_batch("batch")
                    if pins:
                        self.assertEqual(cache.entries[hot_dir].ref_count, 0)
                        if expected_keys is None:
                            expected_keys = hot_keys().clone()
                        else:
                            torch.testing.assert_close(hot_keys(), expected_keys, atol=0, rtol=0)
                if pins:
                    counts = cache.prefix_build_counts if mode == FULL_RECONSTRUCT else cache.retained_build_counts
                    self.assertEqual(counts[hot_dir], 1)
                    self.assertFalse(cache.evict_many([hot_dir]))
            if reference is None:
                reference = collected
            else:
                for actual, expected in zip(collected, reference):
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
            stats = server.exposed_reset_lowra_runtime()
            self.assertEqual(stats["free_slots_after"], pool.tot_size)
            if cache:
                self.assertFalse(cache.entries)


if __name__ == "__main__":
    unittest.main()
