"""Retention accounting, lifecycle, attention addresses and real-layer equivalence."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from slora.common.mem_allocator import MemoryAllocator
from slora.common.lowra_retained import RetainedPrefixKVCache, RetainedLayerRuntime
from slora.common.prefix_modes import LAYER_RETAINED
from slora.server.router.lowra_fifo_req_queue import LowRAFIFOReqQueue


class RetentionAccountingTests(unittest.TestCase):
    def pool(self, size=80):
        return MemoryAllocator(size, size, torch.float16, 1, 16, 4,
                               dynamic_layer_workspace=True, device="cpu")

    def test_exact_packing_refcounts_and_private_storage_isolation(self):
        memory = self.pool()
        private = memory.alloc(5)
        for tensor in memory.key_buffer + memory.value_buffer:
            tensor[private] = 7
        cache = RetainedPrefixKVCache(7, 2, memory)
        cache.acquire_many(["a", "a", "b"])
        self.assertEqual(memory.can_use_mem_size, 80 - 5 - 2 * 4)
        self.assertEqual(cache.entries["a"].ref_count, 2)
        self.assertEqual(cache.entries["a"].prefix_indices.shape, (2, 7))
        for entry in cache.entries.values():
            memory.prefix_workspace_k[entry.prefix_indices] = 2
            memory.prefix_workspace_v[entry.prefix_indices] = -2
        workspace = memory.alloc_layer_workspace(14)
        self.assertEqual(memory.workspace_token_equiv, 4)
        memory.prefix_workspace_k[workspace] = 9
        for entry in cache.entries.values():
            self.assertTrue(torch.all(memory.prefix_workspace_k[entry.prefix_indices] == 2))
        for tensor in memory.key_buffer + memory.value_buffer:
            self.assertTrue(torch.all(tensor[private] == 7))
        memory.free_layer_workspace()
        cache.release_many(["a"])
        self.assertEqual(cache.entries["a"].ref_count, 1)
        cache.release_many(["a", "b"])
        self.assertEqual(cache.entries, {})
        self.assertEqual(memory.can_use_mem_size, 75)
        memory.free(private)
        self.assertEqual(memory.can_use_mem_size, 80)

    def test_failed_acquisition_rolls_back_and_does_not_increment_hits(self):
        memory = self.pool(8)
        cache = RetainedPrefixKVCache(7, 2, memory)
        cache.acquire_many(["a"])
        with self.assertRaisesRegex(Exception, "no enough pool space"):
            cache.acquire_many(["a", "b", "c"])
        self.assertEqual(set(cache.entries), {"a"})
        self.assertEqual(cache.entries["a"].ref_count, 1)
        self.assertEqual(memory.can_use_mem_size, 4)
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            cache.release_many(["a", "a"])
        self.assertEqual(cache.entries["a"].ref_count, 1)
        with self.assertRaisesRegex(RuntimeError, "active"):
            cache.evict_many(["a"])

    def test_reconstruction_only_on_first_k_misses_and_every_suffix_layer(self):
        for retained in (0, 2, 4):
            memory = self.pool()
            static = SimpleNamespace(layer_num=4, prefix_len=7, base_k=[torch.zeros(1)])
            cache = RetainedPrefixKVCache(7, retained, memory)
            runtime = RetainedLayerRuntime(static, memory, cache)
            dirs = ["b", "a", "a"]
            cache.acquire_many(dirs)

            def metadata(static, directories, adapter, locations):
                return SimpleNamespace(prefix_locs=locations)

            def reconstruct(static, adapter, meta, layer, keys, values):
                keys[meta.prefix_locs] = layer + 1
                values[meta.prefix_locs] = -layer - 1

            with patch("slora.common.lowra_retained.prepare_reconstruction_metadata", side_effect=metadata), \
                    patch("slora.common.lowra_retained.reconstruct_prefix_layer", side_effect=reconstruct) as call:
                for step in range(2):
                    frame = runtime.prepare(dirs, object())
                    for layer in range(4):
                        frame.before_layer(layer)
                        locations = frame.prefix_locations(layer)
                        keys, values = frame.buffers(layer)
                        self.assertTrue(torch.equal(locations[1], locations[2]))
                        self.assertTrue(torch.all(keys[locations] == layer + 1))
                        self.assertTrue(torch.all(values[locations] == -layer - 1))
                    frame.finish()
                    frame.finish()
                self.assertEqual(call.call_count, 4 + 4 - retained)
            self.assertEqual(cache.retained_layer_builds, retained * 2)
            self.assertEqual(cache.retained_layer_reuses, retained * 2)
            self.assertEqual(cache.serial_layer_builds, (4 - retained) * 2 * 2)
            self.assertEqual(memory.workspace_token_equiv, 0)
            cache.release_many(dirs)
            self.assertTrue(all(entry.ref_count == 0 for entry in cache.entries.values()))
            cache.evict_many(list(cache.entries))
            self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def test_fifo_accounts_for_retained_rounding_and_union_not_requests(self):
        def request(directory, ident):
            return SimpleNamespace(adapter_dir=directory, input_len=3, max_output_len=2,
                                   output_ids=[], aborted=False, request_id=ident)
        queue = LowRAFIFOReqQueue(28, 100, 10, 7, LAYER_RETAINED, layer_num=4, retain_layers=2)
        running = SimpleNamespace(reqs=[request("a", "old")])
        queue.append(request("a", "same"))
        queue.append(request("b", "new"))
        selected = queue.generate_new_batch(running, {"a": 1, "b": 1})
        self.assertEqual([req.request_id for req in selected.reqs], ["same"])
        self.assertEqual(queue.waiting_req_list[0].request_id, "new")
        with self.assertRaises(ValueError):
            RetainedPrefixKVCache(7, 5, self.pool())
        with self.assertRaises(ValueError):
            LowRAFIFOReqQueue(28, 100, 10, 7, LAYER_RETAINED, layer_num=4, retain_layers=5)

    def test_cli_forwards_retention(self):
        from benchmarks.lowra_saturation_exp import parse_args, server_command
        from pathlib import Path
        with patch("sys.argv", ["bench", "--mode", LAYER_RETAINED, "--retain-layers", "16",
                                "--retain-top-k", "2", "--reconstruction-kernel", "bounded",
                                "--reconstruction-workers", "1296", "--output-dir", "/tmp/retention-cli-test"]):
            args = parse_args()
        self.assertEqual(args.port, 8003)
        command = server_command(args, Path("/tmp/runtime.jsonl"))
        self.assertEqual(command[command.index("--lowra-retain-layers") + 1], "16")
        self.assertEqual(command[command.index("--lowra-retain-top-k") + 1], "2")
        self.assertEqual(command[command.index("--lowra-reconstruction-kernel") + 1], "bounded")
        self.assertEqual(command[command.index("--lowra-reconstruction-workers") + 1], "1296")

    def test_invalid_reconstruction_options_are_rejected(self):
        memory = self.pool()
        static = SimpleNamespace(layer_num=4, prefix_len=7)
        cache = RetainedPrefixKVCache(7, 4, memory, retain_top_k=1)
        for kwargs in (dict(reconstruction_kernel="unknown"), dict(reconstruction_workers=1296),
                       dict(reconstruction_kernel="bounded", reconstruction_workers=0),
                       dict(reconstruction_kernel="bounded", reconstruction_workers="1296")):
            with self.assertRaises(ValueError):
                RetainedLayerRuntime(static, memory, cache, **kwargs)

    def test_hot_selection_union_ties_demotion_and_no_startup_allocation(self):
        memory = self.pool(28)
        cache = RetainedPrefixKVCache(7, 4, memory, retain_top_k=2)
        cache.acquire_many(["a", "a", "b", "c"])
        self.assertEqual(memory.can_use_mem_size, 28)
        cache.rebalance()
        self.assertEqual(cache.state_dict()["cached_lora_dirs"], {"a", "b"})
        self.assertEqual(memory.can_use_mem_size, 14)
        self.assertEqual(cache.batch_unique_hits, 0)
        cache.entries["a"].ready = True
        cache.entries["b"].ready = True
        memory.prefix_workspace_k[cache.entries["a"].prefix_indices] = 11
        cache.acquire_many(["c", "c"])
        self.assertEqual(cache.batch_unique_hits, 0)
        cache.rebalance()
        self.assertEqual(cache.state_dict()["cached_lora_dirs"], {"a", "c"})
        self.assertTrue(torch.all(memory.prefix_workspace_k[cache.entries["a"].prefix_indices] == 11))
        self.assertEqual(cache.entries["b"].allocated_slots, 0)
        self.assertEqual(memory.can_use_mem_size, 14)
        cache.release_many(["c", "c"])
        cache.rebalance()
        self.assertEqual(cache.state_dict()["cached_lora_dirs"], {"a", "c"})
        cache.release_many(["a", "a", "b", "c"])
        self.assertEqual(cache.state_dict()["inactive_lora_dirs"], {"a"})
        cache.evict_many(list(cache.entries))
        self.assertEqual(memory.can_use_mem_size, 28)

    def test_mixed_hot_cold_addresses_and_partial_layers(self):
        for retained in (0, 2, 4):
            for top_k in (0, 1, 2, 10):
                memory = self.pool()
                static = SimpleNamespace(layer_num=4, prefix_len=7, base_k=[torch.zeros(1)])
                cache = RetainedPrefixKVCache(7, retained, memory, retain_top_k=top_k)
                runtime = RetainedLayerRuntime(static, memory, cache)
                dirs = ["b", "a", "a", "c"]
                cache.acquire_many(dirs)

                def metadata(static, directories, adapter, locations):
                    return SimpleNamespace(prefix_locs=locations, directories=directories)

                def reconstruct(static, adapter, meta, layer, keys, values):
                    for directory, locations in zip(meta.directories, meta.prefix_locs):
                        keys[locations] = (ord(directory) - 96) * 10 + layer
                        values[locations] = -(ord(directory) - 96) * 10 - layer

                with patch("slora.common.lowra_retained.prepare_reconstruction_metadata", side_effect=metadata), \
                        patch("slora.common.lowra_retained.reconstruct_prefix_layer", side_effect=reconstruct):
                    for _ in range(2):
                        frame = runtime.prepare(dirs, object())
                        for layer in range(4):
                            frame.before_layer(layer)
                            for directory, locations in zip(dirs, frame.prefix_locations(layer)):
                                keys, values = frame.buffers(layer)
                                self.assertTrue(torch.all(keys[locations] == (ord(directory) - 96) * 10 + layer))
                                self.assertTrue(torch.all(values[locations] == -(ord(directory) - 96) * 10 - layer))
                        frame.finish()
                hot = min(top_k, 3)
                self.assertEqual(cache.retained_layer_builds, hot * retained)
                self.assertEqual(cache.retained_layer_reuses, hot * retained)
                self.assertEqual(cache.serial_layer_builds, 2 * (3 * 4 - hot * retained))
                scratch_count = 3 if retained < 4 else 3 - hot
                self.assertEqual(memory.peak_workspace_token_equiv, (scratch_count * 7 + 3) // 4)
                cache.release_many(dirs)
                self.assertTrue(all(entry.ref_count == 0 for entry in cache.entries.values()))
                cache.evict_many(list(cache.entries))
                self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def test_top_k_admission_charges_hot_layers_and_cold_workspace(self):
        for top_k in (0, 1, 2, 10):
            for retained in (0, 2, 4):
                unique, prefix, layers = 3, 7, 4
                hot = min(unique, top_k)
                scratch = unique if retained < layers else unique - hot
                prefix_slots = hot * ((retained * prefix + layers - 1) // layers)
                prefix_slots += (scratch * prefix + layers - 1) // layers
                # Three requests each reserve private 3+2 tokens and rank-1 adapter 4 slots.
                exact = 3 * (3 + 2 + 4) + prefix_slots
                for budget in (exact, exact + 1):
                    queue = LowRAFIFOReqQueue(budget, 100, 20, prefix, LAYER_RETAINED,
                                             layer_num=layers, retain_layers=retained,
                                             retain_top_k=top_k)
                    for directory in ("a", "b", "c"):
                        queue.append(SimpleNamespace(adapter_dir=directory, input_len=3, max_output_len=2,
                                                     output_ids=[], aborted=False, request_id=directory))
                    batch = queue.generate_new_batch(None, {"a": 1, "b": 1, "c": 1})
                    self.assertEqual(len(batch.reqs), 3 if budget > exact else 2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RetentionServingTests(unittest.TestCase):
    def test_bounded_values_promotions_reuse_and_workspace_isolation(self):
        from slora.common.lowra_static import LowRAStaticPrefixStore

        torch.manual_seed(42)
        layers, prefix, heads, dim = 3, 257, 2, 128
        memory = MemoryAllocator(1600, 1600, torch.float16, heads, dim, layers,
                                 dynamic_layer_workspace=True)
        ranges = {"a": 16, "b": 8, "c": 16}
        static = LowRAStaticPrefixStore(prefix, list(ranges), ranges, layers, heads, heads, dim)
        for tensor in static.base_k + static.base_v + static.mini_k + static.mini_v:
            tensor.normal_(std=0.1)
        adapter_locs = memory.alloc(160)[torch.randperm(160, device="cuda")]
        for tensor in memory.value_buffer:
            tensor[adapter_locs] = torch.randn(160, heads, dim, device="cuda", dtype=torch.float16) * 0.1
        adapter = SimpleNamespace(mem_manager=memory, idx_map={"a": 0, "b": 1, "c": 2},
                                  a_loc=adapter_locs, a_start=torch.tensor([0, 64, 96], device="cuda"),
                                  a_len=torch.tensor([64, 32, 64], device="cuda"),
                                  a_scaling=torch.tensor([0.25, 0.5, 1.75], device="cuda", dtype=torch.float16))
        original_b = [buffer[adapter_locs].clone() for buffer in memory.value_buffer]
        cache = RetainedPrefixKVCache(prefix, layers, memory, retain_top_k=1)
        runtime = RetainedLayerRuntime(static, memory, cache, reconstruction_kernel="bounded",
                                       reconstruction_workers=1296)
        dirs = ["a", "a", "b", "c"]
        cache.acquire_many(dirs)
        from slora.common.lowra_layer_runtime import reconstruct_prefix_layer_bounded
        calls = []

        def reconstruct(*args, **kwargs):
            calls.append((args[3], kwargs["workers"]))
            return reconstruct_prefix_layer_bounded(*args, **kwargs)

        with patch("slora.common.lowra_retained.reconstruct_prefix_layer_bounded", new=reconstruct), \
                patch("slora.common.lowra_retained.reconstruct_prefix_layer") as legacy:
            for step in range(4):
                if step == 2:
                    cache.acquire_many(["b", "b"])
                    dirs += ["b", "b"]
                elif step == 3:
                    cache.release_many(["b", "b"])
                    dirs = dirs[:-2]
                count_before = len(calls)
                frame = runtime.prepare(dirs, adapter)
                self.assertEqual(cache.state_dict()["cached_lora_dirs"], {"b" if step == 2 else "a"})
                for layer in range(layers):
                    frame.before_layer(layer)
                    keys, values = frame.buffers(layer)
                    for directory, locations in zip(dirs, frame.prefix_locations(layer)):
                        idx = adapter.idx_map[directory]
                        rank = ranges[directory]
                        start, end = static.get_lora_rank_range_by_dir(directory)
                        packed_start = (0, 64, 96)[idx]
                        for actual, bases, minis, offset in (
                            (keys[locations], static.base_k, static.mini_k, 1),
                            (values[locations], static.base_v, static.mini_v, 2),
                        ):
                            b = original_b[layer][packed_start + offset * rank:packed_start + (offset + 1) * rank]
                            expected = bases[layer].float().view(prefix, -1) + (
                                minis[layer][:, start:end].float() @ b.float().reshape(-1, rank).T
                            ) * adapter.a_scaling[idx].float()
                            torch.testing.assert_close(actual.float().view(prefix, -1), expected,
                                                       atol=0.002, rtol=0.01)
                frame.finish()
                self.assertEqual(len(calls) - count_before, layers if step == 1 else 2 * layers)
                if step == 1:
                    self.assertGreater(runtime.metadata_reuses, 0)
                self.assertTrue(all(entry[3].prefix_locs is None for entry in runtime._metadata_cache.values()))
                self.assertEqual(memory.workspace_token_equiv, 0)
            legacy.assert_not_called()
        self.assertTrue(all(workers == 1296 for _, workers in calls))
        self.assertEqual((runtime.last_reconstruction_kernel, runtime.last_reconstruction_workers), ("bounded", 1296))
        self.assertEqual(cache.promotions, 3)
        self.assertEqual(cache.demotions, 2)
        self.assertEqual(cache.retained_layer_builds, 3 * layers)
        self.assertEqual(cache.retained_layer_reuses, layers)
        cache.release_many(dirs)
        cache.evict_many(list(cache.entries))
        for layer, buffer in enumerate(memory.value_buffer):
            torch.testing.assert_close(buffer[adapter_locs], original_b[layer], atol=0, rtol=0)
        memory.free(adapter_locs)
        self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def args(self, retained, mode=LAYER_RETAINED):
        return SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                               adapter_base="dummy-lora-7b-rank-16", mode=mode, prefix_len=17,
                               query_len=8, output_len=5, batch_size=3, unique_loras=2,
                               num_adapters=2, rank=16, nonzero_prefix=True, retain_layers=retained)

    def test_nonzero_logits_match_serial_and_full_reconstruction_for_all_retention_counts(self):
        from benchmarks.lowra_layer_profile import build_server, run_round
        from slora.server.router.model_infer.post_process import sample
        reference = None
        for mode, retained, kernel in (("layer_serial", 0, "auto"), ("full_reconstruct", 0, "auto"),
                                       (LAYER_RETAINED, 0, "auto"), (LAYER_RETAINED, 1, "auto"),
                                       (LAYER_RETAINED, 2, "auto"), (LAYER_RETAINED, 3, "auto"),
                                       (LAYER_RETAINED, 0, "bounded"), (LAYER_RETAINED, 1, "bounded"),
                                       (LAYER_RETAINED, 2, "bounded"), (LAYER_RETAINED, 3, "bounded")):
            args = self.args(retained, mode)
            args.reconstruction_kernel = kernel
            args.reconstruction_workers = 1296 if kernel == "bounded" else None
            server, profiler, requests, _ = build_server(args)
            requests[0]["input_id"] = [10] * 3
            requests[1]["input_id"] = [10] * 5
            collected = []

            def capture(logits, batch):
                collected.append(logits.detach().clone())
                return sample(logits, batch)

            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture):
                run_round(server, profiler, requests, args.output_len)
            torch.cuda.synchronize()
            if reference is None:
                reference = collected
            else:
                for actual, expected in zip(collected, reference):
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
            if mode == LAYER_RETAINED:
                cache = server.lowra_prefix_cache
                self.assertEqual(cache.retained_layer_builds, 2 * retained)
                self.assertEqual(cache.serial_layer_builds, 2 * (3 - retained) * args.output_len)
                self.assertFalse(cache.entries)
                self.assertEqual(server.model.mem_manager.workspace_token_equiv, 0)

    def test_merge_filter_cold_adapter_and_final_release(self):
        from benchmarks.lowra_layer_profile import build_server
        args = self.args(2)
        args.batch_size = 4
        server, profiler, requests, _ = build_server(args)
        server.exposed_add_batch("old", [requests[0], requests[2]], "fp16")
        server.exposed_prefill_batch("old")
        cache = server.lowra_prefix_cache
        self.assertEqual(cache.retained_layer_builds, 2)
        directory = requests[0]["adapter_dir"]
        locations = cache.entries[directory].prefix_indices.clone()
        keys = server.model.mem_manager.prefix_workspace_k[locations].clone()
        server.exposed_decode_batch("old")
        same = dict(requests[0], request_id="same")
        server.exposed_add_batch("mini", [same, requests[1]], "fp16")
        server.exposed_prefill_batch("mini")
        self.assertEqual(cache.retained_layer_builds, 4)
        self.assertEqual(cache.entries[directory].ref_count, 3)
        torch.testing.assert_close(server.model.mem_manager.prefix_workspace_k[locations], keys, atol=0, rtol=0)
        server.exposed_merge_batch("old", "mini")
        server.exposed_decode_batch("old")
        server.exposed_filter_batch("old", ["1", "same"])
        self.assertEqual(cache.entries[directory].ref_count, 1)
        server.exposed_decode_batch("old")
        server.exposed_filter_batch("old", ["1"])
        self.assertEqual(cache.entries[directory].ref_count, 0)
        torch.testing.assert_close(cache.entries[directory].prefix_indices, locations)
        server.exposed_decode_batch("old")
        self.assertEqual(cache.retained_layer_builds, 4)
        server.exposed_remove_batch("old")
        server.infer_adapter.offload_adapters([])
        cache.evict_many(list(cache.entries))
        self.assertFalse(cache.entries)
        self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)

    def test_hot_promotions_merge_filter_match_serial_nonzero_logits(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.server.router.model_infer.post_process import sample
        reference = None
        for mode, layers, top_k, kernel in (
            ("layer_serial", 0, None, "auto"), (LAYER_RETAINED, 1, 1, "auto"),
            (LAYER_RETAINED, 3, 1, "auto"), (LAYER_RETAINED, 1, 2, "auto"),
            (LAYER_RETAINED, 3, 2, "auto"), (LAYER_RETAINED, 1, 1, "bounded"),
            (LAYER_RETAINED, 3, 1, "bounded"), (LAYER_RETAINED, 1, 2, "bounded"),
            (LAYER_RETAINED, 3, 2, "bounded"),
        ):
            args = self.args(layers, mode)
            args.reconstruction_kernel = kernel
            args.reconstruction_workers = 1296 if kernel == "bounded" else None
            args.retain_top_k = top_k
            args.unique_loras = args.num_adapters = 3
            args.batch_size = 7
            server, _, requests, _ = build_server(args)
            collected = []

            def capture(logits, batch):
                collected.append(logits.detach().clone())
                return sample(logits, batch)

            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture):
                old = [dict(requests[0], request_id="a1"), dict(requests[0], request_id="a2"),
                       dict(requests[1], request_id="b1"), dict(requests[2], request_id="c1")]
                server.exposed_add_batch("old", old, "fp16")
                server.exposed_prefill_batch("old")
                server.exposed_decode_batch("old")
                mini = [dict(requests[1], request_id="b2"), dict(requests[1], request_id="b3")]
                server.exposed_add_batch("mini", mini, "fp16")
                server.exposed_prefill_batch("mini")
                if mode == LAYER_RETAINED:
                    self.assertEqual(server.lowra_prefix_cache.state_dict()["cached_lora_dirs"],
                                     {requests[1]["adapter_dir"]} if top_k == 1 else
                                     {requests[0]["adapter_dir"], requests[1]["adapter_dir"]})
                server.exposed_merge_batch("old", "mini")
                server.exposed_decode_batch("old")
                server.exposed_filter_batch("old", ["a2", "b1", "a1", "c1"])
                server.exposed_decode_batch("old")
                if mode == LAYER_RETAINED:
                    cache = server.lowra_prefix_cache
                    self.assertEqual(cache.state_dict()["cached_lora_dirs"],
                                     {requests[0]["adapter_dir"]} if top_k == 1 else
                                     {requests[0]["adapter_dir"], requests[1]["adapter_dir"]})
                    self.assertEqual(cache.promotions, 3 if top_k == 1 else 2)
                    self.assertEqual(cache.demotions, 2 if top_k == 1 else 0)
                server.exposed_remove_batch("old")
                server.infer_adapter.offload_adapters([])
                if server.lowra_prefix_cache is not None:
                    server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
            if reference is None:
                reference = collected
            else:
                self.assertEqual(len(collected), len(reference))
                for actual, expected in zip(collected, reference):
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
                self.assertFalse(server.lowra_prefix_cache.entries)
            self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)


if __name__ == "__main__":
    unittest.main()
