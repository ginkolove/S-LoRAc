import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from slora.common.mem_allocator import MemoryAllocator
from slora.server.router.lowra_fifo_req_queue import LowRAFIFOReqQueue


class AccountingTests(unittest.TestCase):
    def test_common_bounded_override_has_equal_serial_and_parallel_configuration(self):
        from slora.common.lowra_layer_runtime import LowRALayerRuntime

        for mode in ("layer_serial", "layer_prefetch"):
            memory = MemoryAllocator(100, 100, torch.float16, 1, 16, 4,
                                     dynamic_layer_workspace=True, device="cpu")
            static = SimpleNamespace(prefix_len=7, base_k=[torch.zeros(1)],
                                     get_lora_rank_range_by_dir=lambda _: (0, 2))
            adapter = SimpleNamespace(idx_map={"a": 0}, a_loc=torch.arange(8),
                                      a_start=torch.tensor([0]), a_len=torch.tensor([8]),
                                      a_scaling=torch.tensor([0.5]))
            with patch("torch.cuda.Stream"), \
                    patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer_bounded") as bounded, \
                    patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer") as legacy:
                runtime = LowRALayerRuntime(static, memory, mode, reconstruction_kernel="bounded",
                                            reconstruction_workers=1296)
                for _ in range(2):
                    frame = runtime.prepare(["a", "a"], adapter)
                    frame._reconstruct(0)
                    self.assertEqual(runtime.last_reconstruction_kernel, "bounded")
                    self.assertEqual(runtime.last_reconstruction_workers, 1296)
                    self.assertEqual(memory.workspace_token_equiv, 2)
                    frame.finish()
                self.assertEqual(bounded.call_count, 2)
                self.assertEqual(bounded.call_args.kwargs["workers"], 1296)
                legacy.assert_not_called()
                self.assertEqual((runtime.metadata_builds, runtime.metadata_reuses), (1, 1))
                self.assertEqual(memory.can_use_mem_size, 100)
                self.assertEqual(runtime.reconstruct_stream is None, mode == "layer_serial")

    def test_invalid_common_reconstruction_options_are_rejected(self):
        from slora.common.lowra_layer_runtime import LowRALayerRuntime

        for kwargs in (dict(reconstruction_kernel="unknown"),
                       dict(reconstruction_workers=1296),
                       dict(reconstruction_kernel="bounded", reconstruction_workers=0),
                       dict(reconstruction_kernel="bounded", prefetch_reconstruction="split")):
            with self.assertRaises(ValueError):
                LowRALayerRuntime(None, None, "layer_serial", **kwargs)
        with self.assertRaises(ValueError):
            LowRALayerRuntime(None, None, "full", reconstruction_kernel="bounded")

    def test_auto_bounded_workers_are_cached_without_changing_workspace_budget(self):
        from slora.common.lowra_layer_runtime import LowRALayerRuntime

        memory = MemoryAllocator(100, 100, torch.float16, 1, 16, 4,
                                 dynamic_layer_workspace=True, device="cpu")
        static = SimpleNamespace(prefix_len=7, base_k=[torch.zeros(1)],
                                 get_lora_rank_range_by_dir=lambda _: (0, 2))
        adapter = SimpleNamespace(idx_map={"a": 0}, a_loc=torch.arange(8),
                                  a_start=torch.tensor([0]), a_len=torch.tensor([8]),
                                  a_scaling=torch.tensor([0.5]))
        with patch("torch.cuda.Stream"), \
                patch("slora.common.lowra_layer_runtime.select_prefetch_reconstruction", return_value="bounded"), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(multi_processor_count=108)) as properties:
            runtime = LowRALayerRuntime(static, memory, "layer_prefetch")
            for _ in range(2):
                frame = runtime.prepare(["a", "a"], adapter)
                self.assertEqual(frame.prefetch_reconstruction, "bounded")
                self.assertEqual(frame.prefetch_workers, 1296)
                self.assertEqual(memory.workspace_token_equiv, 2)
                self.assertTrue(torch.equal(frame.prefix_b_loc[0], frame.prefix_b_loc[1]))
                frame.finish()
                self.assertEqual(memory.can_use_mem_size, 100)
            properties.assert_called_once()
        for bad in (0, -1, 1.5, "648"):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                LowRALayerRuntime(None, None, "layer_serial", prefetch_workers=bad)

    def test_prefetch_kernel_policy_is_limited_to_profiled_shapes(self):
        from slora.common.lowra_layer_runtime import select_prefetch_reconstruction

        for prefix, count, embed, rank, capability, expected in (
            (4608, 20, 4096, 16, (8, 0), "bounded"),
            (4608, 8, 4096, 16, (8, 0), "bounded"),
            (8193, 8, 4096, 16, (8, 0), "bounded"),
            (4608, 1, 4096, 16, (8, 0), "fused"),
            (257, 20, 4096, 16, (8, 0), "fused"),
            (4608, 20, 4096, 32, (8, 0), "fused"),
            (4608, 20, 5120, 16, (8, 0), "fused"),
            (4608, 20, 4096, 16, (9, 0), "fused"),
        ):
            with self.subTest(prefix=prefix, count=count, embed=embed, rank=rank, capability=capability):
                static = SimpleNamespace(prefix_len=prefix, kv_embed_dim_k=embed, kv_embed_dim_v=embed)
                metadata = SimpleNamespace(prefix_locs=torch.empty(count, prefix),
                                           k_b_locs=torch.empty(count, rank))
                self.assertEqual(select_prefetch_reconstruction(static, metadata, capability), expected)

    def test_prefetch_split_dispatch_does_not_change_serial_dispatch(self):
        from slora.common.lowra_layer_runtime import LowRALayerFrame

        for mode in ("layer_serial", "layer_prefetch"):
            for kernel in ("split", "fused", "bounded"):
                runtime = SimpleNamespace(mode=mode, profiler=None, static_store=object())
                frame = LowRALayerFrame(runtime, object(), object(), None, 1)
                frame.prefetch_reconstruction = kernel
                with patch.object(frame, "buffers", return_value=(None, None)), \
                        patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer") as separate, \
                        patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer_fused_kv") as fused, \
                        patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer_bounded") as bounded:
                    frame._reconstruct(0)
                    use_fused = mode == "layer_prefetch" and kernel == "fused"
                    use_bounded = mode == "layer_prefetch" and kernel == "bounded"
                    self.assertEqual(fused.call_count, int(use_fused))
                    self.assertEqual(bounded.call_count, int(use_bounded))
                    self.assertEqual(separate.call_count, int(not (use_fused or use_bounded)))

    def test_prefetch_waits_at_attention_and_reuses_event_pairs(self):
        from slora.common.lowra_layer_runtime import LowRALayerFrame

        compute, side = Mock(), Mock()
        consumed, ready = Mock(), Mock()
        runtime = SimpleNamespace(mode="layer_prefetch", profiler=None, events={},
                                  reconstruct_stream=side, mem_manager=Mock(),
                                  static_store=SimpleNamespace(layer_num=1))
        metadata = SimpleNamespace(locations=Mock())
        with patch("torch.cuda.current_stream", return_value=compute), \
                patch("torch.cuda.Event", side_effect=[consumed, ready]) as create_event, \
                patch("torch.cuda.stream", return_value=nullcontext()), \
                patch.object(LowRALayerFrame, "_reconstruct") as reconstruct:
            for _ in range(3):
                frame = LowRALayerFrame(runtime, object(), metadata, None, 1)
                compute.reset_mock()
                frame.before_layer(0)
                compute.wait_event.assert_not_called()
                frame.before_attention(0)
                compute.wait_event.assert_called_once_with(ready)
                frame.after_attention(0)
                frame.finish()
            self.assertEqual(create_event.call_count, 2)
            self.assertEqual(reconstruct.call_count, 3)

    def test_metadata_cache_invalidates_mapping_but_never_caches_workspace(self):
        from slora.common.lowra_layer_runtime import LowRALayerRuntime

        memory = MemoryAllocator(100, 100, torch.float16, 1, 16, 4,
                                 dynamic_layer_workspace=True, device="cpu")
        ranges = {"a": (0, 2), "b": (2, 3)}
        static = SimpleNamespace(prefix_len=7, base_k=[torch.zeros(1)],
                                 get_lora_rank_range_by_dir=ranges.__getitem__)
        adapter = SimpleNamespace(idx_map={"a": 0, "b": 1}, a_loc=torch.arange(12),
                                  a_start=torch.tensor([0, 8]), a_len=torch.tensor([8, 4]),
                                  a_scaling=torch.tensor([0.5, 0.25]))
        with patch("torch.cuda.Stream"):
            runtime = LowRALayerRuntime(static, memory, "layer_prefetch")
        frame = runtime.prepare(["a", "a", "b"], adapter)
        old_metadata = frame.metadata
        old_locations = frame.prefix_b_loc.clone()
        frame.finish()
        private = memory.alloc(11)
        frame = runtime.prepare(["a", "b", "b"], adapter)
        self.assertIs(frame.metadata.k_b_locs, old_metadata.k_b_locs)
        self.assertFalse(torch.equal(frame.prefix_b_loc[0], old_locations[0]))
        self.assertTrue(torch.equal(frame.prefix_b_loc[1], frame.prefix_b_loc[2]))
        self.assertIsNone(runtime._metadata_cache[3].prefix_locs)
        self.assertEqual((runtime.metadata_builds, runtime.metadata_reuses), (1, 1))
        frame.finish()
        adapter.a_scaling[0] = 2
        frame = runtime.prepare(["a", "b"], adapter)
        self.assertEqual(frame.metadata.scalings.tolist(), [2, 0.25])
        frame.finish()
        adapter.a_loc = adapter.a_loc + 20
        frame = runtime.prepare(["a", "b"], adapter)
        self.assertEqual(frame.metadata.k_b_locs[0].tolist(), [22, 23])
        frame.finish()
        frame = runtime.prepare(["b", "a"], adapter)
        self.assertEqual(frame.metadata.rank_lens.tolist(), [1, 2])
        frame.finish()
        self.assertEqual(runtime.metadata_builds, 4)
        memory.free(private)
        self.assertEqual(memory.can_use_mem_size, 100)

    def test_prefetch_failure_fences_side_stream_before_workspace_release(self):
        from slora.common.lowra_layer_runtime import LowRALayerFrame

        memory = Mock()
        compute, reconstruct = Mock(), Mock()
        consumed, ready = Mock(), Mock()
        metadata = SimpleNamespace(locations=Mock())
        runtime = SimpleNamespace(mode="layer_prefetch", mem_manager=memory, profiler=None,
                                  reconstruct_stream=reconstruct, events={},
                                  static_store=SimpleNamespace(layer_num=3))
        frame = LowRALayerFrame(runtime, object(), metadata, None, 1)
        with patch("torch.cuda.current_stream", return_value=compute), \
                patch("torch.cuda.Event", side_effect=[consumed, ready]), \
                patch("torch.cuda.stream", return_value=nullcontext()), \
                patch.object(frame, "_reconstruct", side_effect=RuntimeError("partial launch")):
            with self.assertRaisesRegex(RuntimeError, "partial launch"):
                frame.after_attention(0)
            consumed.record.assert_called_once_with(compute)
            reconstruct.wait_event.assert_called_once_with(consumed)
            ready.record.assert_called_once_with(reconstruct)
            metadata.locations.record_stream.assert_called_once_with(reconstruct)
            frame.finish()
            frame.finish()
            compute.wait_event.assert_called_once_with(ready)
            memory.free_layer_workspace.assert_called_once_with()

    def test_serial_and_prefetch_have_identical_admission_budget(self):
        selected = []
        for mode in ("layer_serial", "layer_prefetch"):
            queue = LowRAFIFOReqQueue(80, 100, 10, 40, mode, layer_num=4)
            for i, directory in enumerate(("a", "a", "b", "c", "b", "c")):
                queue.append(SimpleNamespace(adapter_dir=directory, input_len=10, max_output_len=4,
                                             output_ids=[], aborted=False, request_id=str(i)))
            batch = queue.generate_new_batch(None, {"a": 1, "b": 1, "c": 1})
            selected.append([req.request_id for req in batch.reqs])
        self.assertTrue(selected[0])
        self.assertEqual(selected[0], selected[1])

    def test_profile_memory_deduplicates_underlying_tensor_storage(self):
        from benchmarks.lowra_layer_profile import tensor_storage_bytes

        tensor = torch.arange(80, dtype=torch.float16)
        other = torch.zeros(10, dtype=torch.float32)
        self.assertEqual(tensor_storage_bytes([tensor[:5], tensor[20:], tensor, other]), 200)

    def test_only_prefetch_dispatches_to_fused_kv_reconstruction(self):
        from slora.common.lowra_layer_runtime import LowRALayerFrame

        memory = MemoryAllocator(10, 10, torch.float16, 1, 16, 1,
                                 dynamic_layer_workspace=True, device="cpu")
        for mode in ("full", "layer_serial", "layer_prefetch"):
            runtime = SimpleNamespace(mode=mode, mem_manager=memory, profiler=None,
                                      static_store=object())
            frame = LowRALayerFrame(runtime, object(), object(), None, 1)
            with patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer") as separate, \
                    patch("slora.common.lowra_layer_runtime.reconstruct_prefix_layer_fused_kv") as fused:
                frame._reconstruct(0)
                self.assertEqual(fused.call_count, int(mode == "layer_prefetch"))
                self.assertEqual(separate.call_count, int(mode != "layer_prefetch"))

    def test_full_cache_pins_batch_hits_before_allocating_misses(self):
        from slora.common.lowra_static import LowRAPrefixKVCache

        memory = MemoryAllocator(4, 4, torch.float16, 1, 16, 1, device="cpu")
        cache = LowRAPrefixKVCache(SimpleNamespace(prefix_len=2), memory)
        with patch.object(cache, "_reconstruct_many_into_dynamic_pool") as reconstruct:
            cache.acquire_many(["hot", "victim"], object())
            cache.release_many(["hot", "victim"])
            hot_indices = cache.entries["hot"].prefix_indices
            cache.entries["victim"].adapter_heat = 10
            reconstruct.reset_mock()
            selected = cache.acquire_many(["cold", "hot", "hot"], object())
            reconstruct.assert_called_once_with(["cold"], unittest.mock.ANY)
            self.assertIs(selected["hot"], hot_indices)
            self.assertEqual(cache.entries["hot"].ref_count, 2)
            self.assertNotIn("victim", cache.entries)

    def test_workspace_has_one_physical_layer_and_one_budget(self):
        from slora.common.lowra_runtime_stats import storage_bytes

        memory = MemoryAllocator(100, 100, torch.float16, 2, 16, 4,
                                 dynamic_layer_workspace=True, device="cpu")
        self.assertEqual(memory.workspace_token_equiv, 0)
        self.assertEqual(memory.can_use_mem_size, 100)
        private = memory.alloc(10)
        for tensor in memory.key_buffer + memory.value_buffer:
            tensor[private] = 7
        locations = memory.alloc_layer_workspace(21)
        self.assertEqual(memory.workspace_token_equiv, 6)
        self.assertEqual(memory.tot_size, 100)
        self.assertEqual(memory.can_use_mem_size, 84)
        self.assertEqual(locations.numel(), 21)
        actual = storage_bytes(memory.key_buffer + memory.value_buffer +
                               [memory.prefix_workspace_k, memory.prefix_workspace_v])
        self.assertEqual(actual, memory.get_memory_size())
        self.assertEqual(actual, 100 * 4 * 2 * 2 * 16 * 2)
        memory.prefix_workspace_k[locations] = 3
        memory.prefix_workspace_v[locations] = -3
        for tensor in memory.key_buffer + memory.value_buffer:
            self.assertTrue(torch.all(tensor[private] == 7))
        pointer = memory.prefix_workspace_k.data_ptr()
        memory.free_layer_workspace()
        memory.free_layer_workspace()
        self.assertEqual(memory.can_use_mem_size, 90)
        self.assertEqual(memory.workspace_token_equiv, 0)
        memory.alloc_layer_workspace(8)
        self.assertEqual(memory.workspace_token_equiv, 2)
        self.assertEqual(memory.can_use_mem_size, 88)
        memory.free_layer_workspace()
        memory.free(private)
        memory.free_all()
        self.assertEqual(memory.can_use_mem_size, 100)
        self.assertEqual(pointer, memory.prefix_workspace_k.data_ptr())

    def test_workspace_failure_does_not_consume_slots(self):
        memory = MemoryAllocator(4, 4, torch.float16, 1, 16, 4,
                                 dynamic_layer_workspace=True, device="cpu")
        private = memory.alloc(3)
        with self.assertRaisesRegex(Exception, "no enough pool space"):
            memory.alloc_layer_workspace(5)
        self.assertEqual(memory.can_use_mem_size, 1)
        self.assertEqual(memory.workspace_token_equiv, 0)
        memory.free(private)

    def test_metadata_failure_releases_workspace(self):
        from slora.common.lowra_layer_runtime import LowRALayerRuntime

        memory = MemoryAllocator(20, 20, torch.float16, 1, 16, 4,
                                 dynamic_layer_workspace=True, device="cpu")
        static = SimpleNamespace(prefix_len=7, base_k=[torch.zeros(1)])
        runtime = LowRALayerRuntime(static, memory, "layer_serial")
        with patch("slora.common.lowra_layer_runtime.prepare_reconstruction_metadata",
                   side_effect=ValueError("invalid adapter")):
            with self.assertRaisesRegex(ValueError, "invalid adapter"):
                runtime.prepare(["a", "a", "b"], object())
        self.assertEqual(memory.can_use_mem_size, 20)
        self.assertEqual(memory.layer_workspace_tokens, 0)

    def test_forward_workspace_tracks_unique_adapters_and_shrinks_after_filter(self):
        from slora.common.lowra_layer_runtime import LowRALayerRuntime

        memory = MemoryAllocator(20, 20, torch.float16, 1, 16, 4,
                                 dynamic_layer_workspace=True, device="cpu")
        static = SimpleNamespace(prefix_len=7, base_k=[torch.zeros(1)])
        runtime = LowRALayerRuntime(static, memory, "layer_serial")
        with patch("slora.common.lowra_layer_runtime.prepare_reconstruction_metadata"):
            frame = runtime.prepare(["a", "a", "b"], object())
            self.assertEqual(frame.unique_count, 2)
            self.assertEqual(memory.layer_workspace_tokens, 14)
            self.assertEqual(memory.workspace_token_equiv, 4)
            self.assertTrue(torch.equal(frame.prefix_b_loc[0], frame.prefix_b_loc[1]))
            self.assertFalse(torch.equal(frame.prefix_b_loc[0], frame.prefix_b_loc[2]))
            frame.finish()
            frame.finish()
            self.assertEqual(memory.can_use_mem_size, 20)
            smaller = runtime.prepare(["b", "b"], object())
            self.assertEqual(memory.layer_workspace_tokens, 7)
            self.assertEqual(memory.workspace_token_equiv, 2)
            smaller.finish()
        self.assertEqual(memory.can_use_mem_size, 20)
        self.assertEqual(memory.peak_workspace_tokens, 14)

    def test_one_hundred_sources_charge_only_five_active_workspace_prefixes(self):
        from slora.common.lowra_static import LowRAStaticPrefixStore

        prefix, layers, count, rank = 8192, 32, 100, 16
        static_tokens = LowRAStaticPrefixStore.token_equivalent_static_size(
            prefix, [rank] * count, 4096, 4096)
        self.assertEqual(static_tokens, 11392)
        memory = MemoryAllocator(36864 - static_tokens, 36864 - static_tokens,
                                 torch.float16, 1, 1, layers,
                                 dynamic_layer_workspace=True, device="cpu")
        self.assertEqual(memory.workspace_token_equiv, 0)
        memory.alloc_layer_workspace(5 * prefix)
        self.assertEqual(memory.workspace_token_equiv, 1280)
        self.assertEqual(memory.can_use_mem_size, 24192)
        memory.free_layer_workspace()
        self.assertEqual(memory.can_use_mem_size, 25472)

    def test_fifo_not_cache_or_group_priority(self):
        queue = LowRAFIFOReqQueue(120, 100, 10, 40, "full")
        queue.update_prefix_cache_state(dict(cached_lora_dirs={"hot"}, inactive_lora_dirs={"hot"}))
        requests = [SimpleNamespace(adapter_dir=directory, input_len=10, max_output_len=2,
                                    output_ids=[], aborted=False, request_id=str(i))
                    for i, directory in enumerate(["cold", "hot", "hot"])]
        for request in requests:
            queue.append(request)
        batch = queue.generate_new_batch(None, {"cold": 1, "hot": 1})
        self.assertEqual([r.request_id for r in batch.reqs], ["0", "1"])
        self.assertEqual(queue.waiting_req_list, requests[2:])

    def test_layered_prefix_is_not_charged_per_request(self):
        queue = LowRAFIFOReqQueue(60, 100, 10, 40, "layer_serial", layer_num=4)
        for i in range(3):
            queue.append(SimpleNamespace(adapter_dir="a", input_len=10, max_output_len=2,
                                         output_ids=[], aborted=False, request_id=str(i)))
        self.assertEqual(len(queue.generate_new_batch(None, {"a": 1}).reqs), 3)

    def test_layer_admission_charges_union_after_merge_not_all_sources(self):
        queue = LowRAFIFOReqQueue(50, 100, 10, 40, "layer_serial", layer_num=4)
        def request(directory, request_id):
            return SimpleNamespace(adapter_dir=directory, input_len=10, max_output_len=2,
                                   output_ids=[], aborted=False, request_id=request_id)
        running = SimpleNamespace(reqs=[request("a", "old")])
        same, missing = request("a", "same"), request("b", "missing")
        queue.append(same)
        queue.append(missing)
        ranks = {str(i): 1 for i in range(100)}
        ranks.update(a=1, b=1)
        batch = queue.generate_new_batch(running, ranks)
        self.assertEqual([req.request_id for req in batch.reqs], ["same"])
        self.assertEqual(queue.waiting_req_list, [missing])

    def test_optional_workspace_limit_controls_admission_without_reservation(self):
        queue = LowRAFIFOReqQueue(1000, 100, 10, 40, "layer_serial",
                                 layer_num=4, workspace_lora_limit=1)
        for i, directory in enumerate(("a", "a", "b")):
            queue.append(SimpleNamespace(adapter_dir=directory, input_len=10, max_output_len=2,
                                         output_ids=[], aborted=False, request_id=str(i)))
        batch = queue.generate_new_batch(None, {"a": 1, "b": 1})
        self.assertEqual(len(batch.reqs), 2)
        self.assertEqual(queue.waiting_req_list[0].adapter_dir, "b")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KernelTests(unittest.TestCase):
    def test_bounded_count_changes_do_not_recompile_or_overwrite_other_adapters(self):
        from slora.common.lowra_reconstruct_bounded import (
            _lowra_reconstruct_bounded_kernel, lowra_reconstruct_bounded,
        )

        torch.manual_seed(42)
        prefix, embed, total, rank = 257, 128, 33, 16
        base = torch.randn(prefix, embed, device="cuda", dtype=torch.float16) * 0.1
        mini = torch.randn(prefix, total * rank, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(total * rank, embed, device="cuda", dtype=torch.float16) * 0.1
        locations = torch.randperm(total * prefix + 7, device="cuda")[:total * prefix].view(total, prefix)
        b_locs = torch.arange(total * rank, device="cuda").view(total, rank)
        starts = torch.arange(total, device="cuda") * rank
        lengths = torch.full((total,), rank, device="cuda", dtype=torch.long)
        scales = torch.ones(total, device="cuda", dtype=torch.float16)
        variants = []
        for count in (1, 7, 8, 9, 16, 17, 20, 33, 9, 1):
            out = torch.full((total * prefix + 7, embed), -7.0, device="cuda", dtype=torch.float16)
            lowra_reconstruct_bounded(base, mini, b, out, locations[:count], b_locs[:count],
                                      starts[:count], lengths[:count], scales[:count], workers=4,
                                      launch_config=(32, 128, 16, 4))
            for adapter in range(count):
                expected = base.float() + mini[:, adapter * rank:(adapter + 1) * rank].float() @ b[b_locs[adapter]].float()
                torch.testing.assert_close(out[locations[adapter]].float(), expected, atol=0.002, rtol=0.01)
            untouched = torch.ones(out.shape[0], device="cuda", dtype=torch.bool)
            untouched[locations[:count].reshape(-1)] = False
            self.assertTrue(bool((out[untouched] == -7).all()))
            variants.append(sum(len(entries) for entries in _lowra_reconstruct_bounded_kernel.cache.values()))
        self.assertTrue(all(v == variants[0] for v in variants), variants)

    def setUp(self):
        torch.manual_seed(42)

    def test_profile_excludes_only_base_weights_and_buffers_not_runtime_pools(self):
        from benchmarks.lowra_layer_profile import base_model_memory_bytes

        weight = torch.zeros(80, device="cuda", dtype=torch.float16)
        rope = torch.zeros(10, device="cuda", dtype=torch.float16)
        model = SimpleNamespace(pre_post_weight=SimpleNamespace(weight=weight[:5]),
                                trans_layers_weight=[SimpleNamespace(weight=weight[20:])],
                                rope=rope, weight_alias=weight,
                                mem_manager=SimpleNamespace(pool=torch.zeros(100, device="cuda")))
        measured = base_model_memory_bytes(model)
        self.assertEqual(measured["base_model_weight_bytes"], 160)
        self.assertEqual(measured["base_model_buffer_bytes"], 20)
        self.assertEqual(measured["base_model_resident_bytes"], 180)

    def test_fused_kv_matches_separate_and_reference_with_scattered_indices(self):
        for dimensions in ((19, 70, (8, 16, 40)), (257, 256, (8, 16)),
                           (4609, 128, (8, 16)), (8193, 128, (8, 16))):
            with self.subTest(dimensions=dimensions):
                self.check_fused_kv(*dimensions)

    def check_fused_kv(self, prefix, embed, ranks):
        from slora.common.lowra_reconstruct import lowra_reconstruct_batched
        from slora.common.lowra_reconstruct_kv import lowra_reconstruct_kv_batched
        from slora.common.lowra_reconstruct_bounded import lowra_reconstruct_bounded

        count, total_rank, max_rank = len(ranks), sum(ranks), max(ranks)
        base_k = torch.randn(prefix, embed, device="cuda", dtype=torch.float16) * 0.1
        base_v = torch.randn_like(base_k) * 0.1
        mini_k = torch.randn(prefix, total_rank, device="cuda", dtype=torch.float16) * 0.1
        mini_v = torch.randn_like(mini_k) * 0.1
        b = torch.randn(2 * total_rank + 7, embed, device="cuda", dtype=torch.float16) * 0.1
        b_rows = torch.randperm(b.shape[0], device="cuda")[:2 * total_rank]
        k_locs = torch.zeros(count, max_rank, device="cuda", dtype=torch.long)
        v_locs = torch.zeros_like(k_locs)
        cursor = 0
        starts = []
        for row, rank in enumerate(ranks):
            starts.append(cursor)
            k_locs[row, :rank] = b_rows[cursor:cursor + rank]
            v_locs[row, :rank] = b_rows[total_rank + cursor:total_rank + cursor + rank]
            cursor += rank
        rank_starts = torch.tensor(starts, device="cuda", dtype=torch.long)
        rank_lens = torch.tensor(ranks, device="cuda", dtype=torch.long)
        scales = torch.tensor([0.25, 0.5, 1.75][:count], device="cuda", dtype=torch.float16)
        rows = count * prefix + 7
        prefix_locs = torch.randperm(rows, device="cuda")[:count * prefix].view(count, prefix)
        out_k = torch.full((rows, embed), 5.5, device="cuda", dtype=torch.float16)
        out_v = torch.full_like(out_k, -3.5)
        separate_k, separate_v = out_k.clone(), out_v.clone()
        lowra_reconstruct_kv_batched(base_k, base_v, mini_k, mini_v, b, out_k, out_v,
                                     prefix_locs, k_locs, v_locs, rank_starts, rank_lens, scales)
        for base, mini, locs, out in ((base_k, mini_k, k_locs, separate_k),
                                      (base_v, mini_v, v_locs, separate_v)):
            lowra_reconstruct_batched(base, mini, b, out, prefix_locs, locs,
                                      rank_starts, rank_lens, scales)
        torch.testing.assert_close(out_k, separate_k, atol=0, rtol=0)
        torch.testing.assert_close(out_v, separate_v, atol=0, rtol=0)
        for workers in (27, 648):
            bounded_k = torch.full_like(out_k, 5.5)
            bounded_v = torch.full_like(out_v, -3.5)
            for base, mini, locs, actual, expected in ((base_k, mini_k, k_locs, bounded_k, separate_k),
                                                      (base_v, mini_v, v_locs, bounded_v, separate_v)):
                lowra_reconstruct_bounded(base, mini, b, actual, prefix_locs, locs,
                                          rank_starts, rank_lens, scales, workers=workers,
                                          launch_config=(32, 128, 16, 4))
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
        for base, mini, locs, actual in ((base_k, mini_k, k_locs, out_k),
                                        (base_v, mini_v, v_locs, out_v)):
            for row, (start, rank) in enumerate(zip(starts, ranks)):
                expected = base.float() + (mini[:, start:start + rank].float()
                                          @ b[locs[row, :rank]].float()) * scales[row]
                torch.testing.assert_close(actual[prefix_locs[row]].float(), expected,
                                           atol=0.002, rtol=0.01)

    def test_prefill_and_decode_with_shared_prefix_ragged_lengths_and_gqa(self):
        from slora.models.lowra.prefix_attention import context_attention_fwd_with_prefix
        from slora.models.lowra.decode_attention import token_attention_fwd_with_prefix

        for prefix_len, private_lengths in ((17, (3, 5)), (129, (3, 65)), (8193, (17, 129))):
            heads, kv_heads, dim = 4, 2, 128
            first, maximum = private_lengths
            total = sum(private_lengths)
            lengths = torch.tensor(private_lengths, device="cuda", dtype=torch.int32)
            starts = torch.tensor([0, first], device="cuda", dtype=torch.int32)
            prefix_k = torch.randn(prefix_len + 5, kv_heads, dim, device="cuda", dtype=torch.float16) * 0.2
            prefix_v = torch.randn_like(prefix_k) * 0.2
            indices = torch.randperm(prefix_len + 5, device="cuda")[:prefix_len]
            prefix_locs = indices[None].expand(2, -1).contiguous()
            q = torch.randn(total, heads, dim, device="cuda", dtype=torch.float16) * 0.2
            k = torch.randn(total, kv_heads, dim, device="cuda", dtype=torch.float16) * 0.2
            v = torch.randn_like(k) * 0.2
            actual = torch.empty_like(q)
            context_attention_fwd_with_prefix(q, k, v, actual, starts, lengths, maximum,
                                              prefix_locs, prefix_len, prefix_k, prefix_v)
            for row, (start, length) in enumerate(((0, first), (first, maximum))):
                full_k = torch.cat([prefix_k[indices], k[start:start + length]]).float().repeat_interleave(2, 1)
                full_v = torch.cat([prefix_v[indices], v[start:start + length]]).float().repeat_interleave(2, 1)
                scores = torch.einsum("qhd,khd->hqk", q[start:start + length].float(), full_k) / dim ** 0.5
                mask = torch.arange(prefix_len + length, device="cuda")[None, :] <= (
                    prefix_len + torch.arange(length, device="cuda")[:, None])
                probabilities = scores.masked_fill(~mask[None], -float("inf")).softmax(-1)
                expected = torch.einsum("hqk,khd->qhd", probabilities, full_v)
                torch.testing.assert_close(actual[start:start + length].float(), expected, atol=0.002, rtol=0.01)
            b_loc = torch.zeros((2, maximum), device="cuda", dtype=torch.long)
            b_loc[0, maximum - first:] = torch.arange(first, device="cuda")
            b_loc[1] = torch.arange(first, total, device="cuda")
            last = torch.tensor([first - 1, total - 1], device="cuda")
            query = q[last]
            decoded = token_attention_fwd_with_prefix(query, prefix_k, prefix_v, k, v,
                                                      prefix_locs, b_loc, lengths, maximum)
            torch.testing.assert_close(decoded, actual[last],
                                       atol=0.002, rtol=0.01)

    def test_nonzero_reconstruction_unique_mapping_and_single_buffer_overwrite(self):
        from slora.common.lowra_static import LowRAStaticPrefixStore, LowRAPrefixKVCache
        from slora.common.lowra_layer_runtime import LowRALayerRuntime

        for mode, kernel in (("full", "auto"), ("layer_serial", "auto"),
                             ("layer_prefetch", "fused"), ("layer_prefetch", "split"),
                             ("layer_prefetch", "bounded")):
            layers, prefix, heads, dim = 3, 19, 2, 16
            memory = MemoryAllocator(256, 256, torch.float16, heads, dim, layers,
                                     dynamic_layer_workspace=mode != "full")
            static = LowRAStaticPrefixStore(prefix, ["a", "b"], {"a": 16, "b": 8},
                                            layers, heads, heads, dim)
            for tensor in static.base_k + static.base_v + static.mini_k + static.mini_v:
                tensor.normal_(std=0.1)
            adapter_locs = memory.alloc(96)
            adapter_locs = adapter_locs[torch.randperm(96, device="cuda")]
            for tensor in memory.value_buffer:
                tensor[adapter_locs] = torch.randn(96, heads, dim, device="cuda", dtype=torch.float16) * 0.1
            adapter = SimpleNamespace(mem_manager=memory, idx_map={"a": 0, "b": 1},
                                       a_loc=adapter_locs, a_start=torch.tensor([0, 64], device="cuda"),
                                       a_len=torch.tensor([64, 32], device="cuda"),
                                       a_scaling=torch.tensor([0.25, 0.5], device="cuda", dtype=torch.float16))
            dirs = ["b", "a", "a"]
            cached = LowRAPrefixKVCache(static, memory) if mode == "full" else None
            prefix_by_dir = cached.acquire_many(dirs, adapter) if cached else None
            free_before = memory.can_use_mem_size
            frame = LowRALayerRuntime(static, memory, mode,
                                      prefetch_reconstruction=kernel,
                                      prefetch_workers=27).prepare(dirs, adapter, prefix_by_dir)
            self.assertEqual(frame.unique_count, 2)
            self.assertTrue(torch.equal(frame.prefix_b_loc[1], frame.prefix_b_loc[2]))
            before = [tensor[adapter_locs].clone() for tensor in memory.value_buffer]
            observed = []
            pointers = []
            for layer in range(layers):
                frame.before_layer(layer)
                frame.before_attention(layer)
                k_buffer, v_buffer = frame.buffers(layer)
                observed.append((k_buffer[frame.prefix_b_loc].clone(), v_buffer[frame.prefix_b_loc].clone()))
                pointers.append(k_buffer.data_ptr())
                frame.after_attention(layer)
                # Independent GPU work represents the FFN; no synchronization between layers.
                work = torch.randn(128, 128, device="cuda", dtype=torch.float16)
                torch.mm(work, work)
            frame.finish()
            frame.finish()
            torch.cuda.synchronize()
            if mode != "full":
                self.assertEqual(len(set(pointers)), 1)
                self.assertEqual(memory.can_use_mem_size, free_before)
                self.assertEqual(memory.workspace_token_equiv, 0)
            for layer, (keys, values) in enumerate(observed):
                torch.testing.assert_close(memory.value_buffer[layer][adapter_locs], before[layer])
                for row, directory in enumerate(dirs):
                    rank_start, rank_end = static.get_lora_rank_range_by_dir(directory)
                    rank = rank_end - rank_start
                    idx = adapter.idx_map[directory]
                    base = int(adapter.a_start[idx])
                    for actual, bases, minis, offset in ((keys[row], static.base_k, static.mini_k, 1),
                                                        (values[row], static.base_v, static.mini_v, 2)):
                        b = before[layer][base + offset * rank:base + (offset + 1) * rank].reshape(-1, rank).T
                        expected = bases[layer].float().view(prefix, -1) + (
                            minis[layer][:, rank_start:rank_end].float() @ b.float()) * adapter.a_scaling[idx]
                        torch.testing.assert_close(actual.float().view(prefix, -1), expected, atol=0.002, rtol=0.01)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ServingLifecycleTests(unittest.TestCase):
    def test_ragged_batch_logits_after_filter_merge_and_adapter_replacement(self):
        from benchmarks.lowra_layer_profile import build_server
        from slora.server.router.model_infer.post_process import sample

        reference = None
        for mode, kernel in (("layer_serial", "split"), ("layer_prefetch", "fused"),
                             ("layer_prefetch", "split"), ("layer_prefetch", "bounded"),
                             ("layer_serial", "common_bounded"), ("layer_prefetch", "common_bounded")):
            args = SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                                   adapter_base="dummy-lora-7b-rank-16", mode=mode, prefix_len=257,
                                   query_len=40, output_len=8, batch_size=3, unique_loras=2,
                                   num_adapters=3, rank=16, nonzero_prefix=True)
            if kernel == "common_bounded":
                args.reconstruction_kernel, args.reconstruction_workers = "bounded", 1296
            server, _, requests, _ = build_server(args)
            if kernel != "common_bounded":
                server.lowra_runtime.prefetch_reconstruction = kernel
            collected = []

            def capture(logits, batch):
                collected.append(logits.detach().clone())
                return sample(logits, batch)

            for request, length in zip(requests, (3, 17, 33)):
                request["input_id"] = [10] * length
            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture):
                server.exposed_add_batch("old", requests, "fp16")
                server.exposed_prefill_batch("old")
                server.exposed_decode_batch("old")
                server.exposed_decode_batch("old")
                server.exposed_filter_batch("old", ["2", "0"])
                server.infer_adapter.offload_adapters([requests[0]["adapter_dir"]])
                server.infer_adapter.load_adapters([server.adapters[2]])
                mini = dict(requests[1], request_id="new", input_id=[11] * 9,
                            adapter_dir=server.adapters[2].lora_dir,
                            sampling_param=dict(requests[1]["sampling_param"]))
                server.exposed_add_batch("mini", [mini], "fp16")
                server.exposed_prefill_batch("mini")
                server.exposed_merge_batch("old", "mini")
                server.exposed_decode_batch("old")
                server.exposed_filter_batch("old", ["new", "0"])
                server.exposed_decode_batch("old")
                server.exposed_remove_batch("old")
            torch.cuda.synchronize()
            server.infer_adapter.offload_adapters([])
            self.assertEqual(server.model.mem_manager.can_use_mem_size, server.model.mem_manager.tot_size)
            if mode == "layer_prefetch":
                self.assertGreater(server.lowra_runtime.metadata_reuses, 0)
                self.assertGreater(server.lowra_runtime.metadata_builds, 1)
                self.assertEqual(len(server.lowra_runtime.events), 3)
            if reference is None:
                reference = collected
            else:
                self.assertEqual(len(collected), len(reference))
                for actual, expected in zip(collected, reference):
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)

    def test_real_layers_filter_merge_decode_and_release(self):
        from benchmarks.lowra_layer_profile import build_server

        for mode in ("full", "layer_serial", "layer_prefetch"):
            args = SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                                   adapter_base="dummy-lora-7b-rank-16", mode=mode, prefix_len=17,
                                   query_len=8, output_len=5, batch_size=3, unique_loras=2,
                                   num_adapters=2, rank=16, nonzero_prefix=True)
            server, profiler, requests, _ = build_server(args)
            requests[0]["input_id"] = [10] * 3
            requests[1]["input_id"] = [10] * 5
            requests[2]["input_id"] = [10] * 2
            server.exposed_add_batch("old", requests, "fp16")
            server.exposed_prefill_batch("old")
            server.exposed_decode_batch("old")
            server.exposed_filter_batch("old", ["2", "0"])
            mini = dict(requests[1], request_id="new", input_id=[10] * 4,
                        sampling_param=dict(requests[1]["sampling_param"]))
            server.exposed_add_batch("mini", [mini], "fp16")
            server.exposed_prefill_batch("mini")
            server.exposed_merge_batch("old", "mini")
            server.exposed_decode_batch("old")
            server.exposed_decode_batch("old")
            server.exposed_remove_batch("old")
            if server.lowra_prefix_cache is not None:
                self.assertTrue(all(entry.ref_count == 0 for entry in server.lowra_prefix_cache.entries.values()))
                server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
            torch.cuda.synchronize()
            self.assertEqual(server.model.mem_manager.tot_size - server.model.mem_manager.can_use_mem_size, 128)
            server.infer_adapter.offload_adapters([])
            self.assertEqual(server.model.mem_manager.tot_size, server.model.mem_manager.can_use_mem_size)

    def test_actual_logits_match_across_serial_and_prefetch_modes(self):
        from benchmarks.lowra_layer_profile import build_server, run_round
        from slora.common.lowra_layer_runtime import LowRALayerFrame
        from slora.server.router.model_infer.post_process import sample

        reference = None
        for mode, kernel in (("layer_serial", "split"), ("layer_prefetch", "fused"),
                             ("layer_prefetch", "split"), ("layer_prefetch", "bounded"),
                             ("layer_serial", "common_bounded"), ("layer_prefetch", "common_bounded")):
            args = SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                                   adapter_base="dummy-lora-7b-rank-16", mode=mode, prefix_len=129,
                                   query_len=5, output_len=4, batch_size=3, unique_loras=2,
                                   num_adapters=2, rank=16, nonzero_prefix=True)
            if kernel == "common_bounded":
                args.reconstruction_kernel, args.reconstruction_workers = "bounded", 1296
            server, profiler, requests, _ = build_server(args)
            if kernel != "common_bounded":
                server.lowra_runtime.prefetch_reconstruction = kernel
            collected = []

            def capture(logits, batch):
                collected.append(logits.detach().clone())
                return sample(logits, batch)

            release = LowRALayerFrame.after_attention
            released_layers = []

            def check_release(frame, layer):
                # The callback must precede O projection, not follow the sublayer.
                phase, last_layer, operation, _, _ = profiler.intervals[-1]
                self.assertEqual((last_layer, operation), (layer, "attention_kernel"))
                released_layers.append((phase, layer))
                release(frame, layer)

            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=capture), \
                    patch.object(LowRALayerFrame, "after_attention", check_release):
                run_round(server, profiler, requests, args.output_len)
            torch.cuda.synchronize()
            self.assertEqual(len(released_layers), server.model.layers_num * args.output_len)
            intervals = profiler.collect()
            if mode == "layer_prefetch":
                for item in intervals:
                    if item["operation"] != "reconstruct" or item["layer"] == 0:
                        continue
                    reader = next(row for row in intervals if row["phase"] == item["phase"]
                                  and row["layer"] == item["layer"] - 1
                                  and row["operation"] == "attention_kernel")
                    self.assertGreaterEqual(item["start_ms"] + 0.001, reader["end_ms"])
            if reference is None:
                reference = collected
            else:
                for actual, expected in zip(collected, reference):
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
