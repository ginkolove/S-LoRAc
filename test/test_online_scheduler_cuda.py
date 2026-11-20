"""Opt-in real-model online scheduling checks on a small nonzero-weight model.

CUDA_VISIBLE_DEVICES=0 RUN_ONLINE_CUDA_TESTS=1 python -m unittest discover \
    -s test -p test_online_scheduler_cuda.py -v
"""

import asyncio
import gc
import os
from types import SimpleNamespace
import unittest

import torch

from test_online_scheduler import request, make_router, drain
from slora.common.prefix_modes import FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, LAYER_RETAINED
from slora.server.router.model_infer.model_rpc import ModelRpcClient
from slora.server.router.online_req_queue import OnlineReqQueue


@unittest.skipUnless(os.environ.get("RUN_ONLINE_CUDA_TESTS") == "1" and torch.cuda.is_available(),
                     "set RUN_ONLINE_CUDA_TESTS=1 on an approved GPU")
class OnlineModelTests(unittest.TestCase):
    def test_staggered_arrivals_merge_and_top_k_rebalance(self):
        from benchmarks.lowra_layer_profile import build_server
        for mode, retained, top_k in ((FULL_RECONSTRUCT, 0, None),
                                      (LAYER_SERIAL, 0, None),
                                      (LAYER_RETAINED, 3, 1)):
            with self.subTest(mode=mode):
                args = SimpleNamespace(seed=42, dummy=True, smoke=True,
                    model_dir="huggyllama/llama-7b", adapter_base="dummy-lora-7b-rank-16",
                    mode=mode, prefix_len=17, query_len=5, output_len=25,
                    batch_size=1, unique_loras=2, num_adapters=2, rank=16,
                    nonzero_prefix=True, retain_layers=retained, retain_top_k=top_k)
                server, profiler, _, _ = build_server(args)
                server.lowra_runtime.profiler = None
                dirs = list(server.adapter_id)
                reqs = [request("first", dirs[0], query=5, output=16),
                        request("second", dirs[1], query=7, output=10),
                        request("third", dirs[1], query=3, output=8),
                        request("fourth", dirs[0], query=4, output=12)]
                expected = {}
                for req in reqs:
                    server.exposed_offload_adapters([])
                    server.exposed_load_adapters([req.adapter_dir])
                    server.exposed_add_batch("ref", [req.to_rpc_obj()], "fp16")
                    tokens = [server.exposed_prefill_batch("ref")[req.request_id][0]]
                    for _ in range(req.max_output_len - 1):
                        tokens.append(server.exposed_decode_batch("ref")[req.request_id][0])
                    expected[req.request_id] = tokens
                    server.exposed_remove_batch("ref")
                    if server.lowra_prefix_cache is not None:
                        server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
                server.exposed_offload_adapters([])
                pool = server.model.mem_manager
                queue = OnlineReqQueue(pool.tot_size, 1000, 10, 17, mode,
                                       layer_num=3, retain_layers=retained, retain_top_k=top_k)
                router = make_router(queue, ModelRpcClient(server, 1), {d: 16 for d in dirs})
                router.input_params.lowra_prefix_mode = mode
                router.input_params.enable_lowra = True

                async def staggered():
                    for req in reqs:
                        queue.append(req)
                        await router._step()
                        await router._step()
                    await drain(router)

                asyncio.run(staggered())
                self.assertEqual(router.online_rejections, 0)
                for req in reqs:
                    self.assertEqual(req.output_ids, expected[req.request_id])
                self.assertEqual(server.cache, {})
                if server.lowra_prefix_cache is not None:
                    self.assertTrue(all(entry.ref_count == 0 for entry in server.lowra_prefix_cache.entries.values()))
                    server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
                self.assertEqual(pool.can_use_mem_size, pool.tot_size)
                print("online staggered GPU check: %s preemptions=%s free=%s/%s" % (
                    mode, router.online_preemptions, pool.can_use_mem_size, pool.tot_size), flush=True)
                del router, server, pool, profiler
                gc.collect()
                torch.cuda.empty_cache()

    def test_pressure_matches_uninterrupted_generation_in_every_prefix_mode(self):
        from benchmarks.lowra_layer_profile import build_server
        configs = [(FULL_REPREFILL, 0, None), (FULL_RECONSTRUCT, 0, None),
                   (LAYER_SERIAL, 0, None), ("layer_prefetch", 0, None),
                   (LAYER_RETAINED, 2, None), (LAYER_RETAINED, 3, 1)]
        for mode, retained, top_k in configs:
            with self.subTest(mode=mode, retained=retained, top_k=top_k):
                args = SimpleNamespace(seed=42, dummy=True, smoke=True,
                    model_dir="huggyllama/llama-7b", adapter_base="dummy-lora-7b-rank-16",
                    mode=mode, prefix_len=17, query_len=5, output_len=17,
                    batch_size=1, unique_loras=1, num_adapters=2, rank=16,
                    nonzero_prefix=True, retain_layers=retained, retain_top_k=top_k)
                server, profiler, _, _ = build_server(args)
                server.lowra_runtime.profiler = None
                dirs = list(server.adapter_id)
                reqs = [request("a1", dirs[0], output=12), request("a2", dirs[0], query=7, output=10),
                        request("b", dirs[1], query=3, output=11)]
                for req in reqs:
                    req.prefix_input_ids = [10] * args.prefix_len
                    req.sample_params.presence_penalty = 0.1
                    req.sample_params.frequency_penalty = 0.2
                expected = {}
                for req in reqs:
                    server.exposed_offload_adapters([])
                    server.exposed_load_adapters([req.adapter_dir])
                    server.exposed_add_batch("reference", [req.to_rpc_obj()], "fp16")
                    output = server.exposed_prefill_batch("reference")
                    tokens = [output[req.request_id][0]]
                    for _ in range(req.max_output_len - 1):
                        tokens.append(server.exposed_decode_batch("reference")[req.request_id][0])
                    expected[req.request_id] = tokens
                    server.exposed_remove_batch("reference")
                    if server.lowra_prefix_cache is not None:
                        server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
                server.exposed_offload_adapters([])
                pool = server.model.mem_manager
                self.assertEqual(pool.can_use_mem_size, pool.tot_size)
                queue = OnlineReqQueue(pool.tot_size, 1000, 10, args.prefix_len, mode,
                                       layer_num=3, retain_layers=retained, retain_top_k=top_k)
                router = make_router(queue, ModelRpcClient(server, 1), {d: 16 for d in dirs})
                router.input_params.lowra_prefix_mode = mode
                router.input_params.enable_lowra = True
                for req in reqs:
                    router.add_req(req.adapter_dir, req.prompt_ids, req.sample_params, req.request_id,
                                   {"prefix_input_ids": req.prefix_input_ids})
                online_reqs = queue.waiting_req_list[:]
                asyncio.run(drain(router))
                self.assertGreater(router.online_preemptions, 0)
                self.assertEqual(router.online_rejections, 0)
                for req in online_reqs:
                    self.assertIsNone(req.max_output_len)
                    self.assertEqual(req.output_ids, expected[req.request_id])
                    self.assertEqual(req.prompt_ids, [10] * req.input_len)
                self.assertEqual(router.online_output_limits, {})
                self.assertEqual(server.cache, {})
                if server.lowra_prefix_cache is not None:
                    self.assertTrue(all(entry.ref_count == 0 for entry in server.lowra_prefix_cache.entries.values()))
                    server.lowra_prefix_cache.evict_many(list(server.lowra_prefix_cache.entries))
                self.assertEqual(pool.can_use_mem_size, pool.tot_size)
                self.assertEqual(pool.workspace_token_equiv, 0)
                print("online GPU check: %s K=%s top_k=%s preemptions=%s recomputed=%s free=%s/%s" % (
                    mode, retained, top_k, router.online_preemptions,
                    router.online_recomputed_tokens, pool.can_use_mem_size, pool.tot_size), flush=True)
                del router, server, pool, profiler
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
