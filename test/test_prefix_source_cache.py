import unittest
from types import SimpleNamespace

import torch

from slora.common.full_reconstruct import FullReconstructPrefixKVCache
from slora.common.prefix_modes import ACTIVE_REPREFILL, CPU_READ, CPU_FIXED, CPU_FIXED_R64, CPU_MODES, FULL_RECONSTRUCT, IMPRESS
from slora.common.prefix_source_cache import ActivePrefixCache, HostPrefixBank
from slora.server.router.online_req_queue import OnlineReqQueue
from test_full_reprefill import CPUPool, request


class PrefixSources(unittest.TestCase):
    def setUp(self):
        self.pool = CPUPool(24)
        self.pool.key_buffer = [torch.arange(48).view(24, 1, 2) + i * 100 for i in (1, 2)]
        self.pool.value_buffer = [t + 1000 for t in self.pool.key_buffer]
        self.bank = self.make_bank(CPU_READ)

    def make_bank(self, mode):
        bank = HostPrefixBank(self.pool, [7, 8], "abc", mode=mode)
        for i, d in enumerate("abc"):
            bank.capture(d, torch.tensor([2*i, 2*i+1]))
        return bank

    @staticmethod
    def req(dirs):
        return [dict(adapter_dir=d, prefix_input_ids=[7, 8]) for d in dirs]

    def test_fragmented_and_contiguous_copies_all_layers(self):
        for indices in ([8, 16], [13, 14]):
            ids = torch.tensor(indices)
            self.bank.load(["b"], [7, 8], {"b": ids})
            for pair, k, v in zip(self.bank.entries["b"], self.pool.key_buffer, self.pool.value_buffer):
                for src, dst in zip(pair, (k, v)):
                    torch.testing.assert_close(dst[ids], src)
        with self.assertRaises(ValueError):
            self.bank.load(["b"], [8, 7], {"b": ids})

    def test_capped_bank_aliases_only_host_addresses(self):
        size = self.bank.entry_bytes
        bank = HostPrefixBank(self.pool, [7, 8], "abcde", byte_cap=2*size+1)
        self.assertEqual(bank.physical_dirs, ("a", "b"))
        self.assertEqual(bank.source_for, dict(zip("abcde", "ababa")))
        for i, d in enumerate(bank.physical_dirs):
            bank.capture(d, torch.tensor([2*i, 2*i+1]))
        cache = ActivePrefixCache(2, self.pool, CPU_READ, bank=bank)
        for t in (*self.pool.key_buffer, *self.pool.value_buffer):
            t.fill_(-1)
        locs = cache.acquire_many(self.req("aaccd"), None)
        self.assertEqual(len(set(torch.cat(list(locs.values())).tolist())), 6)
        for d, loc in locs.items():
            for pair, k, v in zip(bank.entries[bank.source_for[d]], self.pool.key_buffer, self.pool.value_buffer):
                torch.testing.assert_close(k[loc], pair[0])
                torch.testing.assert_close(v[loc], pair[1])
        cache.acquire_many(self.req("c"), None)
        stats = cache.access_stats()
        self.assertEqual((stats["host_prefix_entries"], stats["host_prefix_logical_entries"]), (2, 5))
        self.assertEqual((stats["host_prefix_bytes"], stats["prefix_h2d_bytes"]), (2*size, 3*size))
        self.assertTrue(stats["host_prefix_aliased"])
        with self.assertRaisesRegex(ValueError, "cannot fit one"):
            HostPrefixBank(self.pool, [7, 8], "ab", byte_cap=size-1)
        with self.assertRaisesRegex(ValueError, "one full prefix"):
            bank.load(["c"], [7, 8], {"c": locs["c"][:1]})

    def test_impress_copies_seeded_positions_to_original_slots_and_zeroes_only_padding(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for length, percent, transfer in ((1, None, 0), (7, None, 3), (8, None, 4),
                                              (7, 60, 4), (8, 0, 0), (8, 100, 8)):
                with self.subTest(device=device, length=length, percent=percent):
                    pool = SimpleNamespace(key_buffer=[t.clone().to(device) for t in self.pool.key_buffer],
                                           value_buffer=[t.clone().to(device) for t in self.pool.value_buffer])
                    buffers = pool.key_buffer + pool.value_buffer
                    originals = [t.clone() for t in buffers]
                    prefix = list(range(length))
                    # Two physical entries; logical c reads a's backing.
                    entry_bytes = length * sum(t[0].numel() * t.element_size() for t in buffers)
                    state = torch.get_rng_state()
                    bank = HostPrefixBank(pool, prefix, "abc", byte_cap=2 * entry_bytes, mode=IMPRESS,
                                          impress_read_percent=percent)
                    self.assertTrue(torch.equal(torch.get_rng_state(), state))
                    rows = torch.arange(length, device=device)
                    bank.capture("a", rows)
                    bank.capture("b", rows + length)
                    for t in buffers:
                        t.fill_(-1)
                    locations = {"c": 2 * rows, "b": 2 * length + rows}
                    bank.load(["c", "b"], prefix, locations)

                    kept = torch.randperm(length, generator=torch.Generator().manual_seed(42))[:transfer].to(device)
                    for dst, original in zip(buffers, originals):
                        expected = torch.full_like(dst, -1)
                        for directory, start in (("c", 0), ("b", length)):
                            slots = locations[directory]
                            expected[slots] = 0
                            expected[slots[kept]] = original[start + kept]
                        torch.testing.assert_close(dst, expected)
                    self.assertEqual(bank.transfer_bytes, transfer * (entry_bytes // length))
                    self.assertEqual(bank.entry_bytes, entry_bytes)

    def test_consecutive_batches_keep_only_intersection_and_fixed_pins(self):
        for mode in (ACTIVE_REPREFILL, *CPU_MODES, FULL_RECONSTRUCT):
            with self.subTest(mode=mode):
                pinned = {CPU_FIXED: ("a", "c"), CPU_FIXED_R64: ("a",)}.get(mode, ())
                bank = self.make_bank(IMPRESS) if mode == IMPRESS else self.bank
                cache = (FullReconstructPrefixKVCache(2, self.pool, reuse_next_batch=True)
                         if mode == FULL_RECONSTRUCT else
                         ActivePrefixCache(2, self.pool, mode, pinned, bank if mode in CPU_MODES else None))
                queue = OnlineReqQueue(100, 100, 20, prefix_len=2, prefix_mode=mode)
                for dirs in ("aab", "bc", "c", "a"):
                    queue.update_prefix_cache_state(cache.state_dict())
                    old = {d: e.prefix_indices for d, e in cache.entries.items()}
                    reqs = [request(d, str(i), query=1) for i, d in enumerate(dirs)]
                    plan = queue.plan([], reqs, dict.fromkeys("abc", 1))
                    self.assertTrue(plan.fits)
                    self.assertEqual(set(plan.evictions), set(old) - set(dirs) - set(pinned))
                    cache.evict_many(plan.evictions)
                    slots = cache.acquire_many(self.req(dirs), lambda *args: None)
                    for d in set(old) & set(dirs):
                        self.assertIs(slots[d], old[d])
                    self.assertEqual(set(cache.entries), set(dirs) | (set(old) & set(pinned)))
                    cache.release_many(dirs)
                    self.assertTrue(all(not e.ref_count for e in cache.entries.values()))
                stats = cache.access_stats()
                self.assertEqual(stats["batch_unique_misses"], 3 if pinned else 4)
                if mode in CPU_MODES:
                    per_load = bank.entry_bytes // 2 if mode == IMPRESS else bank.entry_bytes
                    self.assertEqual(stats["prefix_h2d_bytes"], stats["batch_unique_misses"] * per_load)
                    if mode == IMPRESS:
                        self.assertEqual(stats["impress_seed"], 42)
                        self.assertEqual(stats["prefix_h2d_tokens_per_load"], 1)
                        self.assertEqual(stats["host_prefix_bytes"], 3 * bank.entry_bytes)
                self.pool.used.clear()

    def test_cpu_loads_charge_slots_not_prefill_compute(self):
        req = request("c", "1", query=3)
        for mode, fits, tokens in ((ACTIVE_REPREFILL, False, 8), (CPU_READ, True, 3), (IMPRESS, True, 3)):
            queue = OnlineReqQueue(20, 5, 2, prefix_len=5, prefix_mode=mode)
            plan = queue.plan([], [req], {"c": 1})
            self.assertEqual((plan.fits, plan.prefill_tokens, plan.required_slots), (fits, tokens, 13))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for end-to-end comparison")
    def test_uploaded_prefix_matches_prefilled_logits(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from benchmarks.lowra_layer_profile import build_server
        from slora.common.full_reprefill import FullReprefillRuntime
        from slora.server.router.model_infer.post_process import sample
        args = SimpleNamespace(seed=42, dummy=True, smoke=True, model_dir="huggyllama/llama-7b",
                               adapter_base="dummy-lora-7b-rank-16", mode="full", prefix_len=129,
                               query_len=5, output_len=4, batch_size=3, unique_loras=2,
                               num_adapters=2, rank=16, nonzero_prefix=False)
        server, _, reqs, _ = build_server(args)
        pool = server.model.mem_manager
        bank = HostPrefixBank(pool, reqs[0]["prefix_input_ids"], list(dict.fromkeys(r["adapter_dir"] for r in reqs)))

        def run(name, save_cpu=False):
            logits = []
            def record(x, batch):
                logits.append(x.detach().clone())
                return sample(x, batch)
            with patch("slora.server.router.model_infer.model_rpc.sample", side_effect=record):
                server.exposed_add_batch(name, reqs, "fp16")
                if save_cpu:
                    for d, e in server.lowra_prefix_cache.entries.items():
                        bank.capture(d, e.prefix_indices)
                server.exposed_prefill_batch(name)
                for _ in range(3):
                    server.exposed_decode_batch(name)
            server.exposed_remove_batch(name)
            return logits

        expected = run("built", True)
        cache = server.lowra_prefix_cache
        for d, e in list(cache.entries.items()):
            for tensor in (*pool.key_buffer, *pool.value_buffer):
                tensor.index_fill_(0, e.prefix_indices.long(), float("nan"))
            cache.evict(d)
        cache = ActivePrefixCache(args.prefix_len, pool, CPU_READ, bank=bank)
        server.lowra_prefix_cache = cache
        server.lowra_runtime = FullReprefillRuntime(args.prefix_len, pool, cache)
        actual = run("loaded")
        self.assertEqual(cache.access_stats()["prefix_h2d_loads"], 2)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
