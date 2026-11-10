"""Canonical names preserve routing and runtime semantics of legacy modes."""

from types import SimpleNamespace
import unittest

import torch

from slora.common.full_reprefill import FullPrefixKVCache, FullReprefillRuntime
from slora.common.full_reconstruct import FullReconstructPrefixKVCache, FullReconstructRuntime
from slora.common.lowra_layer_runtime import LowRALayerRuntime
from slora.common.mem_allocator import MemoryAllocator
from slora.common.prefix_modes import FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, normalize_prefix_mode
from slora.server.router.full_reprefill_req_queue import FullReprefillReqQueue
from slora.server.router.lowra_fifo_req_queue import LowRAFIFOReqQueue


class PrefixModeTests(unittest.TestCase):
    def test_aliases_normalize_and_unknown_modes_are_rejected(self):
        for legacy, canonical in (("full", FULL_REPREFILL), ("full_reprefill", FULL_REPREFILL),
                                  ("full_reconstruct", FULL_RECONSTRUCT), ("layer_serial", LAYER_SERIAL)):
            self.assertEqual(normalize_prefix_mode(legacy), canonical)
            self.assertEqual(normalize_prefix_mode(canonical), canonical)
        with self.assertRaises(ValueError):
            normalize_prefix_mode("unknown")

    def test_runtime_mode_fields_are_canonical(self):
        pool = MemoryAllocator(20, 20, torch.float16, 1, 16, 3, device="cpu")
        static = SimpleNamespace(prefix_len=5, layer_num=3)
        full = FullReprefillRuntime(5, pool, FullPrefixKVCache(5, pool))
        reconstructed = FullReconstructRuntime(static, pool, FullReconstructPrefixKVCache(5, pool))
        self.assertEqual(full.mode, FULL_REPREFILL)
        self.assertEqual(reconstructed.mode, FULL_RECONSTRUCT)
        for mode in ("layer_serial", LAYER_SERIAL):
            runtime = LowRALayerRuntime(static, pool, mode)
            self.assertEqual(runtime.mode, LAYER_SERIAL)

    def test_router_dispatch_is_identical_for_aliases(self):
        from slora.server.router.manager import get_scheduler
        for canonical, legacy, expected in ((FULL_REPREFILL, "full_reprefill", FullReprefillReqQueue),
                                            (FULL_RECONSTRUCT, "full_reconstruct", FullReprefillReqQueue),
                                            (LAYER_SERIAL, "layer_serial", LowRAFIFOReqQueue)):
            for mode in (canonical, legacy):
                params = SimpleNamespace(enable_lowra=True, lowra_prefix_mode=mode,
                                         max_total_token_num=100, batch_max_tokens=100,
                                         running_max_req_size=10, lowra_shared_prefix_len=5,
                                         lowra_layer_num=3, lowra_workspace_loras=None)
                self.assertIs(type(get_scheduler(params, [])), expected)


if __name__ == "__main__":
    unittest.main()
