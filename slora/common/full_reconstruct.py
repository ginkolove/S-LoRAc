"""Complete MiniKV reconstruction with the full-prefix FIFO/LRU cache lifecycle."""

import torch

from slora.common.full_reprefill import FullPrefixKVCache, FullReprefillRuntime
from slora.common.prefix_modes import FULL_RECONSTRUCT
from slora.common.lowra_layer_runtime import (
    prepare_reconstruction_metadata, profile_interval, reconstruct_prefix_layer,
)


class FullReconstructPrefixKVCache(FullPrefixKVCache):
    def reset_access_stats(self):
        super().reset_access_stats()
        self.prefix_reconstruct_builds = 0
        self.prefix_reconstruct_tokens = 0

    def access_stats(self):
        return dict(super().access_stats(),
                    prefix_reconstruct_builds=self.prefix_reconstruct_builds,
                    prefix_reconstruct_tokens=self.prefix_reconstruct_tokens)

    def _validate_prefix_inputs(self, requests):
        # The immutable static store supplies the prefix, not per-request tokens.
        return None

    def _record_prefix_builds(self, misses):
        self.prefix_reconstruct_builds += len(misses)
        self.prefix_reconstruct_tokens += len(misses) * self.prefix_len
        for directory in misses:
            self.prefix_build_counts[directory] = self.prefix_build_counts.get(directory, 0) + 1


class FullReconstructRuntime(FullReprefillRuntime):
    def __init__(self, static_store, mem_manager, cache, profiler=None):
        if static_store.prefix_len != cache.prefix_len:
            raise ValueError("static and reconstructed prefix lengths must match")
        if cache.mem_manager is not mem_manager:
            raise ValueError("full reconstruction must use the cache's Unified Paging pool")
        super().__init__(static_store.prefix_len, mem_manager, cache, profiler)
        self.static_store = static_store
        self.mode = FULL_RECONSTRUCT

    @torch.no_grad()
    def acquire_many(self, requests, infer_adapter):
        def build_prefixes(lora_dirs, prefix_ids, prefix_indices):
            prefix_locs = torch.stack([prefix_indices[directory] for directory in lora_dirs])
            metadata = prepare_reconstruction_metadata(
                self.static_store, lora_dirs, infer_adapter, prefix_locs)
            for layer_id in range(self.static_store.layer_num):
                with profile_interval(self.profiler, "reconstruct", layer_id):
                    reconstruct_prefix_layer(
                        self.static_store, infer_adapter, metadata, layer_id,
                        self.mem_manager.key_buffer[layer_id],
                        self.mem_manager.value_buffer[layer_id])

        return self.cache.acquire_many(requests, build_prefixes)
