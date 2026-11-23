"""Measure elastic page remaps and share reconstruction source metadata in V3."""
from collections import OrderedDict, deque
import time

import torch

from slora.common.lowra_adaptive import AdaptiveLayerRuntime, AdaptivePrefixKVCache
from slora.common.lowra_layer_runtime import cached_reconstruction_metadata


class AdaptiveV3PrefixKVCache(AdaptivePrefixKVCache):
    def __init__(self, prefix_len, mem_manager):
        super().__init__(prefix_len, mem_manager)
        self._pending_applies = deque(maxlen=32)
        self._apply_samples = deque(maxlen=16)

    def reset_access_stats(self):
        super().reset_access_stats()
        self.source_metadata_builds = 0
        self.source_metadata_reuses = 0

    def access_stats(self):
        return dict(super().access_stats(), source_metadata_builds=self.source_metadata_builds,
                    source_metadata_reuses=self.source_metadata_reuses)

    def apply_plan(self, layers_by_dir, evictions=()):
        evictions = tuple(evictions)
        changed = sum(k != (self.entries[d].layers if d in self.entries else 0)
                      for d, k in layers_by_dir.items())
        changed += sum(d in self.entries for d in evictions)
        if not changed:
            return super().apply_plan(layers_by_dir, evictions)
        begin = end = None
        if torch.device(self.mem_manager.device).type == "cuda":
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        start = time.perf_counter()
        super().apply_plan(layers_by_dir, evictions)
        host_ms = (time.perf_counter() - start) * 1000.
        if end is not None:
            end.record()
        self._pending_applies.append((self.plan_version, changed, host_ms, begin, end))

    def state_dict(self):
        # This state is polled between forwards. Never synchronize the GPU to train the planner.
        while self._pending_applies:
            seq, changed, host_ms, begin, end = self._pending_applies[0]
            if end is not None and not end.query():
                break
            self._pending_applies.popleft()
            self._apply_samples.append(dict(seq=seq, changed_dirs=changed, host_ms=host_ms,
                                            device_ms=begin.elapsed_time(end) if end is not None else 0.))
        return dict(super().state_dict(), v3_apply_samples=list(self._apply_samples))


class AdaptiveV3LayerRuntime(AdaptiveLayerRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata_cache = OrderedDict()

    def _prepare_metadata(self, role, directories, infer_adapter, locations):
        # Identical ordered missing adapters share source indices across layers.
        # The existing helper verifies packed-tensor identity, versions, ranks and
        # row mapping, and binds the destination for this specific invocation.
        group = tuple(directories)
        metadata, cached, hit = cached_reconstruction_metadata(
            self.static_store, group, infer_adapter, locations, self._metadata_cache.pop(group, None))
        if cached is not None:
            self._metadata_cache[group] = cached
            if len(self._metadata_cache) > self.static_store.layer_num:
                self._metadata_cache.popitem(last=False)
        if hit:
            self.metadata_reuses += 1
            self.cache.source_metadata_reuses += 1
        else:
            self.metadata_builds += 1
            self.cache.source_metadata_builds += 1
        return metadata
