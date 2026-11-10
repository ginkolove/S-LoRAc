"""Keep the first K reconstructed layers while streaming the remaining layers.

Entries are shared by all active requests of an adapter. Ready retained layers
remain cached after the final release and are evicted under pressure by LRU.
Optional top-K retention ranks all live requests (including pending mini-batches).
Ties prefer already retained adapters, then lexical adapter ID.
"""

from collections import Counter
from dataclasses import dataclass

import torch

from slora.common.lowra_layer_runtime import (
    prepare_reconstruction_metadata, profile_interval, reconstruct_prefix_layer,
    reconstruct_prefix_layer_bounded, cached_reconstruction_metadata,
)
from slora.common.prefix_modes import LAYER_RETAINED
from slora.common.prefix_indices import PrefixIndexCache, PrefixIndexedFrame


@dataclass
class RetainedPrefixEntry:
    prefix_indices: torch.Tensor
    pool_indices: torch.Tensor
    prefix_len: int
    ref_count: int
    ready: bool = False
    last_used: int = 0

    @property
    def allocated_slots(self):
        return self.pool_indices.numel()


class RetainedPrefixKVCache:
    def __init__(self, prefix_len, retain_layers, mem_manager, retain_top_k=None, pinned_lora_dirs=()):
        if not 0 <= retain_layers <= mem_manager.layer_num:
            raise ValueError("retain_layers must be between zero and the model layer count")
        if not mem_manager.dynamic_layer_workspace:
            raise ValueError("retained prefixes require flattened Unified Paging backing")
        self.prefix_len = int(prefix_len)
        self.retain_layers = int(retain_layers)
        if retain_top_k is not None and retain_top_k < 0:
            raise ValueError("retain_top_k must be nonnegative or None (all adapters)")
        self.retain_top_k = retain_top_k
        self.pinned_lora_dirs = frozenset(pinned_lora_dirs)
        if self.pinned_lora_dirs and (retain_top_k is not None or not retain_layers):
            raise ValueError("pinned retention requires positive retained layers and no dynamic top-K")
        self.selection_dirty = False
        self.mem_manager = mem_manager
        self.entries = {}
        self.clock = 0
        self.reset_access_stats()

    def reset_access_stats(self):
        self.batch_unique_hits = 0
        self.batch_unique_misses = 0
        self.releases = 0
        self.evictions = 0
        self.retained_layer_builds = 0
        self.retained_layer_reuses = 0
        self.serial_layer_builds = 0
        self.promotions = 0
        self.demotions = 0
        self.retained_build_counts = {}

    def access_stats(self):
        return dict(batch_unique_hits=self.batch_unique_hits,
                    batch_unique_misses=self.batch_unique_misses,
                    retain_layers=self.retain_layers, retain_top_k=self.retain_top_k,
                    hotness_scope="fixed_configured_adapter_order" if self.pinned_lora_dirs else "active_request_count",
                    pinned_lora_dirs=sorted(self.pinned_lora_dirs),
                    retained_build_counts=dict(self.retained_build_counts),
                    batch_unique_access_scope=("active_adapter_entry" if self.retain_top_k is None and not self.pinned_lora_dirs
                                               else "selected_ready_retained_prefix_at_admission"),
                    retained_lora_dirs=sorted(d for d, e in self.entries.items() if e.allocated_slots),
                    promotions=self.promotions, demotions=self.demotions, releases=self.releases,
                    evictions=self.evictions,
                    retained_layer_builds=self.retained_layer_builds,
                    retained_layer_reuses=self.retained_layer_reuses,
                    serial_layer_builds=self.serial_layer_builds)

    def acquire_many(self, lora_dirs, infer_adapter=None):
        counts = Counter(lora_dirs)
        if None in counts:
            raise ValueError("retained prefixes require an adapter for every request")
        hits = [directory for directory in counts if directory in self.entries]
        created = []
        try:
            for directory in counts:
                if directory in self.entries:
                    continue
                layers = self.retain_layers if self.retain_top_k is None else 0
                if self.pinned_lora_dirs and directory not in self.pinned_lora_dirs:
                    layers = 0
                if layers:
                    self.ensure_available(
                        (layers * self.prefix_len + self.mem_manager.layer_num - 1) // self.mem_manager.layer_num,
                        protected=counts)
                    pages, locations = self.mem_manager.alloc_layer_storage(
                        layers * self.prefix_len)
                else:
                    pages = locations = torch.empty(0, device=self.mem_manager.device, dtype=torch.long)
                try:
                    self.entries[directory] = RetainedPrefixEntry(
                        locations.view(layers, self.prefix_len), pages,
                        self.prefix_len, 0, ready=layers == 0)
                except Exception:
                    self.mem_manager.free(pages)
                    raise
                created.append(directory)
        except Exception:
            for directory in created:
                self.mem_manager.free(self.entries.pop(directory).pool_indices)
            raise
        for directory, count in counts.items():
            self.entries[directory].ref_count += count
        if self.retain_top_k is not None or self.pinned_lora_dirs:
            selected = self._selected_hot()
            hits = [d for d in counts if d in selected and self.entries[d].allocated_slots
                    and self.entries[d].ready]
        self.batch_unique_hits += len(hits)
        self.batch_unique_misses += len(counts) - len(hits)
        self.selection_dirty = True
        self.touch(counts)

    def touch(self, lora_dirs):
        self.clock += 1
        for directory in set(lora_dirs):
            self.entries[directory].last_used = self.clock

    def ensure_available(self, slots, protected=()):
        while slots > self.mem_manager.can_use_mem_size:
            if not self.evict_one(protected):
                raise RuntimeError("no enough pool space for retained prefixes/workspace; admission budget is inconsistent")

    def _selected_hot(self):
        if self.pinned_lora_dirs:
            return self.pinned_lora_dirs.intersection(self.entries)
        ranked = sorted((d for d, e in self.entries.items() if e.ref_count), key=lambda d: (
            -self.entries[d].ref_count, -bool(self.entries[d].allocated_slots), d))
        return set(ranked[:self.retain_top_k]) if self.retain_layers else set()

    def rebalance(self):
        """Select by live request counts across running and pending mini-batches."""
        if self.retain_top_k is None or not self.selection_dirty:
            return
        selected = self._selected_hot()
        demote = [d for d, e in self.entries.items() if e.ref_count and e.allocated_slots and d not in selected]
        promote = sorted(d for d in selected if not self.entries[d].allocated_slots)
        slots = (self.retain_layers * self.prefix_len + self.mem_manager.layer_num - 1) // self.mem_manager.layer_num
        for directory in demote:
            entry = self.entries[directory]
            self.mem_manager.free(entry.pool_indices)
            entry.pool_indices = entry.pool_indices[:0]
            entry.prefix_indices = entry.prefix_indices[:0]
            entry.ready = True
            self.demotions += 1
        self.ensure_available(len(promote) * slots)
        created = []
        try:
            for directory in promote:
                pages, locations = self.mem_manager.alloc_layer_storage(self.retain_layers * self.prefix_len)
                entry = self.entries[directory]
                entry.pool_indices = pages
                entry.prefix_indices = locations.view(self.retain_layers, self.prefix_len)
                entry.ready = False
                created.append(directory)
        except Exception:
            for directory in created:
                entry = self.entries[directory]
                self.mem_manager.free(entry.pool_indices)
                entry.pool_indices = entry.pool_indices[:0]
                entry.prefix_indices = entry.prefix_indices[:0]
                entry.ready = True
            raise
        self.promotions += len(created)
        self.selection_dirty = False

    def release_many(self, lora_dirs):
        counts = Counter(lora_dirs)
        for directory, count in counts.items():
            if directory not in self.entries or self.entries[directory].ref_count < count:
                raise RuntimeError("retained prefix reference count underflow")
        for directory, count in counts.items():
            entry = self.entries[directory]
            entry.ref_count -= count
            if entry.ref_count == 0:
                self.releases += 1
                if not entry.allocated_slots or not entry.ready:
                    self.mem_manager.free(entry.pool_indices)
                    del self.entries[directory]
        self.selection_dirty = True

    def evict(self, directory):
        entry = self.entries.get(directory)
        if entry is None or entry.ref_count or directory in self.pinned_lora_dirs:
            return False
        self.mem_manager.free(entry.pool_indices)
        del self.entries[directory]
        self.evictions += 1
        return True

    def evict_one(self, protected=()):
        protected = set(protected) | self.pinned_lora_dirs
        candidates = [d for d, e in self.entries.items() if not e.ref_count and d not in protected]
        if not candidates:
            return False
        return self.evict(min(candidates, key=lambda d: (self.entries[d].last_used, d)))

    def evict_many(self, lora_dirs):
        lora_dirs = list(lora_dirs)
        if any(directory in self.entries and self.entries[directory].ref_count
               and directory not in self.pinned_lora_dirs for directory in lora_dirs):
            raise RuntimeError("cannot evict an active retained prefix")
        for directory in lora_dirs:
            self.evict(directory)

    def clear_metadata_if_inactive(self):
        if any(entry.ref_count for entry in self.entries.values()):
            raise RuntimeError("retained prefix reset requires no active requests")
        count = len(self.entries)
        self.entries.clear()
        self.clock = 0
        self.selection_dirty = False
        return count

    def state_dict(self):
        return dict(prefix_len=self.prefix_len,
                    cached_lora_dirs={d for d, e in self.entries.items() if e.allocated_slots},
                    inactive_lora_dirs={d for d, e in self.entries.items() if not e.ref_count},
                    prefix_sizes={d: e.allocated_slots for d, e in self.entries.items()},
                    last_used={d: e.last_used for d, e in self.entries.items()},
                    ref_counts={d: e.ref_count for d, e in self.entries.items()},
                    **self.access_stats())


class RetainedLayerRuntime:
    def __init__(self, static_store, mem_manager, cache, profiler=None, workspace_lora_limit=None,
                 reconstruction_kernel="auto", reconstruction_workers=None):
        if static_store.layer_num != mem_manager.layer_num or static_store.prefix_len != cache.prefix_len:
            raise ValueError("retained prefix dimensions do not match the pool/static store")
        if cache.mem_manager is not mem_manager:
            raise ValueError("retained prefix and workspace must share the same pool")
        if reconstruction_kernel not in ("auto", "bounded"):
            raise ValueError("unknown reconstruction kernel: %s" % reconstruction_kernel)
        if reconstruction_workers is not None and (
                reconstruction_kernel != "bounded" or not isinstance(reconstruction_workers, int)
                or reconstruction_workers <= 0):
            raise ValueError("reconstruction workers require bounded and a positive integer")
        self.static_store = static_store
        self.mem_manager = mem_manager
        self.cache = cache
        self.profiler = profiler
        self.workspace_lora_limit = workspace_lora_limit
        self.mode = LAYER_RETAINED
        self.reconstruction_kernel = reconstruction_kernel
        self.reconstruction_workers = reconstruction_workers
        self.last_reconstruction_kernel = None
        self.last_reconstruction_workers = None
        self._metadata_cache = {}
        self.metadata_builds = 0
        self.metadata_reuses = 0
        self.index_cache = PrefixIndexCache()

    def _prepare_metadata(self, role, directories, infer_adapter, locations):
        if self.reconstruction_kernel != "bounded":
            return prepare_reconstruction_metadata(self.static_store, directories, infer_adapter, locations)
        metadata, cached, reused = cached_reconstruction_metadata(
            self.static_store, directories, infer_adapter, locations, self._metadata_cache.get(role))
        self._metadata_cache[role] = cached
        if reused:
            self.metadata_reuses += 1
        else:
            self.metadata_builds += 1
        return metadata

    def prepare(self, adapter_dirs, infer_adapter, prefix_by_dir=None, reserve_tokens=0):
        if self.reconstruction_kernel == "bounded" and self.reconstruction_workers is None:
            device = self.static_store.base_k[0].device
            self.reconstruction_workers = 12 * torch.cuda.get_device_properties(device).multi_processor_count
        self.last_reconstruction_kernel = "bounded" if self.reconstruction_kernel == "bounded" else "split"
        self.last_reconstruction_workers = self.reconstruction_workers
        self.cache.rebalance()
        self.cache.touch(adapter_dirs)
        device = self.static_store.base_k[0].device
        unique, request_ids = self.index_cache.request_mapping(adapter_dirs, device)
        if self.workspace_lora_limit is not None and len(unique) > self.workspace_lora_limit:
            raise RuntimeError("unique LoRAs exceed the configured workspace limit")
        hot = [d for d in unique if self.cache.entries[d].allocated_slots]
        hot_set = set(hot)
        cold = [d for d in unique if d not in hot_set]
        layers = self.cache.retain_layers
        owns_workspace = False
        try:
            retained_requests = build_metadata = build_locs = None
            scratch_requests = scratch_metadata = cold_metadata = None
            missing = [d for d in hot if not self.cache.entries[d].ready]
            scratch_dirs = unique if layers < self.static_store.layer_num else cold
            workspace_slots = (len(scratch_dirs) * self.static_store.prefix_len
                               + self.mem_manager.layer_num - 1) // self.mem_manager.layer_num
            self.cache.ensure_available(workspace_slots + reserve_tokens)
            scratch_by_dir = {}
            if scratch_dirs:
                scratch = self.mem_manager.alloc_layer_workspace(len(scratch_dirs) * self.static_store.prefix_len)
                owns_workspace = True
                scratch = scratch.view(len(scratch_dirs), self.static_store.prefix_len)
                scratch_by_dir = dict(zip(scratch_dirs, scratch.unbind(0)))
                if layers < self.static_store.layer_num:
                    scratch_requests = scratch
                    scratch_metadata = self._prepare_metadata("scratch", unique, infer_adapter, scratch)
            if layers:
                # Cold adapters address the same scratch cells at every retained layer.
                retained_requests = self.index_cache.stack("retained", [
                    self.cache.entries[d].prefix_indices if d in hot_set
                    else scratch_by_dir[d].expand(layers, -1) for d in unique], dim=1)
                if missing:
                    build_locs = torch.stack([self.cache.entries[d].prefix_indices for d in missing], dim=1)
                    build_metadata = self._prepare_metadata("build", missing, infer_adapter, build_locs[0])
                if cold:
                    cold_metadata = self._prepare_metadata("cold", cold, infer_adapter,
                        torch.stack([scratch_by_dir[d] for d in cold]))
            return RetainedLayerFrame(self, infer_adapter, unique, missing, retained_requests,
                                      build_locs, build_metadata, scratch_requests,
                                      scratch_metadata, owns_workspace, len(hot), len(cold), cold_metadata,
                                      request_ids)
        except Exception:
            if owns_workspace:
                self.mem_manager.free_layer_workspace()
            raise


class RetainedLayerFrame(PrefixIndexedFrame):
    def __init__(self, runtime, infer_adapter, unique, missing, retained_requests,
                 build_locs, build_metadata, scratch_requests, scratch_metadata, owns_workspace,
                 hot_count, cold_count, cold_metadata, request_ids):
        self.runtime = runtime
        self.infer_adapter = infer_adapter
        self.unique_count = len(unique)
        self.missing = missing
        self.retained_requests = retained_requests
        self.build_locs = build_locs
        self.build_metadata = build_metadata
        self.scratch_requests = scratch_requests
        self.scratch_metadata = scratch_metadata
        self.owns_workspace = owns_workspace
        self.hot_count = hot_count
        self.cold_count = cold_count
        self.cold_metadata = cold_metadata
        self.finished = False
        self.prefix_request_ids = request_ids

    def prefix_index_table(self, layer_id):
        if layer_id < self.runtime.cache.retain_layers:
            return self.retained_requests[layer_id]
        return self.scratch_requests

    def buffers(self, layer_id):
        return self.runtime.mem_manager.prefix_workspace_k, self.runtime.mem_manager.prefix_workspace_v

    def before_layer(self, layer_id):
        cache = self.runtime.cache
        if layer_id < cache.retain_layers:
            cache.retained_layer_reuses += self.hot_count - len(self.missing)
            if self.missing:
                self.build_metadata.prefix_locs = self.build_locs[layer_id]
                self._reconstruct(self.build_metadata, layer_id)
                cache.retained_layer_builds += len(self.missing)
                if layer_id == cache.retain_layers - 1:
                    for directory in self.missing:
                        cache.entries[directory].ready = True
                        cache.retained_build_counts[directory] = cache.retained_build_counts.get(directory, 0) + 1
            if self.cold_metadata is not None:
                self._reconstruct(self.cold_metadata, layer_id)
                cache.serial_layer_builds += self.cold_count
        else:
            self._reconstruct(self.scratch_metadata, layer_id)
            cache.serial_layer_builds += self.unique_count

    def _reconstruct(self, metadata, layer_id):
        with profile_interval(self.runtime.profiler, "reconstruct", layer_id):
            if self.runtime.reconstruction_kernel == "bounded":
                reconstruct_prefix_layer_bounded(self.runtime.static_store, self.infer_adapter,
                    metadata, layer_id, *self.buffers(layer_id), workers=self.runtime.reconstruction_workers)
            else:
                reconstruct_prefix_layer(self.runtime.static_store, self.infer_adapter,
                                         metadata, layer_id, *self.buffers(layer_id))

    def before_attention(self, layer_id):
        pass

    def after_attention(self, layer_id):
        pass

    def finish(self):
        if not self.finished and self.owns_workspace:
            self.runtime.mem_manager.free_layer_workspace()
        self.finished = True
