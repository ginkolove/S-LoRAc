"""Full prefix KV built by ordinary LoRA prefill in the Unified Paging pool."""

from collections import Counter
from dataclasses import dataclass

import torch

from slora.common.prefix_modes import FULL_REPREFILL, FULL_REPREFILL_MODES, FULL_PREFIX_CACHE_MODES
from slora.common.prefix_indices import PrefixIndexCache, PrefixIndexedFrame


@dataclass
class FullPrefixEntry:
    lora_dir: str
    prefix_indices: torch.Tensor
    prefix_len: int
    ref_count: int = 0
    last_used: int = 0


class FullPrefixKVCache:
    def __init__(self, prefix_len, mem_manager, pinned_lora_dirs=(), reuse_next_batch=False):
        self.prefix_len = int(prefix_len)
        self.mem_manager = mem_manager
        self.entries = {}
        self.pinned_lora_dirs = frozenset(pinned_lora_dirs)
        self.reuse_next_batch = reuse_next_batch
        self.shared_prefix_ids = None
        self.clock = 0
        self.reset_access_stats()

    def reset_access_stats(self):
        self.batch_unique_hits = 0
        self.idle_hits = self.pinned_idle_hits = 0
        self.batch_unique_misses = 0
        self.evictions = 0
        self.prefix_prefill_builds = 0
        self.prefix_prefill_tokens = 0
        self.prefix_build_counts = {}

    def access_stats(self):
        return dict(batch_unique_hits=self.batch_unique_hits,
                    reuse_next_batch=self.reuse_next_batch,
                    active_shared_hits=self.batch_unique_hits - self.idle_hits,
                    idle_pinned_hits=self.pinned_idle_hits,
                    idle_unpinned_hits=self.idle_hits - self.pinned_idle_hits,
                    batch_unique_misses=self.batch_unique_misses, evictions=self.evictions,
                    prefix_prefill_builds=self.prefix_prefill_builds,
                    prefix_prefill_tokens=self.prefix_prefill_tokens,
                    pinned_lora_dirs=sorted(self.pinned_lora_dirs),
                    prefix_build_counts=dict(self.prefix_build_counts))

    def touch(self, lora_dirs):
        self.clock += 1
        for directory in set(lora_dirs):
            self.entries[directory].last_used = self.clock

    def _validate_prefix_inputs(self, requests):
        prefix_ids = None
        for request in requests:
            tokens = tuple(request.get("prefix_input_ids", ()))
            if len(tokens) != self.prefix_len:
                raise ValueError("full reprefill requires the actual shared prefix tokens")
            if prefix_ids is not None and tokens != prefix_ids:
                raise ValueError("all requests must use the same shared prefix")
            prefix_ids = tokens
        if self.shared_prefix_ids is not None and prefix_ids != self.shared_prefix_ids:
            raise ValueError("shared prefix changed; reset the idle runtime before changing it")
        return prefix_ids

    def _record_prefix_builds(self, misses):
        self.prefix_prefill_builds += len(misses)
        self.prefix_prefill_tokens += len(misses) * self.prefix_len
        for directory in misses:
            self.prefix_build_counts[directory] = self.prefix_build_counts.get(directory, 0) + 1

    def acquire_many(self, requests, build_prefixes):
        counts = Counter(request["adapter_dir"] for request in requests)
        if None in counts:
            raise ValueError("full reprefill requires a LoRA for every request")
        prefix_ids = self._validate_prefix_inputs(requests)
        hits = [directory for directory in counts if directory in self.entries]
        idle_hits = [d for d in hits if self.entries[d].ref_count == 0]
        misses = [directory for directory in counts if directory not in self.entries]
        created = []
        for directory in hits:
            self.entries[directory].ref_count += counts[directory]
        try:
            for directory in misses:
                while self.prefix_len > self.mem_manager.can_use_mem_size:
                    if not self.evict_one():
                        raise RuntimeError("no pool space for a full prefix; admission budget is inconsistent")
                indices = self.mem_manager.alloc(self.prefix_len)
                self.entries[directory] = FullPrefixEntry(
                    directory, indices, self.prefix_len, counts[directory])
                created.append(directory)
            if misses:
                build_prefixes(misses, prefix_ids,
                               {directory: self.entries[directory].prefix_indices for directory in misses})
        except Exception:
            for directory in created:
                entry = self.entries.pop(directory)
                self.mem_manager.free(entry.prefix_indices)
            for directory in hits:
                self.entries[directory].ref_count -= counts[directory]
            raise
        self.shared_prefix_ids = prefix_ids
        self.touch(counts)
        self.batch_unique_hits += len(hits)
        self.idle_hits += len(idle_hits)
        self.pinned_idle_hits += sum(d in self.pinned_lora_dirs for d in idle_hits)
        self.batch_unique_misses += len(misses)
        self._record_prefix_builds(misses)
        return {directory: self.entries[directory].prefix_indices for directory in counts}

    def release_many(self, lora_dirs):
        counts = Counter(lora_dirs)
        for directory, count in counts.items():
            if directory not in self.entries or self.entries[directory].ref_count < count:
                raise RuntimeError("full prefix reference count underflow")
        for directory, count in counts.items():
            self.entries[directory].ref_count -= count

    def evict_one(self):
        inactive = [entry for entry in self.entries.values()
                    if entry.ref_count == 0 and entry.lora_dir not in self.pinned_lora_dirs]
        if not inactive:
            return False
        victim = min(inactive, key=lambda entry: (entry.last_used, entry.lora_dir))
        return self.evict(victim.lora_dir)

    def evict(self, directory):
        entry = self.entries.get(directory)
        if entry is None or entry.ref_count or directory in self.pinned_lora_dirs:
            return False
        self.mem_manager.free(entry.prefix_indices)
        del self.entries[directory]
        self.evictions += 1
        return True

    def evict_many(self, lora_dirs):
        for directory in lora_dirs:
            self.evict(directory)

    def clear_metadata_if_inactive(self):
        if any(entry.ref_count for entry in self.entries.values()):
            raise RuntimeError("full prefix runtime reset requires no active requests")
        count = len(self.entries)
        self.entries.clear()
        self.shared_prefix_ids = None
        self.clock = 0
        return count

    def state_dict(self):
        return dict(prefix_len=self.prefix_len, cached_lora_dirs=set(self.entries),
                    inactive_lora_dirs={directory for directory, entry in self.entries.items()
                                        if entry.ref_count == 0},
                    prefix_sizes={directory: entry.prefix_len for directory, entry in self.entries.items()},
                    last_used={directory: entry.last_used for directory, entry in self.entries.items()},
                    ref_counts={directory: entry.ref_count for directory, entry in self.entries.items()},
                    **self.access_stats())


class FullReprefillRuntime:
    def __init__(self, prefix_len, mem_manager, cache, profiler=None):
        self.prefix_len = int(prefix_len)
        self.mem_manager = mem_manager
        self.cache = cache
        self.profiler = profiler
        self.mode = FULL_REPREFILL
        self.index_cache = PrefixIndexCache()

    def prepare(self, adapter_dirs, infer_adapter=None, prefix_by_dir=None, reserve_tokens=0):
        self.cache.touch(adapter_dirs)
        device = self.cache.entries[adapter_dirs[0]].prefix_indices.device
        unique, request_ids = self.index_cache.request_mapping(adapter_dirs, device)
        prefix_locs = self.index_cache.stack("full", [self.cache.entries[directory].prefix_indices
                                                     for directory in unique])
        return FullPrefixFrame(self, prefix_locs, request_ids)


class FullPrefixFrame(PrefixIndexedFrame):
    def __init__(self, runtime, prefix_locs, request_ids):
        self.runtime = runtime
        self.prefix_len = runtime.prefix_len
        self.prefix_locs = prefix_locs
        self.prefix_request_ids = request_ids

    def prefix_index_table(self, layer_id):
        return self.prefix_locs

    def buffers(self, layer_id):
        return (self.runtime.mem_manager.key_buffer[layer_id],
                self.runtime.mem_manager.value_buffer[layer_id])

    def before_layer(self, layer_id):
        pass

    def before_attention(self, layer_id):
        pass

    def after_attention(self, layer_id):
        pass

    def finish(self):
        pass
