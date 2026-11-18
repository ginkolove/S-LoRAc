"""Per-adapter elastic retained layers; existing reconstruction/attention kernels.

Persistent cells are page-major so extending or shrinking an entry never moves
the surviving KV. Private token slots remain full-layer allocations.
"""

from collections import Counter, deque
from dataclasses import dataclass

import torch

from slora.common.lowra_retained import RetainedPrefixKVCache, RetainedLayerRuntime
from slora.common.lowra_layer_runtime import profile_interval
from slora.common.prefix_indices import PrefixIndexedFrame
from slora.common.prefix_modes import LAYER_ADAPTIVE


@dataclass
class AdaptivePrefixEntry:
    prefix_indices: torch.Tensor
    pool_indices: torch.Tensor
    prefix_len: int
    ref_count: int = 0
    ready_layers: int = 0
    last_used: int = 0

    @property
    def layers(self):
        return self.prefix_indices.shape[0]

    @property
    def allocated_slots(self):
        return self.pool_indices.numel()

    @property
    def ready(self):
        return self.ready_layers == self.layers


class AdaptivePrefixKVCache(RetainedPrefixKVCache):
    def __init__(self, prefix_len, mem_manager):
        super().__init__(prefix_len, 0, mem_manager)
        self.plan_version = 0

    def _empty(self):
        empty = torch.empty(0, device=self.mem_manager.device, dtype=torch.long)
        return AdaptivePrefixEntry(empty.view(0, self.prefix_len), empty, self.prefix_len)

    def acquire_many(self, lora_dirs, infer_adapter=None):
        counts = Counter(lora_dirs)
        if None in counts:
            raise ValueError("adaptive prefixes require adapters")
        for directory, count in counts.items():
            if directory not in self.entries:
                self.entries[directory] = self._empty()
            entry = self.entries[directory]
            hit = bool(entry.layers and entry.ready)
            self.batch_unique_hits += int(hit)
            self.batch_unique_misses += int(not hit)
            entry.ref_count += count
        self.touch(counts)

    def rebalance(self):
        # Only the router can change allocations; prepare must honor its budget.
        pass

    def _resize(self, entry, layers):
        memory = self.mem_manager
        slots = (layers * self.prefix_len + memory.layer_num - 1) // memory.layer_num
        old = entry.allocated_slots
        if slots < old:
            memory.free(entry.pool_indices[slots:])
            entry.pool_indices = entry.pool_indices[:slots]
        elif slots > old:
            extra = memory.alloc(slots - old)
            entry.pool_indices = torch.cat((entry.pool_indices, extra))
        offsets = torch.arange(memory.layer_num, device=memory.device, dtype=torch.long)
        cells = (entry.pool_indices[:, None] + offsets[None, :] * memory.tot_size).reshape(-1)
        entry.prefix_indices = cells[:layers * self.prefix_len].view(layers, self.prefix_len)
        entry.ready_layers = min(entry.ready_layers, layers)

    def apply_plan(self, layers_by_dir, evictions=()):
        if self.mem_manager._workspace_indices is not None:
            raise RuntimeError("cannot change retention during an active forward")
        targets = dict(layers_by_dir)
        for layers in targets.values():
            if not isinstance(layers, int) or not 0 <= layers <= self.mem_manager.layer_num:
                raise ValueError("invalid adaptive retained layer count")
        for directory in evictions:
            if directory in self.entries and self.entries[directory].ref_count:
                raise RuntimeError("cannot evict active private dependencies")
        # Validate the final persistent allocation before releasing anything.
        available = self.mem_manager.can_use_mem_size
        available += sum(e.allocated_slots for d, e in self.entries.items()
                         if d in targets or d in evictions)
        required = sum((k * self.prefix_len + self.mem_manager.layer_num - 1)
                       // self.mem_manager.layer_num for k in targets.values())
        if required > available:
            raise RuntimeError("adaptive plan exceeds allocator capacity")
        self.evict_many(evictions)
        for directory, layers in targets.items():
            if directory not in self.entries:
                self.entries[directory] = self._empty()
            entry = self.entries[directory]
            if layers < entry.layers:
                self._resize(entry, layers)
                self.demotions += 1
        for directory, layers in targets.items():
            entry = self.entries[directory]
            if layers > entry.layers:
                self._resize(entry, layers)
                self.promotions += 1
        self.plan_version += 1

    def state_dict(self):
        state = super().state_dict()
        state.update(retained_layers={d: e.layers for d, e in self.entries.items()},
                     ready_layers={d: e.ready_layers for d, e in self.entries.items()},
                     plan_version=self.plan_version)
        return state

    def access_stats(self):
        state = super().access_stats()
        state.update(hotness_scope="measured_short_window_planner",
                     retained_layers={d: e.layers for d, e in self.entries.items()},
                     plan_version=getattr(self, "plan_version", 0))
        return state


class AdaptiveLayerRuntime(RetainedLayerRuntime):
    def __init__(self, *args, sample_interval=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = LAYER_ADAPTIVE
        self.sample_interval = sample_interval
        self.forward_count = 0
        self.sample_seq = 0
        self.profile_records = deque(maxlen=24)
        self.pending_profile = deque(maxlen=8)
        self.active_profile = None

    def start_observation(self, adapter_dirs, private_tokens, is_prefill):
        self.collect_observations()
        self.forward_count += 1
        # Early warmup samples are collected but not fitted by the planner.
        enabled = is_prefill or self.forward_count <= 6 or self.forward_count % self.sample_interval == 0
        if not enabled or not torch.cuda.is_available():
            self.active_profile = None
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        ranks = self.static_store.lora_ranks
        self.active_profile = dict(start=start, calls=[],
            batch=len(adapter_dirs), private_tokens=private_tokens, prefill=is_prefill,
            rank_sum=sum(ranks[d] for d in adapter_dirs), prefix_len=self.static_store.prefix_len)

    def end_observation(self):
        if self.active_profile is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.active_profile.update(end=end)
            self.pending_profile.append(self.active_profile)
            self.active_profile = None

    def collect_observations(self):
        while self.pending_profile and self.pending_profile[0]["end"].query():
            sample = self.pending_profile.popleft()
            self.sample_seq += 1
            groups = {}
            reconstruct_ms = 0.0
            for start, end, count, ranks in sample["calls"]:
                elapsed = start.elapsed_time(end)
                reconstruct_ms += elapsed
                times = groups.setdefault((count, ranks), [])
                times.append(elapsed)
            total_ms = sample["start"].elapsed_time(sample["end"])
            row = {key: sample[key] for key in ("batch", "private_tokens", "rank_sum", "prefill", "prefix_len")}
            row.update(seq=self.sample_seq, total_ms=total_ms,
                       core_ms=max(0.001, total_ms - reconstruct_ms),
                       reconstruct_ms=reconstruct_ms,
                       reconstruction=[dict(unique=count, rank_sum=ranks, ms=sum(times) / len(times))
                                       for (count, ranks), times in groups.items()])
            self.profile_records.append(row)

    def prepare(self, adapter_dirs, infer_adapter, prefix_by_dir=None, reserve_tokens=0):
        if self.reconstruction_kernel == "bounded" and self.reconstruction_workers is None:
            self.reconstruction_workers = 12 * torch.cuda.get_device_properties(
                self.static_store.base_k[0].device).multi_processor_count
        self.last_reconstruction_kernel = self.reconstruction_kernel
        self.last_reconstruction_workers = self.reconstruction_workers
        self.cache.touch(adapter_dirs)
        device = self.static_store.base_k[0].device
        unique, request_ids = self.index_cache.request_mapping(adapter_dirs, device)
        if self.workspace_lora_limit is not None and len(unique) > self.workspace_lora_limit:
            raise RuntimeError("adaptive workspace adapter limit exceeded")
        scratch_dirs = [d for d in unique if self.cache.entries[d].layers < self.static_store.layer_num]
        slots = (len(scratch_dirs) * self.static_store.prefix_len + self.static_store.layer_num - 1) // self.static_store.layer_num
        self.cache.ensure_available(slots + reserve_tokens)
        scratch_by_dir = {}
        owns_workspace = False
        try:
            if scratch_dirs:
                scratch = self.mem_manager.alloc_layer_workspace(len(scratch_dirs) * self.static_store.prefix_len)
                owns_workspace = True
                scratch_by_dir = dict(zip(scratch_dirs, scratch.view(len(scratch_dirs), -1).unbind(0)))
            tables, metadata, missing = [], [], []
            for layer in range(self.static_store.layer_num):
                rows = [self.cache.entries[d].prefix_indices[layer]
                        if layer < self.cache.entries[d].layers else scratch_by_dir[d] for d in unique]
                table = self.index_cache.stack(("adaptive_table", layer), rows)
                tables.append(table)
                build = [d for d in unique if layer >= self.cache.entries[d].ready_layers]
                missing.append(build)
                if build:
                    locations = self.index_cache.stack(("adaptive_build", layer), [
                        self.cache.entries[d].prefix_indices[layer]
                        if layer < self.cache.entries[d].layers else scratch_by_dir[d] for d in build])
                    metadata.append(self._prepare_metadata(("adaptive", layer), build, infer_adapter, locations))
                else:
                    metadata.append(None)
            return AdaptiveLayerFrame(self, infer_adapter, tables, metadata, missing, unique, request_ids, owns_workspace)
        except Exception:
            if owns_workspace:
                self.mem_manager.free_layer_workspace()
            raise


class AdaptiveLayerFrame(PrefixIndexedFrame):
    def __init__(self, runtime, infer_adapter, tables, metadata, missing, unique, request_ids, owns_workspace):
        self.runtime, self.tables, self.metadata, self.missing = runtime, tables, metadata, missing
        self.infer_adapter = infer_adapter
        self.unique, self.prefix_request_ids = unique, request_ids
        self.owns_workspace, self.finished = owns_workspace, False

    def prefix_index_table(self, layer_id):
        return self.tables[layer_id]

    def buffers(self, layer_id):
        return self.runtime.mem_manager.prefix_workspace_k, self.runtime.mem_manager.prefix_workspace_v

    def before_layer(self, layer_id):
        from slora.common.lowra_layer_runtime import reconstruct_prefix_layer, reconstruct_prefix_layer_bounded
        runtime, meta = self.runtime, self.metadata[layer_id]
        build = self.missing[layer_id]
        runtime.cache.retained_layer_reuses += len(self.unique) - len(build)
        if meta is None:
            return
        observation = runtime.active_profile
        start = end = None
        if observation is not None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        with profile_interval(runtime.profiler, "reconstruct", layer_id):
            if runtime.reconstruction_kernel == "bounded":
                reconstruct_prefix_layer_bounded(runtime.static_store, self.infer_adapter,
                    meta, layer_id, *self.buffers(layer_id), workers=runtime.reconstruction_workers)
            else:
                reconstruct_prefix_layer(runtime.static_store, self.infer_adapter,
                                         meta, layer_id, *self.buffers(layer_id))
        if observation is not None:
            end.record()
            observation["calls"].append((start, end, len(build),
                sum(runtime.static_store.lora_ranks[d] for d in build)))
        for directory in build:
            entry = runtime.cache.entries[directory]
            if layer_id < entry.layers:
                entry.ready_layers = layer_id + 1
                runtime.cache.retained_layer_builds += 1
            else:
                runtime.cache.serial_layer_builds += 1

    def before_attention(self, layer_id):
        pass

    def after_attention(self, layer_id):
        pass

    def finish(self):
        if not self.finished and self.owns_workspace:
            self.runtime.mem_manager.free_layer_workspace()
        self.finished = True
