from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch
from slora.common.prefix_indices import PrefixIndexCache, PrefixIndexedFrame

from slora.common.lowra_reconstruct import lowra_reconstruct_batched
from slora.common.lowra_reconstruct_kv import lowra_reconstruct_kv_batched
from slora.common.lowra_reconstruct_bounded import lowra_reconstruct_bounded
from slora.common.lowra_key_rope import rotate_delta_k_add_base
from slora.common.prefix_modes import FULL_REPREFILL, LAYER_SERIAL, normalize_prefix_mode


PREFIX_MODES = (FULL_REPREFILL, LAYER_SERIAL, "layer_prefetch")


class LowRAProfileEvents:
    """CUDA intervals and NVTX ranges, collected without per-layer synchronization."""

    def __init__(self):
        self.intervals = []
        self.phase = "prefill"
        self.origin = None

    @contextmanager
    def measure(self, name, layer_id, stream=None):
        stream = stream or torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        label = "lowra/%s/%s/%s" % (self.phase, layer_id, name)
        torch.cuda.nvtx.range_push(label)
        try:
            with torch.profiler.record_function(label):
                yield
        finally:
            end.record(stream)
            torch.cuda.nvtx.range_pop()
            self.intervals.append((self.phase, layer_id, name, start, end))

    def collect(self):
        torch.cuda.synchronize()
        return [dict(phase=phase, layer=layer, operation=name,
                     milliseconds=start.elapsed_time(end),
                     start_ms=self.origin.elapsed_time(start) if self.origin else None,
                     end_ms=self.origin.elapsed_time(end) if self.origin else None)
                for phase, layer, name, start, end in self.intervals]

    def clear(self):
        self.intervals.clear()
        self.origin = None


@contextmanager
def profile_interval(profiler, name, layer_id, stream=None):
    if profiler is None:
        yield
    else:
        with profiler.measure(name, layer_id, stream):
            yield


def run_profiled(profiler, name, layer_id, function, *args, **kwargs):
    with profile_interval(profiler, name, layer_id):
        return function(*args, **kwargs)


@dataclass
class ReconstructionMetadata:
    prefix_locs: torch.Tensor
    k_b_locs: torch.Tensor
    v_b_locs: torch.Tensor
    rank_starts: torch.Tensor
    rank_lens: torch.Tensor
    scalings: torch.Tensor


def prepare_reconstruction_metadata(static, lora_dirs, infer_adapter, prefix_locs):
    device = prefix_locs.device
    ranges = [static.get_lora_rank_range_by_dir(directory) for directory in lora_dirs]
    ranks = [end - start for start, end in ranges]
    max_rank = max(ranks)
    k_locs = torch.zeros((len(lora_dirs), max_rank), dtype=torch.long, device=device)
    v_locs = torch.zeros_like(k_locs)
    starts = infer_adapter.a_start.cpu().tolist()
    lengths = infer_adapter.a_len.cpu().tolist()
    adapter_ids = [infer_adapter.idx_map[directory] for directory in lora_dirs]
    for row, (adapter_id, rank) in enumerate(zip(adapter_ids, ranks)):
        if lengths[adapter_id] != 4 * rank:
            raise ValueError("LowRA requires the packed q/k/v/o adapter layout")
        start = starts[adapter_id]
        locs = infer_adapter.a_loc[start:start + lengths[adapter_id]]
        k_locs[row, :rank] = locs[rank:2 * rank]
        v_locs[row, :rank] = locs[2 * rank:3 * rank]
    return ReconstructionMetadata(
        prefix_locs, k_locs, v_locs,
        torch.tensor([start for start, _ in ranges], device=device, dtype=torch.long),
        torch.tensor(ranks, device=device, dtype=torch.long),
        infer_adapter.a_scaling[torch.tensor(adapter_ids, device=device, dtype=torch.long)],
    )


def cached_reconstruction_metadata(static, unique, infer_adapter, prefix_locs, cached):
    # Cache source indices only; workspace and retained-prefix destinations can move.
    names = ("a_loc", "a_start", "a_len", "a_scaling")
    sources = tuple(getattr(infer_adapter, name, None) for name in names)
    if not all(isinstance(t, torch.Tensor) for t in sources):
        return prepare_reconstruction_metadata(static, unique, infer_adapter, prefix_locs), None, False
    versions = tuple(t._version for t in sources)
    key = (tuple(unique), tuple(infer_adapter.idx_map[d] for d in unique),
           tuple(static.get_lora_rank_range_by_dir(d) for d in unique))
    if (cached is not None and cached[0] == key and cached[2] == versions
            and all(old is new for old, new in zip(cached[1], sources))):
        return replace(cached[3], prefix_locs=prefix_locs), cached, True
    metadata = prepare_reconstruction_metadata(static, unique, infer_adapter, prefix_locs)
    return metadata, (key, sources, versions, replace(metadata, prefix_locs=None)), False


def reconstruct_prefix_layer(static, infer_adapter, metadata, layer_id, out_k, out_v):
    key_rope = getattr(static, "key_rope_cos", None) is not None
    b_buffer = infer_adapter.mem_manager.value_buffer[layer_id]
    if b_buffer.shape[1:] != (static.tp_k_head_num, static.head_dim):
        raise ValueError("adapter B packing does not match the local KV dimensions")
    for base, mini, b_locs, out, embed in (
        (static.base_k[layer_id], static.mini_k[layer_id], metadata.k_b_locs,
         out_k, static.kv_embed_dim_k),
        (static.base_v[layer_id], static.mini_v[layer_id], metadata.v_b_locs,
         out_v, static.kv_embed_dim_v),
    ):
        lowra_reconstruct_batched(
            base.view(static.prefix_len, embed), mini, b_buffer.view(-1, embed),
            out.view(-1, embed), metadata.prefix_locs, b_locs,
            metadata.rank_starts, metadata.rank_lens, metadata.scalings,
            b_packed=True,
            add_base=not key_rope or out is out_v,
        )
    if key_rope:
        rotate_delta_k_add_base(static.base_k[layer_id], out_k, metadata.prefix_locs,
                                static.key_rope_cos, static.key_rope_sin)


def reconstruct_prefix_layer_fused_kv(static, infer_adapter, metadata, layer_id, out_k, out_v):
    key_rope = getattr(static, "key_rope_cos", None) is not None
    b_buffer = infer_adapter.mem_manager.value_buffer[layer_id]
    if (static.kv_embed_dim_k != static.kv_embed_dim_v
            or b_buffer.shape[1:] != (static.tp_k_head_num, static.head_dim)):
        raise ValueError("fused reconstruction requires matching packed MHA K/V dimensions")
    prefix, embed = static.prefix_len, static.kv_embed_dim_k
    lowra_reconstruct_kv_batched(
        static.base_k[layer_id].view(prefix, embed),
        static.base_v[layer_id].view(prefix, embed),
        static.mini_k[layer_id], static.mini_v[layer_id],
        b_buffer.view(-1, embed), out_k.view(-1, embed), out_v.view(-1, embed),
        metadata.prefix_locs, metadata.k_b_locs, metadata.v_b_locs,
        metadata.rank_starts, metadata.rank_lens, metadata.scalings,
        b_packed=True,
        add_k_base=not key_rope,
    )
    if key_rope:
        rotate_delta_k_add_base(static.base_k[layer_id], out_k, metadata.prefix_locs,
                                static.key_rope_cos, static.key_rope_sin)


def reconstruct_prefix_layer_bounded(static, infer_adapter, metadata, layer_id, out_k, out_v,
                                     workers=648, block_r=16, rank_loop="max"):
    if block_r not in (16, 32):
        raise ValueError("bounded reconstruction block_r must be 16 or 32")
    if rank_loop not in ("max", "actual"):
        raise ValueError("rank_loop must be 'max' or 'actual'")
    key_rope = getattr(static, "key_rope_cos", None) is not None
    b_buffer = infer_adapter.mem_manager.value_buffer[layer_id]
    if b_buffer.shape[1:] != (static.tp_k_head_num, static.head_dim):
        raise ValueError("adapter B packing does not match the local KV dimensions")
    for base, mini, b_locs, out, embed in (
        (static.base_k[layer_id], static.mini_k[layer_id], metadata.k_b_locs, out_k, static.kv_embed_dim_k),
        (static.base_v[layer_id], static.mini_v[layer_id], metadata.v_b_locs, out_v, static.kv_embed_dim_v),
    ):
        lowra_reconstruct_bounded(base.view(static.prefix_len, embed), mini, b_buffer.view(-1, embed),
                                  out.view(-1, embed), metadata.prefix_locs, b_locs,
                                  metadata.rank_starts, metadata.rank_lens, metadata.scalings,
                                  workers=workers, launch_config=(32, 128, block_r, 4), b_packed=True,
                                  rank_loop=rank_loop,
                                  add_base=not key_rope or out is out_v)
    if key_rope:
        rotate_delta_k_add_base(static.base_k[layer_id], out_k, metadata.prefix_locs,
                                static.key_rope_cos, static.key_rope_sin)


def select_prefetch_reconstruction(static, metadata, capability):
    # Historical A100 shape policy; reprofile after the packed-B address fix.
    # Keep the existing dispatch unchanged while validating correctness.
    if (capability == (8, 0) and static.kv_embed_dim_k == static.kv_embed_dim_v == 4096
            and metadata.k_b_locs.shape[1] <= 16 and static.prefix_len >= 4096
            and metadata.prefix_locs.shape[0] * static.prefix_len >= 32768):
        return "bounded"
    return "fused"


class LowRALayerRuntime:
    def __init__(self, static_store, mem_manager, mode, profiler=None, workspace_lora_limit=None,
                 prefetch_reconstruction="auto", prefetch_workers=None,
                 reconstruction_kernel="auto", reconstruction_workers=None):
        mode = normalize_prefix_mode(mode)
        if mode not in PREFIX_MODES:
            raise ValueError("unknown prefix mode: %s" % mode)
        if reconstruction_kernel not in ("auto", "bounded"):
            raise ValueError("unknown reconstruction kernel: %s" % reconstruction_kernel)
        if reconstruction_kernel == "bounded":
            if mode not in (LAYER_SERIAL, "layer_prefetch"):
                raise ValueError("bounded reconstruction requires a serial or prefetch layer mode")
            if prefetch_reconstruction not in ("auto", "bounded"):
                raise ValueError("conflicting reconstruction kernel overrides")
            prefetch_reconstruction = "bounded"
        if reconstruction_workers is not None:
            if reconstruction_kernel != "bounded":
                raise ValueError("reconstruction workers require the bounded kernel")
            if prefetch_workers is not None and prefetch_workers != reconstruction_workers:
                raise ValueError("conflicting reconstruction worker overrides")
            prefetch_workers = reconstruction_workers
        if prefetch_reconstruction not in ("auto", "split", "fused", "bounded"):
            raise ValueError("unknown prefetch reconstruction: %s" % prefetch_reconstruction)
        if prefetch_workers is not None and (not isinstance(prefetch_workers, int) or prefetch_workers <= 0):
            raise ValueError("prefetch workers must be a positive integer")
        self.static_store = static_store
        self.mem_manager = mem_manager
        self.mode = mode
        self.profiler = profiler
        self.workspace_lora_limit = workspace_lora_limit
        self.prefetch_reconstruction = prefetch_reconstruction
        self.prefetch_workers = prefetch_workers
        self.reconstruction_kernel = reconstruction_kernel
        self.last_reconstruction_kernel = None
        self.last_reconstruction_workers = None
        self._default_prefetch_workers = None
        self.capability = None
        self.reconstruct_stream = torch.cuda.Stream() if mode == "layer_prefetch" else None
        self.events = {}
        self._metadata_cache = None
        self.metadata_builds = 0
        self.metadata_reuses = 0
        self.index_cache = PrefixIndexCache()

    def _prepare_metadata(self, unique, infer_adapter, prefix_locs):
        if self.mode != "layer_prefetch" and self.reconstruction_kernel != "bounded":
            return prepare_reconstruction_metadata(self.static_store, unique, infer_adapter, prefix_locs)
        metadata, self._metadata_cache, reused = cached_reconstruction_metadata(
            self.static_store, unique, infer_adapter, prefix_locs, self._metadata_cache)
        if reused:
            self.metadata_reuses += 1
        else:
            self.metadata_builds += 1
        return metadata

    def prepare(self, adapter_dirs, infer_adapter, prefix_by_dir=None, reserve_tokens=0):
        if any(directory is None for directory in adapter_dirs):
            raise ValueError("LowRA profiling requires a LoRA adapter for every request")
        device = self.static_store.base_k[0].device
        unique, request_ids = self.index_cache.request_mapping(adapter_dirs, device)
        prefix_len = self.static_store.prefix_len
        if self.mode == FULL_REPREFILL:
            prefix_locs = self.index_cache.stack("full", [prefix_by_dir[directory] for directory in unique])
            metadata = None
        else:
            tokens = len(unique) * prefix_len
            if self.workspace_lora_limit is not None and len(unique) > self.workspace_lora_limit:
                raise RuntimeError("unique LoRAs exceed the configured workspace limit")
            prefix_locs = self.mem_manager.alloc_layer_workspace(tokens).view(-1, prefix_len)
        try:
            if self.mode != FULL_REPREFILL:
                metadata = self._prepare_metadata(unique, infer_adapter, prefix_locs)
            frame = LowRALayerFrame(self, infer_adapter, metadata, prefix_locs, len(unique), request_ids)
            if self.mode == "layer_prefetch" or self.reconstruction_kernel == "bounded":
                if self.prefetch_reconstruction == "auto":
                    if self.capability is None:
                        self.capability = (torch.cuda.get_device_capability(device) if device.type == "cuda" else None)
                    frame.prefetch_reconstruction = select_prefetch_reconstruction(
                        self.static_store, metadata, self.capability)
                else:
                    frame.prefetch_reconstruction = self.prefetch_reconstruction
                if frame.prefetch_reconstruction == "bounded":
                    if self.prefetch_workers is not None:
                        frame.prefetch_workers = self.prefetch_workers
                    else:
                        if self._default_prefetch_workers is None:
                            sms = torch.cuda.get_device_properties(device).multi_processor_count
                            self._default_prefetch_workers = 12 * sms
                        frame.prefetch_workers = self._default_prefetch_workers
            self.last_reconstruction_kernel = (frame.prefetch_reconstruction
                if self.mode == "layer_prefetch" or self.reconstruction_kernel == "bounded" else "split")
            self.last_reconstruction_workers = (frame.prefetch_workers
                if self.last_reconstruction_kernel == "bounded" else None)
        except Exception:
            if self.mode != FULL_REPREFILL:
                self.mem_manager.free_layer_workspace()
            raise
        return frame


class LowRALayerFrame(PrefixIndexedFrame):
    def __init__(self, runtime, infer_adapter, metadata, prefix_locs, unique_count, request_ids=None):
        self.runtime = runtime
        self.infer_adapter = infer_adapter
        self.metadata = metadata
        self.prefix_locs = prefix_locs
        self.prefix_request_ids = request_ids
        self.unique_count = unique_count
        self.ready = None
        self.finished = False
        self.metadata_recorded = False
        self.prefetch_reconstruction = "fused"
        self.prefetch_workers = 648

    def prefix_index_table(self, layer_id):
        return self.prefix_locs

    def buffers(self, layer_id):
        memory = self.runtime.mem_manager
        if self.runtime.mode == FULL_REPREFILL:
            return memory.key_buffer[layer_id], memory.value_buffer[layer_id]
        return memory.prefix_workspace_k, memory.prefix_workspace_v

    def _reconstruct(self, layer_id):
        with profile_interval(self.runtime.profiler, "reconstruct", layer_id):
            if ((self.runtime.mode == "layer_prefetch" and self.prefetch_reconstruction == "bounded")
                    or (self.runtime.mode == LAYER_SERIAL
                        and getattr(self.runtime, "reconstruction_kernel", "auto") == "bounded")):
                reconstruct_prefix_layer_bounded(self.runtime.static_store, self.infer_adapter,
                                                  self.metadata, layer_id, *self.buffers(layer_id),
                                                  workers=self.prefetch_workers)
                return
            reconstruct = (reconstruct_prefix_layer_fused_kv
                           if self.runtime.mode == "layer_prefetch" and self.prefetch_reconstruction == "fused"
                           else reconstruct_prefix_layer)
            reconstruct(self.runtime.static_store, self.infer_adapter,
                        self.metadata, layer_id, *self.buffers(layer_id))

    def before_layer(self, layer_id):
        if self.runtime.mode == FULL_REPREFILL:
            return
        if self.runtime.mode == LAYER_SERIAL:
            self._reconstruct(layer_id)
        elif layer_id == 0:
            # Even the first prefix can overlap this layer's norm and QKV work.
            self._prefetch_layer(layer_id)

    def before_attention(self, layer_id):
        if self.runtime.mode == "layer_prefetch":
            with profile_interval(self.runtime.profiler, "wait_prefix", layer_id):
                torch.cuda.current_stream().wait_event(self.ready)

    def after_attention(self, layer_id):
        """Overwrite scratch only after the final prefix reader, before O/FFN."""
        if self.runtime.mode != "layer_prefetch" or layer_id + 1 == self.runtime.static_store.layer_num:
            return
        self._prefetch_layer(layer_id + 1)

    def _prefetch_layer(self, layer_id):
        if layer_id not in self.runtime.events:
            self.runtime.events[layer_id] = (torch.cuda.Event(), torch.cuda.Event())
        consumed, ready = self.runtime.events[layer_id]
        consumed.record(torch.cuda.current_stream())
        stream = self.runtime.reconstruct_stream
        with torch.cuda.stream(stream):
            stream.wait_event(consumed)
            self.ready = ready
            try:
                self._reconstruct(layer_id)
            finally:
                # Also fence partially submitted work if reconstruction raises.
                self.ready.record(stream)
                if not self.metadata_recorded:
                    for tensor in vars(self.metadata).values():
                        tensor.record_stream(stream)
                    self.metadata_recorded = True

    def finish(self):
        if self.finished:
            return
        if self.ready is not None:
            torch.cuda.current_stream().wait_event(self.ready)
        if self.runtime.mode != FULL_REPREFILL:
            self.runtime.mem_manager.free_layer_workspace()
        self.finished = True
