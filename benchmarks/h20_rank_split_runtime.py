"""H20 runtime combining the full MiniKV cache and the serial MiniKV workspace."""
import torch

from slora.common.full_reconstruct import FullReconstructPrefixKVCache, FullReconstructRuntime
from slora.common.lowra_layer_runtime import LowRALayerRuntime
from slora.common.prefix_indices import PrefixIndexedFrame
from slora.common.prefix_modes import LAYER_SERIAL


class RankSplitCache(FullReconstructPrefixKVCache):
    """Only high-rank adapters acquire full-prefix slots and references."""

    def __init__(self, static, pool, serial_max_rank):
        self.serial_max_rank = serial_max_rank
        self.ranks = static.lora_ranks
        super().__init__(static.prefix_len, pool, reuse_next_batch=True)

    def full(self, directory):
        return self.ranks[directory] > self.serial_max_rank

    def acquire_many(self, requests, build_prefixes):
        return super().acquire_many([r for r in requests if self.full(r["adapter_dir"])], build_prefixes)

    def release_many(self, lora_dirs):
        super().release_many(d for d in lora_dirs if self.full(d))

    def access_stats(self):
        return dict(super().access_stats(), serial_max_rank=self.serial_max_rank,
                    full_ranks=sorted({r for r in self.ranks.values() if r > self.serial_max_rank}),
                    serial_ranks=sorted({r for r in self.ranks.values() if r <= self.serial_max_rank}))


class RankSplitRuntime(FullReconstructRuntime):
    def __init__(self, static, pool, cache, mode, workers, workspace_lora_limit=None):
        super().__init__(static, pool, cache)
        self.mode = mode
        self.serial = LowRALayerRuntime(static, pool, LAYER_SERIAL, workspace_lora_limit=workspace_lora_limit,
                                        reconstruction_kernel="bounded", reconstruction_workers=workers)

    def prepare(self, adapter_dirs, infer_adapter, prefix_by_dir=None, reserve_tokens=0):
        low = [d for d in adapter_dirs if not self.cache.full(d)]
        if not low:
            return super().prepare(adapter_dirs, infer_adapter, prefix_by_dir, reserve_tokens)
        serial = self.serial.prepare(low, infer_adapter, reserve_tokens=reserve_tokens)
        self.last_reconstruction_kernel = self.serial.last_reconstruction_kernel
        self.last_reconstruction_workers = self.serial.last_reconstruction_workers
        if len(low) == len(adapter_dirs):
            return serial
        try:
            device = serial.prefix_locs.device
            unique, request_ids = self.index_cache.request_mapping(adapter_dirs, device)
            low_locs = dict(zip(dict.fromkeys(low), serial.prefix_locs.unbind(0)))
            full = [d for d in unique if d not in low_locs]
            self.cache.touch(full)
            # Full slots index a single layer's buffer. Scratch slots already
            # index the flattened all-layer buffer and stay fixed between layers.
            base = torch.stack([low_locs[d] if d in low_locs else self.cache.entries[d].prefix_indices
                                for d in unique])
            strides = base.new_tensor([0 if d in low_locs else self.mem_manager.tot_size
                                       for d in unique]).view(-1, 1)
            return RankSplitFrame(self, serial, base, strides, request_ids)
        except Exception:
            serial.finish()
            raise


class RankSplitFrame(PrefixIndexedFrame):
    def __init__(self, runtime, serial, base, strides, request_ids):
        self.runtime = runtime
        self.serial = serial
        self.base = base
        self.strides = strides
        self.prefix_request_ids = request_ids
        self.layer_table = torch.empty_like(base)

    def prefix_index_table(self, layer_id):
        if layer_id == 0:
            return self.base
        return torch.add(self.base, self.strides, alpha=layer_id, out=self.layer_table)

    def buffers(self, layer_id):
        return self.serial.buffers(layer_id)

    def before_layer(self, layer_id):
        self.serial.before_layer(layer_id)

    def before_attention(self, layer_id):
        self.serial.before_attention(layer_id)

    def after_attention(self, layer_id):
        self.serial.after_attention(layer_id)

    def finish(self):
        self.serial.finish()
