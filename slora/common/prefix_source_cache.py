"""Next-batch GPU prefix reuse with full or sampled prefix transfers from CPU."""

import hashlib
from pathlib import Path

import torch

from slora.common.full_reprefill import FullPrefixKVCache
from slora.common.prefix_modes import ACTIVE_REPREFILL, CPU_PIN_COUNTS, CPU_MODES, CPU_READ, IMPRESS


IMPRESS_SEED = 42


class HostPrefixBank:
    def __init__(self, pool, prefix_ids, adapter_dirs, byte_cap=None, *, mode=CPU_READ, impress_read_percent=None):
        self.pool = pool
        self.prefix_ids = tuple(prefix_ids)
        self.entries = {}
        self.entry_bytes = len(self.prefix_ids) * sum(
            t[0].numel() * t.element_size() for t in (*pool.key_buffer, *pool.value_buffer))
        dirs = tuple(adapter_dirs)
        if not dirs or len(set(dirs)) != len(dirs) or not self.entry_bytes:
            raise ValueError("CPU prefix bank needs distinct adapters and nonempty prefixes")
        count = len(dirs) if byte_cap is None else min(len(dirs), byte_cap // self.entry_bytes)
        if count < 1:
            raise ValueError("host prefix pool cap %d bytes cannot fit one complete prefix (%d bytes)"
                             % (byte_cap, self.entry_bytes))
        self.byte_cap = byte_cap
        self.physical_dirs = dirs[:count]
        # Only the host address aliases. Admission, device slots and copies are
        # still keyed by the requested adapter, not by its representative.
        self.source_for = {d: self.physical_dirs[i % count] for i, d in enumerate(dirs)}

        length = len(self.prefix_ids)
        self.transfer_tokens = length
        self.impress_order = None
        if mode == IMPRESS:
            percent = 50 if impress_read_percent is None else impress_read_percent
            if type(percent) is not int or not 0 <= percent <= 100:
                raise ValueError("Impress read percentage must be an integer between 0 and 100")
            self.impress_read_percent = percent
            # One CPU permutation for every adapter, layer and K/V tensor, independent of model RNG.
            generator = torch.Generator(device="cpu").manual_seed(IMPRESS_SEED)
            permutation = torch.randperm(length, device="cpu", generator=generator)
            self.transfer_tokens = length * percent // 100
            kept = permutation[:self.transfer_tokens].sort().values
            dropped = permutation[self.transfer_tokens:].sort().values
            self.impress_positions_sha256 = hashlib.sha256(kept.numpy().tobytes()).hexdigest()
            self.impress_order = torch.cat((kept, dropped)).to(pool.key_buffer[0].device)
        self.transfer_bytes = self.transfer_tokens * (self.entry_bytes // length)

    def check_capacity(self):
        # Pageable host tensors, not hundreds of GB of page-locked allocations.
        available = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
        groups = Path("/proc/self/cgroup").read_text().splitlines()
        for line in groups:
            if line.startswith("0::"):
                root = Path("/sys/fs/cgroup")
                group = root / line.split("::", 1)[1].lstrip("/")
                for parent in (group, *group.parents):
                    if parent != root and root not in parent.parents:
                        break
                    limit = parent / "memory.max"
                    if limit.exists() and limit.read_text().strip() != "max":
                        available = min(available, int(limit.read_text()) -
                                        int((parent / "memory.current").read_text()))
        needed = self.entry_bytes * len(self.physical_dirs)
        if needed > available * 0.9:
            raise RuntimeError("CPU prefix bank needs %d bytes; available %d (10%% headroom required). "
                               "Reduce adapter count or prefix length." % (needed, available))

    def capture(self, directory, slots):
        if self.source_for[directory] != directory or slots.numel() != len(self.prefix_ids):
            raise ValueError("capture requires a physical source and one full prefix")
        slots = slots.long()
        if self.impress_order is not None:
            # Pack the selected rows first during capture, so each miss reads a contiguous CPU slice.
            slots = slots.index_select(0, self.impress_order)
        self.entries[directory] = [
            tuple(t.index_select(0, slots).cpu() for t in pair)
            for pair in zip(self.pool.key_buffer, self.pool.value_buffer)]

    def _buffer_pairs(self, directory):
        entry = self.entries[self.source_for[directory]]
        for (k, v), dk, dv in zip(entry, self.pool.key_buffer, self.pool.value_buffer):
            yield k, dk
            yield v, dv

    def load(self, misses, prefix_ids, locations):
        if tuple(prefix_ids) != self.prefix_ids:
            raise ValueError("prefix token IDs differ from the prefilled CPU bank")
        for directory in misses:
            slots = locations[directory].long()
            if slots.numel() != len(self.prefix_ids):
                raise ValueError("load requires device slots for one full prefix")
            if self.impress_order is not None:
                ordered_slots = slots.index_select(0, self.impress_order)
                kept = ordered_slots[:self.transfer_tokens]
                padding = ordered_slots[self.transfer_tokens:]
                for src, dst in self._buffer_pairs(directory):
                    dst.index_fill_(0, padding, 0)
                    payload = src[:self.transfer_tokens].to(dst.device, non_blocking=True)
                    dst.index_copy_(0, kept, payload)
                continue

            first = int(slots[0])
            contiguous = torch.equal(slots, torch.arange(first, first + len(slots), device=slots.device))
            for src, dst in self._buffer_pairs(directory):
                if contiguous:
                    dst.narrow(0, first, len(slots)).copy_(src, non_blocking=True)
                else:
                    # dst[slots].copy_(...) would only update an advanced-indexing temporary.
                    dst.index_copy_(0, slots, src.to(dst.device, non_blocking=True))


class ActivePrefixCache(FullPrefixKVCache):
    def __init__(self, prefix_len, pool, mode, fixed=(), bank=None):
        if (mode in CPU_MODES) != (bank is not None) or mode not in (ACTIVE_REPREFILL, *CPU_MODES):
            raise ValueError("invalid prefix source")
        if bank is not None and (mode == IMPRESS) != (bank.impress_order is not None):
            raise ValueError("CPU prefix bank layout does not match " + mode)
        if len(fixed) != CPU_PIN_COUNTS.get(mode, 0) or len(set(fixed)) != len(fixed):
            raise ValueError("wrong number of distinct pinned adapters for " + mode)
        self.source_mode = mode
        self.bank = bank
        super().__init__(prefix_len, pool, pinned_lora_dirs=fixed, reuse_next_batch=True)

    def acquire_many(self, requests, build_prefixes):
        return super().acquire_many(requests, self.bank.load if self.bank else build_prefixes)

    def _record_prefix_builds(self, misses):
        if self.bank is None:
            super()._record_prefix_builds(misses)

    def access_stats(self):
        stats = dict(super().access_stats(), source_mode=self.source_mode,
                    hot_lora_dirs=sorted(self.pinned_lora_dirs),
                    host_prefix_entries=len(self.bank.entries) if self.bank else 0,
                    host_prefix_bytes=len(self.bank.entries) * self.bank.entry_bytes if self.bank else 0,
                    host_prefix_logical_entries=len(self.bank.source_for) if self.bank else 0,
                    host_prefix_entry_bytes=self.bank.entry_bytes if self.bank else 0,
                    host_prefix_pool_cap_bytes=self.bank.byte_cap if self.bank else None,
                    host_prefix_aliased=bool(self.bank and len(self.bank.physical_dirs) < len(self.bank.source_for)),
                    prefix_h2d_loads=self.batch_unique_misses if self.bank else 0,
                    prefix_h2d_bytes=self.batch_unique_misses * self.bank.transfer_bytes if self.bank else 0)
        if self.source_mode == IMPRESS:
            stats.update(impress_seed=IMPRESS_SEED, impress_read_percent=self.bank.impress_read_percent,
                         impress_positions_sha256=self.bank.impress_positions_sha256,
                         prefix_h2d_tokens_per_load=self.bank.transfer_tokens)
        return stats
