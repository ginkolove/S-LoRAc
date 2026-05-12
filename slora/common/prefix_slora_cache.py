from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Optional

import torch


@dataclass
class PrefixSLoraEntry:
    lora_dir: str
    prefix_len: int
    prefix_indices: Optional[torch.Tensor]
    ref_count: int
    heat: int
    last_access: int
    cpu_index: int


class PrefixSLoraPrefixCache:
    """Prefix-S-LoRA prefix KV placement metadata.

    All adapter prefix KVs are stored on CPU. GPU prefix KVs are normal entries
    in the unified mem_manager pool: they are allocated on demand, kept after
    requests finish, and evicted by LRU only under space pressure.
    """

    def __init__(
        self,
        mem_manager,
        adapter_dirs: List[str],
        shared_prefix_len: int,
        gpu_prefix_num: int = 0,
        cpu_prefix_num: Optional[int] = None,
    ) -> None:
        assert shared_prefix_len > 0
        assert gpu_prefix_num >= 0
        if cpu_prefix_num is not None:
            assert cpu_prefix_num >= 0

        self.mem_manager = mem_manager
        self.adapter_dirs = list(adapter_dirs)
        self.shared_prefix_len = shared_prefix_len
        self.cpu_prefix_num = len(self.adapter_dirs)
        self.entries: Dict[str, PrefixSLoraEntry] = {}
        self.cpu_lora_to_index: Dict[str, int] = {}
        self.cpu_key_buffer = []
        self.cpu_value_buffer = []
        self.clock = 0

        self._register_cpu_prefixes()

    def _register_cpu_prefixes(self) -> None:
        if self.cpu_prefix_num > 0:
            for _ in range(self.mem_manager.layer_num):
                self.cpu_key_buffer.append(
                    torch.zeros(
                        (
                            self.cpu_prefix_num,
                            self.shared_prefix_len,
                            self.mem_manager.head_num,
                            self.mem_manager.head_dim,
                        ),
                        dtype=self.mem_manager.dtype,
                        device="cpu",
                    )
                )
                self.cpu_value_buffer.append(
                    torch.zeros(
                        (
                            self.cpu_prefix_num,
                            self.shared_prefix_len,
                            self.mem_manager.head_num,
                            self.mem_manager.head_dim,
                        ),
                        dtype=self.mem_manager.dtype,
                        device="cpu",
                    )
                )

        for lora_dir in self.adapter_dirs:
            self.cpu_lora_to_index[lora_dir] = len(self.cpu_lora_to_index)
            self.entries[lora_dir] = PrefixSLoraEntry(
                lora_dir=lora_dir,
                prefix_len=self.shared_prefix_len,
                prefix_indices=None,
                ref_count=0,
                heat=0,
                last_access=0,
                cpu_index=self.cpu_lora_to_index[lora_dir],
            )

    def _zero_gpu_prefix(self, prefix_indices: torch.Tensor) -> None:
        for layer_id in range(self.mem_manager.layer_num):
            self.mem_manager.key_buffer[layer_id][prefix_indices].zero_()
            self.mem_manager.value_buffer[layer_id][prefix_indices].zero_()

    def get_entry(self, lora_dir: str) -> Optional[PrefixSLoraEntry]:
        return self.entries.get(lora_dir)

    def copy_cpu_prefix_to_gpu(self, lora_dir: str, prefix_indices: torch.Tensor) -> None:
        """Copy a CPU-resident prefix into already allocated GPU prefix slots."""
        cpu_index = self.cpu_lora_to_index[lora_dir]
        assert prefix_indices.numel() == self.shared_prefix_len
        for layer_id in range(self.mem_manager.layer_num):
            self.mem_manager.key_buffer[layer_id][prefix_indices].copy_(
                self.cpu_key_buffer[layer_id][cpu_index], non_blocking=True
            )
            self.mem_manager.value_buffer[layer_id][prefix_indices].copy_(
                self.cpu_value_buffer[layer_id][cpu_index], non_blocking=True
            )

    def acquire_many(self, lora_dirs: List[str]) -> Dict[str, torch.Tensor]:
        counts = Counter(lora_dirs)
        needed = set(counts.keys())
        registered = {
            lora_dir: self.entries[lora_dir]
            for lora_dir in counts
            if lora_dir in self.entries
        }

        out: Dict[str, torch.Tensor] = {}
        for lora_dir, entry in registered.items():
            if entry.prefix_indices is None:
                entry.prefix_indices = self._alloc_prefix_slots_with_lru(needed)
                self.copy_cpu_prefix_to_gpu(lora_dir, entry.prefix_indices)

        for lora_dir, entry in registered.items():
            count = counts[lora_dir]
            self.clock += 1
            entry.heat += count
            entry.ref_count += count
            entry.last_access = self.clock
            out[lora_dir] = entry.prefix_indices
        return out

    def prepare_many(self, lora_dirs: List[str], extra_token_num: int = 0) -> List[str]:
        needed = set(lora_dirs)
        missing_prefix_num = 0
        for lora_dir in needed:
            entry = self.entries.get(lora_dir)
            if entry is not None and entry.prefix_indices is None:
                missing_prefix_num += 1
        token_num = missing_prefix_num * self.shared_prefix_len + max(0, int(extra_token_num))
        return self.evict_until_can_allocate(token_num, keep_lora_dirs=needed)

    def _alloc_prefix_slots_with_lru(self, protected_lora_dirs) -> torch.Tensor:
        protected_lora_dirs = set(protected_lora_dirs)
        while self.mem_manager.can_use_mem_size < self.shared_prefix_len:
            victim = self._select_lru_victim(protected_lora_dirs)
            if victim is None:
                raise RuntimeError(
                    "Prefix-S-LoRA cannot allocate prefix KV: no inactive "
                    "resident prefix can be evicted from the unified pool"
                )
            self._evict_entry(victim)
        return self.mem_manager.alloc(self.shared_prefix_len)

    def _select_lru_victim(self, protected_lora_dirs) -> Optional[PrefixSLoraEntry]:
        candidates = [
            entry for entry in self.entries.values()
            if entry.prefix_indices is not None
            and entry.ref_count == 0
            and entry.lora_dir not in protected_lora_dirs
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda entry: entry.last_access)

    def _evict_entry(self, entry: PrefixSLoraEntry) -> None:
        assert entry.prefix_indices is not None
        self.mem_manager.free(entry.prefix_indices)
        entry.prefix_indices = None

    def release_many(self, lora_dirs: List[str]) -> None:
        counts = Counter(lora_dirs)
        for lora_dir, count in counts.items():
            entry = self.entries.get(lora_dir)
            if entry is None:
                continue
            entry.ref_count = max(0, entry.ref_count - count)

    def evict_inactive_except(self, keep_lora_dirs) -> List[str]:
        keep_lora_dirs = set(keep_lora_dirs)
        evicted = []
        while True:
            entry = self._select_lru_victim(keep_lora_dirs)
            if entry is None:
                break
            lora_dir = entry.lora_dir
            self._evict_entry(entry)
            evicted.append(lora_dir)
        return evicted

    def evict_until_can_allocate(self, token_num: int, keep_lora_dirs=None) -> List[str]:
        keep_lora_dirs = set(keep_lora_dirs or [])
        evicted = []
        while self.mem_manager.can_use_mem_size < token_num:
            entry = self._select_lru_victim(keep_lora_dirs)
            if entry is None:
                break
            evicted.append(entry.lora_dir)
            self._evict_entry(entry)
        return evicted

    def resident_lora_dirs(self) -> List[str]:
        return [
            lora_dir for lora_dir, entry in self.entries.items()
            if entry.prefix_indices is not None
        ]

    def active_lora_dirs(self) -> List[str]:
        return [
            lora_dir for lora_dir, entry in self.entries.items()
            if entry.prefix_indices is not None and entry.ref_count > 0
        ]

    def inactive_lora_dirs(self) -> List[str]:
        return [
            lora_dir for lora_dir, entry in self.entries.items()
            if entry.prefix_indices is not None and entry.ref_count == 0
        ]

    def state_dict(self):
        resident = self.resident_lora_dirs()
        active = self.active_lora_dirs()
        return {
            "enabled": True,
            "shared_prefix_len": self.shared_prefix_len,
            "cpu_prefix_num": self.cpu_prefix_num,
            "cpu_lora_dirs": list(self.adapter_dirs),
            "resident_gpu_prefix_num": len(resident),
            "resident_gpu_prefix_tokens": len(resident) * self.shared_prefix_len,
            "active_gpu_prefix_num": len(active),
            "active_gpu_prefix_tokens": len(active) * self.shared_prefix_len,
            "cpu_prefix_tokens": self.cpu_prefix_num * self.shared_prefix_len,
            "free_unified_tokens": int(self.mem_manager.can_use_mem_size),
        }
