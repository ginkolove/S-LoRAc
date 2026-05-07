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
    is_gpu_hot: bool
    is_cpu_backed: bool


class PrefixSLoraPrefixCache:
    """Prefix-S-LoRA prefix KV placement metadata.

    Hot prefixes are physically reserved in the unified mem_manager pool at
    startup. Cold prefixes are stored as zero-initialized CPU tensors; later
    swap-in can allocate dynamic GPU slots and copy these tensors.
    """

    def __init__(
        self,
        mem_manager,
        adapter_dirs: List[str],
        shared_prefix_len: int,
        gpu_prefix_num: int,
        cpu_prefix_num: int,
    ) -> None:
        assert shared_prefix_len > 0
        assert gpu_prefix_num >= 0
        assert cpu_prefix_num >= 0
        assert gpu_prefix_num + cpu_prefix_num <= len(adapter_dirs)

        self.mem_manager = mem_manager
        self.adapter_dirs = list(adapter_dirs)
        self.shared_prefix_len = shared_prefix_len
        self.gpu_prefix_num = gpu_prefix_num
        self.cpu_prefix_num = cpu_prefix_num
        self.gpu_lora_dirs = self.adapter_dirs[:gpu_prefix_num]
        self.cpu_lora_dirs = self.adapter_dirs[gpu_prefix_num:gpu_prefix_num + cpu_prefix_num]
        self.entries: Dict[str, PrefixSLoraEntry] = {}
        self.cpu_lora_to_index: Dict[str, int] = {}
        self.cpu_key_buffer = []
        self.cpu_value_buffer = []

        self._reserve_hot_gpu_prefixes()
        self._register_cpu_prefixes()

    def _reserve_hot_gpu_prefixes(self) -> None:
        if self.gpu_prefix_num == 0:
            return

        total_prefix_tokens = self.gpu_prefix_num * self.shared_prefix_len
        all_indices = self.mem_manager.alloc(total_prefix_tokens)

        for i, lora_dir in enumerate(self.gpu_lora_dirs):
            start = i * self.shared_prefix_len
            end = start + self.shared_prefix_len
            prefix_indices = all_indices[start:end].contiguous()
            self._zero_gpu_prefix(prefix_indices)
            self.entries[lora_dir] = PrefixSLoraEntry(
                lora_dir=lora_dir,
                prefix_len=self.shared_prefix_len,
                prefix_indices=prefix_indices,
                ref_count=0,
                heat=0,
                is_gpu_hot=True,
                is_cpu_backed=False,
            )

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

        for lora_dir in self.cpu_lora_dirs:
            self.cpu_lora_to_index[lora_dir] = len(self.cpu_lora_to_index)
            self.entries[lora_dir] = PrefixSLoraEntry(
                lora_dir=lora_dir,
                prefix_len=self.shared_prefix_len,
                prefix_indices=None,
                ref_count=0,
                heat=0,
                is_gpu_hot=False,
                is_cpu_backed=True,
            )

    def _zero_gpu_prefix(self, prefix_indices: torch.Tensor) -> None:
        for layer_id in range(self.mem_manager.layer_num):
            self.mem_manager.key_buffer[layer_id][prefix_indices].zero_()
            self.mem_manager.value_buffer[layer_id][prefix_indices].zero_()

    def get_entry(self, lora_dir: str) -> Optional[PrefixSLoraEntry]:
        return self.entries.get(lora_dir)

    def copy_cpu_prefix_to_gpu(self, lora_dir: str, prefix_indices: torch.Tensor) -> None:
        """Copy a cold CPU prefix into already allocated GPU prefix slots."""
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
        self.evict_inactive_except(needed)

        out: Dict[str, torch.Tensor] = {}
        for lora_dir, count in counts.items():
            entry = self.entries.get(lora_dir)
            if entry is None:
                continue
            entry.heat += count
            entry.ref_count += count
            if entry.prefix_indices is None:
                assert entry.is_cpu_backed
                entry.prefix_indices = self.mem_manager.alloc(self.shared_prefix_len)
                self.copy_cpu_prefix_to_gpu(lora_dir, entry.prefix_indices)
            out[lora_dir] = entry.prefix_indices
        return out

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
        for lora_dir, entry in self.entries.items():
            if entry.is_gpu_hot:
                continue
            if lora_dir in keep_lora_dirs:
                continue
            if entry.ref_count != 0 or entry.prefix_indices is None:
                continue
            self.mem_manager.free(entry.prefix_indices)
            entry.prefix_indices = None
            evicted.append(lora_dir)
        return evicted

    def state_dict(self):
        return {
            "enabled": True,
            "shared_prefix_len": self.shared_prefix_len,
            "gpu_prefix_num": self.gpu_prefix_num,
            "cpu_prefix_num": self.cpu_prefix_num,
            "gpu_lora_dirs": list(self.gpu_lora_dirs),
            "cpu_lora_dirs": list(self.cpu_lora_dirs),
            "reserved_gpu_prefix_tokens": self.gpu_prefix_num * self.shared_prefix_len,
            "cpu_prefix_tokens": self.cpu_prefix_num * self.shared_prefix_len,
            "free_tokens_after_prefix_reserve": int(self.mem_manager.can_use_mem_size),
        }
