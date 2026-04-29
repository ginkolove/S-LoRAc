import os
from dataclasses import dataclass


@dataclass
class PrefixKVCacheEntry:
    lora_dir: str
    prefix_indices: object
    prefix_len: int
    ref_count: int = 0
    adapter_heat: int = 0


class PrefixKVCacheManager:
    def __init__(self, mem_manager):
        self.mem_manager = mem_manager
        self.entries = {}
        self.debug_enabled = os.environ.get("SLORA_PREFIX_DEBUG", "0") == "1"

    def _debug(self, message):
        if self.debug_enabled:
            print(f"[prefix-cache] {message}", flush=True)

    def clear(self):
        self.entries = {}

    def is_enabled(self):
        return bool(getattr(self.mem_manager, "has_static_shared_prefix", False))

    def _normalize_lora_dirs(self, adapter_dirs):
        if not self.is_enabled():
            return []
        normalized = []
        seen = set()
        valid_dirs = getattr(self.mem_manager, "lora_dir_to_index", {})
        for lora_dir in adapter_dirs or []:
            if lora_dir is None or lora_dir not in valid_dirs or lora_dir in seen:
                continue
            normalized.append(lora_dir)
            seen.add(lora_dir)
        return normalized

    def get(self, lora_dir):
        return self.entries.get(lora_dir)

    def contains(self, lora_dir):
        return lora_dir in self.entries

    def insert(self, lora_dir, prefix_indices, prefix_len=None):
        if not self.is_enabled() or lora_dir is None:
            return None
        entry = PrefixKVCacheEntry(
            lora_dir=lora_dir,
            prefix_indices=prefix_indices,
            prefix_len=int(prefix_len or self.mem_manager.shared_prefix_length),
        )
        self.entries[lora_dir] = entry
        self._debug(
            f"insert lora={lora_dir} prefix_len={entry.prefix_len} "
            f"indices={entry.prefix_indices.shape[0]}"
        )
        return entry

    def acquire(self, lora_dir, count=1):
        entry = self.get(lora_dir)
        if entry is None or count <= 0:
            return
        entry.ref_count += int(count)
        self._debug(f"acquire lora={lora_dir} ref_count={entry.ref_count}")

    def release(self, lora_dir, count=1):
        entry = self.get(lora_dir)
        if entry is None or count <= 0:
            return
        entry.ref_count = max(entry.ref_count - int(count), 0)
        self._debug(f"release lora={lora_dir} ref_count={entry.ref_count}")

    def count_missing_prefixes(self, batch_adapter_dirs):
        missing = 0
        for lora_dir in self._normalize_lora_dirs(batch_adapter_dirs):
            if lora_dir not in self.entries:
                missing += 1
        return missing

    def estimate_required_slots(self, batch_adapter_dirs, prompt_token_num, extra_slots=0):
        prompt_token_num = int(prompt_token_num)
        extra_slots = int(extra_slots)
        if not self.is_enabled():
            return prompt_token_num + extra_slots
        missing_prefixes = self.count_missing_prefixes(batch_adapter_dirs)
        return prompt_token_num + missing_prefixes * self.mem_manager.shared_prefix_length + extra_slots

    def _evict_entry(self, entry):
        self.mem_manager.free(entry.prefix_indices)
        self.entries.pop(entry.lora_dir, None)

    def ensure_available(self, required_slots, pinned_loras=None, future_reuse=None, adapter_heat=None):
        required_slots = int(required_slots)
        future_reuse = future_reuse or {}
        adapter_heat = adapter_heat or {}
        if not self.is_enabled():
            return {
                "admitted": True,
                "required_slots": required_slots,
                "free_slots": self.mem_manager.can_use_mem_size,
                "evicted": [],
            }

        pinned_loras = set(self._normalize_lora_dirs(pinned_loras))
        evicted = []

        while self.mem_manager.can_use_mem_size < required_slots:
            candidates = [
                entry for entry in self.entries.values()
                if entry.ref_count == 0 and entry.lora_dir not in pinned_loras
            ]
            if len(candidates) == 0:
                break
            candidates.sort(
                key=lambda entry: (
                    int(future_reuse.get(entry.lora_dir, 0)),
                    int(adapter_heat.get(entry.lora_dir, entry.adapter_heat)),
                    entry.lora_dir,
                )
            )
            victim = candidates[0]
            evicted.append(victim.lora_dir)
            self._debug(
                "evict "
                f"lora={victim.lora_dir} future_reuse={int(future_reuse.get(victim.lora_dir, 0))} "
                f"heat={int(adapter_heat.get(victim.lora_dir, victim.adapter_heat))} "
                f"free_before={self.mem_manager.can_use_mem_size} required={required_slots}"
            )
            self._evict_entry(victim)

        for lora_dir, heat in adapter_heat.items():
            entry = self.entries.get(lora_dir)
            if entry is not None:
                entry.adapter_heat = int(heat)

        result = {
            "admitted": self.mem_manager.can_use_mem_size >= required_slots,
            "required_slots": required_slots,
            "free_slots": self.mem_manager.can_use_mem_size,
            "evicted": evicted,
        }
        return result

    def prepare_for_admission(self, batch_adapter_dirs, prompt_token_num, future_reuse=None, adapter_heat=None, extra_slots=0):
        prompt_token_num = int(prompt_token_num)
        extra_slots = int(extra_slots)
        if not self.is_enabled():
            return {
                "admitted": True,
                "required_slots": prompt_token_num + extra_slots,
                "free_slots": self.mem_manager.can_use_mem_size,
                "evicted": [],
                "missing_prefixes": 0,
                "extra_slots": extra_slots,
            }

        normalized_batch_loras = self._normalize_lora_dirs(batch_adapter_dirs)
        missing_prefixes = self.count_missing_prefixes(batch_adapter_dirs)
        required_slots = prompt_token_num + missing_prefixes * self.mem_manager.shared_prefix_length + extra_slots
        result = self.ensure_available(
            required_slots=required_slots,
            pinned_loras=normalized_batch_loras,
            future_reuse=future_reuse,
            adapter_heat=adapter_heat,
        )
        result["missing_prefixes"] = missing_prefixes
        result["extra_slots"] = extra_slots
        self._debug(
            "prepare "
            f"batch_loras={normalized_batch_loras} "
            f"prompt={prompt_token_num} missing_prefixes={missing_prefixes} "
            f"extra_slots={extra_slots} required={required_slots} free_after={result['free_slots']} "
            f"admitted={result['admitted']} evicted={result['evicted']}"
        )
        return result
