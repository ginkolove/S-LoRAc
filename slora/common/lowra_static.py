import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from slora.common.lowra_reconstruct import lowra_reconstruct_batched


@dataclass
class LowRAStaticPrefixView:
    lora_dir: str
    lora_index: int
    rank_start: int
    rank_end: int


@dataclass
class LowRAPrefixCacheEntry:
    lora_dir: str
    prefix_indices: torch.Tensor
    prefix_len: int
    ref_count: int = 0
    adapter_heat: int = 0


class LowRAStaticPrefixStore:
    """Static BaseKV/MiniKV storage for LowRA.

    The tensors are intentionally zero-initialized in this phase. Shapes use
    TP-local K/V dimensions.
    """

    def __init__(
        self,
        prefix_len: int,
        lora_dirs: List[str],
        lora_ranks: Dict[str, int],
        layer_num: int,
        tp_k_head_num: int,
        tp_v_head_num: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.prefix_len = int(prefix_len)
        self.lora_dirs = list(lora_dirs)
        self.lora_dir_to_index = {lora_dir: i for i, lora_dir in enumerate(self.lora_dirs)}
        self.lora_ranks = {lora_dir: int(lora_ranks[lora_dir]) for lora_dir in self.lora_dirs}
        self.layer_num = int(layer_num)
        self.tp_k_head_num = int(tp_k_head_num)
        self.tp_v_head_num = int(tp_v_head_num)
        self.head_dim = int(head_dim)
        self.kv_embed_dim_k = self.tp_k_head_num * self.head_dim
        self.kv_embed_dim_v = self.tp_v_head_num * self.head_dim
        self.total_rank = sum(self.lora_ranks.values())

        self.rank_ranges: List[Tuple[int, int]] = []
        rank_cursor = 0
        for lora_dir in self.lora_dirs:
            rank = self.lora_ranks[lora_dir]
            self.rank_ranges.append((rank_cursor, rank_cursor + rank))
            rank_cursor += rank

        self.base_k = [
            torch.zeros(
                (self.prefix_len, self.tp_k_head_num, self.head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.layer_num)
        ]
        self.base_v = [
            torch.zeros(
                (self.prefix_len, self.tp_v_head_num, self.head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.layer_num)
        ]
        self.mini_k = [
            torch.zeros((self.prefix_len, self.total_rank), dtype=dtype, device=device)
            for _ in range(self.layer_num)
        ]
        self.mini_v = [
            torch.zeros((self.prefix_len, self.total_rank), dtype=dtype, device=device)
            for _ in range(self.layer_num)
        ]

    @staticmethod
    def token_equivalent_static_size(
        prefix_len: int,
        lora_ranks: Iterable[int],
        kv_embed_dim_k: int,
        kv_embed_dim_v: int,
    ) -> int:
        prefix_len = int(prefix_len)
        rank_sum = sum(int(r) for r in lora_ranks)
        if prefix_len <= 0:
            return 0
        full_base_tokens = prefix_len
        mini_k_tokens = int(math.ceil(prefix_len * rank_sum / kv_embed_dim_k)) if rank_sum > 0 else 0
        mini_v_tokens = int(math.ceil(prefix_len * rank_sum / kv_embed_dim_v)) if rank_sum > 0 else 0
        return full_base_tokens + max(mini_k_tokens, mini_v_tokens)

    @classmethod
    def token_equivalent_static_size_from_config(
        cls,
        prefix_len: int,
        lora_ranks: Iterable[int],
        network_config: Dict,
        world_size: int,
    ) -> int:
        hidden_size = int(network_config["hidden_size"])
        num_attention_heads = int(network_config["num_attention_heads"])
        num_kv_heads = int(network_config.get("num_key_value_heads", num_attention_heads))
        head_dim = hidden_size // num_attention_heads
        tp_k_head_num = num_kv_heads // int(world_size)
        tp_v_head_num = tp_k_head_num
        return cls.token_equivalent_static_size(
            prefix_len,
            lora_ranks,
            tp_k_head_num * head_dim,
            tp_v_head_num * head_dim,
        )

    def get_lora_rank_range(self, lora_index: int) -> Tuple[int, int]:
        return self.rank_ranges[lora_index]

    def get_lora_rank_range_by_dir(self, lora_dir: str) -> Tuple[int, int]:
        return self.get_lora_rank_range(self.lora_dir_to_index[lora_dir])

    def get_static_prefix_view(self, lora_index: int) -> LowRAStaticPrefixView:
        rank_start, rank_end = self.get_lora_rank_range(lora_index)
        return LowRAStaticPrefixView(
            lora_dir=self.lora_dirs[lora_index],
            lora_index=lora_index,
            rank_start=rank_start,
            rank_end=rank_end,
        )

    def get_static_prefix_view_by_dir(self, lora_dir: str) -> LowRAStaticPrefixView:
        return self.get_static_prefix_view(self.lora_dir_to_index[lora_dir])


class LowRAPrefixKVCache:
    """Cache-owned reconstructed prefix KV entries in the dynamic pool."""

    def __init__(self, static_store: LowRAStaticPrefixStore, mem_manager):
        self.static_store = static_store
        self.mem_manager = mem_manager
        self.entries: Dict[str, LowRAPrefixCacheEntry] = {}

    def acquire_many(self, lora_dirs: Iterable[Optional[str]], infer_adapter=None) -> Dict[str, torch.Tensor]:
        counts: Dict[str, int] = {}
        for lora_dir in lora_dirs:
            if lora_dir is None:
                continue
            counts[lora_dir] = counts.get(lora_dir, 0) + 1

        prefix_indices_by_dir = {}
        misses = []
        for lora_dir, count in counts.items():
            if lora_dir in self.entries:
                entry = self.entries[lora_dir]
                entry.ref_count += count
                entry.adapter_heat += 1
            else:
                if infer_adapter is None:
                    raise RuntimeError(
                        f"LowRA prefix cache miss for {lora_dir}, but no InferAdapter was provided"
                    )
                prefix_indices = self._alloc_prefix_indices()
                entry = LowRAPrefixCacheEntry(
                    lora_dir=lora_dir,
                    prefix_indices=prefix_indices,
                    prefix_len=self.static_store.prefix_len,
                    ref_count=count,
                    adapter_heat=1,
                )
                self.entries[lora_dir] = entry
                misses.append(lora_dir)
            prefix_indices_by_dir[lora_dir] = entry.prefix_indices

        if misses:
            self._reconstruct_many_into_dynamic_pool(misses, infer_adapter)
        return prefix_indices_by_dir

    def acquire(self, lora_dir: str, infer_adapter=None, ref_delta: int = 1) -> LowRAPrefixCacheEntry:
        if lora_dir in self.entries:
            entry = self.entries[lora_dir]
            entry.ref_count += ref_delta
            entry.adapter_heat += 1
            return entry

        if infer_adapter is None:
            raise RuntimeError(f"LowRA prefix cache miss for {lora_dir}, but no InferAdapter was provided")

        prefix_indices = self._alloc_prefix_indices()
        entry = LowRAPrefixCacheEntry(
            lora_dir=lora_dir,
            prefix_indices=prefix_indices,
            prefix_len=self.static_store.prefix_len,
            ref_count=ref_delta,
            adapter_heat=1,
        )
        self.entries[lora_dir] = entry
        self._reconstruct_many_into_dynamic_pool([lora_dir], infer_adapter)
        return entry

    def release_many(self, lora_dirs: Iterable[Optional[str]]):
        for lora_dir in lora_dirs:
            if lora_dir is not None:
                self.release(lora_dir)

    def release(self, lora_dir: str):
        entry = self.entries.get(lora_dir)
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)

    def evict_one(self) -> bool:
        evictable = [entry for entry in self.entries.values() if entry.ref_count == 0]
        if not evictable:
            return False
        victim = min(evictable, key=lambda entry: entry.adapter_heat)
        self.evict(victim.lora_dir)
        return True

    def evict(self, lora_dir: str) -> bool:
        entry = self.entries.get(lora_dir)
        if entry is None or entry.ref_count > 0:
            return False
        self.mem_manager.free(entry.prefix_indices)
        del self.entries[lora_dir]
        return True

    def evict_many(self, lora_dirs: Iterable[str]):
        for lora_dir in lora_dirs:
            self.evict(lora_dir)

    def clear_metadata_if_inactive(self) -> int:
        active = [lora_dir for lora_dir, entry in self.entries.items() if entry.ref_count > 0]
        if active:
            raise RuntimeError(f"LowRA prefix cache still has active entries: {active}")
        cleared = len(self.entries)
        self.entries.clear()
        return cleared

    def state_dict(self) -> Dict:
        return {
            "prefix_len": self.static_store.prefix_len,
            "cached_lora_dirs": set(self.entries.keys()),
            "active_lora_dirs": {
                lora_dir for lora_dir, entry in self.entries.items()
                if entry.ref_count > 0
            },
            "inactive_lora_dirs": {
                lora_dir for lora_dir, entry in self.entries.items()
                if entry.ref_count == 0
            },
            "heat_by_lora": {
                lora_dir: entry.adapter_heat
                for lora_dir, entry in self.entries.items()
            },
            "free_dynamic_slots": self.mem_manager.can_use_mem_size,
        }

    def _alloc_prefix_indices(self) -> torch.Tensor:
        prefix_len = self.static_store.prefix_len
        while prefix_len > self.mem_manager.can_use_mem_size:
            if not self.evict_one():
                break
        if prefix_len > self.mem_manager.can_use_mem_size:
            raise RuntimeError(
                f"LowRA cannot allocate reconstructed prefix KV: need {prefix_len}, "
                f"left {self.mem_manager.can_use_mem_size}"
            )
        return self.mem_manager.alloc(prefix_len)

    @torch.no_grad()
    def _reconstruct_many_into_dynamic_pool(self, lora_dirs: List[str], infer_adapter):
        static = self.static_store
        max_rank = max(static.get_lora_rank_range_by_dir(lora_dir)[1] -
                       static.get_lora_rank_range_by_dir(lora_dir)[0]
                       for lora_dir in lora_dirs)
        prefix_locs = torch.empty(
            (len(lora_dirs), static.prefix_len),
            dtype=torch.long,
            device="cuda",
        )
        k_b_locs = torch.empty((len(lora_dirs), max_rank), dtype=torch.long, device="cuda")
        v_b_locs = torch.empty((len(lora_dirs), max_rank), dtype=torch.long, device="cuda")
        rank_starts = torch.empty((len(lora_dirs),), dtype=torch.long, device="cuda")
        rank_lens = torch.empty((len(lora_dirs),), dtype=torch.long, device="cuda")
        scalings = torch.empty((len(lora_dirs),), dtype=torch.float16, device="cuda")

        for batch_idx, lora_dir in enumerate(lora_dirs):
            view = static.get_static_prefix_view_by_dir(lora_dir)
            rank = view.rank_end - view.rank_start
            adapter_idx = infer_adapter.idx_map[lora_dir]
            adapter_start = int(infer_adapter.a_start[adapter_idx].item())
            adapter_len = int(infer_adapter.a_len[adapter_idx].item())
            adapter_locs = infer_adapter.a_loc[adapter_start: adapter_start + adapter_len]

            prefix_locs[batch_idx] = self.entries[lora_dir].prefix_indices
            k_b_locs[batch_idx, :rank] = adapter_locs[rank: 2 * rank]
            v_b_locs[batch_idx, :rank] = adapter_locs[2 * rank: 3 * rank]
            if rank < max_rank:
                k_b_locs[batch_idx, rank:].fill_(0)
                v_b_locs[batch_idx, rank:].fill_(0)
            rank_starts[batch_idx] = view.rank_start
            rank_lens[batch_idx] = rank
            scalings[batch_idx] = infer_adapter.a_scaling[adapter_idx]

        for layer_id in range(static.layer_num):
            lowra_reconstruct_batched(
                static.base_k[layer_id].view(static.prefix_len, static.kv_embed_dim_k),
                static.mini_k[layer_id],
                infer_adapter.mem_manager.value_buffer[layer_id].view(-1, static.kv_embed_dim_k),
                self.mem_manager.key_buffer[layer_id].view(-1, static.kv_embed_dim_k),
                prefix_locs,
                k_b_locs,
                rank_starts,
                rank_lens,
                scalings,
            )
            lowra_reconstruct_batched(
                static.base_v[layer_id].view(static.prefix_len, static.kv_embed_dim_v),
                static.mini_v[layer_id],
                infer_adapter.mem_manager.value_buffer[layer_id].view(-1, static.kv_embed_dim_v),
                self.mem_manager.value_buffer[layer_id].view(-1, static.kv_embed_dim_v),
                prefix_locs,
                v_b_locs,
                rank_starts,
                rank_lens,
                scalings,
            )
