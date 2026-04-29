import gc
import math
import torch

from slora.common.prefix_kv_cache import PrefixKVCacheManager


def suffix_cumsum(tensor, dim=-1, dtype=torch.int32):
    return torch.cumsum(tensor.flip(dim), dim, dtype=torch.int32).flip(dim)


def calculate_static_token_equivalent(shared_prefix_length, total_lora_rank, cell_size):
    if shared_prefix_length <= 0 or total_lora_rank < 0 or cell_size <= 0:
        return 0
    base_tokens = shared_prefix_length
    mini_tokens = math.ceil(shared_prefix_length * total_lora_rank / cell_size)
    return base_tokens + mini_tokens


class MemoryAllocator:
    def __init__(
        self,
        tot_size,
        dtype,
        head_num,
        head_dim,
        layer_num,
        shared_prefix_length=0,
        lora_ranks=None,
        lora_dirs=None,
    ):
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cell_size = head_num * head_dim
        self.total_budget_size = int(tot_size)

        # Static region: one base prefix kv plus one mini kv slice per LoRA.
        self.shared_prefix_length = int(shared_prefix_length)
        self.lora_ranks = [int(rank) for rank in (lora_ranks or [])]
        self.lora_dirs = list(lora_dirs or [])
        if self.lora_dirs and len(self.lora_dirs) != len(self.lora_ranks):
            raise ValueError("lora_dirs and lora_ranks must have the same length")
        self.lora_num = len(self.lora_ranks)
        self.total_lora_rank = sum(self.lora_ranks)
        self.has_static_shared_prefix = (
            self.shared_prefix_length > 0 and self.lora_num > 0 and self.total_lora_rank > 0
        )
        self.static_token_equivalent = calculate_static_token_equivalent(
            self.shared_prefix_length, self.total_lora_rank, self.cell_size
        )
        self.static_base_token_equivalent = self.shared_prefix_length if self.has_static_shared_prefix else 0
        self.static_mini_token_equivalent = max(
            self.static_token_equivalent - self.static_base_token_equivalent, 0
        )
        if self.static_token_equivalent > self.total_budget_size:
            raise ValueError(
                "shared prefix static region exceeds the total token budget: "
                f"need {self.static_token_equivalent}, total budget {self.total_budget_size}"
            )

        # cache_size is the remaining Unified Paging pool after static
        # reservation. Restored prefix KV, suffix KV, and adapter weights all
        # allocate from this same dynamic pool.
        self.cache_size = self.total_budget_size - self.static_token_equivalent
        self.tot_size = self.cache_size

        self._init_static_layout()
        self.prefix_cache = PrefixKVCacheManager(self) if self.has_static_shared_prefix else None
        self.reset_all_pool()

    def _dtype_size_bytes(self):
        if self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            return 2
        if self.dtype == torch.float32:
            return 4
        raise ValueError(f"unsupported dtype for size accounting: {self.dtype}")

    def _init_static_layout(self):
        rank_offsets = [0]
        for rank in self.lora_ranks:
            rank_offsets.append(rank_offsets[-1] + rank)
        self.lora_rank_offsets_cpu = rank_offsets
        self.lora_rank_offsets = torch.tensor(rank_offsets, dtype=torch.long, device="cuda")
        self.lora_rank_tensor = torch.tensor(self.lora_ranks, dtype=torch.long, device="cuda")
        self.lora_dir_to_index = {lora_dir: idx for idx, lora_dir in enumerate(self.lora_dirs)}
        self.lora_index_to_dir = list(self.lora_dirs)

        self.static_base_key_buffer = None
        self.static_base_value_buffer = None
        self.static_mini_key_buffer = None
        self.static_mini_value_buffer = None
        if not self.has_static_shared_prefix:
            return

        # Layout rationale:
        # - base kv: [layer, prefix_token, head, dim]
        # - mini kv: [layer, prefix_token, packed_rank]
        # Each LoRA occupies one contiguous rank slice inside packed_rank so recovery can
        # read a single contiguous mini-kv slice and multiply it with the matching lora_B.
        self.static_base_key_buffer = [
            torch.empty(
                (self.shared_prefix_length, self.head_num, self.head_dim),
                dtype=self.dtype,
                device="cuda",
            )
            for _ in range(self.layer_num)
        ]
        self.static_base_value_buffer = [
            torch.empty(
                (self.shared_prefix_length, self.head_num, self.head_dim),
                dtype=self.dtype,
                device="cuda",
            )
            for _ in range(self.layer_num)
        ]
        self.static_mini_key_buffer = [
            torch.empty(
                (self.shared_prefix_length, self.total_lora_rank),
                dtype=self.dtype,
                device="cuda",
            )
            for _ in range(self.layer_num)
        ]
        self.static_mini_value_buffer = [
            torch.empty(
                (self.shared_prefix_length, self.total_lora_rank),
                dtype=self.dtype,
                device="cuda",
            )
            for _ in range(self.layer_num)
        ]
        self._zero_init_static_layout()

    def _zero_init_static_layout(self):
        if not self.has_static_shared_prefix:
            return
        for buf in (
            self.static_base_key_buffer,
            self.static_base_value_buffer,
            self.static_mini_key_buffer,
            self.static_mini_value_buffer,
        ):
            for layer_buf in buf:
                layer_buf.zero_()

    def get_memory_size(self):
        dsize = self._dtype_size_bytes()
        dynamic_size = 2 * self.layer_num * self.tot_size * self.cell_size * dsize
        static_size = 0
        if self.has_static_shared_prefix:
            static_size += 2 * self.layer_num * self.shared_prefix_length * self.cell_size * dsize
            static_size += 2 * self.layer_num * self.shared_prefix_length * self.total_lora_rank * dsize
        return dynamic_size + static_size

    def get_static_kv_shape(self):
        return {
            "shared_prefix_length": self.shared_prefix_length,
            "lora_num": self.lora_num,
            "lora_dirs": list(self.lora_dirs),
            "lora_ranks": list(self.lora_ranks),
            "total_lora_rank": self.total_lora_rank,
            "total_budget_size": self.total_budget_size,
            "static_token_equivalent": self.static_token_equivalent,
            "unified_paging_cache_size": self.cache_size,
            "dynamic_tot_size": self.tot_size,
            "base_shape": None
            if not self.has_static_shared_prefix
            else tuple(self.static_base_key_buffer[0].shape),
            "mini_shape": None
            if not self.has_static_shared_prefix
            else tuple(self.static_mini_key_buffer[0].shape),
        }

    def get_lora_rank_range(self, lora_index):
        start = self.lora_rank_offsets_cpu[lora_index]
        end = self.lora_rank_offsets_cpu[lora_index + 1]
        return start, end

    def get_lora_rank_range_by_dir(self, lora_dir):
        if lora_dir not in self.lora_dir_to_index:
            raise KeyError(f"unknown lora dir: {lora_dir}")
        return self.get_lora_rank_range(self.lora_dir_to_index[lora_dir])

    def get_static_prefix_view(self, lora_index):
        if not self.has_static_shared_prefix:
            raise ValueError("shared prefix static region is disabled")
        start, end = self.get_lora_rank_range(lora_index)
        return {
            "base_key": self.static_base_key_buffer,
            "base_value": self.static_base_value_buffer,
            "mini_key": [layer_buf[:, start:end] for layer_buf in self.static_mini_key_buffer],
            "mini_value": [layer_buf[:, start:end] for layer_buf in self.static_mini_value_buffer],
            "rank_start": start,
            "rank_end": end,
        }

    def get_static_prefix_view_by_dir(self, lora_dir):
        if lora_dir not in self.lora_dir_to_index:
            raise KeyError(f"unknown lora dir: {lora_dir}")
        return self.get_static_prefix_view(self.lora_dir_to_index[lora_dir])

    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            raise Exception(
                f"warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}"
            )

        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= len(select_index)
        return select_index

    def alloc_contiguous(self, need_size):
        if need_size > self.can_use_mem_size:
            raise Exception(
                f"warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}"
            )

        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = (
            self._mem_cum_sum[need_size - 1 : self.tot_size]
            - self._mem_cum_sum[0 : self.tot_size - need_size + 1]
            + self.mem_state[0 : self.tot_size - need_size + 1]
        )
        can_used_loc = self.indexes[0 : self.tot_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]

        self.mem_state[select_index] = 0
        self.can_use_mem_size -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end

    def alloc_strip(self, need_block, block_size):
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = (
            self._mem_cum_sum[block_size - 1 : self.tot_size]
            - self._mem_cum_sum[0 : self.tot_size - block_size + 1]
            + self.mem_state[0 : self.tot_size - block_size + 1]
        )
        loc_use = loc_sums == block_size
        torch.cumsum(loc_use, dim=0, dtype=torch.int32, out=loc_sums)

        block_start = torch.empty((loc_use.shape[0]), dtype=torch.int32, device="cuda")
        block_start[0] = loc_use[0]
        block_start[1:] = (loc_use[:-1] == 0) & (loc_use[1:] == 1)

        cum_max, _ = torch.cummax(block_start, dim=0)
        mask = block_size - 1
        loc_use = (((loc_sums - cum_max) & mask) == 0) & loc_use
        can_use_loc = self.indexes[0 : self.tot_size - block_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_block:
            raise Exception(
                f"no enough pool space for alloc_strip, need {need_block} blocks, {can_use_loc.shape[0]} left"
            )
        can_use_loc = can_use_loc[:need_block]
        select_index = torch.empty((block_size, need_block), dtype=torch.int32, device="cuda")
        for i in range(block_size):
            select_index[i] = can_use_loc + i
        select_index = select_index.T.reshape(-1)

        self.mem_state[select_index] = 0
        self.can_use_mem_size -= select_index.shape[0]
        return select_index

    def alloc_grid(self, need_grid, grid_size):
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = (
            self._mem_cum_sum[grid_size - 1 : self.tot_size]
            - self._mem_cum_sum[0 : self.tot_size - grid_size + 1]
            + self.mem_state[0 : self.tot_size - grid_size + 1]
        )
        loc_use = loc_sums == grid_size

        mask = grid_size - 1
        loc_use = ((self.indexes[: self.tot_size - grid_size + 1] & mask) == 0) & loc_use
        can_use_loc = self.indexes[0 : self.tot_size - grid_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_grid:
            raise Exception(
                f"no enough pool space for alloc_strip, need {need_grid} grids, {can_use_loc.shape[0]} left"
            )
        can_use_loc = can_use_loc[:need_grid]
        select_index = torch.empty((grid_size, need_grid), dtype=torch.int32, device="cuda")
        for i in range(grid_size):
            select_index[i] = can_use_loc + i
        select_index = select_index.T.reshape(-1)

        self.mem_state[select_index] = 0
        self.can_use_mem_size -= select_index.shape[0]
        return select_index

    def alloc_prefix(self, need_size):
        assert False

    def alloc_contiguous_prefix(self, need_size):
        assert False

    def alloc_suffix(self, need_size):
        assert False

    def alloc_contiguous_suffix(self, need_size):
        assert False

    def free(self, free_index):
        self.can_use_mem_size += free_index.shape[0]
        self.mem_state[free_index] = 1
        return

    def free_all(self):
        self.mem_state[:] = 1
        self.can_use_mem_size = self.tot_size

    def delete_all_pool(self):
        self.mem_state = None
        self._mem_cum_sum = None
        self.indexes = None
        self.can_use_mem_size = 0
        self.key_buffer = None
        self.value_buffer = None
        self.static_base_key_buffer = None
        self.static_base_value_buffer = None
        self.static_mini_key_buffer = None
        self.static_mini_value_buffer = None
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
        gc.collect()

    def delete_all_cache(self):
        self.delete_all_pool()

    def reset_all_pool(self):
        self.mem_state = torch.ones((self.tot_size,), dtype=torch.bool, device="cuda")
        self._mem_cum_sum = torch.empty((self.tot_size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, self.tot_size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = self.tot_size
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
        self.key_buffer = [
            torch.empty((self.tot_size, self.head_num, self.head_dim), dtype=self.dtype, device="cuda")
            for _ in range(self.layer_num)
        ]
        self.value_buffer = [
            torch.empty((self.tot_size, self.head_num, self.head_dim), dtype=self.dtype, device="cuda")
            for _ in range(self.layer_num)
        ]

    def reset_all_cache(self):
        self.reset_all_pool()
