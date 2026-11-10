import gc
import torch


# TODO: will it slow down the program?
def suffix_cumsum(tensor, dim=-1, dtype=torch.int32):
    return torch.cumsum(tensor.flip(dim), dim, dtype=torch.int32).flip(dim)


class MemoryAllocator:
    def __init__(self, tot_size, cache_size, dtype, head_num, head_dim, layer_num,
                 dynamic_layer_workspace=False, device="cuda"):
        assert tot_size >= cache_size
        if layer_num <= 0:
            raise ValueError("invalid layer count")
        self.device = device
        self.dynamic_layer_workspace = bool(dynamic_layer_workspace)
        self.budget_size = int(tot_size)
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cell_size = head_num * head_dim

        self.tot_size = int(tot_size)
        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError("no KV budget")

        self.reset_all_pool()


    def get_memory_size(self):
        dsize = torch.empty((), dtype=self.dtype).element_size()
        return 2 * self.layer_num * self.tot_size * self.cell_size * dsize

    def alloc_layer_workspace(self, token_count):
        if not self.dynamic_layer_workspace:
            raise RuntimeError("dynamic layer workspace is not enabled")
        if token_count <= 0 or self._workspace_indices is not None:
            raise ValueError("workspace needs positive tokens and exclusive ownership")
        page_count = (token_count + self.layer_num - 1) // self.layer_num
        indices = self.alloc(page_count)
        try:
            # Each reserved Unified Paging slot has storage in every layer. Flatten
            # those layer slices into one scratch address space, without copying KV.
            offsets = torch.arange(self.layer_num, device=self.device, dtype=torch.long)
            locations = (offsets[:, None] * self.tot_size + indices[None, :]).reshape(-1)[:token_count]
        except Exception:
            self.free(indices)
            raise
        self._workspace_indices = indices
        self.layer_workspace_tokens = int(token_count)
        self.workspace_token_equiv = page_count
        self.peak_workspace_tokens = max(self.peak_workspace_tokens, token_count)
        self.peak_workspace_token_equiv = max(self.peak_workspace_token_equiv, page_count)
        return locations

    def alloc_layer_storage(self, token_count):
        """Reserve persistent single-layer cells in ordinary full-layer slots."""
        if not self.dynamic_layer_workspace or token_count <= 0:
            raise ValueError("layer storage requires positive tokens and flattened pool backing")
        indices = self.alloc((token_count + self.layer_num - 1) // self.layer_num)
        try:
            offsets = torch.arange(self.layer_num, device=self.device, dtype=torch.long)
            locations = (offsets[:, None] * self.tot_size + indices[None, :]).reshape(-1)[:token_count]
            return indices, locations
        except Exception:
            self.free(indices)
            raise

    def free_layer_workspace(self):
        if self._workspace_indices is not None:
            self.free(self._workspace_indices)
            self._workspace_indices = None
        self.layer_workspace_tokens = 0
        self.workspace_token_equiv = 0
  

    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= len(select_index)
        return select_index


    def alloc_contiguous(self, need_size):
        if need_size > self.can_use_mem_size:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[need_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - need_size + 1] + self.mem_state[0:self.tot_size - need_size + 1]
        can_used_loc = self.indexes[0:self.tot_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size}')
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
        loc_sums = self._mem_cum_sum[block_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - block_size + 1] + self.mem_state[0:self.tot_size - block_size + 1]
        loc_use = (loc_sums == block_size)
        torch.cumsum(loc_use, dim=0, dtype=torch.int32, out=loc_sums)

        block_start = torch.empty((loc_use.shape[0]), dtype=torch.int32, device="cuda")
        block_start[0] = loc_use[0]
        block_start[1:] = (loc_use[:-1] == 0) & (loc_use[1:] == 1)

        cum_max, _ = torch.cummax(block_start, dim=0)
        # (diff % block_size == 0) & loc_use
        mask = block_size - 1
        loc_use = (((loc_sums - cum_max) & mask) == 0) & loc_use
        can_use_loc = self.indexes[0:self.tot_size - block_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_block:
            raise Exception(f"no enough pool space for alloc_strip, "
                            f"need {need_block} blocks, {can_use_loc.shape[0]} left")
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
        loc_sums = self._mem_cum_sum[grid_size - 1:self.tot_size] - self._mem_cum_sum[0:self.tot_size - grid_size + 1] + self.mem_state[0:self.tot_size - grid_size + 1]
        loc_use = (loc_sums == grid_size)

        mask = grid_size - 1
        loc_use = ((self.indexes[:self.tot_size - grid_size + 1] & mask) == 0) & loc_use
        can_use_loc = self.indexes[0:self.tot_size - grid_size + 1][loc_use == 1]
        if can_use_loc.shape[0] < need_grid:
            raise Exception(f"no enough pool space for alloc_strip, "
                            f"need {need_grid} grids, {can_use_loc.shape[0]} left")
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
        if need_size > self.can_use_mem_size_prefix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_prefix}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size_prefix -= len(select_index)
        return select_index
    

    def alloc_contiguous_prefix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_prefix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_prefix}')
        
        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        loc_sums = self._mem_cum_sum[need_size - 1:self.cache_size] - self._mem_cum_sum[0:self.cache_size - need_size + 1] + self.mem_state[0:self.cache_size - need_size + 1]
        can_used_loc = self.indexes[0:self.cache_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size_prefix}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size_prefix -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end


    def alloc_suffix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_suffix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        
        self._mem_cum_sum = suffix_cumsum(self.mem_state, dim=0, dtype=torch.int32)
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size_suffix -= len(select_index)
        return select_index
    

    def alloc_contiguous_suffix(self, need_size):
        assert False
        if need_size > self.can_use_mem_size_suffix:
            raise Exception(f'warn no enough pool space: need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        
        self._mem_cum_sum = suffix_cumsum(self.mem_state, dim=0, dtype=torch.int32)
        assert len(self._mem_cum_sum) == self.cache_size
        loc_sums = (self._mem_cum_sum[0:self.cache_size - need_size + 1] - self._mem_cum_sum[need_size - 1:] +
                    self.mem_state[need_size - 1:])
        can_used_loc = self.indexes[0:self.cache_size - need_size + 1][loc_sums == need_size]
        if can_used_loc.shape[0] == 0:
            # print(f'warn no enough pool space: to contiguous need_size {need_size} left_size {self.can_use_mem_size_suffix}')
            return None
        start_loc = can_used_loc[0]
        select_index = self.indexes[start_loc : start_loc + need_size]
        
        self.mem_state[select_index] = 0
        self.can_use_mem_size_suffix -= need_size
        start = start_loc.item()
        end = start + need_size
        return select_index, start, end
 
    
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        self.can_use_mem_size += free_index.shape[0]
        # self.can_use_mem_size_prefix += torch.sum(free_index < self.cache_size)
        # self.can_use_mem_size_suffix += torch.sum(free_index >= self.cache_size)
        self.mem_state[free_index] = 1

        # if self.can_use_mem_size_prefix + self.can_use_mem_size_suffix == self.tot_size:
        #     print(f"freed all gpu mem size {self.tot_size}")
        # print(f"free state {self.can_use_mem_size_prefix} + {self.can_use_mem_size_suffix} all {self.tot_size}")
        return
    
    def free_all(self):
        self._workspace_indices = None
        self.layer_workspace_tokens = 0
        self.workspace_token_equiv = 0
        self.peak_workspace_tokens = 0
        self.peak_workspace_token_equiv = 0
        self.mem_state[:] = 1
        self.can_use_mem_size = self.tot_size
        # self.can_use_mem_size_prefix = self.cache_size
        # self.can_use_mem_size_suffix = self.tot_size - self.cache_size
    

    def delete_all_pool(self):
        self.mem_state = None
        self._mem_cum_sum = None
        self.indexes = None
        self.can_use_mem_size = 0
        # self.can_use_mem_size_prefix = 0
        # self.can_use_mem_size_suffix = 0
        self.buffer = None
        gc.collect()

    def delete_all_cache(self):
        self.delete_all_pool()


    def reset_all_pool(self):
        self._workspace_indices = None
        self.layer_workspace_tokens = 0
        self.workspace_token_equiv = 0
        self.peak_workspace_tokens = 0
        self.peak_workspace_token_equiv = 0
        self.mem_state = torch.ones((self.tot_size,), dtype=torch.bool, device=self.device)
        self._mem_cum_sum = torch.empty((self.tot_size,), dtype=torch.int32, device=self.device)
        self.indexes = torch.arange(0, self.tot_size, dtype=torch.long, device=self.device)
        self.can_use_mem_size = self.tot_size
        # self.can_use_mem_size_prefix = self.cache_size
        # self.can_use_mem_size_suffix = self.tot_size - self.cache_size
        if self.dynamic_layer_workspace:
            shape = (self.layer_num, self.tot_size, self.head_num, self.head_dim)
            keys = torch.empty(shape, dtype=self.dtype, device=self.device)
            values = torch.empty(shape, dtype=self.dtype, device=self.device)
            self.key_buffer = list(keys.unbind(0))
            self.value_buffer = list(values.unbind(0))
            self.prefix_workspace_k = keys.view(-1, self.head_num, self.head_dim)
            self.prefix_workspace_v = values.view(-1, self.head_num, self.head_dim)
        else:
            self.key_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                           dtype=self.dtype, device=self.device)
                               for _ in range(self.layer_num)]
            self.value_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                           dtype=self.dtype, device=self.device)
                                 for _ in range(self.layer_num)]
            self.prefix_workspace_k = None
            self.prefix_workspace_v = None
 

    def reset_all_cache(self):
        self.reset_all_pool()
