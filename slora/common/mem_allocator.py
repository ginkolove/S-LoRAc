import gc
import torch


# TODO: will it slow down the program?
def suffix_cumsum(tensor, dim=-1, dtype=torch.int32):
    return torch.cumsum(tensor.flip(dim), dim, dtype=torch.int32).flip(dim)


class MemoryAllocator:
    def __init__(self, tot_size, cache_size, dtype, head_num, head_dim, layer_num, system_prompt_lens=0,adapter_dirs=None):
        assert tot_size >= cache_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cell_size = head_num * head_dim

        self.tot_size = tot_size
        self.cache_size = cache_size
        # 新增：system_prompt专用空间大小
        self.system_prompt_lens = system_prompt_lens
        self.adapter_dirs = adapter_dirs
        # system_prompt缓存相关
        self.system_prompt_kv_caches = {}  # {adapter_dir: (key_cache, value_cache)}
        self.base_system_prompt_kv = None  # 基础模型的system_prompt KV缓存

        self.reset_all_pool()
    

    def get_memory_size(self):
        dsize = 2 if self.dtype == torch.float16 else None
        total_system_prompt_lens = self.system_prompt_lens * len(self.system_prompt_kv_caches)
        return 2 * self.layer_num * (self.tot_size + total_system_prompt_lens) * self.cell_size * dsize


    def init_system_prompt_cache(self):
        """
        初始化system_prompt的KV缓存空间
        
        Args:
            adapter_dirs (List[str]): 所有adapter的目录列表
            system_prompt_tokens (int): system_prompt的token数量
        """
        
        # 为基础模型分配system_prompt KV缓存
        if self.system_prompt_lens > 0:
            self.base_system_prompt_kv = {
                'key_cache': [torch.empty((self.system_prompt_lens, self.head_num, self.head_dim),
                                        dtype=self.dtype, device="cuda")
                            for _ in range(self.layer_num)],
                'value_cache': [torch.empty((self.system_prompt_lens, self.head_num, self.head_dim),
                                          dtype=self.dtype, device="cuda")
                              for _ in range(self.layer_num)]
            }
            
            # 为每个adapter分配system_prompt KV缓存
            for adapter_dir in self.adapter_dirs:
                self.system_prompt_kv_caches[adapter_dir] = {
                    'key_cache': [torch.empty((self.system_prompt_lens, self.head_num, self.head_dim),
                                            dtype=self.dtype, device="cuda")
                                for _ in range(self.layer_num)],
                    'value_cache': [torch.empty((self.system_prompt_lens, self.head_num, self.head_dim),
                                              dtype=self.dtype, device="cuda")
                                  for _ in range(self.layer_num)]
                }


    def get_system_prompt_kv(self, adapter_dir=None):
        """
        获取system_prompt的KV缓存
        
        Args:
            adapter_dir (str, optional): adapter目录，None表示基础模型
            
        Returns:
            dict: 包含key_cache和value_cache的字典，如果不存在则返回None
        """
        if adapter_dir is None:
            return self.base_system_prompt_kv
        else:
            return self.system_prompt_kv_caches.get(adapter_dir, None)


    def set_system_prompt_kv(self, key_states_list, value_states_list, adapter_dir=None):
        """
        设置system_prompt的KV缓存
        
        Args:
            key_states_list (List[torch.Tensor]): 每层的key状态
            value_states_list (List[torch.Tensor]): 每层的value状态  
            adapter_dir (str, optional): adapter目录，None表示基础模型
        """
        if adapter_dir is None:
            if self.base_system_prompt_kv is not None:
                for i in range(self.layer_num):
                    if i < len(self.base_system_prompt_kv['key_cache']) and i < len(key_states_list):
                        self.base_system_prompt_kv['key_cache'][i].copy_(key_states_list[i])
                    if i < len(self.base_system_prompt_kv['value_cache']) and i < len(value_states_list):
                        self.base_system_prompt_kv['value_cache'][i].copy_(value_states_list[i])
        else:
            if adapter_dir in self.system_prompt_kv_caches:
                cache = self.system_prompt_kv_caches[adapter_dir]
                for i in range(self.layer_num):
                    if i < len(cache['key_cache']) and i < len(key_states_list):
                        cache['key_cache'][i].copy_(key_states_list[i])
                    if i < len(cache['value_cache']) and i < len(value_states_list):
                        cache['value_cache'][i].copy_(value_states_list[i])

    
    def has_system_prompt_cache(self, adapter_dir=None):
        """
        检查是否有system_prompt缓存
        
        Args:
            adapter_dir (str, optional): adapter目录，None表示基础模型
            
        Returns:
            bool: 是否有缓存
        """
        if adapter_dir is None:
            return self.base_system_prompt_kv is not None
        else:
            return adapter_dir in self.system_prompt_kv_caches
  

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
        self.mem_state = torch.ones((self.tot_size,), dtype=torch.bool, device="cuda")
        self._mem_cum_sum = torch.empty((self.tot_size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, self.tot_size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = self.tot_size
        # self.can_use_mem_size_prefix = self.cache_size
        # self.can_use_mem_size_suffix = self.tot_size - self.cache_size
        self.key_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                       dtype=self.dtype, device="cuda")
                           for _ in range(self.layer_num)]
        self.value_buffer = [torch.empty((self.tot_size, self.head_num, self.head_dim),
                                       dtype=self.dtype, device="cuda")
                           for _ in range(self.layer_num)]
        self.init_system_prompt_cache()

    def reset_all_cache(self):
        self.reset_all_pool()
