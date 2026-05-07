import uuid
import numpy as np
from typing import List

from ..io_struct import Batch, Req


class PrefixSLoraReqQueue:
    def __init__(
        self,
        max_total_tokens,
        batch_max_tokens,
        running_max_req_size,
        adapter_dirs: List[str],
        shared_prefix_len: int,
        gpu_prefix_num: int,
        cpu_prefix_num: int,
    ) -> None:
        self.shared_prefix_len = shared_prefix_len
        self.gpu_lora_dirs = set(adapter_dirs[:gpu_prefix_num])
        self.cpu_lora_dirs = set(adapter_dirs[gpu_prefix_num:gpu_prefix_num + cpu_prefix_num])
        self.hot_reserved_tokens = gpu_prefix_num * shared_prefix_len
        self.max_total_tokens = max_total_tokens - self.hot_reserved_tokens
        assert self.max_total_tokens > 0
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []

    def append(self, req):
        self.waiting_req_list.append(req)
        return

    def _is_cold_prefix_req(self, req):
        return req.adapter_dir in self.cpu_lora_dirs

    def _init_cache_list(self, current_batch: Batch, lora_ranks):
        self.cache_len_list = []
        self.adapters = set()
        self.adapter_size = 0
        self.cold_prefixes = set()
        self.cold_prefix_size = 0
        if current_batch is None:
            return

        for req in current_batch.reqs:
            self.cache_len_list.append((
                req.input_len + len(req.output_ids),
                req.max_output_len - len(req.output_ids) - 1,
            ))
            if req.adapter_dir not in self.adapters:
                self.adapter_size += lora_ranks[req.adapter_dir] * 4
                self.adapters.add(req.adapter_dir)
            if self._is_cold_prefix_req(req) and req.adapter_dir not in self.cold_prefixes:
                self.cold_prefixes.add(req.adapter_dir)
                self.cold_prefix_size += self.shared_prefix_len

    def _can_add_new_req(self, req, lora_ranks):
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        if self._is_cold_prefix_req(req) and req.adapter_dir not in self.cold_prefixes:
            self.cold_prefixes.add(req.adapter_dir)
            self.cold_prefix_size += self.shared_prefix_len

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (
            need_max_token_num
            < self.max_total_tokens - self.adapter_size - self.cold_prefix_size
            and len(self.cache_len_list) <= self.running_max_req_size
        ):
            return True
        return False

    def update_counter(self, req):
        pass

    def generate_new_batch(self, current_batch: Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.aborted:
                aborted_count += 1
                continue
            if (
                self._can_add_new_req(req, lora_ranks)
                and new_batch_total_tokens + req.input_len <= self.batch_max_tokens
            ):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        return None

    def next_batch(self):
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            return Batch(uuid.uuid4().hex, next_batch)
        return None
