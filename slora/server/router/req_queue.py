import uuid
import asyncio
import numpy as np
import os
from typing import List, Optional, Set
from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time


class ReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, shared_prefix_length=0) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.shared_prefix_length = int(shared_prefix_length or 0)
        self.waiting_req_list: List[Req] = []
        self.cache_aware_window_size = 32
        self.starvation_priority_threshold = 2
        self.scheduler_debug_enabled = os.environ.get("SLORA_SCHED_DEBUG", "0") == "1"

    def _debug(self, message):
        if self.scheduler_debug_enabled:
            print(f"[sched-queue] {message}", flush=True)
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return

    def prepend_reqs(self, reqs):
        if len(reqs) == 0:
            return
        self.waiting_req_list = list(reqs) + self.waiting_req_list
        return
    
    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        self.prefix_cost_adapters = set()
        self.prefix_cost_total = 0
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                           req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        prefix_cost = self._reserve_prefix_cost(req)
        self.cache_len_list.append((req.input_len + prefix_cost + 1, req.max_output_len - 1)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False
    
    def update_counter(self, req):
        pass 

    def _snapshot_state(self):
        return (
            list(self.cache_len_list),
            set(self.adapters),
            self.adapter_size,
            set(self.prefix_cost_adapters),
            self.prefix_cost_total,
        )

    def _restore_state(self, snapshot):
        (
            self.cache_len_list,
            self.adapters,
            self.adapter_size,
            self.prefix_cost_adapters,
            self.prefix_cost_total,
        ) = snapshot
        return

    def _estimate_prefix_cost(self, req):
        if self.shared_prefix_length <= 0 or req.adapter_dir is None:
            return 0
        if req.adapter_dir in self.cached_prefix_adapters or req.adapter_dir in self.prefix_cost_adapters:
            return 0
        return self.shared_prefix_length

    def _reserve_prefix_cost(self, req):
        prefix_cost = self._estimate_prefix_cost(req)
        if prefix_cost > 0:
            self.prefix_cost_adapters.add(req.adapter_dir)
            self.prefix_cost_total += prefix_cost
        return prefix_cost

    def _try_add_req(self, req, lora_ranks):
        snapshot = self._snapshot_state()
        can_add = self._can_add_new_req(req, lora_ranks)
        self._restore_state(snapshot)
        return can_add

    def _get_candidate_window(self, waiting_req_list=None):
        waiting_req_list = self.waiting_req_list if waiting_req_list is None else waiting_req_list
        candidates = []
        aborted_reqs = []
        for req in waiting_req_list:
            if req.aborted:
                aborted_reqs.append(req)
                continue
            candidates.append(req)
            if len(candidates) >= self.cache_aware_window_size:
                break
        return candidates, aborted_reqs

    def _update_skip_counts(self, candidate_window, selected_reqs):
        selected_ids = {req.request_id for req in selected_reqs}
        for req in candidate_window:
            if req.request_id in selected_ids:
                req.schedule_skip_count = 0
            else:
                req.schedule_skip_count = min(
                    req.schedule_skip_count + 1,
                    self.starvation_priority_threshold,
                )
        return

    def _select_cache_aware_batch(
        self,
        current_batch: Batch,
        lora_ranks: dict[str, int],
        waiting_req_list=None,
        preferred_adapter_dirs: Optional[Set[str]] = None,
        mutate_counters: bool = True,
    ):
        waiting_req_list = self.waiting_req_list if waiting_req_list is None else waiting_req_list
        preferred_adapter_dirs = set(preferred_adapter_dirs or [])
        self.cached_prefix_adapters = set(preferred_adapter_dirs)
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        self._init_cache_list(current_batch, lora_ranks)
        candidate_window, _ = self._get_candidate_window(waiting_req_list)
        if len(candidate_window) == 0:
            self._debug("select-window empty")
            return None

        can_run_list = []
        new_batch_total_tokens = 0
        remaining_candidates = list(candidate_window)

        while len(remaining_candidates) > 0:
            feasible = []
            for order_idx, req in enumerate(remaining_candidates):
                effective_req_tokens = req.input_len + self._estimate_prefix_cost(req)
                if new_batch_total_tokens + effective_req_tokens > self.batch_max_tokens:
                    continue
                if self._try_add_req(req, lora_ranks):
                    feasible.append((
                        0 if req.schedule_skip_count >= self.starvation_priority_threshold else 1,
                        0 if req.adapter_dir in preferred_adapter_dirs else 1,
                        order_idx,
                        req,
                    ))

            if len(feasible) == 0:
                self._debug(
                    f"no-feasible current_batch_size={0 if current_batch is None else len(current_batch.reqs)} "
                    f"window={len(candidate_window)} selected={len(can_run_list)} "
                    f"selected_tokens={new_batch_total_tokens}"
                )
                break

            _, _, _, chosen_req = min(feasible, key=lambda item: item[:3])
            chosen_prefix_cost = self._estimate_prefix_cost(chosen_req)
            assert self._can_add_new_req(chosen_req, lora_ranks)
            can_run_list.append(chosen_req)
            new_batch_total_tokens += chosen_req.input_len + chosen_prefix_cost
            remaining_candidates = [req for req in remaining_candidates if req.request_id != chosen_req.request_id]

        if len(can_run_list) == 0:
            self._debug(
                f"select-empty current_batch_size={0 if current_batch is None else len(current_batch.reqs)} "
                f"waiting={len(waiting_req_list)} window={len(candidate_window)} "
                f"preferred={sorted(preferred_adapter_dirs)}"
            )
            return None

        if mutate_counters:
            self._update_skip_counts(candidate_window, can_run_list)

        self._debug(
            f"select-batch size={len(can_run_list)} effective_tokens={new_batch_total_tokens} "
            f"prefix_cost={self.prefix_cost_total} "
            f"waiting={len(waiting_req_list)} window={len(candidate_window)} "
            f"preferred={sorted(preferred_adapter_dirs)} "
            f"adapters={[req.adapter_dir for req in can_run_list]}"
        )

        return Batch(uuid.uuid4().hex, can_run_list)

    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int], preferred_adapter_dirs: Optional[Set[str]] = None):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        new_batch = self._select_cache_aware_batch(
            current_batch,
            lora_ranks,
            waiting_req_list=self.waiting_req_list,
            preferred_adapter_dirs=preferred_adapter_dirs,
            mutate_counters=True,
        )
        if new_batch is None:
            return None
        selected_ids = {req.request_id for req in new_batch.reqs}
        self.waiting_req_list = [
            req for req in self.waiting_req_list
            if (not req.aborted) and req.request_id not in selected_ids
        ]
        return new_batch

    def peek_generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int], waiting_req_list=None, preferred_adapter_dirs: Optional[Set[str]] = None):
        return self._select_cache_aware_batch(
            current_batch,
            lora_ranks,
            waiting_req_list=waiting_req_list,
            preferred_adapter_dirs=preferred_adapter_dirs,
            mutate_counters=False,
        )

    def peek_future_batches(self, current_batch:Batch, lora_ranks: dict[str, int], num_batches: int, preferred_adapter_dirs: Optional[Set[str]] = None):
        if num_batches <= 0:
            return []
        simulated_waiting = list(self.waiting_req_list)
        simulated_current = current_batch
        future_batches = []
        for _ in range(num_batches):
            next_batch = self.peek_generate_new_batch(
                simulated_current,
                lora_ranks,
                simulated_waiting,
                preferred_adapter_dirs=preferred_adapter_dirs,
            )
            if next_batch is None:
                break
            future_batches.append(next_batch)
            consumed_ids = {req.request_id for req in next_batch.reqs}
            simulated_waiting = [req for req in simulated_waiting if req.request_id not in consumed_ids]
            if simulated_current is None:
                simulated_current = next_batch
            else:
                simulated_current = Batch(uuid.uuid4().hex, list(simulated_current.reqs) + list(next_batch.reqs))
        return future_batches


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
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            return next_batch
        else:
            return None
