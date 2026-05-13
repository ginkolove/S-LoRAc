import uuid
import time
from collections import OrderedDict

from ..io_struct import Batch
from .req_queue import ReqQueue


class LowRAPrefixAwareReqQueue(ReqQueue):
    """Naive cache-aware LoRA clustering for LowRA admission."""

    def __init__(
        self,
        max_total_tokens,
        batch_max_tokens,
        running_max_req_size,
        prefix_len,
        window_n,
        wait_slo_ms=0.0,
    ) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.prefix_len = int(prefix_len)
        self.window_size = max(1, int(window_n) - 2)
        self.wait_slo_s = max(0.0, float(wait_slo_ms) / 1000.0)
        self.prefix_cache_state = {
            "cached_lora_dirs": set(),
            "active_lora_dirs": set(),
            "inactive_lora_dirs": set(),
            "heat_by_lora": {},
            "free_dynamic_slots": self.max_total_tokens,
        }

    def update_prefix_cache_state(self, state):
        if state is None:
            return
        self.prefix_cache_state = state

    def _non_aborted_waiting_with_order(self):
        return [
            (idx, req)
            for idx, req in enumerate(self.waiting_req_list)
            if not req.aborted
        ]

    def _unique_loras(self, reqs):
        return {req.adapter_dir for req in reqs if req.adapter_dir is not None}

    def _query_len(self, req):
        return int(getattr(req, "lowra_query_len", req.input_len))

    def _decode_len(self, req):
        return int(getattr(req, "lowra_decode_len", req.max_output_len))

    def _remaining_decode_after_current_step(self, req):
        return max(0, self._decode_len(req) - len(req.output_ids) - 1)

    def _private_peak_need(self, current_batch, candidate):
        cache_len_list = []
        if current_batch is not None:
            for req in current_batch.reqs:
                cache_len_list.append((
                    self._query_len(req) + len(req.output_ids),
                    self._remaining_decode_after_current_step(req),
                ))
        for req in candidate:
            cache_len_list.append((self._query_len(req) + 1, max(0, self._decode_len(req) - 1)))
        if not cache_len_list:
            return 0

        cache_len_list.sort(key=lambda x: -x[1])
        cum_run = 0
        need = 0
        for i, (has_run_len, left_out_len) in enumerate(cache_len_list, start=1):
            cum_run += has_run_len
            need = max(need, left_out_len * i + cum_run)
        return need

    def _adapter_need(self, current_batch, candidate, lora_ranks):
        loras = set()
        if current_batch is not None:
            loras |= self._unique_loras(current_batch.reqs)
        loras |= self._unique_loras(candidate)
        return sum(lora_ranks[lora_dir] * 4 for lora_dir in loras)

    def _prefix_misses(self, candidate):
        cached = self.prefix_cache_state.get("cached_lora_dirs", set())
        return self._unique_loras(candidate) - cached

    def _resident_prefix_need(self, candidate, evict_prefixes=()):
        cached = set(self.prefix_cache_state.get("cached_lora_dirs", set()))
        retained = cached - set(evict_prefixes)
        misses = self._prefix_misses(candidate)
        return self.prefix_len * len(retained | misses)

    def _total_need(self, current_batch, candidate, lora_ranks, evict_prefixes=()):
        return (
            self._private_peak_need(current_batch, candidate)
            + self._adapter_need(current_batch, candidate, lora_ranks)
            + self._resident_prefix_need(candidate, evict_prefixes)
        )

    def _fits(self, current_batch, candidate, lora_ranks, evict_prefixes=()):
        if not candidate:
            return False
        if current_batch is not None and len(current_batch.reqs) + len(candidate) > self.running_max_req_size:
            return False
        return self._total_need(current_batch, candidate, lora_ranks, evict_prefixes) <= self.max_total_tokens

    def _sorted_evictable_prefixes_for_candidate(self, candidate, current_batch):
        running_loras = self._unique_loras(current_batch.reqs) if current_batch is not None else set()
        pinned = running_loras | self._unique_loras(candidate)
        inactive = self.prefix_cache_state.get("inactive_lora_dirs", set())
        heat = self.prefix_cache_state.get("heat_by_lora", {})
        evictable = [lora_dir for lora_dir in inactive if lora_dir not in pinned]
        evictable.sort(key=lambda lora_dir: heat.get(lora_dir, 0))
        return evictable

    def _eviction_plan_for_candidate(self, candidate, current_batch, lora_ranks, base_plan=()):
        running_loras = self._unique_loras(current_batch.reqs) if current_batch is not None else set()
        pinned = running_loras | self._unique_loras(candidate)
        plan = [lora_dir for lora_dir in (base_plan or []) if lora_dir not in pinned]
        if self._fits(current_batch, candidate, lora_ranks, evict_prefixes=plan):
            return plan

        planned = set(plan)
        for lora_dir in self._sorted_evictable_prefixes_for_candidate(candidate, current_batch):
            if lora_dir in planned:
                continue
            plan.append(lora_dir)
            planned.add(lora_dir)
            if self._fits(current_batch, candidate, lora_ranks, evict_prefixes=plan):
                return plan
        return None

    def _attach_lowra_plan(self, batch, evict_prefixes):
        batch.lowra_evict_prefix_lora_dirs = list(evict_prefixes or [])
        batch.lowra_prefix_miss_lora_dirs = list(self._prefix_misses(batch.reqs))
        return batch

    def _make_batch(self, reqs, evict_prefixes=()):
        selected_ids = {req.request_id for req in reqs}
        self.waiting_req_list = [
            req for req in self.waiting_req_list
            if req.request_id not in selected_ids and not req.aborted
        ]
        return self._attach_lowra_plan(Batch(uuid.uuid4().hex, reqs), evict_prefixes)

    def _cluster_waiting_groups(self, exclude_ids=None):
        exclude_ids = exclude_ids or set()
        groups = OrderedDict()
        for idx, req in self._non_aborted_waiting_with_order():
            if req.request_id in exclude_ids:
                continue
            if req.adapter_dir not in groups:
                groups[req.adapter_dir] = {
                    "lora_dir": req.adapter_dir,
                    "reqs": [],
                    "first_order": idx,
                }
            groups[req.adapter_dir]["reqs"].append(req)

        cached = self.prefix_cache_state.get("cached_lora_dirs", set())
        clustered = list(groups.values())
        clustered.sort(key=lambda group: (
            group["lora_dir"] not in cached,
            -len(group["reqs"]),
            group["first_order"],
        ))
        return clustered

    def _urgent_waiting_reqs(self, now):
        if self.wait_slo_s <= 0:
            return []
        urgent = []
        for idx, req in self._non_aborted_waiting_with_order():
            arrival_time = getattr(req, "arrival_time", now)
            if now - arrival_time >= self.wait_slo_s:
                urgent.append((arrival_time + self.wait_slo_s, idx, req))
        urgent.sort(key=lambda item: (item[0], item[1]))
        return [req for _, _, req in urgent]

    def _try_select_req(self, selected, req, current_batch, lora_ranks, evict_plan):
        trial = selected + [req]
        plan = self._eviction_plan_for_candidate(
            trial, current_batch, lora_ranks, evict_plan
        )
        if plan is None:
            return None
        return trial, plan

    def _try_select_group_fifo(self, selected, reqs, current_batch, lora_ranks, evict_plan):
        full_trial = selected + reqs
        full_plan = self._eviction_plan_for_candidate(
            full_trial, current_batch, lora_ranks, evict_plan
        )
        if full_plan is not None:
            return full_trial, full_plan

        cur_selected = selected
        cur_plan = evict_plan
        progressed = False
        for req in reqs:
            result = self._try_select_req(
                cur_selected, req, current_batch, lora_ranks, cur_plan
            )
            if result is None:
                break
            cur_selected, cur_plan = result
            progressed = True
        if not progressed:
            return None
        return cur_selected, cur_plan

    def _same_lora_followers(self, lora_dir, exclude_ids):
        return [
            req
            for _, req in self._non_aborted_waiting_with_order()
            if req.request_id not in exclude_ids and req.adapter_dir == lora_dir
        ]

    def _select_slo_urgent_prefix(self, current_batch, lora_ranks):
        selected = []
        evict_plan = []
        selected_ids = set()
        for req in self._urgent_waiting_reqs(time.time()):
            if req.request_id in selected_ids:
                continue
            result = self._try_select_req(
                selected, req, current_batch, lora_ranks, evict_plan
            )
            if result is None:
                continue
            selected, evict_plan = result
            selected_ids.add(req.request_id)

            same_lora_reqs = self._same_lora_followers(req.adapter_dir, selected_ids)
            if not same_lora_reqs:
                continue
            result = self._try_select_group_fifo(
                selected, same_lora_reqs, current_batch, lora_ranks, evict_plan
            )
            if result is None:
                continue
            selected, evict_plan = result
            selected_ids = {req.request_id for req in selected}
        return selected, evict_plan

    def generate_new_batch(self, current_batch: Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        selected, evict_plan = self._select_slo_urgent_prefix(current_batch, lora_ranks)
        selected_ids = {req.request_id for req in selected}

        clustered_groups = self._cluster_waiting_groups(exclude_ids=selected_ids)
        if not clustered_groups:
            self.waiting_req_list = [req for req in self.waiting_req_list if not req.aborted]
            if selected:
                return self._make_batch(selected, evict_plan)
            return None

        for group in clustered_groups:
            group_reqs = group["reqs"]
            result = self._try_select_group_fifo(
                selected, group_reqs, current_batch, lora_ranks, evict_plan
            )
            if result is not None:
                selected, evict_plan = result

        if not selected:
            return None
        return self._make_batch(selected, evict_plan)

    def next_batch(self):
        selected = []
        for group in self._cluster_waiting_groups():
            if len(selected) >= self.running_max_req_size:
                break
            for req in group["reqs"]:
                if len(selected) >= self.running_max_req_size:
                    break
                selected.append(req)
        if selected:
            return Batch(uuid.uuid4().hex, selected)
        return None
