import uuid
from collections import Counter, OrderedDict

from ..io_struct import Batch
from .req_queue import ReqQueue


class LowRAPrefixAwareReqQueue(ReqQueue):
    """FIFO-window LowRA admission with prefix-aware eviction planning."""

    def __init__(
        self,
        max_total_tokens,
        batch_max_tokens,
        running_max_req_size,
        prefix_len,
        window_n,
    ) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.prefix_len = int(prefix_len)
        self.window_size = max(1, int(window_n) - 2)
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

    def _front_non_aborted(self, limit, start=0):
        out = []
        seen = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if seen < start:
                seen += 1
                continue
            if len(out) >= limit:
                break
            out.append(req)
            seen += 1
        return out

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

    def _candidate_tokens(self, candidate):
        return sum(self._query_len(req) for req in candidate)

    def _fits(self, current_batch, candidate, lora_ranks, evict_prefixes=()):
        if not candidate:
            return False
        if current_batch is not None and len(current_batch.reqs) + len(candidate) > self.running_max_req_size:
            return False
        if self._candidate_tokens(candidate) > self.batch_max_tokens:
            return False
        return self._total_need(current_batch, candidate, lora_ranks, evict_prefixes) <= self.max_total_tokens

    def _sorted_evictable_prefixes(self, W0, W1, current_batch):
        running_loras = self._unique_loras(current_batch.reqs) if current_batch is not None else set()
        pinned = running_loras | self._unique_loras(W0)
        inactive = self.prefix_cache_state.get("inactive_lora_dirs", set())
        heat = self.prefix_cache_state.get("heat_by_lora", {})
        w1_reuse = Counter(req.adapter_dir for req in W1 if req.adapter_dir is not None)
        evictable = [lora_dir for lora_dir in inactive if lora_dir not in pinned]
        evictable.sort(key=lambda lora_dir: (w1_reuse[lora_dir], heat.get(lora_dir, 0)))
        return evictable

    def _eviction_plan_for_W0(self, W0, W1, current_batch, lora_ranks):
        evictable = self._sorted_evictable_prefixes(W0, W1, current_batch)

        plan = []
        for lora_dir in evictable:
            plan.append(lora_dir)
            if self._fits(current_batch, W0, lora_ranks, evict_prefixes=plan):
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

    def _groups_by_lora(self, reqs):
        groups = OrderedDict()
        for req in reqs:
            groups.setdefault(req.adapter_dir, []).append(req)
        return list(groups.items())

    def _fallback_partial_W0(self, W0, current_batch, lora_ranks, evict_prefixes=()):
        cached = self.prefix_cache_state.get("cached_lora_dirs", set())
        groups = self._groups_by_lora(W0)
        resident_groups = [(lora_dir, reqs) for lora_dir, reqs in groups if lora_dir in cached]
        missing_groups = [(lora_dir, reqs) for lora_dir, reqs in groups if lora_dir not in cached]

        selected = []
        for _, group_reqs in resident_groups + missing_groups:
            for req in group_reqs:
                trial = selected + [req]
                if self._fits(current_batch, trial, lora_ranks, evict_prefixes=evict_prefixes):
                    selected.append(req)
                else:
                    break
        if not selected:
            return None
        return self._make_batch(selected, evict_prefixes)

    def generate_new_batch(self, current_batch: Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        W0 = self._front_non_aborted(self.window_size)
        if not W0:
            self.waiting_req_list = [req for req in self.waiting_req_list if not req.aborted]
            return None
        W1 = self._front_non_aborted(self.window_size, start=len(W0))

        if self._fits(current_batch, W0, lora_ranks):
            return self._make_batch(W0)

        evict_plan = self._eviction_plan_for_W0(W0, W1, current_batch, lora_ranks)
        if evict_plan is not None:
            return self._make_batch(W0, evict_plan)

        all_evictable = self._sorted_evictable_prefixes(W0, W1, current_batch)
        return self._fallback_partial_W0(W0, current_batch, lora_ranks, all_evictable)

    def next_batch(self):
        W0 = self._front_non_aborted(self.window_size)
        if W0:
            return Batch(uuid.uuid4().hex, W0)
        return None
