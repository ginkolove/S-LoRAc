"""S-LoRA FIFO admission with full-prefix sharing and inactive-cache LRU."""

import uuid

from ..io_struct import Batch
from .req_queue import ReqQueue


class FullReprefillReqQueue(ReqQueue):
    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, prefix_len):
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        if prefix_len <= 0:
            raise ValueError("full reprefill requires a positive prefix length")
        self.prefix_len = int(prefix_len)
        self.prefix_cache_state = {}

    def update_prefix_cache_state(self, state):
        self.prefix_cache_state = state or {}

    @staticmethod
    def _private_peak(running, selected):
        lengths = [(req.input_len + len(req.output_ids),
                    max(0, req.max_output_len - len(req.output_ids) - 1))
                   for req in running]
        lengths.extend((req.input_len + 1, req.max_output_len - 1) for req in selected)
        lengths.sort(key=lambda length: -length[1])
        cumulative, peak = 0, 0
        for count, (processed, remaining) in enumerate(lengths, 1):
            cumulative += processed
            peak = max(peak, cumulative + count * remaining)
        return peak

    def generate_new_batch(self, current_batch, lora_ranks):
        running = current_batch.reqs if current_batch is not None else []
        if len(running) >= self.running_max_req_size:
            return None
        state = self.prefix_cache_state
        cached = set(state.get("cached_lora_dirs", ()))
        inactive = set(state.get("inactive_lora_dirs", ()))
        last_used = state.get("last_used", {})
        prefix_sizes = state.get("prefix_sizes", {})
        pinned = set(state.get("pinned_lora_dirs", ()))
        selected, evictions = [], []
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            trial = selected + [req]
            if len(running) + len(trial) > self.running_max_req_size:
                break
            needed = {request.adapter_dir for request in running + trial}
            if None in needed:
                raise ValueError("full reprefill requires a LoRA for every request")
            victims = list(evictions)
            resident = cached - set(victims)
            misses = {request.adapter_dir for request in trial} - resident
            prefill_tokens = sum(request.input_len for request in trial) + len(misses) * self.prefix_len
            if prefill_tokens > self.batch_max_tokens:
                break
            private_peak = self._private_peak(running, trial)
            adapter_slots = sum(4 * lora_ranks[directory] for directory in needed)

            def required_slots():
                prefix_slots = sum(prefix_sizes.get(directory, self.prefix_len)
                                   for directory in resident | needed)
                return private_peak + adapter_slots + prefix_slots

            # Plan eviction only; a rejected candidate must not mutate the live cache.
            candidates = sorted((resident & inactive) - needed - pinned,
                                key=lambda directory: (last_used.get(directory, 0), directory))
            while required_slots() >= self.max_total_tokens and candidates:
                victim = candidates.pop(0)
                resident.remove(victim)
                victims.append(victim)
            if required_slots() >= self.max_total_tokens:
                break
            selected, evictions = trial, victims
        if not selected:
            return None
        selected_ids = {req.request_id for req in selected}
        self.waiting_req_list = [req for req in self.waiting_req_list
                                 if not req.aborted and req.request_id not in selected_ids]
        batch = Batch(uuid.uuid4().hex, selected)
        batch.lowra_evict_prefix_lora_dirs = evictions
        batch.prefix_prefill_tokens = (len({req.adapter_dir for req in selected}
                                           - (cached - set(evictions))) * self.prefix_len)
        batch.prefill_token_num = batch.input_tokens() + batch.prefix_prefill_tokens
        return batch
