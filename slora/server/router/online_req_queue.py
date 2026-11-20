"""FCFS admission using known tokens only, with one-step KV headroom.

The router must check again before every decode and preempt/recompute when the
pool fills. Neither output caps nor replay answer lengths enter this planner.
"""

from collections import Counter
from dataclasses import dataclass
import uuid

from ..io_struct import Batch
from .req_queue import ReqQueue
from slora.common.prefix_modes import (
    FULL_REPREFILL, FULL_PREFIX_CACHE_MODES, PREFILL_BUILD_MODES, LAYER_RETAINED, normalize_prefix_mode,
)


@dataclass
class OnlinePlan:
    fits: bool
    required_slots: int
    evictions: list
    prefill_tokens: int
    prefix_build_tokens: int


class OnlineReqQueue(ReqQueue):
    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 prefix_len=0, prefix_mode=FULL_REPREFILL, layer_num=1,
                 retain_layers=0, retain_top_k=None, pinned_lora_dirs=(),
                 max_active_loras=None, workspace_lora_limit=None):
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.prefix_len = prefix_len
        self.prefix_mode = normalize_prefix_mode(prefix_mode)
        self.layer_num = layer_num
        self.retain_layers = retain_layers
        self.retain_top_k = retain_top_k
        self.pinned_lora_dirs = set(pinned_lora_dirs)
        self.max_active_loras = max_active_loras
        self.workspace_lora_limit = workspace_lora_limit
        self.prefix_cache_state = {}

    def update_prefix_cache_state(self, state):
        self.prefix_cache_state = state or {}

    @staticmethod
    def known_tokens(req):
        return req.input_len + len(req.output_ids)

    def plan(self, running, selected, lora_ranks):
        requests = list(running) + list(selected)
        counts = Counter(req.adapter_dir for req in requests if req.adapter_dir is not None)
        needed = set(counts)
        state = self.prefix_cache_state
        sizes = dict(state.get("prefix_sizes", {}))
        cached = set(state.get("cached_lora_dirs", ()))
        inactive = set(state.get("inactive_lora_dirs", ()))
        pinned = self.pinned_lora_dirs | set(state.get("pinned_lora_dirs", ()))
        private = sum(self.known_tokens(req) for req in requests) + len(selected)
        adapters = sum(4 * lora_ranks[directory] for directory in needed)
        query_tokens = sum(self.known_tokens(req) for req in selected)
        prefix_build_tokens = 0
        scratch = 0
        after = dict(sizes)
        before = dict(sizes)
        if self.prefix_len and self.prefix_mode in FULL_PREFIX_CACHE_MODES:
            for directory in cached | needed:
                before[directory] = after[directory] = self.prefix_len
            prefix_build_tokens = len({r.adapter_dir for r in selected} - cached) * self.prefix_len
        elif self.prefix_len and self.prefix_mode == LAYER_RETAINED:
            slots = (self.prefix_len * self.retain_layers + self.layer_num - 1) // self.layer_num
            if pinned:
                hot = needed & pinned
            elif self.retain_top_k is None:
                hot = needed
            else:
                hot = set(sorted(needed, key=lambda d: (
                    -counts[d], -bool(sizes.get(d, 0)), d))[:self.retain_top_k])
            for directory in needed:
                after[directory] = slots if directory in hot else 0
                # Fixed retention allocates during acquire; dynamic top-K rebalances
                # only after weights load, so also account for its old allocation.
                if self.retain_top_k is None and directory in hot:
                    before[directory] = slots
            scratch_count = len(needed) if self.retain_layers < self.layer_num else len(needed - hot)
            scratch = (scratch_count * self.prefix_len + self.layer_num - 1) // self.layer_num
        elif self.prefix_len:
            scratch = (len(needed) * self.prefix_len + self.layer_num - 1) // self.layer_num

        def required():
            return private + adapters + max(sum(before.values()), sum(after.values())) + scratch

        evictions = []
        last_used = state.get("last_used", {})
        for directory in sorted(inactive - needed - pinned, key=lambda d: (last_used.get(d, 0), d)):
            if not state.get("reuse_next_batch") and required() <= self.max_total_tokens:
                break
            before.pop(directory, None)
            after.pop(directory, None)
            evictions.append(directory)
        # Only reprefill computes missing prefixes as prompt tokens. MiniKV
        # reconstruction consumes memory/compute but not the prefill token budget.
        prefill_tokens = query_tokens + (prefix_build_tokens if self.prefix_mode in PREFILL_BUILD_MODES else 0)
        fits = (required() <= self.max_total_tokens
                and len(requests) <= self.running_max_req_size
                and prefill_tokens <= self.batch_max_tokens
                and (self.max_active_loras is None or len(needed) <= self.max_active_loras)
                and (not self.prefix_len or self.prefix_mode in FULL_PREFIX_CACHE_MODES
                     or self.workspace_lora_limit is None or len(needed) <= self.workspace_lora_limit))
        return OnlinePlan(fits, required(), evictions, prefill_tokens, prefix_build_tokens)

    def generate_new_batch(self, current_batch, lora_ranks):
        running = current_batch.reqs if current_batch else []
        selected, accepted = [], None
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            active = {r.adapter_dir for r in running + selected if r.adapter_dir is not None}
            if (self.max_active_loras is not None and req.adapter_dir is not None
                    and req.adapter_dir not in active and len(active) >= self.max_active_loras):
                continue
            candidate = self.plan(running, selected + [req], lora_ranks)
            if not candidate.fits:
                break
            selected.append(req)
            accepted = candidate
        if not selected:
            return None
        selected_ids = {req.request_id for req in selected}
        self.waiting_req_list = [req for req in self.waiting_req_list
                                 if not req.aborted and req.request_id not in selected_ids]
        batch = Batch(uuid.uuid4().hex, selected)
        batch.lowra_evict_prefix_lora_dirs = accepted.evictions
        batch.prefix_prefill_tokens = accepted.prefix_build_tokens
        batch.prefill_token_num = accepted.prefill_tokens
        batch.online_required_slots = accepted.required_slots
        return batch
