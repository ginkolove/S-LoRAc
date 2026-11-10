import numpy as np

from .req_queue import ReqQueue
from slora.common.prefix_modes import FULL_REPREFILL, LAYER_RETAINED, normalize_prefix_mode


class LowRAFIFOReqQueue(ReqQueue):
    """S-LoRA FIFO policy with representation-aware memory accounting only."""

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 prefix_len, prefix_mode=FULL_REPREFILL, layer_num=1, workspace_lora_limit=None,
                 retain_layers=0, retain_top_k=None, pinned_lora_dirs=()):
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        if layer_num <= 0:
            raise ValueError("layer_num must be positive")
        self.prefix_len = prefix_len
        self.prefix_mode = normalize_prefix_mode(prefix_mode)
        self.layer_num = layer_num
        self.workspace_lora_limit = workspace_lora_limit
        if not 0 <= retain_layers <= layer_num:
            raise ValueError("retain_layers must be between zero and layer_num")
        if retain_layers and self.prefix_mode != LAYER_RETAINED:
            raise ValueError("retained layers require minikv-layer_retained")
        self.retain_layers = retain_layers
        if retain_top_k is not None and (retain_top_k < 0 or self.prefix_mode != LAYER_RETAINED):
            raise ValueError("retain_top_k requires minikv-layer_retained and a nonnegative count")
        self.retain_top_k = retain_top_k
        self.pinned_lora_dirs = frozenset(pinned_lora_dirs)
        if self.pinned_lora_dirs and (retain_top_k is not None or not retain_layers or self.prefix_mode != LAYER_RETAINED):
            raise ValueError("pinned retention requires retained mode/layers and no dynamic top-K")
        self.prefix_cache_state = {}
        self._planned_evictions = []

    def update_prefix_cache_state(self, state):
        self.prefix_cache_state = state or {}

    def _init_cache_list(self, current_batch, lora_ranks):
        super()._init_cache_list(current_batch, lora_ranks)
        self.prefixes = {req.adapter_dir for req in current_batch.reqs
                         if req.adapter_dir is not None} if current_batch is not None else set()
        self._planned_evictions = []

    def _can_add_new_req(self, req, lora_ranks):
        super()._can_add_new_req(req, lora_ranks)
        if req.adapter_dir is not None:
            self.prefixes.add(req.adapter_dir)
        lengths = np.array(self.cache_len_list)
        private_peak = (lengths[:, 1] * np.arange(1, len(lengths) + 1)
                        + np.cumsum(lengths[:, 0])).max()
        prefix_tokens = len(self.prefixes) * self.prefix_len
        if self.prefix_mode != FULL_REPREFILL:
            if self.workspace_lora_limit is not None and len(self.prefixes) > self.workspace_lora_limit:
                return False
            prefix_tokens = (prefix_tokens + self.layer_num - 1) // self.layer_num
            if self.prefix_mode == LAYER_RETAINED:
                # Persistent packing rounds separately for each unique adapter.
                retained_slots = (self.prefix_len * self.retain_layers + self.layer_num - 1) // self.layer_num
                hot_count = (len(self.prefixes) if self.retain_top_k is None
                             else min(len(self.prefixes), self.retain_top_k))
                active_hot_count = hot_count
                if self.pinned_lora_dirs:
                    cached = set(self.prefix_cache_state.get("cached_lora_dirs", ()))
                    hot_count = len(self.pinned_lora_dirs & (cached | self.prefixes))
                    active_hot_count = len(self.pinned_lora_dirs & self.prefixes)
                scratch_count = (len(self.prefixes) if self.retain_layers < self.layer_num
                                 else len(self.prefixes) - active_hot_count)
                prefix_tokens = (scratch_count * self.prefix_len + self.layer_num - 1) // self.layer_num
                prefix_tokens += hot_count * retained_slots
        required = private_peak + self.adapter_size + prefix_tokens
        if self.prefix_mode == LAYER_RETAINED:
            state = self.prefix_cache_state
            # Active prefixes are already budgeted above; inactive pinned prefixes
            # are included in hot_count. Keep other idle entries only if they fit
            # alongside the entire decode reservation, not just this prefill.
            extra = set(state.get("cached_lora_dirs", ())) - self.prefixes - self.pinned_lora_dirs
            sizes = state.get("prefix_sizes", {})
            required += sum(sizes.get(d, retained_slots) for d in extra)
            inactive = set(state.get("inactive_lora_dirs", ()))
            last_used = state.get("last_used", {})
            victims = sorted(extra & inactive, key=lambda d: (last_used.get(d, 0), d))
            self._planned_evictions = []
            for directory in victims:
                if required < self.max_total_tokens:
                    break
                required -= sizes.get(directory, retained_slots)
                self._planned_evictions.append(directory)
        return (required < self.max_total_tokens
                and len(lengths) <= self.running_max_req_size)

    def generate_new_batch(self, current_batch, lora_ranks):
        batch = super().generate_new_batch(current_batch, lora_ranks)
        if batch is None or self.prefix_mode not in (FULL_REPREFILL, LAYER_RETAINED):
            return batch
        # ReqQueue also examines the first rejected request; exclude it from the eviction plan.
        self._init_cache_list(current_batch, lora_ranks)
        for request in batch.reqs:
            self._can_add_new_req(request, lora_ranks)
        if self.prefix_mode == LAYER_RETAINED:
            batch.lowra_evict_prefix_lora_dirs = list(self._planned_evictions)
            return batch
        needed = self.prefixes
        inactive = sorted(self.prefix_cache_state.get("inactive_lora_dirs", set()) - needed)
        lengths = np.array(self.cache_len_list)
        private_peak = (lengths[:, 1] * np.arange(1, len(lengths) + 1)
                        + np.cumsum(lengths[:, 0])).max()
        free_for_inactive = self.max_total_tokens - private_peak - self.adapter_size - len(needed) * self.prefix_len
        keep_count = max(0, int(free_for_inactive) // self.prefix_len)
        batch.lowra_evict_prefix_lora_dirs = inactive[:max(0, len(inactive) - keep_count)]
        return batch
