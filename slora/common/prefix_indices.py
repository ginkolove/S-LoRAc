"""Unique-prefix address tables and a small request-to-prefix indirection."""

import torch


class PrefixIndexCache:
    def __init__(self):
        self._requests = None
        self._mapping = None
        self._tables = {}

    def request_mapping(self, adapter_dirs, device):
        key = (tuple(adapter_dirs), torch.device(device))
        if key != self._requests:
            unique = list(dict.fromkeys(adapter_dirs))
            ids = {directory: index for index, directory in enumerate(unique)}
            request_ids = torch.tensor([ids[d] for d in adapter_dirs], device=device, dtype=torch.long)
            self._requests = key
            self._mapping = unique, request_ids
        return self._mapping

    def stack(self, role, tensors, dim=0):
        sources = tuple(tensors)
        key = (dim, tuple((t.data_ptr(), tuple(t.shape), t.stride(), t._version) for t in sources))
        cached = self._tables.get(role)
        if cached is None or cached[0] != key:
            # Hold source tensors too, so allocator pointer reuse cannot look like a hit.
            cached = (key, torch.stack(sources, dim=dim), sources)
            self._tables[role] = cached
        return cached[1]


class PrefixIndexedFrame:
    """Serving consumes unique tables; expanded views are diagnostic-only."""

    @property
    def prefix_b_loc(self):
        return self.prefix_locations(0)

    def prefix_locations(self, layer_id):
        return self.prefix_index_table(layer_id)[self.prefix_request_ids]
