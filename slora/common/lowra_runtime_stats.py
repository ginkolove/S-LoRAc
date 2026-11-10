"""Low-frequency serving snapshots without CUDA events or synchronization."""

import torch


def storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def base_model_memory_bytes(model):
    weights = [tensor for layer in [model.pre_post_weight, *model.trans_layers_weight]
               for tensor in vars(layer).values()
               if isinstance(tensor, torch.Tensor) and tensor.is_cuda]
    buffers = [tensor for tensor in vars(model).values()
               if isinstance(tensor, torch.Tensor) and tensor.is_cuda]
    weight_bytes = storage_bytes(weights)
    resident_bytes = storage_bytes(weights + buffers)
    return dict(base_model_weight_bytes=weight_bytes,
                base_model_buffer_bytes=resident_bytes - weight_bytes,
                base_model_resident_bytes=resident_bytes)


def snapshot(server):
    model = server.model
    pool = model.mem_manager
    if not hasattr(server, "_lowra_model_memory"):
        server._lowra_model_memory = base_model_memory_bytes(model)
    entries = server.lowra_prefix_cache.entries if server.lowra_prefix_cache else {}
    prefix_slots = sum(getattr(entry, "allocated_slots", entry.prefix_len) for entry in entries.values())
    resident = [entry for entry in entries.values()
                if getattr(entry, "allocated_slots", entry.prefix_len) > 0]
    adapter_slots = sum(adapter.r * 4 for adapter in server.adapters
                        if adapter is not None
                        and adapter.lora_dir in server.infer_adapter.adapter_dirs)
    used_slots = int(pool.tot_size - pool.can_use_mem_size - pool.workspace_token_equiv)
    slot_bytes = 2 * pool.layer_num * pool.cell_size * torch.empty(
        (), dtype=pool.dtype).element_size()
    peak = int(torch.cuda.max_memory_allocated())
    cache_stats = (server.lowra_prefix_cache.access_stats()
                   if server.lowra_prefix_cache else None)
    runtime = getattr(server, "lowra_runtime", None)
    return dict(
        reconstruction_kernel=getattr(runtime, "last_reconstruction_kernel", None),
        reconstruction_workers=getattr(runtime, "last_reconstruction_workers", None),
        persistent_total_slots=int(pool.tot_size - pool.workspace_token_equiv),
        persistent_free_slots=int(pool.can_use_mem_size),
        persistent_used_slots=used_slots,
        prefix_slots=prefix_slots,
        adapter_slots=adapter_slots,
        private_kv_slots=used_slots - prefix_slots - adapter_slots,
        prefix_resident_loras=len(resident),
        prefix_active_loras=sum(entry.ref_count > 0 for entry in resident),
        prefix_idle_loras=sum(entry.ref_count == 0 for entry in resident),
        prefix_idle_slots=sum(getattr(entry, "allocated_slots", entry.prefix_len)
                              for entry in resident if entry.ref_count == 0),
        prefix_cache_stats=cache_stats,
        workspace_capacity_tokens=pool.layer_workspace_tokens,
        workspace_token_equivalent=pool.workspace_token_equiv,
        workspace_backing_bytes=pool.workspace_token_equiv * slot_bytes,
        workspace_peak_tokens=pool.peak_workspace_tokens,
        workspace_peak_token_equivalent=pool.peak_workspace_token_equiv,
        workspace_peak_backing_bytes=pool.peak_workspace_token_equiv * slot_bytes,
        unified_pool_backing_bytes=pool.get_memory_size(),
        slot_bytes=slot_bytes,
        allocated_bytes=int(torch.cuda.memory_allocated()),
        reserved_bytes=int(torch.cuda.memory_reserved()),
        peak_allocated_bytes=peak,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        peak_allocated_without_base_model_bytes=peak - server._lowra_model_memory["base_model_resident_bytes"],
        **server._lowra_model_memory,
    )
