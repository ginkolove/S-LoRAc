"""Bounded TP=1 memory probes for the guarded H20 benchmark worker."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from slora.common.prefix_modes import CPU_PIN_COUNTS


def fit_lengths(minima, maxima, budget):
    """Spend a token budget on the longest histories first, preserving batch width."""
    remaining = budget - sum(minima)
    if remaining < 0 or len(minima) != len(maxima):
        raise ValueError("invalid startup probe token budget")
    lengths = list(minima)
    for i, (low, high) in enumerate(zip(minima, maxima)):
        if low > high:
            raise ValueError("invalid startup probe length bounds")
        extra = min(high - low, remaining)
        lengths[i] += extra
        remaining -= extra
    return lengths


def probe_shape(queue, ranks, directories, queries, histories):
    """Use the staged scheduler's own capacity calculation, including scratch and prefix builds."""
    rows = [SimpleNamespace(adapter_dir=d, input_len=1, output_ids=()) for d in directories]
    lo, hi = 0, min(len(rows), queue.running_max_req_size)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if queue.plan([], rows[:mid], ranks).fits:
            lo = mid
        else:
            hi = mid - 1
    if not lo:
        raise ValueError("Startup probe cannot admit one request with the configured scheduling limits")
    base = queue.plan([], rows[:lo], ranks)
    private = lo + queue.max_total_tokens - base.required_slots
    prefill = lo + queue.batch_max_tokens - base.prefill_tokens
    prompts = fit_lengths([1] * lo, queries[:lo], min(private, prefill))
    contexts = fit_lengths(prompts, histories[:lo], private)
    return directories[:lo], prompts, contexts


def extend_decode_history(batch, prompts, contexts):
    """Populate valid private KV slots for a deep decode without timing thousands of decode steps."""
    extra = sum(contexts) - sum(prompts)
    if not extra:
        return
    pool = batch.mem_manager
    slots = pool.alloc(extra)
    for keys, values in zip(pool.key_buffer, pool.value_buffer):
        keys.index_fill_(0, slots, 0)
        values.index_fill_(0, slots, 0)
    # The output of prefill is the pending input token. Past KV is right-aligned
    # immediately before it, exactly as in InferBatch.merge/filter.
    old_end, end = max(prompts), max(contexts)
    offset = 0
    for i, (prompt, history) in enumerate(zip(prompts, contexts)):
        start = end - history
        old = batch.nopad_b_loc[i, old_end - prompt:old_end]
        if start != old_end - prompt:
            batch.nopad_b_loc[i, start:start + prompt].copy_(old.clone())
        count = history - prompt
        batch.nopad_b_loc[i, start + prompt:end] = slots[offset:offset + count]
        offset += count
    lengths = batch.nopad_b_seq_len.new_tensor([n + 1 for n in contexts])
    batch.nopad_b_seq_len = lengths
    batch.nopad_b_start_loc = lengths.cumsum(0, dtype=lengths.dtype) - lengths
    batch.nopad_total_token_num = sum(contexts) + len(contexts)
    batch.nopad_max_len_in_batch = end + 1
    batch.input_lengths = [n + 1 for n in contexts]


@torch.no_grad()
def run_startup_profile(router, spec_path):
    if router.world_size != 1 or router.input_params.scheduler != "online":
        raise ValueError("H20 slot profiling requires the TP=1 online worker")
    spec = json.loads(Path(spec_path).read_text())
    output = Path(os.environ["SLORA_BENCHMARK_MPS_REPORT"]).with_name("forward_profile.json")
    worker = router.model_rpcs[0].model
    pool = worker.model.mem_manager
    queue, ranks = router.req_queue, router.lora_ranks
    params = router.input_params
    quota = int(os.environ["SLORA_BENCHMARK_MPS_BYTES"])
    phases, shapes = {}, {}

    def snapshot(name):
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        phases[name] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            client_free_bytes=free, other_client_bytes=max(0, quota - free - reserved))
        torch.cuda.reset_peak_memory_stats()

    snapshot("model_startup")
    # Interleave ranks so even a small feasible cold batch exercises different kernels.
    indices = spec["adapter_indices"]
    by_rank = {}
    for i in indices:
        d = router.adapter_dirs[i]
        by_rank.setdefault(ranks[d], []).append(d)
    ordered = [ds[i] for i in range(max(map(len, by_rank.values())))
               for _, ds in sorted(by_rank.items(), reverse=True) if i < len(ds)]
    queries, histories = spec["query_lengths"], spec["history_lengths"]
    prefix = [params.cpu_prefix_token_id] * params.lowra_shared_prefix_len

    def run_batch(name, directories, prompts, contexts, retained=0):
        unique = list(dict.fromkeys(directories))
        shapes[name] = dict(requests=len(prompts), unique_loras=len(unique),
                            ranks=[ranks[d] for d in unique], prefill_private_tokens=sum(prompts),
                            decode_private_tokens=sum(contexts) + len(prompts),
                            longest_private_history=max(contexts), retained_layers=retained)
        print(f"Startup memory probe {name}: {len(prompts)} requests, {len(unique)} LoRAs, "
              f"{sum(prompts)} prefill tokens, {sum(contexts) + len(prompts)} private decode tokens", flush=True)
        worker.exposed_load_adapters(unique)
        if retained:
            worker.exposed_apply_space_plan({unique[0]: retained}, [])
        requests = [dict(request_id=i, adapter_dir=d, input_id=[1] * n, prefix_input_ids=prefix,
                         sampling_param={}) for i, (d, n) in enumerate(zip(directories, prompts))]
        worker.exposed_add_batch(name, requests, "fp16")
        snapshot(name + "_prefix")
        worker.exposed_prefill_batch(name)
        snapshot(name + "_prefill")
        extend_decode_history(worker.cache[name], prompts, contexts)
        snapshot(name + "_history_setup")
        worker.exposed_decode_batch(name)
        snapshot(name + "_decode")
        worker.exposed_remove_batch(name)
        worker.exposed_reset_lowra_runtime()

    def shape_for(name, directories, prefill_limits):
        if params.lowra_prefix_mode in CPU_PIN_COUNTS:
            # A previously used fixed prefix stays resident during a subsequent cold batch.
            hot = [router.adapter_dirs[i] for i in params.cpu_fixed_hot_indices]
            worker.exposed_load_adapters(hot)
            requests = [dict(request_id=i, adapter_dir=d, input_id=[1], prefix_input_ids=prefix,
                             sampling_param={}) for i, d in enumerate(hot)]
            worker.exposed_add_batch("warm_fixed", requests, "fp16")
            worker.exposed_prefill_batch("warm_fixed")
            worker.exposed_remove_batch("warm_fixed")
            worker.exposed_offload_adapters([])
            snapshot(name + "_warm_fixed")
        queue.update_prefix_cache_state(worker.exposed_lowra_prefix_cache_state())
        return probe_shape(queue, ranks, directories, prefill_limits, histories)

    cold = shape_for("cold_loras", ordered, queries)
    run_batch("cold_loras", *cold)
    # Resumed requests can prefill their entire private histories together. Bound a
    # wide recompute by both batch_max_tokens and the pool's one-step capacity.
    wide = shape_for("wide_batch", [ordered[0]] * len(queries), histories)
    run_batch("wide_batch", *wide)
    if 1 < len(cold[0]) < len(wide[0]):
        mixed = [cold[0][i % len(cold[0])] for i in range(len(queries))]
        run_batch("wide_mixed", *shape_for("wide_mixed", mixed, histories))
    if hasattr(worker.lowra_prefix_cache, "apply_plan"):
        # Exercise promotion and a partially retained decode as well as the all-streaming layout.
        d = ordered[0]
        layer_num, prefix_len = pool.layer_num, len(prefix)
        available = pool.tot_size - 4 * ranks[d] - (prefix_len + layer_num - 1) // layer_num - 2
        retained = min(layer_num // 2, available * layer_num // prefix_len)
        if retained > 0:
            run_batch("partial_retention", [d], [1], [1], retained)
    torch.cuda.synchronize()
    static = params.lowra_static_token_num
    output.write_text(json.dumps(dict(mode=params.lowra_prefix_mode, trace_sha256=spec["trace_sha256"],
                                      pool_tokens=pool.tot_size + static, dynamic_pool_slots=pool.tot_size,
                                      static_token_equivalents=static, slot_bytes=pool.get_memory_size() // pool.tot_size,
                                      phases=phases, shapes=shapes), indent=2) + "\n")
