"""Capacity estimates over complete, continuously backlogged service cycles."""

import math
from pathlib import Path


def decode_cache_entries():
    import os
    root = Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache")))
    return {str(path) for name in ("_prefix_decode_scores", "_prefix_decode_reduce")
            for path in root.glob("*/%s.json" % name)}


def summarize_cycles(cycles, requests, origin, warmup, output_len, skip_cycles=2,
                     min_cycles=6, min_seconds=300.0, tolerance=0.1, new_compiles=0):
    by_id = {req.req_id: req for req in requests}
    actual_end = max((req.end_time for req in requests), default=0.0)
    point = sorted((cycle for cycle in cycles if cycle.get("event") == "service_cycle"
                    and origin <= cycle["start"] <= origin + actual_end), key=lambda c: c["start"])
    runs, current, rejected = [], [], []
    for index, cycle in enumerate(point):
        ids = cycle.get("completed_request_ids", [])
        matched = [by_id.get(request_id) for request_id in ids]
        reason = None
        if index < skip_cycles or cycle["start"] - origin < warmup:
            reason = "warmup"
        elif not cycle.get("continuously_backlogged"):
            reason = "queue_not_continuously_backlogged"
        elif (not cycle.get("valid") or not ids or len(set(ids)) != len(ids)
              or len(ids) != cycle.get("completed_requests")
              or cycle.get("generated_tokens") != len(ids) * output_len
              or not math.isfinite(cycle["end"] - cycle["start"])
              or cycle["end"] <= cycle["start"]
              or any(req is None or not req.success or req.output_tokens != output_len for req in matched)):
            reason = "incomplete_or_failed_cycle"
        if reason:
            rejected.append(dict(cycle_id=cycle["cycle_id"], reason=reason))
            if current:
                runs.append(current)
                current = []
        else:
            current.append(cycle)
    if current:
        runs.append(current)
    # Select by duration, never by the highest throughput or a favorable completion burst.
    chosen = max(runs, key=lambda run: run[-1]["end"] - run[0]["start"], default=[])
    base = dict(scope="longest_contiguous_run_of_complete_continuously_backlogged_cycles",
                selection="duration_not_throughput", total_cycles=len(point),
                excluded_cycles=rejected, cycles=len(chosen), warmup_cycles=skip_cycles,
                min_cycles=min_cycles, min_seconds=min_seconds, stability_tolerance=tolerance,
                new_prefix_decode_compiles=new_compiles, qualified=False,
                saturated_completed_req_per_s=None, peak_completed_req_per_s=None)
    if not chosen:
        return dict(base, reasons=["no_complete_backlogged_cycles"])
    ids = {request_id for cycle in chosen for request_id in cycle["completed_request_ids"]}
    start = chosen[0]["start"] - origin
    end = max(chosen[-1]["end"] - origin, max(by_id[request_id].end_time for request_id in ids))
    duration = end - start
    completed = [req for req in requests if req.success and start <= req.end_time <= end]
    count = len(completed)
    blocks = []
    for block in range(3):
        part = chosen[len(chosen) * block // 3:len(chosen) * (block + 1) // 3]
        if not part:
            continue
        seconds = part[-1]["end"] - part[0]["start"]
        number = sum(cycle["completed_requests"] for cycle in part)
        blocks.append(dict(start=part[0]["start"] - origin, end=part[-1]["end"] - origin,
                           cycles=len(part), completed_requests=number, seconds=seconds,
                           req_per_s=number / seconds))
    rates = [block["req_per_s"] for block in blocks]
    rate = count / duration
    spread = (max(rates) - min(rates)) / rate if rate > 0 and rates else None
    reasons = []
    if len(chosen) < min_cycles:
        reasons.append("too_few_cycles")
    if duration < min_seconds:
        reasons.append("measurement_too_short")
    if len(blocks) < 3 or spread is None or spread > tolerance:
        reasons.append("block_throughput_not_stable")
    if set(req.req_id for req in completed) != ids:
        reasons.append("client_completion_boundary_mismatch")
    if not requests or any(not req.success for req in requests):
        reasons.append("failed_requests_in_point")
    if new_compiles:
        reasons.append("prefix_decode_compilation_inside_measurement")
    return dict(base, qualified=not reasons, reasons=reasons, start=start, end=end,
                seconds=duration, completed_requests=count, cycle_completed_requests=len(ids),
                cycle_ids=[cycle["cycle_id"] for cycle in chosen],
                saturated_completed_req_per_s=rate,
                peak_completed_req_per_s=rate if not reasons else None,
                output_tokens_per_s=count * output_len / duration,
                blocks=blocks, block_relative_spread=spread,
                note="Observed backlogged capacity of this policy/configuration, not an oracle optimum.")
