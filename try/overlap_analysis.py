"""CPU-only statistics for the reconstruction/FFN overlap experiment."""

import statistics


def summarize(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty samples")
    pos = (len(ordered) - 1) * 0.95
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return dict(mean=statistics.mean(ordered), median=statistics.median(ordered),
                p95=ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower),
                min=ordered[0], max=ordered[-1])


def compare(samples):
    stats = {mode: {key: summarize([sample[key] for sample in rows])
                    for key in rows[0]} for mode, rows in samples.items()}
    r = stats["reconstruct"].get("reconstruct_ms", stats["reconstruct"]["total_ms"])["median"]
    f = stats["ffn"].get("ffn_ms", stats["ffn"]["total_ms"])["median"]
    result = {}
    for order in ("rf", "fr"):
        serial = stats["serial_" + order]["total_ms"]["median"]
        parallel = stats["overlap_" + order]
        total = parallel["total_ms"]["median"]
        result[order] = dict(
            serial_over_parallel_speedup=serial / total,
            reconstruct_event_span_slowdown=parallel["reconstruct_ms"]["median"] / r if "reconstruct_ms" in parallel else None,
            ffn_event_span_slowdown=parallel["ffn_ms"]["median"] / f if "ffn_ms" in parallel else None,
            ideal_no_contention_ms=max(r, f),
            ideal_max_speedup=(r + f) / max(r, f),
            parallel_over_ideal=total / max(r, f),
            net_saved_ms=serial - total,
            net_saved_fraction_of_shorter_task=(serial - total) / min(r, f),
        )
    return dict(timings_ms=stats, comparisons=result)


def compare_traces(traces):
    if not all(trace["valid"] for trace in traces.values()):
        return dict(valid=False, reason="missing GPU kernels; inspect profiler/CUPTI support")
    r = traces["reconstruct"]["reconstruct_kernel_union_ms"]
    f = traces["ffn"]["ffn_kernel_union_ms"]
    if r <= 0 or f <= 0:
        return dict(valid=False, reason="could not identify isolated reconstruction or FFN kernels")
    return dict(valid=True, comparisons={
        order: dict(reconstruct_kernel_slowdown=traces["overlap_" + order]["reconstruct_kernel_union_ms"] / r,
                    ffn_kernel_slowdown=traces["overlap_" + order]["ffn_kernel_union_ms"] / f,
                    actual_kernel_overlap_ms=traces["overlap_" + order]["actual_kernel_overlap_ms"])
        for order in ("rf", "fr")})


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection_duration(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze_trace(trace):
    kernels = [event for event in trace.get("traceEvents", [])
               if event.get("ph") == "X" and "kernel" in event.get("cat", "").lower()
               and event.get("dur", 0) > 0]
    reconstruction = [event for event in kernels if "lowra_reconstruct_kv_kernel" in event["name"]]
    # Torch 2.0 graph replay updates RNG offset tensors even for this deterministic graph.
    auxiliary = [event for event in kernels if "FillFunctor<long>" in event["name"]]
    compute = [event for event in kernels if "lowra_reconstruct_kv_kernel" not in event["name"]
               and "FillFunctor<long>" not in event["name"]]
    intervals = lambda events: [(e["ts"], e["ts"] + e["dur"]) for e in events]
    rec_intervals, ffn_intervals = intervals(reconstruction), intervals(compute)
    overlap = intersection_duration(rec_intervals, ffn_intervals)
    duration = lambda spans: sum(end - start for start, end in merge_intervals(spans))
    origin = min((e["ts"] for e in kernels), default=0)
    return dict(
        valid=bool(kernels), paired=bool(reconstruction and compute),
        scope="Isolated pair only; graph FillFunctor<long> bookkeeping is excluded from FFN. Profiler timing is diagnostic.",
        reconstruct_kernel_count=len(reconstruction), ffn_kernel_count=len(compute),
        auxiliary_kernel_count=len(auxiliary),
        auxiliary_kernel_union_ms=duration(intervals(auxiliary)) / 1000,
        reconstruct_kernel_union_ms=duration(rec_intervals) / 1000,
        ffn_kernel_union_ms=duration(ffn_intervals) / 1000,
        actual_kernel_overlap_ms=overlap / 1000,
        ffn_kernel_overlap_fraction=overlap / duration(ffn_intervals) if compute else None,
        timeline=[dict(name=e["name"], stream=e.get("args", {}).get("stream"),
                       role=("reconstruct" if "lowra_reconstruct_kv_kernel" in e["name"] else
                             "graph_bookkeeping" if "FillFunctor<long>" in e["name"] else "ffn"),
                       start_ms=(e["ts"] - origin) / 1000, duration_ms=e["dur"] / 1000)
                  for e in sorted(kernels, key=lambda e: e["ts"])],
    )


def analyze_compute_window_trace(trace):
    """Attribute GPU kernels to enclosing CPU launch ranges, not GPU event spans."""
    events = trace.get("traceEvents", [])
    ranges = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("window/")]
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") == "cuda_runtime" and "LaunchKernel" in e.get("name", "")}
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    stages, timeline = {}, []
    origin = min((e["ts"] for e in kernels), default=0)
    for kernel in kernels:
        launch = launches.get(kernel.get("args", {}).get("correlation"))
        enclosing = [e for e in ranges if launch is not None and e.get("tid") == launch.get("tid")
                     and e["ts"] <= launch["ts"] <= e["ts"] + e["dur"]]
        stage = min(enclosing, key=lambda e: e["dur"])["name"][7:] if enclosing else "unattributed"
        if "lowra_reconstruct" in kernel["name"]:
            stage = "reconstruct"
        interval = (kernel["ts"], kernel["ts"] + kernel["dur"])
        stages.setdefault(stage, []).append(interval)
        timeline.append(dict(name=kernel["name"], stage=stage, stream=kernel.get("args", {}).get("stream"),
                             start_ms=(kernel["ts"] - origin) / 1000, duration_ms=kernel["dur"] / 1000))
    rec = stages.get("reconstruct", [])
    return dict(valid=bool(ranges and kernels), scope="Instrumented GPU kernel intervals, not CPU ranges or occupancy counters.",
                kernel_ms={stage: sum(end - start for start, end in merge_intervals(spans)) / 1000
                           for stage, spans in stages.items()},
                overlap_with_reconstruction_ms={stage: intersection_duration(rec, spans) / 1000
                                                for stage, spans in stages.items() if stage != "reconstruct"},
                timeline=sorted(timeline, key=lambda e: e["start_ms"]))
