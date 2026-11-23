"""Collect a single frozen trace replayed by the five prefix-source modes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slora.common.prefix_modes import CPU_FIXED, CPU_FIXED_R64, LAYER_SERIAL, SOURCE_COMPARISON_MODES


def compare(root, modes=SOURCE_COMPARISON_MODES):
    points = {}
    digest = None
    for mode in modes:
        paths = list((root / mode).glob("rate*_seed[0-9]*.json"))
        paths = [p for p in paths if p.stem.split("_seed")[-1].isdigit()]
        if len(paths) != 1:
            raise ValueError("expected one completed rate/seed result for " + mode)
        point = json.loads(paths[0].read_text())
        result = point["result"]
        if point["mode"] != mode or not result["valid"] or not point["trace_sha256"]:
            raise ValueError("invalid point: " + mode)
        if digest is not None and point["trace_sha256"] != digest:
            raise ValueError("trace differs: " + mode)
        digest = point["trace_sha256"]
        cache = point["runtime"]["prefix_cache_stats"]
        if mode != LAYER_SERIAL and not (cache or {}).get("reuse_next_batch"):
            raise ValueError("expected next-batch prefix reuse, not legacy cache lifetime: " + mode)
        points[mode] = {k: result[k] for k in (
            "total_requests", "completed_requests", "benchmark_time", "completed_req_per_s_total",
            "completed_req_per_s_steady", "output_tokens_per_s_total", "latency_mean", "latency_p95",
            "ttft_mean", "ttft_p95")}
        points[mode]["cache"] = cache
        points[mode]["peak_output_window_10s"] = result.get("peak_output_window_10s")
        points[mode]["peak_without_model_bytes"] = point["runtime"]["peak_allocated_without_base_model_bytes"]
    return dict(trace_sha256=digest, methods=points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fixed-mode", choices=(CPU_FIXED, CPU_FIXED_R64), default=CPU_FIXED)
    args = parser.parse_args()
    root = args.root
    modes = [args.fixed_mode if m == CPU_FIXED else m for m in SOURCE_COMPARISON_MODES]
    output = root / "comparison.json"
    output.write_text(json.dumps(compare(root, modes), indent=2, allow_nan=False) + "\n")
    print(output)
