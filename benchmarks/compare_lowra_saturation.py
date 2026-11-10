"""Compare paired saturation points after both managed runs have finished."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slora.common.prefix_modes import FULL_PREFIX_CACHE_MODES, FULL_RECONSTRUCT, LAYER_SERIAL, normalize_prefix_mode


def compare(full_dir, serial_dir):
    manifests = [json.loads((directory / "manifest.json").read_text())["config"]
                 for directory in (full_dir, serial_dir)]
    for field in ("model_dir", "adapter_base", "num_adapters", "prefix_len", "query_len",
                  "output_len", "num_token", "batch_max_tokens", "running_max_req_size",
                  "model_seed", "dummy", "request_rates", "seeds", "duration", "warmup", "steady_end",
                  "alpha", "cv", "prompt_token_id", "stats_interval", "preflight_requests"):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError("unpaired experiment configuration: %s" % field)
    if manifests[0].get("workload_sha256") != manifests[1].get("workload_sha256"):
        raise ValueError("unpaired frozen workload")
    # Old manifests predate the online scheduler and used ten decode steps
    # between admissions. Never silently compare them with the new defaults.
    def scheduling_config(config):
        online = config.get("scheduler", "slora") == "online"
        return (config.get("scheduler", "slora"),
                config.get("schedule_interval", 1) if online else 10,
                config.get("max_active_loras") if online else None,
                config.get("generation_mode", "trace"),
                config.get("generation_max_new_tokens", 512)
                if config.get("generation_mode") == "eos" else None)
    if scheduling_config(manifests[0]) != scheduling_config(manifests[1]):
        raise ValueError("unpaired scheduler or generation semantics")
    modes = [normalize_prefix_mode(manifest["mode"]) for manifest in manifests]
    if modes[0] not in FULL_PREFIX_CACHE_MODES or modes[1] != LAYER_SERIAL:
        raise ValueError("expected --full-dir a full-prefix cache mode and --serial-dir minikv-layer_serial")
    rows = []
    for rate in manifests[0]["request_rates"]:
        for seed in manifests[0]["seeds"]:
            filename = "rate%s_seed%s.json" % (format(rate, "g"), seed)
            full = json.loads((full_dir / filename).read_text())
            serial = json.loads((serial_dir / filename).read_text())
            if full["trace_sha256"] != serial["trace_sha256"]:
                raise ValueError("different request traces: %s" % filename)
            if not full["result"]["valid"] or not serial["result"]["valid"]:
                raise ValueError("invalid benchmark point: %s" % filename)
            a, b = full["result"]["completed_req_per_s_steady"], serial["result"]["completed_req_per_s_steady"]
            rows.append(dict(request_rate=rate, seed=seed, full_req_per_s=a, serial_req_per_s=b,
                             full_mode=modes[0], serial_mode=modes[1],
                             serial_over_full=b / a if a else None,
                             full_batch_mean=full["runtime"]["running_batch_size"]["mean"],
                             serial_batch_mean=serial["runtime"]["running_batch_size"]["mean"],
                             both_backlogged=full["runtime"]["backlogged_window"]
                             and serial["runtime"]["backlogged_window"],
                             full_serving_peak_gib=full["runtime"]["peak_allocated_without_base_model_bytes"] / 2 ** 30,
                             serial_serving_peak_gib=serial["runtime"]["peak_allocated_without_base_model_bytes"] / 2 ** 30))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--serial-dir", type=Path, required=True)
    parser.add_argument("--reconstruct-dir", type=Path,
                        help="Optional minikv-full_reconstruct run for a three-way comparison.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = compare(args.full_dir, args.serial_dir)
    if args.reconstruct_dir:
        mode = json.loads((args.reconstruct_dir / "manifest.json").read_text())["config"]["mode"]
        if normalize_prefix_mode(mode) != FULL_RECONSTRUCT:
            raise ValueError("--reconstruct-dir must contain a minikv-full_reconstruct run")
        reconstructed = compare(args.reconstruct_dir, args.serial_dir)
        for row, reconstructed_row in zip(rows, reconstructed):
            row.update(reconstruct_req_per_s=reconstructed_row["full_req_per_s"],
                       reconstruct_mode=FULL_RECONSTRUCT,
                       serial_over_reconstruct=reconstructed_row["serial_over_full"],
                       reconstruct_batch_mean=reconstructed_row["full_batch_mean"],
                       reconstruct_serving_peak_gib=reconstructed_row["full_serving_peak_gib"])
    print("rate seed     full req/s   serial req/s   ratio   full batch   serial batch   both backlog")
    for row in rows:
        ratio = "%.3f" % row["serial_over_full"] if row["serial_over_full"] is not None else "n/a"
        print("%4g %4s %14.6f %14.6f %7s %12.3f %14.3f %s" % (
            row["request_rate"], row["seed"], row["full_req_per_s"], row["serial_req_per_s"], ratio,
            row["full_batch_mean"], row["serial_batch_mean"], row["both_backlogged"]))
        if args.reconstruct_dir:
            print("          reconstruct req/s=%.6f batch=%.3f serial/reconstruct=%s" % (
                row["reconstruct_req_per_s"], row["reconstruct_batch_mean"],
                "%.3f" % row["serial_over_reconstruct"] if row["serial_over_reconstruct"] is not None else "n/a"))
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
