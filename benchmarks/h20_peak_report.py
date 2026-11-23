"""Show whole-GPU high-water marks for one six-card 13B/10K formal run."""

import argparse
import json
from pathlib import Path


def attempt_peak(attempt):
    report = attempt / "memory.json"
    if report.is_file():
        memory = json.loads(report.read_text())
        if memory["cuda_oom"]:
            status = "OOM"
        elif memory["error"] or memory["monitor_error"] or memory["returncode"] or memory["over_limit"]:
            status = "incomplete"
        else:
            status = "awaiting validation"
        return memory["sampled_peak_bytes"], status

    peak = None
    samples = attempt / "device_memory.csv"
    if samples.is_file():
        with samples.open() as stream:
            for line in stream:
                # Ignore the header and a partial last row from the live writer.
                if line.endswith("\n"):
                    value = line.rstrip().partition(",")[2]
                    if value.isdecimal():
                        used = int(value)
                        peak = used if peak is None else max(peak, used)
    return peak, "sampled so far" if peak is not None else "no samples yet"


def method_peak(point, method):
    directory = point / method
    result = directory / "result.json"
    if result.is_file():
        data = json.loads(result.read_text())
        return data["pool_tokens"], data["experiment_sampled_peak_bytes"], f"done ({data['completed_requests']} req)"

    config = point / "experiment.json"
    slots = json.loads(config.read_text()).get("fixed_pool_tokens") if config.is_file() else None
    peak, status = None, "no samples yet"
    for attempt in sorted(directory.glob("formal_*")):
        observed, status = attempt_peak(attempt)
        if observed is not None:
            peak = observed if peak is None else max(peak, observed)
    if (directory / "failure.json").is_file() and status != "OOM":
        status = "incomplete"
    return slots, peak, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"Run directory does not exist: {args.run_dir}")
    print(f"{'GPU':>3}  {'method':<24} {'slots':>6} {'peak GB':>8} {'peak GiB':>9}  status")
    for log in sorted(args.run_dir.glob("gpu*_*.log")):
        gpu, method = log.stem.split("_", 1)
        slots, peak, status = method_peak(args.run_dir / gpu / "13b_p10000", method)
        gb = "-" if peak is None else f"{peak / 10**9:.2f}"
        gib = "-" if peak is None else f"{peak / 2**30:.2f}"
        print(f"{gpu[3:]:>3}  {method:<24} {slots or '-':>6} {gb:>8} {gib:>9}  {status}")


if __name__ == "__main__":
    main()
