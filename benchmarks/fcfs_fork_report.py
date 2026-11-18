"""Summarize paired forks; fit on one trace and evaluate on a different trace."""

import argparse
import csv
import json
from pathlib import Path

from benchmarks.fcfs_fork_exp import read, write


def points(directory):
    journal = read(directory / "journal.json")
    rows = []
    for i, p in enumerate(journal["points"]):
        arms = [read(directory / ("%03d-%s.json" % (i, a))) for a in ("wait", "shrink")]
        if any(x["signature"] != p["signature"] or x["point"] != i or x["arm"] != a
               for a, x in zip(("wait", "shrink"), arms)):
            raise ValueError("unpaired arms at point %d" % i)
        a = p["action"]
        for h in ("5", "10", "20"):
            w, s = [x["windows"][h] for x in arms]
            if w["window_s"] != s["window_s"] or w["window_s"] <= 0:
                raise ValueError("arms have different observation windows")
            row = dict(point=i, horizon=int(h), b0=a["b0"], n=a["n"], rho=a["rho"],
                trace_time=p["trace_time"], phase="drain" if p["trace_time"] > journal["config"]["duration"] else "arrivals",
                group="new" if a["new_groups"] else "existing", new_groups=len(a["new_groups"]),
                newly_streamed_active=len(a["newly_streamed_active"]),
                lost_layers=sum(a["lost_layers"].values()), lost_ready=sum(a["lost_ready"].values()),
                query_tokens=a["query_tokens"], delta_tokens=s["tokens"] - w["tokens"],
                gain_tokens_s=s["tokens_s"] - w["tokens_s"], window_s=w["window_s"])
            for name, arm in (("wait", w), ("shrink", s)):
                row.update({name + "_" + k: v for k, v in arm.items() if k != "window_s"})
            rows.append(row)
    return journal, rows


def regret(rows, threshold):
    loss = mistakes = 0
    for row in rows:
        delta = row["delta_tokens"]
        shrink = row["rho"] >= threshold
        if (shrink and delta < 0) or (not shrink and delta > 0):
            mistakes += 1
            loss += abs(delta)
    return dict(points=len(rows), wrong_non_ties=mistakes, lost_tokens_vs_paired_best=loss)


def fit(rows):
    xs = sorted({r["rho"] for r in rows})
    if len(xs) < 2 or not any(r["delta_tokens"] > 0 for r in rows) or not any(
            r["delta_tokens"] < 0 for r in rows):
        return None
    candidates = [(x + y) / 2 for x, y in zip(xs, xs[1:])]
    theta = min(candidates, key=lambda t: regret(rows, t)["lost_tokens_vs_paired_best"])
    loss = regret(rows, theta)["lost_tokens_vs_paired_best"]
    constant = min(regret(rows, x)["lost_tokens_vs_paired_best"] for x in (0., float("inf")))
    return theta if loss < constant else None


def plot(path, rows):
    # Standalone SVG keeps the benchmark environment free of plotting dependencies.
    xs = max((r["rho"] for r in rows), default=1.) or 1.
    ys = max((abs(r["gain_tokens_s"]) for r in rows), default=1.) or 1.
    dots = []
    for r in rows:
        x, y = 70 + 600 * r["rho"] / xs, 165 - 125 * r["gain_tokens_s"] / ys
        color = "#136f63" if r["group"] == "existing" else "#ae451e"
        dots.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"><title>pair %d, rho=%.3f, delta=%.2f tok/s</title></circle>'
                    % (x, y, color, r["point"], r["rho"], r["gain_tokens_s"]))
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 730 345">'
        '<rect width="730" height="345" fill="white"/><g font-family="sans-serif" font-size="13">'
        '<text x="12" y="18">Shrink minus wait (output tokens/s), 10-step-time window</text>'
        '<path d="M70 35V300 M70 165H675" stroke="#666" fill="none"/>'
        '<text x="8" y="45">%.1f</text><text x="44" y="169">0</text><text x="5" y="292">-%.1f</text>'
        '<text x="67" y="317">0</text><text x="650" y="317">%.2f</text><text x="300" y="317">n / B0</text>'
        '<text x="370" y="20" fill="#136f63">existing</text><text x="455" y="20" fill="#ae451e">new group</text>'
        '%s</g></svg>' % (ys, ys, xs, "".join(dots)))


def report(directory, train=None):
    journal, rows = points(directory)
    result = dict(pairs=len(journal["points"]), blocked=journal["blocked"], eligible=journal["eligible"], curves={})
    if rows:
        with (directory / "pairs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot(directory / "gain.svg", [r for r in rows if r["horizon"] == 10])
    calibration = rows
    if train:
        source, calibration = points(train)
        if (source["trace_sha256"] == journal["trace_sha256"] or
                source["config"]["data_seed"] == journal["config"]["data_seed"]):
            raise ValueError("validation requires a separate trace and LMSYS sample order")
        for key in ("tokenizer", "prefix_len", "num_adapters", "request_rate", "maf_skewness", "duration"):
            if source["config"][key] != journal["config"][key]:
                raise ValueError("validation configuration changed: " + key)
        if source["pool"] != journal["pool"]:
            raise ValueError("validation requires the calibration KV budget")
        result["training_trace"] = source["trace_sha256"]
    for group in ("existing", "new"):
        def select(data):
            return [r for r in data if r["group"] == group and r["horizon"] == 10]
        trained, evaluated = select(calibration), select(rows)
        theta = fit(trained)
        result["curves"][group] = dict(training_points=len(trained), threshold=theta,
            evaluation="held_out" if train else "in_sample_only",
            threshold_rule=regret(evaluated, theta) if theta is not None else None,
            always_wait=regret(evaluated, float("inf")), always_shrink=regret(evaluated, 0.))
    write(directory / ("validation.json" if train else "summary.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--train", type=Path)
    args = parser.parse_args()
    report(args.directory, args.train)
