"""Sweep bounded reconstruction workers against the same real decode window."""

import json
from pathlib import Path
import random

import torch

from profile_compute_window import ComputeWindow, parse_args
from overlap_analysis import summarize, analyze_compute_window_trace
from slora.common.lowra_reconstruct_bounded import lowra_reconstruct_bounded


@torch.no_grad()
def main(args):
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "result.json"
    if output.exists():
        raise FileExistsError(output)
    window = ComputeWindow(args)
    original = window.reconstruct
    original()
    window.stage("o")
    for stage in ("o_residual", "ffn_norm", "ffn", "ffn_residual", "qkv_norm", "qkv"):
        window.stage(stage)
    window.reference = {name: getattr(window, name).clone() for name in ("activation", "q", "k", "v")}
    window.prefix_reference = [out.clone() for out in window.reconstruction.outputs]
    tiles = tuple(map(tuple, args.bounded_tile)) if args.bounded_tile else (
        (16, 32, 32, 4), (16, 64, 16, 4), (16, 128, 16, 4), (32, 64, 16, 4))
    configs = [("serial_split", workers, tile, None) for workers in args.worker_counts for tile in tiles]
    configs += [("overlap_split", workers, tile, stage) for workers in args.worker_counts
                for tile in tiles for stage in args.start_stages]
    configs += [(mode, None, None, None) for mode in ("serial_split", "overlap_split", "overlap_fused")]
    samples = {config: [] for config in configs}
    graphs = {}

    def select(config):
        mode, workers, tile, _ = config
        if workers is None:
            window.reconstruct = original
        else:
            def bounded(*unused_args, **unused_kwargs):
                w = window.reconstruction
                for kv in range(2):
                    lowra_reconstruct_bounded(w.base[kv], w.mini[kv], w.b, w.outputs[kv],
                                              w.locations, w.b_locations[kv], w.starts, w.ranks,
                                              w.scales, workers=workers, launch_config=tile)
            window.reconstruct = bounded
        key = (workers, tile, mode.endswith("fused"))
        if args.reconstruct_graphs:
            if key not in graphs:
                for _ in range(3):
                    window.reconstruct(fused=mode.endswith("fused"))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=window.side):
                    window.reconstruct(fused=mode.endswith("fused"))
                graphs[key] = graph
                torch.cuda.synchronize()
            window.reconstruct = lambda *unused_args, **unused_kwargs: graphs[key].replay()
        return mode

    for config in configs:
        mode = select(config)
        for _ in range(args.warmup):
            window.run(mode, target=config[3])
        for out in window.reconstruction.outputs:
            out.fill_(-7)
        window.run(mode, target=config[3])
        window.validate(mode)
    rng = random.Random(args.seed)
    for _ in range(args.iterations):
        rng.shuffle(configs)
        for config in configs:
            mode = select(config)
            samples[config].append(window.run(mode, target=config[3])["total_ms"])
    rows = []
    for config, times in samples.items():
        mode, workers, tile, stage = config
        rows.append(dict(mode=mode, workers=workers, tile=tile, start_stage=stage,
                         timings=summarize(times), samples=times))
    rows.sort(key=lambda row: row["timings"]["median"])
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  correctness_passed=True, rows=rows)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.trace:
        selected = rows[:1] + [row for row in rows if row["workers"] is not None and row["mode"].startswith("overlap")][:1]
        selected += [row for row in rows if row["workers"] is None]
        for row in selected:
            mode = select((row["mode"], row["workers"], tuple(row["tile"]) if row["tile"] else None, row["start_stage"]))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                window.run(mode, target=row["start_stage"], instrument=True)
            trace = args.output_dir / ("%s_%s_%s_%s.json" % (mode, row["workers"], row["tile"], row["start_stage"]))
            profiler.export_chrome_trace(str(trace))
            row["trace"] = str(trace)
            row["kernel_trace"] = analyze_compute_window_trace(json.loads(trace.read_text()))
    output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k not in ("samples", "kernel_trace")}), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main(parse_args())
