"""Compare reconstruction tiles using the actual decode compute window."""

import itertools
import json
from pathlib import Path
import random

import torch

from profile_compute_window import ComputeWindow, parse_args
from overlap_analysis import summarize


@torch.no_grad()
def main(args):
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "tuning.json"
    if output.exists():
        raise FileExistsError(output)
    window = ComputeWindow(args)
    window.run("serial_split")
    window.reference = {name: getattr(window, name).clone() for name in ("activation", "q", "k", "v")}
    window.prefix_reference = [out.clone() for out in window.reconstruction.outputs]
    tiles = [(16, 32, 32, 4, 3), (16, 128, 16, 4, 1), (32, 64, 16, 4, 1),
             (32, 128, 16, 4, 1), (32, 256, 16, 4, 1), (64, 128, 16, 4, 1),
             (16, 128, 16, 4, 3), (64, 64, 16, 4, 1)]
    configs = [(mode, tile, order) for tile, order in itertools.product(tiles, (False, True))
               for mode in ("serial_fused", "overlap_fused", "overlap_main_first_fused")]
    configs.extend((mode, None, None) for mode in ("serial_split", "overlap_split", "overlap_main_first_split"))
    results = {config: [] for config in configs}
    for mode, tile, order in configs:
        args.fused_config, args.embed_first = tile, order
        for _ in range(args.warmup):
            window.run(mode)
        window.validate(mode)
    rng = random.Random(args.seed)
    for _ in range(args.iterations):
        rng.shuffle(configs)
        for config in configs:
            mode, args.fused_config, args.embed_first = config
            results[config].append(window.run(mode)["total_ms"])
    rows = [dict(mode=mode, tile=tile, embed_first=order, timings=summarize(samples), samples=samples)
            for (mode, tile, order), samples in results.items()]
    rows.sort(key=lambda row: row["timings"]["median"])
    output.write_text(json.dumps(dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                     correctness_passed=True, rows=rows), indent=2) + "\n")
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main(parse_args())
