"""Profile indexed prefix reconstruction alone and alongside an unchanged FFN."""

import argparse
import json
from pathlib import Path
import sys

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slora.common.lowra_reconstruct_kv import lowra_reconstruct_kv_batched


CONFIGS = ((16, 32, 32, 4, 3), (32, 64, 32, 4, 1),
           (32, 128, 32, 4, 1), (64, 64, 32, 4, 1),
           (64, 128, 32, 4, 1), (32, 128, 16, 4, 1))


def measure(reconstruct, ffn, stream, mode, iterations):
    start, end, ready = (torch.cuda.Event(enable_timing=True) for _ in range(3))

    def run():
        start.record()
        if mode == "overlap":
            with torch.cuda.stream(stream):
                stream.wait_event(start)
                reconstruct()
                ready.record()
            ffn()
            torch.cuda.current_stream().wait_event(ready)
        elif mode == "ffn":
            ffn()
        else:
            reconstruct()
            if mode == "serial":
                ffn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(3):
        run()
    return sum(run() for _ in range(iterations)) / iterations


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=[2048, 4608, 8193])
    parser.add_argument("--unique-loras", nargs="+", type=int, default=[1, 8, 20])
    parser.add_argument("--ranks", nargs="+", type=int, default=[16])
    parser.add_argument("--embed", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--ffn-tokens", nargs="+", type=int, default=[40, 1920])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--embed-first", action="store_true",
                        help="schedule adjacent column tiles before different adapters")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.prefix_lengths, *args.unique_loras, *args.ranks, *args.ffn_tokens,
           args.embed, args.intermediate, args.iterations) <= 0:
        parser.error("dimensions and iterations must be positive")
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    weights = [torch.randn((args.embed, args.intermediate), device="cuda", dtype=torch.float16) * 0.01
               for _ in range(2)]
    down = torch.randn((args.intermediate, args.embed), device="cuda", dtype=torch.float16) * 0.01
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                  scope="single-layer synthetic nonzero tensors; paired unchanged FFN; not serving throughput",
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, rows=[])
    for rank in args.ranks:
        for prefix in args.prefix_lengths:
            for count in args.unique_loras:
                base = [torch.randn(prefix, args.embed, device="cuda", dtype=torch.float16) * 0.02
                        for _ in range(2)]
                mini = [torch.randn(prefix, count * rank, device="cuda", dtype=torch.float16) * 0.02
                        for _ in range(2)]
                b = torch.randn(2 * count * rank, args.embed, device="cuda", dtype=torch.float16) * 0.02
                # Gaps between rows exercise indexed output without another KV copy.
                locs = (torch.arange(count * prefix, device="cuda") * 2).view(count, prefix)
                b_locs = torch.arange(2 * count * rank, device="cuda").view(2, count, rank)
                starts = torch.arange(count, device="cuda") * rank
                ranks = torch.full((count,), rank, device="cuda", dtype=torch.long)
                scales = torch.ones(count, device="cuda", dtype=torch.float16)
                output = [torch.empty(2 * count * prefix, args.embed, device="cuda", dtype=torch.float16)
                          for _ in range(2)]
                for tokens in args.ffn_tokens:
                    inputs = torch.randn(tokens, args.embed, device="cuda", dtype=torch.float16) * 0.02

                    def ffn():
                        gate = torch.mm(inputs, weights[0])
                        torch.nn.functional.silu(gate, inplace=True)
                        up = torch.mm(inputs, weights[1])
                        return torch.mm(gate * up, down)

                    ffn_ms = measure(None, ffn, stream, "ffn", args.iterations)
                    for config in CONFIGS:
                        def reconstruct():
                            lowra_reconstruct_kv_batched(*base, *mini, b, *output, locs,
                                                        *b_locs, starts, ranks, scales,
                                                        launch_config=config,
                                                        embed_first=args.embed_first)

                        row = dict(prefix=prefix, unique_loras=count, rank=rank, ffn_tokens=tokens,
                                   launch_config=config, embed_first=args.embed_first, ffn_ms=ffn_ms)
                        for mode in ("reconstruct", "serial", "overlap"):
                            row[mode + "_ms"] = measure(reconstruct, ffn, stream, mode, args.iterations)
                        report["rows"].append(row)
                        print(json.dumps(row), flush=True)
                del base, mini, b, output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
