import argparse
import json
import time

import torch

from slora.common.lowra_reconstruct import lowra_reconstruct_batched


@torch.no_grad()
def reconstruct_torch_mm(
    base,
    mini,
    b_buffer,
    out_buffer,
    prefix_locs,
    b_locs,
    rank_starts,
    rank_lens,
    scalings,
):
    for i in range(prefix_locs.shape[0]):
        rank_start = int(rank_starts[i].item())
        rank_len = int(rank_lens[i].item())
        mini_slice = mini[:, rank_start: rank_start + rank_len]
        b = b_buffer[b_locs[i, :rank_len]]
        recovered = base + torch.mm(mini_slice, b) * scalings[i]
        out_buffer[prefix_locs[i]] = recovered


def timed_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def build_case(args):
    torch.manual_seed(args.seed)
    rank_lens_cpu = [args.rank + (i % max(1, args.rank_jitter + 1)) for i in range(args.num_loras)]
    rank_starts_cpu = []
    cursor = 0
    for rank in rank_lens_cpu:
        rank_starts_cpu.append(cursor)
        cursor += rank

    max_rank = max(rank_lens_cpu)
    base = torch.randn(args.prefix_len, args.embed_dim, dtype=torch.float16, device="cuda")
    mini = torch.randn(args.prefix_len, cursor, dtype=torch.float16, device="cuda")
    b_buffer = torch.randn(args.num_loras * max_rank + 13, args.embed_dim, dtype=torch.float16, device="cuda")
    out_torch = torch.empty(args.num_loras * args.prefix_len + 32, args.embed_dim, dtype=torch.float16, device="cuda")
    out_triton = torch.empty_like(out_torch)

    prefix_locs = torch.empty(args.num_loras, args.prefix_len, dtype=torch.long, device="cuda")
    b_locs = torch.empty(args.num_loras, max_rank, dtype=torch.long, device="cuda")
    for i, rank in enumerate(rank_lens_cpu):
        prefix_locs[i] = torch.arange(i * args.prefix_len, (i + 1) * args.prefix_len, dtype=torch.long, device="cuda")
        b_locs[i, :rank] = torch.arange(i * max_rank, i * max_rank + rank, dtype=torch.long, device="cuda")
        if rank < max_rank:
            b_locs[i, rank:] = 0

    rank_starts = torch.tensor(rank_starts_cpu, dtype=torch.long, device="cuda")
    rank_lens = torch.tensor(rank_lens_cpu, dtype=torch.long, device="cuda")
    scalings = torch.rand(args.num_loras, dtype=torch.float16, device="cuda")
    return base, mini, b_buffer, out_torch, out_triton, prefix_locs, b_locs, rank_starts, rank_lens, scalings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--rank-jitter", type=int, default=0)
    parser.add_argument("--num-loras", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({
            "skipped": True,
            "reason": "CUDA is not available",
            "torch": torch.__version__,
        }))
        return

    case = build_case(args)
    base, mini, b_buffer, out_torch, out_triton, prefix_locs, b_locs, rank_starts, rank_lens, scalings = case

    def run_torch():
        reconstruct_torch_mm(base, mini, b_buffer, out_torch, prefix_locs, b_locs, rank_starts, rank_lens, scalings)

    def run_triton():
        lowra_reconstruct_batched(base, mini, b_buffer, out_triton, prefix_locs, b_locs, rank_starts, rank_lens, scalings)

    run_torch()
    run_triton()
    torch.cuda.synchronize()
    max_err = (out_torch[prefix_locs.reshape(-1)] - out_triton[prefix_locs.reshape(-1)]).abs().max().item()

    torch_ms = timed_cuda(run_torch, args.warmup, args.iters)
    triton_ms = timed_cuda(run_triton, args.warmup, args.iters)
    print(json.dumps({
        "skipped": False,
        "prefix_len": args.prefix_len,
        "embed_dim": args.embed_dim,
        "rank": args.rank,
        "rank_jitter": args.rank_jitter,
        "num_loras": args.num_loras,
        "max_err": max_err,
        "torch_mm_ms": torch_ms,
        "triton_fused_ms": triton_ms,
        "speedup": torch_ms / triton_ms if triton_ms > 0 else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
