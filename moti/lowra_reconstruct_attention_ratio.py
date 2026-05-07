import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = None
lowra_reconstruct_batched = None
context_attention_fwd_with_prefix = None


def load_runtime_deps():
    global torch, lowra_reconstruct_batched, context_attention_fwd_with_prefix
    if torch is not None:
        return
    import torch as torch_mod
    from slora.common.lowra_reconstruct import lowra_reconstruct_batched as reconstruct_mod
    from slora.models.lowra.prefix_attention import (
        context_attention_fwd_with_prefix as attention_mod,
    )

    torch = torch_mod
    lowra_reconstruct_batched = reconstruct_mod
    context_attention_fwd_with_prefix = attention_mod


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


def lowra_reconstruct_torch_mm(
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


def build_case(args):
    torch.manual_seed(args.seed)
    dtype = torch.float16
    device = "cuda"

    kv_embed_dim = args.kv_heads * args.head_dim
    total_rank = args.unique_loras * args.rank
    pool_tokens = args.unique_loras * args.prefix_len + args.batch_size * args.query_len + 32

    base_k = torch.randn(args.prefix_len, kv_embed_dim, dtype=dtype, device=device)
    base_v = torch.randn(args.prefix_len, kv_embed_dim, dtype=dtype, device=device)
    mini_k = torch.randn(args.prefix_len, total_rank, dtype=dtype, device=device)
    mini_v = torch.randn(args.prefix_len, total_rank, dtype=dtype, device=device)
    b_k = torch.randn(args.unique_loras * args.rank, kv_embed_dim, dtype=dtype, device=device)
    b_v = torch.randn(args.unique_loras * args.rank, kv_embed_dim, dtype=dtype, device=device)

    mem_k = torch.empty(pool_tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device)
    mem_v = torch.empty_like(mem_k)
    prefix_locs = torch.empty(args.unique_loras, args.prefix_len, dtype=torch.long, device=device)
    b_locs = torch.empty(args.unique_loras, args.rank, dtype=torch.long, device=device)
    for i in range(args.unique_loras):
        prefix_locs[i] = torch.arange(
            i * args.prefix_len,
            (i + 1) * args.prefix_len,
            dtype=torch.long,
            device=device,
        )
        b_locs[i] = torch.arange(
            i * args.rank,
            (i + 1) * args.rank,
            dtype=torch.long,
            device=device,
        )

    rank_starts = torch.arange(
        0,
        args.unique_loras * args.rank,
        args.rank,
        dtype=torch.long,
        device=device,
    )
    rank_lens = torch.full((args.unique_loras,), args.rank, dtype=torch.long, device=device)
    scalings = torch.full((args.unique_loras,), args.scaling, dtype=dtype, device=device)

    total_query_tokens = args.batch_size * args.query_len
    q = torch.randn(total_query_tokens, args.q_heads, args.head_dim, dtype=dtype, device=device)
    query_k = torch.randn(total_query_tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device)
    query_v = torch.randn_like(query_k)
    o = torch.empty_like(q)

    b_seq_len = torch.full((args.batch_size,), args.query_len, dtype=torch.int32, device=device)
    b_start_loc = torch.arange(
        0,
        total_query_tokens,
        args.query_len,
        dtype=torch.int32,
        device=device,
    )
    prefix_b_loc = torch.empty(args.batch_size, args.prefix_len, dtype=torch.long, device=device)
    for i in range(args.batch_size):
        lora_idx = i % args.unique_loras
        prefix_b_loc[i] = prefix_locs[lora_idx]

    return {
        "base_k": base_k,
        "base_v": base_v,
        "mini_k": mini_k,
        "mini_v": mini_v,
        "b_k": b_k,
        "b_v": b_v,
        "mem_k": mem_k,
        "mem_v": mem_v,
        "prefix_locs": prefix_locs,
        "b_locs": b_locs,
        "rank_starts": rank_starts,
        "rank_lens": rank_lens,
        "scalings": scalings,
        "q": q,
        "query_k": query_k,
        "query_v": query_v,
        "o": o,
        "b_start_loc": b_start_loc,
        "b_seq_len": b_seq_len,
        "prefix_b_loc": prefix_b_loc,
    }


def run_one(args):
    case = build_case(args)
    kv_embed_dim = args.kv_heads * args.head_dim
    reconstruct_impl = (
        lowra_reconstruct_torch_mm
        if args.reconstruct_backend == "torch"
        else lowra_reconstruct_batched
    )

    def reconstruct_layer():
        reconstruct_impl(
            case["base_k"],
            case["mini_k"],
            case["b_k"],
            case["mem_k"].view(-1, kv_embed_dim),
            case["prefix_locs"],
            case["b_locs"],
            case["rank_starts"],
            case["rank_lens"],
            case["scalings"],
        )
        reconstruct_impl(
            case["base_v"],
            case["mini_v"],
            case["b_v"],
            case["mem_v"].view(-1, kv_embed_dim),
            case["prefix_locs"],
            case["b_locs"],
            case["rank_starts"],
            case["rank_lens"],
            case["scalings"],
        )

    def prefix_attention_layer():
        context_attention_fwd_with_prefix(
            case["q"],
            case["query_k"],
            case["query_v"],
            case["o"],
            case["b_start_loc"],
            case["b_seq_len"],
            args.query_len,
            case["prefix_b_loc"],
            args.prefix_len,
            case["mem_k"],
            case["mem_v"],
        )

    reconstruct_layer()
    prefix_attention_layer()
    torch.cuda.synchronize()

    reconstruct_ms = timed_cuda(reconstruct_layer, args.warmup, args.iters)
    attention_ms = timed_cuda(prefix_attention_layer, args.warmup, args.iters)
    return {
        "skipped": False,
        "prefix_len": args.prefix_len,
        "query_len": args.query_len,
        "batch_size": args.batch_size,
        "unique_loras": args.unique_loras,
        "rank": args.rank,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "reconstruct_backend": args.reconstruct_backend,
        "reconstruct_kv_ms_per_layer": reconstruct_ms,
        "prefix_attention_ms_per_layer": attention_ms,
        "reconstruct_over_attention": (
            reconstruct_ms / attention_ms if attention_ms > 0 else None
        ),
        "reconstruct_total_ms_for_layers": reconstruct_ms * args.layers,
        "prefix_attention_total_ms_for_layers": attention_ms * args.layers,
        "layers": args.layers,
        "warmup": args.warmup,
        "iters": args.iters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-len", type=int, default=960)
    parser.add_argument("--query-len", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--unique-loras", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--scaling", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reconstruct-backend", type=str, default="triton",
                        choices=["triton", "torch"])
    parser.add_argument("--sweep-shared-ratios", action="store_true")
    parser.add_argument("--input-len", type=int, default=1920)
    args = parser.parse_args()

    load_runtime_deps()

    if not torch.cuda.is_available():
        print(json.dumps({
            "skipped": True,
            "reason": "CUDA is not available",
            "torch": torch.__version__,
        }))
        return

    if args.sweep_shared_ratios:
        for ratio in (0.25, 0.5, 0.75, 0.875):
            case_args = argparse.Namespace(**vars(args))
            case_args.prefix_len = int(args.input_len * ratio)
            case_args.query_len = args.input_len - case_args.prefix_len
            result = run_one(case_args)
            result["shared_prefix_ratio"] = ratio
            print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(run_one(args), sort_keys=True))


if __name__ == "__main__":
    main()
