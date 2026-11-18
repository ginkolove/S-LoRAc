"""Prefix reconstruction with a bounded number of resident worker blocks."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT"])
def _lowra_reconstruct_bounded_kernel(
    BASE, MINI, B_BUFFER, OUT, PREFIX_LOCS, B_LOCS,
    RANK_STARTS, RANK_LENS, SCALINGS,
    COUNT,
    P: tl.constexpr, E: tl.constexpr, R_TOTAL: tl.constexpr,
    MAX_RANK: tl.constexpr,
    ACTUAL_RANK: tl.constexpr,
    B_PACKED: tl.constexpr,
    ADD_BASE: tl.constexpr,
    WORKERS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    n_tiles = tl.cdiv(E, BLOCK_N)
    p_tiles = tl.cdiv(P, BLOCK_M)
    group_size = 8
    total = tl.cdiv(COUNT, group_size) * group_size * p_tiles * n_tiles
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    # The grid bounds active workers; the loop covers every LoRA/token/output tile.
    for tile in range(tl.program_id(0), total, WORKERS):
        n = tile % n_tiles
        group = tile // (n_tiles * p_tiles * group_size)
        adapter = group * group_size + (tile // n_tiles) % group_size
        p = (tile // (n_tiles * group_size)) % p_tiles
        rows = p * BLOCK_M + offs_m
        cols = n * BLOCK_N + offs_n
        if adapter < COUNT:
            rank_start = tl.load(RANK_STARTS + adapter)
            rank_len = tl.load(RANK_LENS + adapter)
            scale = tl.load(SCALINGS + adapter)
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            # MAX_RANK remains the B index-table stride in both execution modes.
            rank_limit = rank_len if ACTUAL_RANK else MAX_RANK
            for r_base in range(0, rank_limit, BLOCK_R):
                ranks = r_base + offs_r
                mini = tl.load(MINI + rows[:, None] * R_TOTAL + rank_start + ranks[None, :],
                               mask=(rows[:, None] < P) & (ranks[None, :] < rank_len), other=0.0)
                if B_PACKED:
                    flat = cols[None, :] * rank_len + ranks[:, None]
                    valid = (ranks[:, None] < rank_len) & (cols[None, :] < E)
                    b_rows = tl.load(B_LOCS + adapter * MAX_RANK + flat // E, mask=valid, other=0)
                    b = tl.load(B_BUFFER + b_rows * E + flat % E, mask=valid, other=0.0)
                else:
                    b_rows = tl.load(B_LOCS + adapter * MAX_RANK + ranks, mask=ranks < rank_len, other=0)
                    b = tl.load(B_BUFFER + b_rows[:, None] * E + cols[None, :],
                                mask=(ranks[:, None] < rank_len) & (cols[None, :] < E), other=0.0)
                acc += tl.dot(mini, b)
            base = tl.load(BASE + rows[:, None] * E + cols[None, :],
                           mask=(rows[:, None] < P) & (cols[None, :] < E), other=0.0)
            dest = tl.load(PREFIX_LOCS + adapter * P + rows, mask=rows < P, other=0)
            value = base + acc * scale if ADD_BASE else acc * scale
            tl.store(OUT + dest[:, None] * E + cols[None, :], value,
                     mask=(rows[:, None] < P) & (cols[None, :] < E))


@torch.no_grad()
def lowra_reconstruct_bounded(base, mini, b_buffer, out, prefix_locs, b_locs,
                              rank_starts, rank_lens, scalings, workers=108,
                              launch_config=(16, 64, 16, 4), b_packed=False, add_base=True,
                              rank_loop="max"):
    """Write indexed KV; b_packed selects S-LoRA's paged HF B[E,rank] layout."""
    if rank_loop not in ("max", "actual"):
        raise ValueError("rank_loop must be 'max' or 'actual'")
    count = prefix_locs.shape[0]
    if count == 0:
        return
    if workers <= 0:
        raise ValueError("reconstruction workers must be positive")
    if base.ndim != 2 or mini.ndim != 2 or mini.shape[0] != base.shape[0]:
        raise ValueError("invalid BaseKV/MiniKV dimensions")
    prefix, embed = base.shape
    if prefix_locs.shape != (count, prefix) or b_locs.ndim != 2 or b_locs.shape[0] != count:
        raise ValueError("invalid prefix or B index shape")
    if any(t.ndim != 2 or t.shape[1] != embed for t in (b_buffer, out)):
        raise ValueError("invalid B/output width")
    if any(t.shape != (count,) for t in (rank_starts, rank_lens, scalings)):
        raise ValueError("invalid per-adapter metadata")
    tensors = (base, mini, b_buffer, out, prefix_locs, b_locs, rank_starts, rank_lens, scalings)
    if any(not t.is_contiguous() or t.device != base.device for t in tensors):
        raise ValueError("bounded reconstruction requires contiguous tensors on one device")
    if not base.is_cuda or base.dtype != torch.float16:
        raise ValueError("bounded reconstruction requires CUDA FP16")
    if any(t.dtype != base.dtype for t in (mini, b_buffer, out)):
        raise ValueError("bounded reconstruction requires a common FP16 dtype")
    block_m, block_n, block_r, warps = launch_config
    if (any(not isinstance(v, int) or v <= 0 or v & (v - 1) for v in (block_m, block_n, block_r))
            or warps not in (4, 8)):
        raise ValueError("invalid reconstruction tile or warp count")
    max_rank = b_locs.shape[1]
    if max_rank <= 0 or prefix <= 0:
        raise ValueError("prefix and rank capacity must be positive")
    workers = min(workers, count * triton.cdiv(prefix, block_m) * triton.cdiv(embed, block_n))
    _lowra_reconstruct_bounded_kernel[(workers,)](
        base, mini, b_buffer, out, prefix_locs, b_locs, rank_starts, rank_lens, scalings, count,
        P=prefix, E=embed, R_TOTAL=mini.shape[1], MAX_RANK=max_rank,
        ACTUAL_RANK=rank_loop == "actual",
        B_PACKED=b_packed,
        ADD_BASE=add_base,
        WORKERS=workers, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_R=block_r,
        num_warps=warps, num_stages=1,
    )
