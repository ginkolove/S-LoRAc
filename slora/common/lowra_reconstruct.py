import torch
import triton
import triton.language as tl


@triton.jit
def _lowra_reconstruct_kernel(
    BASE,
    MINI,
    B_BUFFER,
    OUT_BUFFER,
    PREFIX_LOCS,
    B_LOCS,
    RANK_STARTS,
    RANK_LENS,
    SCALINGS,
    P: tl.constexpr,
    R_TOTAL: tl.constexpr,
    E: tl.constexpr,
    MAX_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    lora_pid = tl.program_id(0)
    p_block = tl.program_id(1)
    e_block = tl.program_id(2)

    p_offsets = p_block * BLOCK_M + tl.arange(0, BLOCK_M)
    e_offsets = e_block * BLOCK_N + tl.arange(0, BLOCK_N)
    r_offsets = tl.arange(0, BLOCK_R)

    rank_start = tl.load(RANK_STARTS + lora_pid)
    rank_len = tl.load(RANK_LENS + lora_pid)
    scaling = tl.load(SCALINGS + lora_pid)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for r_base in range(0, MAX_RANK, BLOCK_R):
        r = r_base + r_offsets
        mini = tl.load(
            MINI + p_offsets[:, None] * R_TOTAL + rank_start + r[None, :],
            mask=(p_offsets[:, None] < P) & (r[None, :] < rank_len),
            other=0.0,
        )
        b_rows = tl.load(
            B_LOCS + lora_pid * MAX_RANK + r,
            mask=r < rank_len,
            other=0,
        )
        b = tl.load(
            B_BUFFER + b_rows[:, None] * E + e_offsets[None, :],
            mask=(r[:, None] < rank_len) & (e_offsets[None, :] < E),
            other=0.0,
        )
        acc += tl.dot(mini, b)

    base = tl.load(
        BASE + p_offsets[:, None] * E + e_offsets[None, :],
        mask=(p_offsets[:, None] < P) & (e_offsets[None, :] < E),
        other=0.0,
    )
    out = base + acc * scaling

    out_rows = tl.load(
        PREFIX_LOCS + lora_pid * P + p_offsets,
        mask=p_offsets < P,
        other=0,
    )
    tl.store(
        OUT_BUFFER + out_rows[:, None] * E + e_offsets[None, :],
        out,
        mask=(p_offsets[:, None] < P) & (e_offsets[None, :] < E),
    )


@torch.no_grad()
def lowra_reconstruct_batched(
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
    if prefix_locs.shape[0] == 0:
        return

    P = base.shape[0]
    E = base.shape[1]
    R_TOTAL = mini.shape[1]
    max_rank = b_locs.shape[1]

    block_m = 16
    block_n = 32
    block_r = 32
    grid = (
        prefix_locs.shape[0],
        triton.cdiv(P, block_m),
        triton.cdiv(E, block_n),
    )
    _lowra_reconstruct_kernel[grid](
        base,
        mini,
        b_buffer,
        out_buffer,
        prefix_locs,
        b_locs,
        rank_starts,
        rank_lens,
        scalings,
        P=P,
        R_TOTAL=R_TOTAL,
        E=E,
        MAX_RANK=max_rank,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        num_warps=4,
        num_stages=3,
    )
