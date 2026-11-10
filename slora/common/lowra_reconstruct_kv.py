import torch
import triton
import triton.language as tl


@triton.jit
def _lowra_reconstruct_kv_kernel(
    BASE_K, BASE_V, MINI_K, MINI_V, B_BUFFER, OUT_K, OUT_V,
    PREFIX_LOCS, K_B_LOCS, V_B_LOCS, RANK_STARTS, RANK_LENS, SCALINGS,
    P: tl.constexpr, R_TOTAL: tl.constexpr, E: tl.constexpr,
    MAX_RANK: tl.constexpr,
    B_PACKED: tl.constexpr,
    ADD_K_BASE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    EMBED_FIRST: tl.constexpr,
):
    p_block = tl.program_id(1)
    if EMBED_FIRST:
        lora_pid = tl.program_id(2)
        kv_tile = tl.program_id(0)
    else:
        lora_pid = tl.program_id(0)
        kv_tile = tl.program_id(2)
    e_block = kv_tile // 2

    # Adjacent K/V tiles belong to one launch without doubling per-block state.
    if kv_tile % 2 == 0:
        base_ptr = BASE_K
        mini_ptr = MINI_K
        b_locs = K_B_LOCS
        out_ptr = OUT_K
    else:
        base_ptr = BASE_V
        mini_ptr = MINI_V
        b_locs = V_B_LOCS
        out_ptr = OUT_V

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
            mini_ptr + p_offsets[:, None] * R_TOTAL + rank_start + r[None, :],
            mask=(p_offsets[:, None] < P) & (r[None, :] < rank_len),
            other=0.0,
        )
        if B_PACKED:
            flat = e_offsets[None, :] * rank_len + r[:, None]
            valid = (r[:, None] < rank_len) & (e_offsets[None, :] < E)
            b_rows = tl.load(b_locs + lora_pid * MAX_RANK + flat // E, mask=valid, other=0)
            b = tl.load(B_BUFFER + b_rows * E + flat % E, mask=valid, other=0.0)
        else:
            b_rows = tl.load(b_locs + lora_pid * MAX_RANK + r, mask=r < rank_len, other=0)
            b = tl.load(B_BUFFER + b_rows[:, None] * E + e_offsets[None, :],
                        mask=(r[:, None] < rank_len) & (e_offsets[None, :] < E), other=0.0)
        acc += tl.dot(mini, b)

    base = tl.load(
        base_ptr + p_offsets[:, None] * E + e_offsets[None, :],
        mask=(p_offsets[:, None] < P) & (e_offsets[None, :] < E),
        other=0.0,
    )
    out_rows = tl.load(PREFIX_LOCS + lora_pid * P + p_offsets,
                       mask=p_offsets < P, other=0)
    base_weight = 1.0 if ADD_K_BASE else (kv_tile % 2).to(tl.float32)
    value = base * base_weight + acc * scaling
    tl.store(
        out_ptr + out_rows[:, None] * E + e_offsets[None, :],
        value,
        mask=(p_offsets[:, None] < P) & (e_offsets[None, :] < E),
    )


@torch.no_grad()
def lowra_reconstruct_kv_batched(
    base_k, base_v, mini_k, mini_v, b_buffer, out_k, out_v,
    prefix_locs, k_b_locs, v_b_locs, rank_starts, rank_lens, scalings,
    launch_config=None,
    embed_first=None,
    b_packed=False,
    add_k_base=True,
):
    """Reconstruct K and V into existing indexed buffers with one kernel launch."""
    if prefix_locs.shape[0] == 0:
        return
    if base_k.ndim != 2 or base_k.shape != base_v.shape:
        raise ValueError("fused prefix reconstruction requires matching K/V dimensions")
    if mini_k.ndim != 2 or mini_k.shape != mini_v.shape or mini_k.shape[0] != base_k.shape[0]:
        raise ValueError("invalid packed MiniKV dimensions")
    count, prefix_len = prefix_locs.shape
    embed = base_k.shape[1]
    if prefix_len != base_k.shape[0] or k_b_locs.ndim != 2 or k_b_locs.shape != v_b_locs.shape:
        raise ValueError("invalid prefix/adapter index dimensions")
    if k_b_locs.shape[0] != count or k_b_locs.shape[1] <= 0:
        raise ValueError("invalid adapter rank capacity")
    matrices = (b_buffer, out_k, out_v)
    if any(t.ndim != 2 or t.shape[1] != embed for t in matrices):
        raise ValueError("adapter/output width does not match the prefix width")
    if any(t.shape != (count,) for t in (rank_starts, rank_lens, scalings)):
        raise ValueError("invalid per-LoRA reconstruction metadata")
    tensors = (base_k, base_v, mini_k, mini_v, *matrices, prefix_locs,
               k_b_locs, v_b_locs, rank_starts, rank_lens, scalings)
    if any(not t.is_contiguous() or t.device != base_k.device for t in tensors):
        raise ValueError("fused reconstruction requires contiguous tensors on one device")
    if not base_k.is_cuda or base_k.dtype != torch.float16:
        raise ValueError("fused reconstruction currently requires CUDA FP16")
    if any(t.dtype != base_k.dtype for t in (base_v, mini_k, mini_v, *matrices)):
        raise ValueError("fused reconstruction requires a common FP16 tensor dtype")

    # Profiled low-rank path; keep the original tile for other rank/width regimes.
    # This depends on source dimensions, never query lengths or decode progress.
    low_rank = k_b_locs.shape[1] <= 16 and embed >= 128 and prefix_len >= 128
    if embed_first is None:
        embed_first = low_rank
    block_m, block_n, block_r, warps, stages = launch_config or (
        (32, 128, 16, 4, 1) if low_rank else (16, 32, 32, 4, 3))
    grid = (count, triton.cdiv(prefix_len, block_m), 2 * triton.cdiv(embed, block_n))
    if embed_first:
        grid = (grid[2], grid[1], grid[0])
    _lowra_reconstruct_kv_kernel[grid](
        base_k, base_v, mini_k, mini_v, b_buffer, out_k, out_v,
        prefix_locs, k_b_locs, v_b_locs, rank_starts, rank_lens, scalings,
        P=prefix_len, R_TOTAL=mini_k.shape[1], E=embed,
        MAX_RANK=k_b_locs.shape[1],
        B_PACKED=b_packed,
        ADD_K_BASE=add_k_base,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_R=block_r,
        EMBED_FIRST=embed_first,
        num_warps=warps, num_stages=stages,
    )
