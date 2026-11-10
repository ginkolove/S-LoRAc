"""Rotate reconstructed delta-K before adding an already-rotated BaseK."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_add_base(BASE, OUT, LOCS, COS, SIN,
                     P: tl.constexpr, E: tl.constexpr, D: tl.constexpr,
                     BLOCK_P: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    head = tl.program_id(1)
    adapter = tl.program_id(2)
    half = tl.arange(0, D // 2)
    locations = tl.load(LOCS + adapter * P + rows, mask=rows < P, other=0)
    offsets = head * D + half
    first = tl.load(OUT + locations[:, None] * E + offsets[None, :],
                    mask=rows[:, None] < P, other=0).to(tl.float32)
    second = tl.load(OUT + locations[:, None] * E + offsets[None, :] + D // 2,
                     mask=rows[:, None] < P, other=0).to(tl.float32)
    c = tl.load(COS + rows[:, None] * (D // 2) + half[None, :],
                mask=rows[:, None] < P, other=0).to(tl.float32)
    s = tl.load(SIN + rows[:, None] * (D // 2) + half[None, :],
                mask=rows[:, None] < P, other=0).to(tl.float32)
    base_first = tl.load(BASE + rows[:, None] * E + offsets[None, :],
                         mask=rows[:, None] < P, other=0).to(tl.float32)
    base_second = tl.load(BASE + rows[:, None] * E + offsets[None, :] + D // 2,
                          mask=rows[:, None] < P, other=0).to(tl.float32)
    # One program owns both halves, so its in-place stores cannot race readers.
    tl.store(OUT + locations[:, None] * E + offsets[None, :],
             base_first + (first * c - second * s), mask=rows[:, None] < P)
    tl.store(OUT + locations[:, None] * E + offsets[None, :] + D // 2,
             base_second + (first * s + second * c), mask=rows[:, None] < P)


@torch.no_grad()
def rotate_delta_k_add_base(base, out, locations, cos, sin):
    prefix, heads, dim = base.shape
    if dim < 16 or dim & (dim - 1) or cos.shape != (prefix, dim // 2) or cos.shape != sin.shape:
        raise ValueError("invalid prefix RoPE dimensions")
    if locations.ndim != 2 or locations.shape[1] != prefix or out.shape[1:] != (heads, dim):
        raise ValueError("invalid reconstructed prefix addresses or K shape")
    tensors = (base, out, locations, cos, sin)
    if any(not t.is_cuda or not t.is_contiguous() or t.device != base.device for t in tensors):
        raise ValueError("prefix key RoPE requires contiguous CUDA tensors on one device")
    if any(t.dtype != torch.float16 for t in (base, out, cos, sin)):
        raise ValueError("prefix key RoPE currently requires FP16")
    _rotate_add_base[(triton.cdiv(prefix, 32), heads, locations.shape[0])](
        base, out, locations, cos, sin, P=prefix, E=heads * dim, D=dim,
        BLOCK_P=32, num_warps=4, num_stages=1)
