import torch
import triton
import triton.language as tl


# Triton 2.1 indexes specialization hints consistently when runtime args precede constexprs.
@triton.jit(do_not_specialize=["MAX_PRIVATE"])
def _prefix_decode_scores(
    Q, PREFIX_K, PRIVATE_K, PREFIX_LOC, PRIVATE_LOC, LENGTHS, SCORES, PREFIX_REQUEST_IDS,
    MAX_PRIVATE,
    Q0: tl.constexpr, Q1: tl.constexpr,
    PK0: tl.constexpr, PK1: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr,
    PL0: tl.constexpr, BL0: tl.constexpr,
    HEADS: tl.constexpr, GROUPS: tl.constexpr,
    PREFIX: tl.constexpr,
    D: tl.constexpr, BLOCK: tl.constexpr, SCALE: tl.constexpr,
    INDIRECT_PREFIX: tl.constexpr,
):
    batch = tl.program_id(0)
    prefix_row = batch
    if INDIRECT_PREFIX:
        prefix_row = tl.load(PREFIX_REQUEST_IDS + batch)
    head = tl.program_id(1)
    pos = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    length = tl.load(LENGTHS + batch)
    prefix_mask = pos < PREFIX
    private_pos = tl.maximum(pos - PREFIX, 0)
    private_mask = (pos >= PREFIX) & (private_pos < length)
    prefix_idx = tl.load(PREFIX_LOC + prefix_row * PL0 + pos, mask=prefix_mask, other=0)
    private_idx = tl.load(PRIVATE_LOC + batch * BL0 + MAX_PRIVATE - length + private_pos,
                          mask=private_mask, other=0)
    q = tl.load(Q + batch * Q0 + head * Q1 + dims)
    kv_head = head // GROUPS
    pk = tl.load(PREFIX_K + prefix_idx[:, None] * PK0 + kv_head * PK1 + dims[None, :],
                 mask=prefix_mask[:, None], other=0.0)
    k = tl.load(PRIVATE_K + private_idx[:, None] * K0 + kv_head * K1 + dims[None, :],
                mask=private_mask[:, None], other=0.0)
    score = tl.sum((pk + k).to(tl.float32) * q[None, :].to(tl.float32), 1) * SCALE
    score = tl.where(prefix_mask | private_mask, score, -float("inf"))
    stride = PREFIX + MAX_PRIVATE
    tl.store(SCORES + (batch * HEADS + head) * stride + pos, score, mask=pos < stride)


@triton.jit(do_not_specialize=["MAX_PRIVATE"])
def _prefix_decode_reduce(
    SCORES, PREFIX_V, PRIVATE_V, PREFIX_LOC, PRIVATE_LOC, LENGTHS, OUT, PREFIX_REQUEST_IDS,
    MAX_PRIVATE,
    PV0: tl.constexpr, PV1: tl.constexpr,
    V0: tl.constexpr, V1: tl.constexpr,
    PL0: tl.constexpr, BL0: tl.constexpr,
    O0: tl.constexpr, O1: tl.constexpr,
    HEADS: tl.constexpr, GROUPS: tl.constexpr,
    PREFIX: tl.constexpr,
    D: tl.constexpr, BLOCK: tl.constexpr,
    INDIRECT_PREFIX: tl.constexpr,
):
    batch = tl.program_id(0)
    prefix_row = batch
    if INDIRECT_PREFIX:
        prefix_row = tl.load(PREFIX_REQUEST_IDS + batch)
    head = tl.program_id(1)
    length = tl.load(LENGTHS + batch)
    dims = tl.arange(0, D)
    offsets = tl.arange(0, BLOCK)
    maximum = -float("inf")
    denominator = 0.0
    acc = tl.zeros([D], tl.float32)
    kv_head = head // GROUPS
    stride = PREFIX + MAX_PRIVATE
    for start in range(0, PREFIX + length, BLOCK):
        pos = start + offsets
        valid = pos < PREFIX + length
        scores = tl.load(SCORES + (batch * HEADS + head) * stride + pos,
                         mask=valid, other=-float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 0))
        scale = tl.exp(maximum - next_max)
        probabilities = tl.exp(scores - next_max)
        prefix_mask = pos < PREFIX
        private_pos = tl.maximum(pos - PREFIX, 0)
        private_mask = (pos >= PREFIX) & valid
        prefix_idx = tl.load(PREFIX_LOC + prefix_row * PL0 + pos, mask=prefix_mask, other=0)
        private_idx = tl.load(PRIVATE_LOC + batch * BL0 + MAX_PRIVATE - length + private_pos,
                              mask=private_mask, other=0)
        pv = tl.load(PREFIX_V + prefix_idx[:, None] * PV0 + kv_head * PV1 + dims[None, :],
                     mask=prefix_mask[:, None], other=0.0)
        v = tl.load(PRIVATE_V + private_idx[:, None] * V0 + kv_head * V1 + dims[None, :],
                    mask=private_mask[:, None], other=0.0)
        denominator = denominator * scale + tl.sum(probabilities, 0)
        acc = acc * scale + tl.sum(probabilities[:, None] * (pv + v).to(tl.float32), 0)
        maximum = next_max
    tl.store(OUT + batch * O0 + head * O1 + dims, acc / denominator)


@torch.no_grad()
def token_attention_fwd_with_prefix(q, prefix_k, prefix_v, private_k, private_v,
                                    prefix_b_loc, b_loc, private_seq_len, max_private_len,
                                    prefix_request_ids=None):
    """Decode without copying or persisting the ephemeral prefix into request KV."""
    batch, heads, dim = q.shape
    prefix_len = prefix_b_loc.shape[1]
    if prefix_request_ids is not None and (
            prefix_request_ids.shape != (batch,) or prefix_request_ids.stride(0) != 1):
        raise ValueError("invalid request-to-prefix mapping")
    request_ids = prefix_request_ids if prefix_request_ids is not None else prefix_b_loc
    if dim not in (16, 32, 64, 128) or q.stride(2) != 1:
        raise ValueError("unsupported decode head dimension/stride")
    if heads % private_k.shape[1] or prefix_b_loc.stride(1) != 1 or b_loc.stride(1) != 1:
        raise ValueError("invalid prefix decode layout")
    groups = heads // private_k.shape[1]
    scores = torch.empty((batch, heads, prefix_len + max_private_len),
                         dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    _prefix_decode_scores[(batch, heads, triton.cdiv(prefix_len + max_private_len, 32))](
        q, prefix_k, private_k, prefix_b_loc, b_loc, private_seq_len, scores, request_ids,
        max_private_len,
        q.stride(0), q.stride(1), prefix_k.stride(0), prefix_k.stride(1),
        private_k.stride(0), private_k.stride(1), prefix_b_loc.stride(0), b_loc.stride(0),
        heads, groups, prefix_len, dim, 32, dim ** -0.5, prefix_request_ids is not None,
        num_warps=4, num_stages=1,
    )
    _prefix_decode_reduce[(batch, heads)](
        scores, prefix_v, private_v, prefix_b_loc, b_loc, private_seq_len, out, request_ids,
        max_private_len,
        prefix_v.stride(0), prefix_v.stride(1), private_v.stride(0), private_v.stride(1),
        prefix_b_loc.stride(0), b_loc.stride(0), out.stride(0), out.stride(1),
        heads, groups, prefix_len, dim, 64, prefix_request_ids is not None,
        num_warps=1, num_stages=1,
    )
    return out
