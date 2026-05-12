import torch
import triton
import triton.language as tl


@torch.no_grad()
def _context_attention_torch_with_prefix(
    q,
    o,
    full_b_loc,
    prefix_len,
    mem_key_buffer,
    mem_value_buffer,
    b_start_loc,
    b_seq_len,
):
    q_head_num = q.shape[1]
    kv_head_num = mem_key_buffer.shape[1]
    head_dim = q.shape[-1]
    kv_group_num = q_head_num // kv_head_num
    sm_scale = 1.0 / (head_dim ** 0.5)

    for batch_idx in range(b_seq_len.shape[0]):
        q_len = int(b_seq_len[batch_idx].item())
        q_start = int(b_start_loc[batch_idx].item())
        total_len = prefix_len + q_len

        cur_q = q[q_start:q_start + q_len].transpose(0, 1)
        kv_slots = full_b_loc[batch_idx, 0:total_len].long()
        cur_k = mem_key_buffer.index_select(0, kv_slots)
        cur_v = mem_value_buffer.index_select(0, kv_slots)
        if kv_group_num != 1:
            cur_k = torch.repeat_interleave(cur_k, kv_group_num, dim=1)
            cur_v = torch.repeat_interleave(cur_v, kv_group_num, dim=1)
        cur_k = cur_k.transpose(0, 1)
        cur_v = cur_v.transpose(0, 1)

        scores = torch.matmul(cur_q.float(), cur_k.transpose(1, 2).float()) * sm_scale
        query_pos = prefix_len + torch.arange(q_len, device=q.device)
        key_pos = torch.arange(total_len, device=q.device)
        causal = key_pos[None, :] <= query_pos[:, None]
        scores = scores.masked_fill(~causal[None, :, :], -1.0e20)
        probs = torch.softmax(scores, dim=-1).to(cur_v.dtype)
        out = torch.matmul(probs, cur_v).transpose(0, 1).contiguous()
        o[q_start:q_start + q_len].copy_(out)
    return


@triton.jit
def _context_attention_indexed_prefix_kernel(
    Q,
    MemK,
    MemV,
    FullBLoc,
    BStartLoc,
    BSeqLen,
    Out,
    sm_scale,
    Tmp,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_mkbs,
    stride_mkh,
    stride_mkd,
    stride_mvbs,
    stride_mvh,
    stride_mvd,
    stride_fbs,
    stride_fn,
    stride_obs,
    stride_oh,
    stride_od,
    stride_tmp_b,
    stride_tmp_h,
    stride_tmp_s,
    kv_group_num: tl.constexpr,
    PREFIX_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_q_len = tl.load(BSeqLen + cur_batch)
    cur_q_start = tl.load(BStartLoc + cur_batch)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + (cur_q_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < cur_q_len, other=0.0)
    tmp_ptrs = Tmp + cur_batch * stride_tmp_b + cur_head * stride_tmp_h + offs_m * stride_tmp_s

    block_start_loc = start_m * BLOCK_M
    block_mask = tl.where(block_start_loc < cur_q_len, 1, 0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_mask * (PREFIX_LEN + (start_m + 1) * BLOCK_M), BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_pos = start_n + offs_n
        valid_key = key_pos < (PREFIX_LEN + cur_q_len)

        key_slots = tl.load(
            FullBLoc + cur_batch * stride_fbs + key_pos * stride_fn,
            mask=valid_key,
            other=0,
        )

        k_ptrs = (
            MemK
            + key_slots[None, :] * stride_mkbs
            + cur_kv_head * stride_mkh
            + offs_d[:, None] * stride_mkd
        )
        k = tl.load(k_ptrs, mask=valid_key[None, :], other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        causal = key_pos[None, :] <= (PREFIX_LEN + offs_m[:, None])
        qk = tl.where(valid_key[None, :] & causal, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        tmp_m_i_ptrs = tmp_ptrs
        tmp_m_ij_ptrs = tmp_ptrs + BLOCK_M * stride_tmp_s
        tl.store(tmp_m_i_ptrs, m_i)
        tl.store(tmp_m_ij_ptrs, m_ij)
        m_i_prev = tl.load(tmp_m_i_ptrs)
        m_ij = tl.load(tmp_m_ij_ptrs)

        m_i_new = tl.maximum(m_i_prev, m_ij)
        alpha = tl.exp(m_i_prev - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_candidate = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_candidate
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_candidate * alpha
        tl.store(tmp_ptrs, acc_scale)
        acc_scale = tl.load(tmp_ptrs)
        acc = acc * acc_scale[:, None]

        v_ptrs = (
            MemV
            + key_slots[:, None] * stride_mvbs
            + cur_kv_head * stride_mvh
            + offs_d[None, :] * stride_mvd
        )
        v = tl.load(v_ptrs, mask=valid_key[:, None], other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_candidate
        m_i = m_i_new

    out_ptrs = (
        Out
        + (cur_q_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_q_len)
    return


@torch.no_grad()
def context_attention_fwd_with_prefix(
    q,
    o,
    full_b_loc,
    prefix_len,
    mem_key_buffer,
    mem_value_buffer,
    b_start_loc,
    b_seq_len,
    max_query_len,
):
    # The indexed Triton implementation is kept in this file, but the current
    # Triton version used by the S-LoRA environment repeatedly fails TTGIR
    # lowering on vectors produced by indexed KV loads. Use the PyTorch path by
    # default so Prefix-S-LoRA serving and scheduler experiments are runnable.
    _context_attention_torch_with_prefix(
        q,
        o,
        full_b_loc,
        prefix_len,
        mem_key_buffer,
        mem_value_buffer,
        b_start_loc,
        b_seq_len,
    )
    return

    block = 128
    d_model = q.shape[-1]
    assert d_model == mem_key_buffer.shape[-1] == mem_value_buffer.shape[-1]
    assert d_model in {16, 32, 64, 128}
    assert prefix_len > 0
    assert full_b_loc.shape[1] >= prefix_len + max_query_len

    batch, q_head_num = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q_head_num // mem_key_buffer.shape[1]
    sm_scale = 1.0 / (d_model ** 0.5)
    grid = (batch, q_head_num, triton.cdiv(max_query_len, block))
    num_warps = 4 if d_model <= 64 else 8
    tmp = torch.empty(
        (batch, q_head_num, max_query_len + 256),
        dtype=torch.float32,
        device=q.device,
    )

    _context_attention_indexed_prefix_kernel[grid](
        q,
        mem_key_buffer,
        mem_value_buffer,
        full_b_loc,
        b_start_loc,
        b_seq_len,
        o,
        sm_scale,
        tmp,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        mem_key_buffer.stride(0),
        mem_key_buffer.stride(1),
        mem_key_buffer.stride(2),
        mem_value_buffer.stride(0),
        mem_value_buffer.stride(1),
        mem_value_buffer.stride(2),
        full_b_loc.stride(0),
        full_b_loc.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        tmp.stride(0),
        tmp.stride(1),
        tmp.stride(2),
        kv_group_num=kv_group_num,
        PREFIX_LEN=prefix_len,
        BLOCK_M=block,
        BLOCK_DMODEL=d_model,
        BLOCK_N=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return
