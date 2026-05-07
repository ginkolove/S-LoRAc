import torch
import triton
import triton.language as tl


@triton.jit
def _context_attention_prefix_kernel(
    Q,
    K,
    V,
    PrefixK,
    PrefixV,
    PrefixBLoc,
    BStartLoc,
    BSeqLen,
    Out,
    sm_scale,
    Tmp,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_pkbs,
    stride_pkh,
    stride_pkd,
    stride_pvbs,
    stride_pvh,
    stride_pvd,
    stride_pbs,
    stride_pn,
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

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, PREFIX_LEN + (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_pos = start_n + offs_n
        is_prefix = key_pos < PREFIX_LEN
        query_key_pos = key_pos - PREFIX_LEN
        valid_key = key_pos < (PREFIX_LEN + cur_q_len)

        prefix_slots = tl.load(
            PrefixBLoc + cur_batch * stride_pbs + key_pos * stride_pn,
            mask=is_prefix,
            other=0,
        )

        prefix_k_ptrs = (
            PrefixK
            + prefix_slots[None, :] * stride_pkbs
            + cur_kv_head * stride_pkh
            + offs_d[:, None] * stride_pkd
        )
        query_k_ptrs = (
            K
            + (cur_q_start + query_key_pos)[None, :] * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k_ptrs = tl.where(is_prefix[None, :], prefix_k_ptrs, query_k_ptrs)
        k = tl.load(k_ptrs, mask=valid_key[None, :], other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        valid_row = offs_m < cur_q_len
        causal = is_prefix[None, :] | (query_key_pos[None, :] <= offs_m[:, None])
        qk = tl.where(valid_row[:, None] & valid_key[None, :] & causal, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        has_valid = m_ij != float("-inf")
        m_ij_safe = tl.where(has_valid, m_ij, m_i)
        p = tl.exp(qk - m_ij_safe[:, None])
        p = tl.where(has_valid[:, None], p, 0.0)
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij_safe)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij_safe - m_i_new)
        l_i_candidate = alpha * l_i + beta * l_ij
        l_i_safe = tl.where(l_i_candidate == 0.0, 1.0, l_i_candidate)

        p_scale = tl.where(has_valid, beta / l_i_safe, 0.0)
        p = p * p_scale[:, None]
        acc_scale = tl.where(has_valid, l_i / l_i_safe * alpha, 1.0)
        tl.store(tmp_ptrs, acc_scale)
        acc_scale = tl.load(tmp_ptrs)
        acc = acc * acc_scale[:, None]

        prefix_v_ptrs = (
            PrefixV
            + prefix_slots[:, None] * stride_pvbs
            + cur_kv_head * stride_pvh
            + offs_d[None, :] * stride_pvd
        )
        query_v_ptrs = (
            V
            + (cur_q_start + query_key_pos)[:, None] * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v_ptrs = tl.where(is_prefix[:, None], prefix_v_ptrs, query_v_ptrs)
        v = tl.load(v_ptrs, mask=valid_key[:, None], other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = tl.where(has_valid, l_i_candidate, l_i)
        m_i = tl.where(has_valid, m_i_new, m_i)

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
    k,
    v,
    o,
    prefix_b_loc,
    prefix_len,
    prefix_key_buffer,
    prefix_value_buffer,
    b_start_loc,
    b_seq_len,
    max_query_len,
):
    block = 128
    d_model = q.shape[-1]
    assert d_model == k.shape[-1] == v.shape[-1]
    assert d_model in {16, 32, 64, 128}
    assert prefix_len > 0
    assert prefix_b_loc.shape[1] >= prefix_len

    batch, q_head_num = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q_head_num // k.shape[1]
    sm_scale = 1.0 / (d_model ** 0.5)
    grid = (batch, q_head_num, triton.cdiv(max_query_len, block))
    num_warps = 4 if d_model <= 64 else 8
    tmp = torch.empty(
        (batch, q_head_num, max_query_len + 256),
        dtype=torch.float32,
        device=q.device,
    )

    _context_attention_prefix_kernel[grid](
        q,
        k,
        v,
        prefix_key_buffer,
        prefix_value_buffer,
        prefix_b_loc,
        b_start_loc,
        b_seq_len,
        o,
        sm_scale,
        tmp,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        prefix_key_buffer.stride(0),
        prefix_key_buffer.stride(1),
        prefix_key_buffer.stride(2),
        prefix_value_buffer.stride(0),
        prefix_value_buffer.stride(1),
        prefix_value_buffer.stride(2),
        prefix_b_loc.stride(0),
        prefix_b_loc.stride(1),
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
