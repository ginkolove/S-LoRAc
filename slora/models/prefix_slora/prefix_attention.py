import torch

import triton
import triton.language as tl


@torch.no_grad()
def _context_attention_torch_with_prefix(
    q,
    k,
    v,
    o,
    b_start_loc,
    b_seq_len,
    prefix_b_loc,
    prefix_len,
    mem_key_buffer,
    mem_value_buffer,
):
    q_head_num = q.shape[1]
    kv_head_num = k.shape[1]
    head_dim = q.shape[-1]
    kv_group_num = q_head_num // kv_head_num
    sm_scale = 1.0 / (head_dim ** 0.5)

    for batch_idx in range(b_seq_len.shape[0]):
        q_len = int(b_seq_len[batch_idx].item())
        q_start = int(b_start_loc[batch_idx].item())
        total_len = prefix_len + q_len

        cur_q = q[q_start:q_start + q_len].transpose(0, 1)
        prefix_slots = prefix_b_loc[batch_idx, 0:prefix_len].long()
        prefix_k = mem_key_buffer.index_select(0, prefix_slots)
        prefix_v = mem_value_buffer.index_select(0, prefix_slots)
        query_k = k[q_start:q_start + q_len]
        query_v = v[q_start:q_start + q_len]

        cur_k = torch.cat([prefix_k, query_k], dim=0)
        cur_v = torch.cat([prefix_v, query_v], dim=0)
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
def _prefix_context_attention_kernel(
    Q,
    K,
    V,
    PREFIX_B_LOC,
    MEM_K,
    MEM_V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    TMP,
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_pbs,
    stride_pp,
    stride_mkbs,
    stride_mkh,
    stride_mkd,
    stride_mvbs,
    stride_mvh,
    stride_mvd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_tmp_b,
    stride_tmp_h,
    stride_tmp_s,
    kv_group_num,
    PREFIX_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_start = tl.load(B_Start_Loc + cur_batch)
    block_start_loc = start_m * BLOCK_M

    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + (cur_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < cur_seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_seq_len, 1, 0)
    max_visible_len = PREFIX_LEN + tl.minimum(cur_seq_len, (start_m + 1) * BLOCK_M)

    for start_n in range(0, block_mask * max_visible_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_pos = start_n + offs_n
        is_prefix = key_pos < PREFIX_LEN
        query_key_pos = key_pos - PREFIX_LEN
        safe_query_key_pos = tl.maximum(query_key_pos, 0)

        prefix_locs = tl.load(
            PREFIX_B_LOC + cur_batch * stride_pbs + key_pos * stride_pp,
            mask=is_prefix,
            other=-1,
        )
        valid_prefix = is_prefix & (prefix_locs >= 0)
        valid_query_key = (~is_prefix) & (query_key_pos < cur_seq_len)
        valid_key = valid_prefix | valid_query_key

        prefix_k = tl.load(
            MEM_K
            + prefix_locs[None, :] * stride_mkbs
            + cur_kv_head * stride_mkh
            + offs_d[:, None] * stride_mkd,
            mask=valid_prefix[None, :],
            other=0.0,
        )
        query_k = tl.load(
            K
            + (cur_start + safe_query_key_pos[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd,
            mask=valid_query_key[None, :],
            other=0.0,
        )

        qk_prefix = tl.dot(q, prefix_k)
        qk_query = tl.dot(q, query_k)
        qk = tl.where(is_prefix[None, :], qk_prefix, qk_query)
        qk *= sm_scale

        causal = is_prefix[None, :] | (offs_m[:, None] >= query_key_pos[None, :])
        qk = tl.where(
            valid_key[None, :] & causal,
            qk,
            float("-inf"),
        )

        tmp_ptrs = TMP + cur_batch * stride_tmp_b + cur_head * stride_tmp_h + offs_m * stride_tmp_s
        tmp_m_i_ptrs = tmp_ptrs
        tmp_m_ij_ptrs = tmp_ptrs + BLOCK_M * stride_tmp_s

        m_ij = tl.max(qk, 1)
        tl.store(tmp_m_i_ptrs, m_i)
        tl.store(tmp_m_ij_ptrs, m_ij)
        m_i_prev = tl.load(tmp_m_i_ptrs)
        m_ij = tl.load(tmp_m_ij_ptrs)

        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i_prev, m_ij)
        alpha = tl.exp(m_i_prev - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        tl.store(tmp_ptrs, acc_scale)
        acc_scale = tl.load(tmp_ptrs)
        acc = acc * acc_scale[:, None]

        prefix_v = tl.load(
            MEM_V
            + prefix_locs[:, None] * stride_mvbs
            + cur_kv_head * stride_mvh
            + offs_d[None, :] * stride_mvd,
            mask=valid_prefix[:, None],
            other=0.0,
        )
        query_v = tl.load(
            V
            + (cur_start + safe_query_key_pos[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=valid_query_key[:, None],
            other=0.0,
        )
        prefix_p = tl.where(is_prefix[None, :], p, 0.0).to(prefix_v.dtype)
        query_p = tl.where(is_prefix[None, :], 0.0, p).to(query_v.dtype)
        acc += tl.dot(prefix_p, prefix_v)
        acc += tl.dot(query_p, query_v)
        l_i = l_i_new
        m_i = m_i_new

    out_ptrs = (
        Out
        + (cur_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_seq_len)


@torch.no_grad()
def context_attention_fwd_with_prefix(
    q,
    k,
    v,
    o,
    b_start_loc,
    b_seq_len,
    max_input_len,
    prefix_b_loc,
    prefix_len,
    mem_key_buffer,
    mem_value_buffer,
    attention_fwd=None,
):
    """Run query prefill attention over shared prefix KV plus query KV.

    Prefix KV lives in the unified memory pool and is addressed by
    prefix_b_loc. Query K/V are the query-only prefill tensors, so requests
    sharing the same adapter also share the same prefix slots.
    """

    if prefix_len <= 0:
        if attention_fwd is None:
            raise ValueError("attention_fwd is required when prefix_len <= 0")
        attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len)
        return

    head_dim = q.shape[-1]
    assert head_dim == k.shape[-1] and head_dim == v.shape[-1]
    assert head_dim == mem_key_buffer.shape[-1] == mem_value_buffer.shape[-1]
    assert head_dim in {16, 32, 64, 128}
    assert q.shape[1] % k.shape[1] == 0
    assert prefix_b_loc is not None
    assert prefix_b_loc.shape[1] >= prefix_len

    _context_attention_torch_with_prefix(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        prefix_b_loc,
        prefix_len,
        mem_key_buffer,
        mem_value_buffer,
    )
    return

    block = 128
    sm_scale = 1.0 / (head_dim ** 0.5)
    batch_size, q_head_num = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    grid = (batch_size, q_head_num, triton.cdiv(max_input_len, block))
    num_warps = 4 if head_dim <= 64 else 8
    tmp = torch.empty(
        (batch_size, q_head_num, max_input_len + 256),
        dtype=torch.float32,
        device=q.device,
    )

    _prefix_context_attention_kernel[grid](
        q,
        k,
        v,
        prefix_b_loc,
        mem_key_buffer,
        mem_value_buffer,
        sm_scale,
        b_start_loc,
        b_seq_len,
        tmp,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        prefix_b_loc.stride(0),
        prefix_b_loc.stride(1),
        mem_key_buffer.stride(0),
        mem_key_buffer.stride(1),
        mem_key_buffer.stride(2),
        mem_value_buffer.stride(0),
        mem_value_buffer.stride(1),
        mem_value_buffer.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        tmp.stride(0),
        tmp.stride(1),
        tmp.stride(2),
        kv_group_num=kv_group_num,
        PREFIX_LEN=prefix_len,
        BLOCK_M=block,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=block,
        num_warps=num_warps,
        num_stages=1,
    )
