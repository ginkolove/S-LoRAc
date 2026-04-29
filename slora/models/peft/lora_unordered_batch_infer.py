import numpy as np
import os
import time
import torch
import torch.nn as nn
from typing import final

from slora.common.infer_utils import init_bloc
from slora.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_with_prefix,
)
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from slora.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from slora.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from slora.utils.infer_utils import mark_cost_time
from slora.utils.infer_utils import calculate_time, mark_start, mark_end
from slora._kernels import dispatch_bgmv


class LoraUnorderedBatchInfer:

    def __init__(self, base_model, adapters, infer_adapter=None):
        self.base_model = base_model
        self.adapters = adapters
        self.prefill_prefix_total = 0
        self.prefill_prefix_allocated_total = 0
        self.prefill_effective_max_len = 0
        self.prefill_prefix_lens = []
        self.prefill_prefix_indices = []
        self.prefix_debug_enabled = os.environ.get("SLORA_PREFIX_DEBUG", "0") == "1"
        self.prefix_profile_enabled = os.environ.get("SLORA_PREFIX_PROFILE", "0") == "1"
        self.last_prefix_restore_ms = 0.0
        self.last_prefix_context_ms = 0.0
        self.last_prefix_prefill_total_ms = 0.0
        self.last_prefix_materialize_ms = {}
        self.last_prefix_hit_count = 0
        self.last_prefix_miss_count = 0

        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        self.max_lora_dim = max(lora_layer_dim)

        self.req_bins = torch.zeros(len(adapters), dtype=torch.long, device="cuda")

        if infer_adapter is not None:
            self.infer_adapter = infer_adapter
            if isinstance(infer_adapter, NaiveInferAdapter):
                self.key_buffer = infer_adapter.key_buffer
                self.value_buffer = infer_adapter.value_buffer
            else:
                self.key_buffer = infer_adapter.mem_manager.key_buffer
                self.value_buffer = infer_adapter.mem_manager.value_buffer
            for i, adapter in enumerate(adapters):
                # FIX ME @TODO: currently not supporting adapter is None
                if adapter is None: continue
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx
        
        self.kv_embed_dim = base_model.tp_k_head_num_ * base_model.head_dim_

    def _debug(self, message):
        if self.prefix_debug_enabled:
            print(f"[prefix-prefill] {message}", flush=True)

    def _profile(self, message):
        if self.prefix_profile_enabled:
            print(f"[prefix-profile] {message}", flush=True)


    @torch.no_grad()
    def forward(
            self,
            batch_size, # number of request
            total_token_num,
            max_len_in_batch,
            input_ids, # 1D input tensor
            b_loc, # mapping to memory pool
            b_start_loc, # the start index of each request
            b_seq_len, # the current length of each request
            is_prefill=True,
            use_bmm=True,
            no_lora_compute=False,
            no_lora_copy=False):

        # Notice that batch_lora only support decoding
        assert len(b_loc) == len(b_start_loc) == len(b_seq_len)
        self.delta = []

        self.max_b_seq_len = torch.max(b_seq_len).item()

        if is_prefill:
            assert(len(self.req_bins)==len(b_seq_len))
            self.batch_req_bins = torch.repeat_interleave(self.req_bins, b_seq_len)
            # self.b_start_loc = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device="cuda"), b_seq_len[:-1]]), dim=0)
            for _ in range(3):
                self.delta.append(torch.zeros((len(self.batch_req_bins), self.max_lora_dim), dtype=torch.float16, device="cuda"))

            return self._prefill(batch_size, total_token_num, max_len_in_batch,
                                 input_ids,
                                 b_loc, b_start_loc, b_seq_len, no_lora_compute)
        else:
            for _ in range(3):
                self.delta.append(torch.zeros((len(b_seq_len), self.max_lora_dim), dtype=torch.float16, device="cuda"))
            return self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids,
                                b_loc, b_start_loc, b_seq_len,
                                no_lora_compute, no_lora_copy)


    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_start_loc, b_seq_len, no_lora_compute=False):
        prefill_start = None
        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            prefill_start = time.perf_counter()

        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.prompt_max_len_in_batch = max_len_in_batch
        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        prompt_b_seq_len = b_seq_len.clone()
        prompt_b_start_loc = b_start_loc.clone()

        prefix_lens, prefix_indices = self._restore_shared_prefix_kv()
        total_seq_lens = [int(prompt_b_seq_len[i].item()) + prefix_lens[i] for i in range(batch_size)]
        effective_max_len = max(total_seq_lens) if len(total_seq_lens) > 0 else max_len_in_batch

        b_seq_len_numpy = prompt_b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([
                np.arange(prefix_lens[i], prefix_lens[i] + b_seq_len_numpy[i])
                for i in range(len(b_seq_len_numpy))
            ], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
                self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
                self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = prompt_b_start_loc
        infer_state.b_seq_len = prompt_b_seq_len
        infer_state.prompt_b_seq_len = prompt_b_seq_len
        infer_state.shared_prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")
        infer_state.prefill_total_seq_len = torch.tensor(total_seq_lens, dtype=torch.int32, device="cuda")
        infer_state.prefill_total_token_num_with_prefix = int(sum(total_seq_lens))
        infer_state.prefill_prompt_total_token_num = infer_state.total_token_num
        infer_state.prefill_effective_max_len = effective_max_len
        infer_state.prefill_full_b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        if batch_size > 1:
            infer_state.prefill_full_b_start_loc[1:] = torch.cumsum(
                infer_state.prefill_total_seq_len, dim=0, dtype=torch.int32
            )[:-1]
        infer_state.mem_manager = self.base_model.mem_manager
        infer_state.prefill_mem_index = self.base_model.mem_manager.alloc(infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        self._init_prefill_b_loc(
            b_loc,
            prompt_b_seq_len,
            effective_max_len,
            infer_state.prefill_mem_index,
            prefix_lens,
            prefix_indices,
        )
        
        context_start = None
        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            context_start = time.perf_counter()
        predict_logics = self._context_forward(input_ids, infer_state, no_lora_compute)
        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            self.last_prefix_context_ms = (time.perf_counter() - context_start) * 1000.0
        self._finalize_prefill_prefix_layout(
            b_start_loc,
            b_seq_len,
            total_seq_lens,
            effective_max_len,
        )
        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            self.last_prefix_prefill_total_ms = (time.perf_counter() - prefill_start) * 1000.0
            self._profile(
                "prefill "
                f"batch={batch_size} prompt_tokens={total_token_num} "
                f"logical_prefix={self.prefill_prefix_total} allocated_prefix={self.prefill_prefix_allocated_total} "
                f"hits={self.last_prefix_hit_count} misses={self.last_prefix_miss_count} "
                f"restore_ms={self.last_prefix_restore_ms:.2f} "
                f"context_ms={self.last_prefix_context_ms:.2f} "
                f"total_ms={self.last_prefix_prefill_total_ms:.2f}"
            )
        return predict_logics

    def _restore_shared_prefix_kv(self):
        mem_manager = self.base_model.mem_manager
        batch_size = len(self.adapters)
        prefix_lens = [0] * batch_size
        prefix_indices = [None] * batch_size
        self.prefill_prefix_total = 0
        self.prefill_prefix_allocated_total = 0
        self.prefill_effective_max_len = 0
        self.prefill_prefix_lens = prefix_lens
        self.prefill_prefix_indices = prefix_indices
        self.last_prefix_restore_ms = 0.0
        self.last_prefix_materialize_ms = {}
        self.last_prefix_hit_count = 0
        self.last_prefix_miss_count = 0

        if (not hasattr(mem_manager, "has_static_shared_prefix") or
            not mem_manager.has_static_shared_prefix or
            not hasattr(self, "infer_adapter")):
            return prefix_lens, prefix_indices

        shared_prefix_length = mem_manager.shared_prefix_length
        lora_to_req_indices = {}
        lora_to_adapter = {}
        for i, adapter in enumerate(self.adapters):
            if adapter is None:
                continue
            if adapter.lora_dir not in mem_manager.lora_dir_to_index:
                continue
            prefix_lens[i] = shared_prefix_length
            if adapter.lora_dir not in lora_to_req_indices:
                lora_to_req_indices[adapter.lora_dir] = []
                lora_to_adapter[adapter.lora_dir] = adapter
            lora_to_req_indices[adapter.lora_dir].append(i)

        logical_prefix_total = sum(prefix_lens)
        if logical_prefix_total == 0:
            return prefix_lens, prefix_indices

        self.prefill_prefix_total = logical_prefix_total
        self.prefill_prefix_allocated_total = 0
        restore_start = None
        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            restore_start = time.perf_counter()

        for lora_dir, req_indices in lora_to_req_indices.items():
            prefix_cache = getattr(mem_manager, "prefix_cache", None)
            cache_entry = None if prefix_cache is None else prefix_cache.get(lora_dir)
            if cache_entry is not None:
                shared_prefix_index = cache_entry.prefix_indices
                self.last_prefix_hit_count += 1
                self._debug(
                    f"cache-hit lora={lora_dir} shared_prefix_len={shared_prefix_length} "
                    f"reqs={len(req_indices)}"
                )
            else:
                shared_prefix_index = mem_manager.alloc(shared_prefix_length)
                self.last_prefix_miss_count += 1
                materialize_ms = self._materialize_request_prefix_kv(shared_prefix_index, lora_to_adapter[lora_dir])
                self.last_prefix_materialize_ms[lora_dir] = materialize_ms
                self.prefill_prefix_allocated_total += shared_prefix_length
                if prefix_cache is not None:
                    prefix_cache.insert(lora_dir, shared_prefix_index, shared_prefix_length)
                self._debug(
                    f"cache-miss lora={lora_dir} restored_prefix_len={shared_prefix_length} "
                    f"reqs={len(req_indices)}"
                )
                if self.prefix_profile_enabled:
                    self._profile(
                        f"materialize lora={lora_dir} prefix_len={shared_prefix_length} "
                        f"reqs={len(req_indices)} ms={materialize_ms:.2f}"
                    )
            if prefix_cache is not None:
                prefix_cache.acquire(lora_dir, len(req_indices))
            for req_idx in req_indices:
                prefix_indices[req_idx] = shared_prefix_index

        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            self.last_prefix_restore_ms = (time.perf_counter() - restore_start) * 1000.0

        self.prefill_prefix_lens = prefix_lens
        self.prefill_prefix_indices = prefix_indices
        self._debug(
            f"prefill-prefix-summary logical_total={self.prefill_prefix_total} "
            f"allocated_total={self.prefill_prefix_allocated_total}"
        )
        return prefix_lens, prefix_indices

    def _materialize_request_prefix_kv(self, req_prefix_index, adapter):
        materialize_start = None
        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            materialize_start = time.perf_counter()
        mem_manager = self.base_model.mem_manager
        prefix_view = mem_manager.get_static_prefix_view_by_dir(adapter.lora_dir)
        adapter_idx = self.infer_adapter.idx_map[adapter.lora_dir]
        scaling = self.infer_adapter.a_scaling[adapter_idx].to(dtype=self.base_model.mem_manager.dtype)
        start = int(self.infer_adapter.a_start[adapter_idx].item())
        a_len = int(self.infer_adapter.a_len[adapter_idx].item())
        loc = self.infer_adapter.a_loc[start:start + a_len]
        rank = a_len // 4
        hidden_size = self.kv_embed_dim
        target_dtype = mem_manager.dtype

        for layer_id in range(self.base_model.layers_num):
            base_key = prefix_view["base_key"][layer_id].to(dtype=target_dtype)
            base_value = prefix_view["base_value"][layer_id].to(dtype=target_dtype)
            mini_key = prefix_view["mini_key"][layer_id].to(dtype=target_dtype)
            mini_value = prefix_view["mini_value"][layer_id].to(dtype=target_dtype)

            k_lora_b = self.value_buffer[layer_id][loc[rank:rank * 2]].reshape(hidden_size, rank).transpose(0, 1)
            v_lora_b = self.value_buffer[layer_id][loc[rank * 2:rank * 3]].reshape(hidden_size, rank).transpose(0, 1)
            k_lora_b = k_lora_b.to(dtype=target_dtype)
            v_lora_b = v_lora_b.to(dtype=target_dtype)

            recovered_key = base_key.view(base_key.shape[0], hidden_size) + torch.mm(mini_key, k_lora_b) * scaling
            recovered_value = base_value.view(base_value.shape[0], hidden_size) + torch.mm(mini_value, v_lora_b) * scaling

            mem_manager.key_buffer[layer_id][req_prefix_index] = recovered_key.view(
                base_key.shape[0], self.base_model.tp_k_head_num_, self.base_model.head_dim_
            )
            mem_manager.value_buffer[layer_id][req_prefix_index] = recovered_value.view(
                base_value.shape[0], self.base_model.tp_v_head_num_, self.base_model.head_dim_
            )
        if self.prefix_profile_enabled:
            torch.cuda.synchronize()
            return (time.perf_counter() - materialize_start) * 1000.0
        return 0.0

    def _init_prefill_b_loc(self, b_loc, prompt_b_seq_len, effective_max_len, prompt_mem_index, prefix_lens, prefix_indices):
        prompt_offset = 0
        prompt_lens = prompt_b_seq_len.cpu().tolist()
        for req_idx, prompt_len in enumerate(prompt_lens):
            total_len = prompt_len + prefix_lens[req_idx]
            row_start = effective_max_len - total_len
            if prefix_lens[req_idx] > 0:
                b_loc[req_idx, row_start:row_start + prefix_lens[req_idx]] = prefix_indices[req_idx]
            prompt_slice = prompt_mem_index[prompt_offset:prompt_offset + prompt_len]
            b_loc[
                req_idx,
                row_start + prefix_lens[req_idx]:row_start + prefix_lens[req_idx] + prompt_len,
            ] = prompt_slice
            prompt_offset += prompt_len

    def _finalize_prefill_prefix_layout(self, b_start_loc, b_seq_len, total_seq_lens, effective_max_len):
        if len(total_seq_lens) == 0:
            self.prefill_effective_max_len = effective_max_len
            return
        total_seq_tensor = torch.tensor(total_seq_lens, dtype=torch.int32, device="cuda")
        b_seq_len[:] = total_seq_tensor
        b_start_loc[0] = 0
        if len(total_seq_lens) > 1:
            b_start_loc[1:] = torch.cumsum(total_seq_tensor, dim=0, dtype=torch.int32)[:-1]
        self.prefill_effective_max_len = effective_max_len


    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids,
                b_loc, b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False):
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        
        infer_state.mem_manager = self.base_model.mem_manager
        prefix_cache = getattr(self.base_model.mem_manager, "prefix_cache", None)
        if prefix_cache is not None:
            prefix_cache.ensure_available(batch_size)

        alloc_mem = self.base_model.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.base_model.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        infer_state.init_some_extra_state(self.base_model, batch_size, total_token_num, max_len_in_batch,
                                          input_ids, b_loc, b_start_loc, b_seq_len, False)
        predict_logics = self._token_forward(input_ids, infer_state, no_lora_compute, no_lora_copy)
        return predict_logics


    @final
    def _context_forward(self, input_ids, infer_state, no_lora_compute=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.context_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_context_forward(i, input_embs, infer_state, no_lora_compute)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _token_forward(self, input_ids, infer_state, no_lora_compute=False, no_lora_copy=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state, no_lora_compute, no_lora_copy)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        self._lora_context_attention(layer_id, input_embs, infer_state, no_lora_compute)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        layer_infer._context_ffn(input_embs, infer_state, layer_weight)
        return input_embs


    @final
    # @calculate_time(show=True, min_cost_ms=0)
    def _lora_token_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        self._lora_token_attention(layer_id, input_embs, infer_state, no_lora_compute, no_lora_copy)
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # mark_start("token_ffn")
        layer_infer._token_ffn(input_embs, infer_state, layer_weight)
        # mark_end("token_ffn")
        return input_embs


    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _lora_context_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        if self.prefill_prefix_total > 0:
            o = self._lora_context_attention_with_prefix_cache(
                layer_id, q, cache_k, cache_v, infer_state, layer_weight
            )
        else:
            o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # residual
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return

    def _lora_context_attention_with_prefix_cache(self, layer_id, q, cache_k, cache_v, infer_state, layer_weight):
        o = torch.empty_like(q)
        context_attention_fwd_with_prefix(
            q.view(-1, self.base_model.tp_q_head_num_, self.base_model.head_dim_),
            infer_state.mem_manager.key_buffer[layer_id],
            infer_state.mem_manager.value_buffer[layer_id],
            o.view(-1, self.base_model.tp_q_head_num_, self.base_model.head_dim_),
            infer_state.b_loc,
            infer_state.b_start_loc,
            infer_state.shared_prefix_lens,
            infer_state.prompt_b_seq_len,
            infer_state.prefill_total_seq_len,
            infer_state.prompt_max_len_in_batch,
            infer_state.prefill_effective_max_len,
        )
        return o


    # @calculate_time(show=True, min_cost_ms=0)
    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        # layer normalization
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        # fetch k, v
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        # gen new q, k, v (batch different adapters)
        q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._batch_lora_get_o(layer_id, o, infer_state, no_lora_compute)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return
    

    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False, no_lora_copy=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        # q (bs, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
         # @TODO: fix me, filter requests querying only base model
        assert(len(q)==len(self.req_bins))

        if not no_lora_compute:
            # mark_start("get_q")
            delta_qA = self.delta[0]
            dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 0, self.infer_adapter.a_scaling)
            dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None
            # mark_end("get_q")

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)

        # k (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_k")
            delta_kA = self.delta[1]
            dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None
            # mark_end("get_k")

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (bs, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))

        if not no_lora_compute:
            # mark_start("get_v")
            delta_vA = self.delta[2]
            dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 2, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
            # mark_end("get_v")

        return q        


    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
        assert(len(q)==len(self.batch_req_bins))
        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            delta_qA = self.delta[0]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_qA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_qA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         q, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         0, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id],
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling)
                dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None

        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)

        # k (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if not no_lora_compute:
            delta_kA = self.delta[1]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_kA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_kA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None

        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if not no_lora_compute:
            delta_vA = self.delta[2]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
            # if 1 ==0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_vA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_vA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_v.view(-1, base_model.tp_v_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
        return q
    

    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        
        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        
        if not no_lora_compute:
            # mark_start("get_o")
            delta_oA = self.delta[0]
            dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 3, self.infer_adapter.a_scaling)
            dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 3, self.infer_adapter.a_scaling)
            # delta_oA = None
            # mark_end("get_o")
        return o


    def _lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        if not no_lora_compute:
            delta_oA = self.delta[0]
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_oA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         o, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 3, self.infer_adapter.a_scaling)
            # delta_oA = None
        return o
