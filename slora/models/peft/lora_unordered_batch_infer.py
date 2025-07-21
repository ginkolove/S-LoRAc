import numpy as np
import torch
import torch.nn as nn
from typing import final

from slora.common.infer_utils import init_bloc
from slora.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from slora.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from slora.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from slora.utils.infer_utils import mark_cost_time
from slora.utils.infer_utils import calculate_time, mark_start, mark_end
from slora._kernels import dispatch_bgmv


class LoraUnorderedBatchInfer:

    def __init__(self, base_model, adapters, infer_adapter=None):
        self.base_model = base_model

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
            no_lora_copy=False,
            adapter_dirs=None,
            use_system_cache=False,
            system_prompt_hashes=None):

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
                                 b_loc, b_start_loc, b_seq_len, no_lora_compute, adapter_dirs, 
                                 use_system_cache, system_prompt_hashes)
        else:
            for _ in range(3):
                self.delta.append(torch.zeros((len(b_seq_len), self.max_lora_dim), dtype=torch.float16, device="cuda"))
            return self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids,
                                b_loc, b_start_loc, b_seq_len,
                                no_lora_compute, no_lora_copy, adapter_dirs)


    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_start_loc, b_seq_len, no_lora_compute=False, adapter_dirs=None,
                 use_system_cache=False, system_prompt_hashes=None):

        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        b_seq_len_numpy = b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                        for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
                self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
                self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.mem_manager = self.base_model.mem_manager
        
        # 检查是否使用system_prompt缓存
        infer_state.use_system_cache = use_system_cache
        infer_state.system_prompt_hashes = system_prompt_hashes if system_prompt_hashes else []
        infer_state.adapter_dirs = adapter_dirs if adapter_dirs else []
        
        # 如果使用system_prompt缓存，需要特殊处理
        if use_system_cache and system_prompt_hashes:
            print(f"LoRA prefill using system prompt cache for batch")
            # 检查所有请求是否都有对应的system_prompt缓存
            has_system_cache = True
            for i, (adapter_dir, sys_hash) in enumerate(zip(adapter_dirs, system_prompt_hashes)):
                if sys_hash is None or not self.base_model.mem_manager.has_system_prompt_cache(adapter_dir):
                    has_system_cache = False
                    print(f"Warning: Request {i} (adapter: {adapter_dir}) missing system prompt cache")
                    break
            
            if has_system_cache:
                infer_state.using_system_cache = True
                # 获取system_prompt的token长度（假设所有缓存长度相同）
                first_cache = self.base_model.mem_manager.get_system_prompt_kv(adapter_dirs[0])
                if first_cache:
                    infer_state.system_prompt_tokens = first_cache['key_cache'][0].shape[0]
                    print(f"Using system prompt cache with {infer_state.system_prompt_tokens} tokens")
            else:
                infer_state.using_system_cache = False
        else:
            infer_state.using_system_cache = False
        
        infer_state.prefill_mem_index = self.base_model.mem_manager.alloc(infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)
        
        predict_logics = self._context_forward(input_ids, infer_state, no_lora_compute)
        return predict_logics


    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids,
                b_loc, b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False, adapter_dirs=None):
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
                                          input_ids, b_loc, b_start_loc, b_seq_len, False, adapter_dirs)
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
        
        # 如果使用system_prompt缓存，先复制缓存的KV
        if getattr(infer_state, 'using_system_cache', False):
            system_tokens = getattr(infer_state, 'system_prompt_tokens', 0)
            if system_tokens > 0:
                # 为每个batch中的请求复制system_prompt KV
                for batch_idx in range(infer_state.batch_size):
                    adapter_dir = infer_state.adapter_dirs[batch_idx] if batch_idx < len(infer_state.adapter_dirs) else None
                    
                    # 获取该adapter的system_prompt缓存
                    system_cache = infer_state.mem_manager.get_system_prompt_kv(adapter_dir)
                    if system_cache and layer_id < len(system_cache['key_cache']):
                        start_loc = infer_state.b_start_loc[batch_idx].item()
                        seq_len = infer_state.b_seq_len[batch_idx].item()
                        
                        # 确保序列长度大于system_prompt长度
                        if seq_len >= system_tokens:
                            cached_key = system_cache['key_cache'][layer_id]  # [sys_tokens, head_num, head_dim]
                            cached_value = system_cache['value_cache'][layer_id]
                            
                            # 复制到对应位置
                            cache_k[start_loc:start_loc + system_tokens] = cached_key
                            cache_v[start_loc:start_loc + system_tokens] = cached_value
                            
                            # 调整position信息，跳过system_prompt部分
                            # 这里需要修改input_embs和position信息，只处理用户输入部分
                            user_start = start_loc + system_tokens
                            user_length = seq_len - system_tokens
                            
                            if user_length > 0:
                                # 只对用户输入部分计算QKV
                                user_input1 = input1[user_start:user_start + user_length]
                                user_cache_k = cache_k[user_start:user_start + user_length]
                                user_cache_v = cache_v[user_start:user_start + user_length]
                                
                                # 对用户输入部分重新计算位置编码
                                user_pos_start = system_tokens
                                user_position_cos = infer_state.position_cos[user_start:user_start + user_length]
                                user_position_sin = infer_state.position_sin[user_start:user_start + user_length]
                                
                                # 计算用户输入的QKV
                                q_user = self._compute_user_qkv(layer_id, user_input1, user_cache_k, user_cache_v, 
                                                               user_position_cos, user_position_sin, no_lora_compute)
                                
                                print(f"Layer {layer_id}: Used system cache ({system_tokens} tokens) + computed user input ({user_length} tokens)")
                            else:
                                print(f"Layer {layer_id}: Only using system cache, no user input")
                        else:
                            print(f"Warning: Sequence length {seq_len} < system tokens {system_tokens}, falling back to normal computation")
                    else:
                        print(f"Warning: No system cache found for adapter {adapter_dir} at layer {layer_id}")

        # gen new q, k, v (batch different adapters)
        q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
        input1 = None
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        # compute attention
        o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # residual
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        return

    def _compute_user_qkv(self, layer_id, user_input, user_cache_k, user_cache_v, user_pos_cos, user_pos_sin, no_lora_compute=False):
        """
        为用户输入部分单独计算QKV（当使用system_prompt缓存时）
        """
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        
        # 计算Q
        q = torch.mm(user_input.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
        
        # 应用旋转位置编码到Q
        from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_), user_pos_cos, user_pos_sin)
        
        # 计算K
        torch.mm(user_input.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=user_cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        rotary_emb_fwd(user_cache_k, user_pos_cos, user_pos_sin)
        
        # 计算V
        torch.mm(user_input.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=user_cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        
        # TODO: 这里还需要处理LoRA的delta计算
        if not no_lora_compute:
            print("Warning: LoRA computation for system cache mode not fully implemented yet")
        
        return q


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

