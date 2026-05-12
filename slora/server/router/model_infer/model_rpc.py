import asyncio
import numpy as np
import rpyc
import torch
import traceback
import time
from collections import defaultdict

from datetime import timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain

from transformers.configuration_utils import PretrainedConfig
from slora.server.router.model_infer.infer_batch import InferBatch

from slora.common.configs.config import setting
from slora.models.llama.model import LlamaTpPartModel
from slora.models.llama2.model import Llama2TpPartModel
from slora.models.peft.lora_adapter import LoraTpPartAdapter
from slora.models.peft.lora_unordered_batch_infer import LoraUnorderedBatchInfer
from slora.models.peft.lora_single_batch_infer import LoraPEFTBatchInfer
from slora.models.bmm.lora_bmm_infer import LoraBmmInfer
from slora.server.router.model_infer.infer_adapter import InferAdapter
from slora.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from slora.common.prefix_slora_cache import PrefixSLoraPrefixCache
from slora.utils.infer_utils import set_random_seed
from slora.utils.infer_utils import calculate_time, mark_start, mark_end
from slora.utils.model_utils import get_model_config
from .post_process import sample


class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, rank_id, world_size, weight_dir, adapter_dirs,
                           max_total_token_num, load_way, mode, input_params,
			   prefetch_stream):
        import torch
        import torch.distributed as dist
        if world_size != 1:
            trans_list = [obtain(e) for e in (rank_id, world_size, weight_dir, adapter_dirs,
                                              max_total_token_num, load_way, mode)]
            rank_id, world_size, weight_dir, adapter_dirs, max_total_token_num, load_way, mode = trans_list

        self.tp_rank = rank_id
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params
        self.prefetch_stream = prefetch_stream

        self.cache = {}

        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{setting["nccl_port"]}', rank=rank_id, world_size=world_size)
        torch.cuda.set_device(rank_id)

        model_cfg = get_model_config(weight_dir, dummy=input_params.dummy)

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "llama":
                if "num_key_value_heads" in model_cfg.keys():
                    self.model = Llama2TpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy)
                    
                else:
                    self.model = LlamaTpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy)
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            raise e

        self.prefix_slora_cache = None
        if input_params.enable_prefix_slora:
            self.prefix_slora_cache = PrefixSLoraPrefixCache(
                self.model.mem_manager,
                adapter_dirs,
                input_params.prefix_slora_shared_prefix_len,
                input_params.prefix_slora_gpu_prefix_num,
                input_params.prefix_slora_cpu_prefix_num,
            )
            if rank_id == 0:
                print("Prefix-S-LoRA prefix cache:", self.prefix_slora_cache.state_dict())

        ''' init adapters '''
        # TODO support TP for adapters
        # print("adapter_dirs", adapter_dirs)
        self.adapters = []
        self.adapter_id = {}
        for adapter_dir in tqdm(adapter_dirs, desc="load adapters"):
            self.adapter_id[adapter_dir] = len(self.adapters)
            self.adapters.append(LoraTpPartAdapter(rank_id, world_size, adapter_dir, model_cfg,
                                                   swap=input_params.swap, dummy=input_params.dummy,
                                                   no_lora_swap=input_params.no_lora_swap,
						   prefetch_stream=prefetch_stream))
        self.adapter_id[None] = len(self.adapters)
        self.adapters.append(None)

        if input_params.no_mem_pool:
            head_num = self.model.config["num_attention_heads"]
            self.infer_adapter = NaiveInferAdapter.init(self.model.config["num_hidden_layers"],
                                                        head_num,
                                                        self.model.config["hidden_size"] // head_num)
        else:
            self.infer_adapter = InferAdapter.init(self.model.mem_manager,
                                                   prefetch_stream)
        ''' finish init adapters '''
        
        set_random_seed(2147483647)
        return


    @torch.no_grad()
    def exposed_load_adapters(self, adapter_dirs, prefetch=False):
        if not self.input_params.bmm:
            adapters = []
            for adapter_dir in adapter_dirs:
                if adapter_dir is not None:
                    adapters.append(self.adapters[self.adapter_id[adapter_dir]])
            self.infer_adapter.load_adapters(adapters, prefetch=prefetch)
        else:
            for adapter_dir in adapter_dirs:
                if adapter_dir is not None:
                    self.adapters[self.adapter_id[adapter_dir]].load_to_gpu(prefetch=prefetch, bmm=True)
            print(f"load {len(adapter_dirs)} on gpu")
            # total_num = 0
            # for adapter in self.adapters:
            #     if adapter is not None:
            #         total_num += 1 if adapter.is_on_gpu() else 0
            # print(f"total {total_num} on gpu")

    @torch.no_grad()
    def exposed_prefix_slora_prepare_batch(self, adapter_dirs, extra_token_num=0):
        if self.prefix_slora_cache is None:
            return []
        return self.prefix_slora_cache.prepare_many(adapter_dirs, extra_token_num)


    @torch.no_grad()
    def exposed_offload_adapters(self, reserve_dirs=None, prefetch=False):
        if not self.input_params.bmm:
            self.infer_adapter.offload_adapters(reserve_dirs if reserve_dirs is not None else [])
        else:
            reserve_dirs = reserve_dirs if reserve_dirs is not None else []
            for adapter_dir, id in self.adapter_id.items():
                if adapter_dir is not None and adapter_dir not in reserve_dirs:
                    self.adapters[id].offload_from_gpu()


    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs, dtype):
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), self.model.mem_manager, self.model.vocab_size)
        if self.prefix_slora_cache is not None:
            lora_dirs = [r["adapter_dir"] for r in reqs]
            prefix_map = self.prefix_slora_cache.acquire_many(lora_dirs)
            prefix_len = self.prefix_slora_cache.shared_prefix_len
            prefix_b_loc = torch.empty((len(reqs), prefix_len), dtype=torch.long, device="cuda")
            for i, lora_dir in enumerate(lora_dirs):
                assert lora_dir in prefix_map, f"Prefix-S-LoRA prefix is not registered for {lora_dir}"
                prefix_b_loc[i, :] = prefix_map[lora_dir]
            batch_data.prefix_slora_prefix_len = prefix_len
            batch_data.prefix_slora_b_loc = prefix_b_loc
        self.cache[batch_id] = batch_data
        return
    
    # @calculate_time(show=True, min_cost_ms=300)
    # @calculate_time(show=True, min_cost_ms=0)
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    # @calculate_time(show=True, min_cost_ms=200)
    # @calculate_time(show=True, min_cost_ms=0)
    def exposed_decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list = obtain(batch_id), obtain(req_id_list)
        # print("filter old size:", len(batch.reqs), "new size:", len(req_id_list))
        batch = self.cache.pop(batch_id)
        if self.prefix_slora_cache is not None:
            keep_ids = set(req_id_list)
            removed_lora_dirs = [
                r["adapter_dir"] for r in batch.requests
                if r["request_id"] not in keep_ids
            ]
            self.prefix_slora_cache.release_many(removed_lora_dirs)
        filter_batch = batch.filter(req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def exposed_remove_batch(self, batch_id):
        batch = self.cache.pop(batch_id)
        if self.prefix_slora_cache is not None:
            self.prefix_slora_cache.release_many([r["adapter_dir"] for r in batch.requests])
        batch.free_self()
        del batch
        # torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def _expand_batch_for_prefix_slora(self, batch: InferBatch):
        if batch.prefix_slora_expanded or batch.prefix_slora_prefix_len <= 0:
            return
        prefix_len = batch.prefix_slora_prefix_len
        old_max_len = batch.nopad_max_len_in_batch
        new_max_len = old_max_len + prefix_len
        new_b_loc = torch.empty(
            (len(batch), new_max_len + 12), dtype=torch.long, device="cuda"
        )
        new_b_seq_len = batch.nopad_b_seq_len + prefix_len
        new_b_start_loc = torch.zeros(len(batch), dtype=torch.int32, device="cuda")

        for i in range(len(batch)):
            query_len = int(batch.nopad_b_seq_len[i].item())
            old_start = old_max_len - query_len
            new_start = new_max_len - (prefix_len + query_len)
            new_b_loc[i, new_start:new_start + prefix_len] = batch.prefix_slora_b_loc[i]
            new_b_loc[i, new_start + prefix_len:new_start + prefix_len + query_len] = (
                batch.nopad_b_loc[i, old_start:old_start + query_len]
            )

        new_b_start_loc[1:] = torch.cumsum(new_b_seq_len, dim=0, dtype=torch.int32)[0:-1]
        batch.nopad_b_loc = new_b_loc
        batch.nopad_b_seq_len = new_b_seq_len
        batch.nopad_b_start_loc = new_b_start_loc
        batch.nopad_total_token_num = torch.sum(new_b_seq_len).item()
        batch.nopad_max_len_in_batch = new_max_len
        batch.prefix_slora_b_loc = None
        batch.prefix_slora_expanded = True
        return
    
    def forward(self, batch_id, is_prefill):
        batch: InferBatch = self.cache.pop(batch_id)
        # print(batch.requests)
        # print([req["request_id"] for req in batch.requests])
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": batch.nopad_total_token_num,
            "max_len_in_batch": batch.nopad_max_len_in_batch,
            "input_ids": batch.input_ids,
            "b_loc": batch.nopad_b_loc,
            "b_start_loc": batch.nopad_b_start_loc,
            "b_seq_len": batch.nopad_b_seq_len,
            "is_prefill": is_prefill
        }
        if is_prefill and batch.prefix_slora_prefix_len > 0:
            kwargs["prefix_slora_prefix_len"] = batch.prefix_slora_prefix_len
            kwargs["prefix_slora_b_loc"] = batch.prefix_slora_b_loc

        # assert False, f"{kwargs}"

        assert len(batch.adapter_dirs) == len(batch), "batch.adapter_dirs != batch"

        # always use lora batch infer
        if (self.input_params.no_lora or self.input_params.no_kernel or
            self.input_params.scheduler == "peft" or set(batch.adapter_dirs) == {None}):
            engine = self.model
        else:
            adapters = [self.adapters[self.adapter_id[adapter_dir]] for adapter_dir in batch.adapter_dirs]
            if self.input_params.no_lora_compute:
                engine = LoraUnorderedBatchInfer(self.model, adapters)
            elif self.input_params.bmm:
                torch.cuda.empty_cache()
                compressed_dirs = [batch.adapter_dirs[0]]
                adapter_sep = [0]
                cnt = 1
                for i in range(1, len(batch.adapter_dirs)):
                    if batch.adapter_dirs[i] == batch.adapter_dirs[i-1]:
                        cnt += 1
                    else:
                        compressed_dirs.append(batch.adapter_dirs[i])
                        adapter_sep.append(adapter_sep[-1] + cnt)
                        cnt = 1
                adapters = [self.adapters[self.adapter_id[adapter_dir]] for adapter_dir in compressed_dirs]
                engine = LoraBmmInfer(self.model, adapters, adapter_sep)
            else:
                engine = LoraUnorderedBatchInfer(self.model, adapters, infer_adapter=self.infer_adapter)
            kwargs["no_lora_compute"] = self.input_params.no_lora_compute
            # kwargs["no_lora_copy"] = self.input_params.no_lora_copy 

        logits = engine.forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, batch)
        if is_prefill and batch.prefix_slora_prefix_len > 0:
            self._expand_batch_for_prefix_slora(batch)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        output_dict = {}
        new_input_ids = []        
        for i, (r, all_input_ids, next_token_id, next_token_logprob) in enumerate(zip(batch.requests, batch.all_input_ids, next_token_ids, next_token_logprobs)):
            # all_input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long, device="cuda")
            all_input_ids.append(int(next_token_id))
            # all_input_ids_tensor = None
            new_input_ids.append(next_token_id)
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] += 1
            batch.out_token_id_counts[i][next_token_id] += 1
            metadata = {
                'id': int(next_token_id),
                'logprob': float(next_token_logprob),
            }
            output_dict[r['request_id']] = (int(next_token_id), metadata)
        
        batch.input_ids = torch.tensor(new_input_ids, dtype=torch.long).cuda()
        batch.nopad_b_start_loc = batch.nopad_b_start_loc + torch.arange(0, len(batch), dtype=torch.int32, device="cuda")
        batch.nopad_total_token_num += len(batch)
        batch.nopad_max_len_in_batch += 1
        batch.nopad_b_seq_len += 1
        self.cache[batch.batch_id] = batch
        return output_dict

    def _profile_adapter_prefill(self, adapter, batch_size, max_input_len):
        engine = LoraUnorderedBatchInfer(self.model, [adapter]*batch_size, infer_adapter=self.infer_adapter)
        self._profile_prefill(batch_size, max_input_len, adapter_engine=engine, rank_size=adapter.r)
    
    def _profile_prefill(self, batch_size, max_input_len, adapter_engine=None, rank_size=None):
        # warm up
        input_len = max_input_len
        test_data = np.vstack([np.arange(1, input_len+1) for _ in range(batch_size)])
        test_data = test_data.reshape(-1)
        test_data = torch.from_numpy(test_data).cuda()
        engine = self.model if adapter_engine is None else adapter_engine

        
        b_loc = torch.zeros(batch_size, input_len, dtype=torch.long, device="cuda")
        b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            b_loc[i, 0:input_len] = i * input_len + torch.arange(0, input_len, dtype=torch.int32, device="cuda")
            b_start_loc[i] = i * input_len
            b_seq_len[i] = input_len

        total_token_num = input_len * batch_size
        logics = engine.forward(batch_size, 
                                    total_token_num, 
                                    input_len, 
                                    test_data,
                                    b_loc,
                                    b_start_loc,
                                    b_seq_len,
                                    is_prefill=True)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        
        max_len_in_batch = input_len
        for i in range(batch_size):
            self.model.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
            
        b_loc = None
        b_start_loc = None
        b_seq_len = None
        
        import torch.distributed as dist
        dist.barrier()
        torch.cuda.synchronize()

        prefill_start_time = time.time()

        b_loc = torch.zeros(batch_size, input_len, dtype=torch.long, device="cuda")
        b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            b_start_loc[i] = i * input_len
            b_seq_len[i] = input_len

        total_token_num = batch_size * input_len
        logics = engine.forward(batch_size, total_token_num, input_len, test_data,
                                                    b_loc, b_start_loc, b_seq_len, is_prefill=True)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        torch.cuda.synchronize()
        if adapter_engine is None:
            self.base_prefill[batch_size][input_len] = time.time() - prefill_start_time
        else:
            self.adapter_prefill[rank_size][batch_size][input_len] = time.time() - prefill_start_time
        
        max_len_in_batch = input_len
        for i in range(batch_size):
            self.model.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
        
        return
    
    def exposed_profile_prefill(self):
        max_bs = self.model.mem_manager.tot_size // 2048
        print(max_bs)
        max_input_len = 1024
        self.base_prefill = defaultdict(dict)
        self.adapter_prefill = defaultdict(dict)
        for adapter in self.adapters:
            if adapter is None:
                continue
            if adapter.r in self.adapter_prefill:
                continue
            else:
                self.adapter_prefill[adapter.r] = defaultdict(dict)
                self.infer_adapter.load_adapters([adapter], prefetch=False)
                torch.cuda.synchronize()
                for bs in range(2, max_bs+1, 2):
                    for input_len in tqdm(range(32, max_input_len+1, 32), desc=f"profile prefill bs={bs}, adapter={adapter.r}"):
                        if bs not in self.base_prefill or input_len not in self.base_prefill[bs]:
                            self._profile_prefill(bs, input_len)
                        self._profile_adapter_prefill(adapter, bs, input_len)
                self.infer_adapter.offload_adapters([])
        return self.base_prefill, self.adapter_prefill

    def exposed_unmerge_adapter(self):
        print("len adapters:", len(self.infer_adapter.adapter_dirs))
        assert len(self.infer_adapter.adapter_dirs) == 1
        print("unmerge:", self.infer_adapter.adapter_dirs)
        engine = LoraPEFTBatchInfer(self.model, infer_adapter=self.infer_adapter)
        engine.unmerge_adapter()

    def exposed_merge_adapter(self):
        print("len adapters:", len(self.infer_adapter.adapter_dirs))
        assert len(self.infer_adapter.adapter_dirs) == 1
        print("merge:", self.infer_adapter.adapter_dirs)
        engine = LoraPEFTBatchInfer(self.model, infer_adapter=self.infer_adapter)
        engine.merge_adapter()


class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.world_size != 1
        if self.use_rpc:
            def async_wrap(f):
                f = rpyc.async_(f)
                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value
                return _func
            self._init_model = async_wrap(self.model.init_model)
            self._load_adapters = rpyc.async_(self.model.load_adapters)
            self._prefix_slora_prepare_batch = async_wrap(self.model.prefix_slora_prepare_batch)
            self._offload_adapters = rpyc.async_(self.model.offload_adapters)
            self._unmerge_adapter = rpyc.async_(self.model.unmerge_adapter)
            self._merge_adapter = rpyc.async_(self.model.merge_adapter)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)
            self._decode_batch = async_wrap(self.model.decode_batch)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
            self._profile_prefill = async_wrap(self.model.profile_prefill)
        else:
            self._init_model = self.model.exposed_init_model
            self._load_adapters = self.model.exposed_load_adapters
            self._prefix_slora_prepare_batch = self.model.exposed_prefix_slora_prepare_batch
            self._offload_adapters = self.model.exposed_offload_adapters
            self._merge_adapter = self.model.exposed_merge_adapter
            self._unmerge_adapter = self.model.exposed_unmerge_adapter
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch
            self._decode_batch = self.model.exposed_decode_batch
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
            self._profile_prefill = self.model.exposed_profile_prefill
        return

    async def init_model(self, rank_id, world_size, weight_dir, adapter_dirs,
                         max_total_token_num, load_way, mode, input_params,
			 prefetch_stream):
        ans : rpyc.AsyncResult = self._init_model(rank_id, world_size, weight_dir, adapter_dirs,
                                                  max_total_token_num, load_way, mode, input_params,
						  prefetch_stream)
        if self.use_rpc:
            await ans
            return
        else:
            return


    async def load_adapters(self, reqs, prefetch=False):
        self._load_adapters(reqs, prefetch=prefetch)


    async def prefix_slora_prepare_batch(self, adapter_dirs, extra_token_num=0):
        ans = self._prefix_slora_prepare_batch(adapter_dirs, extra_token_num)
        if self.use_rpc:
            return await ans
        else:
            return ans


    async def offload_adapters(self, reserved_reqs=None, prefetch=False):
        self._offload_adapters(reserved_reqs, prefetch=prefetch)
    
    async def unmerge_adapter(self):
        self._unmerge_adapter()
    
    async def merge_adapter(self):
        self._merge_adapter()


    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def prefill_batch(self, batch_id):
        ans = self._prefill_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def decode_batch(self, batch_id):
        ans = self._decode_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def filter_batch(self, batch_id, req_id_list):
        ans = self._filter_batch(batch_id, req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return 

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return
    
    async def profile_prefill(self):
        ans = self._profile_prefill()
        if self.use_rpc:
            return await ans
        else:
            return ans


def _init_env(port):
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, world_size):
    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(), world_size)
    
    import multiprocessing
    proc = multiprocessing.Process(target=_init_env, args=(port,))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)
