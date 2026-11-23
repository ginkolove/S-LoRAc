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
from slora.common.lowra_static import LowRAStaticPrefixStore
from slora.common.lowra_layer_runtime import LowRALayerRuntime, profile_interval
from slora.common.full_reprefill import (
    FULL_REPREFILL_MODES, FULL_PREFIX_CACHE_MODES, FullPrefixKVCache, FullReprefillRuntime,
)
from slora.common.full_reconstruct import FullReconstructPrefixKVCache, FullReconstructRuntime
from slora.common.lowra_retained import RetainedPrefixKVCache, RetainedLayerRuntime
from slora.common.lowra_adaptive import AdaptivePrefixKVCache, AdaptiveLayerRuntime
from slora.common.prefix_modes import FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_RETAINED, LAYER_ADAPTIVE, normalize_prefix_mode
from slora.common.prefix_modes import ACTIVE_REPREFILL, CPU_PIN_COUNTS, CPU_MODES
from slora.common.prefix_source_cache import ActivePrefixCache, HostPrefixBank
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
        self.input_params.lowra_prefix_mode = normalize_prefix_mode(
            getattr(input_params, "lowra_prefix_mode", FULL_REPREFILL))
        self.prefetch_stream = prefetch_stream
        self.lowra_prefix_cache = None
        self.lowra_runtime = None

        if input_params.enable_lowra and (
            input_params.bmm or input_params.no_kernel or input_params.no_lora or input_params.scheduler == "peft"
        ):
            raise ValueError("LowRA currently requires the LoraUnorderedBatchInfer path")

        self.cache = {}

        # Spawned workers must restore the length used by InferBatch's location table.
        max_req_total_len = int(input_params.max_req_total_len)
        if max_req_total_len <= 0:
            raise ValueError("max_req_total_len must be positive")
        setting["max_req_total_len"] = max_req_total_len

        # Spawned workers re-import module globals, so carry the port in RPC config.
        nccl_port = getattr(input_params, "nccl_port", setting["nccl_port"])
        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{nccl_port}', rank=rank_id, world_size=world_size)
        torch.cuda.set_device(rank_id)

        model_cfg = get_model_config(weight_dir, dummy=input_params.dummy)
        if input_params.enable_lowra:
            if mode or input_params.no_lora_compute or input_params.no_lora_swap:
                raise ValueError("LowRA profiling requires FP16 unified paging with real adapter computation")
            if model_cfg.get("num_key_value_heads", model_cfg["num_attention_heads"]) != model_cfg["num_attention_heads"]:
                raise ValueError("the current S-LoRA adapter packing supports MHA models only")

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "llama":
                if "num_key_value_heads" in model_cfg.keys():
                    self.model = Llama2TpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy,
                                                    dynamic_layer_workspace=(input_params.enable_lowra
                                                                             and input_params.lowra_prefix_mode not in FULL_PREFIX_CACHE_MODES))
                    
                else:
                    self.model = LlamaTpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy,
                                                    dynamic_layer_workspace=(input_params.enable_lowra
                                                                             and input_params.lowra_prefix_mode not in FULL_PREFIX_CACHE_MODES))
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            raise e

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
            if input_params.enable_lowra:
                raise ValueError("LowRA requires the Unified Paging memory pool; do not use --no-mem-pool")
            head_num = self.model.config["num_attention_heads"]
            self.infer_adapter = NaiveInferAdapter.init(self.model.config["num_hidden_layers"],
                                                        head_num,
                                                        self.model.config["hidden_size"] // head_num)
        else:
            self.infer_adapter = InferAdapter.init(self.model.mem_manager,
                                                   prefetch_stream)
        ''' finish init adapters '''

        if input_params.enable_lowra and input_params.lowra_shared_prefix_len > 0:
            if input_params.lowra_prefix_mode in FULL_REPREFILL_MODES:
                source = input_params.lowra_prefix_mode
                length, pool = input_params.lowra_shared_prefix_len, self.model.mem_manager
                bank = None
                if source in CPU_MODES:
                    token = input_params.cpu_prefix_token_id
                    if not 0 <= token < self.model.vocab_size:
                        raise ValueError("invalid CPU bank prefix token")
                    cap = input_params.cpu_prefix_pool_gb
                    bank = HostPrefixBank(pool, [token] * length, adapter_dirs,
                                          byte_cap=cap * 10**9 if cap is not None else None, mode=source,
                                          impress_read_percent=getattr(input_params, "impress_read_percent", None))
                    bank.check_capacity()
                    print("CPU prefix bank: %d logical adapters, %d physical entries, %d bytes/entry"
                          % (len(adapter_dirs), len(bank.physical_dirs), bank.entry_bytes), flush=True)
                if source == ACTIVE_REPREFILL or bank is not None:
                    fixed = [adapter_dirs[i] for i in input_params.cpu_fixed_hot_indices] if source in CPU_PIN_COUNTS else []
                    self.lowra_prefix_cache = ActivePrefixCache(length, pool, source, fixed, bank)
                else:
                    self.lowra_prefix_cache = FullPrefixKVCache(length, pool)
                self.lowra_runtime = FullReprefillRuntime(
                    input_params.lowra_shared_prefix_len, self.model.mem_manager,
                    self.lowra_prefix_cache)
                if bank is not None:
                    # Build each physical source once, outside replay. Runtime
                    # resets clear GPU prefixes but intentionally preserve DRAM.
                    for directory in tqdm(bank.physical_dirs, desc="Prefill CPU prefix bank"):
                        self.exposed_load_adapters([directory])
                        slots = pool.alloc(length)
                        try:
                            self._prefill_full_prefixes([directory], bank.prefix_ids, {directory: slots})
                            bank.capture(directory, slots)
                        finally:
                            pool.free(slots)
                            self.exposed_offload_adapters([])
                print("%s initialized: prefix_len=%s, static_token_equiv=0" % (source, length))
                set_random_seed(2147483647)
                return
            lora_rank_map = {
                adapter.lora_dir: adapter.r
                for adapter in self.adapters
                if adapter is not None
            }
            static_store = LowRAStaticPrefixStore(
                prefix_len=input_params.lowra_shared_prefix_len,
                lora_dirs=[adapter.lora_dir for adapter in self.adapters if adapter is not None],
                lora_ranks=lora_rank_map,
                layer_num=self.model.layers_num,
                tp_k_head_num=self.model.tp_k_head_num_,
                tp_v_head_num=self.model.tp_v_head_num_,
                head_dim=self.model.head_dim_,
            )
            static_store.bind_key_rope(
                self.model._cos_cached[:static_store.prefix_len],
                self.model._sin_cached[:static_store.prefix_len])
            if input_params.lowra_prefix_mode == FULL_RECONSTRUCT:
                self.lowra_prefix_cache = FullReconstructPrefixKVCache(
                    static_store.prefix_len, self.model.mem_manager,
                    reuse_next_batch=getattr(input_params, "reuse_next_reconstructed", False),
                    pinned_lora_dirs=list(lora_rank_map)[:getattr(input_params, "lowra_pin_hot_adapters", 0)])
                self.lowra_runtime = FullReconstructRuntime(
                    static_store, self.model.mem_manager, self.lowra_prefix_cache)
            elif input_params.lowra_prefix_mode == LAYER_ADAPTIVE:
                self.lowra_prefix_cache = AdaptivePrefixKVCache(static_store.prefix_len, self.model.mem_manager)
                self.lowra_runtime = AdaptiveLayerRuntime(
                    static_store, self.model.mem_manager, self.lowra_prefix_cache,
                    workspace_lora_limit=input_params.lowra_workspace_loras,
                    reconstruction_kernel=input_params.lowra_reconstruction_kernel,
                    reconstruction_workers=input_params.lowra_reconstruction_workers,
                    sample_interval=input_params.space_plan_config["interval"])
            elif input_params.lowra_prefix_mode == LAYER_RETAINED:
                self.lowra_prefix_cache = RetainedPrefixKVCache(
                    static_store.prefix_len, input_params.lowra_retain_layers, self.model.mem_manager,
                    retain_top_k=getattr(input_params, "lowra_retain_top_k", None),
                    pinned_lora_dirs=list(lora_rank_map)[:getattr(input_params, "lowra_pin_hot_adapters", 0)])
                self.lowra_runtime = RetainedLayerRuntime(
                    static_store, self.model.mem_manager, self.lowra_prefix_cache,
                    workspace_lora_limit=input_params.lowra_workspace_loras,
                    reconstruction_kernel=getattr(input_params, "lowra_reconstruction_kernel", "auto"),
                    reconstruction_workers=getattr(input_params, "lowra_reconstruction_workers", None))
                print("LowRA reconstruction: mode=%s kernel=%s workers=%s" % (
                    input_params.lowra_prefix_mode, self.lowra_runtime.reconstruction_kernel,
                    self.lowra_runtime.reconstruction_workers or "12*SMs (when bounded)"), flush=True)
            else:
                self.lowra_runtime = LowRALayerRuntime(static_store, self.model.mem_manager,
                                                      input_params.lowra_prefix_mode,
                                                      workspace_lora_limit=input_params.lowra_workspace_loras,
                                                      reconstruction_kernel=getattr(input_params, "lowra_reconstruction_kernel", "auto"),
                                                      reconstruction_workers=getattr(input_params, "lowra_reconstruction_workers", None))
                print("LowRA reconstruction: mode=%s kernel=%s workers=%s" % (
                    input_params.lowra_prefix_mode, self.lowra_runtime.reconstruction_kernel,
                    self.lowra_runtime.prefetch_workers or "12*SMs (when bounded)"), flush=True)
            print(
                f"LowRA static store initialized: prefix_len={static_store.prefix_len}, "
                f"total_rank={static_store.total_rank}, "
                f"static_token_equiv={getattr(input_params, 'lowra_static_token_num', 0)}"
            )
        
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
    def exposed_offload_adapters(self, reserve_dirs=None, prefetch=False):
        if not self.input_params.bmm:
            self.infer_adapter.offload_adapters(reserve_dirs if reserve_dirs is not None else [])
        else:
            reserve_dirs = reserve_dirs if reserve_dirs is not None else []
            for adapter_dir, id in self.adapter_id.items():
                if adapter_dir is not None and adapter_dir not in reserve_dirs:
                    self.adapters[id].offload_from_gpu()

    def exposed_lowra_prefix_cache_state(self):
        if self.lowra_prefix_cache is None:
            return None
        state = self.lowra_prefix_cache.state_dict()
        if isinstance(self.lowra_runtime, AdaptiveLayerRuntime):
            self.lowra_runtime.collect_observations()
            state["adaptive_profile"] = list(self.lowra_runtime.profile_records)
        return state

    def exposed_apply_space_plan(self, layers, evictions):
        if not isinstance(self.lowra_prefix_cache, AdaptivePrefixKVCache):
            raise ValueError("space plans require minikv-adaptive")
        self.lowra_prefix_cache.apply_plan(dict(layers), list(evictions))

    def exposed_evict_lowra_prefixes(self, lora_dirs):
        if self.lowra_prefix_cache is None:
            return
        self.lowra_prefix_cache.evict_many(lora_dirs)

    def exposed_lowra_runtime_stats(self):
        from slora.common.lowra_runtime_stats import snapshot
        return snapshot(self)

    def exposed_reset_lowra_runtime(self):
        stats = {
            "free_slots_before": self.model.mem_manager.can_use_mem_size,
            "total_slots": self.model.mem_manager.tot_size,
        }
        if self.lowra_prefix_cache is not None:
            stats["prefix_access_before"] = self.lowra_prefix_cache.access_stats()
            stats["prefix_cache_before"] = self.lowra_prefix_cache.state_dict()
            stats["prefix_entries_cleared"] = self.lowra_prefix_cache.clear_metadata_if_inactive()
        if hasattr(self, "infer_adapter") and self.infer_adapter is not None:
            stats["adapter_entries_offloaded"] = len(getattr(self.infer_adapter, "adapter_dirs", []))
            self.infer_adapter.offload_adapters([])
        self.model.mem_manager.free_all()
        if self.lowra_prefix_cache is not None:
            stats["prefix_cache_after"] = self.lowra_prefix_cache.state_dict()
            self.lowra_prefix_cache.reset_access_stats()
        torch.cuda.reset_peak_memory_stats()
        stats["free_slots_after"] = self.model.mem_manager.can_use_mem_size
        return stats


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
        if self.lowra_runtime is not None:
            batch_data.lowra_prefix_len = (self.lowra_runtime.prefix_len
                                           if isinstance(self.lowra_runtime, FullReprefillRuntime)
                                           else self.lowra_runtime.static_store.prefix_len)
        if self.lowra_prefix_cache is not None:
            if isinstance(self.lowra_runtime, FullReconstructRuntime):
                self.lowra_runtime.acquire_many(reqs, self.infer_adapter)
            elif isinstance(self.lowra_prefix_cache, FullPrefixKVCache):
                self.lowra_prefix_cache.acquire_many(reqs, self._prefill_full_prefixes)
            else:
                self.lowra_prefix_cache.acquire_many(
                    batch_data.adapter_dirs,
                    infer_adapter=self.infer_adapter,
                )
        self.cache[batch_id] = batch_data
        return

    @torch.no_grad()
    def _prefill_full_prefixes(self, lora_dirs, prefix_ids, prefix_indices):
        requests = [dict(request_id="prefix-%s" % index, adapter_dir=directory,
                         input_id=list(prefix_ids), sampling_param={})
                    for index, directory in enumerate(lora_dirs)]
        prefix_batch = InferBatch.init_batch(
            "full-prefix-build", requests, torch.float16, torch.cuda.current_device(),
            self.model.mem_manager, self.model.vocab_size)
        adapters = [self.adapters[self.adapter_id[directory]] for directory in lora_dirs]
        engine = LoraUnorderedBatchInfer(self.model, adapters, infer_adapter=self.infer_adapter)
        # Cache owns these slots from the start; prefill must not allocate a second copy.
        with profile_interval(self.lowra_runtime.profiler, "prefix_prefill", -1):
            engine.forward(
                batch_size=len(prefix_batch), total_token_num=prefix_batch.nopad_total_token_num,
                max_len_in_batch=prefix_batch.nopad_max_len_in_batch, input_ids=prefix_batch.input_ids,
                b_loc=prefix_batch.nopad_b_loc, b_start_loc=prefix_batch.nopad_b_start_loc,
                b_seq_len=prefix_batch.nopad_b_seq_len, is_prefill=True,
                prefill_mem_index=torch.cat([prefix_indices[directory] for directory in lora_dirs]),
                return_logits=False)
    
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
        if self.lowra_prefix_cache is not None:
            keep_ids = set(req_id_list)
            self.lowra_prefix_cache.release_many(
                req["adapter_dir"] for req in batch.requests if req["request_id"] not in keep_ids
            )
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
        if self.lowra_prefix_cache is not None:
            self.lowra_prefix_cache.release_many(req["adapter_dir"] for req in batch.requests)
        batch.free_self()
        del batch
        # torch.cuda.empty_cache()
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
            if self.lowra_runtime is not None:
                if isinstance(self.lowra_runtime, AdaptiveLayerRuntime):
                    self.lowra_runtime.start_observation(batch.adapter_dirs, batch.nopad_total_token_num, is_prefill)
                prefix_by_dir = ({directory: entry.prefix_indices
                                  for directory, entry in self.lowra_prefix_cache.entries.items()}
                                 if self.lowra_prefix_cache is not None else None)
                frame = self.lowra_runtime.prepare(
                    batch.adapter_dirs, self.infer_adapter, prefix_by_dir,
                    reserve_tokens=batch.nopad_total_token_num if is_prefill else len(batch))
                kwargs["lowra_frame"] = frame
                if is_prefill:
                    kwargs["lowra_prefix_b_loc"] = frame.prefix_index_table(0)
                    kwargs["lowra_prefix_len"] = batch.lowra_prefix_len
            # kwargs["no_lora_copy"] = self.input_params.no_lora_copy 

        try:
            logits = engine.forward(**kwargs)
        finally:
            frame = kwargs.get("lowra_frame")
            if frame is not None:
                frame.finish()
            if isinstance(self.lowra_runtime, AdaptiveLayerRuntime):
                self.lowra_runtime.end_observation()
        next_token_ids, next_token_probs = sample(logits, batch)
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
        # b_loc always owns private query/decode KV only; prefix addresses are per-forward.

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
            self._load_adapters = async_wrap(self.model.load_adapters)
            self._offload_adapters = async_wrap(self.model.offload_adapters)
            self._unmerge_adapter = rpyc.async_(self.model.unmerge_adapter)
            self._merge_adapter = rpyc.async_(self.model.merge_adapter)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)
            self._decode_batch = async_wrap(self.model.decode_batch)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
            self._profile_prefill = async_wrap(self.model.profile_prefill)
            self._lowra_prefix_cache_state = async_wrap(self.model.lowra_prefix_cache_state)
            self._evict_lowra_prefixes = async_wrap(self.model.evict_lowra_prefixes)
            self._reset_lowra_runtime = async_wrap(self.model.reset_lowra_runtime)
            self._lowra_runtime_stats = async_wrap(self.model.lowra_runtime_stats)
            self._apply_space_plan = async_wrap(self.model.apply_space_plan)
        else:
            self._init_model = self.model.exposed_init_model
            self._load_adapters = self.model.exposed_load_adapters
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
            self._lowra_prefix_cache_state = self.model.exposed_lowra_prefix_cache_state
            self._evict_lowra_prefixes = self.model.exposed_evict_lowra_prefixes
            self._reset_lowra_runtime = self.model.exposed_reset_lowra_runtime
            self._lowra_runtime_stats = self.model.exposed_lowra_runtime_stats
            self._apply_space_plan = getattr(self.model, "exposed_apply_space_plan", None)
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
        ans = self._load_adapters(reqs, prefetch=prefetch)
        if self.use_rpc:
            await ans


    async def offload_adapters(self, reserved_reqs=None, prefetch=False):
        ans = self._offload_adapters(reserved_reqs, prefetch=prefetch)
        if self.use_rpc:
            await ans
    
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

    async def lowra_prefix_cache_state(self):
        ans = self._lowra_prefix_cache_state()
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def evict_lowra_prefixes(self, lora_dirs):
        ans = self._evict_lowra_prefixes(lora_dirs)
        if self.use_rpc:
            await ans

    async def reset_lowra_runtime(self):
        ans = self._reset_lowra_runtime()
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def lowra_runtime_stats(self):
        ans = self._lowra_runtime_stats()
        return await ans if self.use_rpc else ans

    async def apply_space_plan(self, layers, evictions):
        ans = self._apply_space_plan(layers, evictions)
        return await ans if self.use_rpc else ans


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
