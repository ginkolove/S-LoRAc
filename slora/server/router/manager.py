import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import json
import pickle
import time
import torch
import zmq
import zmq.asyncio
from typing import Dict, List, Optional

from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from slora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq, LowRAResetReq, LowRAResetAck
from .stats import Stats

from slora.server.input_params import InputParams
from slora.models.peft.lora_adapter import get_lora_config
from slora.common.lowra_static import LowRAStaticPrefixStore
from slora.utils.model_utils import get_model_config
from slora.server.router.profiler import AlphaModel, BetaModel
from slora.server.router.abort_req_queue import AbortReqQueue
from slora.server.router.cluster_req_queue import ClusterReqQueue
from slora.server.router.vtc_req_queue import VTCReqQueue
from slora.server.router.pets_req_queue import PETSReqQueue
from slora.server.router.peft_req_queue import PEFTReqQueue
from slora.server.router.lowra_fifo_req_queue import LowRAFIFOReqQueue
from slora.server.router.full_reprefill_req_queue import FullReprefillReqQueue
from slora.server.router.online_req_queue import OnlineReqQueue
from slora.common.full_reprefill import FULL_REPREFILL_MODES, FULL_PREFIX_CACHE_MODES
from slora.common.prefix_modes import FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_RETAINED, normalize_prefix_mode
from slora.common.service_cycles import ServiceCycleRecorder


def get_scheduler(input_params, adapter_dirs):
    if getattr(input_params, "scheduler", "slora") == "online":
        return OnlineReqQueue(
            input_params.max_total_token_num, input_params.batch_max_tokens,
            input_params.running_max_req_size,
            prefix_len=input_params.lowra_shared_prefix_len if input_params.enable_lowra else 0,
            prefix_mode=input_params.lowra_prefix_mode, layer_num=input_params.lowra_layer_num,
            retain_layers=input_params.lowra_retain_layers,
            retain_top_k=input_params.lowra_retain_top_k,
            pinned_lora_dirs=adapter_dirs[:input_params.lowra_pin_hot_adapters],
            max_active_loras=input_params.max_active_loras,
            workspace_lora_limit=input_params.lowra_workspace_loras)
    if input_params.enable_lowra:
        prefix_mode = normalize_prefix_mode(input_params.lowra_prefix_mode)
        if prefix_mode in FULL_PREFIX_CACHE_MODES:
            return FullReprefillReqQueue(input_params.max_total_token_num,
                                        input_params.batch_max_tokens, input_params.running_max_req_size,
                                        input_params.lowra_shared_prefix_len)
        return LowRAFIFOReqQueue(input_params.max_total_token_num,
                                input_params.batch_max_tokens, input_params.running_max_req_size,
                                input_params.lowra_shared_prefix_len, prefix_mode,
                                input_params.lowra_layer_num, input_params.lowra_workspace_loras,
                                getattr(input_params, "lowra_retain_layers", 0),
                                getattr(input_params, "lowra_retain_top_k", None),
                                adapter_dirs[:getattr(input_params, "lowra_pin_hot_adapters", 0)])
    if input_params.scheduler == "vtc_fair":
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")


class RouterManager:

    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params
        if input_params.scheduler == "online" and mode:
            raise ValueError("online scheduling currently requires the FP16 Unified Paging path")
        self.input_params.lowra_prefix_mode = normalize_prefix_mode(
            getattr(input_params, "lowra_prefix_mode", FULL_REPREFILL))
        retain_layers = getattr(input_params, "lowra_retain_layers", 0)
        retain_top_k = getattr(input_params, "lowra_retain_top_k", None)
        pin_count = getattr(input_params, "lowra_pin_hot_adapters", 0)
        if pin_count < 0 or pin_count > len(adapter_dirs):
            raise ValueError("--lowra-pin-hot-adapters must be between zero and the configured adapter count")
        if pin_count and (not input_params.enable_lowra or input_params.lowra_prefix_mode not in
                          (FULL_RECONSTRUCT, LAYER_RETAINED) or retain_top_k is not None):
            raise ValueError("pinned hot prefixes require a MiniKV cache mode and no dynamic top-K")
        if pin_count and input_params.lowra_prefix_mode == LAYER_RETAINED and retain_layers <= 0:
            raise ValueError("pinned layer retention requires positive --lowra-retain-layers")
        if retain_top_k is not None and (retain_top_k < 0 or input_params.lowra_prefix_mode != LAYER_RETAINED):
            raise ValueError("--lowra-retain-top-k requires minikv-layer_retained and a nonnegative count")
        if retain_layers < 0 or (retain_layers and input_params.lowra_prefix_mode != LAYER_RETAINED):
            raise ValueError("--lowra-retain-layers requires minikv-layer_retained and a nonnegative count")

        if input_params.enable_lowra and (input_params.lowra_shared_prefix_len <= 0 or not adapter_dirs):
            raise ValueError("LowRA requires a positive shared prefix length and at least one adapter")

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[None] = 0

        if (self.input_params.enable_lowra and self.input_params.lowra_shared_prefix_len > 0
                and self.input_params.lowra_prefix_mode not in FULL_REPREFILL_MODES):
            model_cfg = get_model_config(weightdir, dummy=input_params.dummy)
            self.input_params.lowra_layer_num = int(model_cfg["num_hidden_layers"])
            if retain_layers > self.input_params.lowra_layer_num:
                raise ValueError("--lowra-retain-layers exceeds the model layer count")
            static_tokens = LowRAStaticPrefixStore.token_equivalent_static_size_from_config(
                self.input_params.lowra_shared_prefix_len,
                [self.lora_ranks[lora_dir] for lora_dir in adapter_dirs],
                model_cfg,
                world_size,
            )
            self.input_params.lowra_static_token_num = static_tokens
            self.input_params.max_total_token_num -= static_tokens
            if self.input_params.max_total_token_num <= 0:
                raise ValueError(
                    f"LowRA static region consumes {static_tokens} token-equivalent entries, "
                    "leaving no dynamic Unified Paging budget"
                )
            print(
                f"LowRA static region: {static_tokens} token-equivalent entries, "
                f"dynamic Unified Paging budget: {self.input_params.max_total_token_num}"
            )
        else:
            self.input_params.lowra_static_token_num = 0
        if self.input_params.enable_lowra:
            if self.input_params.pool_size_lora:
                raise ValueError("LowRA profiling counts adapters in the total budget; use --pool-size-lora 0")
            model_cfg = get_model_config(weightdir, dummy=input_params.dummy)
            max_sequence_length = model_cfg.get("max_sequence_length", model_cfg.get("max_position_embeddings", self.input_params.max_req_total_len))
            self.input_params.lowra_window_n = max(1, self.input_params.lowra_original_total_token_num // int(max_sequence_length))
            if self.input_params.scheduler not in ("slora", "online") or self.input_params.lowra_wait_slo_ms > 0:
                raise ValueError("LowRA layer profiling uses the default S-LoRA FIFO scheduler, without SLO")
            if self.input_params.prefetch:
                raise ValueError("disable adapter prefetch when profiling layer reconstruction")
            if self.input_params.lowra_prefix_mode in FULL_REPREFILL_MODES:
                print("Full-reprefill: no static region; full prefixes, adapters and request KV "
                      "share the original Unified Paging pool")
            elif self.input_params.lowra_prefix_mode == FULL_RECONSTRUCT:
                print("minikv-full_reconstruct: static BaseKV/MiniKV; complete prefix cache, adapters "
                      "and request KV share Unified Paging; no single-layer workspace")
            elif self.input_params.lowra_prefix_mode == LAYER_RETAINED:
                print("minikv-layer_retained: keep first %s layers for %s; "
                      "other prefixes/layers overwrite one workspace; both charge Unified Paging" %
                      (retain_layers, "fixed first %s configured LoRAs until reset" % pin_count if pin_count
                       else "all active LoRAs" if retain_top_k is None
                       else "top %s LoRAs by active request count" % retain_top_k))
            else:
                requested_limit = self.input_params.lowra_workspace_loras
                if requested_limit is not None and requested_limit <= 0:
                    raise ValueError("workspace LoRA limit must be positive")
                print("LowRA single-layer prefix workspace: allocated on demand from Unified Paging; "
                      "no startup reservation")
            if pin_count:
                print("Pinned hot prefix adapters (lazy build, protected until reset): %s" % adapter_dirs[:pin_count])

        self.req_queue = get_scheduler(input_params, adapter_dirs)
        if input_params.scheduler == "online":
            print("Online FCFS: output lengths are not used for admission; "
                  "decode-step memory checks with preempt/recompute; admission interval=%s" %
                  input_params.schedule_interval, flush=True)

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        self.online_preemptions = 0
        self.online_preempted_request_ids = set()
        self.online_recomputed_tokens = 0
        self.online_rejections = 0
        self.preemption_stats_file = None
        self.runtime_stats_file = None
        self.runtime_stats_interval = 1.0
        self.runtime_stats_last_time = 0.0
        self.service_cycles = None
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)


    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                ))

        await asyncio.gather(*init_model_ret)
        return
    
    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        cache_dir = os.path.expanduser("~/.cache/slora")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir+"/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return


    def add_req(
        self,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str,
        lowra_lengths=None,
    ):
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params, lowra_lengths)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
                if self.service_cycles is not None:
                    self.service_cycles.complete(request_id, aborted=True)
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            if self.service_cycles is not None:
                self.service_cycles.observe_waiting(len(self.req_queue.waiting_req_list))
            await self._record_runtime_stats()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:", self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num)
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _record_runtime_stats(self):
        if self.runtime_stats_file is None:
            return
        now = time.monotonic()
        if now - self.runtime_stats_last_time < self.runtime_stats_interval:
            return
        self.runtime_stats_last_time = now
        ranks = await asyncio.gather(*[rpc.lowra_runtime_stats() for rpc in self.model_rpcs])
        if self.world_size != 1:
            ranks = [obtain(rank) for rank in ranks]
        running = self.running_batch.reqs if self.running_batch else []
        waiting = [req for req in self.req_queue.waiting_req_list if not req.aborted]
        row = dict(monotonic_time=time.monotonic(), wall_time=time.time(),
                   scheduler=self.input_params.scheduler,
                   online_preemptions=self.online_preemptions,
                   online_preempted_requests=len(self.online_preempted_request_ids),
                   online_recomputed_tokens=self.online_recomputed_tokens,
                   online_rejections=self.online_rejections,
                   prefix_mode=self.input_params.lowra_prefix_mode,
                   running_batch_size=len(running),
                   running_unique_loras=len({req.adapter_dir for req in running}),
                   waiting_queue_size=len(waiting),
                   oldest_wait_seconds=max([time.time() - req.arrival_time for req in waiting], default=0.0),
                   total_budget_tokens=self.input_params.lowra_original_total_token_num,
                   static_token_equivalent=self.input_params.lowra_static_token_num,
                   running_max_req_size=self.input_params.running_max_req_size,
                   batch_max_tokens=self.input_params.batch_max_tokens,
                   last_prefill_query_tokens=getattr(self, "last_prefill_query_tokens", 0),
                   last_prefill_tokens=getattr(self, "last_prefill_tokens", 0),
                   last_admission_tokens=getattr(self, "last_admission_tokens", 0),
                   last_prefix_reconstruct_tokens=getattr(self, "last_prefix_reconstruct_tokens", 0),
                   last_prefill_batch_size=getattr(self, "last_prefill_batch_size", 0),
                   per_rank=ranks)
        self.runtime_stats_file.write(json.dumps(row) + "\n")
        self.runtime_stats_file.flush()

    async def _step(self):
        """
        事件处理循环
        """
        if self.input_params.scheduler == "online":
            return await self._step_online()
        # 删除所有已经 finished 的 req
        if self.running_batch is None:
            cycle_start = time.monotonic()
            await self._sync_lowra_prefix_cache_state()
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_batch is not None:
                if self.service_cycles is not None:
                    self.service_cycles.begin(cycle_start)
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch

                await self._evict_lowra_prefixes_for_batch(new_batch)
                if not self.input_params.no_lora:
                    # load adapters
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                # merge adapter to base model
                if self.input_params.scheduler == "peft":
                    torch.cuda.synchronize()
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].merge_adapter())
                    await asyncio.gather(*ret)
            
                torch.cuda.synchronize()
                await self._prefill_batch(self.running_batch)
                await self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            # prefetch
            if (not self.input_params.no_lora and
                self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                next_batch = self.req_queue.next_batch()
                if next_batch is not None:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(
                            next_batch.adapter_dirs, prefetch=True))
                    await asyncio.gather(*ret)
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()

            self.has_wait_tokens += 1
            return
        else:
            await self._sync_lowra_prefix_cache_state()
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)

                await self._evict_lowra_prefixes_for_batch(new_mini_batch)
                if not self.input_params.no_lora:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                await self._prefill_batch(new_mini_batch, minibatch=True)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
        
    async def _step_online(self):
        # Aborts must release state even before another token can be generated.
        self.req_queue.waiting_req_list = [req for req in self.req_queue.waiting_req_list if not req.aborted]
        if self.running_batch and any(req.aborted for req in self.running_batch.reqs):
            await self._handle_finish_req(self.running_batch, True)
            await self._filter_runing_batch()

        await self._sync_lowra_prefix_cache_state()
        preempted = False
        while self.running_batch is not None:
            plan = self.req_queue.plan(self.running_batch.reqs, [], self.lora_ranks)
            if plan.fits:
                await self._evict_online_prefixes(plan.evictions)
                break
            # No allocation happens until a feasible decode plan exists.
            victim = self.running_batch.reqs[-1]
            if len(self.running_batch.reqs) == 1:
                # Even after all unpinned idle prefixes are reclaimed, this
                # request cannot advance. Reject rather than loop or crash.
                self._reject_online_request(victim, "request KV exceeds the dynamic pool capacity")
                await self._handle_finish_req(self.running_batch, True)
                await self._filter_runing_batch()
            else:
                await self._preempt_online_request(victim)
                preempted = True
            await self._sync_lowra_prefix_cache_state()

        interval = self.input_params.schedule_interval
        if not preempted and (self.running_batch is None or self.has_wait_tokens >= interval):
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            while new_batch is None and self.running_batch is None and self.req_queue.waiting_req_list:
                victim = self.req_queue.waiting_req_list.pop(0)
                self._reject_online_request(victim, "known prompt/history cannot fit online admission budgets")
                new_batch = self.req_queue.generate_new_batch(None, self.lora_ranks)
            if new_batch is not None:
                if self.service_cycles is not None and self.service_cycles.current is None:
                    self.service_cycles.begin()
                await self._evict_lowra_prefixes_for_batch(new_batch)
                running_dirs = self.running_batch.adapter_dirs if self.running_batch else set()
                if not self.input_params.no_lora:
                    await asyncio.gather(*[rpc.offload_adapters(running_dirs | new_batch.adapter_dirs)
                                           for rpc in self.model_rpcs])
                    await asyncio.gather(*[rpc.load_adapters(new_batch.adapter_dirs) for rpc in self.model_rpcs])
                self.online_recomputed_tokens += sum(
                    req.input_len + len(req.output_ids) for req in new_batch.reqs if req.preemptions)
                resumed = [dict(req_id=req.request_id, input_tokens=req.input_len,
                                generated_tokens=len(req.output_ids), preemptions=req.preemptions)
                           for req in new_batch.reqs if req.preemptions]
                prefill_size = len(new_batch.reqs)
                self.stats_tool.count_prompt_tokens(new_batch)
                prefill_start = time.monotonic()
                await self._prefill_batch(new_batch, minibatch=True)
                if resumed:
                    # This is the entire resumed/mixed batch, not isolated recompute cost.
                    self._record_preemption_event(
                        "resume_prefill", requests=resumed, batch_size=prefill_size,
                        new_requests=prefill_size - len(resumed),
                        prefill_wall_seconds=time.monotonic() - prefill_start)
                if self.running_batch is None:
                    self.running_batch = new_batch
                elif not new_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_batch)
                    self.running_batch.merge(new_batch)
                await self._filter_runing_batch()
                if self.running_batch is not None and not self.input_params.no_lora:
                    await asyncio.gather(*[rpc.offload_adapters(self.running_batch.adapter_dirs)
                                           for rpc in self.model_rpcs])
                self.has_wait_tokens = 0
                return
        if self.running_batch is not None:
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()
            self.has_wait_tokens += 1
        elif self.service_cycles is not None and self.service_cycles.current is not None:
            self.service_cycles.finish(len(self.req_queue.waiting_req_list))

    async def _evict_online_prefixes(self, directories):
        if directories:
            await asyncio.gather(*[rpc.evict_lowra_prefixes(directories) for rpc in self.model_rpcs])
            await self._sync_lowra_prefix_cache_state()

    def _reject_online_request(self, req, reason):
        req.aborted = req.has_generate_finished = True
        self.online_rejections += 1
        print("online admission rejected %s: %s" % (req.request_id, reason), flush=True)
        self.send_to_detokenization.send_pyobj(BatchAbortReq([req.request_id]))
        if self.service_cycles is not None:
            self.service_cycles.complete(req.request_id, aborted=True)

    async def _preempt_online_request(self, req):
        release_start = time.monotonic()
        batch = self.running_batch
        survivors = [r for r in batch.reqs if r.request_id != req.request_id]
        # Reuse the existing filter/remove lifecycle to free private KV and
        # release precisely one prefix reference, without marking completion.
        if survivors:
            await asyncio.gather(*[rpc.filter_batch(batch.batch_id, [r.request_id for r in survivors])
                                   for rpc in self.model_rpcs])
            batch.reqs = survivors
            batch.id_to_reqs = {r.request_id: r for r in survivors}
            batch.adapter_dirs = {r.adapter_dir for r in survivors}
        else:
            await self._remove_batch(batch)
            self.running_batch = None
        if not self.input_params.no_lora:
            await asyncio.gather(*[rpc.offload_adapters({r.adapter_dir for r in survivors})
                                   for rpc in self.model_rpcs])
        req.preemptions += 1
        self.online_preemptions += 1
        self.online_preempted_request_ids.add(req.request_id)
        self.req_queue.waiting_req_list.insert(0, req)
        self._record_preemption_event(
            "preempt", req_id=req.request_id, adapter_dir=req.adapter_dir,
            preemptions=req.preemptions, input_tokens=req.input_len,
            generated_tokens=len(req.output_ids),
            release_wall_seconds=time.monotonic() - release_start)

    def _record_preemption_event(self, event, **fields):
        if self.preemption_stats_file is not None:
            row = dict(event=event, monotonic_time=time.monotonic(), **fields)
            self.preemption_stats_file.write(json.dumps(row) + "\n")
            self.preemption_stats_file.flush()

    async def _sync_lowra_prefix_cache_state(self):
        if not hasattr(self.req_queue, "update_prefix_cache_state"):
            return
        if not hasattr(self, "model_rpcs"):
            return
        states = await asyncio.gather(*[
            self.model_rpcs[tp_rank].lowra_prefix_cache_state()
            for tp_rank in range(self.world_size)
        ])
        self.req_queue.update_prefix_cache_state(obtain(states[0]) if self.world_size != 1 else states[0])

    async def _evict_lowra_prefixes_for_batch(self, batch):
        evict_lora_dirs = getattr(batch, "lowra_evict_prefix_lora_dirs", [])
        if not evict_lora_dirs:
            return
        await asyncio.gather(*[
            self.model_rpcs[tp_rank].evict_lowra_prefixes(evict_lora_dirs)
            for tp_rank in range(self.world_size)
        ])

    async def _reset_lowra_runtime(self, request_id):
        if not self.input_params.enable_lowra:
            self.send_to_detokenization.send_pyobj(
                LowRAResetAck(request_id, False, "LowRA is not enabled")
            )
            return
        if self.running_batch is not None or len(self.req_queue.waiting_req_list) > 0:
            self.send_to_detokenization.send_pyobj(
                LowRAResetAck(
                    request_id,
                    False,
                    "LowRA runtime reset requires an idle server",
                    {
                        "running_batch_size": 0 if self.running_batch is None else len(self.running_batch.reqs),
                        "waiting_queue_size": len(self.req_queue.waiting_req_list),
                    },
                )
            )
            return
        try:
            stats_list = await asyncio.gather(*[
                self.model_rpcs[tp_rank].reset_lowra_runtime()
                for tp_rank in range(self.world_size)
            ])
            await self._sync_lowra_prefix_cache_state()
            if self.service_cycles is not None:
                self.service_cycles.reset()
            self.online_preemptions = self.online_recomputed_tokens = self.online_rejections = 0
            self.online_preempted_request_ids.clear()
            self.send_to_detokenization.send_pyobj(
                LowRAResetAck(request_id, True, "LowRA runtime reset complete", {"per_rank": stats_list})
            )
        except Exception as exc:
            self.send_to_detokenization.send_pyobj(
                LowRAResetAck(request_id, False, repr(exc))
            )

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch, minibatch=True):
        if self.service_cycles is not None:
            self.service_cycles.admit((req.request_id for req in batch.reqs),
                                      len(self.req_queue.waiting_req_list))
        self.last_prefill_query_tokens = batch.input_tokens()
        self.last_prefill_tokens = getattr(batch, "prefill_token_num", batch.input_tokens())
        self.last_admission_tokens = self.last_prefill_tokens
        self.last_prefix_reconstruct_tokens = 0
        if self.input_params.lowra_prefix_mode == FULL_RECONSTRUCT:
            self.last_prefix_reconstruct_tokens = getattr(batch, "prefix_prefill_tokens", 0)
            self.last_prefill_tokens = batch.input_tokens()
        self.last_prefill_batch_size = len(batch.reqs)
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        return

    async def _decode_batch(self, batch:Batch):
        self.req_queue.update_counter(batch)
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _filter_batch(self, batch: Batch):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        if has_new_finished_req:
            if self.service_cycles is not None:
                for req in batch.reqs:
                    if req.has_generate_finished:
                        self.service_cycles.complete(req.request_id, req.aborted)
            batch.filter_finished()

            # unmerge adapter from base model
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)

            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch)
        return

    async def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            if not self.input_params.no_lora:
                # offload model and adapters
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters())
                await asyncio.gather(*ret)

            if self.service_cycles is not None and not (
                    self.input_params.scheduler == "online" and any(
                        req.preemptions and not req.aborted for req in self.req_queue.waiting_req_list)):
                self.service_cycles.finish(len(self.req_queue.waiting_req_list))
            self.running_batch = None
            return
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        if self.service_cycles is not None:
            self.service_cycles.tokens(len(req_ans))
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
    
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 5:
                adapter_dir, prompt_ids, sampling_params, request_id, lowra_lengths = recv_req
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id, lowra_lengths)
            elif isinstance(recv_req, tuple) and len(recv_req) == 4:
                adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            elif isinstance(recv_req, LowRAResetReq):
                await self._reset_lowra_runtime(recv_req.req_id)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        processes = [rpc.rpc_server_process for rpc in getattr(self, "model_rpcs", [])
                     if rpc.rpc_server_process is not None]
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               schedule_interval=args.schedule_interval,
                               max_active_loras=args.max_active_loras,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                               enable_lowra=args.enable_lowra,
                               lowra_shared_prefix_len=args.lowra_shared_prefix_len,
                               lowra_wait_slo_ms=args.lowra_wait_slo_ms,
                               lowra_prefix_mode=args.lowra_prefix_mode,
                               lowra_reconstruction_kernel=args.lowra_reconstruction_kernel,
                               lowra_reconstruction_workers=args.lowra_reconstruction_workers,
                               lowra_workspace_loras=args.lowra_workspace_loras,
                               lowra_retain_layers=args.lowra_retain_layers,
                               lowra_retain_top_k=args.lowra_retain_top_k,
                               lowra_pin_hot_adapters=args.lowra_pin_hot_adapters,
                               nccl_port=args.nccl_port,
                              )

    try:
        if args.model_seed is not None:
            from slora.utils.infer_utils import set_random_seed
            set_random_seed(args.model_seed)
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval,
        )
    
        asyncio.run(router.wait_to_model_ready())
        if args.lowra_runtime_stats_path:
            directory = os.path.dirname(os.path.abspath(args.lowra_runtime_stats_path))
            os.makedirs(directory, exist_ok=True)
            router.runtime_stats_file = open(args.lowra_runtime_stats_path, "a", buffering=1)
            router.runtime_stats_interval = args.lowra_runtime_stats_interval
            cycle_path = os.path.splitext(args.lowra_runtime_stats_path)[0] + ".cycles.jsonl"
            router.service_cycles = ServiceCycleRecorder(open(cycle_path, "a", buffering=1))
            preemption_path = os.path.splitext(args.lowra_runtime_stats_path)[0] + ".preemptions.jsonl"
            router.preemption_stats_file = open(preemption_path, "a", buffering=1)
        if input_params.profile:
            asyncio.run(router.profile_prefill())
        if input_params.scheduler == "pets" and input_params.profile:
            router.req_queue.alpha = router.alpha_model
            router.req_queue.beta = router.beta_model
        elif input_params.scheduler == "pets":
            # loading from file
            cache_dir = os.path.expanduser("~/.cache/slora")
            router.req_queue.alpha = AlphaModel.from_file(cache_dir+"/profile_results.pkl")
            router.req_queue.beta = BetaModel.from_file(cache_dir+"/profile_results.pkl")
    
    except Exception as e:
        import traceback
        err_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        pipe_writer.send(err_str)
        if "router" in locals():
            router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
