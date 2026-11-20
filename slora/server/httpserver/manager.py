import zmq
import zmq.asyncio
import asyncio
import time
import uvloop
from typing import Union

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from slora.common.prefix_modes import CPU_MODES, FULL_REPREFILL_MODES, FULL_REPREFILL, FULL_RECONSTRUCT, normalize_prefix_mode
from ..io_struct import BatchStrOut, AbortReq, BatchAbortReq, LowRAResetReq, LowRAResetAck


class HttpServerManager:
    def __init__(
        self,
        model_weightdir,
        tokenizor_mode,
        router_port,
        httpserver_port,
        total_token_num,
        max_req_input_len,
        max_req_total_len,
        trust_remote_code,
        dummy=False,
        enable_lowra=False,
        lowra_shared_prefix_len=0,
        lowra_prefix_mode=FULL_REPREFILL,
        scheduler="slora",
        cpu_prefix_token_id=10,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")

        try: 
            self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code) 
        except:
            if dummy:
                self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode) 

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)
        self.reset_operations = {}

        self.total_token_num = total_token_num
        self.scheduler = scheduler
        self.max_req_input_len = max_req_input_len
        self.max_req_total_len = max_req_total_len
        self.enable_lowra = enable_lowra
        self.lowra_shared_prefix_len = lowra_shared_prefix_len
        prefix_mode = normalize_prefix_mode(lowra_prefix_mode)
        self.full_reprefill = prefix_mode in FULL_REPREFILL_MODES
        self.cpu_prefix_ids = ((cpu_prefix_token_id,) * lowra_shared_prefix_len if prefix_mode in CPU_MODES else None)
        self.full_reconstruct = prefix_mode == FULL_RECONSTRUCT
        self.shared_prefix_ids = None

    async def generate(self, adapter_dir, prompt, sampling_params, request_id, lowra_lengths=None):

        if isinstance(prompt, list):
            prompt_ids = [int(token_id) for token_id in prompt]
        else:
            prompt_ids = self.tokenizer.encode(prompt)
        explicit_lowra_lengths = lowra_lengths is not None
        lowra_lengths = dict(lowra_lengths or {})
        prefix_ids = None
        if self.enable_lowra and getattr(self, "full_reconstruct", False) and adapter_dir is None:
            raise ValueError("full reconstruction requires a LoRA for every request")
        if self.enable_lowra and self.full_reprefill and adapter_dir is None:
            raise ValueError("full reprefill requires a LoRA for every request")
        if self.enable_lowra and adapter_dir is not None:
            if "shared_prefix_len" not in lowra_lengths:
                lowra_lengths["shared_prefix_len"] = self.lowra_shared_prefix_len
            if lowra_lengths["shared_prefix_len"] != self.lowra_shared_prefix_len:
                raise ValueError(
                    f"LowRA shared_prefix_len {lowra_lengths['shared_prefix_len']} != server lowra_shared_prefix_len {self.lowra_shared_prefix_len}"
                )
        if self.enable_lowra and adapter_dir is not None:
            shared_prefix_len = lowra_lengths["shared_prefix_len"]
            if getattr(self, "full_reconstruct", False):
                lowra_lengths.setdefault("query_len", len(prompt_ids))
            if self.full_reprefill:
                query_len = int(lowra_lengths.get("query_len", len(prompt_ids) - shared_prefix_len))
                if query_len <= 0 or len(prompt_ids) != shared_prefix_len + query_len:
                    raise ValueError("full reprefill requires full [shared prefix + query] token input")
                prefix_ids = tuple(prompt_ids[:shared_prefix_len])
                if getattr(self, "cpu_prefix_ids", None) is not None and prefix_ids != self.cpu_prefix_ids:
                    raise ValueError("request prefix does not match the CPU bank")
                if self.shared_prefix_ids is not None and prefix_ids != self.shared_prefix_ids:
                    raise ValueError("all full-mode requests must use the same shared prefix")
                lowra_lengths["prefix_input_ids"] = list(prefix_ids)
            explicit_query_len = "query_len" in lowra_lengths
            if explicit_query_len:
                query_len = int(lowra_lengths["query_len"])
                if len(prompt_ids) == shared_prefix_len + query_len:
                    prompt_ids = prompt_ids[shared_prefix_len:]
                elif len(prompt_ids) != query_len:
                    raise ValueError(
                        "LowRA prompt token len must equal query_len for query-only input "
                        "or shared_prefix_len + query_len for full-prefix input"
                    )
            elif shared_prefix_len > 0:
                prompt_ids = prompt_ids[shared_prefix_len:]
            lowra_lengths.setdefault("query_len", len(prompt_ids))
            lowra_lengths.setdefault("decode_len", sampling_params.max_new_tokens)
            if explicit_lowra_lengths and lowra_lengths["query_len"] != len(prompt_ids):
                raise ValueError(
                    f"LowRA query_len metadata {lowra_lengths['query_len']} != query-only token len {len(prompt_ids)}"
                )
            if explicit_lowra_lengths and lowra_lengths["decode_len"] != sampling_params.max_new_tokens:
                raise ValueError(
                    f"LowRA decode_len metadata {lowra_lengths['decode_len']} != max_new_tokens {sampling_params.max_new_tokens}"
                )
        prompt_tokens = len(prompt_ids)
        logical_input_tokens = prompt_tokens
        if self.enable_lowra and self.full_reprefill and adapter_dir is not None:
            logical_input_tokens += lowra_lengths["shared_prefix_len"]
        elif self.enable_lowra and getattr(self, "full_reconstruct", False) and adapter_dir is not None:
            logical_input_tokens += lowra_lengths["shared_prefix_len"]
        if logical_input_tokens > self.max_req_input_len:
            raise ValueError(
                f"the input prompt token len {logical_input_tokens} is too long > {self.max_req_input_len}"
            )
        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if self.enable_lowra and adapter_dir is not None:
            req_total_len = (
                lowra_lengths["shared_prefix_len"]
                + lowra_lengths["query_len"]
                + lowra_lengths["decode_len"]
            )
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if getattr(self, "scheduler", "slora") != "online" and req_total_len + 1 > self.total_token_num:
            raise ValueError(
                f"the req token total len + 1 is too long > max_total_token_num:{self.total_token_num}"
            )
        
        sampling_params.stop_sentences_to_token_ids(self.tokenizer)
        if prefix_ids is not None:
            self.shared_prefix_ids = prefix_ids

        self.send_to_router.send_pyobj((adapter_dir, prompt_ids, sampling_params, request_id, lowra_lengths))
        event = asyncio.Event()
        self.req_id_to_out_inf[request_id] = ("", {}, False, event)
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            event.clear()
            # request_id is aborted by the backend system for traffic control
            if request_id not in self.req_id_to_out_inf:
                yield "", {}, -1
                break
            out_str, metadata, finished, _ = self.req_id_to_out_inf[request_id]
            if len(metadata) != 0:
                self.req_id_to_out_inf[request_id] = ("", {}, finished, event)
                metadata["prompt_tokens"] = logical_input_tokens
                yield out_str, metadata, finished
            if finished:
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                break
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        try:
            del self.req_id_to_out_inf[request_id]
        except:
            pass
        return

    async def reset_lowra_runtime(self, request_id):
        # Retain the pre-reset snapshot longer than the benchmark's 120s retry
        # window. An HTTP timeout/cancellation must not cancel the operation.
        now = time.monotonic()
        for key, (expires, _) in list(self.reset_operations.items()):
            if expires <= now:
                del self.reset_operations[key]
        if request_id not in self.reset_operations:
            if len(self.reset_operations) >= 128:
                return 503, {"ok": False, "message": "Too many recent reset operations", "stats": {}}
            result = asyncio.get_running_loop().create_future()
            self.reset_operations[request_id] = (now + 300, result)
            try:
                await asyncio.shield(self.send_to_router.send_pyobj(LowRAResetReq(request_id)))
            except Exception as exc:
                if not result.done():
                    result.set_result((500, {"ok": False, "message": repr(exc), "stats": {}}))
        else:
            _, result = self.reset_operations[request_id]
        try:
            return await asyncio.wait_for(asyncio.shield(result), timeout=60)
        except asyncio.TimeoutError:
            return 504, {"ok": False, "message": "Still waiting for LowRA runtime reset", "stats": {}}

    def _finish_reset(self, ack):
        operation = self.reset_operations.get(ack.req_id)
        if operation is None or operation[1].done():
            return
        if ack.ok and self.full_reprefill:
            self.shared_prefix_ids = None
        # Only a busy rejection is safe to execute again: the router checked
        # for active work before it touched the pool.
        busy = not ack.ok and "running_batch_size" in ack.stats
        if busy:
            del self.reset_operations[ack.req_id]
        status = 200 if ack.ok else (409 if busy else 500)
        operation[1].set_result((status, {"ok": ack.ok, "message": ack.message, "stats": ack.stats}))

    async def handle_loop(self):
        while True:
            recv_ans:Union(BatchStrOut, BatchAbortReq, LowRAResetAck) = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, (BatchStrOut, BatchAbortReq, LowRAResetAck)), f"error recv type {type(recv_ans)}"
            if isinstance(recv_ans, LowRAResetAck):
                self._finish_reset(recv_ans)
                continue

            if isinstance(recv_ans, BatchStrOut):
                for req_id, text, metadata, finished, abort in recv_ans.reqs_infs:
                    try:
                        if not abort:
                            _, _, _, event = self.req_id_to_out_inf[req_id]
                            self.req_id_to_out_inf[req_id] = (
                                text,
                                metadata,
                                finished,
                                event,
                            )
                            event.set()
                        else:
                            del self.req_id_to_out_inf[req_id]
                    except:
                        pass
            elif isinstance(recv_ans, BatchAbortReq):
                print("abort reqs:", recv_ans.reqs)
                for req_id in recv_ans.reqs:
                    try:
                        del self.req_id_to_out_inf[req_id]
                    except:
                        pass

        return
