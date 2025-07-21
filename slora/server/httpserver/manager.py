import zmq
import zmq.asyncio
import asyncio
import uvloop
from typing import Union

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from ..tokenizer import get_tokenizer
from ..io_struct import BatchStrOut, AbortReq, BatchAbortReq


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

        self.total_token_num = total_token_num
        self.max_req_input_len = max_req_input_len
        self.max_req_total_len = max_req_total_len

    async def generate(self, adapter_dir, prompt, sampling_params, request_id):

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)
        if prompt_tokens > self.max_req_input_len:
            raise ValueError(
                f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}"
            )
        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(
                f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )
        
        sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        self.send_to_router.send_pyobj((adapter_dir, prompt_ids, sampling_params, request_id))
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
                metadata["prompt_tokens"] = prompt_tokens
                yield out_str, metadata, finished
            if finished:
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                break
        return

    async def generate_with_system_prompt(self, adapter_dir, system_prompt_text, user_prompt, sampling_params, request_id):
        """
        使用system_prompt缓存进行推理
        
        Args:
            adapter_dir (str): adapter目录
            system_prompt_text (str): system prompt文本（用于标识缓存）
            user_prompt (str): 用户输入的prompt
            sampling_params: 采样参数
            request_id (str): 请求ID
        """
        # tokenize用户输入
        user_prompt_ids = self.tokenizer.encode(user_prompt)
        user_prompt_tokens = len(user_prompt_ids)
        
        # 验证输入长度
        if user_prompt_tokens > self.max_req_input_len:
            raise ValueError(
                f"the user prompt token len {user_prompt_tokens} is too long > {self.max_req_input_len}"
            )
        
        # 计算总长度（这里用户输入长度 + 输出长度，system prompt长度不计入因为已经缓存）
        req_total_len = user_prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req token total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(
                f"the req token total len + 1 (input len + output len + 1) is too long > max_total_token_num:{self.total_token_num}"
            )
        
        sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        # 发送带system_prompt标识的请求 (adapter_dir, prompt_ids, sampling_params, request_id, use_system_prompt, system_prompt_hash)
        system_prompt_hash = hash(system_prompt_text)
        system_prompt_request = (adapter_dir, user_prompt_ids, sampling_params, request_id, True, system_prompt_hash)
        self.send_to_router.send_pyobj(system_prompt_request)
        
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
                metadata["prompt_tokens"] = user_prompt_tokens  # 只计用户输入的token数
                metadata["using_system_cache"] = True
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

    async def init_system_prompt_cache(self, system_prompt, lora_dirs=None):
        """
        初始化system_prompt的KV缓存
        
        Args:
            system_prompt (str): system prompt文本
            lora_dirs (List[str], optional): 要初始化的lora目录列表，None表示所有lora
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # tokenize system prompt
            prompt_ids = self.tokenizer.encode(system_prompt)
            prompt_tokens = len(prompt_ids)
            
            if prompt_tokens > self.max_req_input_len:
                raise ValueError(f"System prompt too long: {prompt_tokens} > {self.max_req_input_len}")
            
            # 发送system prompt初始化请求到router
            # 使用特殊的request_id来标识这是system prompt初始化请求
            request_id = f"system_prompt_init_{hash(system_prompt)}"
            
            # 发送初始化请求：常规请求+is_system_prompt标志+lora_dirs
            # (adapter_dir, prompt_ids, sampling_params, request_id, is_system_prompt, lora_dirs)
            init_request = ("system_prompt_init", prompt_ids, None, request_id, True, lora_dirs)
            self.send_to_router.send_pyobj(init_request)
            
            # 等待初始化完成的确认
            event = asyncio.Event()
            self.req_id_to_out_inf[request_id] = ("", {}, False, event)
            
            try:
                # 等待最多180秒（3分钟），给每个LoRA足够的计算时间
                await asyncio.wait_for(event.wait(), timeout=180)
                
                # 检查结果
                if request_id in self.req_id_to_out_inf:
                    _, metadata, finished, _ = self.req_id_to_out_inf[request_id]
                    success = finished and metadata.get("success", False)
                    
                    # 清理
                    try:
                        del self.req_id_to_out_inf[request_id]
                    except:
                        pass
                    
                    return success
                else:
                    return False
                    
            except asyncio.TimeoutError:
                # 清理超时的请求
                try:
                    del self.req_id_to_out_inf[request_id]
                except:
                    pass
                return False
                
        except Exception as e:
            print(f"Error initializing system prompt cache: {e}")
            return False

    async def handle_loop(self):
        while True:
            recv_ans:Union(BatchStrOut, BatchAbortReq) = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, (BatchStrOut, BatchAbortReq)), f"error recv type {type(recv_ans)}"
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
