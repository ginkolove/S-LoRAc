import torch

class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size = None
        self.total_token_num = None
        self.b_loc = None
        self.b_start_loc = None
        self.b_seq_len = None
        self.max_len_in_batch = None
        self.is_prefill = None
        
        self.mem_manager = None
        
        self.prefill_mem_index = None
        self.prefill_key_buffer = None
        self.prefill_value_buffer = None
        
        self.decode_is_contiguous = None
        self.decode_mem_start = None 
        self.decode_mem_end = None
        self.decode_mem_index = None
        self.decode_key_buffer = None 
        self.decode_value_buffer = None
    
        # system_prompt缓存相关
        self.use_system_prompt_cache = False
        self.system_prompt_tokens = 0
        self.adapter_dirs = None  # 当前批次的adapter目录列表
    
    def init_some_extra_state(self, 
            model, 
            batch_size, 
            total_token_num,
            max_len_in_batch,
            input_ids : torch.Tensor,
            b_loc : torch.Tensor,
            b_start_loc : torch.Tensor,
            b_seq_len : torch.Tensor,
            is_prefill,
            adapter_dirs=None):
        """
        初始化额外状态，包括system_prompt缓存检查
        
        Args:
            adapter_dirs (List[str], optional): 当前批次的adapter目录列表
        """
        self.adapter_dirs = adapter_dirs
        
        # 检查是否可以使用system_prompt缓存
        if hasattr(model.mem_manager, 'has_system_prompt_cache') and adapter_dirs:
            # 检查是否所有adapter都有system_prompt缓存
            all_have_cache = True
            for adapter_dir in set(adapter_dirs):  # 去重
                if not model.mem_manager.has_system_prompt_cache(adapter_dir):
                    all_have_cache = False
                    break
            
            # 还要检查基础模型是否有缓存
            if all_have_cache and model.mem_manager.has_system_prompt_cache(None):
                self.use_system_prompt_cache = True
                self.system_prompt_tokens = getattr(model.mem_manager, 'system_prompt_tokens', 0)
                print(f"Using system prompt cache with {self.system_prompt_tokens} tokens")
        
        # 调用原有的初始化逻辑（如果有子类需要）
        pass

    def get_system_prompt_kv_cache(self, model, adapter_dir=None):
        """
        获取指定adapter的system prompt KV缓存
        
        Args:
            model: 模型实例
            adapter_dir (str, optional): adapter目录，None表示基础模型
            
        Returns:
            dict: 包含key_cache和value_cache的字典，如果不可用则返回None
        """
        if not self.use_system_prompt_cache:
            return None
            
        if hasattr(model.mem_manager, 'get_system_prompt_kv'):
            return model.mem_manager.get_system_prompt_kv(adapter_dir)
        
        return None
