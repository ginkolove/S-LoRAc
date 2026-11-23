from slora.common.prefix_modes import CPU_MODES, FULL_REPREFILL, LAYER_SERIAL, LAYER_PREFETCH, LAYER_RETAINED, LAYER_ADAPTIVE, normalize_prefix_mode
from slora.common.space_plan_config import SpacePlanConfig


class ServeParams:

    def __init__(
        self,
        first_slo,
        token_slo,
    ) -> None:
        self.first_slo = first_slo
        self.token_slo = token_slo
        return
    
   
    def to_dict(self):
        ret = {}
        ret["first_slo"] = self.first_slo
        ret["token_slo"] = self.token_slo
        return ret


class InputParams:

    def __init__(
        self,
        max_req_total_len,
        # kv cache manager parameters
        max_total_token_num,
        pool_size_lora,
        batch_max_tokens,
        running_max_req_size,
        # mem_ratio,
        # adapter_ratio,
        # heuristic
        swap,
        prefetch,
        prefetch_size,
        scheduler,
        profile,
        batch_num_adapters,
        enable_abort,
        # kernel,
        # # debug
        dummy,
        no_lora_compute,
        no_lora_swap,
        # no_lora_copy,
        no_kernel,
        no_mem_pool,
        bmm,
        no_lora,
        # fairness
        fair_weights,
        # LowRA
        enable_lowra=False,
        lowra_shared_prefix_len=0,
        lowra_wait_slo_ms=0.0,
        lowra_prefix_mode=FULL_REPREFILL,
        lowra_workspace_loras=None,
        nccl_port=28765,
        lowra_retain_layers=0,
        lowra_retain_top_k=None,
        lowra_pin_hot_adapters=0,
        cpu_prefix_token_id=10,
        reuse_next_reconstructed=False,
        cpu_fixed_hot_indices=(0, 1),
        lowra_reconstruction_kernel="auto",
        lowra_reconstruction_workers=None,
        schedule_interval=1,
        max_active_loras=None,
        space_plan_config=None,
        cpu_prefix_pool_gb=None,
        impress_read_percent=None,
    ) -> None:
        self.max_req_total_len = max_req_total_len
        self.nccl_port = int(nccl_port)
        self.max_total_token_num = max_total_token_num
        self.lowra_original_total_token_num = max_total_token_num
        self.lowra_window_n = 1
        self.pool_size_lora = pool_size_lora
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        # self.mem_ratio = mem_ratio
        # self.adapter_ratio = adapter_ratio

        self.swap = swap
        self.prefetch = prefetch
        self.prefetch_size = prefetch_size
        self.scheduler = scheduler
        if schedule_interval <= 0 or (max_active_loras is not None and max_active_loras <= 0):
            raise ValueError("online scheduling interval and adapter limit must be positive")
        if scheduler == "online" and (prefetch or no_mem_pool or bmm or pool_size_lora
                                      or enable_abort or batch_num_adapters is not None
                                      or no_lora_compute or no_lora_swap or no_kernel):
            raise ValueError("online scheduling requires unified paging, on-demand adapters, "
                             "and no legacy prefetch/abort/clustering/debug overrides")
        self.schedule_interval = int(schedule_interval)
        self.max_active_loras = max_active_loras
        self.profile = profile
        self.batch_num_adapters = batch_num_adapters
        self.enable_abort = enable_abort
        # self.kernel = kernel

        self.dummy = dummy
        self.no_lora_compute = no_lora_compute
        self.no_lora_swap = no_lora_swap
        # self.no_lora_copy = no_lora_copy
        self.no_kernel = no_kernel
        self.no_mem_pool = no_mem_pool
        self.bmm = bmm
        self.no_lora = no_lora
        
        self.fair_weights = fair_weights
        self.enable_lowra = enable_lowra
        self.lowra_shared_prefix_len = lowra_shared_prefix_len
        self.lowra_wait_slo_ms = lowra_wait_slo_ms
        self.lowra_prefix_mode = normalize_prefix_mode(lowra_prefix_mode)
        self.space_plan_config = vars(SpacePlanConfig(**(space_plan_config or {})))
        if self.lowra_prefix_mode == LAYER_ADAPTIVE and (
                not enable_lowra or scheduler != "online" or lowra_retain_layers
                or lowra_retain_top_k is not None or lowra_pin_hot_adapters):
            raise ValueError("adaptive mode requires online LowRA without fixed retention/top-K/pinning")
        self.lowra_workspace_loras = lowra_workspace_loras
        self.lowra_retain_layers = int(lowra_retain_layers)
        self.lowra_retain_top_k = lowra_retain_top_k
        self.lowra_pin_hot_adapters = int(lowra_pin_hot_adapters)
        self.cpu_prefix_token_id = cpu_prefix_token_id
        if cpu_prefix_pool_gb is not None and (
                cpu_prefix_pool_gb < 1 or not dummy or not enable_lowra or self.lowra_prefix_mode not in CPU_MODES):
            raise ValueError("--cpu-prefix-pool-gb needs a positive decimal GB cap and a dummy CPU prefix mode")
        self.cpu_prefix_pool_gb = cpu_prefix_pool_gb
        self.impress_read_percent = impress_read_percent
        self.reuse_next_reconstructed = reuse_next_reconstructed
        self.cpu_fixed_hot_indices = cpu_fixed_hot_indices
        if lowra_reconstruction_kernel not in ("auto", "bounded"):
            raise ValueError("unknown LowRA reconstruction kernel")
        if lowra_reconstruction_kernel == "bounded" and (
                not enable_lowra or self.lowra_prefix_mode not in (LAYER_SERIAL, LAYER_PREFETCH, LAYER_RETAINED, LAYER_ADAPTIVE)):
            raise ValueError("bounded reconstruction requires a LowRA layer mode")
        if lowra_reconstruction_workers is not None and (
                lowra_reconstruction_kernel != "bounded"
                or not isinstance(lowra_reconstruction_workers, int) or lowra_reconstruction_workers <= 0):
            raise ValueError("reconstruction workers require bounded and a positive integer")
        self.lowra_reconstruction_kernel = lowra_reconstruction_kernel
        self.lowra_reconstruction_workers = lowra_reconstruction_workers
        self.lowra_layer_num = 1
        return
 
