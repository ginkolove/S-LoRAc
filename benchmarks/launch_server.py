import argparse
import os
import psutil
import sys
from pathlib import Path

from exp_suite import BASE_MODEL, LORA_DIR
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slora.common.prefix_modes import PREFIX_MODES, FULL_REPREFILL, normalize_prefix_mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="debug")
    parser.add_argument("--backend", type=str, default="slora",
                        choices=["slora", "vllm", "lightllm", "vllm-packed"])
    parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--adapter-base", action="append", default=None)

    parser.add_argument("--num-adapter", type=int)
    parser.add_argument("--num-token", type=int)
    parser.add_argument("--max-req-input-len", type=int, default=None)
    parser.add_argument("--max-req-total-len", type=int, default=None)
    parser.add_argument("--batch-max-tokens", type=int, default=None)
    parser.add_argument("--running-max-req-size", type=int, default=None)
    parser.add_argument("--scheduler", default="slora")
    parser.add_argument("--schedule-interval", type=int, default=1)
    parser.add_argument("--max-active-loras", type=int, default=None)

    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--bmm", action="store_true")
    parser.add_argument("--batch-num-adapters", type=int, default=None)
    parser.add_argument("--enable-abort", action="store_true")
    parser.add_argument("--enable-lowra", action="store_true")
    parser.add_argument("--lowra-shared-prefix-len", type=int, default=0)
    parser.add_argument("--lowra-prefix-mode", type=normalize_prefix_mode,
                        choices=PREFIX_MODES, default=FULL_REPREFILL)
    parser.add_argument("--lowra-workspace-loras", type=int, default=None)
    parser.add_argument("--lowra-reconstruction-kernel", choices=("auto", "bounded"), default="auto")
    parser.add_argument("--lowra-reconstruction-workers", type=int, default=None)
    parser.add_argument("--lowra-retain-layers", type=int, default=0)
    parser.add_argument("--lowra-retain-top-k", type=int, default=None)
    parser.add_argument("--lowra-pin-hot-adapters", type=int, default=0)
    parser.add_argument("--lowra-wait-slo-ms", type=float, default=0.0)
    parser.add_argument("--vllm-mem-ratio", type=float, default=0.95)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    base_model = args.model_dir or BASE_MODEL[args.model_setting]
    adapter_dirs = args.adapter_base or LORA_DIR[args.model_setting]

    if args.device == "a10g":
        if args.num_adapter is None: args.num_adapter = 200
        if args.num_token is None: args.num_token = 14000
    elif args.device == "h100":
        if args.num_adapter is None: args.num_adapter = 1000
        if args.num_token is None: args.num_token = 120000
    elif args.device == "debug":
        if args.num_adapter is None: args.num_adapter = 30
        if args.num_token is None: args.num_token = 14000
        if args.no_mem_pool:
            args.num_token -= 64 * 4 * 18
    

    if args.backend == "slora":
        cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
        cmd += f" --model_dir {base_model}"
        cmd += f" --tokenizer_mode auto"
        cmd += f" --scheduler {args.scheduler} --schedule-interval {args.schedule_interval}"
        if args.max_active_loras is not None:
            cmd += f" --max-active-loras {args.max_active_loras}"

        loaded_adapters = 0
        num_iter = args.num_adapter // len(adapter_dirs) + 1
        for i in range(num_iter):
            for adapter_dir in adapter_dirs:
                if loaded_adapters >= args.num_adapter:
                    break
                cmd += f" --lora-dirs {adapter_dir}-{i}"
                loaded_adapters += 1

        if args.dummy:
            cmd += " --dummy"
        if args.max_req_total_len is not None:
            cmd += f" --max_req_total_len {args.max_req_total_len}"
        if args.max_req_input_len is not None:
            cmd += f" --max_req_input_len {args.max_req_input_len}"
        if args.batch_max_tokens is not None:
            cmd += f" --batch_max_tokens {args.batch_max_tokens}"
        if args.running_max_req_size is not None:
            cmd += f" --running_max_req_size {args.running_max_req_size}"
        cmd += " --swap"
        # cmd += " --scheduler pets"
        # cmd += " --profile"
        if args.enable_abort:
            cmd += " --enable-abort"
        if args.batch_num_adapters:
            cmd += f" --batch-num-adapters {args.batch_num_adapters}"
        if args.no_lora_compute:
            cmd += " --no-lora-compute"
        if args.prefetch:
            cmd += " --prefetch"
        if args.no_mem_pool:
            cmd += " --no-mem-pool"
        # cmd += " --no-lora-copy"
        # cmd += " --no-kernel"
        if args.bmm:
            cmd += " --bmm"
        if args.enable_lowra:
            cmd += " --enable-lowra"
            cmd += f" --lowra-shared-prefix-len {args.lowra_shared_prefix_len}"
            cmd += f" --lowra-prefix-mode {args.lowra_prefix_mode}"
            cmd += f" --lowra-reconstruction-kernel {args.lowra_reconstruction_kernel}"
            if args.lowra_reconstruction_workers is not None:
                cmd += f" --lowra-reconstruction-workers {args.lowra_reconstruction_workers}"
            cmd += f" --lowra-retain-layers {args.lowra_retain_layers}"
            if args.lowra_retain_top_k is not None:
                cmd += f" --lowra-retain-top-k {args.lowra_retain_top_k}"
            if args.lowra_pin_hot_adapters:
                cmd += f" --lowra-pin-hot-adapters {args.lowra_pin_hot_adapters}"
            if args.lowra_workspace_loras is not None:
                cmd += f" --lowra-workspace-loras {args.lowra_workspace_loras}"
            if args.lowra_wait_slo_ms > 0:
                cmd += f" --lowra-wait-slo-ms {args.lowra_wait_slo_ms}"

    elif args.backend == "lightllm":
        cmd = f"python -m lightllm.server.api_server" \
              f" --model_dir {base_model} --tp 1 --max_total_token_num {args.num_token}" \
              f" --tokenizer_mode auto" \
              f" --host 127.0.0.1 --port 8000"

    elif args.backend == "vllm":
        cmd = f"python -m vllm.entrypoints.api_server" \
              f" --model {base_model} --swap-space 16" \
              f" --disable-log-requests" \
              f" --host 127.0.0.1 --port 8000"

    elif args.backend == "vllm-packed":
        gpu_memory = args.vllm_mem_ratio / (args.num_adapter)

        for i in range(args.num_adapter):
            pid = os.fork()
            if pid == 0:
                cmd = f"python -m vllm.entrypoints.api_server" \
                    f" --model {base_model} --swap-space 0" \
                    f" --gpu-memory-utilization {gpu_memory}" \
                    f" --disable-log-requests" \
                    f" --host 127.0.0.1 --port {8000 + i}"
                os.system(cmd)
                sys.exit(0)

        for _ in range(args.num_adapter):
            os.wait()

        sys.exit(0)

    if args.print_only:
        print(cmd)
    else:
        os.system(cmd)
