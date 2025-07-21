"""
启动 S-LoRAc 服务器并完成 system_prompt KV cache 初始化
仿照 launch_server.py 实现，支持自动初始化 system_prompt 缓存

使用方法:
python launch_with_system_prompt.py --device debug --model-setting S1 --auto-init
python launch_with_system_prompt.py --device a10g --model-setting S1 --system-prompt "You are a helpful assistant"
"""
import argparse
import asyncio
import json
import os
import psutil
import requests
import subprocess
import sys
import time
from typing import List, Optional
import aiohttp

# 添加项目路径到 Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from benchmarks.exp_suite import BASE_MODEL, LORA_DIR
except ImportError:
    # Fallback 配置
    BASE_MODEL = {
        "S1": "../model_data/Llama-2-7b-hf",
        "S2": "../model_data/Llama-2-13b-hf",
    }
    LORA_DIR = {
        "S1": [
            "../lora_test_data/finiteautomata-bertweet-base-sentiment-analysis",
            "../lora_test_data/winglian-llama-7b-evol-instruct-v2",
        ],
        "S2": [
            "../lora_test_data/finiteautomata-bertweet-base-sentiment-analysis",
            "../lora_test_data/winglian-llama-7b-evol-instruct-v2",
        ]
    }

# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Please provide accurate, helpful, and well-structured responses to user queries. Always be polite and professional in your interactions."""


class ServerManager:
    def __init__(self, args):
        self.args = args
        self.server_process = None
        self.server_url = f"http://localhost:{args.port}"
        self.base_model = BASE_MODEL[args.model_setting]
        self.adapter_dirs = self._prepare_adapter_dirs()
        
    def _prepare_adapter_dirs(self):
        """准备 adapter 目录列表"""
        adapter_dirs = LORA_DIR[self.args.model_setting]
        
        # 扩展 adapter 数量
        extended_adapters = []
        num_iter = self.args.num_adapter // len(adapter_dirs) + 1
        
        for i in range(num_iter):
            for adapter_dir in adapter_dirs:
                extended_adapters.append(f"{adapter_dir}-{i}")
                if len(extended_adapters) >= self.args.num_adapter:
                    break
            if len(extended_adapters) >= self.args.num_adapter:
                break
                
        return extended_adapters[:self.args.num_adapter]
    
    def build_server_command(self):
        """构建服务器启动命令"""
        if self.args.backend == "slora":
            cmd = [
                "python", "-m", "slora.server.api_server",
                "--max_total_token_num", str(self.args.num_token),
                "--model", self.base_model,
                "--tokenizer_mode", "auto",
                "--host", "127.0.0.1",
                "--port", str(self.args.port)
            ]
            
            # 添加 LoRA adapters
            for adapter_dir in self.adapter_dirs:
                cmd.extend(["--lora", adapter_dir])
                
            # 添加其他参数
            if self.args.dummy:
                cmd.append("--dummy")
            cmd.append("--swap")
            
            if self.args.enable_abort:
                cmd.append("--enable-abort")
            if self.args.batch_num_adapters:
                cmd.extend(["--batch-num-adapters", str(self.args.batch_num_adapters)])
            if self.args.no_lora_compute:
                cmd.append("--no-lora-compute")
            if self.args.prefetch:
                cmd.append("--prefetch")
            if self.args.no_mem_pool:
                cmd.append("--no-mem-pool")
            if self.args.bmm:
                cmd.append("--bmm")
                
        elif self.args.backend == "lightllm":
            cmd = [
                "python", "-m", "lightllm.server.api_server",
                "--model_dir", self.base_model,
                "--tp", "1",
                "--max_total_token_num", str(self.args.num_token),
                "--tokenizer_mode", "auto",
                "--host", "127.0.0.1",
                "--port", str(self.args.port)
            ]
            
        elif self.args.backend == "vllm":
            cmd = [
                "python", "-m", "vllm.entrypoints.api_server",
                "--model", self.base_model,
                "--swap-space", "16",
                "--disable-log-requests",
                "--host", "127.0.0.1",
                "--port", str(self.args.port)
            ]
            
        else:
            raise ValueError(f"不支持的后端: {self.args.backend}")
            
        return cmd
    
    def start_server(self):
        """启动服务器"""
        cmd = self.build_server_command()
        print(f"🚀 启动服务器命令: {' '.join(cmd)}")
        
        # 启动服务器进程
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ 服务器进程已启动 (PID: {self.server_process.pid})")
        return self.server_process
    
    def wait_for_server_ready(self, timeout: int = 300):
        """等待服务器启动就绪"""
        print(f"⏳ 等待服务器启动 (最大等待 {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ 服务器启动成功！")
                    return True
            except:
                pass
                
            # 检查进程是否还在运行
            if self.server_process and self.server_process.poll() is not None:
                print("❌ 服务器进程意外终止")
                return False
                
            time.sleep(5)
        
        print("❌ 服务器启动超时")
        return False
    
    async def init_system_prompt_cache(self, system_prompt: str) -> bool:
        """初始化 system_prompt KV cache"""
        if self.args.backend != "slora":
            print(f"⚠️  {self.args.backend} 后端不支持 system_prompt 缓存")
            return True
            
        print("🔧 正在初始化 system_prompt KV cache...")
        
        url = f"{self.server_url}/init_system_prompt"
        data = {
            'model_dir': self.base_model,
            'lora_dirs': self.adapter_dirs,
            'system_prompt': system_prompt
        }
        
        headers = {'Content-Type': 'application/json'}
        timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                
                print(f"📤 发送初始化请求...")
                print(f"   - 基础模型: {self.base_model}")
                print(f"   - LoRA数量: {len(self.adapter_dirs)}")
                print(f"   - System Prompt: {system_prompt[:50]}...")
                
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        init_time = time.time() - start_time
                        
                        print("✅ system_prompt KV cache 初始化成功！")
                        print(f"   - 初始化时间: {init_time:.2f}s")
                        print(f"   - system_prompt_hash: {result.get('system_prompt_hash')}")
                        print(f"   - 基础模型已初始化: {result.get('base_model_initialized', False)}")
                        
                        adapters_init = result.get('adapters_initialized', [])
                        print(f"   - LoRA适配器已初始化: {len(adapters_init)}")
                        
                        if adapters_init:
                            print("   - 已初始化的适配器:")
                            for adapter in adapters_init[:5]:  # 只显示前5个
                                print(f"     * {adapter}")
                            if len(adapters_init) > 5:
                                print(f"     ... 还有 {len(adapters_init) - 5} 个")
                        
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ 初始化失败: HTTP {response.status}")
                        print(f"   错误详情: {error_text}")
                        return False
                        
        except asyncio.TimeoutError:
            print("❌ 初始化超时")
            return False
        except Exception as e:
            print(f"❌ 初始化异常: {e}")
            return False
    
    async def test_inference(self, system_prompt_hash: Optional[str] = None):
        """测试推理功能"""
        print("\n🧪 测试推理功能...")
        
        test_prompts = [
            "Hello! How are you?",
            "What is machine learning?",
            "Explain neural networks briefly."
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n测试 {i+1}: {prompt}")
            
            url = f"{self.server_url}/generate"
            data = {
                'model_dir': self.base_model,
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': 50,
                    'do_sample': False,
                }
            }
            
            # 如果有 system_prompt_hash，使用它
            if system_prompt_hash:
                data['system_prompt_hash'] = system_prompt_hash
                if self.adapter_dirs:
                    data['lora_dir'] = self.adapter_dirs[0]  # 使用第一个adapter测试
            
            try:
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    async with session.post(url, json=data, timeout=60) as response:
                        if response.status == 200:
                            result = await response.json()
                            latency = time.time() - start_time
                            output = result.get('generated_text', result.get('output', ''))
                            
                            print(f"✅ 推理成功 (延迟: {latency:.2f}s)")
                            print(f"   输出: {output[:100]}...")
                        else:
                            error_text = await response.text()
                            print(f"❌ 推理失败: HTTP {response.status}")
                            print(f"   错误: {error_text[:200]}...")
                            
            except Exception as e:
                print(f"❌ 推理异常: {e}")
                
            await asyncio.sleep(1)  # 请求间隔
    
    def stop_server(self):
        """停止服务器"""
        if self.server_process:
            print("🛑 正在停止服务器...")
            self.server_process.terminate()
            
            try:
                self.server_process.wait(timeout=10)
                print("✅ 服务器已停止")
            except subprocess.TimeoutExpired:
                print("⚠️  强制终止服务器进程")
                self.server_process.kill()
                self.server_process.wait()
    
    def show_server_logs(self):
        """显示服务器日志"""
        if not self.server_process:
            return
            
        print("\n📋 服务器日志:")
        print("=" * 60)
        
        # 读取已有的输出
        try:
            output, _ = self.server_process.communicate(timeout=1)
            print(output)
        except subprocess.TimeoutExpired:
            # 进程还在运行，读取部分输出
            pass


async def main():
    parser = argparse.ArgumentParser(description="启动S-LoRAc服务器并初始化system_prompt缓存")
    
    # 服务器配置
    parser.add_argument("--device", type=str, default="debug",
                       choices=["debug", "a10g", "h100"], help="设备类型")
    parser.add_argument("--backend", type=str, default="slora",
                       choices=["slora", "vllm", "lightllm"], help="后端类型")
    parser.add_argument("--model-setting", type=str, default="S1",
                       choices=["S1", "S2"], help="模型设置")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    
    # 资源配置
    parser.add_argument("--num-adapter", type=int, help="LoRA适配器数量")
    parser.add_argument("--num-token", type=int, help="最大token数量")
    
    # 功能开关
    parser.add_argument("--dummy", action="store_true", help="使用dummy模式")
    parser.add_argument("--no-lora-compute", action="store_true", help="禁用LoRA计算")
    parser.add_argument("--prefetch", action="store_true", help="启用预取")
    parser.add_argument("--no-mem-pool", action="store_true", help="禁用内存池")
    parser.add_argument("--bmm", action="store_true", help="使用BMM")
    parser.add_argument("--batch-num-adapters", type=int, help="批处理适配器数量")
    parser.add_argument("--enable-abort", action="store_true", help="启用请求中止")
    
    # System prompt 配置
    parser.add_argument("--auto-init", action="store_true", 
                       help="自动初始化system_prompt缓存")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                       help="系统提示词")
    parser.add_argument("--test-inference", action="store_true",
                       help="测试推理功能")
    
    # 其他选项
    parser.add_argument("--wait-only", action="store_true",
                       help="仅等待现有服务器，不启动新进程")
    parser.add_argument("--init-only", action="store_true",
                       help="仅初始化缓存，不启动服务器")
    
    args = parser.parse_args()
    
    # 设置设备默认参数
    if args.device == "a10g":
        if args.num_adapter is None: args.num_adapter = 200
        if args.num_token is None: args.num_token = 14000
    elif args.device == "h100":
        if args.num_adapter is None: args.num_adapter = 1000
        if args.num_token is None: args.num_token = 120000
    elif args.device == "debug":
        if args.num_adapter is None: args.num_adapter = 5  # 调试模式使用较少适配器
        if args.num_token is None: args.num_token = 8000
        if args.no_mem_pool:
            args.num_token -= 64 * 4 * 18
    
    print("🔧 S-LoRAc 服务器启动器")
    print("=" * 50)
    print(f"设备: {args.device}")
    print(f"后端: {args.backend}")
    print(f"模型: {args.model_setting}")
    print(f"端口: {args.port}")
    print(f"适配器数量: {args.num_adapter}")
    print(f"最大Token数: {args.num_token}")
    print(f"自动初始化: {args.auto_init}")
    
    manager = ServerManager(args)
    system_prompt_hash = None
    
    try:
        if not args.init_only:
            if not args.wait_only:
                # 启动服务器
                manager.start_server()
                
                # 等待服务器就绪
                if not manager.wait_for_server_ready():
                    print("❌ 服务器启动失败")
                    return
            else:
                # 仅等待现有服务器
                if not manager.wait_for_server_ready(timeout=60):
                    print("❌ 无法连接到现有服务器")
                    return
        
        # 初始化 system_prompt 缓存
        if args.auto_init or args.init_only:
            success = await manager.init_system_prompt_cache(args.system_prompt)
            if success:
                print("\n💾 system_prompt KV cache 已就绪！")
                # 这里可以保存 system_prompt_hash 供后续使用
            else:
                print("\n❌ system_prompt KV cache 初始化失败")
                
        # 测试推理
        if args.test_inference and not args.init_only:
            await manager.test_inference(system_prompt_hash)
        
        if not args.init_only:
            print(f"\n🌟 服务器运行中: {manager.server_url}")
            print("按 Ctrl+C 停止服务器")
            
            # 保持运行
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n⌨️  收到停止信号")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not args.init_only and not args.wait_only:
            manager.stop_server()


if __name__ == "__main__":
    asyncio.run(main()) 