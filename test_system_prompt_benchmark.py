"""
System Prompt KV Cache 性能测试脚本
仿照 run_exp.py 实现，测试有无 system_prompt KV cache 复用的性能差异

使用方法:
python test_system_prompt_benchmark.py --server http://localhost:8000 --mode with_cache
python test_system_prompt_benchmark.py --server http://localhost:8000 --mode without_cache
python test_system_prompt_benchmark.py --server http://localhost:8000 --mode compare
"""
import argparse
import asyncio
import json
import numpy as np
import os
import sys
import time
from typing import List, Tuple, Optional
import aiohttp

# System prompt for testing
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Please provide accurate, helpful, and well-structured responses to user queries. Always be polite and professional in your interactions."""

# Test user prompts
TEST_USER_PROMPTS = [
    "What is machine learning?",
    "Explain the concept of neural networks.",
    "How do transformers work in NLP?",
    "What are the advantages of LoRA adapters?",
    "Describe the attention mechanism.",
    "What is the difference between training and inference?",
    "How does batch processing improve efficiency?",
    "Explain gradient descent optimization.",
    "What are the challenges in large language model deployment?",
    "How does KV cache improve inference speed?",
]

# LoRA adapter directories for testing
TEST_ADAPTERS = [
    "../lora_test_data/finiteautomata-bertweet-base-sentiment-analysis",
    "../lora_test_data/winglian-llama-7b-evol-instruct-v2",
]

# (prompt_len, output_len, latency, first_token_latency, mode)
REQUEST_LATENCY: List[Tuple[int, int, float, float, str]] = []


class SystemPromptBenchmark:
    def __init__(self, server: str, model_dir: str = "../model_data/Llama-2-7b-hf"):
        self.server = server
        self.model_dir = model_dir
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.system_prompt_hash = None
        
    async def init_system_prompt_cache(self, lora_dirs: Optional[List[str]] = None) -> bool:
        """初始化 system_prompt KV cache"""
        print("正在初始化 system_prompt KV cache...")
        
        url = self.server + "/init_system_prompt"
        data = {
            'model_dir': self.model_dir,
            'lora_dirs': lora_dirs or [],
            'system_prompt': self.system_prompt
        }
        
        headers = {'Content-Type': 'application/json'}
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.system_prompt_hash = result.get('system_prompt_hash')
                        init_time = time.time() - start_time
                        print(f"✅ system_prompt KV cache 初始化成功！")
                        print(f"   - 初始化时间: {init_time:.2f}s")
                        print(f"   - system_prompt_hash: {self.system_prompt_hash}")
                        print(f"   - 基础模型: {result.get('base_model_initialized', False)}")
                        print(f"   - LoRA适配器: {len(result.get('adapters_initialized', []))}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ 初始化失败: HTTP {response.status}")
                        print(f"   错误详情: {error_text}")
                        return False
        except Exception as e:
            print(f"❌ 初始化异常: {e}")
            return False

    async def send_request_with_cache(
        self,
        req_id: str,
        adapter_dir: Optional[str],
        user_prompt: str,
        output_len: int = 50,
        use_cache: bool = True
    ) -> Tuple[int, int, float, Optional[float]]:
        """发送请求（使用或不使用 system_prompt cache）"""
        
        if use_cache and self.system_prompt_hash:
            # 使用 system_prompt cache
            full_prompt = user_prompt  # 只发送用户输入
            url = self.server + "/generate_stream"
            data = {
                'model_dir': self.model_dir,
                'lora_dir': adapter_dir,
                'inputs': full_prompt,
                'system_prompt_hash': self.system_prompt_hash,  # 使用缓存
                'parameters': {
                    'do_sample': False,
                    'max_new_tokens': output_len,
                    'ignore_eos': True,
                }
            }
        else:
            # 不使用 system_prompt cache，完整输入
            full_prompt = f"{self.system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            url = self.server + "/generate_stream" 
            data = {
                'model_dir': self.model_dir,
                'lora_dir': adapter_dir,
                'inputs': full_prompt,
                'parameters': {
                    'do_sample': False,
                    'max_new_tokens': output_len,
                    'ignore_eos': True,
                }
            }
        
        request_start_time = time.time()
        first_token_latency = None
        headers = {'Content-Type': 'application/json'}
        timeout = aiohttp.ClientTimeout(total=300)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        if first_token_latency is None:
                            first_token_latency = time.time() - request_start_time
                        chunks.append(chunk)
                
                output = b"".join(chunks).decode("utf-8")
                if '\"finished\": -1' in output:
                    print(f"⚠️  请求 {req_id} 被中止")
                    return (len(full_prompt), output_len, -1, None)
                    
        except Exception as e:
            print(f"❌ 请求 {req_id} 异常: {e}")
            return (len(full_prompt), output_len, -1, None)
        
        request_end_time = time.time()
        request_latency = request_end_time - request_start_time
        
        cache_status = "with_cache" if (use_cache and self.system_prompt_hash) else "without_cache"
        print(f"req_id {req_id} ({cache_status}) prompt_len {len(full_prompt)} "
              f"output_len {output_len} latency {request_latency:.2f}s "
              f"first_token {first_token_latency:.2f}s")
        
        return (len(full_prompt), output_len, request_latency, first_token_latency)

    async def run_benchmark_batch(
        self,
        requests: List[Tuple[str, str, str, int]],  # (req_id, adapter_dir, user_prompt, output_len)
        use_cache: bool = True,
        concurrent: int = 1
    ) -> List[Tuple[int, int, float, Optional[float]]]:
        """运行批量测试"""
        
        mode = "with_cache" if use_cache else "without_cache"
        print(f"\n🚀 开始批量测试 ({mode})...")
        print(f"   - 并发数: {concurrent}")
        print(f"   - 请求数: {len(requests)}")
        
        start_time = time.time()
        results = []
        
        # 分批处理请求
        for i in range(0, len(requests), concurrent):
            batch = requests[i:i + concurrent]
            tasks = []
            
            for req_id, adapter_dir, user_prompt, output_len in batch:
                task = self.send_request_with_cache(
                    req_id, adapter_dir, user_prompt, output_len, use_cache
                )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # 添加请求间隔
            if i + concurrent < len(requests):
                await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        print(f"✅ 批量测试完成，总耗时: {total_time:.2f}s")
        
        return results

    def analyze_results(self, results: List[Tuple[int, int, float, Optional[float]]], mode: str):
        """分析测试结果"""
        print(f"\n📊 {mode.upper()} 性能分析:")
        print("=" * 50)
        
        # 过滤失败的请求
        valid_results = [r for r in results if r[2] > 0 and r[3] is not None]
        failed_count = len(results) - len(valid_results)
        
        if failed_count > 0:
            print(f"❌ 失败请求: {failed_count}")
        
        if not valid_results:
            print("❌ 没有有效的测试结果")
            return {}
        
        # 计算统计指标
        latencies = [r[2] for r in valid_results]
        first_token_latencies = [r[3] for r in valid_results]
        
        stats = {
            'mode': mode,
            'total_requests': len(results),
            'successful_requests': len(valid_results),
            'failed_requests': failed_count,
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p90_latency': np.percentile(latencies, 90),
            'p99_latency': np.percentile(latencies, 99),
            'avg_first_token_latency': np.mean(first_token_latencies),
            'p50_first_token_latency': np.percentile(first_token_latencies, 50),
            'p90_first_token_latency': np.percentile(first_token_latencies, 90),
        }
        
        print(f"✅ 成功请求: {stats['successful_requests']}/{stats['total_requests']}")
        print(f"📈 平均延迟: {stats['avg_latency']:.3f}s")
        print(f"📈 P50延迟: {stats['p50_latency']:.3f}s")  
        print(f"📈 P90延迟: {stats['p90_latency']:.3f}s")
        print(f"📈 P99延迟: {stats['p99_latency']:.3f}s")
        print(f"🚀 平均首Token延迟: {stats['avg_first_token_latency']:.3f}s")
        print(f"🚀 P50首Token延迟: {stats['p50_first_token_latency']:.3f}s")
        print(f"🚀 P90首Token延迟: {stats['p90_first_token_latency']:.3f}s")
        
        return stats

    def compare_results(self, stats_with_cache: dict, stats_without_cache: dict):
        """对比有无缓存的性能差异"""
        print(f"\n🔄 性能对比分析:")
        print("=" * 50)
        
        if not stats_with_cache or not stats_without_cache:
            print("❌ 缺少对比数据")
            return
        
        def calc_improvement(with_cache, without_cache):
            if without_cache == 0:
                return 0
            return (without_cache - with_cache) / without_cache * 100
        
        latency_improvement = calc_improvement(
            stats_with_cache['avg_latency'], 
            stats_without_cache['avg_latency']
        )
        
        first_token_improvement = calc_improvement(
            stats_with_cache['avg_first_token_latency'],
            stats_without_cache['avg_first_token_latency'] 
        )
        
        p90_improvement = calc_improvement(
            stats_with_cache['p90_latency'],
            stats_without_cache['p90_latency']
        )
        
        print(f"🚀 总体延迟改善: {latency_improvement:+.2f}%")
        print(f"🚀 首Token延迟改善: {first_token_improvement:+.2f}%") 
        print(f"🚀 P90延迟改善: {p90_improvement:+.2f}%")
        
        if latency_improvement > 0:
            print("✅ system_prompt KV cache 显著提升了性能！")
        else:
            print("⚠️  system_prompt KV cache 未显著提升性能")


async def main():
    parser = argparse.ArgumentParser(description="System Prompt KV Cache 性能测试")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                       help="服务器地址")
    parser.add_argument("--model-dir", type=str, default="../model_data/Llama-2-7b-hf",
                       help="基础模型目录")
    parser.add_argument("--mode", type=str, default="compare",
                       choices=["with_cache", "without_cache", "compare"],
                       help="测试模式")
    parser.add_argument("--concurrent", type=int, default=1,
                       help="并发请求数")
    parser.add_argument("--num-requests", type=int, default=10,
                       help="总请求数")
    parser.add_argument("--output-len", type=int, default=50,
                       help="生成长度")
    parser.add_argument("--output-file", type=str, default="system_prompt_benchmark.json",
                       help="结果输出文件")
    
    args = parser.parse_args()
    
    # 创建测试对象
    benchmark = SystemPromptBenchmark(args.server, args.model_dir)
    
    # 生成测试请求
    requests = []
    for i in range(args.num_requests):
        req_id = f"req_{i:03d}"
        adapter_dir = TEST_ADAPTERS[i % len(TEST_ADAPTERS)] if TEST_ADAPTERS else None
        user_prompt = TEST_USER_PROMPTS[i % len(TEST_USER_PROMPTS)]
        requests.append((req_id, adapter_dir, user_prompt, args.output_len))
    
    results = {}
    
    if args.mode in ["with_cache", "compare"]:
        # 初始化 system_prompt cache
        init_success = await benchmark.init_system_prompt_cache(TEST_ADAPTERS)
        if not init_success:
            print("❌ system_prompt cache 初始化失败，退出测试")
            return
        
        # 测试使用缓存的性能
        cache_results = await benchmark.run_benchmark_batch(
            requests, use_cache=True, concurrent=args.concurrent
        )
        results['with_cache'] = benchmark.analyze_results(cache_results, "with_cache")
    
    if args.mode in ["without_cache", "compare"]:
        # 测试不使用缓存的性能
        no_cache_results = await benchmark.run_benchmark_batch(
            requests, use_cache=False, concurrent=args.concurrent
        )
        results['without_cache'] = benchmark.analyze_results(no_cache_results, "without_cache")
    
    # 对比分析
    if args.mode == "compare" and 'with_cache' in results and 'without_cache' in results:
        benchmark.compare_results(results['with_cache'], results['without_cache'])
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main()) 