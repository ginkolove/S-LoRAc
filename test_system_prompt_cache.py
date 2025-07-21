#!/usr/bin/env python3
"""
测试 S-LoRAc system_prompt KV缓存功能
"""

import json
import time
import requests
import sys
import argparse
from typing import Dict, List, Optional

class SLoRASystemPromptTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def _post(self, endpoint: str, data: Dict) -> Dict:
        """发送POST请求"""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = self.session.post(url, json=data, headers=headers, timeout=200)  # 3分钟以上，匹配服务端超时
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            return {"error": str(e)}
    
    def check_service_health(self) -> bool:
        """检查服务是否运行"""
        try:
            response = self.session.get(f"{self.base_url}/healthz", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def init_system_prompt_cache(self, system_prompt: str, lora_dirs: Optional[List[str]] = None) -> Dict:
        """初始化system_prompt缓存"""
        data = {
            "system_prompt": system_prompt,
            "lora_dirs": lora_dirs if lora_dirs is not None else []
        }
        
        print(f"正在初始化system_prompt缓存...")
        print(f"  system_prompt: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}")
        print(f"  lora_dirs: {lora_dirs if lora_dirs else '[] (所有LoRA)'}")
        
        start_time = time.time()
        result = self._post("/init_system_prompt", data)
        end_time = time.time()
        
        print(f"初始化耗时: {end_time - start_time:.2f}秒")
        
        if "error" in result:
            print(f"❌ 初始化失败: {result['error']}")
            return result
        elif result.get("status") == "success":
            print(f"✅ 初始化成功: {result['message']}")
            return result
        else:
            print(f"❓ 未知响应: {result}")
            return result
    
    def generate_text(self, prompt: str, lora_dir: Optional[str] = None, 
                     use_system_prompt: bool = False, system_prompt: str = "",
                     max_tokens: int = 100, temperature: float = 0.7,
                     stream: bool = False) -> Dict:
        """生成文本"""
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.9
            }
        }
        
        if lora_dir:
            data["lora_dir"] = lora_dir
            
        if use_system_prompt:
            data["use_system_prompt"] = True
            data["system_prompt"] = system_prompt
        
        endpoint = "/generate_stream" if stream else "/generate"
        
        start_time = time.time()
        result = self._post(endpoint, data)
        end_time = time.time()
        
        return {
            **result,
            "generation_time": end_time - start_time,
            "used_system_cache": result.get("using_system_cache", False)
        }
    
    def benchmark_comparison(self, system_prompt: str, user_prompt: str,
                           lora_dir: Optional[str] = None, 
                           max_tokens: int = 50, rounds: int = 3):
        """对比常规推理和缓存推理的性能"""
        print(f"\n{'='*60}")
        print(f"性能对比测试 (轮次: {rounds})")
        print(f"system_prompt: {system_prompt[:30]}...")
        print(f"user_prompt: {user_prompt[:30]}...")
        print(f"lora_dir: {lora_dir}")
        print(f"{'='*60}")
        
        # 常规推理测试
        print(f"\n📊 常规推理测试 (完整prompt):")
        normal_times = []
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        for i in range(rounds):
            print(f"  轮次 {i+1}/{rounds}...", end="")
            result = self.generate_text(
                prompt=full_prompt,
                lora_dir=lora_dir,
                use_system_prompt=False,
                max_tokens=max_tokens
            )
            
            if "error" not in result:
                gen_time = result["generation_time"]
                normal_times.append(gen_time)
                output_tokens = result.get("count_output_tokens", 0)
                print(f" {gen_time:.2f}s (生成{output_tokens}个token)")
            else:
                print(f" ❌ 失败: {result['error']}")
        
        # 缓存推理测试
        print(f"\n🚀 缓存推理测试 (复用system_prompt):")
        cached_times = []
        
        for i in range(rounds):
            print(f"  轮次 {i+1}/{rounds}...", end="")
            result = self.generate_text(
                prompt=user_prompt,
                lora_dir=lora_dir,
                use_system_prompt=True,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
            
            if "error" not in result:
                gen_time = result["generation_time"]
                cached_times.append(gen_time)
                output_tokens = result.get("count_output_tokens", 0)
                using_cache = result.get("used_system_cache", False)
                cache_status = "✅" if using_cache else "❌"
                print(f" {gen_time:.2f}s (生成{output_tokens}个token) {cache_status}")
            else:
                print(f" ❌ 失败: {result['error']}")
        
        # 统计结果
        if normal_times and cached_times:
            avg_normal = sum(normal_times) / len(normal_times)
            avg_cached = sum(cached_times) / len(cached_times)
            speedup = avg_normal / avg_cached if avg_cached > 0 else 0
            
            print(f"\n📈 性能统计:")
            print(f"  常规推理平均时间: {avg_normal:.3f}s")
            print(f"  缓存推理平均时间: {avg_cached:.3f}s")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  时间节省: {((avg_normal - avg_cached) / avg_normal * 100):.1f}%")
        
        return {
            "normal_times": normal_times,
            "cached_times": cached_times,
            "avg_normal": avg_normal if normal_times else 0,
            "avg_cached": avg_cached if cached_times else 0,
            "speedup": speedup if normal_times and cached_times else 0
        }
    
    def test_multiple_lora(self, system_prompt: str, user_prompt: str, 
                          lora_dirs: List[str], max_tokens: int = 30):
        """测试多个LoRA的缓存功能"""
        print(f"\n🔄 多LoRA缓存测试")
        print(f"测试的LoRA目录: {lora_dirs}")
        
        for lora_dir in lora_dirs:
            print(f"\n测试LoRA: {lora_dir}")
            result = self.generate_text(
                prompt=user_prompt,
                lora_dir=lora_dir,
                use_system_prompt=True,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
            
            if "error" not in result:
                gen_time = result["generation_time"]
                output = result.get("generated_text", [""])[0]
                using_cache = result.get("used_system_cache", False)
                cache_status = "✅ 缓存命中" if using_cache else "❌ 缓存未命中"
                
                print(f"  时间: {gen_time:.2f}s | {cache_status}")
                print(f"  输出: {output[:100]}{'...' if len(output) > 100 else ''}")
            else:
                print(f"  ❌ 失败: {result['error']}")


def main():
    parser = argparse.ArgumentParser(description="S-LoRAc System Prompt Cache测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--lora-dirs", nargs="*", help="要测试的LoRA目录列表")
    parser.add_argument("--rounds", type=int, default=3, help="性能测试轮次")
    parser.add_argument("--max-tokens", type=int, default=50, help="最大生成token数")
    parser.add_argument("--skip-init", action="store_true", help="跳过缓存初始化（假设已初始化）")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = SLoRASystemPromptTester(args.url)
    
    # 检查服务状态
    print("🔍 检查服务状态...")
    if not tester.check_service_health():
        print(f"❌ 无法连接到服务: {args.url}")
        print("请确保S-LoRAc服务正在运行")
        sys.exit(1)
    print("✅ 服务运行正常")
    
    # 测试配置
    system_prompt = "你是一个有用的AI助手，专门帮助用户解答编程相关的问题。请用专业且易懂的语言回答问题。"
    test_prompts = [
        "什么是Python中的装饰器？",
        "如何优化深度学习模型的训练速度？",
        "解释一下React的useState钩子。"
    ]
    
    lora_dirs_to_test = args.lora_dirs if args.lora_dirs else [None]
    
    # 步骤1：初始化system_prompt缓存
    if not args.skip_init:
        print(f"\n🚀 步骤1: 初始化System Prompt缓存")
        init_result = tester.init_system_prompt_cache(system_prompt, args.lora_dirs)
        
        if "error" in init_result:
            print("初始化失败，退出测试")
            sys.exit(1)
        
        print("⏳ 等待缓存初始化完成...")
        time.sleep(2)
    else:
        print("⏭️  跳过缓存初始化步骤")
    
    # 步骤2: 基础功能测试
    print(f"\n🧪 步骤2: 基础功能测试")
    test_lora = lora_dirs_to_test[0]
    
    # 测试常规推理
    print("\n📝 测试常规推理...")
    normal_result = tester.generate_text(
        prompt=f"{system_prompt}\n\n{test_prompts[0]}",
        lora_dir=test_lora,
        use_system_prompt=False,
        max_tokens=args.max_tokens
    )
    
    if "error" not in normal_result:
        print(f"✅ 常规推理成功 ({normal_result['generation_time']:.2f}s)")
        print(f"输出: {normal_result.get('generated_text', [''])[0][:100]}...")
    else:
        print(f"❌ 常规推理失败: {normal_result['error']}")
    
    # 测试缓存推理
    print("\n⚡ 测试缓存推理...")
    cached_result = tester.generate_text(
        prompt=test_prompts[0],
        lora_dir=test_lora,
        use_system_prompt=True,
        system_prompt=system_prompt,
        max_tokens=args.max_tokens
    )
    
    if "error" not in cached_result:
        cache_status = "✅ 缓存命中" if cached_result.get("used_system_cache") else "❌ 缓存未命中"
        print(f"✅ 缓存推理成功 ({cached_result['generation_time']:.2f}s) | {cache_status}")
        print(f"输出: {cached_result.get('generated_text', [''])[0][:100]}...")
    else:
        print(f"❌ 缓存推理失败: {cached_result['error']}")
    
    # 步骤3: 性能对比测试
    print(f"\n⚡ 步骤3: 性能对比测试")
    for i, prompt in enumerate(test_prompts[:2]):  # 测试前2个prompt
        print(f"\n📊 测试prompt {i+1}: {prompt[:30]}...")
        benchmark_result = tester.benchmark_comparison(
            system_prompt=system_prompt,
            user_prompt=prompt,
            lora_dir=test_lora,
            max_tokens=args.max_tokens,
            rounds=args.rounds
        )
    
    # 步骤4: 多LoRA测试（如果有多个LoRA）
    if len(lora_dirs_to_test) > 1 and None not in lora_dirs_to_test:
        print(f"\n🔄 步骤4: 多LoRA测试")
        tester.test_multiple_lora(
            system_prompt=system_prompt,
            user_prompt=test_prompts[2],
            lora_dirs=[d for d in lora_dirs_to_test if d is not None],
            max_tokens=args.max_tokens
        )
    
    print(f"\n🎉 测试完成!")
    print(f"🔗 如需查看更多日志，请检查S-LoRAc服务的输出")


if __name__ == "__main__":
    main() 