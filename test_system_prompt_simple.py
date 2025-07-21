#!/usr/bin/env python3
"""
简化的System Prompt KV Cache测试脚本
仅测试基本功能，避免复杂的多LoRA初始化
"""

import requests
import json
import time
from typing import Dict, List, Optional

class SimpleSystemPromptTester:
    def __init__(self, base_url="http://localhost:8000", timeout=200):
        self.base_url = base_url
        self.timeout = timeout
        
    def _post(self, endpoint: str, data: Dict, timeout=None) -> Dict:
        """发送POST请求"""
        if timeout is None:
            timeout = self.timeout
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        print(f"  请求URL: {url}")
        print(f"  请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(url, json=data, headers=headers, timeout=timeout)
        result = response.json()
        
        print(f"  响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return result
    
    def check_service(self):
        """检查服务状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 服务运行正常")
                return True
        except Exception as e:
            print(f"❌ 服务不可用: {e}")
            return False
    
    def test_simple_init(self):
        """测试简单的system prompt初始化（只指定一个LoRA）"""
        print("\n🚀 测试简单初始化（单个LoRA）")
        system_prompt = "你是一个AI助手。"
        
        # 先只测试一个特定的LoRA
        data = {
            "system_prompt": system_prompt,
            "lora_dirs": ["tloen/alpaca-lora-7b"]  # 只测试一个
        }
        
        start_time = time.time()
        try:
            result = self._post("/init_system_prompt", data, timeout=200)  # 增加超时时间到3分钟以上
            elapsed = time.time() - start_time
            print(f"初始化耗时: {elapsed:.2f}秒")
            
            if result.get("status") == "success":
                print("✅ 简单初始化成功")
                return True
            else:
                print(f"❌ 初始化失败: {result}")
                return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"初始化耗时: {elapsed:.2f}秒")
            print(f"❌ 初始化异常: {e}")
            return False
    
    def test_base_model_only(self):
        """测试只初始化基础模型（不包含任何LoRA）"""
        print("\n🚀 测试基础模型初始化")
        system_prompt = "你是一个AI助手。"
        
        # 空列表应该意味着只初始化基础模型
        data = {
            "system_prompt": system_prompt,
            "lora_dirs": []
        }
        
        start_time = time.time()
        try:
            result = self._post("/init_system_prompt", data, timeout=60)
            elapsed = time.time() - start_time
            print(f"初始化耗时: {elapsed:.2f}秒")
            
            if result.get("status") == "success":
                print("✅ 基础模型初始化成功")
                return True
            else:
                print(f"❌ 初始化失败: {result}")
                return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"初始化耗时: {elapsed:.2f}秒")
            print(f"❌ 初始化异常: {e}")
            return False

    def test_simple_inference(self):
        """测试简单的推理请求"""
        print("\n🚀 测试简单推理")
        
        data = {
            "adapter": "tloen/alpaca-lora-7b",
            "inputs": "Hello, how are you?",
            "parameters": {
                "max_new_tokens": 20,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        try:
            result = self._post("/generate", data, timeout=30)
            print("✅ 推理请求成功")
            return True
        except Exception as e:
            print(f"❌ 推理请求失败: {e}")
            return False

def main():
    print("🔍 简化System Prompt缓存测试")
    
    tester = SimpleSystemPromptTester()
    
    # 1. 检查服务状态
    if not tester.check_service():
        print("服务不可用，请先启动服务器")
        return
    
    # 2. 先测试普通推理是否工作
    print("\n" + "="*50)
    print("测试基础功能")
    if not tester.test_simple_inference():
        print("基础推理都不工作，请检查服务配置")
        return
    
    # 3. 测试只初始化基础模型
    print("\n" + "="*50)
    print("测试基础模型初始化")
    if tester.test_base_model_only():
        print("✅ 基础模型初始化成功！")
    
    # 4. 测试单个LoRA初始化
    print("\n" + "="*50)
    print("测试单个LoRA初始化")
    if tester.test_simple_init():
        print("✅ 单个LoRA初始化成功！")
    
    print("\n" + "="*50)
    print("测试完成")

if __name__ == "__main__":
    main() 