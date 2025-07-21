import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
import time
import asyncio
import aiohttp

SYS_PROMPT="You are a helpful, knowledgeable, and friendly AI assistant. Your job is to answer user questions accurately, clearly, and politely. You can help with a wide range of topics, including science, technology, writing, coding, history, and daily life. Always prioritize clarity, truthfulness, and usefulness. If a question is ambiguous, ask for clarification. Do not guess or make up information. Use a professional and respectful tone. When appropriate, offer concise examples. Do not provide medical, legal, or financial advice."

def send_sys_prompt_to_init(server_url="http://127.0.0.1:8000"):
    """
    直接发送SYS_PROMPT到/init_system_prompt接口
    """
    url = f"{server_url}/init_system_prompt"
    payload = {
        "system_prompt": SYS_PROMPT,
        "lora_dirs": []  # 空列表表示初始化所有LoRA
    }
    
    print("🚀 发送SYS_PROMPT到/init_system_prompt接口")
    print("=" * 60)
    print(f"服务器地址: {server_url}")
    print(f"接口地址: {url}")
    print(f"System prompt长度: {len(SYS_PROMPT)} 字符")
    print(f"System prompt内容:\n{SYS_PROMPT[:500]}{'...' if len(SYS_PROMPT) > 600 else ''}")
    print("\n正在发送请求...")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        end_time = time.time()
        
        print(f"请求耗时: {end_time - start_time:.2f}秒")
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 成功！")
            print(f"状态: {result.get('status', 'unknown')}")
            print(f"消息: {result.get('message', '')}")
            print(f"System prompt长度: {result.get('system_prompt_length', 0)} 字符")
            
            # 显示更多详细信息
            if 'detailed_info' in result:
                print(f"详细信息: {result.get('detailed_info', '')}")
            if 'adapter_dirs' in result and result.get('adapter_dirs'):
                print(f"初始化的适配器: {', '.join(result.get('adapter_dirs', []))}")
            
            return True
        else:
            print(f"\n❌ 失败！")
            print(f"错误信息: {response.text}")
            try:
                error_detail = response.json()
                print(f"详细错误: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
            except:
                pass
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保S-LoRAc服务器正在运行")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时！初始化可能需要更长时间")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 直接调用函数发送SYS_PROMPT
    success = send_sys_prompt_to_init()
    if success:
        print("\n🎉 SYS_PROMPT已成功发送并初始化缓存！")
    else:
        print("\n💥 发送失败，请检查服务器状态")
    
    exit(0 if success else 1)
