#!/bin/bash

# S-LoRAc System Prompt KV缓存功能使用示例

echo "=== S-LoRAc System Prompt KV缓存功能演示 ==="

# 1. 启动S-LoRAc服务（请根据实际情况修改路径）
echo "步骤1: 启动S-LoRAc服务..."
echo "请确保已运行以下命令启动服务：" 
echo "python slora/server/api_server.py \\"
echo "  --model_dir /path/to/your/base/model \\"
echo "  --lora-dirs /path/to/lora1 /path/to/lora2 /path/to/lora3 \\"
echo "  --tp 1 \\"
echo "  --max_total_token_num 6000 \\"
echo "  --batch_max_tokens 2000"
echo ""

# 2. 等待用户确认服务已启动
read -p "服务已启动？按回车键继续..."

# 3. 初始化system_prompt缓存
echo "步骤2: 初始化system_prompt缓存..."
curl -X POST http://localhost:8000/init_system_prompt \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "你是一个有用的AI编程助手。请用专业且易懂的语言回答编程相关问题，提供清晰的代码示例。",
    "lora_dirs": []
  }'

echo -e "\n"

# 4. 测试常规推理
echo "步骤3: 测试常规推理（完整prompt）..."
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "你是一个有用的AI编程助手。请用专业且易懂的语言回答编程相关问题，提供清晰的代码示例。\n\n什么是Python装饰器？请提供一个简单的例子。",
    "lora_dir": null,
    "parameters": {
      "max_new_tokens": 100,
      "temperature": 0.7,
      "do_sample": true
    }
  }'

echo -e "\n\n"

# 5. 测试基于缓存的推理
echo "步骤4: 测试缓存推理（复用system_prompt）..."
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "什么是Python装饰器？请提供一个简单的例子。",
    "lora_dir": null,
    "use_system_prompt": true,
    "system_prompt": "你是一个有用的AI编程助手。请用专业且易懂的语言回答编程相关问题，提供清晰的代码示例。",
    "parameters": {
      "max_new_tokens": 100,
      "temperature": 0.7,
      "do_sample": true
    }
  }'

echo -e "\n\n"

# 6. 运行完整测试
echo "步骤5: 运行完整性能测试..."
python test_system_prompt_cache.py --rounds 3 --max-tokens 80

echo ""
echo "=== 演示完成 ==="
echo "注意观察:"
echo "1. 初始化时间：第一次需要预计算所有LoRA的KV缓存"
echo "2. 推理时间对比：缓存推理应该比常规推理快30-70%"
echo "3. 缓存命中状态：应该显示✅缓存命中"
echo "4. 输出质量：两种方式的输出质量应该基本相同" 