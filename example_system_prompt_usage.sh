#!/bin/bash

echo "🚀 S-LoRAc System Prompt KV Cache 使用示例"
echo "=" x50

# 设置变量
SERVER_URL="http://localhost:8000"
MODEL_DIR="../model_data/Llama-2-7b-hf"
LORA_DIR="../lora_test_data/finiteautomata-bertweet-base-sentiment-analysis"

echo "📋 使用配置:"
echo "  - 服务器地址: $SERVER_URL"
echo "  - 基础模型: $MODEL_DIR"
echo "  - LoRA适配器: $LORA_DIR"
echo

# 方法一: 使用启动脚本自动启动服务器并初始化缓存
echo "🔧 方法一: 自动启动服务器并初始化system_prompt缓存"
echo "命令: python launch_with_system_prompt.py --device debug --auto-init --test-inference"
echo

# 方法二: 手动启动服务器然后初始化缓存
echo "🔧 方法二: 手动启动和初始化"
echo

echo "步骤 1: 启动服务器"
echo "python -m slora.server.api_server \\"
echo "  --max_total_token_num 8000 \\"
echo "  --model $MODEL_DIR \\"
echo "  --lora $LORA_DIR \\"
echo "  --tokenizer_mode auto \\"
echo "  --swap"
echo

echo "步骤 2: 等待服务器启动后，初始化system_prompt缓存"
echo "curl -X POST $SERVER_URL/init_system_prompt \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"model_dir\": \"$MODEL_DIR\","
echo "    \"lora_dirs\": [\"$LORA_DIR\"],"
echo "    \"system_prompt\": \"You are a helpful AI assistant.\""
echo "  }'"
echo

echo "步骤 3: 使用system_prompt缓存进行推理"
echo "curl -X POST $SERVER_URL/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"model_dir\": \"$MODEL_DIR\","
echo "    \"lora_dir\": \"$LORA_DIR\","
echo "    \"inputs\": \"What is machine learning?\","
echo "    \"system_prompt_hash\": \"<从步骤2返回的hash>\","
echo "    \"parameters\": {"
echo "      \"max_new_tokens\": 50,"
echo "      \"do_sample\": false"
echo "    }"
echo "  }'"
echo

# 方法三: 使用Python测试脚本
echo "🔧 方法三: 使用性能测试脚本"
echo

echo "对比测试 (有无缓存):"
echo "python test_system_prompt_benchmark.py --mode compare --num-requests 5"
echo

echo "仅测试使用缓存:"
echo "python test_system_prompt_benchmark.py --mode with_cache --num-requests 10"
echo

echo "仅测试不使用缓存:"
echo "python test_system_prompt_benchmark.py --mode without_cache --num-requests 10"
echo

# 实际运行示例
echo "🎯 快速开始 (调试模式)"
echo "如果你想立即测试，运行以下命令:"
echo
echo "# 启动服务器并自动初始化缓存"
echo "python launch_with_system_prompt.py --device debug --auto-init"
echo
echo "# 在另一个终端运行性能测试"
echo "python test_system_prompt_benchmark.py --mode compare --num-requests 3"
echo

echo "💡 提示:"
echo "1. 确保模型和LoRA文件路径正确"
echo "2. 首次运行可能需要较长时间加载模型"
echo "3. system_prompt初始化需要额外的GPU内存"
echo "4. 使用--device debug可以减少资源需求"
echo

echo "📚 更多选项请查看:"
echo "python launch_with_system_prompt.py --help"
echo "python test_system_prompt_benchmark.py --help" 