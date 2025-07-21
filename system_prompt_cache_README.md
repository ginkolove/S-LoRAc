# S-LoRAc System Prompt KV Cache 使用指南

## 🎯 功能简介

System Prompt KV Cache 功能允许您预先计算和存储系统提示词的 KV 状态，在后续推理中复用这些缓存，从而显著提升推理性能。

## 📁 脚本文件

- `launch_with_system_prompt.py`: 服务器启动和初始化脚本
- `test_system_prompt_benchmark.py`: 性能测试和对比脚本
- `example_system_prompt_usage.sh`: 使用示例和命令参考

## 🚀 快速开始

### 1. 一键启动和测试

```bash
# 启动服务器并自动初始化system_prompt缓存
python launch_with_system_prompt.py --device debug --auto-init --test-inference

# 在另一个终端运行性能对比测试
python test_system_prompt_benchmark.py --mode compare --num-requests 5
```

### 2. 手动分步操作

#### 步骤 1: 启动服务器
```bash
python -m slora.server.api_server \
  --max_total_token_num 8000 \
  --model ../model_data/Llama-2-7b-hf \
  --lora ../lora_test_data/finiteautomata-bertweet-base-sentiment-analysis \
  --tokenizer_mode auto \
  --swap
```

#### 步骤 2: 初始化system_prompt缓存
```bash
curl -X POST http://localhost:8000/init_system_prompt \
  -H 'Content-Type: application/json' \
  -d '{
    "model_dir": "../model_data/Llama-2-7b-hf",
    "lora_dirs": ["../lora_test_data/finiteautomata-bertweet-base-sentiment-analysis"],
    "system_prompt": "You are a helpful AI assistant."
  }'
```

#### 步骤 3: 使用缓存进行推理
```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model_dir": "../model_data/Llama-2-7b-hf",
    "lora_dir": "../lora_test_data/finiteautomata-bertweet-base-sentiment-analysis",
    "inputs": "What is machine learning?",
    "system_prompt_hash": "<从步骤2返回的hash>",
    "parameters": {
      "max_new_tokens": 50,
      "do_sample": false
    }
  }'
```

## 📊 性能测试

### 对比测试（推荐）
```bash
# 对比有无缓存的性能差异
python test_system_prompt_benchmark.py --mode compare --num-requests 10

# 自定义测试参数
python test_system_prompt_benchmark.py \
  --mode compare \
  --num-requests 20 \
  --concurrent 2 \
  --output-len 100 \
  --output-file my_benchmark.json
```

### 单独测试
```bash
# 仅测试使用缓存
python test_system_prompt_benchmark.py --mode with_cache --num-requests 10

# 仅测试不使用缓存
python test_system_prompt_benchmark.py --mode without_cache --num-requests 10
```

## 🔧 启动脚本选项

### 基本用法
```bash
# 调试模式（资源需求少）
python launch_with_system_prompt.py --device debug --auto-init

# A10G GPU配置
python launch_with_system_prompt.py --device a10g --auto-init

# 自定义系统提示词
python launch_with_system_prompt.py \
  --device debug \
  --auto-init \
  --system-prompt "你是一个有用的AI助手，请用中文回答。"
```

### 高级选项
```bash
# 仅初始化缓存（不启动服务器）
python launch_with_system_prompt.py --init-only --auto-init

# 连接现有服务器并初始化
python launch_with_system_prompt.py --wait-only --auto-init

# 启用各种功能
python launch_with_system_prompt.py \
  --device debug \
  --auto-init \
  --bmm \
  --enable-abort \
  --prefetch
```

## 📈 测试脚本选项

### 基本参数
```bash
python test_system_prompt_benchmark.py \
  --server http://localhost:8000 \
  --model-dir ../model_data/Llama-2-7b-hf \
  --mode compare \
  --num-requests 10 \
  --output-len 50
```

### 并发测试
```bash
python test_system_prompt_benchmark.py \
  --mode compare \
  --concurrent 3 \
  --num-requests 15
```

## 💡 注意事项

1. **内存需求**: system_prompt缓存需要额外的GPU内存，建议先用`--device debug`测试
2. **初始化时间**: 首次初始化可能需要较长时间，特别是有多个LoRA适配器时
3. **路径配置**: 确保模型和LoRA文件路径正确
4. **超时设置**: 初始化过程可能需要几分钟，脚本已设置合适的超时时间

## 🔍 故障排除

### 常见错误

1. **ImportError**: 运行`export PYTHONPATH=.`或使用`python -m`方式启动
2. **超时错误**: 增加`--num-token`或减少`--num-adapter`
3. **内存不足**: 使用`--device debug`减少资源需求
4. **端口占用**: 使用`--port 8001`指定其他端口

### 调试信息

```bash
# 查看详细日志
python launch_with_system_prompt.py --device debug --auto-init --test-inference

# 检查服务器状态
curl http://localhost:8000/health
```

## 📚 API文档

### 初始化接口
- **URL**: `/init_system_prompt`
- **方法**: POST
- **参数**:
  - `model_dir`: 基础模型目录
  - `lora_dirs`: LoRA适配器目录列表
  - `system_prompt`: 系统提示词文本

### 推理接口
- **URL**: `/generate` 或 `/generate_stream`
- **方法**: POST
- **参数**:
  - `model_dir`: 基础模型目录
  - `lora_dir`: LoRA适配器目录（可选）
  - `inputs`: 用户输入文本
  - `system_prompt_hash`: 缓存哈希值（使用缓存时）
  - `parameters`: 生成参数

## 🎯 最佳实践

1. **开发阶段**: 使用`--device debug`快速测试
2. **生产环境**: 根据硬件选择合适的`--device`参数
3. **性能优化**: 先运行对比测试了解性能提升
4. **监控资源**: 注意GPU内存使用情况
5. **批量处理**: 对于相同system_prompt的请求，复用同一个缓存

## 🚀 进阶使用

查看更多选项和参数：
```bash
python launch_with_system_prompt.py --help
python test_system_prompt_benchmark.py --help
``` 