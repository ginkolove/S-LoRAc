# S-LoRAc System Prompt KV缓存测试指南

## 概述

这个测试脚本验证S-LoRAc的system_prompt KV缓存功能，包括：
- System prompt缓存的初始化
- 基于缓存的推理性能对比
- 多LoRA缓存功能测试

## 前提条件

1. **启动S-LoRAc服务**
   ```bash
   python slora/server/api_server.py \
     --model_dir /path/to/base/model \
     --lora-dirs /path/to/lora1 /path/to/lora2 /path/to/lora3 \
     --tp 1 \
     --max_total_token_num 6000 \
     --batch_max_tokens 2000
   ```

2. **安装依赖**
   ```bash
   pip install requests
   ```

## 关键概念说明

### LoRA目录的处理逻辑

⚠️ **重要**: `lora_dirs: []` 的含义

- 当初始化system_prompt时传入 `"lora_dirs": []`（空数组）
- 系统会**自动为服务启动时的所有LoRA**初始化system_prompt缓存
- 这些LoRA是通过启动参数 `--lora-dirs` 指定的

**示例**：
```bash
# 1. 启动服务时指定3个LoRA
python slora/server/api_server.py --lora-dirs /lora1 /lora2 /lora3 ...

# 2. 初始化缓存时传入空数组
curl -X POST http://localhost:8000/init_system_prompt \
  -d '{"system_prompt": "...", "lora_dirs": []}'

# 结果: 会为/lora1, /lora2, /lora3这3个LoRA都初始化system_prompt缓存
```

## 使用方法

### 1. 基础测试（自动检测所有LoRA）

```bash
python test_system_prompt_cache.py
```

### 2. 指定特定LoRA进行测试

```bash
python test_system_prompt_cache.py --lora-dirs /path/to/lora1 /path/to/lora2
```

### 3. 性能测试（多轮次）

```bash
python test_system_prompt_cache.py --rounds 5 --max-tokens 100
```

### 4. 跳过初始化（缓存已存在）

```bash
python test_system_prompt_cache.py --skip-init
```

### 5. 测试不同服务地址

```bash
python test_system_prompt_cache.py --url http://192.168.1.100:8000
```

## 测试流程

### 步骤1: 缓存初始化
- 发送system_prompt到服务器
- 为指定的LoRA（或所有LoRA）预计算KV缓存
- 验证初始化是否成功

### 步骤2: 基础功能测试
- **常规推理**: 发送完整prompt（system_prompt + user_prompt）
- **缓存推理**: 只发送user_prompt，复用system_prompt缓存
- 验证两种方式都能正常工作

### 步骤3: 性能对比测试
- 多轮次测试常规推理和缓存推理
- 统计平均耗时和加速比
- 展示性能提升效果

### 步骤4: 多LoRA测试
- 测试不同LoRA都能正确使用system_prompt缓存
- 验证缓存隔离性

## 预期结果

### 性能提升指标

- **加速比**: 1.5x - 3x（取决于system_prompt长度）
- **时间节省**: 30% - 70%
- **缓存命中**: ✅ 应该显示缓存命中

### 日志输出示例

```
🔍 检查服务状态...
✅ 服务运行正常

🚀 步骤1: 初始化System Prompt缓存
正在初始化system_prompt缓存...
  system_prompt: 你是一个有用的AI助手，专门帮助用户解答编程相关的问题。请用专业且易懂的...
  lora_dirs: [] (所有LoRA)
初始化耗时: 5.23秒
✅ 初始化成功: System prompt KV cache initialized for 3 adapters

🧪 步骤2: 基础功能测试

📝 测试常规推理...
✅ 常规推理成功 (2.45s)

⚡ 测试缓存推理...
✅ 缓存推理成功 (0.89s) | ✅ 缓存命中

⚡ 步骤3: 性能对比测试

============================================================
性能对比测试 (轮次: 3)
system_prompt: 你是一个有用的AI助手，专门帮助用户解答编程相关的...
user_prompt: 什么是Python中的装饰器？...
lora_dir: /path/to/lora1
============================================================

📊 常规推理测试 (完整prompt):
  轮次 1/3... 2.34s (生成50个token)
  轮次 2/3... 2.41s (生成50个token)
  轮次 3/3... 2.38s (生成50个token)

🚀 缓存推理测试 (复用system_prompt):
  轮次 1/3... 0.87s (生成50个token) ✅
  轮次 2/3... 0.92s (生成50个token) ✅
  轮次 3/3... 0.89s (生成50个token) ✅

📈 性能统计:
  常规推理平均时间: 2.377s
  缓存推理平均时间: 0.893s
  加速比: 2.66x
  时间节省: 62.4%

🎉 测试完成!
```

## 故障排除

### 1. 连接失败
```
❌ 无法连接到服务: http://localhost:8000
```
**解决方案**: 检查S-LoRAc服务是否正在运行

### 2. 初始化失败
```
❌ 初始化失败: system_prompt is required
```
**解决方案**: 检查system_prompt是否为空

### 3. 缓存未命中
```
✅ 缓存推理成功 (2.45s) | ❌ 缓存未命中
```
**可能原因**:
- system_prompt文本不匹配
- 缓存未正确初始化
- LoRA路径不匹配

### 4. 推理失败
```
❌ 缓存推理失败: Error message
```
**解决方案**: 查看S-LoRAc服务日志获取详细错误信息

## API参考

### 初始化缓存 API
```bash
POST /init_system_prompt
Content-Type: application/json

{
  "system_prompt": "系统提示词内容",
  "lora_dirs": []  // 空数组 = 所有LoRA，或指定具体路径
}
```

### 缓存推理 API
```bash
POST /generate
Content-Type: application/json

{
  "inputs": "用户输入",
  "lora_dir": "/path/to/lora1",
  "use_system_prompt": true,
  "system_prompt": "系统提示词内容（必须与初始化时完全一致）",
  "parameters": {
    "max_new_tokens": 100,
    "temperature": 0.7
  }
}
```

## 注意事项

1. **一致性**: 推理时的`system_prompt`必须与初始化时完全一致
2. **内存**: 每个LoRA都会分配system_prompt的KV缓存空间
3. **并发**: 多个请求可以同时使用相同的system_prompt缓存
4. **更新**: 如需更改system_prompt，需要重新初始化缓存 