from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型
model_name = "huggyllama/llama-7b"  # 或者改成你自己的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 第一步：预编码 system prompt
system_prompt = "You are a helpful assistant."
system_inputs = tokenizer.encode(system_prompt)
print(f"System prompt IDs: {system_inputs}")
ids = tokenizer(system_prompt)
print(ids)