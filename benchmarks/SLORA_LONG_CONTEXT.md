# 10,000-token Request Check

This checks full-input prefill and subsequent decode beyond LLaMA's trained
context length. It does not establish generation quality or numerical equivalence
to a long-context-trained model. No sliding window, input truncation, position
wrapping, or computation bypass is used.

The ordinary RoPE table in this repository already has 65,536 extra positions.
For LLaMA-7B, it covers this experiment without changing weights, head dimensions,
layers, or attention kernels. The worker now restores `max_req_total_len` from
the RPC configuration so its `b_loc` table agrees with the HTTP input limits.

## Server (Physical GPU 0)

Run from `/home/qlh/S-LoRAc`. Do not enable LowRA for this check: all 10,000 input
tokens must go through prefill. One rank-16 dummy adapter exercises the existing
LoRA computation as well as the base model. Dummy weights preserve model shapes
and execution, but not meaningful outputs.

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/slora/bin/python -m slora.server.api_server \
  --host 127.0.0.1 --port 8200 \
  --nccl_port 28767 --internal-port-start 14000 \
  --model_dir huggyllama/llama-7b --tokenizer_mode auto \
  --lora-dirs dummy-lora-7b-rank-16-0 \
  --max_total_token_num 16384 \
  --max_req_input_len 10000 --max_req_total_len 10064 \
  --batch_max_tokens 10064 --running_max_req_size 1 \
  --model-seed 42 --dummy --swap
```

The KV budget also needs room for the adapter. These 16,384 slots are independent
of the model's original 2,048-position training limit. Make sure the selected GPU
has enough free memory for weights, the allocated pool, and long-input activations.

## Client

In another terminal, after the server finishes initialization:

```bash
/home/qlh/anaconda3/envs/slora/bin/python benchmarks/slora_long_context_exp.py \
  --server http://127.0.0.1:8200 \
  --lora-dir dummy-lora-7b-rank-16-0 \
  --input-len 10000 --output-len 64 \
  --num-requests 1 --timeout 600 \
  --output benchmarks/results/slora_llama7b_input10000_output64.json
```

The input is a list of exactly 10,000 token IDs (ID 10 by default). There is no
tokenizer round trip or cached prefix injection. A pass requires reported input
length 10,000, a completed stream, and cumulative output count 64, with finite
reported token log probabilities. HTTP 200 alone is not a pass. There are 63
decode forward passes after the first token produced by prefill.

`ttft_s` and `latency_s` are client-observed seconds, including HTTP, scheduling,
and detokenization, not isolated GPU prefill/decode times. SSE events may coalesce
several tokens, so `stream_events` need not equal `output_tokens`.

Use `--num-requests 3` to repeat sequentially. Failure stops the run, writes the
error to JSON, and returns a nonzero exit status. Repeated success alone does not
prove absence of KV leaks. The client does not measure GPU memory peaks.

LLaMA-13B/30B use the same client: change both server model and matching adapter
name. Larger models require more memory; an OOM is not evidence that RoPE refuses
positions beyond 2,048. Real-weight tests should omit `--dummy` and use a real,
compatible adapter directory instead.
