#!/usr/bin/env bash
# Fixed FP16, TP=1 profile: base-model weights + total pool about 79.1 decimal GB.
# 7B/13B keep the original A100-80GB paper budgets; 30B is matched in bytes.
# Model bytes are from runtime.base_model_weight_bytes measured on 2026-11-14.
# Model buffers (including RoPE), activations and CUDA costs are outside this sum.
# Freeze these values so a preceding run's allocator state cannot change budgets.
llama_memory_budget() {
  case "$1" in
    7b) LLAMA_MODEL_WEIGHT_BYTES=13476831232; LLAMA_LAYERS=32; LLAMA_HIDDEN=4096; LLAMA_POOL_TOKENS=125000 ;;
    13b) LLAMA_MODEL_WEIGHT_BYTES=26031728640; LLAMA_LAYERS=40; LLAMA_HIDDEN=5120; LLAMA_POOL_TOKENS=65000 ;;
    30b) LLAMA_MODEL_WEIGHT_BYTES=65057887232; LLAMA_LAYERS=60; LLAMA_HIDDEN=6656; LLAMA_POOL_TOKENS=8800 ;;
    *) printf 'Supported memory profiles: 7b, 13b, 30b\n' >&2; return 2 ;;
  esac
  LLAMA_SLOT_BYTES=$((4 * LLAMA_LAYERS * LLAMA_HIDDEN))
  LLAMA_WEIGHT_POOL_BYTES=$((LLAMA_MODEL_WEIGHT_BYTES + LLAMA_POOL_TOKENS * LLAMA_SLOT_BYTES))
}
