# Real Model End-to-End Audit

## Scope

Executed on physical GPU0 only, with the existing idle GPU0 holder preserved.
No `accexp` files, scheduler rules or KV pool budgets were changed.

- Actual complete `huggyllama/llama-7b` checkpoint: 32 layers, hidden size 4096.
- Real rank-16 attention q/k/v/o adapter: `/data/hub/alpaca_cleaned-lora/7b`.
- Real rank-8 q/v adapter: `/data/hub/alpaca_mt-lora/7b`.
- Missing q/v-only adapter targets receive exactly zero A/B delta. Declared
  targets missing from a checkpoint are rejected, not silently zero-filled.
- Prefix lengths 129 and 513; three variable-length queries per batch, assigned
  to adapters `[0,1,0]`; eight output tokens per request.
- Reference: Transformers 4.57.3, PEFT 0.18.1, Torch 2.11.0+cu130, eager attention.
- Serving: Torch 2.0.1+cu117, Triton 2.1.0, real S-LoRA model/RPC execution.
- Base and adapter weights both FP16; PEFT's adapter FP32 autocast disabled.

BaseKV comes from base-model prefix prefill. MiniK/V are actual forward-hook
outputs of the active adapter's LoRA-A during LoRA prefix prefill. Full LoRA KV
and independently reconstructed MiniKV caches are saved separately.

The independent reconstructed reference computes raw delta-K in FP32 from the
original LoRA-B, rounds that delta to FP16 (as the serving destination does),
applies prefix-position RoPE, and adds already-rotated BaseK. V is added directly.
Transformers consumes these caches using fresh `DynamicCache` instances.

## Bug Found and Fixed

Before this audit the serving reconstruction did not rotate delta-K. Its
linear math was validated, but adding unrotated delta-K to rotated BaseK is not
the actual model's prefix semantics. Serial and parallel agreed with each other
yet both failed the independent reconstructed-cache oracle.

| Prefix | Serial/Parallel vs Cached Oracle Max Logit Error Before | After |
| ---: | ---: | ---: |
| 129 | 1.187500 | 0.031250 |
| 513 | 0.8515625 | 0.02734375 |

The correction writes raw delta-K into the existing destination workspace, then
rotates both halves and adds BaseK in place. It allocates no additional full
prefix KV tensor. V remains unchanged. Each RoPE program owns both half-vectors,
preventing in-place read/write races. The existing stream/event ordering also
includes the new postprocessing work before prefix readiness is signaled.

`LowRAStaticPrefixStore.bind_key_rope` explicitly declares the raw-MiniK /
post-RoPE-BaseK contract. Normal model-RPC initialization and the fixed-batch
profile builder bind their model's prefix RoPE tables, including dummy runs.
Standalone unbound stores retain the generic linear reconstruction contract.
The initial fused optional branch triggered a Triton 2.1 compiler assertion;
the equivalent scalar base-weight form passed all subsequent tests.

## Final Results

Each real-model case checks every vocabulary logit for each decode step.
Reference histories are teacher-forced to make comparisons use the same input
tokens. Every next-token argmax must additionally match the reference; all did,
so the tested greedy histories also coincide through all eight steps.

| Path | Reference | Worst Max Abs Logit Error | Token Agreement |
| --- | --- | ---: | ---: |
| Ordinary S-LoRA | HF/PEFT full-LoRA cache | 0.051758 | 100% |
| Full reprefill/reuse | HF/PEFT full-LoRA cache | 0.078125 | 100% |
| Independently injected full MiniKV cache | HF/PEFT reconstructed cache | 0.039063 | 100% |
| Layer serial | HF/PEFT reconstructed cache | 0.050781 | 100% |
| Layer parallel | HF/PEFT reconstructed cache | 0.050781 | 100% |
| Parallel vs serial | Same source KV and inputs | 0 | 100% |

Declared logit tolerances were `atol=0.1, rtol=0.01` across inference stacks,
`atol=0.05, rtol=0.005` against the same-engine cached MiniKV oracle, and
`atol=0.01, rtol=0.001` for serial vs parallel. All outputs must be finite and
all argmaxes agree. No tolerances were relaxed after observing results.

There are 48 generated token positions per mode (2 prefixes x 3 requests x 8).
All pool allocations used by requests, prefix caches and workspaces were freed
after each mode; loaded source-adapter slots remained protected.

Regression: 70 tests in `test_lowra_*.py` plus 9 in `test_full_reconstruct.py`
passed on GPU0. These regressions include small dummy-model lifecycle tests in
addition to the complete real-checkpoint audit above.

## Artifacts and Reproduction

- Driver: `try/validate_model_e2e.py`.
- Final FP16 reference and source caches: `try/results/model_e2e_gpu0_real7b_fp16_20261106/`.
- `reference.json`: reference environment, paths, precision and workload.
- `prefix129.pt`, `prefix513.pt`: actual sources, original full caches, independent
  reconstructed caches, queries, reference logits and generated tokens.
- `serving_rope0.json`: failed pre-fix negative control; preserved unchanged.
- `serving_rope1.json`: final passing results and source hashes.
- `serving_prefix*_rope1.pt`: actual serving logits.
- Adjacent `*_reference.log`, `*_serving_before.log`, `*_serving_after.log`,
  `*_serving_final.log` contain the actual console outputs.
- Regression logs are inside the result directory. The first failed compiler
  run is preserved as `key_rope_unit_tests.log`; `key_rope_unit_tests_v2.log`
  records the passing replacement.

```bash
MODEL=/home/qlh/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549
RUN="try/results/model_e2e_gpu0_$(date -u +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /home/qlh/anaconda3/envs/transformer/bin/python -u try/validate_model_e2e.py \
  --stage reference --model "$MODEL" \
  --adapters /data/hub/alpaca_cleaned-lora/7b /data/hub/alpaca_mt-lora/7b \
  --prefix-lengths 129 513 --output-len 8 --output-dir "$RUN"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/qlh/anaconda3/envs/slora/bin/python -u try/validate_model_e2e.py \
  --stage serving --key-rope --model "$MODEL" \
  --adapters /data/hub/alpaca_cleaned-lora/7b /data/hub/alpaca_mt-lora/7b \
  --prefix-lengths 129 513 --output-len 8 --output-dir "$RUN"
```

## Limits

This proves the tested complete real-model execution agrees with its independent
reference within the declared FP16 tolerances. It does not prove that MiniKV
approximates full LoRA prefill without loss. First-token full-LoRA versus
MiniKV-reference argmax agreement was 2/3 for each prefix in these examples;
this method-level discrepancy is reported separately and is not masked by
serial/parallel agreement. This is not a dataset accuracy score.

Actual BaseKV/MiniKV sources are injected by the audit driver. The ordinary
server still initializes placeholder source tensors; this change does not add
automatic real-prefix precomputation or an HTTP source-loading interface.
The audit exercises model/RPC prefill and decode, not an HTTP load test, long
workload replay, or 13B/30B/65B checkpoints. No post-fix throughput claim is made.
Old timings without both packed-B decoding and delta-K RoPE must be rerun.
