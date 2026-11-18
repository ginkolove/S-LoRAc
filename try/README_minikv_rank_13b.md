# Llama-13B MiniKV rank reconstruction profile

`profile_minikv_rank_13b.py` calls the serving function
`slora.common.lowra_layer_runtime.reconstruct_prefix_layer_bounded` directly.
It does not implement a replacement reconstruction kernel or change the server.

## Scope

- Llama-13B MHA shape: 40 layers, hidden size 5120, 40 heads, head dimension 128.
- FP16, TP=1, prefix length **10000** (not 10240), seed 42.
- All 50 adapters' MiniK/MiniV and shared BaseK/BaseV reside on GPU.
- Active weights go through `LoraLayerWeight.load_hf_weights_cpu` and
  `InferAdapter.load_adapters`, preserving the serving packed-B layout.
- Nonzero synthetic inputs and weights isolate shape-dependent execution cost.
  No pretrained model checkpoint or dataset is loaded. This is not an accuracy
  experiment or an end-to-end serving throughput benchmark.
- Each sweep reconstructs 40 distinct layers' source tensors into the **same
  one-layer workspace**, just as serial reconstruction overwrites its workspace.
  Allocation uses the actual Unified Paging allocator; it reserves
  `ceil(unique_adapters * prefix_length / layer_count)` full-layer slots for scratch,
  not a full 40-layer prefix for every adapter.
- K includes the production delta-K RoPE and BaseK addition. V includes scaling
  and BaseV addition. No attention, FFN, H2D, allocator or index preparation is
  inside the primary timed interval. The runtime wrapper and its Python launches
  are unchanged, so CUDA stream intervals can include host submission gaps.
- The default launch configuration is `(32, 128, 16, 4)`, with 1296 requested
  workers. `--block-r 32` changes only the rank tile width to 32. No autotuning
  is done; the Triton 2.1 FP16 dot path cannot use a rank tile width of 8.
- `--rank-loop max` is the unchanged default policy: every tile loops to the
  common metadata width. `--rank-loop actual` loops only to that tile's adapter
  rank. Both use the same packed-B indexing, zero-masked tail, workspace and
  K/V/RoPE operations. B index-table row stride remains the common metadata width
  in both modes. The new policy is opt-in; existing server launches stay on max/16.

## Cases

1. Homogeneous ranks 8/16/32/64, each with 1/2/4/8/16 unique active adapters: 20 cases.
2. Eight active adapters with rank mixes: seven 8 + one 64; four 8 + four 64;
   seven 16 + one 64; two each of 8/16/32/64; four 16 + four 32;
   five 16 + three 32; six 16 + one 32 + one 64; six 16 + two 64: 8 cases.
3. Diagnostic padding control: identical data, actual ranks and destinations to
   `r8_u8`, but metadata padded to rank 64. This is **not** a normal production
   configuration. It isolates the effect of the common maximum-rank loop width.

The same seeded permutation chooses active adapter IDs. A homogeneous case has
50 source adapters of its rank; mixed cases have rank-16 inactive adapters. Active
adapter count is not request count: requests using the same LoRA share one prefix.
At the same active count, output KV size is independent of rank. Sources and
weights increase with rank, as in serving.

The same cases, seeded sources and shuffled round order are used for all
`--rank-loop` / `--block-r` combinations. Each invocation is a separate process,
so physical allocation addresses and GPU timing conditions are not guaranteed
identical. Timing excludes compilation through validation and warmup.

## Validation And Timing

Before timing each case for the first time, first/middle/last layers are checked
against PyTorch FP32 `Mini @ original_HF_B.T * scaling`, including the production
FP16 delta-K rounding before RoPE. All active adapters and all elements of these
layers are checked in chunks. Protected allocator slots and original packed B
are also checked for accidental overwrites. A failed check stops the run.

Each case has 10 warmup sweeps and 50 measured sweeps per round, with 3 shuffled
rounds. One CUDA-event pair spans the entire 40-layer sweep, without per-layer
synchronization; synchronized wall time is recorded too. Compilation and setup
are excluded. A separate instrumented run records K matmul, V matmul/add and K
RoPE/add per layer; those event timings are diagnostics, not primary measurements.

This benchmark does not interleave FFN or attention between layers. Results
characterize the reconstruction path in isolation, not an exact prediction of
an online batch's latency or how it changes when admitting more requests.

The script does not inspect other GPU processes, wait for an idle GPU or stop
other processes. GPU occupancy is managed by the user; concurrent workloads can
affect latency measurements. Available-memory and numerical checks still run.

## GPU0 Command

Run on GPU0:

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="try/results/minikv_rank_13b_p10000_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 nohup /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_minikv_rank_13b.py \
  --prefix-len 10000 --pool-adapters 50 \
  --ranks 8 16 32 64 --unique-loras 1 2 4 8 16 \
  --workers 1296 --warmup 10 --iterations 50 --repeats 3 \
  --seed 42 --output-dir "$RUN_DIR" \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &
echo "PID=$! RUN_DIR=$RUN_DIR"
tail -f "$RUN_DIR/run.log"
```

List all cases without initializing CUDA:

```bash
/home/qlh/anaconda3/envs/slora/bin/python try/profile_minikv_rank_13b.py --list-cases
```

Optional short GPU correctness/CLI run. This keeps
the 13B hidden dimension but uses two layers and 129 tokens; its timings are not
13B 10K results:

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_minikv_rank_13b.py --smoke \
  --cases r8_u1 r16_u1 r32_u1 r64_u1 mix_7r8_1r64 r8_u8 r8_u8_padded64 \
  --warmup 1 --iterations 2 --repeats 1 --diagnostic-iterations 1 \
  --output-dir "try/results/rank_smoke_$(date +%Y%m%d_%H%M%S)"
```

## Outputs

- `result.json`: incrementally saved state, model shape, arguments, source hashes,
  correctness errors, memory accounting, execution order, raw CUDA/wall samples,
  and separate per-layer operation breakdowns. Require `state=complete` and
  `all_passed=true` before treating the suite as complete.
- `summary.csv`: one row per finished case, pooling rounds; full-sweep mean,
  p50/p95, synchronized wall mean, per-layer average, between-round variation,
  output workspace size, and validation error. Partial files can contain fewer
  cases/rounds; check `rounds` before comparison.
- `run.log`: preparation, validation, progress bars and per-case timings.

Compare homogeneous ranks at fixed `unique_loras`; compare mixed cases with
`r8_u8`, `r16_u8` and `r64_u8`. The padded diagnostic can reveal effects of max-rank
loop bounds even when actual rank is unchanged. Do not assume latency is linear
in rank, or multiply the unique-adapter reconstruction cost by request count.

## Actual-Rank And Rank-Tile Comparison

The launcher below runs 29 cases per configuration, including the mixed-rank and
padding controls above. It does not inspect other GPU processes, wait for idle,
kill jobs, or start a serving workload. On failure it stops that GPU's sequence
and keeps the log and partial report. Each `summary.csv` records `rank_loop` and
`block_r`; `result.json` also records launch configuration and source hashes.

GPU0 runs actual/16, then max/32. GPU1 runs actual/32. Run both commands from the
same shell so they use the same `RUN_DIR`:

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="try/results/minikv_rank_tiles_13b_p10000_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 nohup bash try/run_minikv_rank_tiles.sh \
  "$RUN_DIR/gpu0" actual 16 max 32 \
  > "$RUN_DIR/gpu0.log" 2>&1 < /dev/null &
echo "GPU0 PID=$! RUN_DIR=$RUN_DIR"

CUDA_VISIBLE_DEVICES=1 nohup bash try/run_minikv_rank_tiles.sh \
  "$RUN_DIR/gpu1" actual 32 \
  > "$RUN_DIR/gpu1.log" 2>&1 < /dev/null &
echo "GPU1 PID=$! RUN_DIR=$RUN_DIR"
```

Compare GPU0's `actual_br16` with `max_br32` and GPU1's `actual_br32`. The earlier
max/16 run remains in its original directory. For a contemporaneous fourth
control, append `max 16` to the GPU0 sequence. Do not infer a small cross-GPU
speedup from a single run without checking clock/load variation.

To run only the two rank-16/rank-32 mixtures, pass
`--cases mix_4r16_4r32 mix_5r16_3r32 --rank-loop actual --block-r 16`
to `profile_minikv_rank_13b.py`. Both cases select the same eight adapter IDs
with the same seed; only the fifth selected adapter changes rank between cases.

For mixtures with one or two rank-64 adapters, select
`--cases mix_7r16_1r64 mix_6r16_1r32_1r64 mix_6r16_2r64`
with `--rank-loop actual --block-r 16`. The same eight adapter IDs are selected
in all three cases; the seventh adapter has rank 16, 32 and 64 respectively.
All three cases use metadata width 64 and rank-16 inactive source adapters.

Short smoke test of the new path, not a formal timing result:

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_minikv_rank_13b.py --smoke --rank-loop actual --block-r 32 \
  --cases r8_u1 r16_u1 r32_u1 r64_u1 mix_7r8_1r64 r8_u8_padded64 \
  --warmup 1 --iterations 2 --repeats 1 --diagnostic-iterations 1 \
  --output-dir "try/results/rank_actual32_smoke_$(date +%Y%m%d_%H%M%S)"
```
