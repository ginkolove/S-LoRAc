# Reconstruction / FFN GPU Contention Diagnostic

**Correctness correction (2026-11-06):** historical serving timings below predate
the S-LoRA packed-B address fix. The old reconstruction wrappers read physical
slots as a rank-major matrix, while the existing S-LoRA loader/expand kernel use
HF B[hidden,rank] flattened across those slots. Current serving wrappers explicitly
select `b_packed=True`; standalone matrix diagnostics retain `False` by default.
Old numerical checks against rank-major synthetic matrices did not catch this
integration bug. Do not treat the historical serving speedups as validated
performance of the corrected operator. See [the audit](PACKED_RECONSTRUCTION_AUDIT.md).

**Real-model correction:** a subsequent complete Llama-7B audit also found the
missing delta-K RoPE operation. Serving/profile initialization now includes it
for raw MiniK plus post-RoPE BaseK, without another full-prefix buffer. Historical
timings without that work also require a rerun. See [the real-model audit](MODEL_E2E_AUDIT.md).

This is a standalone, single-layer diagnostic. It does not change the server,
scheduler, memory budget, or accuracy experiments. It imports the actual
`lowra_reconstruct_kv_batched` fused reconstruction kernel and the LLaMA serving
`LlamaTransformerLayerInfer._ffn` implementation.

## Controlled Comparison

All modes use the same nonzero FP16 inputs, adapter ranks, indexed destinations,
reconstruction kernel, and FFN. Both K and V are reconstructed once per unique
LoRA, not once per request. Destinations and source metadata are prepared before
timing. FFN retains its existing allocation and operator sequence.

Six modes are tested:

| Mode | Work |
| --- | --- |
| `reconstruct` | Reconstruction alone, on its side stream |
| `ffn` | FFN alone |
| `serial_rf` | Reconstruction then FFN, on one stream |
| `serial_fr` | FFN then reconstruction, on one stream |
| `overlap_rf` | Fork two streams, submit reconstruction first, join at end |
| `overlap_fr` | Same dependencies, submit FFN first |

`rf`/`fr` change host submission order, not GPU priority or a guarantee of simultaneous
execution. Both branches depend only on the initial event. A join protects the next
trial from pending writes. Trials are synchronized at their end; modes are shuffled
within each repetition using a fixed seed. Compilation, warmup, numerical checking,
and trace collection are outside the timed samples.

The original serving serial and parallel modes used different reconstruction
kernels (separate K/V vs fused K/V). This diagnostic deliberately removes that
confound: **the fused kernel is identical in every comparison**.

The parallel runtime now supports bounded-worker K/V for the measured large A100
regime; see the follow-up below. This older FFN-only diagnostic
still explicitly benchmarks the fused kernel, not the automatic selector.

## Run on Physical GPU 1

Run a targeted LLaMA-7B decode comparison first (32 requests, 20 unique LoRAs,
prefix 4608, rank 16). No model download, server, or full model weights are needed.

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="try/results/overlap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_reconstruct_ffn.py \
  --prefix-lengths 4608 \
  --unique-loras 20 \
  --ffn-tokens 32 \
  --ranks 16 \
  --hidden-size 4096 --intermediate-size 11008 \
  --launch-modes eager graph \
  --warmup 10 --iterations 100 \
  --trace \
  --output-dir "$RUN_DIR"
```

For a sweep, change `--unique-loras` to `1 8 20`, and `--ffn-tokens` to `32 64 128`.
Keep the other parameters unchanged. Omit `--trace` for a timing-only sweep.
Test prefill-sized FFN separately with `--ffn-tokens 15360` (32 queries x 480 tokens).
This changes the FFN work only, not the prefix reconstruction work.

`--row-stride 2` checks an indexed destination layout with holes; the default is a
contiguous destination. `--prefix-lengths` and `--ranks` accept multiple values.
`--ffn-tokens` is the number of tokens entering this one FFN: in decode it is the
running batch size, **not** the final output length (256/512). Shape sweeps keep the
compute/data flow fixed; they do not simulate a request trace.

## Outputs and Interpretation

`result.json` is updated after each shape/launch-mode group; existing results are
never overwritten. Each row includes raw samples, mean/median/p95, code hashes,
hardware/library metadata, numerical checks, and:

- `reconstruct_event_span_slowdown`: concurrent reconstruction event span / isolated span.
- `ffn_event_span_slowdown`: concurrent FFN event span / isolated span. A high value
  can mean queued work, not slower execution inside the kernels. Check the trace.
- `serial_over_parallel_speedup`: serial total / parallel total; above 1 is faster.
- `ideal_max_speedup`: `(R + F) / max(R, F)`, a no-contention upper bound based on
  isolated work. This is an estimate, not a strict bound when launch/event overhead
  matters (especially tiny shapes and external graph timing). If one side dominates,
  even ideal overlap offers little speedup.
- `net_saved_ms`: serial total minus parallel total; negative means a regression.
- `host_enqueue_ms`, `wall_ms`, and `total_ms`: host submission time, synchronized
  wall time, and GPU event-span time. They are different measurements.
- `event_interval_overlap_ms`: intersection of branch event intervals, **not proof
  that their kernels ran simultaneously**.

`summary.csv` contains one row per shape/launch-mode, with median total times for
all six modes and both parallel speedups. Correctness checks compare all FFN
outputs and 17 prefix positions per adapter (all columns, both K/V) against an FP32
reconstruction reference, with explicit output poisoning and guards on unused
destination rows. They are not an end-to-end model accuracy test.

The `graph` comparison captures the same fork/join and operators in CUDA Graphs.
This reduces Python/driver submission gaps; it is a diagnostic control, not an
assumption that dynamic serving can use a single static graph. Graph failure is
reported explicitly, not silently replaced by eager execution.

On Torch 2.0, captured events become graph dependencies and cannot reliably time
individual branches after replay. Graph `total_ms` uses external CUDA events;
event-based branch slowdowns are `null` for graphs, never fabricated from total
time. Use `kernel_trace_comparisons` for graph branch slowdowns measured from GPU
kernel traces. These are profiler-instrumented diagnostics, not headline timings.

With `--trace`, separate Chrome/Perfetto traces and parsed kernel timelines are
saved for all six modes. `traces.*.actual_kernel_overlap_ms` measures intersections
of actual reconstruction and FFN kernel intervals, excluding CPU ranges and copies.
Graph RNG-offset `FillFunctor<long>` bookkeeping kernels are reported separately,
not charged to FFN, even though no random operations are present in the captured work.
Trace timings are instrumented diagnostics; the headline medians come from the
unprofiled samples. Open the JSON traces in Perfetto to inspect launch gaps and
kernel overlap. A missing kernel trace is marked `valid: false`, not interpreted
as zero contention.

Interpret evidence cautiously:

1. Eager is poor but graph improves substantially: host submission and scheduling
   gaps contribute; do not blame GPU capacity alone.
2. Graph still shows longer kernels, with real overlap but little/no speedup:
   resource interference is supported for these shapes. Hardware counters are
   needed to distinguish SM/Tensor Core, occupancy, L2, or HBM limitations.
3. Almost no actual kernel overlap: inspect kernel residency, grid size, launch
   gaps, and stream dependencies. This does not by itself prove an event bug.
4. A tiny FFN relative to reconstruction limits the available gain even if neither
   kernel slows down. Compare against `ideal_max_speedup`.

This isolates FFN core vs reconstruction, not the entire serving overlap window
(which also contains O projection, residual/norm, and next-layer QKV), not a whole
decode step, and not a hardware theoretical limit. Single-layer reuse has different
cache behavior from cycling through all layers. Timings cannot prove that *every*
possible reconstruction implementation would suffer the same contention. Stop
other workloads on the selected GPU before interpreting results; snapshots record
GPU utilization but do not enforce exclusivity.

CPU statistics tests:

```bash
/home/qlh/anaconda3/envs/slora/bin/python -m unittest discover -s try -p 'test_*.py' -v
```

## Complete Decode Compute Window

`profile_compute_window.py` replays the actual O projection and its LoRA kernels,
residual, FFN norm, FFN, residual, next-layer norm, QKV LoRA computation, and RoPE.
It uses nonzero synthetic tensors and the next layer's actual packed B buffer for
both reconstruction and QKV. Main work stays on CUDA stream 0 because the current
BGMV extension launches there. It does not include attention, sampling, or admission.

In addition to separate/fused serial and parallel runs, `window` trials reconstruct
a prefix-token chunk across all unique adapters alongside one selected stage, then
join before the following stage. The remaining prefix is reconstructed before the
next attention. The chunks use existing kernels and actual physical destination
indices; no extra full prefix or intermediate KV buffer is introduced. Validation
checks all reconstructed elements and main outputs against the split reference.

```bash
cd /home/qlh/S-LoRAc
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_compute_window.py \
  --prefix-len 4608 --query-len 480 --batch-size 32 --unique-loras 20 \
  --rank 16 --hidden-size 4096 --intermediate-size 11008 \
  --chunk-tokens 16 64 256 --warmup 10 --iterations 100 --trace \
  --output-dir "try/results/compute_window_$(date +%Y%m%d_%H%M%S)"
```

`--windows` with no values omits the chunk sweep. Timings come from unprofiled,
randomly interleaved repetitions. Instrumented event spans include launch/wait
overhead and must not be used as individual kernel durations. `kernel_trace`
attributes kernels through launch correlations where available; missing driver
launch records are explicitly `unattributed`. The Chrome traces also include CPU
stage ranges. `tune_compute_window.py` compares fused tile/order choices using the
same window, rather than assuming fewer launches are faster.

### Observations on GPU 1, 2026-11-05

Configuration: A100 80GB, Torch 2.0.1/cu117, Triton 2.1.0, FP16, P=4608,
H=4096, intermediate=11008, rank=16, B=32, U=20. Reconstruct K/V once per
unique LoRA, not once per request. Unprofiled window medians (40 trials):

| Implementation | Window time (ms) |
| --- | ---: |
| Serial split K/V | 3.6920 |
| Original fused parallel | 4.2726 |
| Split K/V parallel | 3.7279 |
| Chunk-window candidates | all slower than serial split |

Norm kernels were about 5 us each, residual kernels about 4 us. A 16-token
reconstruction chunk for all 20 adapters already needed about 21 us of K+V kernel
time, plus launches/events. The candidate light windows were too short for the
tested chunk mechanism to pay for itself. In the U=8 diagnostic trace, split
reconstruction overlapped O kernels by about 47 us (fused about 6 us); FFN had no
observed kernel overlap in that trial. This is not evidence of SM saturation or
proof that no other overlap implementation could work.

Nsight Compute hardware counters were unavailable (`ERR_NVGPUCTRPERM`). Do not
report SM/Tensor Core/HBM utilization percentages based on these tests. Profiler
estimated occupancy fields are not measured utilization.

Recorded results:

- `results/compute_window_20261105/result.json`: complete window/chunk sweep.
- `results/compute_window_tuning_20261105/tuning.json`: tile and launch-order sweep.
- `results/compute_window_u1_20261105/result.json`: one-LoRA check, fused preferable.
- `results/compute_window_u8_20261105/result.json`: eight-LoRA check and annotated traces.
- `results/compute_window_p8193_u8_20261105/result.json`: non-aligned long prefix;
  parallel split 2.8068 ms vs parallel fused 3.1232 ms.

### Serving Change and Full-Layer Validation

Only `layer_prefetch` kernel selection changed. Its single-workspace lifetime,
attention-completion fence, earliest next-layer launch, wait at the next attention,
allocator, and admission policy are unchanged. The serial and full-cache paths
are unchanged. There is no per-request profiling or CUDA synchronization added.

The first selector used split K/V for compute
capability 8.0, K/V width 4096, max rank <=16, prefix >=4096, and
`unique_adapters * prefix_len >=32768`; otherwise it retains the previous fused
path. The follow-up below replaces that selected large-shape path with bounded
workers. This is a conservative measured-shape heuristic, not a universal optimum.
Constructor overrides support controlled experiments. The
normal server needs no new argument. The unprofitable chunk-window method is
kept in the diagnostic, not enabled in production.

`profile_serving_overlap.py` loads the real 32-layer Llama-7B architecture with
dummy weights and nonzero Base/MiniKV. It reuses the same model, adapters and
request batch for serial, old fused prefetch, and new automatic prefetch, checks
all steps' logits, warms compilation, then interleaves unprofiled repetitions.
It explicitly extends the performance-only context without changing RoPE. It
does not load real model weights or measure task accuracy.

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_serving_overlap.py \
  --prefix-len 4608 --query-len 480 --batch-size 32 --unique-loras 20 \
  --output-len 16 --warmup 1 --iterations 3 \
  --output-dir "try/results/serving_overlap_$(date +%Y%m%d_%H%M%S)"
```

`results/serving_overlap_20261105/result.json` passed logits checks. Median aggregate
decode time / 15 decode steps:

| Implementation | ms/step | Peak allocated (GiB, including model) |
| --- | ---: | ---: |
| Serial split | 188.55 | 26.21 |
| Original fused parallel | 206.58 | 26.21 |
| Automatic parallel (split here) | 193.54 | 26.21 |

Decode time decreased 6.31% vs old parallel, but remained 2.65% slower than serial.
This is a fixed-batch implementation regression, not the previous 936-request,
512-output-token saturation workload. It does not establish improved service
throughput or a benefit from phase-selective overlap.

## Bounded-Worker Follow-up

The final parallel implementation is `slora/common/lowra_reconstruct_bounded.py`.
A finite grid of worker CTAs loops over LoRA/token/output-column tiles, instead
of submitting hundreds of thousands of CTAs at once. The serving tile is
32 prefix tokens x 128 output columns, with rank tiles of 16. K/V write directly
into the existing indexed single-layer workspace. There is no host-side loop
over every token chunk, no extra full-prefix buffer, and no additional stage-end
barrier. The existing attention-completion fence and next-attention join remain.

The CTA cap is not a hard SM partition and does not guarantee resource isolation.
Too few workers substantially slow reconstruction, so the limit was swept instead
of assuming that leaving more SMs idle is always beneficial. Grouping eight adapter
tiles keeps divisors constant while `COUNT` remains a non-specialized runtime
argument; changing the selected LoRA count does not itself require compilation.
Query lengths and decode progress are not compilation constants in this kernel.

For the same auto-selector conditions above, `layer_prefetch` now chooses
`bounded`, with 12 times the device SM count as its worker limit (1296 on this
A100). Other regimes retain the previous fused path. Constructor overrides are
`prefetch_reconstruction="split"|"fused"|"bounded"|"auto"` and
`prefetch_workers=<positive integer>`. Normal server commands using
`--lowra-prefix-mode layer_prefetch` need no changes. Serial/full methods, batch
admission, prefix sharing, private query/decode KV, and total KV budget are unchanged.

### Reproduce the Kernel Sweep

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_bounded_reconstruct.py \
  --prefix-len 4608 --query-len 480 --batch-size 32 --unique-loras 20 \
  --worker-counts 648 1296 --bounded-tile 32 128 16 4 \
  --start-stages o ffn_norm ffn qkv_norm \
  --warmup 5 --iterations 50 --trace \
  --output-dir "try/results/bounded_window_$(date +%Y%m%d_%H%M%S)"
```

`--start-stages` tests reconstruction submission before the named stage. All
asynchronous trials join only at the end of the complete window, representing
the next attention dependency, not immediately after the selected light stage.
`--reconstruct-graphs` optionally captures only reconstruction, not BGMV/main
compute. Use it without `--trace`: Torch 2.0 CUPTI Graph collection hung during
post-processing in one diagnostic, which was terminated. The CLI rejects that
combination. CUDA Graphs are not enabled in the serving runtime by this change.
The sweep saves unprofiled timings before collecting optional traces.

### Final Full-Model Regression

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_serving_overlap.py \
  --modes serial_split serial_bounded prefetch_bounded prefetch_auto \
  --prefix-len 4608 --query-len 480 --output-len 16 \
  --batch-size 32 --unique-loras 20 \
  --bounded-workers 1296 --serial-bounded-workers 648 \
  --warmup 1 --iterations 3 \
  --output-dir "try/results/serving_bounded_$(date +%Y%m%d_%H%M%S)"
```

The serial bounded-kernel comparator exists only in this harness, implemented by
direct function replacement. It does not change the production serial mode.
Source hashes for the runtime, bounded kernel, and harness are recorded. No mock
call recording is used in the timed path. All model logits are checked before
timing, using FP16 numerical tolerances.

Final result: `results/serving_bounded_dynamic_final_20261105/result.json`.
Same 32-layer Llama-7B dummy model, nonzero Base/MiniKV, same batch/request input,
FP16, GPU 1; medians of 3 rounds, each containing 15 decode steps:

| Implementation | Workers | Decode ms/step | Peak allocated GiB, including model |
| --- | ---: | ---: | ---: |
| Original serial split | original grid | 188.48 | 26.21 |
| New kernel, serial comparator | 648 | 131.12 | 26.21 |
| New kernel, parallel | 1296 | 130.16 | 26.21 |
| Server auto parallel | 1296 | 130.09 | 26.21 |

Parallel is about 31% faster than the original serial implementation here. Most
of this is kernel efficiency, not overlap. The tuned new serial comparator is
only about 0.7% slower than tuned parallel; different worker settings are shown
explicitly. Three short rounds do not establish a robust pure-overlap advantage,
and this is not the 936-request/decode-512 saturation workload.

`results/bounded_dynamic_count_20261105/result.json` contains the final kernel's
window timings and eager traces. Other bounded-worker directories are exploratory
versions. In particular, `serving_bounded_overlap_20261105` and
`serving_bounded_best_pair_20261105` used a Mock in the serial comparator, retaining
call arguments and causing GC interference. They are superseded, not headline
results. `serving_bounded_fair_20261105` removed that overhead but predates the
runtime COUNT version; use the final result above for the current implementation.

Validation includes scattered destinations, masked dimensions, ranks 8/16/40,
prefixes through 8193 tokens, variable-count compilation-cache reuse, protection
of unselected rows, ragged queries, filter/merge, adapter replacement, workspace
release and logits. Run on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -m unittest \
  discover -s test -p test_lowra_layer_runtime.py -v
```

### Independent Correctness Re-Audit

`audit_bounded_reconstruct.py` checks every output element against both the old
kernel and a separately evaluated FP32 PyTorch formula with TF32 disabled. It
uses nonzero random Base/MiniKV/B, independently shuffled source and destination
indices, varied scalings, NaNs in required destinations, and sentinels in all
unselected rows. It is not a speed test or a pretrained-model accuracy test.

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/audit_bounded_reconstruct.py \
  --output "try/results/bounded_audit_$(date +%Y%m%d_%H%M%S)/result.json"
```

Result: `results/bounded_correctness_audit_20261106/result.json`.

- All four GPU cases matched the old kernel bitwise, for K and V separately.
- The serving-size case checks 377,487,360 elements per K or V tensor:
  20 adapters, prefix 4608, width 4096, rank 16, with both 648 and 1296 workers.
- Other cases cover prefix 4609, width 4097, mixed ranks 1/3/8/16/40, selected
  adapter counts 17/33, and unselected source adapters. No required output stays
  NaN and no sentinel row changes.
- Maximum absolute difference from the independent FP32 formula is
  0.0002442598342895508, including final FP16 rounding.
- 32 CPU schedule checks verify every valid tile is visited exactly once.
- The 27-test GPU suite also passed after extending dynamic COUNT coverage
  across 7/8/9 and 16/17/20/33 adapters and forcing the bounded path through the
  direct per-layer KV/event/workspace test. The production source hashes still
  match the final full-model timing report above.

The old tile is `(16, 32, 32)` and the serving bounded tile is `(32, 128, 16)`.
For one K or V tensor in the serving-size case, the old grid launches 737,280
small CTAs. The new kernel assigns 92,160 valid larger tiles to a bounded worker
grid (plus masked adapter-group padding), iterating until all tiles are written.
This changes execution granularity and rank padding, not the reconstruction
formula or number of required output values. It explains why the new serial
comparator benefits too; the roughly 31% end-to-end step-time reduction cannot
be attributed solely to concurrency.
