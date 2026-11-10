# LowRA Layer Reconstruction Profile

This branch uses S-LoRA FIFO admission and its ten-decode-step minibatch check.
LoRA grouping and waiting-SLO promotion are disabled. The prefill token cap and
running request limit remain the original S-LoRA controls.

## Modes

- `full-reprefill`: perform ordinary LoRA prefill of the actual prefix
  tokens on a cache miss. Cache hits reuse all-layer prefix KV in Unified Paging.
  There is no static GPU region, CPU prefix store, or MiniKV reconstruction.
- `minikv-full_reconstruct`: reconstruct every layer's full prefix on a cache
  miss before query prefill, then keep and reuse it through decode and later batches.
- `minikv-layer_serial`: reconstruct layer N, run its attention and FFN, then overwrite
  the same single-layer workspace for N+1.
- `layer_prefetch`: immediately after layer N's final attention kernel has consumed its prefix,
  a CUDA event allows a side stream to overwrite that same workspace for N+1
  while the compute stream runs N's output projection, residual and FFN.
  Decode's score kernel alone is not sufficient: the value-reduction kernel must
  also finish before overwriting either K or V. K and V use interleaved blocks in one
  fused reconstruction kernel launch. N+1 waits for both outputs only immediately
  before attention: its norm/QKV/RoPE/private-KV writes can run first.
  Layer 0 also reconstructs on the side stream while its norm/QKV work runs.
  This does not
  guarantee simultaneous kernel starts or reserve GPU resources for either
  stream. The `minikv-layer_serial` mode retains its separate K/V reconstruction launches.

In layer modes there is only one layer's prefix workspace, not a double buffer. Each forward
deduplicates LoRA IDs: requests with the same adapter share the same prefix
indices. Private query and generated-token KV remain persistent in every layer.
Layer modes reconstruct again during each decode forward because their previous
layer prefixes have been overwritten. `max_new_tokens=O` entails one prefill
output and O-1 decode forwards.

Parallel mode reuses a bounded pair of synchronization events per model layer.
Adapter-index metadata is cached for the last unique-adapter order and tensor
identities/versions. Adapter loading, offloading, in-place index/scaling changes,
or changes to the unique-adapter mapping invalidate it. Workspace locations are
never cached: every forward obtains its actual slots and returns them afterward.
The optimization does not retain reconstructed KV across forwards or change
FIFO admission, retention policies, or private-KV ownership.

One shared prefix length is configured per server run. It may change between
experiments; queries and current decode lengths may differ within a batch.
Actual final output lengths are not needed by the kernels. Existing request
generation/context limits and admission reservations remain in force. Changing
the shared prefix may compile a new specialization; changing each decode length
does not require a separate decode-attention specialization.

All MiniKV modes keep BaseKV/MiniKV static. With L layers, U unique adapters, prefix length P,
local KV width H, and element size e, prefix backing is `2*L*U*P*H*e` bytes for
the full-prefix cache modes and `2*U*P*H*e` for layer modes. The latter acquires
`ceil(U*P/L)` Unified Paging slots for the actual forward's unique adapters,
then returns those slots after the forward. Workspace addresses are views into
the existing pool, not extra GPU allocations. Rounding wastes at most L-1
single-layer token rows; registered but unused adapters reserve no workspace.
Admission accounts for the running/new batch's union of adapters so that the
merged decode batch also fits. This reduction
applies to prefix backing, not total GPU memory: model weights, static sources,
private KV, intermediates, and allocator reservations still consume memory.

## Standalone Profile

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/slora/bin/python \
  benchmarks/lowra_layer_profile.py \
  --model-dir huggyllama/llama-7b \
  --adapter-base dummy-lora-7b-rank-16 \
  --dummy --mode all \
  --prefix-len 1440 --query-len 480 --output-len 64 \
  --batch-size 8 --unique-loras 8 --rank 16 \
  --warmup 5 --iterations 20 --seed 42 \
  --trace-dir /tmp/lowra_layer_traces \
  --output /tmp/lowra_layer_profile.json
```

This uses real model dimensions and real S-LoRA layer/adapter kernels, but dummy
weights. Add `--nonzero-prefix` to check nonzero reconstruction. Add `--smoke`
for an explicitly smaller three-layer, 1024-hidden test model. `--num-adapters`
can exceed `--unique-loras`; all MiniKV sources are stored, while only the
requested adapters participate in this fixed batch. The adapter path's rank
must match `--rank`.

`--mode all` isolates modes in separate processes. Each measured round starts
with a cold full-prefix cache. CUDA intervals and wall time cover prefill,
sampling, metadata, and subsequent decode; initial adapter H2D is excluded.
The full mode's decode reconstruction cost is zero on cache hits; layer modes
pay reconstruction on every decode. Timing instrumentation is enabled, so use
these results for breakdowns, not uninstrumented serving throughput claims.

Results include phase/layer reconstruction, attention kernel, attention
sublayer, FFN and wait intervals, reconstruction ratios, prefix backing,
actual allocated/reserved GPU peaks, serial/prefetch output-token equality, and slot counts
after cleanup. Attention sublayer includes projection and residual work;
attention kernel measures attention alone. Event-interval overlap includes
stream waits and submission gaps and is not proof of simultaneous GPU kernel
execution. Inspect the exported CUDA kernel timeline for actual concurrency.
Concurrent reconstruction can contend with FFN, so prefetch need not be faster.
Full reports a separate `prefix_prefill` interval inside prefill time. Its real
prefix KV is not numerically equivalent to zero/synthetic BaseKV and MiniKV;
full correctness is instead checked against ordinary uncached full-prompt
prefill and decode in `test_full_reprefill.py`.

Memory accounting also records `base_model_weight_bytes`,
`base_model_buffer_bytes` (resident model buffers such as RoPE), and
`base_model_resident_bytes` (their sum). These count unique actual GPU tensor
storages rather than estimating from the parameter count. The serving-only
metric `peak_allocated_without_base_model_bytes` is allocated peak minus that
resident base model. It retains BaseKV/MiniKV, LoRA weights, dynamic backing,
prefix workspace, and transient compute tensors. A weights-only subtraction is
also available as `peak_allocated_without_model_weights_bytes`. These describe
PyTorch allocated memory, not NVML usage, CUDA context memory, or reserved cache.

The script provisions minimal backing for a fixed workload. It is not a
same-budget throughput experiment. Actual server modes keep a single total
token-equivalent budget. Full uses it entirely as dynamic Unified Paging;
layer modes deduct static sources at startup and charge one-layer workspace
on demand. `--lowra-workspace-loras` is an optional
active unique-adapter limit, not an allocation or a startup reservation.

## Serving Integration

Append `--lowra-prefix-mode minikv-layer_serial` or `--lowra-prefix-mode layer_prefetch`
to the existing LowRA launch command. Default is `full` for the control.
Do not pass a nonzero `--lowra-wait-slo-ms`, adapter `--prefetch`, nonzero
`--pool-size-lora`, or an alternate scheduler for this profile branch.
The workspace is sized to the actual forward's unique adapters. An explicit
`--lowra-workspace-loras` limits admission, without reserving unused workspace.
Full uses its own FIFO queue with inactive LRU eviction; its prefill token cap
includes real prefix tokens only for unique cache misses. Query/decode read
the shared prefix by indices and keep their private KV separate.

The implemented serving path targets FP16 MHA with S-LoRA's q/k/v/o packed
adapter layout. Other formats are rejected rather than silently misaccounted.

## Correctness Tests

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/slora/bin/python \
  -m unittest discover -s test -p test_lowra_layer_runtime.py -v
```

Tests cover single-buffer accounting, FIFO admission, cache-hit protection,
nonzero mixed-rank reconstruction, indexed shared-prefix prefill/decode,
cross-mode logits, filter/merge, refcounts, and private-slot release.

## Reconstruction Tile Microbenchmark

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  benchmarks/lowra_reconstruction_bench.py \
  --prefix-lengths 4608 8193 --unique-loras 1 20 --ranks 16 \
  --ffn-tokens 40 1920 --iterations 10 --embed-first \
  --output /tmp/lowra_reconstruction_tiles.json
```

This compares the same fused reconstruction kernel plus the same FFN in serial
and side-stream execution. It uses nonzero synthetic sources and indexed output
with gaps. It is not a serving throughput experiment. The default low-rank
kernel uses `(BLOCK_M, BLOCK_N, BLOCK_R)=(32,128,16)`, four warps, one stage,
and adjacent-column-first scheduling for rank capacity <=16, width >=128 and
prefix >=128. Other shapes retain the original `(16,32,32)` configuration.
This is a measured conservative choice, not a claim of global optimality or
of equivalent performance on other GPUs. No online autotuning enters serving.

On A100 80GB / Triton 2.1, an illustrative P=8193, U=20, rank=16 case reduced
isolated one-layer reconstruction from 7.26 ms (old tile/order) to 6.60 ms.
With a 40-token FFN, paired overlap completion fell from 7.67 to 6.95 ms.
However, the optimized fused serial pair took 6.85 ms: overlapping kernels was
still slower in that microbenchmark. Event reuse and later waits remove avoidable
host/dependency overhead, but cannot eliminate shared SM/HBM contention.

### Validation (2026-11-05)

The 86-test GPU regression run covered layer runtime, runtime decode lengths,
mode aliases, saturation accounting, full reprefill/reconstruction, retained/pinned
prefixes and service cycles. New tests include non-tile-aligned prefixes up to
8193 tokens, mixed ranks, ragged queries, logits after filter/reorder/merge and
adapter replacement, metadata invalidation and event reuse. Request-private
slots and unchanged full/serial/retained behavior are checked separately.

Raw microbenchmark JSON files and a short service smoke run are saved in
`results/layer_prefetch_optimization_20261105/`. The smoke run used GPU 1,
Llama-7B dummy dimensions, 50 adapters, P/Q/O=4608/480/256, a 36864-token
budget, seed 42, rate 8 for only 3 seconds, after two preflight requests.
All 27 measured requests and both preflight requests completed successfully.
There were zero new prefix-decode compilations during the measured replay;
runtime reset succeeded and the test server was stopped. This run is a
functional check, not a qualified saturated-throughput measurement.
