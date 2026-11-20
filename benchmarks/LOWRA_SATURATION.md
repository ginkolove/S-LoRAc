# Full Cache versus Layer-Serial Saturation

For the five-source, mixed-rank LMSYS replay, see the
[2026-11-20 experiment handoff](PREFIX_SOURCE_HANDOFF_2026-11-20.md) for
the 55-run matrix, verified results, restart commands and validity checks.

**2026-11-06 correction:** all MiniKV serving reconstruction wrappers now decode
the actual S-LoRA packed B layout (`b_packed=True`). Earlier MiniKV timings used
incorrect rank-major reads from that pool. The performance source tensors are
zero-initialized, which hid the numerical error. Preserve those logs as historical
results, but rerun before drawing conclusions about the corrected operator.
Ordinary `full-reprefill` and the original LoRA weight loader are unchanged.
See `try/PACKED_RECONSTRUCTION_AUDIT.md` for the reproduction and validation scope.

The subsequent real-model audit also added delta-K RoPE before BaseK addition.
Normal serving and fixed-batch profile initialization bind these position tables,
including dummy performance runs. The destination workspace is reused in place;
no additional full-prefix tensor is allocated. Old timings omitted this work and
must be rerun. See `try/MODEL_E2E_AUDIT.md` for the verified real-model scope.

`lowra_saturation_exp.py` manages a single real S-LoRA HTTP server and replays
open-loop workloads. It follows the asynchronous replay structure of `run_exp.py`,
not its string prompts, power-distribution adapter sampling, or strip statistics.
The canonical modes are `full-reprefill`, `minikv-full_reconstruct`, and
`minikv-layer_serial`. Legacy names `full`, `full_reprefill`, `full_reconstruct`,
and `layer_serial` are accepted and normalized in new configurations and results.
Full reprefill is ordinary S-LoRA with a shared full-prefix runtime cache:
on a miss it runs real LoRA prefix prefill; on a hit it reuses the indexed KV.
It does not initialize BaseKV/MiniKV or fetch prefix KV from CPU.
Do not start another service for the same invocation.

`minikv-full_reconstruct` uses the same FIFO/LRU admission and full-prefix ownership as
`full-reprefill`, but builds each unique cache miss from static BaseKV/MiniKV and
the loaded adapter B weights. All layers are reconstructed before query prefill.
Decode and later cache hits reuse the full prefix without reconstruction. The
outputs are written directly into ordinary Unified Paging slots, with no layer
workspace or separate KV pool. Static sources use the same zero initialization
and token-equivalent accounting as `minikv-layer_serial` and reduce the dynamic budget.
Only query tokens are sent for this mode, with explicit logical length metadata.
Thus identical policies do not imply identical admitted batches: the available
dynamic budgets differ from `full-reprefill`.

## LMSYS Adapter Profiles

The five prefix-source comparisons (`run_lmsys_preemption.sh --prefix-sources`
or `--prefix-source-group compute|cpu`) default to `--adapter-profile chameleon`.
Adapter IDs cycle through fixed ranks 16, 32, 64, giving 17/17/16 adapters for
N=50. Each rank gets one third of the probability mass; within a group the
j-th adapter has weight proportional to 1/j. These are sampling probabilities,
not quotas. `trace.json` freezes each request's adapter index; `chameleon`
contains the rank and probability vectors indexed by adapter ID, and `summary`
reports observed request counts by rank. The CPU fixed Top-2 uses the two
highest prior probabilities (tie: lower adapter index), not replay counts.
All five methods use the same trace and validate the server adapter ranks.

Use a fresh output directory. Pass `--adapter-profile maf` to reproduce the
rank-16 five-way setup; the standalone generator and ordinary two-mode launcher
still default to MAF. Changing only the profile preserves arrivals, query IDs,
input tokens and output lengths for the same seeds, dataset and tokenizer.

### MAF Profile

`prepare_lmsys_maf_trace.py` prepares a rank-independent JSON trace on CPU. It
uses the first user/assistant pair from each LMSYS-33K A/B branch; subsequent
turns are not replayed as independent requests. Queries are tokenized with the
serving tokenizer without inserting a BOS after the shared prefix. The recorded
answer token count specifies each request's forced output length (`ignore_eos`).
Samples are selected without replacement, rejecting empty or over-context pairs
without truncation. This is a content/length-driven serving benchmark, not an
accuracy test or a dependency-preserving replay of complete conversations.

The default popularity source is MAF 2019 day 1, all minutes 1-1440. Functions are
sorted by descending invocation count (ties by owner/app/function key); sorted
position j maps to adapter `(j // skewness) % num_adapters`. Skewness 1 is this
block size, NOT Zipf alpha 1. Counts are aggregated into adapter probabilities.
MAF timestamps are not replayed: independent exponential interarrival times
generate the requested global Poisson rate. Arrival and adapter sampling use
separate child RNG streams of `--seed` (default 42). Content order independently
uses `numpy.default_rng(--data-seed).permutation(...)`, with `--data-seed 42` by
default. Changing arrival rate, duration, rank, N or arrival seed does not change
the candidate permutation. Context filtering can still change which candidates
are accepted when the tokenizer or prefix/context budget changes.
The trace records source hashes,
mapping parameters, adapter probabilities, query tokens, answer lengths and
sample identifiers. Short Poisson traces need not contain exactly rate*duration
requests or use every configured adapter.

The generator also saves the complete shuffled candidate `source_ids` and the
per-adapter MAF load/probability profile under the output's `prepared/` directory.
These artifacts have SHA256 references in the trace. Previous first-hour traces
remain historical artifacts; they are not rewritten and are not the new default.

Pass `--workload-trace trace.json` to `lowra_saturation_exp.py`. The trace must
match the model/tokenizer, N, prefix/context limits, rate, duration and seed.
Fixed `--query-len` / `--output-len` become trace maxima for server configuration;
each HTTP payload and completion check uses its own lengths. `--alpha` is unused
on this path. Fixed-output saturation-cycle estimation is explicitly rejected.
The preflight uses real pairs and then resets the dynamic state before replay.
At the end of a valid replay, the client first writes
`checkpoints/rate<R>_seed<S>.json` with `replay_valid=true` and
`result.valid=false`. It uses a fresh control connection and the same reset
operation ID across retries for up to 120 s. The HTTP server keeps that
operation's original snapshot for 300 s in the same server process.
Only a confirmed reset with all rank pools free publishes the normal result
and removes the checkpoint. An unconfirmed checkpoint is diagnostic data, not
a successful benchmark or a replacement for missing cache counters.
All runs save their bound request trace; only the adapter directory changes
between rank=8 and rank=16. The original synthetic trace path is unchanged.

For the requested 10K-prefix, 50-adapter, 3 req/s, 10-second comparison:

```bash
RUN_DIR="benchmarks/results/lmsys_maf_rank_compare_$(date +%Y%m%d_%H%M%S)"
/home/qlh/anaconda3/envs/transformer/bin/python -u benchmarks/prepare_lmsys_maf_trace.py \
  --maf-skewness 1 --prefix-len 10000 --num-adapters 50 \
  --maf-minute-start 1 --maf-minute-end 1440 --data-seed 42 \
  --request-rate 3 --duration 10 --seed 42 --output "$RUN_DIR/trace.json"
nohup bash benchmarks/run_lmsys_rank_compare.sh 0 8 "$RUN_DIR/trace.json" "$RUN_DIR/rank8" \
  > "$RUN_DIR/rank8.log" 2>&1 < /dev/null &
nohup bash benchmarks/run_lmsys_rank_compare.sh 1 16 "$RUN_DIR/trace.json" "$RUN_DIR/rank16" \
  > "$RUN_DIR/rank16.log" 2>&1 < /dev/null &
```

The launcher runs full reconstruction then layer-serial sequentially on each
GPU, using the previous 125000-token total budget and 36864 prefill-token cap.
An optional fifth positional argument selects `7b` (default) or `13b`;
13B defaults to a 60000-token total budget. An optional sixth argument overrides
that budget, and an optional seventh argument sets the sending duration in seconds
(default 10). The duration must match the frozen trace's `--duration`.
The trace must have been prepared using the corresponding model's
tokenizer. Use the same MAF profile and data seed for paired workload selection,
and verify tokenized content and output lengths before comparing models.
It retains dummy weights and synthetic shared prefix state; native LMSYS does
not supply this 10K prefix. No serving kernels or scheduling policies change.
Warmup=0 and an end clamped to the final completion measure global throughput,
not peak or steady-state capacity. Latency/TTFT cover all successful requests;
failures are counted and invalidate the point. Rank-dependent static MiniKV and
adapter space remain charged against the same total budget, so this is a system
rank comparison rather than a reconstruction-kernel-only microbenchmark.

## Single-Buffer Parallel Reconstruction

`--mode layer_prefetch` uses the same FIFO admission, static sources and one-layer
workspace budget as `minikv-layer_serial`. No adapter retains any expanded prefix
between forwards. Once layer N's final attention kernel finishes reading the
workspace, a CUDA event allows a side stream to overwrite it with layer N+1's KV,
overlapping with N's output projection/residual/FFN. N+1's norm/QKV/RoPE/private-KV
writes also run before waiting; the wait is immediately before attention reads
the workspace. Layer 0 reconstructs on the side stream as well. The final layer
does not prefetch beyond the model.
Prefill and every decode forward use this ordering; private request KV stays intact.
Default HTTP/NCCL/internal ports are 8004/28769/16000.

Shared prefix length is fixed within each server run, but may vary between
experiments. Query and current generated lengths can differ per request; no
final decode length is required by the overlap mechanism. Event pairs and stable
adapter metadata are reused, but workspace allocation remains per-forward and
is returned to the unified pool. Adapter replacement invalidates cached metadata.
See `LOWRA_LAYER_PROFILE.md` for the isolated kernel/FFN comparison and validation.

Without an explicit kernel override, serial uses the original separate K/V
kernel and prefetch selects bounded workers for the profiled A100/4096-width,
long-prefix, rank<=16 regime (fused K/V otherwise). They compute the same values,
but those defaults do not isolate the effect of changing streams.
Concurrent kernels may contend for resources; overlap does not guarantee speedup.
Use `--measure-saturation` alongside the fixed completion window, with identical
trace settings, rather than treating a short completion burst as peak capacity.

### Matched-Kernel Decode-512 Comparison

Use `--reconstruction-kernel bounded --reconstruction-workers 1296` for both
`minikv-layer_serial` and `layer_prefetch`. The server options have a `lowra-`
prefix. This explicitly selects the same `(32,128,16)` tiled, separate K/V
reconstruction in both modes, including small batches; no shape-based fallback
is used. Both modes cache adapter index metadata with the same invalidation rules.
Only the stream ordering differs. Legacy defaults remain unchanged. Full-cache
modes reject these overrides; the retained layer mode also supports bounded
reconstruction for both initial retained builds and ordinary scratch builds.

The paired launcher starts and stops each server sequentially on the visible
GPU. Its `retention` comparison remains the default for existing commands;
`--comparison overlap` selects the new pair:

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="benchmarks/results/decode512_bounded_pair_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com \
nohup /home/qlh/anaconda3/envs/slora/bin/python -u \
  benchmarks/run_decode_saturation.py \
  --comparison overlap --output-len 512 --reconstruction-workers 1296 \
  --port 8290 --nccl-port 30290 --internal-port-start 18870 \
  --output-dir "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1 < /dev/null &
```

Configuration: Llama-7B dummy weights, 50 rank-16 adapters, prefix/query/output
4608/480/512, total KV budget 36864 tokens, batch-max-tokens 36864, running limit
256, Zipf alpha=1, Gamma CV=1, 8 req/s sent for 120 seconds, seed/model-seed=42.
Both modes replay the same saved 936-request trace. Static Base/MiniKV retain
the performance harness's zero initialization; this is not an accuracy run.
The fixed completion window remains [100,min(180,last completion)]. Qualified
capacity uses complete backlogged service cycles, not this short fixed window.

Outputs: `serial_bounded/`, `parallel_bounded/`, `traces/`, and `comparison.json`.
The comparison checks the trace hash and observed kernel/worker settings.
Each manifest records reconstruction source hashes; runtime JSONL and point
summaries report the selected reconstruction configuration. Compare global
completion throughput, total completion time, all-request latency/TTFT, and
capacity only when `saturation.qualified` is true. Do not treat a null peak as zero.
Add `--dry-run` to the launcher to inspect commands without launching experiments.

### Parallel versus Dynamic Top-1 with the New Kernel

`--comparison prefetch-top1` runs `parallel_bounded` then
`dynamic_top1_bounded` sequentially on one visible GPU. Both use the same bounded
K/V kernel and worker limit. Parallel mode retains no full prefix and overlaps
the single-layer overwrite with O/FFN. Dynamic Top-1 mode retains all 32 layers
of one hot adapter and serially reconstructs cold adapters into one layer of
scratch; it does not additionally enable side-stream reconstruction.

Hotness remains the current live request count across running and pending
mini-batches, not fixed Zipf rank or historical popularity. Selection updates
after request membership changes. Ties favor a retained adapter and then adapter
ID. A ready retained prefix survives its final request release as an inactive
LRU entry; inactive adapters do not participate in the active Top-K ranking. Both modes
cache source index metadata with identity/version checks; neither caches scratch
destinations. Retained promotions, cold builds, and suffix-layer builds all use
the selected new kernel, while a ready retained layer skips reconstruction.

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="benchmarks/results/decode512_prefetch_top1_bounded_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nohup /home/qlh/anaconda3/envs/slora/bin/python -u \
  benchmarks/run_decode_saturation.py \
  --comparison prefetch-top1 --output-len 512 --reconstruction-workers 1296 \
  --port 8294 --nccl-port 30294 --internal-port-start 18910 \
  --output-dir "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1 < /dev/null &
```

All trace/budget settings are the same as the decode-512 comparison above,
including 4608/480/512 tokens, 50 rank-16 adapters, 36864-token total budget,
8 req/s for 120 seconds, and seed 42. Each method manages its own server and
replays the same saved 936-request trace. The new comparison records retention
counters in `comparison.json` as well as validating the actual kernel and worker
configuration. Obtain an otherwise idle GPU before timing; zero utilization in
one snapshot does not establish exclusive access.

## Defaults

The optional `minikv-layer_retained` mode accepts `--retain-layers K` (server:
`--lowra-retain-layers K`). It retains the first K layers, zero-indexed `[0,K)`,
for each active unique adapter and reuses them across decode and mini-batch
merges. The remaining layers reconstruct into the same single-layer workspace
on every forward. K may range from zero to the model layer count. For K>0, the
final request release leaves ready retained layers in an inactive LRU cache.
Later batches reuse them without rebuilding. Admission evicts the oldest idle,
unpinned entries only when needed for private KV growth, adapters and workspace.
Active entries cannot be evicted. K=0 keeps no persistent expanded prefix; its
single-layer workspace is released after each forward. New adapters are still
allocated on demand, never all at startup.

Both retained KV and scratch borrow the original Unified Paging backing. For U
active adapters, prefix length P and L model layers, retained KV costs
`U * ceil(K*P/L)` full-layer-equivalent slots, plus `ceil(U*P/L)` scratch slots
when K < L. Idle retained entries add their actual allocated slots until evicted.
Private KV growth and adapter slots remain in the admission budget.
The HTTP/NCCL/internal default ports for this mode are 8003/28768/15000.
The K=all-layers and full-reconstruction modes now both have inactive LRU
retention, but their reconstruction kernel, initial build ordering and admission
paths still differ. A single-runtime K sweep should use this mode at both K=0
and K=all-layers, not substitute the separate serial/full modes at its endpoints.

All modes use a unique-LoRA prefix address table and a request-to-LoRA map in
prefill and decode attention. Stable request mappings and retained/full tables
are reused; table sources are checked by allocation identity and tensor version.
Serving no longer expands `[K, requests, prefix_tokens]` indices each forward.
Scratch addresses still come from the current allocation and are never assumed
valid after workspace release. Diagnostic `prefix_locations()` materializes a
request-level table only when explicitly called outside the serving path.

For the paired 4608-prefix experiment:

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  benchmarks/lowra_saturation_exp.py \
  --mode minikv-layer_retained --retain-layers 16 \
  --model-dir huggyllama/llama-7b --adapter-base dummy-lora-7b-rank-16 \
  --num-adapters 50 --prefix-len 4608 --query-len 480 --output-len 64 \
  --max-context-len 10000 --num-token 36864 --batch-max-tokens 36864 \
  --running-max-req-size 256 --request-rates 8 --duration 120 \
  --warmup 70 --steady-end 120 --alpha 1 --cv 1 --seeds 42 43 44 \
  --model-seed 42 --dummy \
  --trace-dir benchmarks/results/three_modes_p4608_w70_e120_20261012_081604/traces \
  --output-dir "benchmarks/results/retained16_p4608_$(date +%Y%m%d_%H%M%S)"
```

The runner starts/stops its own server. Runtime snapshots include per-adapter
layer-build/reuse counters and correctly count retained slots separately from
private KV. Prefill and decode use each retained layer's own address table.

### Retain Only Hot Adapters

Add `--retain-top-k H` (server: `--lowra-retain-top-k H`) to limit retention
to H adapters with the largest **current active request counts**, across the
running batch and newly admitted mini-batch. This is not a fixed Zipf ranking
or a historical popularity estimate. Ties prefer an existing retained cache,
then lexical adapter ID. Membership changes mark selection dirty; it is updated
before the next forward, not during a layer's attention computation.

`--retain-top-k 2 --retain-layers 32` on Llama-7B retains every layer of the
two selected adapters. Their first forward builds and keeps each layer as it
executes, without an additional full-prefix reconstruction pass. Other adapters
overwrite one layer of scratch per forward. For partial retention, e.g. 16,
selected adapters also use scratch for layers 16 onward. Omit `--retain-top-k`
to preserve the original all-active-adapter retention policy; zero disables
retention. Counts above the active adapter count are clamped.

With U active adapters, h=min(H,U), K retained layers, P prefix tokens and L
model layers, prefix storage costs `h * ceil(K*P/L)` slots plus
`ceil(C*P/L)` scratch slots, where C=U for K<L and C=U-h for K=L. The scheduler
reserves these slots along with private KV growth and adapter weights. It does
not prioritize hot requests over FIFO or exceed the configured total budget.

When an active adapter leaves the hot set, its retained cells are released and
its next forward reconstructs through scratch. An adapter becoming inactive
instead keeps its ready retained layers as an LRU entry. When a new adapter
enters the hot set, it reuses a ready idle entry or builds missing layers before
attention. Unchanged hot adapters keep valid KV across mini-batch merges. Idle
entries are charged in addition to the active/scratch formula above and are
reclaimable under pressure; fixed pinned entries remain protected.

For the full 4608 experiment on GPU1, use the preceding command with
`--retain-top-k 2 --retain-layers 32` and a fresh output directory.
Runtime snapshots report selected `retained_lora_dirs`, promotions/demotions,
layer builds/reuses, and actual retained slot occupancy. The trace identity
does not include these retention settings, so modes reuse identical workloads.

### Pin a Fixed Hot Prefix Until Reset

`--pin-hot-adapters N` (server: `--lowra-pin-hot-adapters N`) is a separate,
opt-in policy for `minikv-full_reconstruct` and `minikv-layer_retained`. It selects
the first N configured adapter IDs, which are the highest-probability adapters
in this synthetic Zipf workload. This uses known workload popularity, not online
heat prediction or the first arbitrary adapter to appear in a batch. For the
single adapter-base setup, N=1 selects `dummy-lora-7b-rank-16-0` in both modes.

There is no startup allocation or preconstruction. A protected prefix is built
on its first request, remains allocated across idle request gaps, and cannot be
evicted or demoted until `/lowra/reset_runtime`. Each seed starts with an empty
runtime after preflight/reset, including the protected prefix. Weights can still
be offloaded when unused: only the reconstructed prefix is pinned.

- Full reconstruction: add `--pin-hot-adapters 1`; cold prefixes keep ordinary
  complete reconstruction, cross-batch reuse and inactive LRU replacement.
- Layer reconstruction: add `--pin-hot-adapters 1 --retain-layers 32`; the fixed
  hot adapter accumulates all 32 layers during its first forward. Every other
  adapter continues single-layer reconstruction/overwrite on every forward.

Do not combine fixed pinning with dynamic `--retain-top-k`. Pinned KV occupies
the same pool and reduces memory available to requests/cold prefixes even when
its request count is zero. Both FIFO admission paths account for that idle
occupancy and exclude pinned prefixes from eviction. A configuration that leaves
insufficient room for any other request must use fewer pinned prefixes or a
larger total budget; pinning never permits exceeding the budget.

Per-adapter `prefix_build_counts` (full mode) and `retained_build_counts` (layer
mode) verify the protected adapter is built once per seed. Compare the pair on
one GPU sequentially using the same trace and seed, e.g. `--duration 120 --warmup 70
--steady-end 280 --seeds 42`, rather than selecting each method's best short bin.

- Llama-7B, 50 rank-16 dummy adapters when `--dummy` is provided.
- Prefix 1440, query 480, output 64; model context 2048.
- Total budget 36864 entries. Full puts all entries in the original dynamic
  pool; serial deducts its BaseKV/MiniKV static sources and charges workspace
  from its remaining dynamic pool. Both have the same total budget.
- FIFO scheduling, no SLO, no adapter prefetch, no separate adapter pool.
- Batch prefill limit 36864; running-request limit 256.
- Request rates 2, 4, 8, 16; seeds 42, 43, 44.
- Zipf popularity exponent 1.0. Aggregate Gamma renewal arrivals with CV=1
  are Poisson; independent adapter marks then give adapter-level Poisson arrivals.
  CV other than 1 describes the aggregate renewal process, not independent
  per-adapter Gamma processes.
- Send for 120 s, warmup 30 s, measure completions in [30, 90], then drain.
- Identical deterministic dummy-weight initialization with model seed 42.

Full uses HTTP/NCCL/internal ports 8000/28765/12000 by default; minikv-layer_serial uses
8001/28766/13000, and minikv-full_reconstruct uses 8002/28767/14000.
Use `--port`, `--nccl-port`, and `--internal-port-start` if occupied.
The runner refuses occupied ports and never kills an unrelated server.

## Semantics and Measurements

### Complete-Cycle Capacity Measurement

`--measure-saturation` adds `completed_req_per_s_saturated` and
`completed_req_per_s_peak` without changing the existing completion-window,
global, latency or TTFT statistics. The router writes exact lifecycle records to
`runtime.cycles.jsonl`. One cycle is a complete busy period, including all merged
mini-batches, from initial admission planning through final KV/adapter release.
Actual completed request IDs and output-token counts are checked against HTTP
results. Request arrival times do not filter the numerator.

The estimator skips the first two cycles and cycles starting before warmup,
excludes any cycle whose waiting queue was empty at an observed scheduling
boundary, and chooses the longest contiguous eligible run by duration, not the
run with the highest throughput. It divides all successful HTTP completions in
that interval by elapsed time, including gaps between cycles and final response
delivery. Startup compilation and the final underfilled drain are excluded.

A peak/capacity claim requires at least six complete cycles, at least 300 seconds,
matching client/server completion boundaries, no failed requests, no new prefix
decode kernels during the point, and three chronological cycle blocks with a
relative throughput spread of at most 10%. Otherwise `completed_req_per_s_peak`
is null and the raw observed value plus failure reasons remain available. The
word peak denotes observed backlogged capacity of the configured policy, not
proof of a globally optimal scheduler or an SLO-compliant rate. A server that
never drains its running batch may yield too few complete cycles; it is reported
as insufficient evidence rather than assigned a favorable short window.

`run_decode_saturation.py --output-dir PATH` runs the serial and dynamic-Top1
pair sequentially with decode=512, prefix=4608, query=480, 50 rank-16 adapters,
budget=36864, rate=8, duration=120 and seed=42. Select physical GPU1 with
`CUDA_VISIBLE_DEVICES=1`. It produces `comparison.json` after both complete.
Decode attention treats the actual private length as a nonspecialized runtime
argument, preserving the exact right-aligned KV indices; it no longer compiles
a separate scores/reduce kernel for every output position.

Full payloads contain actual prefix + query token IDs. Serial payloads remain
query-only. Both include explicit prefix/query/decode lengths and share the same
arrival and adapter trace. The client parses SSE frames, requires a terminal event and exactly 64
generated tokens, and rejects aborts, malformed streams, and nonfinite logprobs.
The cumulative token counter is required because the HTTP manager may coalesce
intermediate token updates; counting network chunks or stream events is incorrect.
TTFT is the client-observed first valid token event, not a CUDA timestamp.

Steady throughput counts every successful completion in the completion window,
including requests that arrived before warmup. There is no arrival-window filter.
The effective end is min(requested steady end, last request end). Sending duration
and completion-window end are independent: `--duration 120 --warmup 70 --steady-end
280` sends for 120 seconds and measures completions over [70,280]. This window can
include backlog processing after sending stops. Use the same predeclared window
for all paired methods, and inspect runtime snapshots to ensure queues stay
backlogged and batches have not begun to shrink. If a method drains earlier,
select a shorter common window for the whole comparison, not each method's best
individual window. This statistic is backlogged service throughput; a single
offered rate or a short maximum bin does not establish peak/saturated capacity.
Global throughput divides all successful completions by time to the last request
end. Latency and TTFT use all successful requests, including warmup/drain requests.
Failures invalidate a point and stop the sweep rather than inflate throughput.

Each point resets prefix cache, adapters, dynamic slots, cache counters, and memory
peak counters. Model/static sources and Unified Paging backing remain allocated;
the serial workspace borrows slots only during each forward and releases them
afterwards. Unused adapters reserve no workspace. Snapshots between forwards
report zero current workspace and retain its measured peak token/byte occupancy.
JIT compilation and PyTorch's reserved cache survive resets. An eight-request
preflight runs before the sweep; each point still has its own timed warmup.

Optional low-frequency snapshots run after router forward steps, without CUDA
events or added device synchronization. They report waiting/running counts,
unique LoRAs, effective pool occupancy, cache hits/misses/evictions, workspace
backing, and allocated/reserved peaks. The reported serving peak subtracts actual
base-model GPU tensor storage, retaining all static, adapter, KV, and temporary
serving tensors. This is not NVML usage and does not include CUDA context memory.
Queue/batch distributions are snapshot-sample statistics, not time-weighted means.
The prefix hit ratio counts unique adapters per new batch across the entire point.
Full additionally reports `prefix_prefill_builds` and `prefix_prefill_tokens`.
Full reconstruction instead reports `prefix_reconstruct_builds` and
`prefix_reconstruct_tokens`; both prefill-build counters remain zero. To preserve
the exact full-reprefill admission policy, its batch limit still charges unique
miss prefix tokens plus new query tokens. `last_admission_tokens` records that
limit charge, while `last_prefill_tokens` records only actual query prefill in
full reconstruction. Reconstruction work is recorded separately.
Its independent `full_reprefill_req_queue.py` preserves FIFO admission and stops
at the first request that cannot fit even after evicting inactive LRU prefixes.
It charges one prefix per adapter, running/new private KV peak (including future
decode), and adapter weights. The prefill cap charges unique misses' prefix
tokens plus all new query tokens, not repeated hit prefixes. Active prefixes are
pinned; finished requests release private KV and prefix references, not cached KV.

Snapshot occupancy need not reach 100%: the scheduler reserves future decode
capacity. Equal preallocated budgets need not yield lower allocated peaks for
serial; the main possible gain is more active requests at the same budget.

## Output

- `manifest.json`: command, GPU inventory, environment, package versions, config.
- `server.log`: complete server logs, separate from client stdout.
- `runtime.jsonl`: monotonic-timestamped server snapshots, including drain.
- `preflight.json`: output-length and stream-completion checks.
- `rate*_seed*_requests.jsonl`: individual request timestamps/results.
- `rate*_seed*.json`: completion bins, all-request latency/TTFT, runtime statistics.
- `summary.json`: per-rate mean/std and conservative saturation evidence.

Both modes share `--trace-dir`; trace hashes must match for paired points.
All three modes can share the same trace directory. Use distinct HTTP, NCCL,
and internal ports for every concurrently running server. In the comparison
command, `--reconstruct-dir PATH` adds full reconstruction to the existing
full-reprefill versus serial comparison. It can also be used as `--full-dir`
to compare reconstructed full caching directly against layer serial.
Existing run output directories are rejected; use a new directory for reruns.
Old full-mode results from MiniKV reconstruction are a different baseline and
must not be mixed with these real-prefix-prefill results.
A waiting-nonempty snapshot fraction >=95% plus backlog at the window end is one
piece of evidence. Saturation is marked confirmed only after two consecutive
rate-increase pairs show <=5% relative throughput change and all seeds remain
backlogged. This is a heuristic: inspect variance and batch/slot curves as well.
If unconfirmed, add higher rates or lengthen the measurement, not a capacity claim.

## Commands

From `/home/qlh/S-LoRAc`, after confirming both GPUs are available:

```bash
mkdir -p benchmarks/results/lowra_saturation_logs
CUDA_VISIBLE_DEVICES=0 nohup /home/qlh/anaconda3/envs/slora/bin/python -u \
  benchmarks/lowra_saturation_exp.py --mode full-reprefill --dummy \
  --output-dir benchmarks/results/lowra_saturation/full \
  > benchmarks/results/lowra_saturation_logs/full.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup /home/qlh/anaconda3/envs/slora/bin/python -u \
  benchmarks/lowra_saturation_exp.py --mode minikv-layer_serial --dummy \
  --output-dir benchmarks/results/lowra_saturation/layer_serial \
  > benchmarks/results/lowra_saturation_logs/layer_serial.log 2>&1 &
```

Once both finish:

```bash
/home/qlh/anaconda3/envs/slora/bin/python benchmarks/compare_lowra_saturation.py \
  --full-dir benchmarks/results/lowra_saturation/full \
  --serial-dir benchmarks/results/lowra_saturation/layer_serial \
  --output benchmarks/results/lowra_saturation/comparison.json
```

Concurrent GPUs share host CPU and potentially PCIe resources. For formal
measurements, repeat sequentially and swap devices to quantify hardware bias.
The program stops only its own server process group when done/interrupted.

## Tests

The CPU-only tests can be run manually (they do not start servers):

```bash
/home/qlh/anaconda3/envs/slora/bin/python -m unittest discover -s test -p test_lowra_saturation_exp.py -v
```

The existing `test_lowra_layer_runtime.py` covers GPU reconstruction and lifecycle;
these client tests are not a substitute for an end-to-end serving smoke.
