# Online scheduling without output-length admission

Use `--scheduler online` on `benchmarks/launch_server.py` or
`python -m slora.server.api_server`. The managed
`benchmarks/lowra_saturation_exp.py` now defaults to `online` and forwards it to
the server. Use `--scheduler slora` to reproduce the old reservation policy.
Existing server/launcher defaults remain unchanged for compatibility.

This implements vLLM-style incremental admission and recompute preemption in
the existing S-LoRA engine. It does not import vLLM or replace the attention,
MiniKV reconstruction, or Unified Paging allocator.

## H20 comparisons under an 80 GB device budget

From the repository root, select one exclusive physical H20 and a fresh result directory.
The default methods are full reconstruction, layer-serial reconstruction and adaptive V3.

```bash
/opt/conda/bin/python benchmarks/run_h20_memory_cap.py benchmarks/results/h20_cap80 \
  --gpu 0 --point 13b:10000
```

Use `--profile-only` to measure the slot counts on the GPU and save them in
`<model>_p<prefix>/profiles.json`. Use `--dry-run` to print the planned configuration.
`--methods` also accepts CPU prefix reads, fixed-hot CPU reads, Impress, active
reprefill, the two rank-split methods and adaptive V1/V2/V4. Select all with `--methods all`.
For Impress, `--impress-read-percent 60` copies 6,000 positions of a 10,000-token
prefix and zero-fills the remaining 4,000. The default is 50%. All adapters, layers
and K/V tensors use the same prefix-position permutation with seed 42.
The scheduling limits and frozen trace stay common across
pool sizes.

Each method boots its own staged TP=1 worker with the minimum capacity for one request.
The worker records the model initialization peak and runs the actual adapter load,
prefix acquisition, prefill, decode and sampling kernels. Synthetic token IDs have
shapes bounded by the frozen trace's sorted prompt and private-history lengths.
The worker probes cold distinct LoRAs and a wide batch sharing a high-rank LoRA. It
also probes a wide mixed batch and partial retention when those shapes fit. Each probe
uses the staged scheduler's admission limits. Wide prefills include the private history
of requests being resumed after preemption. To exercise deep decode in one forward,
the worker fills the extra private KV positions with zeros. It records the batch widths,
prefix counts, lengths and PyTorch peaks for each phase in `forward_profile.json`.

The launcher takes the smaller of whole-device and MPS client headroom. The client
estimate combines the largest PyTorch reserved peak with the largest observed allocation
outside that allocator. After reserving one decimal GB it divides the remaining bytes by
the worker's FP16 KV bytes per slot to propose a larger pool. It verifies the proposal
with new probes at that actual capacity, including any larger admissible batches.
A CUDA OOM or insufficient margin triggers a smaller candidate. The counts in
`startup_profile.json` distinguish the total `--num-token` budget (`pool_tokens`)
from allocated dynamic slots (`dynamic_pool_slots`) and the MiniKV static region
(`static_token_equivalents`).

A timed experiment starts a clean worker with the verified capacity and replays the
original arrivals and outputs. Startup probes are outside the timing window and their
prefix state and adaptive timing samples stay in the profiling processes. A CUDA OOM
during replay triggers a fresh profile and replay at a smaller capacity.

For a repeated measurement on an exclusive H20, select one method and pass its previous
`startup_profile.json` with `--reuse-profile`. The launcher verifies the method, trace
hash, recorded successful 80 GB probe, pool geometry and remaining client budget. The
benchmark checks the server options, inference source hashes and Torch/Triton versions
before starting the server. The formal replay then uses those slots under the current
GPU's MPS limit and whole-device monitor. A reused capacity that encounters a CUDA OOM
is reported as a failed run. The new experiment's peak covers its own attempts, and the
copied profile records the path to the original measurement.

```bash
/opt/conda/bin/python benchmarks/run_h20_memory_cap.py benchmarks/results/replay_v3 \
  --gpu 5 --point 13b:10000 --methods minikv-adaptive-v3 \
  --reuse-profile benchmarks/results/h20_startup_capacity_13b10k/13b_p10000/minikv-adaptive-v3/startup_profile.json
```

Use `--num-token SLOTS` to go straight to formal replay at an explicit total slot
budget. The MPS quota and whole-device monitoring stay active. A CUDA OOM marks
the fixed-capacity run as failed. For the six 13B, 10K-prefix experiments on
GPUs 0–5, use the following command from the repository root. Both CPU-read
methods default to 45,532 slots. Pass `--num-token SLOTS` to this script to set
one common total slot count for all six GPUs. Each launch creates a fresh
directory and prints the path, controller PIDs and log files. Add `--dry-run`
to preview the settings.

```bash
bash benchmarks/run_h20_13b10k_formal.sh
```

During parallel launches NVML can list a router on several boards while CUDA initializes.
Each guarded router publishes its PID and assigned GPU UUID in its private MPS directory
and holds a lock for its lifetime. The guard excludes a router assigned to another board
only when every NVML entry for that PID on this board reports exactly zero bytes.
An unrecognized process or a peer with local device memory invalidates the attempt.
An interrupted formal replay stops on the first ownership conflict and records its
reason in the method's `failure.json` and the point's `incomplete.json`.
Startup probes may retry up to twice if a transient registration has cleared after teardown.

The launcher creates a private MPS daemon on the selected UUID. The TP=1 router is the
only GPU client. Its driver-enforced quota leaves the idle device usage plus two decimal
GB for overhead outside MPS accounting. Adjust the latter with `--outside-reserve-gb`.
Before loading weights the runner checks that current board usage plus the client's
remaining quota fits within 80,000,000,000 bytes. The whole-device bound depends on
exclusive access and this reserve covering later allocations outside the client quota.

Each attempt writes `memory.json` and `device_memory.csv`. NVML polls the physical GPU
every 20 ms from before MPS starts through teardown. The CSV records changes and the
report includes the longest polling gap. The device peak is sampled. A detected breach
or loss of monitoring invalidates the run. The point's `comparison.json` contains only
completed formal replays with verified request identities, output counts and runtime reset.
`startup_sampled_peak_bytes` includes all profiling attempts, even candidates that hit
the CUDA quota. `experiment_sampled_peak_bytes` also includes every formal attempt.
If a method still runs out of memory at its single-request pool floor, the runner records
that failure, finishes the other methods, and writes `incomplete.json`.

## MiniKV adaptive V4

The H20 launcher stages the V4 router and worker together. For a 13B model and a
10,000-token prefix at a fixed total budget of 61,000 slots, run from the repository root

```bash
PYTHONPATH=. /opt/conda/envs/slora/bin/python benchmarks/run_h20_memory_cap.py \
  benchmarks/results/h20_v4_13b10k --gpu 6 --point 13b:10000 \
  --methods minikv-adaptive-v4 --num-token 61000 --device-capacity
```

V4 changes retention between forwards. It uses the **dynamic** pool after the
static MiniKV allocation. A new rank starts with the full prefix if the pool
can hold one full copy, the smallest configured adapter and two private slots
for a minimal request. A smaller pool uses layer-serial reconstruction throughout.

- At an admission boundary, V4 tries the existing layout for the FCFS queue and
  reclaims idle prefixes in LRU order. If the first remaining request still needs
  space, V4 reduces retention together for all distinct prefixes of the lowest
  rank in the proposed batch.
  It keeps the largest common layer count that admits that request, then admits
  any additional requests that fit. It can move to the next rank once the lower
  rank reaches zero. Each new prefix with an already active rank inherits its depth.
- Before each forward, existing requests' observed private histories must fit.
  Growth can force another grouped reduction even with an empty waiting queue.
  If the serial layout also exhausts the pool, the router uses its preemption
  and recovery path.
- After a completion, cancellation or capacity increase, an empty waiting queue
  and sufficient spare slots allow a promotion trial. V4 restores the highest-rank
  group that can gain layers. Promotions leave headroom based on observed output
  batch size and preemption deficits. Admissions can use the entire dynamic pool.
- For an elective change, the reference is actual output tokens per elapsed
  second in the previous layout. A runnable batch first supplies a reference
  window. If the very first request needs a smaller layout to start, it can
  start immediately. The defaults are a one-second window and a 5% tolerance.
  Two consecutive windows below 95% of the reference trigger rollback. If the
  old layout would displace admitted requests, V4 pauses new admissions and
  retries restoration at subsequent boundaries. Set `--v4-feedback-seconds`
  and `--v4-hysteresis` to change the window and tolerance.

The formal run's `runtime.plans.jsonl` records the trigger, target layers per prefix,
required and available slots, observed output rate, reference rate and rollback state.

## Admission and recovery

- FCFS waiting order. `--max-active-loras N` optionally caps distinct running
  adapters, not requests; incompatible requests can be skipped at this limit.
- `--schedule-interval 1` tries admission after each decode step. Larger values
  change admission frequency only: memory is still checked before every decode.
- Admission uses known prompt/history tokens, one-step headroom, unique adapter
  weight slots, shared prefix allocations, and the single-layer scratch budget.
  It never reads `max_output_len`, `lowra_decode_len`, or reference answer lengths.
- An idle shared prefix is retained for reuse. Under pressure, unpinned idle
  entries are evicted in LRU order before requests are preempted.
- If decode still cannot fit, running requests are preempted from the tail.
  Private KV is freed and shared-prefix references are released. The request
  returns to the front of the waiting queue without completing or emitting a
  duplicate token.
- Resume prefills the original private prompt plus generated token history.
  Output token counts, stopping conditions, and presence/frequency penalties
  continue from the original request. Shared prefixes follow the existing cache
  policy; static BaseKV and MiniKV are not discarded.
- If a single request cannot fit even after reclaiming eligible idle caches,
  it fails explicitly through the existing abort response instead of hanging
  or causing a pool allocation exception. Pinning may reduce usable capacity.

For running request i, private KV after its next decode is `input_i + emitted_i`.
For an admitted/resumed request j, the conservative prefill-plus-next-decode
budget is `input_j + emitted_j + 1`. These are observed lengths, not predictions.
Prefix memory is charged once per adapter. Layer storage keeps the existing
`ceil(prefix_tokens * retained_layers / model_layers)` packing, and transient
scratch space is also charged. Old/new top-K allocations are considered during
rebalance so admission does not rely on memory that has not been released yet.

`max_new_tokens` remains a legal generation cap, not a required known output
length. Context-length checks still apply. The old HTTP check that required the
entire input-plus-output cap to fit the KV pool is disabled for `online`.

## Benchmark generation semantics

Each timed replay also measures delivered output throughput in consecutive 10-second
windows, from replay start through the last request, including drain. The benchmark
prints throughput as each full window closes. Each streaming
response contributes the increase in its cumulative output token count at the time
the benchmark client receives it. The point's `rate*_seed*.json` stores every window
under `result.output_windows_10s`, including empty windows and the final partial
window at its actual duration. `result.peak_output_window_10s` contains the highest
complete window of a valid replay, with its start, end, token count and tokens per
second. Ties select the earliest window. A replay shorter than 10 seconds has a
null peak. `summary.json` identifies the best measured window across rates and
seeds, and the H20 `comparison.json` reports each method's peak.

The default `--generation-mode trace` keeps deterministic recorded/synthetic
output lengths and ignores EOS for controlled performance replay. In online
mode those lengths control termination only, not admission. This is still trace
replay, not a claim that the model naturally produced those answer lengths.

For actual EOS-terminated generation, add:

```bash
--scheduler online --generation-mode eos --generation-max-new-tokens 512
```

This uses one common output cap (clamped to remaining context) independent of
the dataset's answer lengths. Success accepts early EOS, and throughput counts
actual returned tokens. Do not use dummy weights to assess natural answer
quality or realistic EOS behavior.

Manifests record the scheduler, interval, adapter cap, generation mode, and
`output_length_admission`. Runtime JSONL includes cumulative
`online_preemptions`, `online_recomputed_tokens`, and `online_rejections`.
These counters reset with the idle-runtime reset before each measured replay.
The paired-comparison script rejects runs with mismatched scheduling semantics.

## Scope

All existing prefix modes use the same online planner: full-reprefill,
minikv-full_reconstruct, minikv-layer_serial, layer_prefetch, and
minikv-layer_retained, including dynamic top-K and pinned entries.

This version retains whole-query/recompute prefill and S-LoRA token slots.
It does not add chunked prefill, mixed prefill/decode kernel execution,
CPU swapping of private KV, or an SLO guarantee. `batch_max_tokens` still limits
whole-prefill work and running decode width, rather than implementing all V1
token-budget semantics. Large prompts must fit the configured prefill budget.
Legacy adapter prefetch, clustering, abort heuristics, BMM, quantized modes, separate LoRA pools,
and debug weight-bypass modes are rejected with `online`.

## Regression tests

```bash
CUDA_VISIBLE_DEVICES='' /home/qlh/anaconda3/envs/slora/bin/python -m unittest discover \
  -s test -p test_online_scheduler.py -v

CUDA_VISIBLE_DEVICES=0 RUN_ONLINE_CUDA_TESTS=1 \
  /home/qlh/anaconda3/envs/slora/bin/python -m unittest discover \
  -s test -p test_online_scheduler_cuda.py -v
```

The opt-in GPU test uses a small three-layer, 1024-hidden dummy model with
nonzero MiniKV/BaseKV. It compares uninterrupted and preempted/resumed greedy
outputs, with sampling penalties, and verifies private/prefix/workspace cleanup.
It is a correctness test, not a serving-throughput benchmark or a TP=2 test.

# Automatic Replay Limits

The LMSYS launchers no longer impose a 16,384-token context limit, a
36,864-token prefill cap, or a 256-request running limit by default.
`prepare_lmsys_maf_trace.py` selects nonempty examples in a fixed seed order
without filtering by length, then derives the total context limit from
`prefix_len + max(query_len + answer_len)` over the selected requests.
Changing only the prefix length preserves sample IDs, query/output lengths,
arrival times and adapter assignments. This remains a first-turn workload with
a synthetic shared prefix, not a native multi-turn agent trace.

An explicit `--max-context-len` that is too small fails rather than silently
selecting shorter samples. The data-preparation-only `--filter-over-context`
option reproduces the old filtering policy when an explicit cap is supplied.
Existing frozen traces remain readable; their previously filtered-out samples
cannot be restored by raising a server limit. Generate a fresh trace/output
directory for the new sampling policy.

The benchmark inherits context length from the trace. Online prefill capacity
defaults to `num_token`, and the running-request limit defaults to the smaller
of the pool token count and the number of trace requests (pool count for
synthetic workloads). These are nonbinding bounds relative to the pool/trace,
not output-length reservations. The online scheduler continues to use only
known input/history plus immediate KV headroom, checking space before each
decode and using existing eviction/preemption when necessary. A prefill token
cap no longer also limits the decode batch's request count.

The paired launcher accepts explicit `--max-context-len`, `--batch-max-tokens`,
`--running-max-req-size`, and `--schedule-interval` overrides. Context and prefill
budgets are independent for online scheduling: a long shared prefix does not
require an equally large query-prefill cap. Explicit small prefill caps can
still make individual requests inadmissible; chunked prefill is not introduced.
Legacy S-LoRA scheduling keeps its original length-budget validation.

Ordinary LLaMA RoPE tables now cover at least the configured server context
limit without changing frequencies or wrapping positions. This permits long
performance experiments, not a claim of long-context model accuracy. KV pool
budgeting still does not guarantee enough memory for activations, workspaces
and other allocations. Keep GPU headroom and use an explicit prefill cap when
needed; removing fixed experimental caps is not an OOM guarantee. Resolved
limits are printed at launch and saved in the manifest.

## Matched Model Weights + Pool Budget

The three LMSYS shell launchers share `llama_memory_budget.sh`. The current
profile retains the original A100-80GB paper pool budgets for 7B (125,000 tokens)
and 13B (65,000 tokens), and assigns 8,800 tokens to 30B so that **weights plus
the total pool are about 79.1 GB (decimal) for all three models**. This supersedes
the earlier 73-decimal-GB cap and idle-memory-minus-7-GB policy. The comparison
uses measured FP16, TP=1 `base_model_weight_bytes`, excluding model buffers such
as RoPE tables, and `4 * layers * hidden_size` bytes per pool token. The 30B
model is our 60-layer, 6656-hidden model, not the original modified S5 model.
This is a frozen reproducible profile, not a live GPU-memory autosizer.

| Model | Weight bytes | Bytes per token | Default pool tokens | Weights + pool GB |
| --- | ---: | ---: | ---: | ---: |
| Llama-7B | 13,476,831,232 | 524,288 | 125,000 | 79.012831 |
| Llama-13B | 26,031,728,640 | 819,200 | 65,000 | 79.279729 |
| Llama-30B | 65,057,887,232 | 1,597,440 | 8,800 | 79.115359 |

The original budgets are in `paper/synthetic/slora/exp_cmds_a100-80_S1.sh` and
`paper/synthetic/slora/exp_cmds_a100-80_S4.sh`.

The measurements come from per-rank `base_model_weight_bytes` in `runtime.jsonl`:

- `results/llama7b_prefix15k_req3_180s_20261114_121857/`
- `results/lmsys_preemption_13b_p10000_r5_t240_gpu0_20261111_092926/`
- `results/lmsys_preemption_30b_p4000_r3_t180_20261111_062609/`

The pool budget still includes static BaseKV/MiniKV and dynamic prefix, private
KV, adapter and layer-workspace allocations. Both idle A100-80GB GPUs reported
81,153 MiB available on 2026-11-14, approximately 85.095 GB. The remaining
5.815-6.082 GB must cover model buffers, runtime activations, pool-external
temporaries, allocator fragmentation and CUDA costs.
This is not a hard peak-memory limit or an OOM guarantee. Recalibrate for a
different GPU, dtype, TP degree, model or substantially larger model buffers.
Direct `lowra_saturation_exp.py --num-token` values and the rank launcher's
explicit token override remain authoritative and can leave this matched profile.

**30B with a 4,000-token prefix can admit short queries but cannot complete the
whole current trace in full reconstruction at this budget.** Its static region
uses 4,481 equivalent slots; one full prefix and a rank-16 adapter leave only
255 slots for private prompt/history. A one-token query plus immediate headroom
fits (8,547 slots total), so the launcher's minimum-capacity check passes, but
some queries and the longest private history (1,529 tokens) cannot fit. An
explicit `--model-size 30b --prefix-len 3000` passes that single-request history
capacity check. The suite never silently shortens the prefix or drops samples
to fit. The early check is only a minimum capacity check; actual query/history
growth and execution peaks still matter.
