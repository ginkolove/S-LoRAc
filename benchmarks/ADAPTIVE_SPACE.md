# Optional adaptive prefix space planning

`minikv-adaptive` is an experimental additional mode. Existing full, serial,
prefetch, and retained-layer modes keep their own execution and scheduling paths.
It currently requires single-GPU (`tp=1`) LowRA with the online scheduler.

## State and execution

- Static BaseKV and all MiniKV stay in the existing static budget.
- Each adapter has one shared retained prefix with an independent `K` in `[0,L]`.
  The first `K` layers persist; other layers reuse a single-layer workspace.
- Persistent storage uses `ceil(K * prefix_len / L)` ordinary all-layer slots.
  The page-major index mapping preserves existing KV when pages are added or
  removed. No intermediate full prefix is allocated during promotion.
- A plan first evicts eligible idle entries, then shrinks entries, then expands
  them. Newly retained layers are built lazily in the next forward.
- Request-private query/decode KV is never discarded by a retention change.
  If even all-serial cannot advance, the existing online preempt/recompute path
  remains the final safety mechanism.
- Idle ready prefixes remain cached and are evicted by LRU when capacity needs
  them. A zero-layer entry has no persistent prefix allocation.
- Reconstruction and attention use the existing numerical kernels, including
  bounded reconstruction in the supplied experiment commands.

## Decision mechanism

Normal expensive replanning runs every 10 decode steps. Admission still occurs
at `--schedule-interval` (default 1), and finished requests are released each
step. Memory pressure, an empty running batch, or an aged waiting head can force
an earlier replan. Every decode still checks immediate capacity.

Only currently known query/history lengths enter admission. Reference answer
lengths and generation caps are not used to predict memory or remaining work.
The horizon normally reserves 10 steps of private-KV growth, assuming every
running request survives. This is a conservative reservation, not a prediction.
It can fall back to one step if a longer reserve would prevent progress.

For a candidate running set plus an FCFS prefix of the waiting queue, the
planner maximizes the surrogate:

```
score = horizon * candidate_request_count /
        (admission_cost + new_retained_layer_build_cost + horizon * step_cost)
```

`step_cost` includes measured core inference plus reconstruction of missing
layers. Reconstruction is counted per unique adapter, not once per request.
Candidates must fit private KV, horizon growth, resident LoRA weights, retained
prefixes, a single-layer workspace, and any idle entries not reclaimed by LRU.
The exact allocator rounding is used. Shared static memory has already been
subtracted from the dynamic pool capacity.

CUDA events sample all prefills, early decode steps, and then every configured
interval. Nonnegative regressions model core inference and grouped reconstruction
using request count, known context length, unique adapters and rank sum. Adapter
loading uses a wall-time moving average. The first two observations are excluded
from fitting. Until at least three decode and three reconstruction observations
exist, admission is greedy FCFS with all-serial execution.

The search tests endpoint seeds and layer-count changes/swaps with a small beam.
It considers up to 64 FCFS prefixes, refines the best four candidate request sets,
and uses 4-layer then 1-layer increments. Search time is bounded; capacity checks,
candidate enumeration, model fitting and allocation are additional overhead.
A gain margin of at least 5%, increased with measured model error, reduces
oscillation. This is not an exhaustive solver or a global optimality guarantee.

The default 3000 ms waiting-age threshold prioritizes admitting the oldest
waiting request when feasible. It is **not** a 3000 ms TTFT guarantee. The
optional `--space-plan-max-step-ms` checks predicted step/admission costs, not a
hard real-time deadline. Neither mechanism can guarantee SLOs under overload.

## Parameters

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--space-plan-interval` | 10 | Normal replanning / decode observation interval |
| `--space-plan-window` | 10 | Progress and memory reservation horizon |
| `--space-plan-group` | 4 | Coarse retained-layer search increment |
| `--space-plan-beam` | 8 | Search beam size |
| `--space-plan-candidates` | 64 | Maximum waiting requests considered per decision |
| `--space-plan-budget-ms` | 10 | Refinement search time budget, not total router latency |
| `--space-plan-max-wait-ms` | 3000 | Waiting-age priority; 0 disables |
| `--space-plan-max-step-ms` | 0 | Optional predicted step/admission limit; 0 disables |
| `--space-plan-hysteresis` | 0.05 | Minimum fractional improvement before switching |

Do not combine the adaptive mode with fixed `--retain-layers`, `--retain-top-k`
or pinned-hot-adapter options. Use the existing modes for those controls.

## LMSYS/MAF experiments

The existing wrapper defaults to the two baseline modes. Opt in using
`--adaptive-only`, for example:

```bash
bash benchmarks/run_lmsys_preemption.sh OUTPUT_DIR \
  --adaptive-only --model-size 7b --prefix-len 12000 --gpu 0 \
  --request-rate 2 --duration 120 \
  --space-plan-interval 10 --space-plan-window 10
```

Latest matched configurations are 7B/prefix12000/pool125000,
13B/prefix10000/pool65000 and 30B/prefix2000/pool8800. All use 50 rank-16 adapters,
LMSYS-33K sampled with seed 42, MAF d01 minutes 1..1440 at skewness 1, and Poisson
arrivals. The wrapper reads the established memory profiles; these are not an
automatic guarantee against activation OOM for arbitrary workloads.

The trace reference answer lengths control termination only. Whole-run metrics
include draining all requests. Steady starts at zero and ends at actual completion
in these commands. Inspect saturation separately before calling a result peak
throughput. The wrapper refuses to run on an occupied GPU or overwrite a completed
suite; it does not kill other processes.

## Diagnostics and validation

- `minikv-adaptive/rate*_seed42.json`: request metrics and preemption summary.
- `minikv-adaptive/runtime.jsonl`: memory, batch and cache statistics.
- `minikv-adaptive/runtime.plans.jsonl`: per-decision layer counts, selected
  requests, required slots, score, prediction error, calibration state and search
  time. These include preflight; use `benchmark_origin_monotonic` from the result
  JSON to select the actual replay interval.
- `minikv-adaptive/runtime.preemptions.jsonl`: existing preemption events.
- `minikv-adaptive/manifest.json`: arguments and source hashes.

Preflight observations warm the cost model; resetting the runtime clears cached
KV but intentionally keeps timing observations. For ablations, keep the preflight
configuration identical and report this warmup policy.

CPU and opt-in GPU tests:

```bash
python -m unittest discover -s test -p test_adaptive_space.py -v
CUDA_VISIBLE_DEVICES=0 RUN_ADAPTIVE_CUDA_TESTS=1 \
  python -m unittest discover -s test -p test_adaptive_space.py -v
```

GPU tests compare small-model logits against serial reconstruction through mixed
retention changes and exercise real worker admission, merge, filter and drain.
They validate execution, not the accuracy of MiniKV relative to full LoRA
prefill, nor performance optimality on 13B/30B. Formal workload experiments must
establish whether adaptive planning actually improves throughput and latency.
