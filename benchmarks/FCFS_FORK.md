# Wait one decode step or shrink now?

This is a paired measurement, **not a new production scheduler**. Both
branches run the TP=1 router, model forwards, elastic cache and bounded
reconstruction kernel. As in the previous serving runs the Llama-13B model
and adapters have dummy weights but real dimensions and GPU computation.
Timing ends at router token emission; there is no HTTP or detokenizer.

## Run

From the repository root, on an idle 80 GB A100 (allow activation headroom):

```bash
OUT="benchmarks/results/fcfs_fork_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 nohup bash benchmarks/run_fcfs_fork.sh "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
```

Defaults: prefix=10000, pool=65000, 50 rank-16 adapters, first LMSYS
turns, full MAF d01, skewness=1, Poisson 3 req/s for 180 s, seeds 42.
The same trace supplies both arms. Known query lengths are visible to the
planner; recorded answer lengths terminate generation only.
`RATE=5` selects a separate pressure run, not extra samples for the 3 req/s curve.
To reuse an existing frozen trace:

```bash
CUDA_VISIBLE_DEVICES=0 /home/qlh/anaconda3/envs/slora/bin/python -m benchmarks.fcfs_fork_exp \
  --trace "$TRACE" --out "$OUT/pairs" --points 30
```

Set `POINTS=2` on the shell launcher for a pilot. Discovery replays and
drains the trace once. Each selected point costs **two fresh model loads and
two roll-ins up to that boundary**. 180 s is the *arrival* duration, not the
total runtime. Interrupted after discovery? Repeat the Python command with
`--resume`, the same trace, directory, GPU, code revision and sampling
settings. It skips completed arms, not half-finished ones. Do not run two
drivers in the same directory concurrently.

## What is compared

* Normal policy: strict FCFS; release finished requests, count actual private
  KV and next-step headroom, reclaim inactive entries by LRU. Admit what fits
  without shrinking first. Use slack to fully retain a prefix when the visible
  queue clears, without extra idle evictions. Under pressure, decrement active
  donors by rank, partial before full, fewer active requests, then adapter ID.
  This is a reproducible roll-in and continuation policy, **not an optimality claim**.
* A fork requires a running batch, no free admission left, a fresh blocked head,
  and some consecutive FCFS requests that fit after demotion. Prefix/scratch
  rounding goes through the same allocator model as the adaptive engine.
  Try up to 1, 2, 4 or 8 extra requests (cap rotated by decode step); take the
  largest feasible queue prefix within that cap. This is not a sweep over every
  possible action at the same point. Cases without both alternatives are excluded.
* Discovery records the trace cursor at every boundary. After ten measured
  decode steps it samples up to 30 spaced opportunities (including drain),
  uniformly from the spaced candidates. Reproduce prior arrivals at the recorded
  boundaries, not wall-clock playback from an unrelated run. Verify request
  IDs, generated history, queue order, private page mappings, cache targets,
  ready layers, physical prefix slots, allocator mask and resident adapters.
  A mismatch aborts rather than silently contributing data.
* A waits **one decode**, B reclaims just enough under the donor order and
  actually prefills the extra queue prefix. Then both switch to the same policy
  and original wall-clock arrival offsets. Inference speed may legitimately
  change *which step* sees a future arrival. Arm execution order alternates
  across pairs. Clocks start before the intervention, including allocation,
  weight transfer, prefill and reconstruction.

## Interpretation

`journal.json` contains fork features and the roll-in journal; each
`NNN-{wait,shrink}.json` contains token emission times. `pairs.csv` records
5x/10x/20x the median duration of the ten preceding decodes. Windows are
identical wall times across arms; crossing forwards count only tokens emitted
before the cutoff. Head TTFT is from trace arrival; null means not yet emitted
and is reported as censored. Running ITL includes the gap across the fork.
`running_without_token` also counts old requests receiving no token in-window.

`gain.svg` plots rho versus shrink-minus-wait output tokens/s over the 10x
window. New group means admission introduced a previously inactive adapter
that will stream layers; full-cache hits fall in the other group. Demotions
that create additional streams for already active adapters are logged separately.
The fit minimizes the sum of output tokens forgone relative to the better arm
at each sampled point. No threshold is reported unless both signs occur and a
cut beats both constant policies on calibration. This is not evidence of a
universal crossover, an SLO guarantee, or whole-run throughput. Sparse bins,
JIT warm-up and short-window timing noise require inspecting the raw pairs.
For validation use an independently sampled trace, not another window of the
same run:

```bash
CUDA_VISIBLE_DEVICES=0 SEED=43 DATA_SEED=43 bash benchmarks/run_fcfs_fork.sh "$VALID"
/home/qlh/anaconda3/envs/slora/bin/python -m benchmarks.fcfs_fork_report "$VALID/pairs" --train "$OUT/pairs"
```

Compare held-out lost tokens, wrong signs, TTFT censoring and running ITL with
always-shrink and always-wait. If there are no eligible points, this run
provides no threshold evidence; report it rather than reducing the pool silently.
