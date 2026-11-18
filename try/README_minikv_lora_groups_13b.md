# Same rank, different LoRA combinations

`profile_minikv_lora_groups_13b.py` tests whether changing adapter identities
changes reconstruction latency when rank, active adapter count and prefix shape
stay fixed. The original rank comparison script and production code are unchanged.

## Default Experiment

- Llama-13B MHA shape: 40 layers, hidden size 5120, FP16, TP=1.
- Prefix length 10000 tokens; ranks 8, 16, 32, 64, tested separately.
- 50 synthetic nonzero adapters per rank; each measured group selects 6.
- Group `first`: IDs 1, 2, 3, 4, 5, 6.
- Group `second`: IDs 10, 11, 12, 13, 14, 15.
- Eight additional distinct combinations selected with seed 42. Groups may share
  some IDs, but no group contains duplicate IDs. All groups are sorted by ID.
- The same ten combinations are used at every rank and in all three repetitions.
  Each rank runs three shuffled rounds, with 10 warmup and 50 measured sweeps per
  group per round: 40 configurations, 150 measurements each, 6000 measurements.
- A sweep reconstructs K and V for all 40 layers, overwriting the same one-layer
  output workspace. The number 6 counts unique LoRAs, not requests.

## Controls And Reused Code

One source pool is allocated per rank and stays alive across every group and
round. All 50 adapters' Base/MiniKV sources and packed GPU weights stay fixed;
switching groups changes only metadata, not source values or allocation addresses.
Only 6 prefixes' single-layer output cells are reserved, not 50 prefixes' outputs.

Keeping all 50 adapters' weights resident is a microbenchmark control, not a new
serving policy. It differs from the older rank sweep's active-only GPU weight
pool. Compare groups **within this experiment**; do not treat it as a strict
cross-script performance comparison. The rank-64 fixture's static sources and
dynamic pool use approximately 23.4 GiB, plus metadata/validation temporaries;
the script checks available memory with 1 GiB initialization headroom.

`GroupFixture` inherits the earlier profiler's layer execution, full sweep,
PyTorch reference validation, CUDA/wall timing, guard checks and cleanup. The
actual kernel remains `reconstruct_prefix_layer_bounded`, with workers=1296 and
tile configuration `(32,128,16,4)`. K includes delta-K RoPE and BaseK addition;
V includes BaseV addition. Actual HF-style CPU packing and `InferAdapter` GPU
loading are used during setup, outside measurement.

No trained model or dataset is loaded. MiniKV and LoRA weights are independent,
fixed-seed random tensors, not outputs collected from model prefill. This tests
execution cost, not model accuracy. It also changes source addresses along with
adapter identity, so a difference cannot be attributed solely to weight values.

First/middle/last layers of each combination are checked against the original
profiler's FP32 PyTorch reference before its first timing. A mismatch stops the
run. Primary timing includes only reconstruction wrapper launches and GPU work,
not initialization, index preparation, H2D, attention or FFN. Each CUDA interval
spans all 40 layers without per-layer synchronization.

The script does not inspect or wait for other GPU processes. Numerical and
available-memory checks remain enabled. Other workloads may affect timings.

## GPU1 Command

```bash
cd /home/qlh/S-LoRAc
RUN_DIR="try/results/minikv_groups_13b_p10000_u6_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=1 nohup /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_minikv_lora_groups_13b.py \
  --prefix-len 10000 --pool-adapters 50 --unique-loras 6 \
  --ranks 8 16 32 64 --random-groups 8 \
  --workers 1296 --warmup 10 --iterations 50 --repeats 3 \
  --seed 42 --output-dir "$RUN_DIR" \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &
echo "PID=$! RUN_DIR=$RUN_DIR"
tail -f "$RUN_DIR/run.log"
```

The selected physical GPU1 appears as logical `cuda:0`; no `--device cuda:1`
should be added to this command. The script does not automatically start a server.

List the exact one-based adapter IDs without allocating GPU memory:

```bash
/home/qlh/anaconda3/envs/slora/bin/python \
  try/profile_minikv_lora_groups_13b.py --list-groups
```

Optional short CUDA correctness run (two layers, 129 tokens, not formal timings):

```bash
CUDA_VISIBLE_DEVICES=1 /home/qlh/anaconda3/envs/slora/bin/python -u \
  try/profile_minikv_lora_groups_13b.py --smoke --random-groups 0 \
  --warmup 1 --iterations 2 --repeats 1 \
  --output-dir "try/results/groups_smoke_$(date +%Y%m%d_%H%M%S)"
```

## Results

- `result.json`: incremental run state, validation errors, full raw timing samples,
  one-based IDs, per-rank memory usage, execution order and source hashes.
- `groups.csv`: each combination's 40-layer mean/p50/p95, mean per-layer time,
  wall time, within-visit timing CV and between-round CV for the same group.
- `rank_comparison.csv`: each rank's fastest/slowest group, max/min mean ratio,
  between-group CV, median within-group round CV, and normalized between-group CV.
- `run.log`: source/weight setup, validation and timing progress.

`round_normalized_ratio` divides a group's round mean by the median of all group
means in that same round, then averages those ratios. This is a descriptive check
for common round drift, not a statistical significance test. Only interpret the
final comparison after `state=complete`, `all_passed=true`, and all rank summaries
have `complete=true`. Partial output can have unequal group/round coverage.

Small between-group variation relative to repeated-run variation supports using
one cost estimate for these same-rank/same-shape groups on this implementation.
It does not prove exact equality, nor guarantee the same result under other
source layouts, cold-cache conditions, real serving interference or architectures.
