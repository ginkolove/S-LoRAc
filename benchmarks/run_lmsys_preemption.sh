#!/usr/bin/env bash
# Frozen LMSYS/MAF trace: paired baselines, or the optional adaptive mode.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: bash %s OUTPUT_DIR [--model-size 7b|13b|30b] [--prefix-len TOKENS] [--gpu 0|1] [--request-rate REQ_PER_S] [--duration SECONDS] [--max-context-len TOKENS] [--batch-max-tokens TOKENS] [--running-max-req-size COUNT] [--schedule-interval STEPS] [--adaptive-only] [--space-plan-interval STEPS] [--space-plan-window STEPS] [--dry-run]\n' "$0" >&2
  exit 2
fi
OUT="$(realpath -m "$1")"
shift
DRY_RUN=0
SIZE=7b
GPU=1
PREFIX_LEN=
REQUEST_RATE=3
DURATION=180
CONTEXT_ARGS=()
BATCH_ARGS=()
SCHEDULE_INTERVAL=1
ADAPTIVE_ONLY=0
PLAN_ARGS=()
MODES=(minikv-full_reconstruct minikv-layer_serial)
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --adaptive-only) ADAPTIVE_ONLY=1; MODES=(minikv-adaptive); shift ;;
    --space-plan-interval|--space-plan-window|--space-plan-group|--space-plan-beam|--space-plan-candidates|--space-plan-budget-ms|--space-plan-max-wait-ms|--space-plan-max-step-ms|--space-plan-hysteresis)
      if [ "$#" -lt 2 ]; then printf 'Missing planner option value\n' >&2; exit 2; fi
      PLAN_ARGS+=("$1" "$2"); shift 2 ;;
    --model-size|--gpu|--prefix-len|--request-rate|--duration|--max-context-len|--batch-max-tokens|--running-max-req-size|--schedule-interval)
      if [ "$#" -lt 2 ]; then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      case "$1" in
        --max-context-len|--batch-max-tokens|--running-max-req-size|--schedule-interval)
          if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
            printf '%s must be a positive integer.\n' "$1" >&2; exit 2
          fi ;;
      esac
      case "$1" in
        --model-size) SIZE="$2" ;;
        --gpu) GPU="$2" ;;
        --prefix-len) PREFIX_LEN="$2" ;;
        --request-rate) REQUEST_RATE="$2" ;;
        --duration) DURATION="$2" ;;
        --max-context-len) CONTEXT_ARGS=(--max-context-len "$2") ;;
        --batch-max-tokens|--running-max-req-size) BATCH_ARGS+=("$1" "$2") ;;
        --schedule-interval) SCHEDULE_INTERVAL="$2" ;;
      esac
      shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
for VALUE in "$REQUEST_RATE" "$DURATION"; do
  if [[ ! "$VALUE" =~ ^[0-9]+([.][0-9]+)?$ || ! "$VALUE" =~ [1-9] ]]; then
    printf 'Request rate and duration must be positive decimal numbers.\n' >&2
    exit 2
  fi
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/benchmarks/llama_memory_budget.sh"
llama_memory_budget "$SIZE"
TOKENS="$LLAMA_POOL_TOKENS"
if [ -z "$PREFIX_LEN" ]; then
  if [ "$SIZE" = 30b ]; then PREFIX_LEN=4000; else PREFIX_LEN=10000; fi
fi
if [[ ! "$PREFIX_LEN" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Prefix length must be a positive integer.\n' >&2
  exit 2
fi
if [ "${#CONTEXT_ARGS[@]}" -gt 0 ] && [ "$PREFIX_LEN" -ge "${CONTEXT_ARGS[1]}" ]; then
  printf 'Explicit max-context-len must exceed prefix-len.\n' >&2
  exit 2
fi
case "$GPU" in
  0|1) ;;
  *) printf 'Supported GPU indices: 0, 1\n' >&2; exit 2 ;;
esac
# This paired suite includes full reconstruction. Check even a one-token query
# can fit static BaseKV/MiniKV, one full prefix, rank-16 weights and KV headroom.
STATIC_TOKENS=$(((PREFIX_LEN * (LLAMA_HIDDEN + 50 * 16) + LLAMA_HIDDEN - 1) / LLAMA_HIDDEN))
MINIMUM_TOKENS=$((STATIC_TOKENS + PREFIX_LEN + 4 * 16 + 2))
if [ "$ADAPTIVE_ONLY" -eq 1 ]; then
  MINIMUM_TOKENS=$((STATIC_TOKENS + (PREFIX_LEN + LLAMA_LAYERS - 1) / LLAMA_LAYERS + 4 * 16 + 2))
fi
if [ "$MINIMUM_TOKENS" -gt "$TOKENS" ]; then
  printf 'Full reconstruction cannot fit: Llama-%s prefix=%s needs at least %s pool tokens, but the matched weights + pool profile allows %s. Choose a shorter --prefix-len explicitly (e.g. 3000 for 30b; actual private history must also fit); no requests were filtered and no GPU process was started.\n' \
    "$SIZE" "$PREFIX_LEN" "$MINIMUM_TOKENS" "$TOKENS" >&2
  exit 2
fi
PYTHON="${SLORA_PYTHON:-/home/qlh/anaconda3/envs/slora/bin/python}"
TRACE_PYTHON="${SLORA_TRACE_PYTHON:-/home/qlh/anaconda3/envs/transformer/bin/python}"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

run_command() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

check_gpu() {
  local PIDS WAIT_SECONDS="${1:-0}" ATTEMPTS=0
  while true; do
    if ! PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
      printf 'Cannot query GPU%s; refusing to start.\n' "$GPU" >&2
      exit 1
    fi
    if [[ -z "${PIDS//[[:space:]]/}" ]]; then return; fi
    if [ "$ATTEMPTS" -ge "$WAIT_SECONDS" ]; then
      printf 'GPU%s still has compute processes after %s release checks: %s. No processes were killed.\n' \
        "$GPU" "$ATTEMPTS" "$PIDS" >&2
      exit 1
    fi
    if [ "$ATTEMPTS" -eq 0 ]; then
      printf 'Previous run finished; waiting up to %ss for GPU%s CUDA cleanup. PIDs: %s\n' \
        "$WAIT_SECONDS" "$GPU" "$PIDS"
    fi
    sleep 1
    ATTEMPTS=$((ATTEMPTS + 1))
  done
}

if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$OUT" benchmarks/results
  exec 9>"benchmarks/results/lmsys_preemption_gpu${GPU}.lock"
  if ! flock -n 9; then
    printf 'Another GPU%s preemption suite is running.\n' "$GPU" >&2
    exit 1
  fi
  for MODE in "${MODES[@]}"; do
    if [ -e "$OUT/$MODE/manifest.json" ]; then
      printf 'Existing run at %s; choose a fresh output directory.\n' "$OUT/$MODE" >&2
      exit 1
    fi
  done
  check_gpu
fi
printf 'CUDA_VISIBLE_DEVICES=%s; Llama-%s, prefix=%s tokens, pool=%s tokens, rate=%s req/s, duration=%ss.\n' \
  "$GPU" "$SIZE" "$PREFIX_LEN" "$TOKENS" "$REQUEST_RATE" "$DURATION"
printf 'Memory profile: weights + pool=%s bytes (about 79.1 GB decimal), weights=%s bytes, slot=%s bytes; model buffers and activation peaks are additional.\n' \
  "$LLAMA_WEIGHT_POOL_BYTES" "$LLAMA_MODEL_WEIGHT_BYTES" "$LLAMA_SLOT_BYTES"
printf 'Selected modes: %s\n' "${MODES[*]}"
run_command "$TRACE_PYTHON" -u benchmarks/prepare_lmsys_maf_trace.py \
  --dataset /data/hub/LMSYS-33k \
  --maf-file /data/hub/MAF/azurefunctions2019/invocations_per_function_md.anon.d01.csv \
  --maf-skewness 1 --maf-minute-start 1 --maf-minute-end 1440 \
  --tokenizer "huggyllama/llama-${SIZE}" --num-adapters 50 \
  --prefix-len "$PREFIX_LEN" "${CONTEXT_ARGS[@]}" \
  --request-rate "$REQUEST_RATE" --duration "$DURATION" --seed 42 --data-seed 42 \
  --output "$OUT/trace.json"

RELEASE_WAIT=0
for MODE in "${MODES[@]}"; do
  if [ "$DRY_RUN" -eq 0 ]; then check_gpu "$RELEASE_WAIT"; fi
  EXTRA=()
  if [ "$MODE" = minikv-layer_serial ] || [ "$MODE" = minikv-adaptive ]; then
    EXTRA=(--reconstruction-kernel bounded --reconstruction-workers 1296)
  fi
  printf 'Starting %s on GPU%s; output=%s\n' "$MODE" "$GPU" "$OUT/$MODE"
  run_command "$PYTHON" -u benchmarks/lowra_saturation_exp.py \
    --mode "$MODE" "${EXTRA[@]}" "${PLAN_ARGS[@]}" \
    --model-dir "huggyllama/llama-${SIZE}" \
    --adapter-base "dummy-lora-${SIZE}-rank-16" --num-adapters 50 \
    --workload-trace "$OUT/trace.json" \
    --prefix-len "$PREFIX_LEN" "${CONTEXT_ARGS[@]}" \
    --num-token "$TOKENS" "${BATCH_ARGS[@]}" \
    --scheduler online --schedule-interval "$SCHEDULE_INTERVAL" --generation-mode trace \
    --request-rates "$REQUEST_RATE" --duration "$DURATION" --cv 1 --seeds 42 --model-seed 42 \
    --warmup 0 --steady-end 1000000000 \
    --startup-timeout 1800 --request-timeout 10800 --drain-timeout 10800 \
    --port "$((9400 + GPU))" --nccl-port "$((34400 + GPU))" --internal-port-start "$((50000 + GPU * 16))" \
    --output-dir "$OUT/$MODE" --dummy
  # Waiting for the server parent does not guarantee its CUDA contexts are gone.
  RELEASE_WAIT=60
done

if [ "$ADAPTIVE_ONLY" -eq 0 ]; then
  run_command "$PYTHON" benchmarks/compare_lowra_saturation.py \
  --full-dir "$OUT/minikv-full_reconstruct" \
  --serial-dir "$OUT/minikv-layer_serial" \
  --output "$OUT/comparison.json"
fi
if [ "$DRY_RUN" -eq 0 ]; then check_gpu 60; fi
printf 'Finished. Preemption diagnostics are in each rate*_seed42.json (preemption), with events in runtime.preemptions.jsonl.\n'
