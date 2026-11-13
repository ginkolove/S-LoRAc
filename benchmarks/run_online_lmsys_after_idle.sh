#!/usr/bin/env bash
# Run each model on its GPU, optionally waiting for both GPUs to become idle.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: bash %s OUTPUT_DIR [--dry-run] [--no-wait]\n' "$0" >&2
  exit 2
fi
OUTPUT_ARG="$1"
shift
DRY_RUN=0
NO_WAIT=0
for OPTION in "$@"; do
  case "$OPTION" in
    --dry-run) DRY_RUN=1 ;;
    --no-wait) NO_WAIT=1 ;;
    *) printf 'Unknown option: %s\n' "$OPTION" >&2; exit 2 ;;
  esac
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(realpath -m "$OUTPUT_ARG")"
PYTHON="${SLORA_PYTHON:-/home/qlh/anaconda3/envs/slora/bin/python}"
TRACE_PYTHON="${SLORA_TRACE_PYTHON:-$PYTHON}"
cd "$ROOT"
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

prepare_trace() {
  local SIZE="$1"
  run_command "$TRACE_PYTHON" -u benchmarks/prepare_lmsys_maf_trace.py \
    --dataset /data/hub/LMSYS-33k \
    --maf-file /data/hub/MAF/azurefunctions2019/invocations_per_function_md.anon.d01.csv \
    --maf-skewness 1 --maf-minute-start 1 --maf-minute-end 1440 \
    --tokenizer "huggyllama/llama-${SIZE}" --num-adapters 50 \
    --prefix-len 10000 --max-context-len 16384 \
    --request-rate 5 --duration 60 --seed 42 --data-seed 42 \
    --output "$OUT/traces/llama${SIZE}/trace.json"
}

run_model() (
  local GPU="$1" SIZE="$2" TOKENS="$3"
  export CUDA_VISIBLE_DEVICES="$GPU"
  for MODE in minikv-full_reconstruct minikv-layer_serial; do
    EXTRA=()
    if [ "$MODE" = minikv-layer_serial ]; then
      EXTRA=(--reconstruction-kernel bounded --reconstruction-workers 1296)
    fi
    run_command "$PYTHON" -u benchmarks/lowra_saturation_exp.py \
      --mode "$MODE" "${EXTRA[@]}" \
      --model-dir "huggyllama/llama-${SIZE}" \
      --adapter-base "dummy-lora-${SIZE}-rank-16" --num-adapters 50 \
      --workload-trace "$OUT/traces/llama${SIZE}/trace.json" \
      --prefix-len 10000 --max-context-len 16384 \
      --num-token "$TOKENS" --batch-max-tokens 36864 --running-max-req-size 256 \
      --scheduler online --schedule-interval 1 --generation-mode trace \
      --request-rates 5 --duration 60 --cv 1 --seeds 42 --model-seed 42 \
      --warmup 0 --steady-end 1000000000 \
      --startup-timeout 1800 --request-timeout 10800 --drain-timeout 10800 \
      --port "$((9300 + GPU))" --nccl-port "$((34300 + GPU))" \
      --internal-port-start "$((49000 + GPU * 16))" \
      --output-dir "$OUT/llama${SIZE}/$MODE" --dummy
  done
  run_command "$PYTHON" benchmarks/compare_lowra_saturation.py \
    --full-dir "$OUT/llama${SIZE}/minikv-full_reconstruct" \
    --serial-dir "$OUT/llama${SIZE}/minikv-layer_serial" \
    --output "$OUT/llama${SIZE}/comparison.json"
)

if [ "$DRY_RUN" -eq 1 ]; then
  prepare_trace 7b
  prepare_trace 13b
  if [ "$NO_WAIT" -eq 1 ]; then
    printf 'Check GPUs 0 and 1 once; start immediately if free, otherwise exit.\n'
  else
    printf 'Wait until GPUs 0 and 1 have no compute processes for two consecutive checks.\n'
  fi
  printf 'CUDA_VISIBLE_DEVICES=0\n'
  run_model 0 7b 125000
  printf 'CUDA_VISIBLE_DEVICES=1\n'
  run_model 1 13b 60000
  exit 0
fi

mkdir -p "$OUT"
# One waiter owns both GPUs; reject duplicate launches of this launcher.
exec 9>benchmarks/results/online_lmsys_after_idle.lock
if ! flock -n 9; then
  printf 'Another online LMSYS waiter/suite is already running.\n' >&2
  exit 1
fi

# CPU-only preparation happens while the previous GPU jobs are still running.
for SIZE in 7b 13b; do
  if ! prepare_trace "$SIZE" > "$OUT/prepare_${SIZE}.log" 2>&1; then
    printf 'Trace preparation failed for %s; inspect %s/prepare_%s.log\n' "$SIZE" "$OUT" "$SIZE" >&2
    tail -n 40 "$OUT/prepare_${SIZE}.log" >&2
    exit 1
  fi
done
REQUIRED_IDLE_CHECKS=2
if [ "$NO_WAIT" -eq 1 ]; then
  REQUIRED_IDLE_CHECKS=1
  printf 'Traces prepared. Checking GPU availability once; no waiting.\n'
else
  printf 'Traces prepared. Waiting for BOTH GPUs to be free; checking every 2 seconds.\n'
fi
IDLE_CHECKS=0
while [ "$IDLE_CHECKS" -lt "$REQUIRED_IDLE_CHECKS" ]; do
  if ! GPU_PIDS="$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    printf 'Cannot query GPUs; refusing to start experiments.\n' >&2
    exit 1
  fi
  if [[ -z "${GPU_PIDS//[[:space:]]/}" ]]; then
    IDLE_CHECKS=$((IDLE_CHECKS + 1))
  else
    if [ "$NO_WAIT" -eq 1 ]; then
      printf 'GPUs 0 and/or 1 are busy; --no-wait refuses to start competing experiments.\n' >&2
      exit 1
    fi
    IDLE_CHECKS=0
  fi
  if [ "$IDLE_CHECKS" -lt "$REQUIRED_IDLE_CHECKS" ]; then
    sleep 2
  fi
done

printf 'Both GPUs are free. Starting 7B on GPU0 and 13B on GPU1.\n'
run_model 0 7b 125000 > "$OUT/llama7b.log" 2>&1 &
PID7=$!
run_model 1 13b 60000 > "$OUT/llama13b.log" 2>&1 &
PID13=$!
printf 'Worker PIDs: 7B=%s 13B=%s; output=%s\n' "$PID7" "$PID13" "$OUT"
STATUS=0
if ! wait "$PID7"; then
  printf '7B suite failed; inspect %s/llama7b.log\n' "$OUT" >&2
  STATUS=1
fi
if ! wait "$PID13"; then
  printf '13B suite failed; inspect %s/llama13b.log\n' "$OUT" >&2
  STATUS=1
fi
printf 'Both suites exited; status=%s\n' "$STATUS"
exit "$STATUS"
