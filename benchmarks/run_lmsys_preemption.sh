#!/usr/bin/env bash
# Paired runs with a frozen LMSYS/MAF trace and preemption diagnostics.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: bash %s OUTPUT_DIR [--model-size 7b|13b|30b] [--prefix-len TOKENS] [--gpu 0|1] [--request-rate REQ_PER_S] [--duration SECONDS] [--dry-run]\n' "$0" >&2
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
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --model-size|--gpu|--prefix-len|--request-rate|--duration)
      if [ "$#" -lt 2 ]; then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      case "$1" in
        --model-size) SIZE="$2" ;;
        --gpu) GPU="$2" ;;
        --prefix-len) PREFIX_LEN="$2" ;;
        --request-rate) REQUEST_RATE="$2" ;;
        --duration) DURATION="$2" ;;
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
case "$SIZE" in
  7b) TOKENS=125000 ;;
  13b) TOKENS=60000 ;;
  # The old 9216-token synthetic-run pool cannot fit some LMSYS requests even alone.
  30b) TOKENS=10240 ;;
  *) printf 'Supported model sizes: 7b, 13b, 30b\n' >&2; exit 2 ;;
esac
if [ -z "$PREFIX_LEN" ]; then
  if [ "$SIZE" = 30b ]; then PREFIX_LEN=4000; else PREFIX_LEN=10000; fi
fi
if [[ ! "$PREFIX_LEN" =~ ^[1-9][0-9]{0,4}$ ]] || [ "$PREFIX_LEN" -ge 16384 ]; then
  printf 'Prefix length must be an integer between 1 and 16383.\n' >&2
  exit 2
fi
case "$GPU" in
  0|1) ;;
  *) printf 'Supported GPU indices: 0, 1\n' >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
  for MODE in minikv-full_reconstruct minikv-layer_serial; do
    if [ -e "$OUT/$MODE/manifest.json" ]; then
      printf 'Existing run at %s; choose a fresh output directory.\n' "$OUT/$MODE" >&2
      exit 1
    fi
  done
  check_gpu
fi
printf 'CUDA_VISIBLE_DEVICES=%s; Llama-%s, prefix=%s tokens, pool=%s tokens, rate=%s req/s, duration=%ss; full reconstruction, then layer serial.\n' \
  "$GPU" "$SIZE" "$PREFIX_LEN" "$TOKENS" "$REQUEST_RATE" "$DURATION"
run_command "$TRACE_PYTHON" -u benchmarks/prepare_lmsys_maf_trace.py \
  --dataset /data/hub/LMSYS-33k \
  --maf-file /data/hub/MAF/azurefunctions2019/invocations_per_function_md.anon.d01.csv \
  --maf-skewness 1 --maf-minute-start 1 --maf-minute-end 1440 \
  --tokenizer "huggyllama/llama-${SIZE}" --num-adapters 50 \
  --prefix-len "$PREFIX_LEN" --max-context-len 16384 \
  --request-rate "$REQUEST_RATE" --duration "$DURATION" --seed 42 --data-seed 42 \
  --output "$OUT/trace.json"

RELEASE_WAIT=0
for MODE in minikv-full_reconstruct minikv-layer_serial; do
  if [ "$DRY_RUN" -eq 0 ]; then check_gpu "$RELEASE_WAIT"; fi
  EXTRA=()
  if [ "$MODE" = minikv-layer_serial ]; then
    EXTRA=(--reconstruction-kernel bounded --reconstruction-workers 1296)
  fi
  printf 'Starting %s on GPU%s; output=%s\n' "$MODE" "$GPU" "$OUT/$MODE"
  run_command "$PYTHON" -u benchmarks/lowra_saturation_exp.py \
    --mode "$MODE" "${EXTRA[@]}" \
    --model-dir "huggyllama/llama-${SIZE}" \
    --adapter-base "dummy-lora-${SIZE}-rank-16" --num-adapters 50 \
    --workload-trace "$OUT/trace.json" \
    --prefix-len "$PREFIX_LEN" --max-context-len 16384 \
    --num-token "$TOKENS" --batch-max-tokens 36864 --running-max-req-size 256 \
    --scheduler online --schedule-interval 1 --generation-mode trace \
    --request-rates "$REQUEST_RATE" --duration "$DURATION" --cv 1 --seeds 42 --model-seed 42 \
    --warmup 0 --steady-end 1000000000 \
    --startup-timeout 1800 --request-timeout 10800 --drain-timeout 10800 \
    --port "$((9400 + GPU))" --nccl-port "$((34400 + GPU))" --internal-port-start "$((50000 + GPU * 16))" \
    --output-dir "$OUT/$MODE" --dummy
  # Waiting for the server parent does not guarantee its CUDA contexts are gone.
  RELEASE_WAIT=60
done

run_command "$PYTHON" benchmarks/compare_lowra_saturation.py \
  --full-dir "$OUT/minikv-full_reconstruct" \
  --serial-dir "$OUT/minikv-layer_serial" \
  --output "$OUT/comparison.json"
printf 'Finished. Preemption diagnostics are in each rate*_seed42.json (preemption), with events in runtime.preemptions.jsonl.\n'
