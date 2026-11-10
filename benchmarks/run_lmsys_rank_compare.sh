#!/usr/bin/env bash
# One GPU: full reconstruction followed by layer-serial, on the same frozen trace.
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 7 ]; then
  printf 'Usage: bash %s GPU RANK TRACE_JSON OUTPUT_DIR [MODEL_SIZE=7b] [NUM_TOKEN] [DURATION=10]\n' "$0" >&2
  exit 2
fi
GPU="$1"
RANK="$2"
TRACE="$(realpath "$3")"
OUT="$(realpath -m "$4")"
MODEL_SIZE="${5:-7b}"
case "$MODEL_SIZE" in
  7b) DEFAULT_TOKENS=125000 ;;
  13b) DEFAULT_TOKENS=60000 ;;
  *) printf 'Supported MODEL_SIZE values are 7b and 13b.\n' >&2; exit 2 ;;
esac
NUM_TOKEN="${6:-$DEFAULT_TOKENS}"
DURATION="${7:-10}"
if ! [[ "$GPU" =~ ^[0-9]+$ && "$RANK" =~ ^[0-9]+$ ]] || [ "$RANK" -le 0 ]; then
  printf 'GPU must be nonnegative and RANK must be positive.\n' >&2
  exit 2
fi
if ! [[ "$NUM_TOKEN" =~ ^[0-9]+$ ]] || [ "$NUM_TOKEN" -le 0 ]; then
  printf 'NUM_TOKEN must be positive.\n' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="${SLORA_PYTHON:-/home/qlh/anaconda3/envs/slora/bin/python}"

for MODE in minikv-full_reconstruct minikv-layer_serial; do
  EXTRA=()
  if [ "$MODE" = minikv-layer_serial ]; then
    EXTRA=(--reconstruction-kernel bounded --reconstruction-workers 1296)
  fi
  "$PYTHON" -u benchmarks/lowra_saturation_exp.py \
    --mode "$MODE" "${EXTRA[@]}" \
    --model-dir "huggyllama/llama-${MODEL_SIZE}" \
    --adapter-base "dummy-lora-${MODEL_SIZE}-rank-${RANK}" --num-adapters 50 \
    --workload-trace "$TRACE" --prefix-len 10000 --max-context-len 16384 \
    --num-token "$NUM_TOKEN" --batch-max-tokens 36864 --running-max-req-size 256 \
    --request-rates 3 --duration "$DURATION" --cv 1 --seeds 42 --model-seed 42 \
    --warmup 0 --steady-end 1000000000 \
    --startup-timeout 1800 --request-timeout 10800 --drain-timeout 10800 \
    --port "$((9200 + GPU))" --nccl-port "$((34200 + GPU))" \
    --internal-port-start "$((48000 + GPU * 16))" \
    --output-dir "$OUT/$MODE" --dummy
done

"$PYTHON" benchmarks/compare_lowra_saturation.py \
  --full-dir "$OUT/minikv-full_reconstruct" \
  --serial-dir "$OUT/minikv-layer_serial" --output "$OUT/comparison.json"
