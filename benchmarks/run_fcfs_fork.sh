#!/usr/bin/env bash
# GPU selection is explicit via CUDA_VISIBLE_DEVICES. Never kill other jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:?Usage: bash benchmarks/run_fcfs_fork.sh NEW_OUTPUT_DIR}"
if [[ -e "$OUT/pairs" || -e "$OUT/trace.json" ]]; then
  printf 'Output already exists; use a fresh directory.\n' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
TRACE_PYTHON="${SLORA_TRACE_PYTHON:-/home/qlh/anaconda3/envs/transformer/bin/python}"
PYTHON="${SLORA_PYTHON:-/home/qlh/anaconda3/envs/slora/bin/python}"
"$TRACE_PYTHON" benchmarks/prepare_lmsys_maf_trace.py \
  --tokenizer huggyllama/llama-13b --prefix-len 10000 --num-adapters 50 \
  --maf-skewness 1 --maf-minute-start 1 --maf-minute-end 1440 \
  --request-rate "${RATE:-3}" --duration "${DURATION:-180}" \
  --seed "${SEED:-42}" --data-seed "${DATA_SEED:-42}" --output "$OUT/trace.json"
exec "$PYTHON" -m benchmarks.fcfs_fork_exp --trace "$OUT/trace.json" --out "$OUT/pairs" \
  --pool 65000 --points "${POINTS:-30}" --spacing "${SPACING:-10}" --nccl-port "${NCCL_PORT:-34482}"
