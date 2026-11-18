#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || ($# - 1) % 2 != 0 )); then
    printf 'Usage: bash %s OUTPUT_DIR {max|actual} {16|32} [MODE BLOCK_R ...]\n' "$0" >&2
    exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
OUTPUT_DIR=$1
shift
PYTHON=${SLORA_PYTHON:-/home/qlh/anaconda3/envs/slora/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

while (( $# )); do
    MODE=$1
    BLOCK_R=$2
    shift 2
    case "$MODE:$BLOCK_R" in
        max:16|max:32|actual:16|actual:32) ;;
        *) printf 'Invalid reconstruction configuration: %s %s\n' "$MODE" "$BLOCK_R" >&2; exit 2 ;;
    esac
    printf '\nStarting rank_loop=%s block_r=%s GPU=%s\n' "$MODE" "$BLOCK_R" "${CUDA_VISIBLE_DEVICES:-default}"
    "$PYTHON" -u try/profile_minikv_rank_13b.py \
        --prefix-len 10000 --pool-adapters 50 \
        --ranks 8 16 32 64 --unique-loras 1 2 4 8 16 \
        --rank-loop "$MODE" --block-r "$BLOCK_R" \
        --workers 1296 --warmup 10 --iterations 50 --repeats 3 \
        --seed 42 --output-dir "$OUTPUT_DIR/${MODE}_br${BLOCK_R}"
done
