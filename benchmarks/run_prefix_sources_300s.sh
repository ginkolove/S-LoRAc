#!/usr/bin/env bash
# Paired prefix-source sweep. Wait for both groups to drain before the next prefix.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash %s FRESH_OUTPUT_DIR [--start-now] [--dry-run]\n' "$0" >&2
  exit 2
fi
out=$(realpath -m "$1")
shift
start=--queue
extra=()
for opt in "$@"; do
  case "$opt" in
    --start-now) start=--start-now ;;
    --dry-run) extra=(--dry-run) ;;
    *) printf 'Unknown option: %s\n' "$opt" >&2; exit 2 ;;
  esac
done
cd "$(dirname "$0")/.."
for model in 7b 13b 30b; do
  case "$model" in
    7b) prefixes=(5000 8000 10000 12000 15000); reprefill=(--reprefill-num-token 120000) ;;
    13b) prefixes=(5000 8000 10000); reprefill=(--reprefill-num-token 62000) ;;
    30b) prefixes=(1000 2000 3000); reprefill=() ;;
  esac
  for prefix in "${prefixes[@]}"; do
    printf '\n=== Llama-%s, prefix=%s, 3 req/s for 300s ===\n' "$model" "$prefix"
    bash benchmarks/run_prefix_sources_after_idle.sh "$out/${model}_p${prefix}" \
      "$start" --model-size "$model" --prefix-len "$prefix" --duration 300 \
      --adapter-profile chameleon --fixed-rank64 --replay-timeout 86400 "${reprefill[@]}" "${extra[@]}"
  done
done
