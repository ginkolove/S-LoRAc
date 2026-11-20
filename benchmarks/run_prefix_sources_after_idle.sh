#!/usr/bin/env bash
# Start the five Llama-13B prefix comparisons together on the first idle poll.
set -euo pipefail

usage() {
  printf 'Usage: bash %s OUTPUT_DIR [--queue|--start-now] [--model-size SIZE] [--prefix-len N] [--duration SEC] [--adapter-profile chameleon|maf] [--fixed-rank64] [--reprefill-num-token N] [--replay-timeout SEC] [--dry-run]\n' "$0" >&2
  exit 2
}
[[ $# -ge 1 ]] || usage
OUT=$(realpath -m "$1")
shift
dry_run=0
queue=0
start_now=0
profile=chameleon
size=13b
prefix_len=10000
duration=80
fixed=()
fixed_mode=cpu-read-fixed2
timeout_args=()
reprefill=()
while (( $# )); do
  case "$1" in
    --queue) queue=1; shift ;;
    --start-now) start_now=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --fixed-rank64) fixed=("$1"); fixed_mode=cpu-read-fixed1-r64; shift ;;
    --model-size)
      case "${2:-}" in 7b|13b|30b) size="$2"; shift 2 ;; *) usage ;; esac ;;
    --duration|--prefix-len|--replay-timeout)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || usage
      case "$1" in
        --duration) duration="$2" ;;
        --prefix-len) prefix_len="$2" ;;
        --replay-timeout) timeout_args=("$1" "$2") ;;
      esac
      shift 2 ;;
    --adapter-profile)
      case "${2:-}" in chameleon|maf) profile="$2"; shift 2 ;; *) usage ;; esac ;;
    --reprefill-num-token)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || usage
      reprefill=("$1" "$2"); shift 2 ;;
    *) usage ;;
  esac
done
(( ! (queue && start_now) )) || usage
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
common=(--model-size "$size" --prefix-len "$prefix_len" --num-adapters 50 --request-rate 3
        --duration "$duration" --schedule-interval 1 --adapter-profile "$profile" "${fixed[@]}" "${timeout_args[@]}")
suite=(bash benchmarks/run_lmsys_preemption.sh "$OUT")
gpu0=("${suite[@]}" --gpu 0 --prefix-source-group compute "${common[@]}" "${reprefill[@]}")
gpu1=("${suite[@]}" --gpu 1 --prefix-source-group cpu --cpu-prefix-pool-gb 100 "${common[@]}")
compare=("${SLORA_PYTHON:-/home/qlh/anaconda3/envs/slora/bin/python}" benchmarks/compare_prefix_sources.py "$OUT" --fixed-mode "$fixed_mode")
if (( dry_run )); then
  "${gpu0[@]}" --dry-run
  "${gpu1[@]}" --dry-run
  printf 'After both succeed: '; printf '%q ' "${compare[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$OUT" benchmarks/results
if compgen -G "$OUT/*/manifest.json" >/dev/null || [[ -e "$OUT/trace.json" ]]; then
  printf 'Existing experiment in %s; choose a fresh directory.\n' "$OUT" >&2
  exit 1
fi
if (( queue )); then
  exec 6>>benchmarks/results/prefix_sources_queued.lock
  if ! flock -n 6; then
    printf 'A queued comparison already exists; not scheduling a duplicate.\n' >&2
    exit 1
  fi
fi
exec 7>>benchmarks/results/prefix_sources_after_idle.lock
if ! flock -n 7; then
  if (( ! queue )); then
    printf 'A prefix comparison waiter or suite is already running.\n' >&2
    exit 1
  fi
  printf '%s\n' "$$" > "$OUT/waiter.pid"
  printf 'Queued behind the previous comparison; waiting for its lock. Output: %s\n' "$OUT"
  flock 7
  if compgen -G "$OUT/*/manifest.json" >/dev/null || [[ -e "$OUT/trace.json" ]]; then
    printf 'Output already populated while queued: %s; refusing to overwrite.\n' "$OUT" >&2
    exit 1
  fi
fi
printf '%s\n' "$$" > "$OUT/waiter.pid"
if (( ! start_now )); then
  printf 'Waiting for both GPUs 0 and 1; polling now and every 2 seconds. Output: %s\n' "$OUT"
fi
previous=initial
while :; do
  if ! pids=$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits); then
    printf 'GPU query failed; not starting either suite.\n' >&2
    exit 1
  fi
  state=${pids//$'\n'/ }
  if [[ -z ${state//[[:space:]]/} ]]; then
    # A prior suite can be between modes and still own its per-GPU lock.
    if flock -n benchmarks/results/lmsys_preemption_gpu0.lock true &&
       flock -n benchmarks/results/lmsys_preemption_gpu1.lock true; then
      break
    fi
    state='an earlier suite still owns a GPU lock'
  fi
  if (( start_now )); then printf 'Cannot start immediately: %s\n' "$state" >&2; exit 1; fi
  if [[ "$state" != "$previous" ]]; then printf 'Busy: %s\n' "$state"; previous=$state; fi
  sleep 2
done

# The child launchers also hold their own locks and recheck the cards before replay.
printf 'Both GPUs idle; starting compute on GPU0 and CPU-read on GPU1.\n'
"${gpu0[@]}" > "$OUT/gpu0.log" 2>&1 &
pid0=$!
"${gpu1[@]}" > "$OUT/gpu1.log" 2>&1 &
pid1=$!
printf 'Suite PIDs: GPU0=%s GPU1=%s\n' "$pid0" "$pid1"
rc0=0; rc1=0
wait "$pid0" || rc0=$?
wait "$pid1" || rc1=$?
if (( rc0 || rc1 )); then
  printf 'Suite exits: GPU0=%s GPU1=%s. See gpu0.log/gpu1.log; skipping comparison.\n' "$rc0" "$rc1" >&2
  exit 1
fi
"${compare[@]}"
