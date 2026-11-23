#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
dry_run=false
watch_peaks=false
device_capacity=false
common_slots=
gpu_slots=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --watch) watch_peaks=true; shift ;;
        --device-capacity) device_capacity=true; shift ;;
        --num-token)
            if [[ ! ${2:-} =~ ^[1-9][0-9]*$ ]]; then
                echo "--num-token requires a positive total slot count" >&2
                exit 2
            fi
            common_slots=$2
            shift 2
            ;;
        --gpu-slots)
            if (( $# < 7 )); then
                echo "--gpu-slots requires six positive counts in GPU 0 through 5 order" >&2
                exit 2
            fi
            gpu_slots=("${@:2:6}")
            for slots in "${gpu_slots[@]}"; do
                if [[ ! $slots =~ ^[1-9][0-9]*$ ]]; then
                    echo "Invalid --gpu-slots count: $slots" >&2
                    exit 2
                fi
            done
            shift 7
            ;;
        *) echo "Usage: $0 [--num-token SLOTS | --gpu-slots SLOT0 SLOT1 SLOT2 SLOT3 SLOT4 SLOT5] [--device-capacity] [--watch] [--dry-run]" >&2; exit 2 ;;
    esac
done
if [[ -n $common_slots ]] && (( ${#gpu_slots[@]} )); then
    echo "Use only one of --num-token and --gpu-slots" >&2
    exit 2
fi

if [[ "$dry_run" == true ]]; then
    run_dir="$PWD/benchmarks/results/h20_13b10k_formal_preview"
else
    run_dir=$(mktemp -d "$PWD/benchmarks/results/h20_13b10k_formal_XXXXXX")
    printf 'Results: %s\n' "$run_dir"
fi

launch() {
    local gpu=$1 method=$2 slots=${common_slots:-$3}
    if (( ${#gpu_slots[@]} )); then
        slots=${gpu_slots[gpu]}
    fi
    local cmd=(/opt/conda/bin/python -u benchmarks/run_h20_memory_cap.py
        "$run_dir/gpu$gpu" --gpu "$gpu" --point 13b:10000 --methods "$method" --num-token "$slots")
    if [[ "$device_capacity" == true ]]; then
        cmd+=(--device-capacity)
    fi
    if [[ "$dry_run" == true ]]; then
        "${cmd[@]}" --dry-run
    else
        local log="$run_dir/gpu${gpu}_${method}.log"
        # Keep the foreground peak viewer's terminal signals away from the replay.
        nohup setsid "${cmd[@]}" </dev/null >"$log" 2>&1 &
        printf 'GPU %s  %s  slots=%s  PID=%s  %s\n' "$gpu" "$method" "$slots" "$!" "$log"
    fi
}

launch 0 active-reprefill         45532
launch 1 cpu-read                 45532
launch 2 cpu-read-fixed1-r64      45532
launch 3 minikv-full_reconstruct  49970
launch 4 minikv-layer_serial      47692
launch 5 minikv-adaptive-v3       47662

if [[ "$dry_run" == false ]]; then
    report=(/opt/conda/bin/python "$PWD/benchmarks/h20_peak_report.py" "$run_dir")
    printf 'Peak report:'
    printf ' %q' "${report[@]}"
    printf '\n'
    if [[ "$watch_peaks" == true ]]; then
        exec watch -n 2 -x "${report[@]}"
    fi
fi
