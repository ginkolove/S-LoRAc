"""Profile KV capacity under an 80 GB H20 budget or measure a fixed pool's full replay."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess

import h20_runtime as h20
from h20_memory_guard import CAP_BYTES, MIB, ForeignGPUProcess, guarded_replay, physical_gpu, save_json
from h20_pair import replay_env
from lmsys_maf_trace import HF_PROFILE, hf_adapter_layout, hf_rank_counts, load_trace
from run_h20_adaptive_v2 import MODE as V2, build_command, stage_v2_source
from run_h20_adaptive_v3 import POLICY as V3, stage_v3_source
from benchmarks.h20_adaptive_v4 import POLICY as V4
from run_h20_minikv_actual import stage_source
from run_h20_rankmix_suite import ADAPTIVE, ADAPTIVE_POINTS, MODES, RANK_SPLIT_MODES, RECONSTRUCTION, point_commands
from slora.common.prefix_modes import CPU_READ, IMPRESS, add_impress_read_argument


GB = 10**9
DEFAULT_METHODS = (*RECONSTRUCTION, V3)
ALL_METHODS = (*MODES, IMPRESS, *RANK_SPLIT_MODES, ADAPTIVE, V2, V3, V4)
POINT_FILE = "rate3_seed42"
PROFILE_PHASES = {"model_startup", "cold_loras_prefill", "cold_loras_decode", "wide_batch_prefill", "wide_batch_decode"}


class PoolSizeOOM(RuntimeError):
    """The required pool size failed under the client quota."""


def option(command, name):
    return command[command.index(name) + 1]


def set_option(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend((name, str(value)))


def stage_v4_source(source):
    stage_v3_source(source)
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "h20_adaptive_v4.py", source / "slora/server/router")
    subprocess.run(["git", "apply", "--no-index", "-"], cwd=source,
                   input=(here / "h20_adaptive_v4.patch").read_text(), text=True, check=True,
                   env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(source.parent)))


def commands_for(dest, size, prefix, env, methods, *, v4_feedback_seconds=1., v4_hysteresis=.05,
                 impress_read_percent=None):
    baseline = dict(point_commands(dest, size, prefix, env))
    if any(mode in RANK_SPLIT_MODES for mode in methods):
        baseline.update(point_commands(dest, size, prefix, env, rank_split=True))
    if ADAPTIVE in methods:
        baseline.update(point_commands(dest, size, prefix, env, adaptive=True))
    commands = {}
    for mode in methods:
        if mode in (V2, V3, V4):
            options = argparse.Namespace(space_plan_window=8 if mode == V3 else 1, space_plan_interval=10,
                                         space_plan_hysteresis=0., space_plan_max_wait_ms=0., space_plan_max_step_ms=0.)
            cmd = build_command(dest, size, prefix, env, options)
            set_option(cmd, "--mode", mode)
            if mode == V4:
                set_option(cmd, "--output-dir", dest / mode)
                set_option(cmd, "--space-plan-feedback-seconds", v4_feedback_seconds)
                set_option(cmd, "--space-plan-hysteresis", v4_hysteresis)
        elif mode == IMPRESS:
            cmd = baseline[CPU_READ].copy()
            set_option(cmd, "--mode", mode)
            set_option(cmd, "--output-dir", dest / mode)
            if impress_read_percent is not None:
                set_option(cmd, "--impress-read-percent", impress_read_percent)
        else:
            cmd = baseline[mode].copy()
        # The original benchmark otherwise implicitly changes the prefill batch
        # limit along with the KV pool. Keep that scheduling knob fixed.
        if "--batch-max-tokens" not in cmd:
            set_option(cmd, "--batch-max-tokens", option(cmd, "--num-token"))
        commands[mode] = cmd
    return commands


def profile_spec(trace, digest):
    # Shape bounds only. No prompt text, arrival order or per-request outcomes
    # are given to the profiling worker or used by the formal scheduler.
    rows = trace["requests"]
    return dict(trace_sha256=digest, adapter_indices=sorted({r["adapter_index"] for r in rows}),
                query_lengths=sorted((len(r["input_ids"]) for r in rows), reverse=True),
                history_lengths=sorted((len(r["input_ids"]) + r["output_len"] - 1 for r in rows), reverse=True))


def pool_geometry(mode, cfg, trace):
    from slora.common.lowra_static import LowRAStaticPrefixStore

    ranks = trace[HF_PROFILE]["adapter_ranks"]
    prefix = trace["config"]["prefix_len"]
    layers = cfg["num_hidden_layers"]
    kv_width = cfg["hidden_size"] // cfg["num_attention_heads"] * cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    static = 0
    if mode.startswith("minikv-"):
        static = LowRAStaticPrefixStore.token_equivalent_static_size_from_config(prefix, ranks, cfg, 1)
    fixed = None
    if mode == "cpu-read-fixed1-r64":
        probabilities = trace[HF_PROFILE]["adapter_probabilities"]
        fixed = max((i for i, r in enumerate(ranks) if r == 64), key=probabilities.__getitem__)
    minimum = 0
    for req in trace["requests"]:
        i = req["adapter_index"]
        serial = mode in (RECONSTRUCTION[1], ADAPTIVE, V2, V3, V4) or ranks[i] <= RANK_SPLIT_MODES.get(mode, 0)
        prefix_slots = (prefix + layers - 1) // layers if serial else prefix
        cached = prefix_slots + (prefix if fixed is not None and i != fixed else 0)
        private = len(req["input_ids"]) + req["output_len"]
        minimum = max(minimum, static + cached + 4 * ranks[i] + private)
    return 4 * layers * kv_width, minimum  # FP16 keys and values, TP=1.


def smaller_pool(slots, minimum, slot_bytes):
    if slots <= minimum:
        raise PoolSizeOOM(f"CUDA OOM at {slots} pool slots, the floor for one request in this configuration")
    return max(minimum, slots - max((slots - minimum + 3) // 4, (GB + slot_bytes - 1) // slot_bytes))


def sampled_attempt_peak(directory, prefix, verified_peak):
    peak = verified_peak
    for report in directory.glob(f"{prefix}_*/memory.json"):
        peak = max(peak, json.loads(report.read_text())["sampled_peak_bytes"])
    return peak


def validate_replay(directory, mode, slots, trace, digest, command, memory):
    run = directory / "run"
    report = json.loads((directory / "mps_client.json").read_text())
    point = json.loads((run / (POINT_FILE + ".json")).read_text())
    result, runtime = point["result"], point["runtime"]
    expected = {r["req_id"]: r for r in trace["requests"]}
    dynamic_slots = runtime["persistent_total_slots"] + runtime["workspace_token_equivalent"]
    static_slots = runtime["static_token_equivalent"]
    if (not memory["verified_before_model_load"] or report["gpu_uuid"] != memory["gpu_uuid"]
            or report["quota_bytes"] != memory["client_quota_bytes"]
            or point["mode"] != mode or point["trace_sha256"] != digest or not result["valid"]
            or point["reset"]["status"] != "confirmed" or runtime["total_budget_tokens"] != slots
            or dynamic_slots + static_slots != slots
            or result["completed_requests"] != len(expected) or result["total_requests"] != len(expected)
            or not 0 < runtime["peak_reserved_bytes"] <= report["quota_bytes"]):
        raise ValueError(f"Incomplete replay or inconsistent memory accounting in {run}")
    counts = hf_rank_counts(trace["config"]["num_adapters"])
    bases = dict(zip(counts, [command[n + 1] for n, v in enumerate(command) if v == "--adapter-base"]))
    adapters = [f"{bases[rank]}-{replica}" for rank, replica in hf_adapter_layout(counts)]
    tokens = 0
    with (run / (POINT_FILE + "_requests.jsonl")).open() as stream:
        for line in stream:
            row = json.loads(line)
            req = expected.pop(row["req_id"], None)
            if (req is None or not row["success"] or not row["finished"] or row["scheduled_time"] != req["req_time"]
                    or row["adapter_index"] != req["adapter_index"] or row["adapter_dir"] != adapters[req["adapter_index"]]
                    or row["source_id"] != req["source_id"] or row["query_tokens"] != len(req["input_ids"])
                    or row["expected_output_tokens"] != req["output_len"] or row["output_tokens"] != req["output_len"]):
                raise ValueError(f"Request {row['req_id']} differs from frozen workload in {run}")
            tokens += row["output_tokens"]
    if expected:
        raise ValueError(f"Missing {len(expected)} completed requests in {run}")
    if "output_windows_10s" in result and sum(w["output_tokens"] for w in result["output_windows_10s"]) != tokens:
        raise ValueError(f"Windowed output token counts differ from completed requests in {run}")
    impress = {}
    if mode == IMPRESS and "--impress-read-percent" in command:
        percent = int(option(command, "--impress-read-percent"))
        count = trace["config"]["prefix_len"] * percent // 100
        cache = runtime["prefix_cache_stats"]
        if cache["impress_read_percent"] != percent or cache["prefix_h2d_tokens_per_load"] != count:
            raise ValueError(f"Impress transferred a different prefix fraction than requested in {run}")
        impress = dict(impress_read_percent=percent, prefix_h2d_tokens_per_load=count)
    return dict(pool_tokens=slots, dynamic_pool_slots=dynamic_slots, static_token_equivalents=static_slots,
                sampled_board_peak_bytes=memory["sampled_peak_bytes"],
                client_quota_bytes=report["quota_bytes"], torch_peak_reserved_bytes=runtime["peak_reserved_bytes"],
                torch_peak_allocated_bytes=runtime["peak_allocated_bytes"], output_tokens=tokens,
                peak_output_window_10s=result.get("peak_output_window_10s"),
                result_directory=str(directory), **impress, **{k: result[k] for k in (
                    "completed_requests", "benchmark_time", "completed_req_per_s_total", "output_tokens_per_s_total",
                    "ttft_mean", "ttft_p95", "latency_mean", "latency_p95")})


def run_attempt(dest, command, source, env, device, quota, slots, frozen, spec_path=None, reuse_manifest=None):
    trace_path, trace, digest = frozen
    cmd = command.copy()
    set_option(cmd, "--num-token", slots)
    set_option(cmd, "--workload-trace", trace_path)
    set_option(cmd, "--duration", trace["config"]["duration"])
    if "--running-max-req-size" not in cmd:
        set_option(cmd, "--running-max-req-size", len(trace["requests"]))
    run_env = dict(env, PYTHONPATH=os.pathsep.join(filter(None, [str(source), env.get("PYTHONPATH")])))
    run_env.pop("SLORA_BENCHMARK_SLOT_PROFILE", None)
    run_env.pop("SLORA_BENCHMARK_REUSE_MANIFEST", None)
    if spec_path is not None:
        run_env["SLORA_BENCHMARK_SLOT_PROFILE"] = str(spec_path)
    if reuse_manifest is not None:
        run_env["SLORA_BENCHMARK_REUSE_MANIFEST"] = str(reuse_manifest)
    # Only startup probes can restart after a transient ownership conflict.
    max_attempts = 3 if spec_path is not None else 1
    for attempt in range(max_attempts):
        directory = dest if not attempt else dest.with_name(f"{dest.name}_retry{attempt}")
        set_option(cmd, "--output-dir", directory / "run")
        print(f"{directory.parent.name} {directory.name}: {slots} pool slots", flush=True)
        try:
            mem = guarded_replay(cmd, source, run_env, device, quota, directory)
            break
        except ForeignGPUProcess:
            if attempt + 1 == max_attempts:
                raise
            interrupted = json.loads((directory / "memory.json").read_text())
            if interrupted["over_limit"] or interrupted["monitor_error"] or device.processes():
                raise
    dest = directory
    if mem["over_limit"] or mem["monitor_error"]:
        raise RuntimeError(f"Board cap or monitoring failure; inspect {dest / 'memory.json'}")
    if mem["cuda_oom"]:
        return None
    if mem["returncode"]:
        raise RuntimeError(f"Benchmark exited {mem['returncode']}; inspect {dest / 'benchmark.log'}")
    if spec_path is not None:
        profile = json.loads((dest / "forward_profile.json").read_text())
        if (not mem["verified_before_model_load"] or profile["mode"] != option(cmd, "--mode")
                or profile["trace_sha256"] != digest or profile["pool_tokens"] != slots
                or slots != profile["dynamic_pool_slots"] + profile["static_token_equivalents"]
                or not PROFILE_PHASES <= profile["phases"].keys()):
            raise ValueError(f"Incomplete startup profile in {dest}")
        phases = profile["phases"].values()
        reserved = max(p["peak_reserved_bytes"] for p in phases)
        other = max(p["other_client_bytes"] for p in phases)
        headroom = min(mem["cap_bytes"] - mem["sampled_peak_bytes"], quota - reserved - other)
        return dict(profile, sampled_board_peak_bytes=mem["sampled_peak_bytes"], client_quota_bytes=quota,
                    torch_peak_reserved_bytes=reserved, other_client_bytes=other, headroom_bytes=headroom,
                    directory=str(dest))
    return validate_replay(dest, option(cmd, "--mode"), slots, trace, digest, cmd, mem)


def read_verified_profile(path, mode, digest):
    profile = json.loads(path.read_text())
    if profile["mode"] != mode or profile["trace_sha256"] != digest:
        raise ValueError(f"Saved profile does not match the selected method and frozen trace: {path}")
    attempt = Path(profile["directory"])
    raw = json.loads((attempt / "forward_profile.json").read_text())
    memory = json.loads((attempt / "memory.json").read_text())
    if (not {"mode", "trace_sha256", "pool_tokens", "dynamic_pool_slots", "static_token_equivalents",
             "slot_bytes", "phases"} <= raw.keys() or not PROFILE_PHASES <= raw["phases"].keys()
            or any(profile[k] != value for k, value in raw.items()) or not memory["verified_before_model_load"]
            or memory["returncode"] != 0 or memory["error"] or memory["cuda_oom"] or memory["over_limit"]
            or memory["monitor_error"] or memory["cap_bytes"] != CAP_BYTES
            or memory["sampled_peak_bytes"] != profile["sampled_board_peak_bytes"]
            or CAP_BYTES - memory["sampled_peak_bytes"] < GB
            or memory["client_quota_bytes"] != profile["client_quota_bytes"]
            or profile["headroom_bytes"] < GB
            or not (attempt / "run/manifest.json").is_file()):
        raise ValueError(f"Missing or inconsistent evidence for a successful 80 GB profile: {path}")
    return dict(profile, reused_from=str(path.resolve()))


def formal_result(dest, measured, startup_peak=0):
    lifetime_peak = max(startup_peak, sampled_attempt_peak(dest, "formal", measured["sampled_board_peak_bytes"]))
    result = dict(measured, experiment_sampled_peak_bytes=lifetime_peak)
    if (dest / "startup_profile.json").is_file():
        result["startup_profile"] = str(dest / "startup_profile.json")
    return result


def measure_method(dest, command, source, env, device, quota, frozen, spec_path, cfg, profile_only=False,
                   reused=None, fixed_slots=None):
    _, trace, _ = frozen
    mode = option(command, "--mode")
    slot_bytes, minimum = pool_geometry(mode, cfg, trace)
    base_args = (command, source, env, device, quota)
    if reused is not None or fixed_slots is not None:
        slots = reused["pool_tokens"] if reused is not None else fixed_slots
        if slots < minimum:
            raise ValueError(f"{mode} needs at least {minimum} total slots for this trace, got {slots}")
        manifest = None
        if reused is not None:
            if (reused["slot_bytes"] != slot_bytes or reused["single_request_floor"] != minimum
                    or slots != reused["dynamic_pool_slots"] + reused["static_token_equivalents"]
                    or quota - reused["torch_peak_reserved_bytes"] - reused["other_client_bytes"] < GB):
                raise ValueError("Saved capacity is incompatible with this pool geometry or GPU quota")
            dest.mkdir(parents=True, exist_ok=False)
            save_json(dest / "startup_profile.json", reused)
            print(f"{mode}: reusing {slots} total slots from {reused['reused_from']}", flush=True)
            manifest = Path(reused["directory"]) / "run/manifest.json"
        measured = run_attempt(dest / f"formal_{slots}", *base_args, slots, frozen,
                               reuse_manifest=manifest)
        if measured is None:
            raise PoolSizeOOM(f"CUDA OOM in formal replay at {slots} total slots under this GPU's quota")
        return formal_result(dest, measured)

    def probe(slots):
        result = run_attempt(dest / f"profile_{slots}", *base_args, slots, frozen, spec_path)
        if result is not None and result["slot_bytes"] != slot_bytes:
            raise ValueError(f"Model and worker disagree about KV slot bytes for {mode}")
        return result

    floor_profile = probe(minimum)
    if floor_profile is None or floor_profile["headroom_bytes"] < GB:
        raise PoolSizeOOM(f"Startup profile cannot fit the single-request floor of {minimum} slots with 1 GB of margin")
    candidate = minimum + (floor_profile["headroom_bytes"] - GB) // slot_bytes
    while True:
        # At the larger pool the scheduler can admit a different shape. Verify that actual pool.
        profile = floor_profile if candidate == minimum else probe(candidate)
        if profile is None:
            candidate = smaller_pool(candidate, minimum, slot_bytes)
            continue
        headroom = profile["headroom_bytes"]
        if headroom < GB:
            # Include a small rounding allowance for the allocator's segment size.
            candidate = max(minimum, candidate - (GB - headroom + 64 * MIB + slot_bytes - 1) // slot_bytes)
            continue
        report = dict(profile, single_request_floor=minimum, bootstrap_directory=floor_profile["directory"],
                      target_margin_bytes=GB, profile_only=profile_only,
                      startup_sampled_peak_bytes=sampled_attempt_peak(dest, "profile", profile["sampled_board_peak_bytes"]))
        save_json(dest / "startup_profile.json", report)
        print(f"{mode}: verified {candidate} total slots ({profile['dynamic_pool_slots']} dynamic), "
              f"profile peak {profile['sampled_board_peak_bytes'] / GB:.3f} GB", flush=True)
        if profile_only:
            return report
        measured = run_attempt(dest / f"formal_{candidate}", *base_args, candidate, frozen)
        if measured is not None:
            return formal_result(dest, measured, report["startup_sampled_peak_bytes"])
        candidate = smaller_pool(candidate, minimum, slot_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="parent for new per-point results")
    parser.add_argument("--gpu", type=int, choices=range(8), required=True, help="exclusive physical H20 index")
    parser.add_argument("--point", choices=ADAPTIVE_POINTS, default="13b:10000", metavar="MODEL:PREFIX")
    parser.add_argument("--methods", nargs="+", choices=("all", *ALL_METHODS), default=DEFAULT_METHODS)
    add_impress_read_argument(parser)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profile-only", action="store_true", help="measure and verify slot counts without a timed replay")
    selection.add_argument("--reuse-profile", type=Path, metavar="STARTUP_PROFILE_JSON",
                           help="use one method's verified capacity and go directly to guarded formal replay")
    selection.add_argument("--num-token", type=int, metavar="SLOTS",
                           help="run one formal replay at this fixed total KV budget")
    parser.add_argument("--device-capacity", action="store_true",
                        help="use the physical H20 memory capacity as the board budget for a fixed-slot replay")
    parser.add_argument("--outside-reserve-gb", type=int, default=2,
                        help="extra decimal GB above idle usage held outside the MPS client quota (default 2)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--v4-feedback-seconds", type=float, default=1.,
                        help="wall-clock length of each V4 output throughput window")
    parser.add_argument("--v4-hysteresis", type=float, default=.05,
                        help="fractional throughput decline tolerated by V4")
    args = parser.parse_args()
    if (not math.isfinite(args.v4_feedback_seconds) or args.v4_feedback_seconds <= 0
            or not 0 <= args.v4_hysteresis < 1):
        parser.error("V4 requires a positive finite window and a hysteresis in [0, 1)")
    if "all" in args.methods:
        if len(args.methods) != 1:
            parser.error("use --methods all on its own")
        args.methods = ALL_METHODS
    if args.impress_read_percent is not None and IMPRESS not in args.methods:
        parser.error("--impress-read-percent requires impress in --methods")
    if len(set(args.methods)) != len(args.methods):
        parser.error("select distinct methods")
    if args.device_capacity and args.num_token is None:
        parser.error("--device-capacity requires --num-token")
    if args.outside_reserve_gb < 1 or (not args.device_capacity and args.outside_reserve_gb >= CAP_BYTES // GB):
        parser.error("select a positive reserve below the board budget")
    if args.reuse_profile and len(args.methods) != 1:
        parser.error("--reuse-profile requires exactly one method")
    if args.num_token is not None and args.num_token <= 0:
        parser.error("--num-token must be a positive total slot count")
    size, prefix = ADAPTIVE_POINTS[args.point]
    dest = args.output.resolve() / f"{size}_p{prefix}"
    shim = str(Path(h20.__file__).resolve())
    env = replay_env(dict(os.environ, CUDA_VISIBLE_DEVICES="", SLORA_PYTHON=shim, SLORA_TRACE_PYTHON=shim,
                          SLORA_H20_PHYSICAL_GPU=str(args.gpu)))
    commands = commands_for(dest, size, prefix, env, args.methods,
                            v4_feedback_seconds=args.v4_feedback_seconds, v4_hysteresis=args.v4_hysteresis,
                            impress_read_percent=args.impress_read_percent)
    trace_path = Path(option(next(iter(commands.values())), "--workload-trace"))
    trace, digest = load_trace(trace_path)
    reused = read_verified_profile(args.reuse_profile, args.methods[0], digest) if args.reuse_profile else None
    if args.dry_run:
        budget = "physical H20 capacity" if args.device_capacity else f"{CAP_BYTES} bytes"
        print(f"GPU {args.gpu}, cap {budget}, private MPS, outside reserve {args.outside_reserve_gb} GB + idle")
        print(f"{len(trace['requests'])} formal requests, SHA-256 {digest}, output {dest}")
        for mode, cmd in commands.items():
            plan = f"reuse {reused['pool_tokens']} verified slots" if reused else "startup forwards -> verify pool"
            if args.num_token is not None:
                plan = f"fixed {args.num_token} total slots"
            if mode == IMPRESS and args.impress_read_percent is not None:
                plan += f", read {args.impress_read_percent}% of prefix positions"
            print(f"{mode}: {plan}" + ("" if args.profile_only else " -> full replay")
                  + f"; batch_max_tokens={option(cmd, '--batch-max-tokens')}")
        return
    if dest.exists():
        parser.error(f"Point already exists: {dest}. Choose a new output parent")
    h20.prepare([size])
    lock_path = h20.ROOT / "benchmarks/results" / f"lmsys_preemption_gpu{args.gpu}.lock"
    with lock_path.open("a") as lock, physical_gpu(args.gpu, use_device_capacity=args.device_capacity) as device:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if device.processes():
            raise RuntimeError(f"GPU {args.gpu} is in use; select an idle GPU")
        memory = device.memory_info()
        available = min(device.cap_bytes - memory.used, memory.free)
        quota = ((available - args.outside_reserve_gb * GB) // MIB) * MIB
        if quota <= 0:
            raise ValueError("No CUDA client budget remains after reserving device memory")
        dest.mkdir(parents=True, exist_ok=False)
        config = dict(cap_bytes=device.cap_bytes, use_device_capacity=args.device_capacity,
                      gpu_uuid=device.uuid, gpu_index=args.gpu, client_quota_bytes=quota,
                      idle_used_bytes=memory.used, outside_reserve_bytes=args.outside_reserve_gb * GB,
                      board_evidence="NVML sampled; MPS enforces the single CUDA client's allocation quota",
                      trace_sha256=digest, methods=list(commands), profile_only=args.profile_only)
        if reused:
            config["reused_startup_profile"] = reused["reused_from"]
        if args.num_token is not None:
            config["fixed_pool_tokens"] = args.num_token
        if args.impress_read_percent is not None:
            config["impress_read_percent"] = args.impress_read_percent
        save_json(dest / "experiment.json", config)
        profile_path = dest / "probe_shapes.json"
        if not reused and args.num_token is None:
            save_json(profile_path, profile_spec(trace, digest))
        cfg = json.loads((h20.model_dir(size) / "config.json").read_text())
        results = {}
        failed = {}
        for mode, command in commands.items():
            if mode in (V2, V3, V4):
                source = dest / f"source_{mode}"
                {V2: stage_v2_source, V3: stage_v3_source, V4: stage_v4_source}[mode](source)
            elif mode == ADAPTIVE:
                source = dest / "source_adaptive"
                stage_source(source, 16, adaptive=True)
            elif mode in RECONSTRUCTION or mode in RANK_SPLIT_MODES:
                source = dest / ("source_rank_split" if mode in RANK_SPLIT_MODES else "source_actual16")
                if not source.exists():
                    stage_source(source, 16, rank_split=mode in RANK_SPLIT_MODES)
            else:
                source = h20.ROOT
            try:
                measured = measure_method(dest / mode, command, source, env, device, quota,
                                          (trace_path, trace, digest), profile_path, cfg, args.profile_only, reused,
                                          args.num_token)
            except (PoolSizeOOM, ForeignGPUProcess) as exc:
                failed[mode] = str(exc)
                save_json(dest / mode / "failure.json", dict(reason=str(exc)))
                print(f"{mode}: {exc}", flush=True)
                if isinstance(exc, ForeignGPUProcess):
                    break
                continue
            if not args.profile_only:
                save_json(dest / mode / "result.json", measured)
                peak = measured["peak_output_window_10s"]
                if peak is not None:
                    print(f"{mode} peak 10 s output throughput {peak['output_tokens_per_s']:.2f} tok/s "
                          f"from {peak['start']:g} to {peak['end']:g} s", flush=True)
            results[mode] = measured
        if failed:
            save_json(dest / "incomplete.json", dict(config=config, completed=results, failed=failed))
            raise SystemExit(f"Experiment incomplete. See {dest / 'incomplete.json'}")
        summary_path = dest / ("profiles.json" if args.profile_only else "comparison.json")
        save_json(summary_path, dict(config=config, methods=results))
        print(summary_path, flush=True)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    # NVML sees physical devices independently. Only the guarded model worker needs CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
