"""Give only the TP=1 benchmark router access to its quota-limited GPU."""

import ctypes
from contextlib import contextmanager
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def verify_mps_client():
    # Check the driver before constructing the model, including on a machine
    # where CUDA would silently run without the requested MPS daemon.
    import torch

    driver = ctypes.CDLL("libcuda.so.1")
    enabled = ctypes.c_int()
    if (driver.cuInit(0) or driver.cuDeviceGetAttribute(ctypes.byref(enabled), 133, 0)
            or enabled.value != 1):  # CU_DEVICE_ATTRIBUTE_MPS_ENABLED
        raise RuntimeError("The benchmark router is not attached to CUDA MPS")
    uuid = os.environ["SLORA_BENCHMARK_MPS_GPU"]
    quota = int(os.environ["SLORA_BENCHMARK_MPS_BYTES"])
    if (torch.cuda.device_count() != 1 or os.environ["CUDA_VISIBLE_DEVICES"] != uuid
            or os.environ["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] != f"{uuid}={quota // 2**20}M"):
        raise RuntimeError("Unexpected GPU visibility or MPS quota in benchmark router")
    torch.empty(1, device="cuda:0")  # Create this process's primary context.
    free, total = torch.cuda.mem_get_info()
    if not 0 < free <= quota < total:
        raise RuntimeError(f"MPS client free/total memory does not match the quota: {free}/{total}, {quota}")
    report = Path(os.environ["SLORA_BENCHMARK_MPS_REPORT"])
    pending = report.with_suffix(".tmp")
    pending.write_text(json.dumps(dict(pid=os.getpid(), gpu_uuid=uuid, quota_bytes=quota,
                                       initial_cuda_free_bytes=free, cuda_total_bytes=total)) + "\n")
    pending.replace(report)
    # Hold the CUDA context steady while the independent NVML observer
    # measures board memory already committed outside the client's free quota.
    deadline = time.monotonic() + 30
    while not report.with_name("startup_budget.json").is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("Board memory verification timed out before model loading")
        time.sleep(.02)


@contextmanager
def register_mps_router():
    """Publish the worker's single device before cuInit enumerates the boards."""
    gpu = os.environ["SLORA_BENCHMARK_MPS_GPU"]
    if not gpu or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError("Guarded worker has unexpected CUDA visibility")
    marker = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"]) / "slora-router.json"
    with marker.open("x") as stream:
        json.dump(dict(gpu_uuid=gpu, pid=os.getpid(), pid_namespace=os.readlink("/proc/self/ns/pid")), stream)
        stream.flush()
        # Process death releases the lock, including during a failed startup.
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _run_mps_router(target, args):
    with register_mps_router():
        try:
            verify_mps_client()
        except Exception as exc:
            args[-1].send(f"MPS verification failed: {exc}")
            raise
        return target(*args)


def spawn_router(target, args):
    """The API and detokenizer stay CPU-only in a guarded replay."""
    gpu = os.environ.get("SLORA_BENCHMARK_MPS_GPU")
    if gpu is None:
        process = mp.Process(target=target, args=args)
        process.start()
        return process
    if args[0].tp != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or mp.get_start_method() != "spawn":
        raise ValueError("Guarded replay requires TP=1, spawn, and a GPU-hidden API process")
    process = mp.Process(target=_run_mps_router, args=(target, args))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        process.start()
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pidfile = Path(os.environ["SLORA_BENCHMARK_MPS_REPORT"]).with_name("router.pid")
    pending = pidfile.with_suffix(".tmp")
    pending.write_text(str(process.pid))
    pending.replace(pidfile)
    return process
