"""Run a TP=1 benchmark under a CUDA MPS quota and sample the physical GPU."""

from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time


CAP_BYTES = 80_000_000_000
MIB = 2**20
CUDA_OOM = re.compile(rb"cuda (?:error: )?out of memory|cuda_error_out_of_memory|cublas_status_alloc_failed", re.I)


class ForeignGPUProcess(RuntimeError):
    """Exclusive GPU ownership was lost during an attempt."""


def save_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


@contextmanager
def physical_gpu(index, *, use_device_capacity=False):
    import pynvml as nv

    nv.nvmlInit()
    try:
        yield PhysicalGPU(nv, index, use_device_capacity=use_device_capacity)
    finally:
        nv.nvmlShutdown()


class PhysicalGPU:
    def __init__(self, nv, index, *, use_device_capacity=False):
        self.nv = nv
        self.handle = nv.nvmlDeviceGetHandleByIndex(index)
        self.uuid = nv.nvmlDeviceGetUUID(self.handle)
        if isinstance(self.uuid, bytes):
            self.uuid = self.uuid.decode()
        if "H20" not in str(nv.nvmlDeviceGetName(self.handle)):
            raise ValueError(f"Expected an H20 at NVML index {index}")
        if nv.nvmlDeviceGetMigMode(self.handle)[0]:
            raise ValueError("Select an exclusive, unpartitioned H20 for whole-device accounting")
        self.cap_bytes = self.memory_info().total if use_device_capacity else CAP_BYTES

    def memory_info(self):
        return self.nv.nvmlDeviceGetMemoryInfo(self.handle, self.nv.nvmlMemory_v2)

    def used(self):
        return self.memory_info().used

    def processes(self):
        readers = (self.nv.nvmlDeviceGetComputeRunningProcesses, self.nv.nvmlDeviceGetGraphicsRunningProcesses,
                   self.nv.nvmlDeviceGetMPSComputeRunningProcesses)
        rows = [p for reader in readers for p in reader(self.handle)]
        # The same PID may have entries in multiple NVML lists. Only a known
        # worker on another board with *all* entries exactly zero may be omitted.
        pids = {p.pid for p in rows}
        zero_only = pids - {p.pid for p in rows if p.usedGpuMemory != 0}
        if zero_only:
            pids -= zero_only & other_board_routers(self.uuid)
        return pids


class DeviceSampler:
    def __init__(self, device, directory, interval):
        self.device, self.directory, self.interval = device, directory, interval
        self.start = time.monotonic()
        self.peak = self.count = self.max_gap = 0
        self.previous = self.peak_time = self.last = self.error = None
        self.over_limit = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run)

    def _sample(self):
        now = time.monotonic() - self.start
        used = self.device.used()
        if self.previous is not None:
            self.max_gap = max(self.max_gap, now - self.previous)
        if self.peak_time is None or used > self.peak:
            self.peak, self.peak_time = used, now
        self.over_limit |= used > self.device.cap_bytes
        if self.count == 0 or used != self.last or self.stop.is_set():
            self.writer.writerow((round(now, 6), used))
        self.previous, self.last = now, used
        self.count += 1

    def _run(self):
        try:
            while not self.stop.wait(self.interval):
                self._sample()
        except Exception as exc:
            self.error = repr(exc)

    def __enter__(self):
        self.file = (self.directory / "device_memory.csv").open("x", buffering=1)
        try:
            self.writer = csv.writer(self.file)
            self.writer.writerow(("elapsed_seconds", "nvml_device_used_bytes"))
            self._sample()  # Synchronous baseline before even starting MPS.
            self.initial = self.last
            self.thread.start()
            return self
        except BaseException:
            self.file.close()
            raise

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()
        try:
            self._sample()  # After the server and our private MPS daemon exited.
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self.file.close()

    def summary(self):
        return dict(cap_bytes=self.device.cap_bytes, gpu_uuid=self.device.uuid, source="NVML device memory info v2.used",
                    sampled_peak_bytes=self.peak, peak_elapsed_s=self.peak_time, samples=self.count,
                    initial_used_bytes=self.initial, final_used_bytes=self.last,
                    interval_ms=self.interval * 1000, max_sample_gap_ms=self.max_gap * 1000,
                    over_limit=self.over_limit, monitor_error=self.error)


def stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def private_mps(env, device, quota, directory):
    with tempfile.TemporaryDirectory(prefix="slora-mps-") as tmp, (directory / "mps.log").open("x") as log:
        pipe = Path(tmp)
        mps_env = dict(env, CUDA_VISIBLE_DEVICES=device.uuid, CUDA_MPS_PIPE_DIRECTORY=str(pipe),
                       CUDA_MPS_LOG_DIRECTORY=tmp, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=f"{device.uuid}={quota // MIB}M")
        daemon = subprocess.Popen(["nvidia-cuda-mps-control", "-f"], env=mps_env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            pidfile = pipe / "nvidia-cuda-mps-control.pid"
            deadline = time.monotonic() + 10
            while not (pidfile.exists() and pidfile.read_text().strip() == str(daemon.pid)):
                if daemon.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"Private MPS daemon did not start; see {log.name}")
                time.sleep(.05)
            yield daemon, dict(mps_env, CUDA_VISIBLE_DEVICES="", SLORA_BENCHMARK_MPS_GPU=device.uuid,
                               SLORA_BENCHMARK_MPS_BYTES=str(quota),
                               SLORA_BENCHMARK_MPS_REPORT=str(directory / "mps_client.json"))
        finally:
            # Signal only the foreground process we created. Never send a
            # control command to a shared MPS service.
            stop_owned(daemon)
            deadline = time.monotonic() + 15
            while device.processes():
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"GPU {device.uuid} still has active contexts after teardown")
                time.sleep(.05)


def nvml_host_pids(pids):
    # NVML reports host PIDs, while MPS control returns PIDs in its namespace.
    # Depending on the procfs mount, the outer PID appears in sched or NStgid.
    host_pids = set()
    for pid in pids:
        proc = Path("/proc/self") if pid == os.getpid() else Path(f"/proc/{pid}")
        try:
            with (proc / "sched").open() as f:
                match = re.search(r"\((\d+), #threads:", f.readline())
            if match:
                host_pids.add(int(match.group(1)))
            else:
                status = (proc / "status").read_text()
                match = re.search(r"^NStgid:\s+(\d+)", status, re.M)
                host_pids.add(int(match.group(1)) if match else pid)
        except FileNotFoundError:
            pass
    return host_pids


def other_board_routers(device_uuid):
    """Return live benchmark routers bound to a different UUID in this PID namespace."""
    peers = set()
    namespace = os.readlink("/proc/self/ns/pid")
    for marker in Path(tempfile.gettempdir()).glob("slora-mps-*/slora-router.json"):
        try:
            if marker.parent.stat().st_uid != os.getuid():
                continue
            with marker.open() as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    worker = json.load(stream)
                    if worker["gpu_uuid"] != device_uuid and worker["pid_namespace"] == namespace:
                        peers.add(int(worker["pid"]))
        except (FileNotFoundError, PermissionError):
            # A private daemon may finish between enumeration and opening its marker.
            continue
    return nvml_host_pids(peers)


def check_gpu_owners(device, pids, daemon, env, report, require_client=False):
    if not pids:
        if require_client:
            raise RuntimeError("NVML cannot see the active benchmark CUDA context")
        return
    # NVML can list the MPS server, its CUDA client, or both depending on driver.
    def query(command):
        out = subprocess.check_output(["nvidia-cuda-mps-control"], env=env, text=True,
                                      input=command + "\n", timeout=5)
        return {int(p) for p in out.split()}

    servers = query("get_server_list")
    clients = set()
    for pid in servers:
        clients.update(query(f"get_client_list {pid}"))
    expected = {int(report.read_text())} if report.is_file() else set()
    if len(servers) > 1 or clients - expected:
        raise RuntimeError("Unexpected CUDA clients on the benchmark's private MPS server")
    if require_client and (not expected or expected != clients or len(servers) != 1):
        raise RuntimeError("The model worker is missing from the private MPS server's client list")
    # A starting MPS server can appear in NVML before get_server_list.
    children = Path(f"/proc/{daemon.pid}/task/{daemon.pid}/children").read_text()
    owned = nvml_host_pids(servers | clients | expected | set(map(int, children.split())))
    unknown = pids - owned
    if unknown and unknown & device.processes():
        raise ForeignGPUProcess(f"Unrecognized GPU processes {sorted(unknown)} on {device.uuid}")


def verify_startup_budget(device, directory, quota):
    report = json.loads((directory / "mps_client.json").read_text())
    used, free = device.used(), report["initial_cuda_free_bytes"]
    if (report["gpu_uuid"] != device.uuid or report["quota_bytes"] != quota or not 0 < free <= quota
            or used + free > device.cap_bytes):
        raise RuntimeError(f"Unsafe startup budget: board used {used} + MPS client free {free}; "
                           "increase --outside-reserve-gb")
    save_json(directory / "startup_budget.json", dict(board_used_bytes=used, client_free_bytes=free,
                                                       board_used_plus_client_free_bytes=used + free))


def read_cuda_oom(path, offset):
    if not path.is_file():
        return False, offset
    with path.open("rb") as stream:
        stream.seek(max(0, offset - 64))
        return CUDA_OOM.search(stream.read()) is not None, stream.tell()


def guarded_replay(command, source, env, device, quota, directory, interval=.02):
    """Return diagnostics even for OOM or interruption. Infrastructure errors propagate."""
    directory.mkdir(parents=True, exist_ok=False)
    if quota <= 0 or quota % MIB or quota + device.used() > device.cap_bytes or device.processes():
        raise RuntimeError(f"Need an idle GPU and a client quota within the {device.cap_bytes}-byte board budget")
    monitor = DeviceSampler(device, directory, interval)
    code = None
    error = None
    oom = False
    offset = 0
    previous_pids = None
    verified = False
    server_log = directory / "run/server.log"
    try:
        with monitor, private_mps(env, device, quota, directory) as (daemon, child_env):
            with (directory / "benchmark.log").open("x") as log:
                process = subprocess.Popen(command, cwd=source, env=child_env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        if monitor.error or daemon.poll() is not None:
                            raise RuntimeError(f"Lost the GPU monitor or MPS daemon: {monitor.error}")
                        if monitor.over_limit:
                            break
                        oom, offset = read_cuda_oom(server_log, offset)
                        if oom:
                            break
                        pids = device.processes()
                        worker_ready = not verified and (directory / "mps_client.json").is_file()
                        if pids != previous_pids or worker_ready:
                            check_gpu_owners(device, pids, daemon, child_env, directory / "router.pid", worker_ready)
                            previous_pids = pids
                        if worker_ready:
                            verify_startup_budget(device, directory, quota)
                            verified = True
                        try:
                            process.wait(timeout=.5)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    stop_owned(process)
                    code = process.returncode
                    if not oom:
                        oom, offset = read_cuda_oom(server_log, offset)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if monitor.count:
            record = dict(monitor.summary(), client_quota_bytes=quota, returncode=code,
                          scope="before MPS startup through service and MPS teardown", error=error, cuda_oom=oom,
                          verified_before_model_load=verified)
            save_json(directory / "memory.json", record)
    if monitor.error:
        raise RuntimeError(f"NVML monitoring failed: {monitor.error}")
    return record
