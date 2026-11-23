#!/opt/conda/bin/python
"""H20 environment adapter for the paired launcher's two Python executables."""
from contextlib import ExitStack
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "benchmarks/results/.h20"
SERVE_PYTHON = Path("/opt/conda/envs/slora/bin/python")
TRACE_PYTHON = Path("/opt/conda/bin/python")
DATASET = ROOT.parent / "dataset/lmsys"
UVLOOP = CACHE / "uvloop-0.21.0-py39"
# Immutable upstream revisions. Only the tokenizer and model metadata are used.
REVISIONS = {
    "7b": "4782ad278652c7c71b72204d462d6d01eaaf7549",
    "13b": "bf57045473f207bb1de1ed035ace226f4d9f9bba",
    "30b": "2b1edcdb3c7ced7bce6c1aa75c94545777c3118b",
}
FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer.model")
# Upstream publishes these same three tokenizer files at all three revisions.
SHARED_SHA256 = {
    "tokenizer.json": "f9ffc4aede0845ab65324ce5dccb823dca2427f9a0710981e5bc2398d73d8162",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "special_tokens_map.json": "ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531",
}


def model_dir(size):
    return CACHE / REVISIONS[size] / ("llama-" + size)


def prepare(sizes):
    for python in (TRACE_PYTHON, SERVE_PYTHON):
        if not os.access(python, os.X_OK):
            raise SystemExit(f"Missing Python executable: {python}")
    if not any(DATASET.glob("*.parquet")):
        raise SystemExit(f"No LMSYS parquet files in {DATASET}")
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not UVLOOP.is_dir():
            with TemporaryDirectory(dir=CACHE) as tmp:
                target = Path(tmp) / "packages"
                subprocess.run([SERVE_PYTHON, "-m", "pip", "install", "--no-deps",
                                "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                                "--target", target, "uvloop==0.21.0"],
                               check=True)
                target.rename(UVLOOP)
        for size in sizes:
            destination = model_dir(size)
            if destination.is_dir():
                continue
            destination.parent.mkdir(exist_ok=True)
            url = f"https://huggingface.co/huggyllama/llama-{size}/resolve/{REVISIONS[size]}"
            with TemporaryDirectory(dir=CACHE) as tmp:
                staged = Path(tmp) / destination.name
                staged.mkdir()
                for name in FILES:
                    local = ROOT.parents[1] / "models/llama-13b" / name
                    if (name in SHARED_SHA256 and local.is_file() and
                            hashlib.sha256(local.read_bytes()).hexdigest() == SHARED_SHA256[name]):
                        shutil.copyfile(local, staged / name)
                        continue
                    subprocess.run(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "10",
                                    "--max-time", "90", "-o", staged / name, f"{url}/{name}"],
                                   check=True)
                staged.rename(destination)


def port_base(gpu):
    # Disjoint ranges below the host's ephemeral ports for the two TP=1 services.
    start = 24000 + 1000 * gpu
    for base in range(start, start + 990, 6):
        try:
            with ExitStack() as stack:
                for port in range(base, base + 6):
                    sock = stack.enter_context(socket.socket())
                    sock.bind(("", port))
            return base
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise RuntimeError(f"No six free ports for GPU {gpu}")


def command(args, plan_gpu=None):
    """Translate only machine-specific options. Preserve all experiment parameters."""
    args = list(args)
    script = Path(args[1] if args[0] == "-u" else args[0]).name
    if script == "compare_prefix_sources.py":
        return [str(SERVE_PYTHON), *args]
    if script == "prepare_lmsys_maf_trace.py":
        python = TRACE_PYTHON
        key = "--tokenizer"
        args[args.index("--dataset") + 1] = str(DATASET)
    elif script == "lowra_saturation_exp.py":
        python = SERVE_PYTHON
        key = "--model-dir"
        if plan_gpu is None:
            gpu = os.environ.get("SLORA_H20_PHYSICAL_GPU") or os.environ["CUDA_VISIBLE_DEVICES"]
            base = port_base(int(gpu))
        else:
            base = 24000 + 1000 * plan_gpu
        for flag, port in (("--port", base), ("--nccl-port", base + 1),
                           ("--internal-port-start", base + 2)):
            args[args.index(flag) + 1] = str(port)
    else:
        raise ValueError(f"Unknown H20 entry point: {script}")
    index = args.index(key) + 1
    size = args[index].removeprefix("huggyllama/llama-")
    args[index] = str(model_dir(size))
    return [str(python), *args]


if __name__ == "__main__":
    cmd = command(sys.argv[1:])
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    if cmd[0] == str(SERVE_PYTHON):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(UVLOOP), env.get("PYTHONPATH")]))
    os.execve(cmd[0], cmd, env)
