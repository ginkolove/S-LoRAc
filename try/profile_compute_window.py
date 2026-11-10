"""Replay the real decode O/residual/FFN/next-QKV window on CUDA stream 0."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slora.common.lowra_reconstruct import lowra_reconstruct_batched
from slora.common.lowra_reconstruct_kv import lowra_reconstruct_kv_batched
from slora.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from slora.models.peft.lora_unordered_batch_infer import LoraUnorderedBatchInfer
from profile_reconstruct_ffn import Workload, gpu_snapshot, random_tensor
from overlap_analysis import summarize, analyze_compute_window_trace


STAGES = ("o", "o_residual", "ffn_norm", "ffn", "ffn_residual", "qkv_norm", "qkv")


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        torch.cuda.nvtx.range_pop()


class ComputeWindow:
    def __init__(self, args):
        self.args = args
        h, d, u, b, r = args.hidden_size, 128, args.unique_loras, args.batch_size, args.rank
        config = dict(hidden_size=h, intermediate_size=args.intermediate_size,
                      num_attention_heads=h // d, rms_norm_eps=1e-6)
        self.layers = [LlamaTransformerLayerInfer(i, 0, 1, config) for i in range(2)]
        self.weight = SimpleNamespace(
            **{name: random_tensor((h, h), "cuda") for name in ("o_weight_", "q_weight_", "k_weight_", "v_weight_")},
            gate_proj=random_tensor((h, args.intermediate_size), "cuda"),
            up_proj=random_tensor((h, args.intermediate_size), "cuda"),
            down_proj=random_tensor((args.intermediate_size, h), "cuda"),
            att_norm_weight_=torch.ones(h, device="cuda", dtype=torch.float16),
            ffn_norm_weight_=torch.ones(h, device="cuda", dtype=torch.float16))
        memory = SimpleNamespace(
            key_buffer=[random_tensor((4 * u * r, h // d, d), "cuda") for _ in range(2)],
            value_buffer=[random_tensor((4 * u * r, h // d, d), "cuda") for _ in range(2)])
        adapter = SimpleNamespace(mem_manager=memory, adapter_dirs=list(range(u)),
                                 a_start=torch.arange(u, device="cuda") * 4 * r,
                                 a_len=torch.full((u,), 4 * r, device="cuda", dtype=torch.long),
                                 a_loc=torch.arange(4 * u * r, device="cuda"),
                                 a_scaling=torch.ones(u, device="cuda", dtype=torch.float16))
        model = SimpleNamespace(tp_k_head_num_=h // d, head_dim_=d, trans_layers_weight=[self.weight] * 2,
                                layers_infer=self.layers)
        self.engine = LoraUnorderedBatchInfer(model, [SimpleNamespace(r=r, lora_dir=i % u) for i in range(b)], adapter)
        self.engine.delta = [torch.zeros((b, r), device="cuda", dtype=torch.float16) for _ in range(3)]
        positions = (args.prefix_len + args.query_len + torch.arange(b, device="cuda") % 64).float()
        frequency = 1.0 / (10000 ** (torch.arange(0, d, 2, device="cuda").float() / d))
        angles = positions[:, None] * frequency[None, :]
        self.state = SimpleNamespace(lowra_async_execution=True,
                                     position_cos=angles.cos().half(), position_sin=angles.sin().half())
        self.initial = random_tensor((b, h), "cuda")
        self.activation = self.initial.clone()
        self.attention_output = random_tensor((b, h), "cuda")
        self.k = torch.empty((b, h // d, d), device="cuda", dtype=torch.float16)
        self.v = torch.empty_like(self.k)
        self.reconstruction = Workload(SimpleNamespace(hidden_size=h, intermediate_size=1,
                                                       device="cuda", row_stride=1), args.prefix_len, u, 1, r)
        # Use actual next-layer packed adapter B rows also used by QKV LoRA computation.
        self.reconstruction.b = memory.value_buffer[1].view(-1, h)
        starts = torch.arange(u, device="cuda")[:, None] * 4 * r
        rank_columns = torch.arange(r, device="cuda")[None, :]
        self.reconstruction.b_locations = torch.stack((starts + r + rank_columns, starts + 2 * r + rank_columns))
        self.reconstruction.scales.fill_(1)
        self.spans = {}
        for width in set([0, args.prefix_len, *args.chunk_tokens]):
            self.spans[(0, width)] = self.reconstruction.locations[:, :width].contiguous()
            self.spans[(width, args.prefix_len)] = self.reconstruction.locations[:, width:].contiguous()
        self.side = torch.cuda.Stream()
        self.start, self.ready, self.end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        self.consumed = torch.cuda.Event()
        self.stage_events = {stage: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                             for stage in STAGES}

    def stage(self, name):
        layer = self.layers[0]
        if name == "o":
            self.o = self.engine._batch_lora_get_o(0, self.attention_output, self.state)
        elif name == "o_residual":
            self.activation.add_(self.o)
        elif name == "ffn_norm":
            self.norm = layer._ffn_norm(self.activation, self.state, self.weight)
        elif name == "ffn":
            self.ffn = layer._ffn(self.norm, self.state, self.weight)
        elif name == "ffn_residual":
            self.activation.add_(self.ffn)
        elif name == "qkv_norm":
            self.norm = self.layers[1]._att_norm(self.activation, self.state, self.weight)
        elif name == "qkv":
            self.q = self.engine._batch_lora_get_qkv(1, self.norm, self.k, self.v, self.state)

    def reconstruct(self, start=0, end=None, fused=False):
        w = self.reconstruction
        end = self.args.prefix_len if end is None else end
        if start == end:
            return
        if fused:
            lowra_reconstruct_kv_batched(*w.base, *w.mini, w.b, *w.outputs,
                                        w.locations, *w.b_locations, w.starts, w.ranks, w.scales,
                                        launch_config=self.args.fused_config,
                                        embed_first=self.args.embed_first)
            return
        for kv in range(2):
            lowra_reconstruct_batched(w.base[kv][start:end], w.mini[kv][start:end], w.b,
                                      w.outputs[kv], self.spans[(start, end)], w.b_locations[kv],
                                      w.starts, w.ranks, w.scales)

    def run(self, mode, target=None, chunk=0, instrument=False):
        if torch.cuda.current_stream() != torch.cuda.default_stream():
            raise RuntimeError("BGMV uses CUDA stream 0; keep the main window on that stream")
        self.activation.copy_(self.initial)
        self.start.record()
        asynchronous = mode.startswith("overlap")
        if mode.startswith("serial") or mode == "reconstruct":
            with label("window/reconstruct") if instrument else _null():
                self.reconstruct(fused=mode.endswith("fused"))
        if asynchronous and target is None and not mode.startswith("overlap_main_first"):
            self.side.wait_event(self.start)
            with torch.cuda.stream(self.side):
                with label("window/reconstruct") if instrument else _null():
                    self.reconstruct(fused=mode.endswith("fused"))
                self.ready.record()
        if mode != "reconstruct":
            for stage in STAGES:
                begin, end = self.stage_events[stage]
                if instrument:
                    begin.record()
                if mode == "window" and stage == target:
                    self.consumed.record()
                    self.side.wait_event(self.consumed)
                    with torch.cuda.stream(self.side):
                        with label("window/reconstruct_chunk") if instrument else _null():
                            self.reconstruct(0, chunk)
                        self.ready.record()
                if asynchronous and stage == target:
                    self.consumed.record()
                    self.side.wait_event(self.consumed)
                    with torch.cuda.stream(self.side):
                        with label("window/reconstruct") if instrument else _null():
                            self.reconstruct(fused=mode.endswith("fused"))
                        self.ready.record()
                with label("window/" + stage) if instrument else _null():
                    self.stage(stage)
                if mode == "window" and stage == target:
                    torch.cuda.current_stream().wait_event(self.ready)
                if instrument:
                    end.record()
        if mode.startswith("overlap_main_first"):
            self.side.wait_event(self.start)
            with torch.cuda.stream(self.side):
                self.reconstruct(fused=mode.endswith("fused"))
                self.ready.record()
        if mode == "window":
            with label("window/reconstruct_tail") if instrument else _null():
                self.reconstruct(chunk, self.args.prefix_len)
        elif asynchronous:
            torch.cuda.current_stream().wait_event(self.ready)
        self.end.record()
        self.end.synchronize()
        result = dict(total_ms=self.start.elapsed_time(self.end))
        if instrument and mode != "reconstruct":
            result["stages_ms"] = {name: begin.elapsed_time(end) for name, (begin, end) in self.stage_events.items()}
        return result

    def validate(self, mode):
        if mode != "reconstruct":
            for name in ("activation", "q", "k", "v"):
                torch.testing.assert_close(getattr(self, name), self.reference[name], atol=1e-4, rtol=1e-3)
        if mode != "main":
            for actual, expected in zip(self.reconstruction.outputs, self.prefix_reference):
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)


@contextmanager
def _null():
    yield


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-len", type=int, default=4608)
    parser.add_argument("--query-len", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unique-loras", type=int, default=20)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--fused-config", type=int, nargs=5)
    parser.add_argument("--embed-first", type=int, choices=[0, 1])
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[27, 54, 108, 216, 432])
    parser.add_argument("--bounded-tile", type=int, nargs=4, action="append")
    parser.add_argument("--reconstruct-graphs", action="store_true")
    parser.add_argument("--start-stages", choices=STAGES, nargs="+", default=["o"])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--chunk-tokens", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--windows", nargs="*", choices=STAGES, default=list(STAGES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--profile-mode", choices=["main", "reconstruct", "serial_split", "overlap_split", "overlap_fused"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.trace and args.reconstruct_graphs:
        parser.error("use separate timing-only Graph and eager trace runs; Torch 2.0 CUPTI Graph tracing can hang")
    if not 0 < args.unique_loras <= args.batch_size:
        parser.error("unique LoRAs must fit the batch")
    if args.hidden_size % 128 or min(args.prefix_len, args.rank, args.iterations, args.warmup, *args.chunk_tokens) <= 0:
        parser.error("invalid dimensions or repetition counts")
    if max(args.chunk_tokens) > args.prefix_len:
        parser.error("chunk size exceeds prefix")
    return args


@torch.no_grad()
def main(args):
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "result.json"
    if path.exists():
        raise FileExistsError(path)
    window = ComputeWindow(args)
    window.run("serial_split")
    window.reference = {name: getattr(window, name).clone() for name in ("activation", "q", "k", "v")}
    window.prefix_reference = [out.clone() for out in window.reconstruction.outputs]
    choices = [(mode, None, 0) for mode in ("main", "reconstruct", "serial_split", "serial_fused", "overlap_split", "overlap_fused",
                                           "overlap_main_first_split", "overlap_main_first_fused")]
    choices.extend(("window", stage, chunk) for stage in args.windows for chunk in args.chunk_tokens)
    if args.profile_mode:
        choices = [(args.profile_mode, None, 0)]
    samples = {choice: [] for choice in choices}
    for choice in choices:
        for _ in range(args.warmup):
            window.run(*choice)
        for out in window.reconstruction.outputs:
            out.fill_(-7)
        torch.cuda.synchronize()
        window.run(*choice)
        window.validate(choice[0])
    rng = random.Random(args.seed)
    torch.cuda.cudart().cudaProfilerStart()
    for iteration in range(args.iterations):
        order = list(choices)
        rng.shuffle(order)
        for choice in order:
            samples[choice].append(window.run(*choice, instrument=bool(args.profile_mode)))
    torch.cuda.cudart().cudaProfilerStop()
    rows = []
    for choice, values in samples.items():
        mode, stage, chunk = choice
        row = dict(mode=mode, window=stage, chunk_tokens=chunk,
                   timings=summarize([v["total_ms"] for v in values]), samples=values)
        if args.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                row["instrumented"] = window.run(*choice, instrument=True)
            trace_path = args.output_dir / ("%s_%s_%s.json" % choice)
            profiler.export_chrome_trace(str(trace_path))
            row["trace"] = str(trace_path)
            row["kernel_trace"] = analyze_compute_window_trace(json.loads(trace_path.read_text()))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("samples", "kernel_trace")}), flush=True)
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  status="complete", correctness_passed=True, gpu=gpu_snapshot(), rows=rows,
                  scope="Actual decode O/LoRA, residual, FFN, next Norm/QKV/LoRA/RoPE, synthetic nonzero tensors. "
                        "Stream 0 main, side reconstruction, same packed B weights. Excludes attention and scheduling. "
                        "Window chunks are fenced at the end of the chosen stage; remainder runs before next attention.",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "window", "chunk_tokens", "mean", "median", "p95", "min", "max"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(mode=row["mode"], window=row["window"], chunk_tokens=row["chunk_tokens"], **row["timings"]))
    print("Results: %s" % path, flush=True)


if __name__ == "__main__":
    main(parse_args())
