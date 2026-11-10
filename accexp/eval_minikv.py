"""Three-way accuracy evaluation using only PyTorch, Transformers and PEFT.

Input JSONL: {"id": "...", "prefix": "...", "query": "...",
              "references": ["..."], "metadata": {}}
Prefixes and queries must already contain the dataset's prompt formatting.
"""

import argparse
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, ExitStack, nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import re
import string
import sys
import threading
import time

import torch
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache
import peft
from peft import PeftConfig, PeftModel

if __package__:
    from .data.common import DATASETS, add_dataset_arguments, load_dataset_samples, validate_dataset_arguments
else:
    from data.common import DATASETS, add_dataset_arguments, load_dataset_samples, validate_dataset_arguments


MODES = ("base", "lora_full", "lora_minikv")


class EvaluationControl:
    def __init__(self):
        self.stop = threading.Event()
        self.error = None
        self.lock = threading.Lock()

    def fail(self, error):
        with self.lock:
            if self.error is None:
                self.error = error
            self.stop.set()

    def wait(self, event):
        while not event.wait(0.1):
            self.check()
        self.check()

    def check(self):
        if self.stop.is_set():
            raise RuntimeError("Evaluation cancelled after a worker failure") from self.error


class MiniKVQueue:
    """Atomically claim disjoint, length-sorted batches from each prefix group."""

    def __init__(self, groups, exact_lengths=False):
        self.groups = deque((key, deque(sorted(items, key=lambda item: len(item[1]["query_ids"]))))
                            for key, items in groups.items())
        self.exact_lengths = exact_lengths
        self.lock = threading.Lock()
        self.assigned, self.completed = {}, set()
        self.worker_counts = Counter()
        self.total = sum(len(items) for items in groups.values())

    def claim(self, worker, batch_size):
        with self.lock:
            if not self.groups:
                return None
            key, pending = self.groups[0]
            length = len(pending[0][1]["query_ids"])
            batch = []
            while pending and len(batch) < batch_size:
                if self.exact_lengths and len(pending[0][1]["query_ids"]) != length:
                    break
                index, item = pending.popleft()
                if index in self.assigned:
                    raise RuntimeError(f"MiniKV sample {index} was assigned twice")
                self.assigned[index] = worker
                batch.append((index, item))
            if not pending:
                self.groups.popleft()
            return key, batch

    def complete(self, worker, batch):
        with self.lock:
            for index, _ in batch:
                if self.assigned.get(index) != worker or index in self.completed:
                    raise RuntimeError(f"Invalid MiniKV completion for sample {index}")
                self.completed.add(index)
                self.worker_counts[worker] += 1

    def validate(self):
        with self.lock:
            if self.groups or len(self.assigned) != self.total or len(self.completed) != self.total:
                raise RuntimeError("MiniKV evaluation has missing or unfinished samples")


class AccuracyProgress:
    """Three mode counters; redirected logs use occasional plain-text lines."""

    def __init__(self, mode, total, position):
        self.description = f"Accuracy {mode}"
        self.count, self.total = 0, total
        self.postfix = {}
        self.last_log = time.perf_counter()
        self.lock = threading.Lock()
        self.terminal = sys.stderr.isatty()
        self.bar = tqdm(total=total, desc=self.description, unit="req", position=position,
                        disable=not self.terminal)

    def set_description(self, description):
        with self.lock:
            if description == self.description:
                return
            self.description = description
            self.bar.set_description(description)
            if not self.terminal:
                self.log()

    def set_postfix(self, refresh=False, **values):
        with self.lock:
            self.postfix = values
            self.bar.set_postfix(refresh=refresh, **values)

    def update(self, count):
        with self.lock:
            self.count += count
            self.bar.update(count)
            if not self.terminal and (self.count == self.total or
                                      time.perf_counter() - self.last_log >= 30):
                self.log()

    def log(self):
        fields = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        print(f"{self.description}: {self.count}/{self.total} {fields}", file=sys.stderr, flush=True)
        self.last_log = time.perf_counter()

    def close(self):
        self.bar.close()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_stage(stats, name, device):
    synchronize(device)
    started = time.perf_counter()
    yield
    synchronize(device)
    stats["stage_seconds"][name] = stats["stage_seconds"].get(name, 0.0) + time.perf_counter() - started


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def load_samples(path, limit=0):
    """Dataset boundary: consume prepared records; do no dataset-specific work."""
    samples, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise ValueError(f"Line {line_number}: expected an object")
            sample = dict(sample)
            sample["id"] = str(sample.get("id", line_number))
            if sample["id"] in seen:
                raise ValueError(f"Duplicate sample id: {sample['id']}")
            seen.add(sample["id"])
            for field in ("prefix", "query"):
                if not isinstance(sample.get(field), str) or not sample[field]:
                    raise ValueError(f"Sample {sample['id']}: {field} must be a nonempty string")
            references = sample.get("references")
            if not isinstance(references, list) or not references or not all(
                isinstance(answer, str) for answer in references
            ):
                raise ValueError(f"Sample {sample['id']}: references must be a nonempty list of strings")
            samples.append(sample)
            if limit and len(samples) >= limit:
                break
    if not samples:
        raise ValueError("The input dataset is empty")
    return samples


def prepare_inputs(samples, tokenizer, max_new_tokens, context_limit, add_special_tokens=True):
    prepared = []
    for sample in samples:
        full_ids = tokenizer.encode(sample["prefix"] + sample["query"],
                                    add_special_tokens=add_special_tokens)
        prefix_ids = tokenizer.encode(sample["prefix"], add_special_tokens=add_special_tokens)
        # A BPE token can straddle the text boundary. Cache only the exact common
        # token prefix, leaving that boundary token in the query.
        boundary = 0
        for left, right in zip(prefix_ids, full_ids):
            if left != right:
                break
            boundary += 1
        if not 0 < boundary < len(full_ids):
            raise ValueError(f"Sample {sample['id']}: need a nonempty token prefix and query")
        if len(full_ids) + max_new_tokens > context_limit:
            raise ValueError(
                f"Sample {sample['id']}: {len(full_ids)} input + {max_new_tokens} output "
                f"exceeds context limit {context_limit}; preprocess the sample explicitly"
            )
        prefix_ids = full_ids[:boundary]
        digest = hashlib.sha256(json.dumps(prefix_ids).encode("ascii")).hexdigest()
        prepared.append(dict(sample=sample, full_ids=full_ids, prefix_ids=prefix_ids,
                             query_ids=full_ids[boundary:], prefix_key=digest))
    return prepared


def core_model(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def input_device(model):
    return core_model(model).get_input_embeddings().weight.device


def cache_pairs(cache):
    if hasattr(cache, "to_legacy_cache"):
        return list(cache.to_legacy_cache())
    return [(layer.keys, layer.values) for layer in cache.layers]


def snapshot_cache(cache, device="cpu"):
    return tuple((key.detach().to(device=device or key.device, copy=True),
                  value.detach().to(device=device or value.device, copy=True))
                 for key, value in cache_pairs(cache))


def fresh_cache(model, pairs=(), batch_size=1):
    # Preserve the whole prefix, including on sliding-window models. Their native
    # attention mask still enforces the configured window. Absolute positions
    # therefore remain independent of any cache truncation policy.
    cache = DynamicCache()
    layers = core_model(model).model.layers
    for index, (key, value) in enumerate(pairs):
        device = layers[index].self_attn.k_proj.weight.device
        key, value = key.to(device=device), value.to(device=device)
        if batch_size == 1:
            key, value = key.clone(), value.clone()
        else:
            key = key.repeat_interleave(batch_size, dim=0)
            value = value.repeat_interleave(batch_size, dim=0)
        cache.update(key, value, index)
    return cache


def place_prefix(model, pairs):
    layers = core_model(model).model.layers
    return tuple((key.to(layer.self_attn.k_proj.weight.device, copy=True),
                  value.to(layer.self_attn.v_proj.weight.device, copy=True))
                 for layer, (key, value) in zip(layers, pairs))


def right_align_cache(cache, valid_lengths):
    """Move trailing padding before each complete history, one working layer at a time."""
    for layer in cache.layers:
        width = layer.keys.shape[-2]
        offsets = width - valid_lengths.to(layer.keys.device)
        indices = (torch.arange(width, device=layer.keys.device)[None, :] - offsets[:, None]).clamp_min(0)
        for name in ("keys", "values"):
            states = getattr(layer, name)
            gather = indices[:, None, :, None].expand_as(states)
            # Leading padding may contain duplicates; the attention mask hides it.
            setattr(layer, name, states.gather(2, gather))


def place_minikv(model, captured):
    return [{name: value.to(device=layer.self_attn.k_proj.weight.device, copy=True)
             if isinstance(value, torch.Tensor) else value
             for name, value in state.items()}
            for layer, state in zip(core_model(model).model.layers, captured)]


@torch.inference_mode()
def forward_tokens(model, token_ids, cache):
    device = input_device(model)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    start = cache.get_seq_length()
    positions = torch.arange(start, start + len(token_ids), device=device)
    kwargs = dict(input_ids=ids, attention_mask=torch.ones(
        (1, start + len(token_ids)), dtype=torch.long, device=device),
        position_ids=positions.unsqueeze(0), cache_position=positions,
        past_key_values=cache, use_cache=True, return_dict=True)
    signature = inspect.signature(core_model(model).forward).parameters
    if "logits_to_keep" in signature:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs)


@torch.inference_mode()
def generate(model, query_ids, prefix_pairs, max_new_tokens, eos_ids):
    cache = fresh_cache(model, prefix_pairs)
    output = forward_tokens(model, query_ids, cache)
    first_logits = output.logits[:, -1].float().cpu()
    generated = []
    for step in range(max_new_tokens):
        if not torch.isfinite(output.logits[:, -1]).all():
            raise RuntimeError("Nonfinite generation logits")
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        output = forward_tokens(model, [token], output.past_key_values)
    return generated, first_logits


@torch.inference_mode()
def generate_batch_transformers(model, queries, prefix_pairs, max_new_tokens, eos_ids, pad_id=0):
    """Supply independent prefix KV to native generation; never decode tokens here."""
    device = input_device(model)
    width = max(map(len, queries))
    ids = torch.tensor([[pad_id] * (width - len(query)) + query for query in queries],
                       dtype=torch.long, device=device)
    lengths = torch.tensor([len(query) for query in queries], device=device)
    cache = fresh_cache(model, prefix_pairs, batch_size=len(queries))
    prefix_len = cache.get_seq_length()
    mask = torch.cat((torch.ones((len(queries), prefix_len), dtype=torch.long, device=device),
                      (torch.arange(width, device=device)[None, :] >= width - lengths[:, None]).long()), dim=1)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
        repetition_penalty=1.0, eos_token_id=sorted(eos_ids), pad_token_id=pad_id,
        bos_token_id=getattr(core_model(model).generation_config, "bos_token_id", None))
    output = model.generate(
        input_ids=ids, attention_mask=mask, past_key_values=cache,
        cache_position=torch.arange(prefix_len, prefix_len + width, device=device),
        generation_config=generation_config, use_model_defaults=False)
    rows = output[:, width:].cpu().tolist()
    # Native generation pads rows that finished before the rest of their batch.
    return [row[:next((index + 1 for index, token in enumerate(row) if token in eos_ids), len(row))]
            for row in rows]


@torch.inference_mode()
def generate_batch(model, queries, prefix_pairs, max_new_tokens, eos_ids, pad_id=0):
    """Reuse immutable prefix KV without putting padding inside a sliding history."""
    if core_model(model).config.model_type == "qwen2":
        return generate_batch_transformers(model, queries, prefix_pairs, max_new_tokens, eos_ids, pad_id)
    device = input_device(model)
    batch_size, width = len(queries), max(map(len, queries))
    cache = fresh_cache(model, prefix_pairs, batch_size=batch_size)
    prefix_len = cache.get_seq_length()
    lengths = torch.tensor([len(query) for query in queries], device=device)
    config = core_model(model).config
    sliding = bool(getattr(config, "sliding_window", None)) and getattr(config, "use_sliding_window", True)
    right_padded = sliding and any(
        len(query) != width for query in queries)
    if right_padded:
        ids = torch.tensor([query + [pad_id] * (width - len(query)) for query in queries],
                           dtype=torch.long, device=device)
        query_mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    else:
        ids = torch.tensor([[pad_id] * (width - len(query)) + query for query in queries],
                           dtype=torch.long, device=device)
        query_mask = torch.arange(width, device=device)[None, :] >= (width - lengths[:, None])
    mask = torch.cat((torch.ones((batch_size, prefix_len), dtype=torch.bool, device=device),
                      query_mask), dim=1)
    positions = prefix_len + (query_mask.long().cumsum(-1) - 1).clamp_min(0)
    cache_positions = torch.arange(prefix_len, prefix_len + width, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    generated = torch.full((batch_size, max_new_tokens), pad_id, dtype=torch.long, device=device)
    output_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    eos = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)
    signature = inspect.signature(core_model(model).forward).parameters
    extra = {"logits_to_keep": 1} if "logits_to_keep" in signature else {}
    prefill_extra = extra
    if right_padded:
        if extra:
            # Project only the distinct last-token positions, not every article token.
            keep, last_logits = torch.unique(lengths - 1, sorted=True, return_inverse=True)
            prefill_extra = {"logits_to_keep": keep}
        else:
            last_logits = lengths - 1
    for step in range(max_new_tokens):
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                       cache_position=cache_positions, past_key_values=cache,
                       use_cache=True, return_dict=True, **(prefill_extra if step == 0 else extra))
        logits = (output.logits[torch.arange(batch_size, device=device), last_logits]
                  if right_padded and step == 0 else output.logits[:, -1])
        if not torch.isfinite(logits[active]).all():
            raise RuntimeError("Nonfinite generation logits")
        tokens = logits.argmax(-1)
        generated[:, step] = torch.where(active, tokens, pad_id)
        output_lengths += active.long()
        active = active & ~torch.isin(tokens, eos)
        if not active.any() or step + 1 == max_new_tokens:
            break
        cache = output.past_key_values
        if right_padded and step == 0:
            # Decode sees [left padding, prefix, query], with no internal gap.
            # RoPE keeps logical positions; native window distances use contiguous KV.
            right_align_cache(cache, prefix_len + lengths)
            mask = torch.arange(prefix_len + width, device=device)[None, :] >= (
                width - lengths[:, None])
        ids = torch.where(active, tokens, pad_id)[:, None]
        mask = torch.cat((mask, active[:, None]), dim=1)
        positions = (prefix_len + lengths + step)[:, None]
        cache_positions = torch.tensor([cache.get_seq_length()], device=device)
    rows, sizes = generated.cpu().tolist(), output_lengths.cpu().tolist()
    return [row[:size] for row, size in zip(rows, sizes)]


@torch.inference_mode()
def check_cache(model, item, prefix_pairs, atol=None, rtol=0.005):
    full = forward_tokens(model, item["full_ids"], fresh_cache(model)).logits[:, -1]
    # Splitting prefill changes GEMM/attention shapes; low-precision roundoff
    # grows across layers even with a correctly constructed cache.
    if atol is None:
        atol = {torch.float16: 0.03, torch.bfloat16: 0.15}.get(full.dtype, 0.0001)
    full = full.float().cpu()
    split = forward_tokens(model, item["query_ids"], fresh_cache(model, prefix_pairs))
    split = split.logits[:, -1].float().cpu()
    if not torch.isfinite(full).all() or not torch.isfinite(split).all():
        raise RuntimeError("Nonfinite logits in the prefix-cache consistency check")
    maximum = float((full - split).abs().max())
    if not torch.allclose(full, split, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"Prefix-cache injection failed for {item['sample']['id']}: "
            f"max logit difference {maximum:.6g} (atol={atol}, rtol={rtol})"
        )
    return dict(max_abs_logit_difference=maximum, atol=atol, rtol=rtol,
                relative_l2_logit_difference=float((full - split).norm() / full.norm().clamp_min(1e-12)),
                same_first_token=bool(torch.equal(full.argmax(-1), split.argmax(-1))))


@contextmanager
def capture_minikv(model, keep_on_device=False):
    """Capture LoRA A outputs on the adapted prefix forward, before RoPE."""
    captured, handles = [], []
    try:
        for layer in core_model(model).model.layers:
            attention = layer.self_attn
            state = {"head_dim": attention.head_dim, "mini_k": None, "mini_v": None,
                     "expected": []}
            captured.append(state)

            def save_positions(module, args, kwargs, state=state):
                embeddings = kwargs.get("position_embeddings")
                if embeddings is None:
                    raise RuntimeError("This Transformers attention does not expose position_embeddings")
                state["cos"], state["sin"] = (
                    t.detach().to(device=t.device if keep_on_device else "cpu", copy=True)
                    for t in embeddings)

            handles.append(attention.register_forward_pre_hook(save_positions, with_kwargs=True))
            for name in ("k", "v"):
                projection = getattr(attention, name + "_proj")
                if not hasattr(projection, "lora_A") or "default" not in projection.lora_A:
                    continue
                if getattr(projection, "lora_variant", {}).get("default") is not None:
                    raise ValueError("MiniKV currently requires standard additive LoRA on K/V")
                state["expected"].append("mini_" + name)

                def save_a(module, args, output, state=state, name=name):
                    state["mini_" + name] = output.detach().to(
                        device=output.device if keep_on_device else "cpu", copy=True)

                handles.append(projection.lora_A["default"].register_forward_hook(save_a))
        yield captured
        if any("cos" not in state or any(state[name] is None for name in state["expected"])
               for state in captured):
            raise RuntimeError("The prefix forward did not execute the expected attention/LoRA hooks")
    finally:
        for handle in handles:
            handle.remove()


def rotate_delta(delta, cos, sin):
    first, second = delta.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return delta * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


@torch.inference_mode()
def reconstruct_prefix(model, base_pairs, captured, keep_on_device=False):
    """Approximate adapted KV: shared BaseKV + that adapter's low-rank delta."""
    layers = core_model(model).model.layers
    if len(base_pairs) != len(layers) or len(captured) != len(layers):
        raise ValueError("BaseKV, MiniKV and model layer counts differ")
    reconstructed = []
    for layer, (base_k, base_v), state in zip(layers, base_pairs, captured):
        pair = []
        for name, base in (("k", base_k), ("v", base_v)):
            mini = state["mini_" + name]
            if mini is None:
                pair.append(base.clone())
                continue
            projection = getattr(layer.self_attn, name + "_proj")
            b = projection.lora_B["default"]
            delta = b(mini.to(device=b.weight.device, dtype=b.weight.dtype))
            delta = delta * projection.scaling["default"]
            delta = delta.view(1, base.shape[2], base.shape[1], state["head_dim"]).transpose(1, 2)
            if name == "k":
                delta = rotate_delta(delta, state["cos"].to(delta.device, delta.dtype),
                                     state["sin"].to(delta.device, delta.dtype))
            result = (base.to(delta.device).float() + delta.float()).to(base.dtype)
            if not torch.isfinite(result).all():
                raise RuntimeError("Nonfinite reconstructed prefix KV")
            pair.append(result if keep_on_device else result.to("cpu", copy=True))
        reconstructed.append(tuple(pair))
    return tuple(reconstructed)


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def default_score(predictions, references, samples):
    """Generic English EM/whitespace-token F1, not an official dataset scorer."""
    exact, f1 = [], []
    for prediction, answers in zip(predictions, references):
        normalized = normalize_answer(prediction)
        exact.append(max(float(normalized == normalize_answer(answer)) for answer in answers))
        token_scores = []
        for answer in answers:
            left, right = normalized.split(), normalize_answer(answer).split()
            if not left or not right:
                token_scores.append(float(left == right))
                continue
            overlap = sum((Counter(left) & Counter(right)).values())
            token_scores.append(2.0 * overlap / (len(left) + len(right)))
        f1.append(max(token_scores))
    return dict(exact_match=sum(exact) / len(exact), generic_token_f1=sum(f1) / len(f1))


def load_scorer(specification):
    if not specification:
        return default_score
    source, separator, name = specification.rpartition(":")
    if not separator:
        raise ValueError("--scorer must be module:function or /path/to/file.py:function")
    if source.endswith(".py"):
        spec = importlib.util.spec_from_file_location("accexp_dataset_scorer", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)
    function = getattr(module, name)
    if not callable(function):
        raise ValueError("The requested scorer is not callable")
    return function


class PartialResults:
    """Persist each completed batch before publishing its cumulative scores."""

    def __init__(self, destination, samples, scorer):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path = destination.with_suffix(".partial.json")
        self.journal_path = destination.with_suffix(".predictions.jsonl")
        self.samples, self.scorer = samples, scorer
        self.rows = {mode: {} for mode in MODES}
        self.values = {mode: {} for mode in MODES}
        self.totals = {mode: Counter() for mode in MODES}
        self.lock = threading.RLock()
        self.state, self.error = "running", None
        self.started = time.time()
        self.journal = self.journal_path.open("x", encoding="utf-8")
        self.write_summary()

    def mode_scores(self, mode, indices=None):
        indices = sorted(self.rows[mode]) if indices is None else indices
        if not indices:
            return None
        if self.scorer is default_score:
            totals = (self.totals[mode] if len(indices) == len(self.rows[mode]) else
                      {name: sum(self.values[mode][index][name] for index in indices)
                       for name in ("exact_match", "generic_token_f1")})
            return {name: value / len(indices) for name, value in totals.items()}
        return self.scorer([self.rows[mode][i]["text"] for i in indices],
                           [self.samples[i]["references"] for i in indices],
                           [self.samples[i] for i in indices])

    def write_summary(self):
        common = sorted(self.rows["lora_full"].keys() & self.rows["lora_minikv"].keys())
        paired = {mode: self.mode_scores(mode, common) for mode in ("lora_full", "lora_minikv")}
        delta = None if not common else {
            name: value - paired["lora_full"][name]
            for name, value in paired["lora_minikv"].items()
            if isinstance(value, (int, float)) and isinstance(paired["lora_full"].get(name), (int, float))}
        summary = dict(state=self.state, error=self.error, total_samples=len(self.samples),
                       started_at_unix=self.started, updated_at_unix=time.time(),
                       journal=str(self.journal_path),
                       completed_requests={mode: len(rows) for mode, rows in self.rows.items()},
                       scores={mode: self.mode_scores(mode) for mode in MODES},
                       paired_lora=dict(samples=len(common), scores=paired, minikv_minus_lora=delta))
        temporary = self.summary_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.summary_path)

    def commit(self, mode, batch):
        with self.lock:
            entries = []
            for index, record in batch:
                if index in self.rows[mode]:
                    raise RuntimeError(f"Duplicate saved prediction: {mode}, sample {index}")
                entry = dict(record, sample_index=index,
                             predictions={mode: record["predictions"][mode]})
                entries.append(entry)
            payload = dict(mode=mode, completed_at_unix=time.time(), records=entries)
            self.journal.write(json.dumps(payload, ensure_ascii=False, default=json_default,
                                          allow_nan=False) + "\n")
            self.journal.flush()
            os.fsync(self.journal.fileno())
            for entry in entries:
                index = entry["sample_index"]
                prediction = entry["predictions"][mode]
                self.rows[mode][index] = prediction
                if self.scorer is default_score:
                    values = default_score([prediction["text"]], [self.samples[index]["references"]],
                                           [self.samples[index]])
                    self.values[mode][index] = values
                    self.totals[mode].update(values)
            self.write_summary()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        with self.lock:
            if error is not None:
                self.state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                self.error = str(error)
            elif all(len(rows) == len(self.samples) for rows in self.rows.values()):
                self.state = "complete"
            else:
                self.state = "incomplete"
            try:
                self.write_summary()
            finally:
                self.journal.close()


def prepare_records(prepared):
    groups = defaultdict(list)
    for index, item in enumerate(prepared):
        groups[item["prefix_key"]].append((index, item))
    records = [dict(id=item["sample"]["id"], references=item["sample"]["references"],
                    metadata=item["sample"].get("metadata", {}), prefix_key=item["prefix_key"],
                    prefix_tokens=len(item["prefix_ids"]), query_tokens=len(item["query_ids"]),
                    input_tokens=len(item["full_ids"]), predictions={}) for item in prepared]
    return groups, records


def iter_batches(model, items, batch_size):
    ordered = sorted(items, key=lambda entry: len(entry[1]["query_ids"]))
    batch = []
    for item in ordered:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def run_mode(model, items, records, pairs, mode, tokenizer, args, eos_ids, progress=None,
             stop_event=None, writer=None):
    if progress is not None:
        progress.set_description(f"Accuracy {mode}")
    size_arg = {"base": "base_batch_size", "lora_full": "lora_batch_size",
                "lora_minikv": "minikv_batch_size"}[mode]
    batch_size = getattr(args, size_arg, None)
    batch_size = batch_size or getattr(args, "batch_size", 1)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = next(iter(sorted(eos_ids)), 0)
    for batch in iter_batches(model, items, batch_size):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Evaluation stopped because another worker failed")
        started = time.perf_counter()
        outputs = generate_batch(model, [item["query_ids"] for _, item in batch], pairs,
                                 args.max_new_tokens, eos_ids, pad_id)
        for (index, item), ids in zip(batch, outputs):
            records[index]["predictions"][mode] = dict(
                text=tokenizer.decode(ids, skip_special_tokens=True), token_ids=ids,
                output_tokens=len(ids), stopped_on_eos=bool(ids and ids[-1] in eos_ids))
        if writer is not None:
            writer.commit(mode, [(index, records[index]) for index, _ in batch])
        if progress is not None:
            progress.set_postfix(sample=batch[-1][1]["sample"]["id"],
                                 output_tokens=sum(map(len, outputs)), batch=len(batch),
                                 last_s=f"{time.perf_counter() - started:.2f}", refresh=False)
            progress.update(len(batch))


def prepare_base_prefixes(model, groups, args, progress, checks):
    prefixes = {}
    progress.set_description("Accuracy base: prepare prefix/check cache")
    for key, items in groups.items():
        first = items[0][1]
        output = forward_tokens(model, first["prefix_ids"], fresh_cache(model))
        prefixes[key] = snapshot_cache(output.past_key_values, device=None)
        del output
        check = check_cache(model, first, prefixes[key], args.cache_check_atol, args.cache_check_rtol)
        checks.append(dict(prefix_key=key, mode="base", **check))
    return prefixes


def prepare_lora_prefixes(model, groups, args, progress, checks, collect_minikv=True):
    prefixes, minis = {}, {}
    progress.set_description("Accuracy lora_full: prepare prefix/check cache")
    for key, items in groups.items():
        first = items[0][1]
        capture = capture_minikv(model, keep_on_device=True) if collect_minikv else nullcontext(None)
        with capture as captured:
            output = forward_tokens(model, first["prefix_ids"], fresh_cache(model))
        prefixes[key] = snapshot_cache(output.past_key_values, device=None)
        if captured is not None:
            minis[key] = captured
        del output
        check = check_cache(model, first, prefixes[key], args.cache_check_atol, args.cache_check_rtol)
        checks.append(dict(prefix_key=key, mode="lora_full", **check))
    return prefixes, minis


def prepare_independent_minikv_sources(model, groups, progress, stats):
    """Re-prefill clean sources after both reference evaluations have finished."""
    bases, minis = {}, {}
    device = input_device(model)
    for key, items in groups.items():
        prefix = items[0][1]["prefix_ids"]
        progress.set_description("Accuracy lora_minikv: prefill BaseKV with LoRA disabled")
        with timed_stage(stats, "minikv_base_prefix_prepare", device), model.disable_adapter():
            output = forward_tokens(model, prefix, fresh_cache(model))
            bases[key] = snapshot_cache(output.past_key_values, device=None)
            del output
        progress.set_description("Accuracy lora_minikv: prefill adapted prefix/capture MiniKV")
        with timed_stage(stats, "minikv_adapted_prefix_prepare", device):
            with capture_minikv(model, keep_on_device=True) as captured:
                output = forward_tokens(model, prefix, fresh_cache(model))
            minis[key] = captured
            # This native adapted cache is not an input to reconstruction.
            del output
    return bases, minis


def evaluate_native(model, groups, records, prefixes, mode, tokenizer, args, eos_ids,
                    progress, control, release=False, writer=None):
    for key, items in groups.items():
        run_mode(model, items, records, prefixes[key], mode, tokenizer, args,
                 eos_ids, progress, control.stop, writer)
        if release:
            del prefixes[key]


def evaluate_mini_queue(model, first_job, queue, worker, bases, minis, records, tokenizer,
                        args, eos_ids, progress, control, stats, writer=None):
    reconstructed = {}
    job = first_job
    with timed_stage(stats, "minikv_stage", input_device(model)):
        while job is not None:
            control.check()
            key, batch = job
            if key not in reconstructed:
                with timed_stage(stats, "reconstruct", input_device(model)):
                    reconstructed[key] = reconstruct_prefix(
                        model, bases.pop(key), minis.pop(key), keep_on_device=True)
            with timed_stage(stats, "minikv_generation", input_device(model)):
                run_mode(model, batch, records, reconstructed[key], "lora_minikv", tokenizer,
                         args, eos_ids, progress, control.stop, writer)
            queue.complete(worker, batch)
            job = queue.claim(worker, args.minikv_batch_size or args.batch_size)
    bases.clear()
    minis.clear()


def evaluate_dual(model, load_model, groups, records, tokenizer, args, eos_ids, loading, progresses, writer=None):
    base_progress, full_progress, mini_progress = progresses
    control = EvaluationControl()
    base_ready, mini_ready = threading.Event(), threading.Event()
    base_copied, mini_copied = threading.Event(), threading.Event()
    shared = {}
    queue = MiniKVQueue(groups)
    stats = {worker: dict(device=device, stage_seconds={}, received_bytes={}, minikv_requests=0)
             for worker, device in (("base_worker", args.device), ("lora_worker", args.lora_device))}
    checks = {"base_worker": [], "lora_worker": []}

    def base_worker():
        worker = "base_worker"
        local_stats = stats[worker]
        with torch.inference_mode():
            with timed_stage(local_stats, "prefix_prepare_and_check", args.device):
                bases = prepare_base_prefixes(model, groups, args, base_progress, checks[worker])
            shared["bases"] = bases
            base_ready.set()
            control.wait(mini_ready)
            with timed_stage(local_stats, "receive_minikv", args.device):
                minis = {key: place_minikv(model, mini) for key, mini in shared["minis"].items()}
            local_stats["received_bytes"]["minikv_with_rope"] = tensor_bytes(minis)
            mini_copied.set()
            with timed_stage(local_stats, "base_generation", args.device):
                evaluate_native(model, groups, records, bases, "base", tokenizer, args,
                                eos_ids, base_progress, control, writer=writer)
            control.wait(base_copied)
            first_job = queue.claim(worker, args.minikv_batch_size or args.batch_size)
            if first_job is not None:
                base_progress.set_description("Accuracy base: complete; attaching LoRA for MiniKV")
                with timed_stage(local_stats, "attach_lora", args.device):
                    adapted = PeftModel.from_pretrained(
                        model, args.lora, is_trainable=False, **loading).eval()
                evaluate_mini_queue(adapted, first_job, queue, worker, bases, minis,
                                    records, tokenizer, args, eos_ids, mini_progress, control, local_stats, writer)
            bases.clear()
            minis.clear()

    def lora_worker():
        worker = "lora_worker"
        local_stats = stats[worker]
        with torch.inference_mode():
            full_progress.set_description("Accuracy lora_full: load model/adapter")
            with timed_stage(local_stats, "load_model_and_lora", args.lora_device):
                adapted = PeftModel.from_pretrained(
                    load_model(args.lora_device), args.lora, is_trainable=False, **loading).eval()
            with timed_stage(local_stats, "prefix_prepare_and_check", args.lora_device):
                full, minis = prepare_lora_prefixes(adapted, groups, args, full_progress, checks[worker])
            shared["minis"] = minis
            mini_ready.set()
            control.wait(base_ready)
            with timed_stage(local_stats, "receive_basekv", args.lora_device):
                bases = {key: place_prefix(adapted, pairs) for key, pairs in shared["bases"].items()}
            local_stats["received_bytes"]["basekv"] = tensor_bytes(bases)
            base_copied.set()
            with timed_stage(local_stats, "full_generation", args.lora_device):
                evaluate_native(adapted, groups, records, full, "lora_full", tokenizer, args,
                                eos_ids, full_progress, control, release=True, writer=writer)
            control.wait(mini_copied)
            first_job = queue.claim(worker, args.minikv_batch_size or args.batch_size)
            if first_job is not None:
                evaluate_mini_queue(adapted, first_job, queue, worker, bases, minis,
                                    records, tokenizer, args, eos_ids, mini_progress, control, local_stats, writer)
            bases.clear()
            minis.clear()

    def guarded(function):
        try:
            return function()
        except BaseException as error:
            control.fail(error)
            raise

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(guarded, base_worker)
    try:
        guarded(lora_worker)
        future.result()
        queue.validate()
    except BaseException:
        if control.error is not None:
            raise control.error
        raise
    finally:
        control.stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
    for worker in stats:
        stats[worker]["minikv_requests"] = queue.worker_counts[worker]
    return checks["base_worker"] + checks["lora_worker"], stats


def evaluate_single(model, groups, records, tokenizer, args, eos_ids, loading, progresses, writer=None):
    base_progress, full_progress, mini_progress = progresses
    checks, control = [], EvaluationControl()
    worker = "single_worker"
    stats = dict(device=args.device, stage_seconds={}, received_bytes={}, minikv_requests=0)
    independent = bool(getattr(args, "minikv_independent_prefill", False))
    stats["prefix_source_policy"] = "independent third-stage prefill" if independent else "reuse reference-stage sources"
    stats["prefix_prepare_forwards"] = dict(base=len(groups), lora_full=len(groups),
                                            minikv_base=len(groups) if independent else 0,
                                            minikv_adapted=len(groups) if independent else 0)
    stats["cache_consistency_checks"] = "additional direct/split forwards, separate from prefix preparation counts"
    device = input_device(model)
    with timed_stage(stats, "base_prefix_prepare_and_check", device):
        bases = prepare_base_prefixes(model, groups, args, base_progress, checks)
    with timed_stage(stats, "base_generation", device):
        evaluate_native(model, groups, records, bases, "base", tokenizer, args,
                        eos_ids, base_progress, control, release=independent, writer=writer)
    with timed_stage(stats, "attach_lora", device):
        model = PeftModel.from_pretrained(model, args.lora, is_trainable=False, **loading).eval()
    with timed_stage(stats, "lora_prefix_prepare_and_check", device):
        full, minis = prepare_lora_prefixes(model, groups, args, full_progress, checks,
                                          collect_minikv=not independent)
    with timed_stage(stats, "full_generation", device):
        evaluate_native(model, groups, records, full, "lora_full", tokenizer, args,
                        eos_ids, full_progress, control, release=True, writer=writer)
    if independent:
        bases, minis = prepare_independent_minikv_sources(model, groups, mini_progress, stats)
    queue = MiniKVQueue(groups)
    evaluate_mini_queue(model, queue.claim(worker, args.minikv_batch_size or args.batch_size),
                        queue, worker, bases, minis, records, tokenizer, args, eos_ids,
                        mini_progress, control, stats, writer)
    queue.validate()
    stats["minikv_requests"] = queue.worker_counts[worker]
    return checks, {worker: stats}


def run(args):
    run_started = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    scorer = load_scorer(args.scorer)
    loading = dict(local_files_only=args.local_files_only)
    config = AutoConfig.from_pretrained(args.model, **loading)
    if config.model_type not in ("llama", "mistral", "qwen2"):
        raise ValueError(f"MiniKV extraction currently supports llama/mistral/qwen2, got {config.model_type}")
    adapter_config = PeftConfig.from_pretrained(args.lora, **loading)
    if adapter_config.peft_type != "LORA" or getattr(adapter_config, "use_dora", False):
        raise ValueError("This experiment requires an additive LoRA checkpoint")
    if getattr(adapter_config, "bias", "none") != "none":
        raise ValueError("LoRA checkpoints that modify base biases are not supported")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, **loading)
    limits = [int(value) for value in (getattr(config, "max_position_embeddings", None),
                                      getattr(config, "max_sequence_length", None)) if value]
    if not limits:
        raise ValueError("The model config does not declare a context length")
    dataset_report = None
    if args.dataset:
        samples, dataset_report = load_dataset_samples(args, tokenizer, min(limits))
    else:
        samples = load_samples(args.data, args.max_samples)
    prepared = prepare_inputs(samples, tokenizer, args.max_new_tokens, min(limits),
                              not args.no_add_special_tokens)
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    weight_options = {} if args.weight_format == "auto" else {
        "use_safetensors": args.weight_format == "safetensors"}
    def load_model(device):
        return AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=dtype,
            device_map="auto" if args.device_map == "auto" else {"": device},
            attn_implementation=args.attn_implementation, **weight_options, **loading).eval()

    model = load_model(args.device)
    generation_config = model.generation_config
    eos = generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    groups, records = prepare_records(prepared)
    started = time.perf_counter()
    dual = args.lora_device is not None
    with torch.inference_mode(), ExitStack() as stack:
        writer = stack.enter_context(PartialResults(args.output, samples, scorer))
        progresses = [AccuracyProgress(mode, len(prepared), position) for position, mode in enumerate(MODES)]
        for progress in progresses:
            stack.callback(progress.close)
        if dual:
            checks, worker_stats = evaluate_dual(
                model, load_model, groups, records, tokenizer, args, eos_ids, loading, progresses, writer)
        else:
            checks, worker_stats = evaluate_single(
                model, groups, records, tokenizer, args, eos_ids, loading, progresses, writer)

    for record in records:
        if set(record["predictions"]) != set(MODES):
            raise RuntimeError(f"Missing evaluation modes for sample {record['id']}")

    references = [sample["references"] for sample in samples]
    scores = {mode: scorer([record["predictions"][mode]["text"] for record in records],
                           references, samples) for mode in MODES}
    deltas = {}
    for name, left, right in (("lora_minus_base", "lora_full", "base"),
                              ("minikv_minus_lora", "lora_minikv", "lora_full")):
        deltas[name] = {metric: value - scores[right][metric]
                        for metric, value in scores[left].items()
                        if isinstance(value, (int, float))
                        and isinstance(scores[right].get(metric), (int, float))}
    result = dict(
        config=vars(args), versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                        peft=peft.__version__),
        model_config=config.to_dict(), adapter_config=adapter_config.to_dict(),
        experiment=dict(samples=len(samples), unique_prefixes=len(groups),
                        elapsed_seconds=time.perf_counter() - started,
                        total_wall_seconds=time.perf_counter() - run_started,
                        minikv_source="K/V LoRA A outputs from the adapted prefix forward",
                        decoding="Transformers model.generate, greedy, EOS or max_new_tokens"
                                 if config.model_type == "qwen2" else
                                 "greedy argmax, EOS or max_new_tokens; no logits processors",
                        generation_engine="transformers.generate" if config.model_type == "qwen2" else "legacy_decode_loop",
                        cache_storage="Device-resident prefix templates; independent batched DynamicCache",
                        base_batch_size=args.base_batch_size or args.batch_size,
                        lora_batch_size=args.lora_batch_size or args.batch_size,
                        minikv_batch_size=args.minikv_batch_size or args.batch_size,
                        parallel_base_lora=dual,
                        minikv_independent_prefill=bool(getattr(args, "minikv_independent_prefill", False)),
                        schedule="staged references, shared MiniKV queue" if dual else "sequential stages",
                        workers=worker_stats,
                        prediction_journal=str(writer.journal_path),
                        partial_results=str(writer.summary_path),
                        scores_scale="default EM/F1 are fractions in [0, 1]"),
        dataset_preparation=dataset_report, cache_checks=checks, scores=scores,
        score_deltas=deltas, predictions=records)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(samples=len(samples), unique_prefixes=len(groups), scores=scores,
                          output=str(destination)), indent=2, ensure_ascii=False))
    return result


def json_default(value):
    if isinstance(value, (set, tuple)):
        return sorted(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path or Hugging Face ID")
    parser.add_argument("--lora", required=True, help="Standalone PEFT LoRA path or Hugging Face ID")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Prepared JSONL samples")
    source.add_argument("--dataset", choices=DATASETS, help="Build a fixed train few-shot prefix and test queries")
    parser.add_argument("--output", required=True, help="JSON results, configuration and predictions")
    parser.add_argument("--tokenizer", help="Optional tokenizer path; defaults to the base model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-device", help="Separate GPU for full LoRA; each GPU joins MiniKV after its reference run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-batch-size", type=int, help="Override Base batch size")
    parser.add_argument("--lora-batch-size", type=int, help="Override full LoRA batch size")
    parser.add_argument("--minikv-batch-size", type=int, help="MiniKV batch size on each GPU")
    parser.add_argument("--device-map", choices=["auto"], help="Use Accelerate for a model spanning GPUs")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--weight-format", choices=["auto", "bin", "safetensors"], default="auto",
                        help="Base weight format; use bin if cached safetensors shards are incomplete")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates all input samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scorer", help="module:function or file.py:function; default generic English EM/token F1")
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--minikv-independent-prefill", action="store_true",
                        help="Single-device only: prepare fresh BaseKV/MiniKV after Base and LoRA Full finish")
    parser.add_argument("--cache-check-atol", type=float,
                        help="Default by logits dtype: FP16 0.03, BF16 0.15, FP32 0.0001")
    parser.add_argument("--cache-check-rtol", type=float, default=0.005)
    add_dataset_arguments(parser)
    args = parser.parse_args(argv)
    validate_dataset_arguments(parser, args)
    if args.minikv_independent_prefill and args.device_map:
        parser.error("minikv-independent-prefill requires a single explicit device, not device-map auto")
    if any(value is not None and value <= 0 for value in
           (args.batch_size, args.base_batch_size, args.lora_batch_size, args.minikv_batch_size)):
        parser.error("batch sizes must be positive")
    if args.lora_device:
        if args.minikv_independent_prefill:
            parser.error("minikv-independent-prefill requires single-device sequential evaluation")
        if args.device_map:
            parser.error("--lora-device cannot be combined with --device-map")
        base_device, lora_device = torch.device(args.device), torch.device(args.lora_device)
        if base_device.type != "cuda" or lora_device.type != "cuda":
            parser.error("--lora-device requires two CUDA devices")
        if (base_device.index or 0) == (lora_device.index or 0):
            parser.error("Base and LoRA must use different GPUs in dual-device mode")
    if args.max_new_tokens <= 0 or args.max_samples < 0:
        parser.error("max-new-tokens must be positive; max-samples must be nonnegative")
    if (args.cache_check_atol is not None and args.cache_check_atol < 0) or args.cache_check_rtol < 0:
        parser.error("cache check tolerances must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
