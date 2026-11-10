"""Offline tests with real tiny Transformers models and nonzero PEFT weights."""

import json
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (LlamaConfig, LlamaForCausalLM, MistralConfig,
                          MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast)

from eval_minikv import (EvaluationControl, MiniKVQueue, PartialResults, capture_minikv, check_cache, default_score, forward_tokens,
                         fresh_cache, generate, generate_batch, iter_batches, load_samples, parse_args,
                         prepare_inputs, reconstruct_prefix, right_align_cache, run, run_mode, snapshot_cache)


def tiny_model(kind="llama", attention="eager", device="cpu", dtype=torch.float32):
    torch.manual_seed(42)
    config_class, model_class = {"llama": (LlamaConfig, LlamaForCausalLM),
                                 "mistral": (MistralConfig, MistralForCausalLM),
                                 "qwen2": (Qwen2Config, Qwen2ForCausalLM)}[kind]
    kwargs = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
                  pad_token_id=0, attention_dropout=0.0)
    if kind == "mistral":
        kwargs["sliding_window"] = 8
    config = config_class(**kwargs)
    config._attn_implementation = attention
    model = model_class(config).to(device=device, dtype=dtype).eval()
    if kind == "qwen2":
        with torch.no_grad():
            for layer in model.model.layers:
                for name in ("q_proj", "k_proj", "v_proj"):
                    getattr(layer.self_attn, name).bias.normal_(std=0.1)
    return model


def attach_lora(model, targets=None, zero=False):
    targets = targets or ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                            lora_dropout=0, task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                if zero:
                    parameter.zero_()
                else:
                    parameter.normal_(std=0.08)
    return model.eval()


class AccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def exercise(self, kind, attention="eager", targets=None, zero=False,
                 device="cpu", dtype=torch.float32):
        model = tiny_model(kind, attention, device, dtype)
        prefix, query = list(range(3, 16)), [20, 21, 22, 23, 24]
        item = dict(full_ids=prefix + query, prefix_ids=prefix, query_ids=query,
                    sample={"id": "tiny"})
        base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        check_cache(model, item, base)
        model = attach_lora(model, targets=targets, zero=zero)
        with capture_minikv(model) as mini:
            output = forward_tokens(model, prefix, fresh_cache(model))
        full = snapshot_cache(output.past_key_values)
        check_cache(model, item, full)
        reconstructed = reconstruct_prefix(model, base, mini)
        atol = 0.003 if dtype != torch.float32 else 2e-6
        for truth, actual in zip(full[0], reconstructed[0]):
            torch.testing.assert_close(actual, truth, atol=atol, rtol=atol)
        if targets == ["q_proj", "v_proj"]:
            self.assertTrue(all(state["mini_k"] is None for state in mini))
        if zero:
            for expected, actual in zip(base, reconstructed):
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        else:
            self.assertGreater(float((full[1][1] - reconstructed[1][1]).abs().max()), 1e-7)
        originals = tuple((k.clone(), v.clone()) for k, v in reconstructed)
        generated, _ = generate(model, query, reconstructed, 4, set())
        self.assertEqual(len(generated), 4)
        for before, after in zip(originals, reconstructed):
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        split_ids, _ = generate(model, query, full, 4, set())
        direct_ids, _ = generate(model, prefix + query, (), 4, set())
        self.assertEqual(split_ids, direct_ids)
        native = model.generate(torch.tensor([prefix + query], device=device),
                                max_new_tokens=4, do_sample=False, eos_token_id=None,
                                pad_token_id=0)
        self.assertEqual(direct_ids, native[0, len(prefix + query):].tolist())

    def test_llama_nonzero_lora(self):
        self.exercise("llama")

    def test_llama_zero_lora(self):
        self.exercise("llama", zero=True)

    def test_qv_only_adapter(self):
        self.exercise("llama", targets=["q_proj", "v_proj"])

    def test_mistral_sliding_window_and_gqa(self):
        self.exercise("mistral")

    def test_mistral_sdpa(self):
        self.exercise("mistral", attention="sdpa")

    def test_qwen2_nonzero_lora_gqa_and_projection_bias(self):
        self.exercise("qwen2")

    def test_qwen2_zero_lora(self):
        self.exercise("qwen2", zero=True)

    def test_qwen2_sdpa(self):
        self.exercise("qwen2", attention="sdpa")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_half(self):
        for kind in ("llama", "mistral", "qwen2"):
            with self.subTest(model=kind):
                self.exercise(kind, attention="sdpa", device="cuda:0", dtype=torch.float16)

    def test_config_dimensions_and_per_projection_rank(self):
        config = LlamaConfig(vocab_size=64, hidden_size=48, intermediate_size=80,
                             num_hidden_layers=3, num_attention_heads=6, num_key_value_heads=3,
                             max_position_embeddings=128)
        model = LlamaForCausalLM(config).eval()
        prefix = [3, 4, 5, 6]
        base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["k_proj", "v_proj"],
                                                rank_pattern={"k_proj": 2, "v_proj": 3},
                                                task_type="CAUSAL_LM")).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.08)
        with capture_minikv(model) as captured:
            output = forward_tokens(model, prefix, fresh_cache(model))
        self.assertEqual(captured[0]["mini_k"].shape, (1, 4, 2))
        self.assertEqual(captured[0]["mini_v"].shape, (1, 4, 3))
        actual = reconstruct_prefix(model, base, captured)
        self.assertEqual(len(actual), 3)
        self.assertEqual(actual[0][0].shape, (1, 3, 4, 8))
        for expected, result in zip(snapshot_cache(output.past_key_values)[0], actual[0]):
            torch.testing.assert_close(expected, result, atol=2e-6, rtol=2e-6)

    def test_cache_check_detects_corruption(self):
        model = tiny_model()
        prefix, query = [3, 4, 5], [6, 7]
        pairs = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        corrupt = tuple((k, v + 10) for k, v in pairs)
        with self.assertRaises(RuntimeError):
            check_cache(model, dict(full_ids=prefix + query, query_ids=query, sample={"id": "bad"}), corrupt)

    def test_dataset_and_tokenization(self):
        class BoundaryTokenizer:
            def encode(self, text, **kwargs):
                return [1, 3, 4] if text == "prefix" else [1, 3, 5, 6]

        sample = dict(id="x", prefix="prefix", query="query", references=["answer"])
        prepared = prepare_inputs([sample], BoundaryTokenizer(), 4, 16)[0]
        self.assertEqual(prepared["prefix_ids"], [1, 3])
        self.assertEqual(prepared["prefix_ids"] + prepared["query_ids"], [1, 3, 5, 6])
        with self.assertRaises(ValueError):
            prepare_inputs([sample], BoundaryTokenizer(), 20, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(sample) + "\n" + json.dumps(sample) + "\n")
            with self.assertRaises(ValueError):
                load_samples(path)
        self.assertEqual(default_score(["The answer"], [["answer"]], [sample])["exact_match"], 1.0)

    def test_progress_updates_per_request_without_changing_predictions(self):
        model = tiny_model()
        prefix = [3, 4, 5]
        pairs = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        items = [(i, dict(sample={"id": str(i)}, full_ids=prefix + [6 + i], query_ids=[6 + i]))
                 for i in range(2)]
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
        args = SimpleNamespace(max_new_tokens=2)
        for mode in ("base", "lora_full", "lora_minikv"):
            with self.subTest(mode=mode):
                expected = [dict(predictions={}) for _ in items]
                actual = [dict(predictions={}) for _ in items]
                run_mode(model, items, expected, pairs, mode, tokenizer, args, set())
                progress = Mock()
                run_mode(model, items, actual, pairs, mode, tokenizer, args, set(), progress)
                self.assertEqual(actual, expected)
                progress.set_description.assert_called_once_with(f"Accuracy {mode}")
                self.assertEqual(progress.update.call_args_list, [call(1), call(1)])
                self.assertEqual(progress.set_postfix.call_count, len(items))
                for index, request_call in enumerate(progress.set_postfix.call_args_list):
                    self.assertEqual(request_call.kwargs["sample"], str(index))
                    self.assertEqual(request_call.kwargs["output_tokens"], 2)
                    self.assertGreaterEqual(float(request_call.kwargs["last_s"]), 0)

    def test_batched_generation_matches_single_and_preserves_prefix(self):
        for kind in ("llama", "mistral"):
            with self.subTest(kind=kind):
                model = tiny_model(kind, attention="sdpa")
                prefix = list(range(3, 16))
                queries = [[20], [21, 22, 23, 24], [25, 26], [27, 28, 29, 30]]
                pairs = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
                originals = tuple((k.clone(), v.clone()) for k, v in pairs)
                first_token = generate(model, queries[0], pairs, 1, set())[0][0]
                for eos in (set(), {first_token}):
                    expected = [generate(model, query, pairs, 5, eos)[0] for query in queries]
                    items = [(i, {"query_ids": query}) for i, query in enumerate(queries)]
                    actual = [None] * len(queries)
                    for batch in iter_batches(model, items, 3):
                        outputs = generate_batch(model, [item["query_ids"] for _, item in batch],
                                                 pairs, 5, eos)
                        for (i, _), ids in zip(batch, outputs):
                            actual[i] = ids
                    self.assertEqual(expected, actual)
                for before, after in zip(originals, pairs):
                    for a, b in zip(before, after):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_right_align_cache_keeps_complete_histories_and_not_templates(self):
        model = tiny_model("mistral")
        pairs = tuple((torch.arange(6).reshape(1, 1, 6, 1).float() + layer * 100,
                       torch.arange(6).reshape(1, 1, 6, 1).float() + layer * 100 + 10)
                      for layer in range(2))
        originals = tuple((key.clone(), value.clone()) for key, value in pairs)
        working = fresh_cache(model, pairs, batch_size=3)
        lengths = torch.tensor([2, 6, 4])
        right_align_cache(working, lengths)
        for original, layer in zip(pairs, working.layers):
            for expected, actual in zip(original, (layer.keys, layer.values)):
                for row, length in enumerate(lengths.tolist()):
                    torch.testing.assert_close(actual[row, :, -length:], expected[0, :, :length])
        for original, actual in zip(originals, pairs):
            for expected, observed in zip(original, actual):
                torch.testing.assert_close(expected, observed, atol=0, rtol=0)

    def sliding_batch_equivalence(self, attention, device="cpu", dtype=torch.float32):
        queries = [[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [31],
                   [32, 33, 34, 35, 36], [37, 38], [39, 40, 41, 42, 43]]

        def trace_generation(active_model, rows, pairs, eos):
            traces = []

            def save_logits(module, unused, kwargs, output):
                self.assertEqual(kwargs["attention_mask"].ndim, 2)
                logits = output.logits
                if not traces and isinstance(kwargs.get("logits_to_keep"), torch.Tensor):
                    keep = kwargs["logits_to_keep"]
                    indices = torch.searchsorted(keep, torch.tensor(
                        [len(row) - 1 for row in rows], device=keep.device))
                    logits = logits[torch.arange(len(rows), device=logits.device), indices]
                else:
                    logits = logits[:, -1]
                traces.append(logits.detach().float().cpu().clone())

            core = active_model.get_base_model() if hasattr(active_model, "get_base_model") else active_model
            handle = core.register_forward_hook(save_logits, with_kwargs=True)
            try:
                result = generate_batch(active_model, rows, pairs, 12, eos, pad_id=0)
            finally:
                handle.remove()
            return result, traces

        for prefix in ([3, 4, 5], list(range(3, 16))):
            # Exercise windows shorter/longer than the prefix and query, then decode past it.
            model = tiny_model("mistral", attention=attention, device=device, dtype=dtype)
            base = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
            adapted = attach_lora(model, targets=["q_proj", "v_proj"])
            with capture_minikv(adapted) as mini:
                full = snapshot_cache(forward_tokens(adapted, prefix, fresh_cache(adapted)).past_key_values)
            reconstructed = reconstruct_prefix(adapted, base, mini)
            for mode, pairs in (("base", base), ("lora_full", full), ("lora_minikv", reconstructed)):
                with self.subTest(attention=attention, prefix=len(prefix), mode=mode, device=device):
                    context = adapted.disable_adapter() if mode == "base" else torch.inference_mode()
                    with context:
                        templates = tuple((k.clone(), v.clone()) for k, v in pairs)
                        singles = [trace_generation(adapted, [query], pairs, set()) for query in queries]
                        actual, traces = trace_generation(adapted, queries, pairs, set())
                        self.assertEqual(actual, [rows[0] for rows, _ in singles])
                        tolerance = 0.003 if dtype != torch.float32 else 3e-6
                        for row, (_, single_trace) in enumerate(singles):
                            for expected, observed in zip(single_trace, traces):
                                torch.testing.assert_close(observed[row], expected[0],
                                                           atol=tolerance, rtol=tolerance)
                        eos = {actual[1][0]}
                        stopped, _ = trace_generation(adapted, queries, pairs, eos)
                        expected = [generate(adapted, query, pairs, 12, eos)[0] for query in queries]
                        self.assertEqual(stopped, expected)
                        self.assertEqual(len(stopped[1]), 1)
                        for before, after in zip(templates, pairs):
                            for left, right in zip(before, after):
                                torch.testing.assert_close(left, right, atol=0, rtol=0)
    def test_mixed_length_sliding_batch_logits_match_single_in_all_modes(self):
        for attention in ("eager", "sdpa"):
            self.sliding_batch_equivalence(attention)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_half_mixed_length_sliding_batch(self):
        self.sliding_batch_equivalence("sdpa", device="cuda:0", dtype=torch.float16)

    def test_mistral_batch_and_minikv_queue_fill_across_query_lengths(self):
        model = tiny_model("mistral", attention="sdpa")
        items = [(i, dict(query_ids=[3] * (i + 1))) for i in reversed(range(35))]
        batches = list(iter_batches(model, items, 32))
        self.assertEqual([len(batch) for batch in batches], [32, 3])
        self.assertEqual([len(item["query_ids"]) for _, item in batches[0]], list(range(1, 33)))
        queue = MiniKVQueue({"shared": items})
        for size in (32, 3):
            key, batch = queue.claim("single_worker", 32)
            self.assertEqual(key, "shared")
            self.assertEqual(len(batch), size)
            queue.complete("single_worker", batch)
        self.assertIsNone(queue.claim("single_worker", 32))
        queue.validate()

    def test_all_modes_use_cached_query_batches(self):
        model = tiny_model()
        prefix = [3, 4, 5]
        pairs = snapshot_cache(forward_tokens(model, prefix, fresh_cache(model)).past_key_values)
        queries = [[6, 7], [8], [9, 10, 11]]
        # full_ids deliberately invalid: production generation must use query_ids.
        items = [(i, dict(sample={"id": str(i)}, full_ids=[999], query_ids=query))
                 for i, query in enumerate(queries)]
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
        args = SimpleNamespace(max_new_tokens=3, batch_size=2)
        expected = [generate(model, query, pairs, 3, set())[0] for query in queries]
        for mode in ("base", "lora_full", "lora_minikv"):
            records = [dict(predictions={}) for _ in items]
            run_mode(model, items, records, pairs, mode, tokenizer, args, set())
            self.assertEqual([row["predictions"][mode]["token_ids"] for row in records], expected)

    def test_partial_answers_saved_before_next_batch_and_survive_failure(self):
        model = tiny_model()
        samples = [dict(id=str(i), references=["answer"]) for i in range(2)]
        items = [(i, dict(sample=samples[i], query_ids=[6 + i])) for i in range(2)]
        records = [dict(id=sample["id"], references=sample["references"], predictions={}) for sample in samples]
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "answer")
        args = SimpleNamespace(max_new_tokens=2, batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            calls = []

            def interrupted_generation(*arguments, **kwargs):
                calls.append(True)
                if len(calls) == 2:
                    partial = json.loads(destination.with_suffix(".partial.json").read_text())
                    self.assertEqual(partial["completed_requests"]["base"], 1)
                    self.assertEqual(partial["scores"]["base"]["exact_match"], 1)
                    self.assertEqual(len(destination.with_suffix(".predictions.jsonl").read_text().splitlines()), 1)
                    raise RuntimeError("interrupted second batch")
                return generate_batch(*arguments, **kwargs)

            with self.assertRaisesRegex(RuntimeError, "second batch"), \
                    PartialResults(destination, samples, default_score) as writer, \
                    patch("eval_minikv.generate_batch", side_effect=interrupted_generation):
                run_mode(model, items, records, (), "base", tokenizer, args, set(), writer=writer)
            partial = json.loads(destination.with_suffix(".partial.json").read_text())
            self.assertEqual(partial["state"], "failed")
            self.assertEqual(partial["completed_requests"]["base"], 1)
            entry = json.loads(destination.with_suffix(".predictions.jsonl").read_text())["records"][0]
            self.assertEqual(entry["predictions"]["base"]["text"], "answer")
            self.assertEqual(entry["references"], ["answer"])

    def test_partial_scores_are_thread_safe_and_paired_on_same_samples(self):
        samples = [dict(id=str(i), references=["answer"]) for i in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            with PartialResults(destination, samples, default_score) as writer:
                def save_mode(mode):
                    for i in reversed(range(20)):
                        prediction = dict(text="answer" if i % 2 else "wrong", token_ids=[3],
                                          output_tokens=1, stopped_on_eos=False)
                        writer.commit(mode, [(i, dict(id=str(i), references=["answer"],
                                                     predictions={mode: prediction}))])

                with ThreadPoolExecutor(max_workers=3) as executor:
                    for future in [executor.submit(save_mode, mode)
                                   for mode in ("base", "lora_full", "lora_minikv")]:
                        future.result(timeout=10)
            partial = json.loads(destination.with_suffix(".partial.json").read_text())
            self.assertEqual(partial["state"], "complete")
            self.assertEqual(partial["paired_lora"]["samples"], 20)
            self.assertEqual(partial["paired_lora"]["minikv_minus_lora"]["exact_match"], 0)
            for mode in ("base", "lora_full", "lora_minikv"):
                self.assertEqual(partial["completed_requests"][mode], 20)
                self.assertEqual(partial["scores"][mode], dict(exact_match=0.5, generic_token_f1=0.5))
            journal = [json.loads(line) for line in destination.with_suffix(".predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(journal), 60)
            with self.assertRaises(FileExistsError):
                PartialResults(destination, samples, default_score)

    def test_partial_paired_scores_use_intersection(self):
        samples = [dict(id=str(i), references=["answer"]) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            with PartialResults(destination, samples, default_score) as writer:
                for mode, index, text in (("lora_full", 0, "answer"), ("lora_full", 1, "wrong"),
                                          ("lora_minikv", 0, "answer"), ("lora_minikv", 2, "wrong")):
                    writer.commit(mode, [(index, dict(id=str(index), references=["answer"],
                                                     predictions={mode: dict(text=text)}))])
                partial = json.loads(destination.with_suffix(".partial.json").read_text())
                self.assertEqual(partial["paired_lora"]["samples"], 1)
                self.assertEqual(partial["paired_lora"]["scores"]["lora_full"]["exact_match"], 1)
                self.assertEqual(partial["scores"]["lora_full"]["exact_match"], 0.5)

    def test_minikv_queue_disjoint_claims_and_length_groups(self):
        groups = {str(key): [(key * 20 + i, {"query_ids": [3] * (i % 4 + 1)})
                            for i in range(20)] for key in range(2)}
        for exact in (False, True):
            with self.subTest(exact_lengths=exact):
                queue = MiniKVQueue(groups, exact_lengths=exact)
                first_claims = threading.Barrier(2)

                def consume(worker, size):
                    job = queue.claim(worker, size)
                    first_claims.wait(timeout=5)
                    while job is not None:
                        key, batch = job
                        self.assertTrue(all(index // 20 == int(key) for index, _ in batch))
                        if exact:
                            self.assertEqual(len({len(item["query_ids"]) for _, item in batch}), 1)
                        queue.complete(worker, batch)
                        job = queue.claim(worker, size)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(consume, "a", 3), executor.submit(consume, "b", 5)]
                    for future in futures:
                        future.result(timeout=10)
                queue.validate()
                self.assertEqual(len(queue.completed), 40)
                self.assertGreater(queue.worker_counts["a"], 0)
                self.assertGreater(queue.worker_counts["b"], 0)

    def test_worker_failure_cancels_waiter(self):
        control = EvaluationControl()
        started = threading.Event()

        def wait_for_unpublished_cache():
            started.set()
            control.wait(threading.Event())

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wait_for_unpublished_cache)
            self.assertTrue(started.wait(5))
            error = ValueError("prefix preparation failed")
            control.fail(error)
            with self.assertRaises(RuntimeError) as raised:
                future.result(timeout=5)
            self.assertIs(raised.exception.__cause__, error)

    def test_staged_workers_join_minikv_before_other_reference_finishes(self):
        reference = self.exercise_cli()
        for slow_mode in ("base", "lora_full"):
            with self.subTest(slow_mode=slow_mode):
                slow_started, first_mini, second_mini = (threading.Event() for _ in range(3))
                slow_finished = threading.Event()
                lock, entries = threading.Lock(), []

                def staged_mode(*args, **kwargs):
                    mode = args[4]
                    if mode == slow_mode:
                        slow_started.set()
                        self.assertTrue(first_mini.wait(10))
                        result = run_mode(*args, **kwargs)
                        slow_finished.set()
                        return result
                    if mode == "lora_minikv":
                        self.assertTrue(slow_started.wait(10))
                        with lock:
                            entries.append(threading.current_thread().name)
                            first = len(entries) == 1
                        if first:
                            self.assertFalse(slow_finished.is_set())
                            first_mini.set()
                            self.assertTrue(second_mini.wait(10))
                        else:
                            second_mini.set()
                    return run_mode(*args, **kwargs)

                with patch("eval_minikv.run_mode", side_effect=staged_mode):
                    result = self.exercise_cli(simulate_dual=True, mini_batch_size=1)
                self.assertEqual(result["experiment"]["workers"]["base_worker"]["minikv_requests"], 1)
                self.assertEqual(result["experiment"]["workers"]["lora_worker"]["minikv_requests"], 1)
                self.assertEqual(result["predictions"], reference["predictions"])

    def test_prefix_preparation_failure_exits_both_workers(self):
        for function in ("prepare_base_prefixes", "prepare_lora_prefixes"):
            with self.subTest(function=function), \
                    patch("eval_minikv." + function, side_effect=ValueError("bad prefix")):
                with self.assertRaisesRegex(ValueError, "bad prefix"):
                    self.exercise_cli(simulate_dual=True)

    def test_end_to_end_cli(self):
        self.exercise_cli()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two GPUs required")
    def test_dual_gpu_end_to_end(self):
        single = self.exercise_cli(device="cuda:0")
        dual = self.exercise_cli(device="cuda:0", lora_device="cuda:1")
        self.assertTrue(dual["experiment"]["parallel_base_lora"])
        self.assertEqual(single["predictions"], dual["predictions"])

    def exercise_cli(self, device="cpu", lora_device=None, simulate_dual=False, mini_batch_size=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, adapter_path = root / "model", root / "adapter"
            model = tiny_model()
            model.save_pretrained(model_path)
            vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                          "System": 4, "context": 5, "Question": 6, "one": 7, "two": 8}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                                bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
            tokenizer.save_pretrained(model_path)
            attach_lora(model).save_pretrained(adapter_path)
            samples = [dict(id=str(i), prefix="System context ", query="Question " + query,
                            references=["one"]) for i, query in enumerate(("one", "two"))]
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")
            scorer = root / "score.py"
            scorer.write_text("def score(predictions, references, samples):\n"
                              "    assert len(predictions) == len(references) == len(samples)\n"
                              "    return {'samples_scored': len(samples)}\n")
            cli = ["--model", str(model_path), "--lora", str(adapter_path),
                               "--data", str(data), "--output", str(root / "result.json"),
                               "--device", device, "--dtype", "float32", "--max-new-tokens", "3",
                               "--batch-size", "2", "--local-files-only", "--scorer", str(scorer) + ":score"]
            if lora_device:
                cli += ["--lora-device", lora_device]
            if mini_batch_size:
                cli += ["--minikv-batch-size", str(mini_batch_size)]
            args = parse_args(cli)
            if simulate_dual:
                args.lora_device = "cpu"
            result = run(args)
            saved = json.loads((root / "result.json").read_text())
            partial = json.loads((root / "result.partial.json").read_text())
            self.assertEqual(partial["state"], "complete")
            self.assertEqual(partial["scores"], result["scores"])
            self.assertEqual(saved["experiment"]["unique_prefixes"], 1)
            self.assertEqual(len(result["predictions"]), 2)
            self.assertEqual(set(result["scores"]), {"base", "lora_full", "lora_minikv"})
            for row in result["predictions"]:
                self.assertEqual(set(row["predictions"]), set(result["scores"]))
            for score in result["scores"].values():
                self.assertEqual(score["samples_scored"], 2)
            self.assertEqual(sum(worker["minikv_requests"] for worker in
                                 result["experiment"]["workers"].values()), 2)
            return result


if __name__ == "__main__":
    unittest.main()
