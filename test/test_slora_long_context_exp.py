"""CPU-only regression checks for long input configuration and HTTP validation."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.slora_long_context_exp import build_payload, decode_event, parse_args, send_request


def event(count=64, finished=True, prompt_tokens=10000, logprob=-0.1):
    return dict(token=dict(id=10, logprob=logprob), prompt_tokens=prompt_tokens,
                count_output_tokens=count, finished=finished)


class FakeContent:
    def __init__(self, frames):
        self.lines = iter(frames)

    async def readline(self):
        return next(self.lines, b"")


class FakeResponse:
    status = 200

    def __init__(self, frames):
        self.content = FakeContent(frames)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeSession:
    def __init__(self, frames):
        self.frames = frames
        self.payload = None

    def post(self, url, json, timeout):
        self.payload = json
        return FakeResponse(self.frames)


def frames(*events):
    lines = []
    for value in events:
        lines.extend([("data:" + json.dumps(value) + "\n").encode(), b"\n"])
    return lines


class LongContextTests(unittest.TestCase):
    def test_payload_is_exact_and_does_not_inject_a_cached_prefix(self):
        payload = build_payload(10000, 64, 10, "adapter", "request")
        self.assertEqual(payload["inputs"], [10] * 10000)
        self.assertEqual(payload["parameters"]["max_new_tokens"], 64)
        self.assertTrue(payload["parameters"]["ignore_eos"])
        self.assertNotIn("lowra_lengths", payload)

    def test_default_lengths(self):
        args = parse_args([])
        self.assertEqual((args.input_len, args.output_len, args.num_requests), (10000, 64, 1))

    def test_counts_may_coalesce_multiple_output_tokens(self):
        self.assertEqual(decode_event([json.dumps(event())], 10000, 64, 1), (10000, 64, True))

    def test_missing_prompt_count_and_truncated_input_fail(self):
        missing = event()
        del missing["prompt_tokens"]
        for value in (missing, event(prompt_tokens=2048), event(prompt_tokens=True)):
            with self.assertRaises(ValueError):
                decode_event([json.dumps(value)], 10000, 64, 0)

    def test_incomplete_nonfinite_and_invalid_counts_fail(self):
        for value in (event(count=63), event(count=65), event(count=True),
                      event(logprob=float("nan")), event(finished=-1)):
            with self.assertRaises(ValueError):
                decode_event([json.dumps(value)], 10000, 64, 0)
        with self.assertRaises(ValueError):
            decode_event([json.dumps(event(count=1, finished=False))], 10000, 64, 1)

    def test_worker_restores_location_table_length_before_cuda_initialization(self):
        from slora.common.configs.config import setting
        from slora.server.router.model_infer.model_rpc import ModelRpcServer

        params = SimpleNamespace(enable_lowra=False, max_req_total_len=10064, nccl_port=28767)
        with patch.dict(setting, max_req_total_len=5120), \
                patch("torch.distributed.init_process_group", side_effect=RuntimeError("stop before CUDA")):
            with self.assertRaisesRegex(RuntimeError, "stop before CUDA"):
                ModelRpcServer().exposed_init_model(0, 1, "model", [], 16384, "HF", [], params, None)
            self.assertEqual(setting["max_req_total_len"], 10064)

    def test_existing_rope_covers_long_prefill_and_decode_without_wrapping(self):
        import torch
        from slora.models.llama.model import LlamaTpPartModel
        from slora.mprophet.model_config import get_config_json

        model = object.__new__(LlamaTpPartModel)
        model.config = get_config_json("huggyllama/llama-7b")
        model.head_dim_ = model.config["hidden_size"] // model.config["num_attention_heads"]
        with patch.dict(os.environ, SLORA_NTK_ALPHA="1"), \
                patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
            model._init_to_get_rotary()
        self.assertEqual(model.config["max_sequence_length"], 2048)
        self.assertGreaterEqual(model._cos_cached.shape[0], 10064)
        positions = torch.tensor([9999, 10062])
        actual = model._cos_cached[positions, 0]
        torch.testing.assert_close(actual, torch.cos(positions.float()).half(), atol=0, rtol=0)

    def test_long_location_table_and_slot_lifecycle_on_cpu(self):
        import torch
        from slora.common.configs.config import setting
        from slora.common.infer_utils import init_bloc
        from slora.common.mem_allocator import MemoryAllocator
        from slora.server.router.model_infer.infer_batch import InferBatch

        memory = MemoryAllocator(16384, 16384, torch.float16, 1, 16, 1, device="cpu")

        def cpu_factory(factory):
            def allocate(*args, **kwargs):
                kwargs["device"] = "cpu"
                return factory(*args, **kwargs)
            return allocate

        for iteration in range(2):
            request = dict(request_id=str(iteration), adapter_dir="adapter",
                           input_id=[10] * 10000, sampling_param={})
            with patch.dict(setting, max_req_total_len=10064), \
                    patch("torch.empty", cpu_factory(torch.empty)), \
                    patch("torch.zeros", cpu_factory(torch.zeros)), \
                    patch("torch.tensor", cpu_factory(torch.tensor)):
                batch = InferBatch.init_batch(iteration, [request], torch.float16, "cpu", memory, 32000)
            self.assertEqual(batch.nopad_b_loc.shape, (1, 10076))
            self.assertEqual(batch.input_ids.numel(), 10000)
            self.assertEqual(batch.nopad_total_token_num, 10000)
            init_bloc(batch.nopad_b_loc, batch.nopad_b_seq_len,
                      batch.nopad_max_len_in_batch, memory.alloc(10000))

            # Match the location/length updates after prefill and each decode.
            batch.nopad_max_len_in_batch += 1
            batch.nopad_b_seq_len += 1
            for _ in range(63):
                allocation, _, _ = memory.alloc_contiguous(1)
                batch.nopad_b_loc[:, batch.nopad_max_len_in_batch - 1] = allocation
                batch.nopad_max_len_in_batch += 1
                batch.nopad_b_seq_len += 1
            self.assertEqual(memory.can_use_mem_size, 16384 - 10063)
            self.assertEqual(batch._free_indices_for_req(0).numel(), 10063)
            batch.free_self()
            self.assertEqual(memory.can_use_mem_size, 16384)
            self.assertTrue(torch.all(memory.mem_state == 1).item())


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_requires_exact_input_and_complete_decode(self):
        session = FakeSession(frames(event(count=1, finished=False), event()))
        result = await send_request(session, parse_args([]), "request")
        self.assertTrue(result["success"])
        self.assertEqual((result["server_prompt_tokens"], result["output_tokens"]), (10000, 64))
        self.assertEqual(result["stream_events"], 2)
        self.assertEqual(len(session.payload["inputs"]), 10000)
        self.assertGreaterEqual(result["latency_s"], result["ttft_s"])

    async def test_stream_closed_early_is_failure_even_with_http_200(self):
        session = FakeSession(frames(event(count=1, finished=False)))
        result = await send_request(session, parse_args([]), "request")
        self.assertFalse(result["success"])
        self.assertEqual(result["http_status"], 200)
        self.assertIn("without completing", result["error"])

    async def test_truncated_frame_fails(self):
        session = FakeSession([("data:" + json.dumps(event()) + "\n").encode()])
        result = await send_request(session, parse_args([]), "request")
        self.assertFalse(result["success"])
        self.assertIn("truncated SSE", result["error"])

    async def test_tokens_after_completion_fail(self):
        session = FakeSession(frames(event(), event()))
        result = await send_request(session, parse_args([]), "request")
        self.assertFalse(result["success"])
        self.assertIn("after completion", result["error"])


if __name__ == "__main__":
    unittest.main()
