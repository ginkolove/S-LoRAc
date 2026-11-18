import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("minikv_rank_profile", ROOT / "try/profile_minikv_rank_13b.py")
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class RankProfileTest(unittest.TestCase):
    def test_default_matrix(self):
        cases = profile.make_cases()
        self.assertEqual(len(cases), 29)
        self.assertEqual(len({case.name for case in cases}), 29)
        homogeneous = [case for case in cases if not case.name.startswith("mix_") and not case.diagnostic_only]
        self.assertEqual({(case.ranks[0], len(case.ranks)) for case in homogeneous},
                         {(r, u) for r in (8, 16, 32, 64) for u in (1, 2, 4, 8, 16)})

    def test_mixed_ranks_and_cli_selection(self):
        expected = {
            "mix_4r16_4r32": (16,) * 4 + (32,) * 4,
            "mix_5r16_3r32": (16,) * 5 + (32,) * 3,
            "mix_7r16_1r64": (16,) * 7 + (64,),
            "mix_6r16_1r32_1r64": (16,) * 6 + (32, 64),
            "mix_6r16_2r64": (16,) * 6 + (64,) * 2,
        }
        cases = {case.name: case for case in profile.make_cases()}
        selected_groups = []
        for name, ranks in expected.items():
            case = cases[name]
            self.assertEqual(case.ranks, ranks)
            self.assertEqual(case.metadata_width or max(case.ranks), max(ranks))
            self.assertFalse(case.diagnostic_only)
            source_ranks, selected = profile.source_layout(case, 50, 42)
            self.assertEqual(tuple(source_ranks[i] for i in selected), ranks)
            self.assertTrue(all(rank == 16 for i, rank in enumerate(source_ranks) if i not in selected))
            selected_groups.append(selected)
        self.assertTrue(all(selected == selected_groups[0] for selected in selected_groups))
        args = profile.parse_args(["--list-cases", "--rank-loop", "actual",
                                   "--block-r", "16", "--cases", *expected])
        with patch.object(torch.cuda, "set_device", side_effect=AssertionError("unexpected CUDA")), \
                contextlib.redirect_stdout(io.StringIO()) as captured:
            profile.main(args)
        self.assertEqual({line.split()[0] for line in captured.getvalue().splitlines()}, set(expected))

    def test_padding_control_uses_identical_sources(self):
        cases = {case.name: case for case in profile.make_cases()}
        normal, padded = cases["r8_u8"], cases["r8_u8_padded64"]
        self.assertEqual(normal.ranks, padded.ranks)
        self.assertEqual(profile.source_layout(normal, 50, 42), profile.source_layout(padded, 50, 42))
        self.assertTrue(padded.diagnostic_only)
        self.assertEqual(padded.metadata_width, 64)

    def test_source_selection_is_seeded_and_unique(self):
        case = next(case for case in profile.make_cases() if case.name == "mix_7r8_1r64")
        ranks, selected = profile.source_layout(case, 50, 42)
        self.assertEqual(len(ranks), 50)
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(tuple(ranks[i] for i in selected), case.ranks)
        self.assertEqual(profile.source_layout(case, 50, 42), (ranks, selected))
        self.assertNotEqual(profile.source_layout(case, 50, 43)[1], selected)
        with self.assertRaises(ValueError):
            profile.source_layout(case, 7, 42)

    def test_workspace_budget_is_one_layer_not_full_prefix(self):
        ranks = (16,) * 8
        slots = profile.pool_slots(10000, 40, ranks)
        self.assertEqual(slots, 4 * sum(ranks) + 2000 + 128)
        self.assertEqual(profile.pool_slots(10001, 40, ranks), slots + 1)
        workspace_bytes = 2000 * 40 * 5120 * 2 * 2
        self.assertEqual(workspace_bytes, 8 * 10000 * 5120 * 2 * 2)

    def test_real_allocator_workspace_is_disjoint_from_weight_slots(self):
        memory = profile.MemoryAllocator(64, 64, torch.float16, 1, 16, 4,
                                         dynamic_layer_workspace=True, device="cpu")
        held = memory.alloc(16)
        memory.free(held[::2])
        guards = held[1::2]
        weights = memory.alloc(16)
        locations = memory.alloc_layer_workspace(33)
        self.assertEqual(memory.workspace_token_equiv, 9)
        self.assertEqual(locations.unique().numel(), 33)
        self.assertFalse(set((locations % 64).tolist()) & set(weights.tolist() + guards.tolist()))
        self.assertEqual(memory.prefix_workspace_k.shape, (256, 1, 16))
        memory.prefix_workspace_k[locations] = 1
        for layer in range(4):
            local = locations[(locations // 64) == layer] % 64
            self.assertTrue(bool((memory.key_buffer[layer][local] == 1).all()))
        memory.free_layer_workspace()
        memory.free(weights)
        memory.free(guards)
        self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def test_actual_cpu_weight_packer_preserves_original_hf_b(self):
        rank, hidden = 8, 32
        weights = {}
        for module in ("q", "k", "v", "o"):
            stem = "base_model.model.model.layers.0.self_attn.%s_proj" % module
            weights[stem + ".lora_A.weight"] = torch.arange(rank * hidden).half().reshape(rank, hidden)
            weights[stem + ".lora_B.weight"] = torch.arange(rank * hidden).half().reshape(hidden, rank)
        layer = profile.LoraLayerWeight(0, 0, 1, {"r": rank},
                                       {"hidden_size": hidden, "num_attention_heads": 2})
        # Pinning is irrelevant to CPU layout verification and initializes CUDA on some builds.
        with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
            layer.load_hf_weights_cpu(weights)
        for offset, module in ((rank, "k"), (2 * rank, "v")):
            original = weights["base_model.model.model.layers.0.self_attn.%s_proj.lora_B.weight" % module]
            packed = layer.w_combined_home[1, offset:offset + rank]
            torch.testing.assert_close(packed.reshape(hidden, rank), original, atol=0, rtol=0)

    def test_widen_metadata_preserves_rank_and_addresses(self):
        metadata = profile.runtime.ReconstructionMetadata(
            torch.arange(6).view(2, 3), torch.arange(16).view(2, 8),
            torch.arange(16, 32).view(2, 8), torch.tensor([0, 8]),
            torch.tensor([8, 8]), torch.tensor([0.5, 0.5]))
        widened = profile.widen_metadata(metadata, 64)
        self.assertEqual(widened.k_b_locs.shape, (2, 64))
        torch.testing.assert_close(widened.k_b_locs[:, :8], metadata.k_b_locs)
        torch.testing.assert_close(widened.v_b_locs[:, :8], metadata.v_b_locs)
        self.assertEqual(widened.k_b_locs[:, 8:].count_nonzero().item(), 0)
        self.assertIs(widened.rank_lens, metadata.rank_lens)
        self.assertIs(widened.rank_starts, metadata.rank_starts)
        self.assertIs(widened.prefix_locs, metadata.prefix_locs)
        self.assertIs(profile.widen_metadata(metadata, 8), metadata)
        with self.assertRaises(ValueError):
            profile.widen_metadata(metadata, 4)

    def test_reference_value_formula(self):
        mini = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float16)
        b = torch.tensor([[2., 0.], [0., 4.], [1., 1.], [-2., 1.]], dtype=torch.float16)
        base = torch.ones(2, 1, 4, dtype=torch.float16)
        expected = torch.tensor([[2., 5., 2.5, 1.], [0., 7., 2., 3.5]])
        torch.testing.assert_close(profile.reference_chunk(base, mini, b, 0.5), expected)

    def test_reference_key_rounds_before_rope_then_adds_base(self):
        mini = torch.tensor([[0.3, 0.7]], dtype=torch.float16)
        b = torch.tensor([[0.123, 0.931], [0.72, -0.88], [0.54, 0.56], [0.33, 0.91]], dtype=torch.float16)
        base = torch.tensor([[[2., 3., 4., 5.]]], dtype=torch.float16)
        cos = torch.tensor([[0.5, 0.75]], dtype=torch.float16)
        sin = torch.tensor([[0.25, -0.5]], dtype=torch.float16)
        delta = ((mini.float() @ b.float().T) * 0.5).half().float()[0]
        expected = torch.tensor([[[2 + delta[0] * 0.5 - delta[2] * 0.25,
                                   3 + delta[1] * 0.75 + delta[3] * 0.5,
                                   4 + delta[0] * 0.25 + delta[2] * 0.5,
                                   5 - delta[1] * 0.5 + delta[3] * 0.75]]])
        torch.testing.assert_close(profile.reference_chunk(base, mini, b, 0.5, cos, sin), expected,
                                   atol=0, rtol=0)

    def test_sweep_reuses_output_and_visits_all_layers(self):
        fixture = profile.Fixture.__new__(profile.Fixture)
        fixture.layers = 40
        fixture.static, fixture.infer, fixture.metadata = object(), object(), object()
        fixture.memory = profile.SimpleNamespace(prefix_workspace_k=object(), prefix_workspace_v=object())
        fixture.args = profile.SimpleNamespace(workers=1296)
        with patch.object(profile.runtime, "reconstruct_prefix_layer_bounded") as kernel:
            fixture.sweep()
        self.assertEqual(kernel.call_count, 40)
        self.assertEqual([call.args[3] for call in kernel.call_args_list], list(range(40)))
        for call in kernel.call_args_list:
            self.assertIs(call.args[4], fixture.memory.prefix_workspace_k)
            self.assertIs(call.args[5], fixture.memory.prefix_workspace_v)
            self.assertEqual(call.kwargs, {"workers": 1296})

    def test_rank_loop_and_tile_options_reach_the_serving_wrapper(self):
        fixture = profile.Fixture.__new__(profile.Fixture)
        fixture.static, fixture.infer, fixture.metadata = object(), object(), object()
        fixture.memory = profile.SimpleNamespace(prefix_workspace_k=object(), prefix_workspace_v=object())
        for rank_loop, block_r in (("actual", 16), ("actual", 32), ("max", 32)):
            fixture.args = profile.SimpleNamespace(workers=1296, rank_loop=rank_loop, block_r=block_r)
            with patch.object(profile.runtime, "reconstruct_prefix_layer_bounded") as kernel:
                fixture.layer(3)
            self.assertEqual(kernel.call_args.kwargs,
                             dict(workers=1296, rank_loop=rank_loop, block_r=block_r))

    def test_layer_wrapper_forwards_options_to_both_k_and_v(self):
        static = profile.SimpleNamespace(prefix_len=3, tp_k_head_num=1, head_dim=16,
            kv_embed_dim_k=16, kv_embed_dim_v=16, base_k=[torch.zeros(3, 1, 16)],
            base_v=[torch.zeros(3, 1, 16)], mini_k=[torch.zeros(3, 8)],
            mini_v=[torch.zeros(3, 8)], key_rope_cos=torch.ones(3, 8),
            key_rope_sin=torch.zeros(3, 8))
        infer = profile.SimpleNamespace(mem_manager=profile.SimpleNamespace(value_buffer=[torch.zeros(32, 1, 16)]))
        metadata = profile.runtime.ReconstructionMetadata(
            torch.arange(3).view(1, 3), torch.zeros(1, 8, dtype=torch.long),
            torch.ones(1, 8, dtype=torch.long), torch.tensor([0]), torch.tensor([8]), torch.tensor([0.5]))
        out_k, out_v = torch.empty(3, 1, 16), torch.empty(3, 1, 16)
        with patch.object(profile.runtime, "lowra_reconstruct_bounded") as kernel, \
                patch.object(profile.runtime, "rotate_delta_k_add_base") as rope:
            profile.runtime.reconstruct_prefix_layer_bounded(
                static, infer, metadata, 0, out_k, out_v, workers=1296, block_r=32, rank_loop="actual")
        self.assertEqual(kernel.call_count, 2)
        for call in kernel.call_args_list:
            self.assertEqual(call.kwargs["launch_config"], (32, 128, 32, 4))
            self.assertEqual(call.kwargs["rank_loop"], "actual")
            self.assertTrue(call.kwargs["b_packed"])
        self.assertFalse(kernel.call_args_list[0].kwargs["add_base"])
        self.assertTrue(kernel.call_args_list[1].kwargs["add_base"])
        rope.assert_called_once()

    def test_default_cli_preserves_max_rank_and_tile16(self):
        args = profile.parse_args(["--list-cases"])
        self.assertEqual((args.rank_loop, args.block_r), ("max", 16))
        args = profile.parse_args(["--list-cases", "--rank-loop", "actual", "--block-r", "32"])
        self.assertEqual((args.rank_loop, args.block_r), ("actual", 32))

    def test_bad_loop_or_tile_is_rejected_before_kernel_launch(self):
        from slora.common.lowra_reconstruct_bounded import lowra_reconstruct_bounded

        with self.assertRaisesRegex(ValueError, "rank_loop"):
            lowra_reconstruct_bounded(*([None] * 10), rank_loop="unknown")
        with self.assertRaisesRegex(ValueError, "block_r"):
            profile.runtime.reconstruct_prefix_layer_bounded(*([None] * 6), block_r=8)

    def test_summary_and_report(self):
        stats = profile.summarize([1., 2., 3., 4.])
        self.assertEqual(stats["mean_ms"], 2.5)
        self.assertAlmostEqual(stats["p95_ms"], 3.85)
        rounds = [dict(cuda_ms=[40., 80.], wall_ms=[42., 82.], cuda=profile.summarize([40., 80.])),
                  dict(cuda_ms=[80., 120.], wall_ms=[82., 122.], cuda=profile.summarize([80., 120.]))]
        report = dict(config={"num_hidden_layers": 40}, cases={"r8_u1": dict(
            ranks=[8], diagnostic_only=False, metadata_width=8, rounds=rounds,
            memory={"workspace_bytes": 1234}, validation={"max_abs_error": 0.001})})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            profile.write_report(output, report)
            self.assertEqual(json.loads((output / "result.json").read_text()), report)
            with (output / "summary.csv").open() as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertEqual(float(row["cuda_mean_ms"]), 80.)
            self.assertEqual(float(row["mean_layer_ms"]), 2.)
            self.assertEqual(float(row["round_mean_stdev_ms"]), 20.)
            self.assertEqual((row["rank_loop"], row["block_r"]), ("max", "16"))
            report["arguments"] = dict(rank_loop="actual", block_r=32)
            profile.write_report(output, report)
            with (output / "summary.csv").open() as handle:
                row = list(csv.DictReader(handle))[0]
            self.assertEqual((row["rank_loop"], row["block_r"]), ("actual", "32"))
            self.assertFalse((output / "result.json.tmp").exists())

    def test_list_cases_does_not_initialize_cuda(self):
        args = profile.parse_args(["--list-cases"])
        with patch.object(torch.cuda, "set_device", side_effect=AssertionError("unexpected CUDA")), \
                contextlib.redirect_stdout(io.StringIO()) as captured:
            profile.main(args)
        self.assertEqual(len(captured.getvalue().splitlines()), 29)

    def test_execution_does_not_inspect_gpu_processes(self):
        args = profile.parse_args(["--output-dir", "unused"])
        with patch("subprocess.run", side_effect=AssertionError("unexpected process inspection")), \
                patch.object(torch.cuda, "set_device", side_effect=RuntimeError("GPU_SETUP_REACHED")) as setup:
            with self.assertRaisesRegex(RuntimeError, "GPU_SETUP_REACHED"):
                profile.main(args)
            setup.assert_called_once_with(0)

    def test_cli_rejects_invalid_runs(self):
        for argv in (["--list-cases", "--iterations", "0"],
                     ["--list-cases", "--unique-loras", "0"],
                     ["--list-cases", "--block-r", "8"],
                     ["--list-cases", "--rank-loop", "unknown"],
                     ["--list-cases", "--ranks", "8", "8"], []):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                profile.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
