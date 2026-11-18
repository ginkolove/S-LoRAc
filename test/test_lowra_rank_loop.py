"""CUDA regressions for per-adapter rank loops with real packed B and RoPE."""

from dataclasses import replace
import unittest

import torch

from slora.common.lowra_layer_runtime import (
    prepare_reconstruction_metadata, reconstruct_prefix_layer_bounded,
)
import test_lowra_packed_reconstruction as packing


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ActualRankReconstructionTests(unittest.TestCase):
    def check_case(self, prefix, ranks, pad_width=None):
        helper = packing.PackedReconstructionTests()
        static, memory, infer, original, held = helper.fixture(
            prefix=prefix, layers=2, hidden=5120, ranks=ranks)
        helper.bind_key_rope(static)
        dirs = list(reversed(static.lora_dirs))
        locations = memory.alloc_layer_workspace(len(dirs) * prefix).view(len(dirs), -1)
        metadata = prepare_reconstruction_metadata(static, dirs, infer, locations)
        if pad_width is not None:
            def pad(tensor):
                # Invalid padding must never be dereferenced, even in MaxRank mode.
                result = tensor.new_full((len(dirs), pad_width), -1)
                result[:, :tensor.shape[1]] = tensor
                return result
            metadata = replace(metadata, k_b_locs=pad(metadata.k_b_locs), v_b_locs=pad(metadata.v_b_locs))
        for layer in range(2):
            memory.key_buffer[layer][held] = 0.625
            memory.value_buffer[layer][held] = -0.75
        try:
            for block_r in (16, 32):
                for layer in range(2):
                    reference = None
                    for rank_loop in ("max", "actual"):
                        with self.subTest(prefix=prefix, block_r=block_r, layer=layer, rank_loop=rank_loop):
                            for output in (memory.prefix_workspace_k, memory.prefix_workspace_v):
                                output[locations.reshape(-1)] = float("nan")
                            reconstruct_prefix_layer_bounded(
                                static, infer, metadata, layer, memory.prefix_workspace_k,
                                memory.prefix_workspace_v, workers=1296,
                                block_r=block_r, rank_loop=rank_loop)
                            current = []
                            for row, directory in enumerate(dirs):
                                for module, output in (("k", memory.prefix_workspace_k),
                                                       ("v", memory.prefix_workspace_v)):
                                    actual = output[locations[row]].view(prefix, -1)
                                    expected = helper.expected(static, original, directory, layer, module)
                                    torch.testing.assert_close(actual.float(), expected, atol=5e-4, rtol=2e-3)
                                    current.append(actual)
                                    rank = ranks[directory]
                                    b_locs = metadata.k_b_locs if module == "k" else metadata.v_b_locs
                                    packed = memory.value_buffer[layer][b_locs[row, :rank]].reshape(5120, rank)
                                    key = "base_model.model.model.layers.%d.self_attn.%s_proj.lora_B.weight" % (layer, module)
                                    torch.testing.assert_close(packed, original[directory][layer][key].cuda(), atol=0, rtol=0)
                            if reference is None:
                                reference = current
                            else:
                                for actual, expected in zip(current, reference):
                                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                            for i in range(2):
                                self.assertTrue(bool((memory.key_buffer[i][held] == 0.625).all()))
                                self.assertTrue(bool((memory.value_buffer[i][held] == -0.75).all()))
        finally:
            memory.free_layer_workspace()
            infer.offload_adapters([])
            memory.free(held)
        self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def test_mixed_ranks_group_boundary_and_tail_tokens(self):
        self.check_case(129, {"lora%d" % i: (8, 16, 32, 64)[i % 4] for i in range(9)})

    def test_13b_10k_prefix_mixed_ranks(self):
        self.check_case(10000, {"a": 8, "b": 16, "c": 32, "d": 64})

    def test_actual_rank_does_not_change_padded_B_index_stride(self):
        self.check_case(129, {"a": 8, "b": 8}, pad_width=64)


if __name__ == "__main__":
    unittest.main()
