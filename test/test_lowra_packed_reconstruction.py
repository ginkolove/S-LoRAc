"""Check reconstruction against HF-format weights through the real S-LoRA packer."""

from types import SimpleNamespace
import unittest

import torch

from slora.common.lowra_layer_runtime import (
    LowRALayerRuntime, prepare_reconstruction_metadata,
    reconstruct_prefix_layer, reconstruct_prefix_layer_fused_kv, reconstruct_prefix_layer_bounded,
)
from slora.common.lowra_static import LowRAStaticPrefixStore
from slora.common.mem_allocator import MemoryAllocator
from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
from slora.server.router.model_infer.infer_adapter import InferAdapter
from slora.common.lowra_retained import RetainedPrefixKVCache, RetainedLayerRuntime
from slora.models.lowra.prefix_attention import context_attention_fwd_with_prefix
from slora.models.lowra.decode_attention import token_attention_fwd_with_prefix


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PackedReconstructionTests(unittest.TestCase):
    def fixture(self, prefix=129, layers=2, hidden=1024, ranks=None, seed=93):
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        heads = hidden // 128
        ranks = ranks or {"a": 8, "b": 16}
        config = dict(hidden_size=hidden, num_attention_heads=heads, num_hidden_layers=layers)
        capacity = max(1024, 6 * sum(ranks.values()) + len(ranks) * prefix + 256)
        memory = MemoryAllocator(capacity, capacity, torch.float16, heads, 128, layers,
                                 dynamic_layer_workspace=True)
        adapters, original = [], {}
        for directory, rank in ranks.items():
            weights = []
            original[directory] = []
            for layer in range(layers):
                hf = {}
                for module in ("q", "k", "v", "o"):
                    root = "base_model.model.model.layers.%s.self_attn.%s_proj" % (layer, module)
                    hf[root + ".lora_A.weight"] = torch.randn(rank, hidden, dtype=torch.float16) * 0.1
                    hf[root + ".lora_B.weight"] = torch.randn(hidden, rank, dtype=torch.float16) * 0.1
                weight = LoraLayerWeight(layer, 0, 1, {"r": rank}, config)
                weight.load_hf_weights_cpu(hf)
                weights.append(weight)
                original[directory].append(hf)
            adapters.append(SimpleNamespace(lora_dir=directory, r=rank, scaling=0.5,
                                            network_config=config, layers=weights))
        # Force non-contiguous adapter slots, including within one packed B matrix.
        held = memory.alloc(192)
        memory.free(held[::2])
        held = held[1::2]
        infer = InferAdapter.init(memory, None)
        infer.load_adapters(adapters)
        static = LowRAStaticPrefixStore(prefix, list(ranks), ranks, layers, heads, heads, 128)
        for layer in range(layers):
            static.base_k[layer].normal_(std=0.1)
            static.base_v[layer].normal_(std=0.1)
            x = torch.randn(prefix, hidden, device="cuda", dtype=torch.float16) * 0.1
            for directory in ranks:
                start, end = static.get_lora_rank_range_by_dir(directory)
                for module, mini in (("k", static.mini_k[layer]), ("v", static.mini_v[layer])):
                    key = "base_model.model.model.layers.%s.self_attn.%s_proj.lora_A.weight" % (layer, module)
                    mini[:, start:end] = (x.float() @ original[directory][layer][key].cuda().float().T).half()
        return static, memory, infer, original, held

    def expected(self, static, original, directory, layer, module):
        start, end = static.get_lora_rank_range_by_dir(directory)
        mini = (static.mini_k if module == "k" else static.mini_v)[layer][:, start:end]
        base = (static.base_k if module == "k" else static.base_v)[layer].view(static.prefix_len, -1)
        key = "base_model.model.model.layers.%s.self_attn.%s_proj.lora_B.weight" % (layer, module)
        delta = (mini.float() @ original[directory][layer][key].cuda().float().T) * 0.5
        if module == "k" and static.key_rope_cos is not None:
            first, second = delta.half().float().view(static.prefix_len, static.tp_k_head_num, static.head_dim).chunk(2, -1)
            cos, sin = static.key_rope_cos.float()[:, None], static.key_rope_sin.float()[:, None]
            delta = torch.cat((first * cos - second * sin, first * sin + second * cos), -1).view_as(delta)
        return base.float() + delta

    def bind_key_rope(self, static):
        positions = torch.arange(static.prefix_len, device="cuda", dtype=torch.float32)[:, None]
        inv = 10000.0 ** (-torch.arange(0, static.head_dim, 2, device="cuda").float() / static.head_dim)
        angles = positions * inv[None]
        static.bind_key_rope(angles.cos().half(), angles.sin().half())

    def test_raw_mini_k_is_rotated_before_adding_post_rope_base(self):
        fixture = self.fixture(prefix=129, layers=2, hidden=4096)
        self.bind_key_rope(fixture[0])
        self.check_reconstruction(*fixture)

    def test_real_packing_matches_hf_formula_for_every_reconstruction_kernel(self):
        self.check_reconstruction(*self.fixture())

    def test_packed_width4096_twenty_adapters_and_mixed_rank_group_boundary(self):
        for ranks in ({"lora%s" % i: 16 for i in range(20)},
                      {"lora%s" % i: (8, 16, 32, 64)[i % 4] for i in range(9)}):
            self.check_reconstruction(*self.fixture(prefix=257, layers=1, hidden=4096, ranks=ranks))

    def test_packed_long_prefix_with_a_masked_final_tile(self):
        self.check_reconstruction(*self.fixture(prefix=4609, layers=1, hidden=4096))

    def test_llama7b_and_13b_rank64_long_prefix_with_rope(self):
        for hidden in (4096, 5120):
            with self.subTest(hidden=hidden):
                fixture = self.fixture(prefix=10000, layers=2, hidden=hidden,
                                       ranks={"a": 64, "b": 64})
                self.bind_key_rope(fixture[0])
                self.check_reconstruction(*fixture)

    def check_reconstruction(self, static, memory, infer, original, held):
        dirs = list(reversed(static.lora_dirs))
        locations = memory.alloc_layer_workspace(len(dirs) * static.prefix_len).view(len(dirs), -1)
        metadata = prepare_reconstruction_metadata(static, dirs, infer, locations)
        for reconstruct in (reconstruct_prefix_layer, reconstruct_prefix_layer_fused_kv,
                            reconstruct_prefix_layer_bounded):
            for layer in range(static.layer_num):
                reconstruct(static, infer, metadata, layer,
                            memory.prefix_workspace_k, memory.prefix_workspace_v)
                for row, directory in enumerate(dirs):
                    for module, output in (("k", memory.prefix_workspace_k), ("v", memory.prefix_workspace_v)):
                        actual = output[locations[row]].view(static.prefix_len, -1).float()
                        expected = self.expected(static, original, directory, layer, module)
                        torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.002)
        memory.free_layer_workspace()
        infer.offload_adapters([])
        memory.free(held)
        self.assertEqual(memory.can_use_mem_size, memory.tot_size)

    def test_existing_bgmv_expand_agrees_with_original_hf_B(self):
        from slora._kernels import dispatch_bgmv

        static, memory, infer, original, held = self.fixture()
        for layer in range(static.layer_num):
            for module, qkvo in (("k", 1), ("v", 2)):
                x = torch.randn(2, 16, device="cuda", dtype=torch.float16) * 0.1
                output = torch.zeros(2, 1024, device="cuda", dtype=torch.float16)
                ids = torch.tensor([infer.idx_map["b"], infer.idx_map["a"]], device="cuda")
                dispatch_bgmv(output, x, memory.value_buffer[layer], infer.a_start, infer.a_len,
                              infer.a_loc, ids, qkvo, infer.a_scaling)
                for row, directory in enumerate(("b", "a")):
                    rank = static.lora_ranks[directory]
                    key = "base_model.model.model.layers.%s.self_attn.%s_proj.lora_B.weight" % (layer, module)
                    expected = x[row, :rank].float() @ original[directory][layer][key].cuda().float().T * 0.5
                    torch.testing.assert_close(output[row].float(), expected, atol=0.0001, rtol=0.002)
        infer.offload_adapters([])
        memory.free(held)

    def test_negative_control_detects_the_old_slot_row_interpretation(self):
        from slora.common.lowra_reconstruct_bounded import lowra_reconstruct_bounded

        static, memory, infer, original, held = self.fixture(layers=1)
        locations = memory.alloc_layer_workspace(static.prefix_len).view(1, -1)
        meta = prepare_reconstruction_metadata(static, ["b"], infer, locations)
        lowra_reconstruct_bounded(static.base_k[0].view(static.prefix_len, -1), static.mini_k[0],
            memory.value_buffer[0].view(-1, 1024), memory.prefix_workspace_k.view(-1, 1024),
            locations, meta.k_b_locs, meta.rank_starts, meta.rank_lens, meta.scalings,
            workers=1296, launch_config=(32, 128, 16, 4), b_packed=False)
        actual = memory.prefix_workspace_k[locations[0]].view(static.prefix_len, -1).float()
        expected = self.expected(static, original, "b", 0, "k")
        self.assertGreater((actual - expected).abs().max().item(), 0.05)
        memory.free_layer_workspace()
        infer.offload_adapters([])
        memory.free(held)

    def test_attention_consumes_the_correct_layer_and_preserves_private_KV(self):
        self.check_attention(False)

    def test_attention_with_key_rope_serial_parallel_and_hot_retention(self):
        self.check_attention(True)

    def check_attention(self, key_rope):
        static, memory, infer, original, held = self.fixture(layers=3)
        if key_rope:
            self.bind_key_rope(static)
        dirs, lengths = ["a", "b", "a"], (3, 17, 65)
        total, maximum = sum(lengths), max(lengths)
        private_locs = memory.alloc(total)
        private = []
        for layer in range(static.layer_num):
            k = torch.randn(total, 8, 128, device="cuda", dtype=torch.float16) * 0.2
            v = torch.randn_like(k) * 0.2
            memory.key_buffer[layer][private_locs] = k
            memory.value_buffer[layer][private_locs] = v
            private.append((k, v))
        sizes = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, lengths[0], lengths[0] + lengths[1]], device="cuda", dtype=torch.int32)
        b_loc = torch.full((3, maximum), -1, device="cuda", dtype=torch.long)
        cursor = 0
        for row, length in enumerate(lengths):
            b_loc[row, maximum - length:] = private_locs[cursor:cursor + length]
            cursor += length
        queries = [torch.randn(total, 8, 128, device="cuda", dtype=torch.float16) * 0.5
                   for _ in range(static.layer_num)]
        expected_outputs = []
        last = torch.tensor([lengths[0] - 1, sum(lengths[:2]) - 1, total - 1], device="cuda")
        for layer in range(static.layer_num):
            expected = torch.empty_like(queries[layer], dtype=torch.float32)
            cursor = 0
            for row, (directory, length) in enumerate(zip(dirs, lengths)):
                pk = self.expected(static, original, directory, layer, "k").half().view(-1, 8, 128)
                pv = self.expected(static, original, directory, layer, "v").half().view(-1, 8, 128)
                k = torch.cat((pk, private[layer][0][cursor:cursor + length])).float()
                v = torch.cat((pv, private[layer][1][cursor:cursor + length])).float()
                q = queries[layer][cursor:cursor + length].float()
                scores = torch.einsum("qhd,khd->hqk", q, k) / 128 ** 0.5
                allowed = torch.arange(static.prefix_len + length, device="cuda")[None] <= (
                    static.prefix_len + torch.arange(length, device="cuda")[:, None])
                expected[cursor:cursor + length] = torch.einsum(
                    "hqk,khd->qhd", scores.masked_fill(~allowed[None], -float("inf")).softmax(-1), v)
                cursor += length
            expected_outputs.append(expected)
        for mode in ("layer_serial", "layer_prefetch", "retained"):
            cache = None
            if mode == "retained":
                cache = RetainedPrefixKVCache(static.prefix_len, static.layer_num, memory, retain_top_k=1)
                cache.acquire_many(dirs)
                runtime = RetainedLayerRuntime(static, memory, cache, reconstruction_kernel="bounded",
                                               reconstruction_workers=1296)
            else:
                runtime = LowRALayerRuntime(static, memory, mode, reconstruction_kernel="bounded",
                                            reconstruction_workers=1296)
            observed = []
            for decode in (False, True):
                frame = runtime.prepare(dirs, infer)
                for layer in range(static.layer_num):
                    frame.before_layer(layer)
                    frame.before_attention(layer)
                    pk, pv = frame.buffers(layer)
                    locations = frame.prefix_locations(layer)
                    if decode:
                        actual = token_attention_fwd_with_prefix(
                            queries[layer][last], pk, pv, memory.key_buffer[layer], memory.value_buffer[layer],
                            locations, b_loc, sizes, maximum)
                        expected = expected_outputs[layer][last]
                    else:
                        actual = torch.empty_like(queries[layer])
                        context_attention_fwd_with_prefix(queries[layer], *private[layer], actual,
                            starts, sizes, maximum, locations, static.prefix_len, pk, pv)
                        expected = expected_outputs[layer]
                    observed.append((actual, expected))
                    frame.after_attention(layer)
                    work = torch.ones(128, 128, device="cuda", dtype=torch.float16)
                    torch.mm(work, work)
                frame.finish()
            torch.cuda.synchronize()
            for actual, expected in observed:
                torch.testing.assert_close(actual.float(), expected, atol=0.001, rtol=0.01)
            for layer in range(static.layer_num):
                torch.testing.assert_close(memory.key_buffer[layer][private_locs], private[layer][0], atol=0, rtol=0)
                torch.testing.assert_close(memory.value_buffer[layer][private_locs], private[layer][1], atol=0, rtol=0)
            if cache is not None:
                self.assertGreater(cache.retained_layer_reuses, 0)
                cache.release_many(dirs)
        memory.free(private_locs)
        infer.offload_adapters([])
        memory.free(held)
        self.assertEqual(memory.can_use_mem_size, memory.tot_size)


if __name__ == "__main__":
    unittest.main()
