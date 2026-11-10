import unittest

import torch
from transformers import LlamaConfig

from benchmarks.kv_cache_memory import cache_geometry, measure_cache


class KVCacheMemoryTests(unittest.TestCase):
    def test_original_llama_sizes(self):
        for hidden, heads, layers, expected in (
            (4096, 32, 32, 5242880000),
            (5120, 40, 40, 8192000000),
            (6656, 52, 60, 15974400000),
        ):
            with self.subTest(hidden=hidden):
                config = LlamaConfig(hidden_size=hidden, num_attention_heads=heads,
                                     num_hidden_layers=layers)
                g = cache_geometry(config, 10000, 1, torch.float16)
                self.assertEqual(g["full_kv_bytes"], expected)
                self.assertEqual(g["head_dim"], 128)

    def test_gqa_uses_kv_not_query_heads(self):
        config = LlamaConfig(hidden_size=32, num_attention_heads=8,
                             num_key_value_heads=2, num_hidden_layers=3)
        g = cache_geometry(config, 10, 2, torch.float32)
        self.assertEqual(g["full_kv_bytes"], 2 * 2 * 10 * 2 * 4 * 3 * 4)

    def test_cache_payload_and_layer_projection(self):
        config = LlamaConfig(hidden_size=32, num_attention_heads=4,
                             num_key_value_heads=2, num_hidden_layers=3)
        g = cache_geometry(config, 7, 2, torch.float16)
        for mode in ("full", "layer"):
            with self.subTest(mode=mode):
                result = measure_cache(config, g, torch.float16, torch.device("cpu"), mode, 0)
                self.assertTrue(result["storage_matches_formula"])
                self.assertEqual(result["projected_full_kv_bytes"], g["full_kv_bytes"])
                self.assertEqual(result["full_cache_actually_allocated"], mode == "full")
        self.assertEqual(config.num_hidden_layers, 3)

    def test_reject_invalid_lengths_and_sliding_cache(self):
        config = LlamaConfig()
        with self.assertRaises(ValueError):
            cache_geometry(config, 0, 1, torch.float16)
        config.sliding_window = 4096
        with self.assertRaises(ValueError):
            cache_geometry(config, 10000, 1, torch.float16)


if __name__ == "__main__":
    unittest.main()
