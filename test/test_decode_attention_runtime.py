import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeRuntimeLengthTests(unittest.TestCase):
    def test_mixed_lengths_match_reference_and_do_not_compile_each_integer(self):
        from slora.models.lowra.decode_attention import (
            _prefix_decode_reduce, _prefix_decode_scores, token_attention_fwd_with_prefix,
        )
        torch.manual_seed(42)
        device = "cuda"
        batch, heads, kv_heads, dim, prefix = 3, 4, 2, 16, 7
        q = torch.randn(batch, heads, dim, device=device, dtype=torch.float16)
        prefix_k = torch.randn(64, kv_heads, dim, device=device, dtype=torch.float16)
        prefix_v = torch.randn_like(prefix_k)
        private_k = torch.randn(256, kv_heads, dim, device=device, dtype=torch.float16)
        private_v = torch.randn_like(private_k)
        prefix_loc = torch.randint(0, 64, (batch, prefix), device=device)
        b_loc = torch.zeros(batch, 1024, device=device, dtype=torch.long)
        variants = []
        for maximum in (19, 20, 65, 66, 511, 512):
            lengths = [maximum, maximum - 3, maximum // 2]
            lens = torch.tensor(lengths, device=device, dtype=torch.int32)
            expected = []
            for index, length in enumerate(lengths):
                indices = torch.randint(0, 256, (length,), device=device)
                b_loc[index, maximum - length:maximum] = indices
                k = torch.cat((prefix_k[prefix_loc[index]], private_k[indices])).float()
                v = torch.cat((prefix_v[prefix_loc[index]], private_v[indices])).float()
                k = k.repeat_interleave(heads // kv_heads, dim=1)
                v = v.repeat_interleave(heads // kv_heads, dim=1)
                score = torch.einsum("hd,shd->hs", q[index].float(), k) * dim ** -0.5
                expected.append(torch.einsum("hs,shd->hd", score.softmax(-1), v))
            actual = token_attention_fwd_with_prefix(
                q, prefix_k, prefix_v, private_k, private_v, prefix_loc, b_loc, lens, maximum)
            torch.testing.assert_close(actual.float(), torch.stack(expected), atol=2e-3, rtol=2e-3)
            variants.append(tuple(sum(len(entries) for entries in kernel.cache.values())
                                  for kernel in (_prefix_decode_scores, _prefix_decode_reduce)))
        self.assertTrue(all(value == variants[0] for value in variants), variants)


if __name__ == "__main__":
    unittest.main()
