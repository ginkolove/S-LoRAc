"""CPU-only checks for matched model weights + pool budgets near 79.1 GB."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def profile(size):
    command = ('source benchmarks/llama_memory_budget.sh\n'
               'llama_memory_budget "$1"\n'
               'printf "%s %s %s %s %s %s\\n" '
               '"$LLAMA_WEIGHT_POOL_BYTES" "$LLAMA_MODEL_WEIGHT_BYTES" '
               '"$LLAMA_SLOT_BYTES" "$LLAMA_POOL_TOKENS" "$LLAMA_LAYERS" "$LLAMA_HIDDEN"')
    output = subprocess.check_output(["bash", "-euc", command, "budget", size], cwd=ROOT, text=True)
    return tuple(map(int, output.split()))


class LlamaMemoryBudgetTests(unittest.TestCase):
    def test_original_7b_13b_pools_and_matched_30b_total(self):
        totals = []
        for size, expected_tokens in (("7b", 125000), ("13b", 65000), ("30b", 8800)):
            with self.subTest(size=size):
                combined, model, slot, tokens, layers, hidden = profile(size)
                self.assertEqual(slot, 2 * 2 * layers * hidden)
                self.assertEqual(tokens, expected_tokens)
                self.assertEqual(combined, model + slot * tokens)
                self.assertGreaterEqual(combined, 79_000_000_000)
                self.assertLess(combined, 79_300_000_000)
                totals.append(combined)
        self.assertLess(max(totals) - min(totals), 300_000_000)
        self.assertLess(totals[0], totals[2])
        self.assertLess(totals[2], totals[1])

    def test_30b_capacity_matches_real_static_accounting_and_online_planner(self):
        from types import SimpleNamespace
        from slora.common.lowra_static import LowRAStaticPrefixStore
        from slora.common.prefix_modes import FULL_RECONSTRUCT
        from slora.server.router.online_req_queue import OnlineReqQueue

        _, _, _, total, layers, hidden = profile("30b")
        config = dict(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=52)
        for prefix, fits in ((4000, True), (5000, False)):
            static = LowRAStaticPrefixStore.token_equivalent_static_size_from_config(
                prefix, [16] * 50, config, 1)
            shell_static = (prefix * (hidden + 50 * 16) + hidden - 1) // hidden
            self.assertEqual(static, shell_static)
            queue = OnlineReqQueue(total - static, total, total, prefix, FULL_RECONSTRUCT, layers)
            req = SimpleNamespace(adapter_dir="a", input_len=1, output_ids=[])
            plan = queue.plan([], [req], {"a": 16})
            self.assertEqual(plan.fits, fits)
            self.assertEqual(static + plan.required_slots, static + prefix + 64 + 2)
        # 4K can admit a short query, but its longest observed private history
        # cannot fit alongside a full prefix. A 3K prefix fits that history.
        req = SimpleNamespace(adapter_dir="a", input_len=686, output_ids=[0] * 843)
        for prefix, fits in ((4000, False), (3000, True)):
            static = LowRAStaticPrefixStore.token_equivalent_static_size_from_config(prefix, [16] * 50, config, 1)
            queue = OnlineReqQueue(total - static, total, total, prefix, FULL_RECONSTRUCT, layers)
            self.assertEqual(queue.plan([req], [], {"a": 16}).fits, fits)


if __name__ == "__main__":
    unittest.main()
