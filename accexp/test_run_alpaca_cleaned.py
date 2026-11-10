"""Sampling must be reproducible across models and preserve prepared prompts."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from run_alpaca_cleaned import mark_partial, sample_jsonl


class SamplingTests(unittest.TestCase):
    def test_fixed_random_subset_preserves_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [dict(id=str(i), prefix="fixed few-shot prefix", query=str(i), references=[str(i)])
                    for i in range(50)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            reports = [sample_jsonl(source, root / f"sample{i}.jsonl", 10, 42) for i in range(2)]
            indices = sorted(random.Random(42).sample(range(50), 10))
            self.assertEqual(reports[0]["indices"], indices)
            self.assertNotEqual(indices, list(range(10)))
            self.assertEqual(reports[0]["sample_ids"], reports[1]["sample_ids"])
            self.assertEqual(reports[0]["output_sha256"], reports[1]["output_sha256"])
            selected = [json.loads(line) for line in (root / "sample0.jsonl").read_text().splitlines()]
            self.assertEqual(selected, [rows[i] for i in indices])
            self.assertEqual(reports[0]["unique_prefixes"], 1)

    def test_sample_size_cannot_silently_shrink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps(dict(id="0", prefix="prefix")) + "\n")
            with self.assertRaises(ValueError):
                sample_jsonl(source, root / "sample.jsonl", 1000, 42)

    def test_termination_keeps_partial_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            path = destination.with_suffix(".partial.json")
            scores = {"lora_full": {"exact_match": 0.25}}
            path.write_text(json.dumps(dict(state="running", scores=scores)))
            mark_partial(destination, -15)
            result = json.loads(path.read_text())
            self.assertEqual(result["state"], "interrupted")
            self.assertEqual(result["scores"], scores)


if __name__ == "__main__":
    unittest.main()
