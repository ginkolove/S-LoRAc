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
SPEC = importlib.util.spec_from_file_location("minikv_lora_groups_profile", ROOT / "try/profile_minikv_lora_groups_13b.py")
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class GroupProfileTests(unittest.TestCase):
    def test_groups_match_requested_examples_and_are_reproducible(self):
        groups = profile.make_groups()
        self.assertEqual(len(groups), 10)
        self.assertEqual(groups[0].adapter_ids, (1, 2, 3, 4, 5, 6))
        self.assertEqual(groups[1].adapter_ids, (10, 11, 12, 13, 14, 15))
        self.assertEqual(groups, profile.make_groups(seed=42))
        self.assertNotEqual(groups[2:], profile.make_groups(seed=43)[2:])
        self.assertEqual(len({g.adapter_ids for g in groups}), 10)
        for group in groups:
            self.assertEqual(len(set(group.adapter_ids)), 6)
            self.assertTrue(all(1 <= i <= 50 for i in group.adapter_ids))

    def test_impossible_combinations_rejected(self):
        for kwargs in (dict(count=0), dict(pool_size=14), dict(random_groups=-1),
                       dict(pool_size=10, count=1, random_groups=9)):
            with self.assertRaises(ValueError):
                profile.make_groups(**kwargs)

    def test_workspace_does_not_scale_with_source_pool_count(self):
        capacity = profile.resident_pool_slots(10000, 40, 16, 50, 6)
        self.assertEqual(capacity, 4 * 16 * 50 + 1500 + 128)
        larger_pool = profile.resident_pool_slots(10000, 40, 16, 100, 6)
        self.assertEqual(larger_pool - capacity, 4 * 16 * 50)
        self.assertEqual(profile.resident_pool_slots(10001, 40, 16, 50, 6), capacity + 1)

    def make_cpu_selection_fixture(self, rank=8):
        fixture = profile.GroupFixture.__new__(profile.GroupFixture)
        fixture.args = profile.SimpleNamespace(pool_adapters=50, unique_loras=6)
        fixture.rank = rank
        names = ["adapter_%03d" % i for i in range(1, 51)]
        fixture.static = profile.baseline.LowRAStaticPrefixStore(
            3, names, dict.fromkeys(names, rank), 2, 1, 1, 16, device="cpu")
        fixture.locations = torch.arange(18).reshape(6, 3)
        fixture.infer = profile.SimpleNamespace(a_start=torch.arange(50) * 4 * rank,
            a_len=torch.full((50,), 4 * rank), a_loc=torch.arange(50 * 4 * rank),
            a_scaling=torch.full((50,), 0.5, dtype=torch.float16), idx_map=dict(zip(names, range(50))))
        return fixture

    def test_select_only_switches_indices_in_the_same_pool(self):
        for rank in (8, 16, 32, 64):
            fixture = self.make_cpu_selection_fixture(rank)
            source, weight_locations, output_locations = fixture.static, fixture.infer.a_loc, fixture.locations
            groups = profile.make_groups()
            with patch.object(profile.baseline.InferAdapter, "load_adapters", side_effect=AssertionError("unexpected H2D")):
                fixture.select(groups[0])
                first_metadata = fixture.metadata
                fixture.select(groups[1])
                torch.testing.assert_close(fixture.metadata.rank_starts, torch.arange(9, 15) * rank)
                torch.testing.assert_close(fixture.metadata.k_b_locs[0], torch.arange(9 * 4 * rank + rank, 9 * 4 * rank + 2 * rank))
                torch.testing.assert_close(fixture.metadata.v_b_locs[0], torch.arange(9 * 4 * rank + 2 * rank, 9 * 4 * rank + 3 * rank))
                self.assertEqual(fixture.metadata.k_b_locs.shape, (6, rank))
                fixture.select(groups[0])
            self.assertIs(fixture.static, source)
            self.assertIs(fixture.infer.a_loc, weight_locations)
            self.assertIs(fixture.locations, output_locations)
            self.assertIs(fixture.metadata.prefix_locs, output_locations)
            for name, tensor in vars(first_metadata).items():
                torch.testing.assert_close(getattr(fixture.metadata, name), tensor)

    def test_invalid_ids_cannot_change_selection(self):
        fixture = self.make_cpu_selection_fixture()
        for ids in ((0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 51), (1, 2, 3, 4, 5, 5), (1, 2)):
            with self.assertRaises(ValueError):
                fixture.select(profile.Group("bad", ids))

    def test_production_reconstruction_and_oracle_are_inherited(self):
        for name in ("layer", "sweep", "validate", "time_sweeps", "close", "check_guards"):
            self.assertIs(getattr(profile.GroupFixture, name), getattr(profile.baseline.Fixture, name))

    def example_report(self):
        def case(name, means):
            rounds = []
            for i, mean in enumerate(means):
                samples = [mean - 1, mean + 1]
                rounds.append(dict(round=i + 1, cuda_ms=samples, wall_ms=[v + 0.1 for v in samples],
                                   cuda=profile.baseline.summarize(samples)))
            return dict(rank=8, group=name, adapter_ids=[1, 2, 3, 4, 5, 6],
                        validation={"max_abs_error": 0.0001}, rounds=rounds)
        return dict(config={"num_hidden_layers": 40}, arguments=dict(ranks=[8], repeats=2),
                    groups=[dict(name="first"), dict(name="second")],
                    cases=dict(r8_first=case("first", [10, 20]), r8_second=case("second", [20, 40])))

    def test_summary_separates_between_group_and_within_group_variation(self):
        report = self.example_report()
        groups, ranks = profile.aggregate(report)
        self.assertEqual(groups[0]["cuda_mean_ms"], 15)
        self.assertEqual(groups[0]["mean_layer_ms"], 15 / 40)
        self.assertEqual(groups[0]["samples"], 4)
        self.assertAlmostEqual(groups[0]["within_group_round_cv_pct"], 100 / 3)
        self.assertAlmostEqual(groups[0]["round_normalized_ratio"], 2 / 3)
        self.assertEqual(ranks[0]["max_min_ratio"], 2)
        self.assertTrue(ranks[0]["complete"])
        self.assertEqual(ranks[0]["fastest_group"], "first")
        report["cases"]["r8_second"]["rounds"].pop()
        self.assertFalse(profile.aggregate(report)[1][0]["complete"])

    def test_common_round_drift_is_not_mistaken_for_group_difference(self):
        report = self.example_report()
        report["cases"]["r8_second"]["rounds"] = report["cases"]["r8_first"]["rounds"]
        _, ranks = profile.aggregate(report)
        self.assertEqual(ranks[0]["between_group_cv_pct"], 0)
        self.assertEqual(ranks[0]["normalized_between_group_cv_pct"], 0)
        self.assertGreater(ranks[0]["median_within_group_round_cv_pct"], 0)

    def test_reports_are_incremental_and_parseable(self):
        report = self.example_report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            profile.write_report(output, report)
            self.assertEqual(json.loads((output / "result.json").read_text()), report)
            with (output / "groups.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with (output / "rank_comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_list_groups_is_cpu_only(self):
        with patch.object(torch.cuda, "set_device", side_effect=AssertionError("unexpected CUDA")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            profile.main(profile.parse_args(["--list-groups"]))
        self.assertEqual(len(output.getvalue().splitlines()), 10)

    def test_no_process_occupancy_guard(self):
        with patch("subprocess.run", side_effect=AssertionError("unexpected process check")), \
                patch.object(torch.cuda, "set_device", side_effect=RuntimeError("SETUP_REACHED")):
            with self.assertRaisesRegex(RuntimeError, "SETUP_REACHED"):
                profile.main(profile.parse_args(["--output-dir", "unused"]))

    def test_invalid_cli(self):
        for argv in (["--list-groups", "--iterations", "0"], ["--list-groups", "--ranks", "8", "8"],
                     ["--list-groups", "--pool-adapters", "10"], []):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                profile.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
