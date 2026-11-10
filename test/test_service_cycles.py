import io
import json
from types import SimpleNamespace
import unittest

from benchmarks.saturation_cycles import summarize_cycles
from slora.common.service_cycles import ServiceCycleRecorder


class CycleRecorderTests(unittest.TestCase):
    def test_records_merges_tokens_and_completions_without_sampling(self):
        clock = iter((20.0,))
        output = io.StringIO()
        recorder = ServiceCycleRecorder(output, clock=lambda: next(clock))
        recorder.begin(10.0)
        recorder.admit(["a", "b"], 5)
        recorder.tokens(2)
        recorder.complete("a")
        recorder.admit(["c"], 3)
        recorder.tokens(4)
        recorder.complete("b")
        recorder.complete("c")
        recorder.finish(2)
        cycle = json.loads(output.getvalue())
        self.assertEqual(cycle["completed_request_ids"], ["a", "b", "c"])
        self.assertEqual(cycle["duration"], 10)
        self.assertEqual(cycle["generated_tokens"], 6)
        self.assertEqual(cycle["admissions"], 2)
        self.assertTrue(cycle["valid"])
        self.assertTrue(cycle["continuously_backlogged"])
        recorder.reset()

    def test_transient_empty_queue_and_aborts_are_not_saturated(self):
        output = io.StringIO()
        recorder = ServiceCycleRecorder(output, clock=lambda: 5)
        recorder.begin(1)
        recorder.admit(["a"], 4)
        recorder.observe_waiting(0)
        recorder.complete("a", aborted=True)
        recorder.finish(10)
        cycle = json.loads(output.getvalue())
        self.assertFalse(cycle["continuously_backlogged"])
        self.assertFalse(cycle["valid"])
        self.assertEqual(cycle["aborted_requests"], 1)

    def test_reset_of_active_cycle_is_rejected(self):
        recorder = ServiceCycleRecorder(io.StringIO())
        recorder.begin()
        with self.assertRaises(RuntimeError):
            recorder.reset()


def fixtures(count=10):
    cycles, requests = [], []
    for index in range(count):
        number = 20
        start, end = index * 10.0, (index + 1) * 10.0
        ids = ["%s-%s" % (index, j) for j in range(number)]
        cycles.append(dict(event="service_cycle", cycle_id=index + 1, start=start, end=end,
                           continuously_backlogged=True, valid=True,
                           completed_requests=number, completed_request_ids=ids,
                           generated_tokens=number * 512))
        requests.extend(SimpleNamespace(req_id=request_id, start_time=0, end_time=end - 0.1,
                                        success=True, output_tokens=512) for request_id in ids)
    return cycles, requests


class CycleSummaryTests(unittest.TestCase):
    def summarize(self, cycles, requests, **kwargs):
        options = dict(origin=0, warmup=0, output_len=512, skip_cycles=2,
                       min_cycles=6, min_seconds=50, tolerance=0.1)
        options.update(kwargs)
        return summarize_cycles(cycles, requests, **options)

    def test_full_cycles_remove_burst_boundary_bias(self):
        cycles, requests = fixtures()
        result = self.summarize(cycles, requests)
        self.assertEqual(result["cycles"], 8)
        self.assertEqual(result["completed_requests"], 160)
        self.assertEqual(result["seconds"], 80)
        self.assertEqual(result["peak_completed_req_per_s"], 2)
        self.assertTrue(result["qualified"])

    def test_warmup_and_drain_are_excluded_without_arrival_filter(self):
        cycles, requests = fixtures(12)
        cycles[-1]["continuously_backlogged"] = False
        result = self.summarize(cycles, requests, warmup=30)
        self.assertEqual(result["start"], 30)
        self.assertEqual(result["end"], 110)
        self.assertEqual(result["completed_requests"], 160)
        self.assertEqual(result["peak_completed_req_per_s"], 2)

    def test_select_longest_period_not_fastest_period(self):
        cycles, requests = fixtures(12)
        cycles[3]["continuously_backlogged"] = False
        result = self.summarize(cycles, requests, skip_cycles=0)
        self.assertEqual(result["cycle_ids"], list(range(5, 13)))

    def test_jit_or_insufficient_measurement_prevents_peak_claim(self):
        cycles, requests = fixtures()
        for options, reason in ((dict(new_compiles=1), "prefix_decode_compilation_inside_measurement"),
                                (dict(min_cycles=20), "too_few_cycles"),
                                (dict(min_seconds=200), "measurement_too_short")):
            result = self.summarize(cycles, requests, **options)
            self.assertFalse(result["qualified"])
            self.assertIsNone(result["peak_completed_req_per_s"])
            self.assertEqual(result["saturated_completed_req_per_s"], 2)
            self.assertIn(reason, result["reasons"])

    def test_unstable_blocks_are_reported_not_cherry_picked(self):
        cycles, requests = fixtures(12)
        cycles[-1]["end"] += 30
        for req in requests[-20:]:
            req.end_time += 30
        result = self.summarize(cycles, requests)
        self.assertIn("block_throughput_not_stable", result["reasons"])
        self.assertIsNone(result["peak_completed_req_per_s"])

    def test_failed_requests_and_wrong_token_counts_are_not_valid_cycles(self):
        cycles, requests = fixtures(4)
        for req in requests:
            req.output_tokens = 511
        result = self.summarize(cycles, requests, skip_cycles=0)
        self.assertIsNone(result["peak_completed_req_per_s"])
        self.assertEqual(result["cycles"], 0)

    def test_preflight_cycles_are_filtered_by_origin(self):
        cycles, requests = fixtures()
        cycles.insert(0, dict(event="service_cycle", start=-100, end=-50))
        result = self.summarize(cycles, requests)
        self.assertEqual(result["total_cycles"], 10)

    def test_client_response_tail_is_included(self):
        cycles, requests = fixtures()
        requests[-1].end_time = 100.2
        result = self.summarize(cycles, requests)
        self.assertAlmostEqual(result["end"], 100.2)
        self.assertEqual(result["completed_requests"], 160)
        self.assertAlmostEqual(result["saturated_completed_req_per_s"], 160 / 80.2)

    def test_failure_outside_selected_period_still_prevents_peak_claim(self):
        cycles, requests = fixtures()
        requests[0].success = False
        result = self.summarize(cycles, requests)
        self.assertIn("failed_requests_in_point", result["reasons"])
        self.assertIsNone(result["peak_completed_req_per_s"])


if __name__ == "__main__":
    unittest.main()
