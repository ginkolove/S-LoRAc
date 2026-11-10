import unittest

from overlap_analysis import (analyze_trace, analyze_compute_window_trace, compare, compare_traces,
                              intersection_duration, merge_intervals, summarize)


class AnalysisTests(unittest.TestCase):
    def test_compute_window_attribution_follows_launch_not_execution_time(self):
        events = [dict(name="window/ffn", ph="X", ts=0, dur=100, tid=1),
                  dict(name="cudaLaunchKernel", ph="X", cat="cuda_runtime", ts=50, dur=5,
                       tid=1, args=dict(correlation=12)),
                  dict(name="gemm", cat="kernel", ts=300, dur=200, args=dict(correlation=12)),
                  dict(name="_lowra_reconstruct_kernel", cat="kernel", ts=200, dur=200)]
        result = analyze_compute_window_trace(dict(traceEvents=events))
        self.assertTrue(result["valid"])
        self.assertEqual(result["kernel_ms"]["ffn"], 0.2)
        self.assertEqual(result["overlap_with_reconstruction_ms"]["ffn"], 0.1)
        self.assertFalse(analyze_compute_window_trace({})["valid"])

    def test_percentile(self):
        self.assertEqual(summarize([3, 1, 2])["median"], 2)
        self.assertAlmostEqual(summarize([3, 1, 2])["p95"], 2.9)
        self.assertEqual(summarize([4])["p95"], 4)
        with self.assertRaises(ValueError):
            summarize([])

    def test_union_not_double_counted(self):
        self.assertEqual(merge_intervals([(0, 4), (2, 5), (7, 8), (8, 9)]), [(0, 5), (7, 9)])
        self.assertEqual(intersection_duration([(0, 4), (2, 5)], [(1, 2), (3, 6)]), 3)
        self.assertEqual(intersection_duration([(0, 1)], [(1, 2)]), 0)

    def test_trace_uses_kernels_not_cpu_ranges(self):
        events = [dict(name="_lowra_reconstruct_kv_kernel", ph="X", cat="kernel", ts=1000, dur=4000,
                       args=dict(stream=7)),
                  dict(name="gemm", ph="X", cat="kernel", ts=3000, dur=3000, args=dict(stream=9)),
                  dict(name="aten::mm", ph="X", cat="cpu_op", ts=0, dur=10000),
                  dict(name="memcpy", ph="X", cat="gpu_memcpy", ts=0, dur=10000)]
        result = analyze_trace(dict(traceEvents=events))
        self.assertEqual(result["actual_kernel_overlap_ms"], 2)
        self.assertEqual(result["reconstruct_kernel_union_ms"], 4)
        self.assertEqual(result["ffn_kernel_union_ms"], 3)
        self.assertEqual(len(result["timeline"]), 2)
        self.assertTrue(result["paired"])
        self.assertFalse(analyze_trace({})["valid"])

    def test_graph_bookkeeping_is_not_ffn(self):
        events = [dict(name="_lowra_reconstruct_kv_kernel", ph="X", cat="kernel", ts=10, dur=20),
                  dict(name="vectorized_elementwise_kernel<FillFunctor<long>>", ph="X", cat="kernel", ts=0, dur=5)]
        result = analyze_trace(dict(traceEvents=events))
        self.assertEqual(result["ffn_kernel_count"], 0)
        self.assertEqual(result["auxiliary_kernel_count"], 1)
        self.assertFalse(result["paired"])
        self.assertFalse(compare_traces({"reconstruct": dict(valid=False)})["valid"])

    def test_graph_only_total_does_not_fabricate_branch_slowdown(self):
        samples = {mode: [dict(total_ms=time)] for mode, time in (
            ("reconstruct", 4), ("ffn", 1), ("serial_rf", 5), ("serial_fr", 5),
            ("overlap_rf", 4.5), ("overlap_fr", 4.5))}
        result = compare(samples)["comparisons"]["rf"]
        self.assertIsNone(result["reconstruct_event_span_slowdown"])
        self.assertIsNone(result["ffn_event_span_slowdown"])
        self.assertAlmostEqual(result["serial_over_parallel_speedup"], 5 / 4.5)

    def test_no_gain_when_parallel_work_slows_down(self):
        samples = dict(reconstruct=[dict(total_ms=4, reconstruct_ms=4)],
                       ffn=[dict(total_ms=1, ffn_ms=1)])
        for order in ("rf", "fr"):
            samples["serial_" + order] = [dict(total_ms=5, reconstruct_ms=4, ffn_ms=1)]
            samples["overlap_" + order] = [dict(total_ms=6, reconstruct_ms=6, ffn_ms=2)]
        result = compare(samples)["comparisons"]["rf"]
        self.assertAlmostEqual(result["serial_over_parallel_speedup"], 5 / 6)
        self.assertEqual(result["reconstruct_event_span_slowdown"], 1.5)
        self.assertEqual(result["ffn_event_span_slowdown"], 2)
        self.assertEqual(result["ideal_max_speedup"], 1.25)
        self.assertEqual(result["net_saved_ms"], -1)
        self.assertEqual(result["net_saved_fraction_of_shorter_task"], -1)


if __name__ == "__main__":
    unittest.main()
