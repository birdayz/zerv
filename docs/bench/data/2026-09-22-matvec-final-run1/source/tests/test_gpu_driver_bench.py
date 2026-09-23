import copy
import json
from pathlib import Path
import unittest
from bench import run_gpu_driver as bench


class GpuDriverBenchTests(unittest.TestCase):
    def setUp(self):
        self.goldens = json.loads((Path(__file__).parent / "fixtures/gpu/dispatch.json").read_text())
        self.records = [dict(kind="timing", workload=k, count=n, trial=t, iterations=i, elapsed_ns=100)
                        for (k, n), i in bench.WORKLOADS.items() for t in range(7)]
        self.records += [c for c in self.goldens["cases"] if (c["kind"], c["count"]) in bench.WORKLOADS]

    def test_exact_checks_and_matched_timing_sets(self):
        self.assertEqual(len(bench.validate(self.records, self.goldens, True)), 42)
        self.assertEqual(bench.validate(self.goldens["cases"], self.goldens, False), [])
        for records in (self.records[:-1], self.records+[self.records[0]], self.records[1:], []):
            with self.assertRaises(ValueError):
                bench.validate(records, self.goldens, True)

    def test_allocation_metadata_matches_real_fixture(self):
        identity, allocations = bench.metadata(self.goldens["device"])
        self.assertTrue(identity.startswith("device="))
        self.assertEqual(len(allocations), 10)
        lines = self.goldens["device"].splitlines()
        for value in ("", "unknown", "\n".join(lines+[lines[-1]]), "\n".join(lines).replace("memory_types=", "other=")):
            with self.assertRaises(ValueError):
                bench.metadata(value)

    def test_bad_timing_and_schema(self):
        for key, value in [("count", 64), ("trial", True), ("iterations", 2), ("elapsed_ns", 0),
                           ("elapsed_ns", 1.5), ("workload", "other"), ("extra", 1)]:
            records = copy.deepcopy(self.records); records[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate(records, self.goldens, True)

    def test_missing_or_corrupted_full_output_hash_rejects_timings(self):
        for key, value in [("input_sha256", "wrong"), ("output_sha256", "wrong"), ("bytes", True), ("count", True)]:
            records = copy.deepcopy(self.records); records[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate(records, self.goldens, True)
        with self.assertRaises(ValueError):
            bench.validate(self.records, self.goldens, False)


if __name__ == "__main__":
    unittest.main()
