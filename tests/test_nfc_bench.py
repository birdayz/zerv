import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nfc_bench", ROOT / "bench/run_nfc.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class NfcBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.corpus = json.loads(bench.FIXTURE.read_text())
        self.expected = bench.expected_cases(self.corpus)
        self.rows = [dict(workload=name, trial=i, iterations=bench.ITERATIONS[name], elapsed_ns=1000, **fields)
                     for name, fields in self.expected.items() for i in range(7)]

    def test_valid(self):
        bench.validate(self.rows, self.expected)

    def test_invalid_sets(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]], [self.rows[0]] * 21):
            with self.assertRaises(ValueError):
                bench.validate(rows, self.expected)
        for workloads in (self.corpus["workloads"][:-1], self.corpus["workloads"] + [self.corpus["workloads"][0]]):
            with self.assertRaises(ValueError):
                bench.expected_cases({"workloads": workloads})

    def test_corrupt_fields(self):
        for key, value in [("trial", 7), ("elapsed_ns", 0), ("elapsed_ns", True), ("iterations", 1),
                           ("workload", "unknown"), ("workload", []), ("input_bytes", 1), ("output_bytes", 0),
                           ("output_sha256", "bad"), ("extra", 0)]:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                bench.validate(rows, self.expected)
        rows = copy.deepcopy(self.rows)
        del rows[0]["output_bytes"]
        with self.assertRaises(ValueError):
            bench.validate(rows, self.expected)

    def test_provenance(self):
        for path, key in [(ROOT / "tests/reference/generate_nfc.py", "generator_sha256"),
                          (ROOT / "src/text/data/nfc9.bin", "data_sha256"),
                          (bench.FIXTURE.parent / "cases.bin", "golden_sha256")]:
            self.assertEqual(bench.sha(path), self.corpus[key])
        self.assertIn("COPYRIGHT AND PERMISSION NOTICE", (ROOT / "src/text/data/UNICODE-LICENSE.txt").read_text())


if __name__ == "__main__":
    unittest.main()
