import copy
import hashlib
import json
from pathlib import Path
import unittest

from bench import run_model_quant as bench


class ModelQuantBenchTests(unittest.TestCase):
    def setUp(self):
        self.checks = {r: dict(input_sha256="a" * 64, output_sha256="b" * 64) for r in bench.ROWS}
        self.rows = [dict(format="q4_1", rows=r, values=17408*r, trial=i, iterations=n, elapsed_ns=10, **self.checks[r])
                     for r, n in bench.ROWS.items() for i in range(7)]

    def test_exact_record_set(self):
        bench.validate(self.rows, self.checks)
        for records in (self.rows[:-1], self.rows + [self.rows[0]], [self.rows[0]] * len(self.rows)):
            with self.assertRaises(ValueError):
                bench.validate(records, self.checks)

    def test_corrupt_timing_shape_and_results(self):
        for key, value in [("format", "q4_0"), ("rows", 2), ("trial", True), ("iterations", 0), ("values", 32),
                           ("elapsed_ns", 0), ("elapsed_ns", 1.2), ("output_sha256", "wrong"), ("input_sha256", "wrong"), ("extra", 0)]:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate(rows, self.checks)

    def test_independent_q4_1_fixture_provenance_and_real_shapes(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "tests/fixtures/q4_1.json").read_text())
        self.assertEqual(data["generator_sha256"], hashlib.sha256((root / "tests/reference/generate_q4_1_goldens.py").read_bytes()).hexdigest())
        self.assertEqual(data["fingerprint"]["values"], 44695552)
        actual = [c for c in data["examples"] if "tensor" in c]
        self.assertEqual(len(actual), 8)
        for c in actual:
            self.assertEqual(c["tensor"]["dims"], [17408, 5120, 1, 1])
            self.assertEqual(len(c["block_indices"]), 8)


if __name__ == "__main__":
    unittest.main()
