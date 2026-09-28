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

    def test_q5_k_profile_records_are_not_confused_with_q4_1(self):
        rows = [dict(r, format="q5_k", values=6144*r["rows"]) for r in self.rows]
        bench.validate(rows, self.checks, "q5_k")
        with self.assertRaises(ValueError):
            bench.validate(rows, self.checks)
        with self.assertRaises(ValueError):
            bench.validate(self.rows, self.checks, "q5_k")

    def test_q6_k_uses_bounded_vocabulary_slice_not_whole_tensor(self):
        workloads = bench.PROFILES["q6_k"]["workloads"]
        checks = {r: dict(input_sha256="a" * 64, output_sha256="b" * 64) for r in workloads}
        rows = [dict(format="q6_k", rows=r, values=5120*r, trial=i, iterations=n, elapsed_ns=10, **checks[r])
                for r, n in workloads.items() for i in range(7)]
        bench.validate(rows, checks, "q6_k")
        self.assertEqual(bench.PROFILES["q6_k"]["tensor_rows"], 248320)
        for count in (5120, 248320):
            bad = [dict(r, rows=count, values=count*5120) if r["rows"] == 4096 else r for r in rows]
            with self.assertRaises(ValueError):
                bench.validate(bad, checks, "q6_k")
        with self.assertRaises(ValueError):
            bench.validate(rows, checks, "q5_k")

    def test_independent_q6_k_fixture_and_helper_provenance(self):
        root = Path(__file__).parents[1]
        data = json.loads((root / "tests/fixtures/q6_k.json").read_text())
        for key, name in [("generator_sha256", "generate_q6_k_goldens.py"), ("helper_sha256", "generate_q4_1_goldens.py")]:
            self.assertEqual(data[key], hashlib.sha256((root / "tests/reference" / name).read_bytes()).hexdigest())
        self.assertEqual(data["fingerprint"]["values"], 65011712)
        self.assertEqual(data["scales_fingerprint"]["values"], 4194304)
        actual = [c for c in data["examples"] if "tensor" in c]
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["tensor"]["dims"], [5120, 248320, 1, 1])
        self.assertEqual(len(actual[0]["block_indices"]), 70)

    def test_independent_q5_k_fixture_and_helper_provenance(self):
        root = Path(__file__).parents[1]
        data = json.loads((root / "tests/fixtures/q5_k.json").read_text())
        for key, name in [("generator_sha256", "generate_q5_k_goldens.py"), ("helper_sha256", "generate_q4_1_goldens.py")]:
            self.assertEqual(data[key], hashlib.sha256((root / "tests/reference" / name).read_bytes()).hexdigest())
        self.assertEqual(data["fingerprint"]["values"], 97517568)
        self.assertEqual(len([c for c in data["examples"] if "tensor" in c]), 48)

    def test_independent_q4_1_fixture_provenance_and_real_shapes(self):
        root = Path(__file__).parents[1]
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
