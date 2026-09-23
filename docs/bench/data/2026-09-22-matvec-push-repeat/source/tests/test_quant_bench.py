import copy
import unittest

from bench import run_quant


class BenchmarkValidationTests(unittest.TestCase):
    def setUp(self):
        self.hashes = {"input_sha256": "a" * 64, "output_sha256": "b" * 64}
        self.cases = {fmt: (None, None, None, self.hashes) for fmt in ("q4_0", "q8_0")}
        self.records = [
            dict(format=fmt, trial=trial, iterations=run_quant.ITERATIONS,
                 values_per_call=run_quant.VALUES, elapsed_ns=100, **self.hashes)
            for fmt in self.cases for trial in range(run_quant.TRIALS)
        ]

    def test_complete_results(self):
        run_quant.validate(self.records, self.cases)

    def test_missing_extra_and_duplicate_trials(self):
        invalid = [self.records[:-1], self.records + [self.records[0]],
                   self.records[:-1] + [self.records[0]]]
        for records in invalid:
            with self.subTest(records=len(records)), self.assertRaises(RuntimeError):
                run_quant.validate(records, self.cases)

    def test_corrupt_results_are_not_accepted_as_benchmarks(self):
        changes = {
            "input_sha256": "c" * 64,
            "output_sha256": "c" * 64,
            "iterations": run_quant.ITERATIONS + 1,
            "values_per_call": run_quant.VALUES - 1,
            "format": "unsupported",
            "trial": True,
            "elapsed_ns": 0,
        }
        for name, value in changes.items():
            records = copy.deepcopy(self.records)
            records[0][name] = value
            with self.subTest(field=name), self.assertRaises(RuntimeError):
                run_quant.validate(records, self.cases)
        for value in (-1, 1.0, True):
            records = copy.deepcopy(self.records)
            records[0]["elapsed_ns"] = value
            with self.subTest(time=value), self.assertRaises(RuntimeError):
                run_quant.validate(records, self.cases)

    def test_result_schema(self):
        for missing in (True, False):
            records = copy.deepcopy(self.records)
            if missing:
                del records[0]["output_sha256"]
            else:
                records[0]["unexpected"] = 1
            with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                run_quant.validate(records, self.cases)


if __name__ == "__main__":
    unittest.main()
