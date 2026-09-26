import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("gguf_bench", Path(__file__).parents[1] / "bench/run_gguf.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class GgufBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.container = dict(metadata=[{}] * 51, tensors=[{}] * 866, data_offset=320)
        self.rows = [dict(trial=i, iterations=10, elapsed_ns=1000, metadata=51, tensors=866, data_offset=320) for i in range(7)]

    def test_valid(self):
        bench.validate(self.rows, self.container)

    def test_invalid_trial_sets(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]], [self.rows[0]] * 7):
            with self.assertRaises(ValueError):
                bench.validate(rows, self.container)

    def test_invalid_fields(self):
        for field, value in (("trial", 7), ("iterations", 0), ("elapsed_ns", 0), ("elapsed_ns", -1),
                             ("elapsed_ns", True), ("elapsed_ns", 1.0), ("metadata", 0), ("tensors", 0),
                             ("data_offset", 0), ("extra", 1)):
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    bench.validate(rows, self.container)
        rows = copy.deepcopy(self.rows)
        del rows[0]["metadata"]
        with self.assertRaises(ValueError):
            bench.validate(rows, self.container)


if __name__ == "__main__":
    unittest.main()
