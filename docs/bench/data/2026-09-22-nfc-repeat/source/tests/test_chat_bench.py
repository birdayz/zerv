import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("chat_bench", Path(__file__).resolve().parents[1] / "bench/run_chat.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ChatBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(renders=95, bytes=200, output_sha256="a" * 64)
        self.rows = [dict(trial=i, iterations=100, elapsed_ns=1000, **self.expected) for i in range(7)]

    def test_valid(self):
        bench.validate(self.rows, self.expected)

    def test_invalid_sets(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]], [self.rows[0]] * 7):
            with self.assertRaises(ValueError):
                bench.validate(rows, self.expected)

    def test_corrupt_fields(self):
        for key, value in [("trial", 7), ("elapsed_ns", 0), ("elapsed_ns", True), ("iterations", 1),
                           ("renders", 1), ("bytes", 0), ("output_sha256", "bad"), ("extra", 0)]:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                bench.validate(rows, self.expected)
        rows = copy.deepcopy(self.rows)
        del rows[0]["renders"]
        with self.assertRaises(ValueError):
            bench.validate(rows, self.expected)


if __name__ == "__main__":
    unittest.main()
