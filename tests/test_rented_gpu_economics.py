import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("rented", ROOT / "tools/rented_gpu_economics.py")
rented = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rented)
DATA = ROOT / "docs/research/2026-09-26-rented-gpu-economics"


class RentedGpuEconomicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((DATA / "inputs.json").read_text(encoding="utf-8"))
        self.rows = json.loads((DATA / "inferencex-llama70b.json").read_text(encoding="utf-8"))["rows"]
        self.llama = self.cfg["models"]["llama-3.3-70b"]
        self.qwen = self.cfg["models"]["qwen3.8-27b"]
        req = self.cfg["request"]
        self.work = rented.Workload(req["uncached_input"], req["cached_input"], req["output"])

    def test_model_that_does_not_fit_is_infeasible(self):
        # Llama-3.3-70B FP8 weights (~72 GB) leave no KV room on one 80 GB H100.
        work = rented.Workload(8192, 0, 1024)
        h100 = self.cfg["gpus"]["h100"]
        self.assertEqual(rented.replica_rate(self.llama, h100, "fp8", work, 30, 1.0, self.cfg, 1), 0)
        self.assertGreater(rented.replica_rate(self.llama, h100, "fp8", work, 30, 1.0, self.cfg, 2), 0)

    def test_fitted_efficiency_reproduces_the_measurement(self):
        fits = rented.fitted_etas(self.cfg, self.rows, 50)
        work = rented.Workload(8192, 0, 1024)
        for key, fit in fits.items():
            if not fit or fit[3] is None:
                continue
            tokens, _, tp, eta, _ = fit
            got = rented.replica_rate(self.llama, self.cfg["gpus"][key], "fp8", work, 50, eta, self.cfg, tp)
            self.assertAlmostEqual(got * work.tokens() / tokens, 1.0, delta=0.02)

    def test_faster_users_never_raise_throughput(self):
        gpu = self.cfg["gpus"]["mi355x"]
        rates = [rented.best_rate(self.qwen, gpu, "fp8", self.work, s, 0.7, self.cfg, tps=(1,))
                 for s in (20, 30, 50, 100, 150)]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_report_renders(self):
        text = rented.report(self.cfg, self.rows)
        for heading in ("### A30.", "### B50.", "### C.", "### D.", "### E."):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
