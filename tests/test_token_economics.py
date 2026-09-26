import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("token_economics", ROOT / "tools/token_economics.py")
econ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(econ)
INPUTS = ROOT / "docs/research/2026-09-26-token-economics/inputs.json"


class TokenEconomicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(INPUTS.read_text(encoding="utf-8"))

    def test_model_reproduces_the_measured_aggregate(self):
        # The pure decode rate, put back through the full-load model at the measured
        # workload's input/output ratio, must return the measured aggregate.
        prefill = econ.value(self.cfg["prefill_tok_s"])
        for mode in self.cfg["modes"]:
            decode = econ.pure_decode_rate(mode["aggregate_tok_s"], mode["workload_prompt_tokens"],
                                           mode["workload_output_tokens"], prefill)
            ratio = mode["workload_prompt_tokens"] / mode["workload_output_tokens"]
            out, _ = econ.full_load_rates(decode, prefill, ratio)
            self.assertAlmostEqual(out, mode["aggregate_tok_s"], places=6)

    def test_margin_is_zero_at_break_even(self):
        cfg = self.cfg
        for cost in cfg["cost_bases"]:
            for price in cfg["prices_usd_per_mtok"]:
                s = econ.Scenario(cfg, cfg["modes"][0], cfg["mixes"][1], price, 0.35, cost)
                u = s.break_even()
                if u is not None:
                    self.assertAlmostEqual(s.margin(u), 0.0, places=6)
                else:
                    self.assertLess(s.margin(1.0), 0.0)

    def test_inconsistent_measurement_is_rejected(self):
        with self.assertRaises(ValueError):
            econ.pure_decode_rate(100.0, 10_000, 10, 1200.0)

    def test_report_renders(self):
        text = econ.report(self.cfg)
        for heading in ("### T1.", "### T2.", "### T3.", "### T4.", "### T5.", "### T6.", "### T7.", "### T8."):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
