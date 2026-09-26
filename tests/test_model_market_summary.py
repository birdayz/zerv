import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("market", ROOT / "tools/model_market_summary.py")
market = importlib.util.module_from_spec(spec)
spec.loader.exec_module(market)
DATA = ROOT / "docs/research/2026-09-26-bigger-models"


class ModelMarketSummaryTests(unittest.TestCase):
    def test_every_model_has_market_and_config_data(self):
        orm = market.load(DATA, "openrouter-models.json")
        hf = market.load(DATA, "huggingface-models.json")
        rows = market.market_rows(orm, hf)
        self.assertEqual(len(rows), len(market.MODELS))

    def test_serving_rows_cover_every_inferencex_view(self):
        pe = market.load(DATA, "inferencex-profit-estimator.json")
        deploy = market.load(DATA, "inferencex-deployments.json")
        rows = market.serving_rows(pe, deploy)
        self.assertEqual({r[0] for r in rows}, set(pe["views"]))


if __name__ == "__main__":
    unittest.main()
