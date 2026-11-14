import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.preflight import _positive, audit


class PreflightTests(unittest.TestCase):
    def test_missing_bank_fails_without_side_effects(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result = audit(root)
            self.assertFalse(result["ready"])
            self.assertIn("Cannot read bank", result["blockers"][0])
            self.assertFalse((root / "runs").exists())

    def test_money_inputs_require_positive_finite_numbers(self):
        self.assertTrue(_positive(1.5))
        for value in (None, 0, -1, True, float("inf"), float("nan")):
            self.assertFalse(_positive(value))

    def test_token_accounting_is_labeled_only_with_non_test_rates(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tasks").mkdir()
            (root / "config").mkdir()
            (root / "tasks/bank.json").write_text(json.dumps({
                "schema": "bank/1", "version": "v1", "frozen": False, "tasks": [],
            }), encoding="utf-8")
            prices_path = root / "config/prices.json"
            prices = {"schema": "prices/1", "currency": "TOKENS", "unit_tokens": 1,
                      "test_only": False, "models": {"local-model": {
                          "input": 1, "cached_input": 1, "output": 1, "reasoning": 1}}}
            prices_path.write_text(json.dumps(prices), encoding="utf-8")
            result = audit(root, "local-model")
            self.assertEqual(result["accounting_scope"], "internal_token_study")
            self.assertIn("非官方比赛成绩", result["accounting_note"])
            self.assertFalse(result["ready"])
            prices["test_only"] = True
            prices_path.write_text(json.dumps(prices), encoding="utf-8")
            self.assertIsNone(audit(root, "local-model")["accounting_scope"])
            prices["test_only"] = False
            prices["unit_tokens"] = 0
            prices_path.write_text(json.dumps(prices), encoding="utf-8")
            self.assertIsNone(audit(root, "local-model")["accounting_scope"])
            prices["unit_tokens"] = 1
            prices["currency"] = "CNY"
            prices_path.write_text(json.dumps(prices), encoding="utf-8")
            self.assertIsNone(audit(root, "local-model")["accounting_scope"])


if __name__ == "__main__":
    unittest.main()
