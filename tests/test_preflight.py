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


if __name__ == "__main__":
    unittest.main()
