from __future__ import annotations

import unittest

from quant_research.signals import trailing_momentum


class SignalTests(unittest.TestCase):
    def test_uses_only_supplied_history(self):
        earlier = {"A": (100.0, 102.0, 104.0), "B": (100.0, 101.0, 99.0)}
        scores = trailing_momentum(earlier, lookback=2)
        self.assertAlmostEqual(scores["A"], 0.04)
        self.assertAlmostEqual(scores["B"], -0.01)
        self.assertEqual(earlier["A"], (100.0, 102.0, 104.0))

    def test_warmup_and_bad_lookback(self):
        self.assertEqual(trailing_momentum({"A": (100.0,)}, lookback=2), {"A": 0.0})
        for value in (0, True, 1.2):
            with self.assertRaisesRegex(ValueError, "lookback"):
                trailing_momentum({}, lookback=value)


if __name__ == "__main__":
    unittest.main()
