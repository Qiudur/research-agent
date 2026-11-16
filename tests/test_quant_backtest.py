"""Timing, fee and failure-path tests for the quant research backtester."""

from __future__ import annotations

from datetime import date
import json
import math
import unittest

from quant_research.backtest import equal_weight_signal, run_backtest
from quant_research.data import PricePanel


DATES = tuple(date(2020, 1, day) for day in range(1, 5))


def panel(a=(100.0, 100.0, 110.0, 121.0),
          b=(100.0, 100.0, 90.0, 81.0),
          symbols=("A", "B")) -> PricePanel:
    return PricePanel(DATES, symbols, {"A": a, "B": b}, "0" * 64)


class BacktestTests(unittest.TestCase):
    def test_signal_sees_only_history_two_closes_before_realized_return(self):
        seen = []

        def signal(history):
            seen.append({symbol: values for symbol, values in history.items()})
            with self.assertRaises(TypeError):
                history["A"] = ()
            return {"A": history["A"][-1], "B": history["B"][-1]}

        result = run_backtest(panel(), signal, first_result_date=DATES[2],
                              last_result_date=DATES[3], top_k=1)
        self.assertEqual(seen, [
            {"A": (100.0,), "B": (100.0,)},
            {"A": (100.0, 100.0), "B": (100.0, 100.0)},
        ])
        self.assertEqual(result.return_dates, DATES[2:])
        self.assertEqual(result.weights, ((1.0, 0.0), (1.0, 0.0)))
        self.assertAlmostEqual(result.gross_returns[0], 0.1)
        self.assertAlmostEqual(result.gross_returns[1], 0.1)
        self.assertAlmostEqual(result.equity[-1], 1.21)
        self.assertEqual(result.turnovers, (1.0, 0.0))
        saved = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        self.assertEqual(saved["schema"], "quant_backtest/1")
        self.assertEqual(saved["return_dates"], ["2020-01-03", "2020-01-04"])
        self.assertEqual(saved["weights"], [[1.0, 0.0], [1.0, 0.0]])

    def test_future_prices_cannot_change_an_earlier_selection(self):
        first = panel()
        altered = panel(a=(100.0, 999.0, 1.0, 2.0),
                        b=(100.0, 1.0, 999.0, 2.0))

        def signal(history):
            return {symbol: prices[-1] for symbol, prices in history.items()}

        normal = run_backtest(first, signal, first_result_date=DATES[2],
                              last_result_date=DATES[2], top_k=1)
        changed = run_backtest(altered, signal, first_result_date=DATES[2],
                               last_result_date=DATES[2], top_k=1)
        self.assertEqual(normal.weights, changed.weights)
        self.assertEqual(normal.weights, ((1.0, 0.0),))
        self.assertNotEqual(normal.gross_returns, changed.gross_returns)

    def test_fees_slippage_and_drifted_weight_turnover(self):
        price_panel = panel(a=(100.0, 100.0, 110.0, 110.0),
                            b=(100.0, 100.0, 90.0, 90.0))
        kwargs = dict(first_result_date=DATES[2], last_result_date=DATES[3],
                      top_k=2)
        free = run_backtest(price_panel, equal_weight_signal, **kwargs)
        paid = run_backtest(price_panel, equal_weight_signal,
                            fee_bps=50, slippage_bps=50, **kwargs)
        self.assertAlmostEqual(free.turnovers[0], 1.0)
        self.assertAlmostEqual(free.turnovers[1], 0.1)
        self.assertEqual(free.turnovers, paid.turnovers)
        self.assertAlmostEqual(paid.net_returns[0], -0.01)
        self.assertLess(paid.metrics["total_return"], free.metrics["total_return"])
        self.assertAlmostEqual(paid.metrics["max_drawdown"], 1 - paid.equity[-1])

    def test_equal_scores_use_symbol_name_as_stable_tiebreaker(self):
        price_panel = PricePanel(DATES, ("A", "B"),
                                 {"B": (100.0,) * 4, "A": (100.0,) * 4}, "0" * 64)
        result = run_backtest(price_panel, equal_weight_signal,
                              first_result_date=DATES[2], last_result_date=DATES[2],
                              top_k=1)
        self.assertEqual(result.symbols, ("A", "B"))
        self.assertEqual(result.weights, ((1.0, 0.0),))

    def test_zero_volatility_reports_no_sharpe(self):
        flat = panel(a=(100.0,) * 4, b=(100.0,) * 4)
        result = run_backtest(flat, equal_weight_signal,
                              first_result_date=DATES[2], last_result_date=DATES[3],
                              top_k=2)
        self.assertEqual(result.metrics["volatility"], 0.0)
        self.assertIsNone(result.metrics["sharpe"])
        self.assertEqual(result.metrics["total_return"], 0.0)

    def test_invalid_scores_and_parameters_fail_closed(self):
        price_panel = panel()
        kwargs = dict(first_result_date=DATES[2], last_result_date=DATES[3], top_k=1)
        bad_scores = (
            {"A": 1.0}, {"A": 1.0, "B": 0.0, "C": 0.0},
            {"A": math.nan, "B": 0.0}, {"A": True, "B": 0.0},
            {"A": "one", "B": 0.0},
        )
        for scores in bad_scores:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                run_backtest(price_panel, lambda _: scores, **kwargs)
        for changed in (
            {"top_k": 0}, {"top_k": True}, {"top_k": 3},
            {"fee_bps": -1}, {"slippage_bps": math.inf},
            {"first_result_date": DATES[1]},
            {"last_result_date": DATES[1]},
            {"first_result_date": date(2020, 2, 1)},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                run_backtest(price_panel, equal_weight_signal, **{**kwargs, **changed})


if __name__ == "__main__":
    unittest.main()
