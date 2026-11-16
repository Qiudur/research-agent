"""Deterministic daily-close backtests for a fixed-universe research demo.

A signal for date t receives only prices through t. Its target weights are
traded after the t+1 close, then earn the t+1-to-t+2 close return. This extra
session of delay is deliberate: daily close data cannot justify a fill at the
same close that produced the signal. The candidate sandbox must additionally
isolate user code from the underlying full panel; this module only supplies
an immutable prefix to the signal callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Callable, Mapping

from .data import PricePanel


TRADING_DAYS_PER_YEAR = 252
Signal = Callable[[Mapping[str, tuple[float, ...]]], Mapping[str, float]]


@dataclass(frozen=True)
class BacktestResult:
    """Period returns and reviewable holdings in source-symbol order.

    Equity starts at 1 before the first trade, so its length is one greater
    than the number of return dates. Turnover is gross traded notional:
    sum(abs(new_weight - pre_trade_weight)); the first full investment is 1.
    """

    return_dates: tuple[date, ...]
    symbols: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    gross_returns: tuple[float, ...]
    net_returns: tuple[float, ...]
    turnovers: tuple[float, ...]
    equity: tuple[float, ...]
    metrics: Mapping[str, float | None]

    def to_dict(self) -> dict:
        """JSON-ready evidence without dropping dates, holdings or negative returns."""
        return {
            "schema": "quant_backtest/1",
            "return_dates": [day.isoformat() for day in self.return_dates],
            "symbols": list(self.symbols),
            "weights": [list(row) for row in self.weights],
            "gross_returns": list(self.gross_returns),
            "net_returns": list(self.net_returns),
            "turnovers": list(self.turnovers),
            "equity": list(self.equity),
            "metrics": dict(self.metrics),
        }


def equal_weight_signal(history: Mapping[str, tuple[float, ...]]) -> dict[str, float]:
    """A no-prediction benchmark; pass top_k=len(panel.symbols) to use it."""
    return {symbol: 0.0 for symbol in history}


def _nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return amount


def _validate_panel(panel: PricePanel) -> None:
    if len(panel.dates) < 3 or not panel.symbols or len(set(panel.symbols)) != len(panel.symbols):
        raise ValueError("Backtest needs at least three dates and unique symbols")
    if any(type(day) is not date for day in panel.dates) or any(
            left >= right for left, right in zip(panel.dates, panel.dates[1:])):
        raise ValueError("Panel dates must be strictly increasing calendar dates")
    if set(panel.closes) != set(panel.symbols):
        raise ValueError("Panel prices must match the symbol universe")
    for symbol in panel.symbols:
        prices = panel.closes[symbol]
        if len(prices) != len(panel.dates) or any(
                isinstance(price, bool) or not isinstance(price, (int, float))
                or not math.isfinite(price) or price <= 0 for price in prices):
            raise ValueError(f"Invalid close series for {symbol}")


def _target_weights(scores: Mapping[str, float], symbols: tuple[str, ...],
                    top_k: int) -> tuple[float, ...]:
    if not isinstance(scores, Mapping) or set(scores) != set(symbols):
        raise ValueError("Signal must score every symbol and no unknown symbols")
    for symbol, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)) \
                or not math.isfinite(score):
            raise ValueError(f"Signal returned a non-finite score for {symbol}")
    ranked = sorted(symbols, key=lambda symbol: (-scores[symbol], symbol))
    selected = set(ranked[:top_k])
    return tuple(1.0 / top_k if symbol in selected else 0.0 for symbol in symbols)


def _metrics(returns: tuple[float, ...], equity: tuple[float, ...],
             turnovers: tuple[float, ...]) -> Mapping[str, float | None]:
    count = len(returns)
    mean = fmean(returns)
    period_volatility = pstdev(returns)
    annualized_volatility = period_volatility * math.sqrt(TRADING_DAYS_PER_YEAR)
    peak = 1.0
    worst_drawdown = 0.0
    for value in equity[1:]:
        peak = max(peak, value)
        worst_drawdown = max(worst_drawdown, 1.0 - value / peak)
    return MappingProxyType({
        "total_return": equity[-1] - 1.0,
        "annualized_return": equity[-1] ** (TRADING_DAYS_PER_YEAR / count) - 1.0,
        "volatility": annualized_volatility,
        "sharpe": mean / period_volatility * math.sqrt(TRADING_DAYS_PER_YEAR)
                   if period_volatility > 0 else None,
        "max_drawdown": worst_drawdown,
        "mean_turnover": fmean(turnovers),
    })


def run_backtest(
    panel: PricePanel,
    signal_fn: Signal,
    *,
    first_result_date: date,
    last_result_date: date,
    top_k: int,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """Run a prefix-only signal over an inclusive range of realized return dates.

    Transaction costs apply to gross traded notional at the rebalance close.
    The initial cash-to-position purchase pays cost too. There is no claim that
    a fixed-universe close-only demonstration handles real delistings or fills.
    """
    _validate_panel(panel)
    if not callable(signal_fn):
        raise TypeError("signal_fn must be callable")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(panel.symbols):
        raise ValueError("top_k must be between 1 and the number of symbols")
    if type(first_result_date) is not date or type(last_result_date) is not date:
        raise ValueError("Result bounds must be calendar dates")
    fee = _nonnegative_finite(fee_bps, "fee_bps")
    slippage = _nonnegative_finite(slippage_bps, "slippage_bps")
    try:
        first = panel.dates.index(first_result_date)
        last = panel.dates.index(last_result_date)
    except ValueError as exc:
        raise ValueError("Result date is absent from the panel") from exc
    if first < 2 or first > last:
        raise ValueError("Result window must begin after two observed dates")

    return_dates: list[date] = []
    weights: list[tuple[float, ...]] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    equity = [1.0]
    pre_trade_weights = tuple(0.0 for _ in panel.symbols)
    cost_rate = (fee + slippage) / 10_000.0

    for result_index in range(first, last + 1):
        signal_index = result_index - 2
        # A read-only mapping of immutable copies is the only panel view passed
        # to a signal. The later sandbox must also prevent global file reads.
        history = MappingProxyType({
            symbol: tuple(panel.closes[symbol][:signal_index + 1])
            for symbol in panel.symbols
        })
        target = _target_weights(signal_fn(history), panel.symbols, top_k)
        turnover = math.fsum(abs(new - old) for new, old in zip(target, pre_trade_weights))
        trade_cost = cost_rate * turnover
        if trade_cost >= 1:
            raise ValueError("Transaction costs would exhaust the portfolio at rebalance")
        relatives = tuple(
            panel.closes[symbol][result_index] / panel.closes[symbol][result_index - 1]
            for symbol in panel.symbols
        )
        gross = math.fsum(weight * (relative - 1.0) for weight, relative in zip(target, relatives))
        net = (1.0 - trade_cost) * (1.0 + gross) - 1.0
        next_equity = equity[-1] * (1.0 + net)
        if not math.isfinite(next_equity) or next_equity <= 0:
            raise ValueError("Portfolio equity is invalid")
        gross_relative = 1.0 + gross
        pre_trade_weights = tuple(
            weight * relative / gross_relative for weight, relative in zip(target, relatives)
        )
        return_dates.append(panel.dates[result_index])
        weights.append(target)
        gross_returns.append(gross)
        net_returns.append(net)
        turnovers.append(turnover)
        equity.append(next_equity)

    net_tuple = tuple(net_returns)
    equity_tuple = tuple(equity)
    turnover_tuple = tuple(turnovers)
    return BacktestResult(
        return_dates=tuple(return_dates), symbols=panel.symbols, weights=tuple(weights),
        gross_returns=tuple(gross_returns), net_returns=net_tuple,
        turnovers=turnover_tuple, equity=equity_tuple,
        metrics=_metrics(net_tuple, equity_tuple, turnover_tuple),
    )
