"""Record a planner quote before the first experiment, with auditable corrections."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import math
from statistics import median

from ledger import Ledger


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def record_quote(ledger: Ledger, opening_plan: Mapping, *, history: Iterable[Mapping] | None = None) -> dict:
    """Write one quote; history is completed prior-task predicted/actual pairs."""
    prior = ledger._events()
    if any(item["kind"] in {"quote", "experiment"} for item in prior):
        raise ValueError("Quote must be recorded once, before the first experiment")
    prediction = opening_plan.get("predicted")
    if not isinstance(prediction, Mapping):
        raise ValueError("Opening plan has no predicted values")
    predicted = {
        "cost": _number(prediction.get("cost"), "predicted.cost", nonnegative=True),
        "seconds": _number(prediction.get("seconds"), "predicted.seconds", nonnegative=True),
        "score": _number(prediction.get("score"), "predicted.score"),
    }
    cost_ratios: list[float] = []
    time_ratios: list[float] = []
    score_errors: list[float] = []
    for index, item in enumerate(history or []):
        if not isinstance(item, Mapping) or not isinstance(item.get("predicted"), Mapping) or not isinstance(item.get("actual"), Mapping):
            raise ValueError(f"Invalid quote history item {index}")
        previous = item["predicted"]
        actual = item["actual"]
        old_cost = _number(previous.get("cost"), "history.predicted.cost", nonnegative=True)
        old_seconds = _number(previous.get("seconds"), "history.predicted.seconds", nonnegative=True)
        old_score = _number(previous.get("score"), "history.predicted.score")
        actual_cost = _number(actual.get("cost"), "history.actual.cost", nonnegative=True)
        actual_seconds = _number(actual.get("seconds"), "history.actual.seconds", nonnegative=True)
        actual_score = _number(actual.get("score"), "history.actual.score")
        if old_cost > 0:
            cost_ratios.append(actual_cost / old_cost)
        if old_seconds > 0:
            time_ratios.append(actual_seconds / old_seconds)
        score_errors.append(actual_score - old_score)
    factors = {
        "cost_ratio": median(cost_ratios) if cost_ratios else 1.0,
        "seconds_ratio": median(time_ratios) if time_ratios else 1.0,
        "score_delta": median(score_errors) if score_errors else 0.0,
    }
    corrected = {
        "cost": predicted["cost"] * factors["cost_ratio"],
        "seconds": predicted["seconds"] * factors["seconds_ratio"],
        "score": predicted["score"] + factors["score_delta"],
    }
    basis = {"source": "planner_opening_plan", "method": "median_prior_error" if score_errors else "uncorrected",
             "history_n": len(score_errors), "uncorrected": predicted, "factors": factors}
    ledger.emit("quote", predicted=corrected, basis=basis)
    return {"predicted": corrected, "basis": basis}


def make_quote_callback(ledger: Ledger, history: Iterable[Mapping] | None = None) -> Callable[[dict], None]:
    """Pass to TaskRunner(on_initial_plan=...) to quote before execution."""
    frozen_history = tuple(history or ())

    def on_initial_plan(opening_plan: dict) -> None:
        record_quote(ledger, opening_plan, history=frozen_history)

    return on_initial_plan


__all__ = ["make_quote_callback", "record_quote"]
