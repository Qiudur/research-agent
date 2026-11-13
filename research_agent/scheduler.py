"""Deterministic budget gates, review triggers and experiment selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Mapping, Sequence


class ScheduleError(ValueError):
    """The task budget, settings or hypothesis is invalid."""


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ScheduleError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ScheduleError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class Scheduler:
    """Rules-only scheduler: no manager model and no hidden budget changes."""

    config: Mapping[str, Any]
    hard_caps: Mapping[str, Any]
    budget: float
    wall_seconds: float

    def __post_init__(self) -> None:
        if self.config.get("schema") != "agent/1":
            raise ScheduleError("Unsupported agent settings schema")
        if self.hard_caps.get("schema") != "hard_caps/1":
            raise ScheduleError("Unsupported hard cap schema")
        if _number(self.budget, "budget") <= 0:
            raise ScheduleError("A task needs a positive priced budget")
        if _number(self.wall_seconds, "wall_seconds") <= 0:
            raise ScheduleError("A task needs a positive time budget")
        ratio = _number(self.hard_caps.get("budget_ratio"), "budget_ratio")
        reserve = _number(self.hard_caps.get("reserve_for_wrapup"), "reserve_for_wrapup")
        if not 0 < ratio < 1 or not 0 <= reserve < 1 or ratio + reserve > 1.000000001:
            raise ScheduleError("Hard cap and wrap-up reserve are invalid")
        if int(self.config.get("plateau_nodes", 0)) <= 0 or int(self.config.get("explore_every", 0)) <= 0:
            raise ScheduleError("Review and exploration intervals must be positive")

    @property
    def spend_cap(self) -> float:
        return self.budget * float(self.hard_caps["budget_ratio"])

    def budget_remaining(self, spent: float) -> float:
        return max(0.0, self.spend_cap - _number(spent, "spent"))

    def stop_reason(self, *, spent: float, elapsed_wall: float, best_score: float) -> str | None:
        """Return a published stop reason when a hard upper bound is reached."""
        if _number(spent, "spent") >= self.spend_cap:
            return "budget_cap"
        if _number(elapsed_wall, "elapsed_wall") >= self.wall_seconds:
            return "time"
        if _number(best_score, "best_score") >= 1:
            return "ceiling"
        return None

    def can_expand(
        self,
        *,
        spent: float,
        elapsed_wall: float,
        estimated_cost: float,
        estimated_seconds: float = 0.0,
    ) -> bool:
        """Preserve the hard reserve before starting an estimated experiment."""
        cost = _number(estimated_cost, "estimated_cost")
        seconds = _number(estimated_seconds, "estimated_seconds")
        if cost < 0 or seconds < 0:
            raise ScheduleError("Estimated cost and seconds cannot be negative")
        return (
            _number(spent, "spent") + cost <= self.spend_cap
            and _number(elapsed_wall, "elapsed_wall") + seconds <= self.wall_seconds
        )

    def review_trigger(
        self,
        *,
        non_improving_nodes: int,
        spent: float,
        best_score: float,
        baseline_score: float = 0.0,
        expected_score_at_spend: float | None = None,
    ) -> str | None:
        """Check plateau first, then the planner's score-at-current-spend forecast."""
        if non_improving_nodes >= int(self.config["plateau_nodes"]):
            return "plateau"
        threshold = self.config["behind_schedule"]
        if expected_score_at_spend is None:
            return None
        if _number(spent, "spent") / self.budget < _number(threshold["spent_ratio"], "spent_ratio"):
            return None
        planned_gain = _number(expected_score_at_spend, "expected_score_at_spend") - _number(baseline_score, "baseline_score")
        if planned_gain <= 0:
            return None
        actual_gain = _number(best_score, "best_score") - _number(baseline_score, "baseline_score")
        if actual_gain < planned_gain * _number(threshold["progress_ratio"], "progress_ratio"):
            return "behind_schedule"
        return None

    @staticmethod
    def priority_for(
        estimate: Mapping[str, float],
        *,
        cited_effects: Sequence[Mapping[str, Any]] = (),
        past_predictions: Sequence[Mapping[str, float]] = (),
    ) -> float:
        """Rank gain per cost with guarded historical and forecast calibration.

        Each cited effect can contain uses, mean_gain, and mean_cost. Past predictions
        contain gain and actual_gain. Three pseudo-observations keep small samples
        near the planner estimate; the past-gain multiplier is bounded to 0.5–1.5.
        """
        plan_gain = max(0.0, _number(estimate.get("gain"), "estimate.gain"))
        plan_cost = _number(estimate.get("cost"), "estimate.cost")
        if plan_cost <= 0:
            raise ScheduleError("Estimated cost must be positive to rank hypotheses")
        weighted_gain = 0.0
        weighted_cost = 0.0
        weight_sum = 0.0
        for effect in cited_effects:
            uses = max(0.0, _number(effect.get("uses", 0), "effect.uses"))
            if not uses:
                continue
            gain = max(0.0, _number(effect.get("mean_gain", 0), "effect.mean_gain"))
            cost = _number(effect.get("mean_cost", plan_cost), "effect.mean_cost")
            if cost <= 0:
                continue
            weight = min(uses, 10.0)
            weighted_gain += weight * gain
            weighted_cost += weight * cost
            weight_sum += weight
        if weight_sum:
            share = weight_sum / (weight_sum + 3.0)
            plan_gain = (1.0 - share) * plan_gain + share * (weighted_gain / weight_sum)
            plan_cost = (1.0 - share) * plan_cost + share * (weighted_cost / weight_sum)
        ratios = []
        for observation in past_predictions:
            prior = _number(observation.get("gain", 0), "prediction.gain")
            if prior > 0:
                actual = _number(observation.get("actual_gain", 0), "prediction.actual_gain")
                ratios.append(actual / prior)
        calibration = min(1.5, max(0.5, median(ratios))) if ratios else 1.0
        return max(0.0, plan_gain * calibration / plan_cost)

    def exploration_turn(self, expanded_count: int) -> bool:
        return (expanded_count + 1) % int(self.config["explore_every"]) == 0

    def choose_next(
        self,
        pending: Sequence[Mapping[str, Any]],
        *,
        expanded_count: int,
        best_node_id: str,
    ) -> Mapping[str, Any] | None:
        """One in every explore_every expansions favors an alternative direction."""
        queued = [node for node in pending if node.get("status", "queued") == "queued"]
        if not queued:
            return None
        if self.exploration_turn(expanded_count):
            preferred = [node for node in queued if node.get("origin") == "explore" or node.get("parent") != best_node_id]
        else:
            preferred = [node for node in queued if node.get("parent") == best_node_id and node.get("origin") != "explore"]
        pool = preferred or queued
        return max(pool, key=lambda node: (_number(node.get("priority", 0), "priority"), -int(str(node.get("node_id", "n9999"))[1:])))

    def choose_parent(self, accepted: Sequence[Mapping[str, Any]], *, expanded_count: int) -> str:
        """Default to best; an exploration turn branches from the runner-up if any."""
        ranked = sorted(
            (node for node in accepted if node.get("status") == "accepted"),
            key=lambda node: (-_number(node["score"]["dev"], "score.dev"), str(node["node_id"])),
        )
        if not ranked:
            raise ScheduleError("No accepted root node")
        index = 1 if self.exploration_turn(expanded_count) and len(ranked) > 1 else 0
        return str(ranked[index]["node_id"])
