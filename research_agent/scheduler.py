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


# Optional agent/1 switches. Every key is absent in config/agent.json, and an
# absent (or null) key keeps the original behavior.
SCENARIO_KEYS = ("spd", "tab", "quant")  # task_id prefixes
STOP_LOSS_KEYS = ("first_n_no_gain", "consecutive_no_gain")
PLANNER_MODES = ("standard", "lean")
SUMMARIZER_RESERVES = ("standard", "worst_case")
BASELINE_NODE_ID = "n0000"
MAX_STOP_LOSS_NODES = 100
MAX_HYPOTHESES_PER_PLAN = 20


def scenario_key(task_id: str | None) -> str | None:
    """Return the settings key (spd, tab or quant) for a task id, else None."""
    prefix = str(task_id or "").partition("-")[0]
    return prefix if prefix in SCENARIO_KEYS else None


def _bounded_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ScheduleError(f"{name} must be an integer from {low} to {high}")
    return value


def _scenario_table(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleError(f"{name} must be an object keyed by {'/'.join(SCENARIO_KEYS)}")
    unknown = sorted(str(key) for key in value if key not in SCENARIO_KEYS)
    if unknown:
        raise ScheduleError(f"{name} has unknown scenario keys {unknown}")
    return value


def validate_options(config: Mapping[str, Any]) -> None:
    """Type and range checks for the optional switches (null means unset)."""
    stop_loss = config.get("stop_loss")
    if stop_loss is not None:
        for scenario, rule in _scenario_table(stop_loss, "stop_loss").items():
            if not isinstance(rule, Mapping):
                raise ScheduleError(f"stop_loss.{scenario} must be an object")
            unknown = sorted(str(key) for key in rule if key not in STOP_LOSS_KEYS)
            if unknown:
                raise ScheduleError(f"stop_loss.{scenario} has unknown keys {unknown}")
            for key, value in rule.items():
                if value is not None:
                    _bounded_int(value, f"stop_loss.{scenario}.{key}", 1, MAX_STOP_LOSS_NODES)
    for name in ("ceiling_excludes_baseline", "failure_feedback"):
        if config.get(name) is not None and not isinstance(config[name], bool):
            raise ScheduleError(f"{name} must be true or false")
    mode = config.get("planner_mode")
    if mode is not None and mode not in PLANNER_MODES:
        raise ScheduleError(f"planner_mode must be one of {list(PLANNER_MODES)}")
    reserve = config.get("summarizer_reserve")
    if reserve is not None and reserve not in SUMMARIZER_RESERVES:
        raise ScheduleError(f"summarizer_reserve must be one of {list(SUMMARIZER_RESERVES)}")
    per_plan = config.get("hypotheses_per_plan")
    if per_plan is not None:
        _bounded_int(per_plan, "hypotheses_per_plan", 1, MAX_HYPOTHESES_PER_PLAN)
    priors = config.get("cost_priors")
    if priors is not None:
        for scenario, value in _scenario_table(priors, "cost_priors").items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ScheduleError(f"cost_priors.{scenario} must be a positive finite number")


@dataclass(frozen=True)
class Scheduler:
    """Rules-only scheduler: no manager model and no hidden budget changes.

    task_id selects the per-scenario (spd/tab/quant) entries of the optional
    stop_loss and cost_priors settings; without it they are ignored.
    """

    config: Mapping[str, Any]
    hard_caps: Mapping[str, Any]
    budget: float
    wall_seconds: float
    task_id: str = ""

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
        validate_options(self.config)

    @property
    def spend_cap(self) -> float:
        return self.budget * float(self.hard_caps["budget_ratio"])

    @property
    def scenario(self) -> str | None:
        """spd, tab or quant, taken from the task id prefix."""
        return scenario_key(self.task_id)

    def _scenario_setting(self, name: str) -> Any:
        table = self.config.get(name)
        if not table or self.scenario is None:
            return None
        return table.get(self.scenario)

    @property
    def cost_prior(self) -> float | None:
        """Fixed per-hypothesis cost for this scenario, replacing the planner's guess."""
        value = self._scenario_setting("cost_priors")
        return None if value is None else float(value)

    def estimated_cost(self, planner_cost: float) -> float:
        """The cost to schedule with: the scenario prior when set, else the planner's estimate."""
        prior = self.cost_prior
        return prior if prior is not None else _number(planner_cost, "estimated_cost")

    def budget_remaining(self, spent: float) -> float:
        return max(0.0, self.spend_cap - _number(spent, "spent"))

    def stop_reason(
        self,
        *,
        spent: float,
        elapsed_wall: float,
        best_score: float,
        best_node_id: str | None = None,
    ) -> str | None:
        """Return a published stop reason when a hard upper bound is reached.

        With ceiling_excludes_baseline, a baseline (n0000) that already scores
        1 or more is not a ceiling; only a better, accepted node can reach it.
        """
        if _number(spent, "spent") >= self.spend_cap:
            return "budget_cap"
        if _number(elapsed_wall, "elapsed_wall") >= self.wall_seconds:
            return "time"
        if _number(best_score, "best_score") >= 1:
            if self.config.get("ceiling_excludes_baseline") and best_node_id == BASELINE_NODE_ID:
                return None
            return "ceiling"
        return None

    def stop_loss_reason(self, statuses: Sequence[str]) -> str | None:
        """Return "stop_loss" when this scenario's early-exit rule is met.

        statuses holds the final status (accepted, rejected or failed) of each
        completed non-root node in completion order. first_n_no_gain stops when
        the first N of them include no accepted node; consecutive_no_gain stops
        when the latest K include none.
        """
        rule = self._scenario_setting("stop_loss")
        if not rule:
            return None
        first_n = rule.get("first_n_no_gain")
        if first_n and len(statuses) >= first_n and "accepted" not in statuses[:first_n]:
            return "stop_loss"
        streak = rule.get("consecutive_no_gain")
        if streak and len(statuses) >= streak and "accepted" not in statuses[-streak:]:
            return "stop_loss"
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
        cost = self.estimated_cost(estimated_cost)
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
        cost_prior: float | None = None,
    ) -> float:
        """Rank gain per cost with guarded historical and forecast calibration.

        Each cited effect can contain uses, mean_gain, and mean_cost. Past predictions
        contain gain and actual_gain. Three pseudo-observations keep small samples
        near the planner estimate; the past-gain multiplier is bounded to 0.5–1.5.
        A cost_prior replaces the planner's own cost estimate (which must still be valid).
        """
        plan_gain = max(0.0, _number(estimate.get("gain"), "estimate.gain"))
        plan_cost = _number(estimate.get("cost"), "estimate.cost")
        if plan_cost <= 0:
            raise ScheduleError("Estimated cost must be positive to rank hypotheses")
        if cost_prior is not None:
            plan_cost = _number(cost_prior, "cost_prior")
            if plan_cost <= 0:
                raise ScheduleError("cost_prior must be positive")
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
