"""Bounded, evidence-calibrated routing between the Qwen proxy and two remote models."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .model import BudgetReserveError, ModelResult


UPGRADABLE = frozenset({"planner", "executor", "diagnoser"})


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema") != "route_policy/1":
        raise ValueError("Expected route_policy/1")
    default = policy.get("default_model")
    role_models = policy.get("role_models")
    eligible = policy.get("eligible_roles")
    thresholds = policy.get("thresholds")
    if not isinstance(default, str) or not default.strip():
        raise ValueError("Routing default model is missing")
    if (not isinstance(role_models, dict) or set(role_models) != UPGRADABLE
            or any(not isinstance(value, str) or not value.strip() or value == default
                   for value in role_models.values())):
        raise ValueError("Routing needs one non-default model for each upgradable role")
    if (not isinstance(eligible, list) or len(set(eligible)) != len(eligible)
            or not set(eligible) <= UPGRADABLE):
        raise ValueError("Invalid eligible routing roles")
    if not isinstance(thresholds, dict) or set(thresholds) != {"executor_failures", "plateau_nodes", "score_drop"}:
        raise ValueError("Invalid route thresholds")
    for key, low, high in (("executor_failures", 1, 3), ("plateau_nodes", 2, 4)):
        value = thresholds[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid route threshold {key}")
    drop = thresholds["score_drop"]
    if isinstance(drop, bool) or not isinstance(drop, (int, float)) or not math.isfinite(drop) or not 0.05 <= drop <= 0.5:
        raise ValueError("Invalid score-drop threshold")
    cap = policy.get("remote_call_cap")
    if type(cap) is not int or not 1 <= cap <= 2:
        raise ValueError("Per-task remote-call cap must be one or two")
    calibration = policy.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("schema") != "route_calibration/1":
        raise ValueError("Routing policy must cite a calibration record")


class RoutingClient:
    """Expose ModelClient.call_json while sealing every choice in the task ledger.

    There is no automatic retry or Qwen fallback after a remote failure: its
    failed request, reported token usage, and negative outcome remain visible.
    """

    def __init__(self, clients: Mapping[str, Any], ledger: Any, policy: Mapping[str, Any]):
        validate_policy(policy)
        catalog = {policy["default_model"], *policy["role_models"].values()}
        if set(clients) != catalog:
            raise ValueError("Routing clients differ from the policy model catalog")
        self.clients = dict(clients)
        self.ledger = ledger
        self.policy = dict(policy)

    @staticmethod
    def _score(value: Any) -> float | None:
        if isinstance(value, Mapping):
            value = value.get("dev")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None

    def _trigger(self, role: str, payload: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
        thresholds = self.policy["thresholds"]
        state: dict[str, Any] = {}
        if role == "executor":
            attempt = payload.get("attempt")
            previous_error = bool(payload.get("previous_error"))
            state.update(attempt=attempt, previous_error=previous_error)
            if type(attempt) is int and attempt > thresholds["executor_failures"] and previous_error:
                return "consecutive_executor_failures", state
        elif role == "planner":
            nodes = payload.get("recent_experiments", payload.get("recent_nodes", []))
            trailing = 0
            if isinstance(nodes, list):
                for node in reversed(nodes):
                    if isinstance(node, Mapping) and node.get("status") in {"rejected", "failed"}:
                        trailing += 1
                    else:
                        break
            state["trailing_non_improving_nodes"] = trailing
            if trailing >= thresholds["plateau_nodes"]:
                return "score_plateau", state
        elif role == "diagnoser":
            trigger = payload.get("trigger")
            node = payload.get("node", {})
            score = self._score(node.get("score")) if isinstance(node, Mapping) else None
            best = self._score(payload.get("best_score"))
            drop = best - score if best is not None and score is not None else None
            state.update(diagnosis_trigger=trigger, score_drop=drop)
            if trigger == "failed_after_retries":
                return "failed_after_retries", state
            if trigger == "abnormal_score" and drop is not None and drop >= thresholds["score_drop"]:
                return "score_drop", state
        return None, state

    def call_json(self, *, role: str, purpose: str, effort: str,
                  system_prompt: str, payload: Mapping[str, Any],
                  schema: Mapping[str, Any], node_id: str | None = None) -> ModelResult:
        prior = self.ledger._events()
        remote_models = set(self.policy["role_models"].values())
        remote_calls = sum(event["kind"] == "llm_call" and event["model"] in remote_models for event in prior)
        trigger, state = self._trigger(role, payload)
        model = self.policy["default_model"]
        reason = trigger or "default_qwen"
        if role in self.policy["eligible_roles"] and trigger:
            if remote_calls < self.policy["remote_call_cap"]:
                model = self.policy["role_models"][role]
            else:
                reason = "remote_cap_exhausted"
        elif trigger:
            reason = "role_not_eligible"
        error: Exception | None = None
        result: ModelResult | None = None
        try:
            result = self.clients[model].call_json(
                role=role, purpose=purpose, effort=effort,
                system_prompt=system_prompt, payload=payload,
                schema=schema, node_id=node_id,
            )
        except Exception as exc:
            error = exc
        after = self.ledger._events()
        model_calls = [event for event in after[len(prior):] if event["kind"] == "llm_call"]
        if len(model_calls) > 1:
            raise ValueError("One routed role invocation produced multiple model calls")
        call = model_calls[0] if model_calls else None
        self.ledger.emit(
            "route_decision", node_id=node_id, role=role, purpose=purpose,
            selected_model=model, default_model=self.policy["default_model"],
            reason=reason, state={**state, "remote_calls_before": remote_calls},
            remote_call_cap=self.policy["remote_call_cap"],
            spent_before=sum(float(event["cost"]) for event in prior if event["kind"] == "llm_call"),
            llm_seq=call["seq"] if call else None,
            ok=error is None, error=str(error)[:300] if error is not None else None,
        )
        if error is not None:
            raise error
        assert result is not None
        return result


__all__ = ["RoutingClient", "validate_policy"]
