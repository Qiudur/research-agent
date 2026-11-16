"""Four research roles with bounded, structured handoffs to the runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .experience_script import writer_mode
from .model import DEFAULT_THINKING_SWITCH_STYLE, ModelClient, check_thinking_switch_style, parse_role_controls
from .scheduler import validate_options


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_STRING = {"type": "string"}
_NUMBER = {"type": "number"}
_STRINGS = {"type": "array", "items": _STRING}
_HYPOTHESIS = _object(
    {
        "hypothesis": _STRING,
        "rationale": _STRING,
        "estimate": _object({"gain": _NUMBER, "cost": _NUMBER, "seconds": _NUMBER}),
        "cites": _STRINGS,
    }
)

PLAN_SCHEMA = _object(
    {
        "hypotheses": {"type": "array", "items": _HYPOTHESIS},
        "predicted": _object({"cost": _NUMBER, "seconds": _NUMBER, "score": _NUMBER}),
        "budget_allocation": {"type": "array", "items": _object({"direction": _STRING, "amount": _NUMBER})},
        "expected_progress": {"type": "array", "items": _object({"spent_ratio": _NUMBER, "score": _NUMBER})},
    }
)

# planner_mode "lean": budget_allocation is no longer required. It stays an
# allowed property, so a plan may omit it or return an empty array; a filled
# one is still validated. The company gateway does not enforce schemas (the
# client validates field by field); an OpenAI-style strict endpoint would
# insist that every property be required.
PLAN_SCHEMA_LEAN = {
    **PLAN_SCHEMA,
    "required": [key for key in PLAN_SCHEMA["required"] if key != "budget_allocation"],
}

EXECUTE_SCHEMA = _object({"code": _STRING, "notes": _STRING})

DIAGNOSE_SCHEMA = _object({"hypothesis": _STRING, "rationale": _STRING, "advice": _STRING})

REVIEW_SCHEMA = _object(
    {
        "decision": {"type": "string", "enum": ["continue", "switch", "reallocate", "stop"]},
        "budget_moves": {"type": "array", "items": _object({"from": _STRING, "to": _STRING, "amount": _NUMBER})},
        "note": _STRING,
    }
)

_DRAFT = _object(
    {
        "scope": {"type": "string", "enum": ["domain", "method"]},
        "scenario": {"type": "string", "enum": ["tabular", "speedup", "quant", "any"]},
        "triggers": _STRINGS,
        "conditions": _STRING,
        "action": _STRING,
        "negative": {"type": "boolean"},
        "node_ids": _STRINGS,
    }
)

_NARROWING = _object({"entry_id": _STRING, "conditions": _STRING, "reason": _STRING})

SUMMARIZE_SCHEMA = _object({
    "drafts": {"type": "array", "items": _DRAFT},
    "narrowings": {"type": "array", "items": _NARROWING},
})

SCHEMAS = {
    "plan": PLAN_SCHEMA,
    "code": EXECUTE_SCHEMA,
    "fix": EXECUTE_SCHEMA,
    "diagnose": DIAGNOSE_SCHEMA,
    "review": REVIEW_SCHEMA,
    "summarize": SUMMARIZE_SCHEMA,
}


class RoleAgents:
    """Load fixed role prompts once and send only compact changing state.

    Methods return JSON objects matching the keys consumed by the A runner.
    ModelClient logs each call through the C-owned ledger implementation.
    """

    def __init__(
        self,
        model: ModelClient,
        config: Mapping[str, Any],
        *,
        project_root: Path,
        prompt_texts: Mapping[str, str] | None = None,
    ) -> None:
        if config.get("schema") != "agent/1":
            raise ValueError("Unknown agent config schema")
        validate_options(config)
        self.model = model
        self.planner_mode = config.get("planner_mode") or "standard"
        self.effort = dict(config["effort"])
        # Optional agent/1 keys; absent means every role behaves exactly as before.
        self.role_controls = parse_role_controls(config)
        self.thinking_switch_style = check_thinking_switch_style(
            config.get("thinking_switch_style", DEFAULT_THINKING_SWITCH_STYLE))
        writer_mode(config)  # a typo in experience_writer should fail now, not after the budget is spent
        self.summarizer_reserve = config.get("summarizer_reserve") or "standard"
        self.prompts: dict[str, str] = {}
        resolved_root = project_root.resolve()
        for role in ("planner", "executor", "diagnoser", "summarizer"):
            if prompt_texts is not None and role in prompt_texts:
                prompt = prompt_texts[role]
            else:
                prompt_path = Path(config["prompts"][role])
                if not prompt_path.is_absolute():
                    prompt_path = project_root / prompt_path
                if not prompt_path.resolve().is_relative_to(resolved_root):
                    raise ValueError(f"Prompt path for {role} escapes project root")
                prompt = prompt_path.read_text(encoding="utf-8")
            if not prompt.strip():
                raise ValueError(f"Empty prompt for {role}")
            self.prompts[role] = prompt

    def _controls(self, role: str) -> dict[str, Any]:
        """Per-call limits from role_controls; empty (no new arguments) by default."""
        controls = dict(self.role_controls.get(role, {}))
        if controls.get("thinking") is False:
            controls["thinking_switch_style"] = self.thinking_switch_style
        if role == "summarizer" and self.summarizer_reserve == "worst_case":
            controls["wrapup_reserve"] = True
        return controls

    def _call(
        self, role: str, purpose: str, payload: Mapping[str, Any], node_id: str | None,
        schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.model.call_json(
            role=role,
            purpose=purpose,
            effort=self.effort[role],
            system_prompt=self.prompts[role],
            payload=payload,
            schema=SCHEMAS[purpose] if schema is None else schema,
            node_id=node_id,
            **self._controls(role),
        ).data

    def plan(self, payload: Mapping[str, Any], *, node_id: str | None = None) -> dict[str, Any]:
        """Return hypotheses plus predicted cost, seconds and score.

        planner_mode "lean" uses the plan schema without budget_allocation.
        """
        schema = PLAN_SCHEMA_LEAN if self.planner_mode == "lean" else None
        return self._call("planner", "plan", payload, node_id, schema)

    def execute(self, payload: Mapping[str, Any], *, fix: bool = False, node_id: str | None = None) -> dict[str, Any]:
        """Return complete solution.py code; fix=True logs purpose=fix."""
        return self._call("executor", "fix" if fix else "code", payload, node_id)

    def diagnose(self, payload: Mapping[str, Any], *, node_id: str | None = None) -> dict[str, Any]:
        return self._call("diagnoser", "diagnose", payload, node_id)

    def review(self, payload: Mapping[str, Any], *, node_id: str | None = None) -> dict[str, Any]:
        return self._call("planner", "review", payload, node_id)

    def summarize(self, payload: Mapping[str, Any], *, node_id: str | None = None) -> dict[str, Any]:
        return self._call("summarizer", "summarize", payload, node_id)
