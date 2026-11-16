"""Responses API adapter with auditable, exclusive token accounting.

The stable role instructions are sent before the changing JSON task state so
identical prefixes can benefit from prompt caching. Each invocation is one
stateless request; the runner decides what compact state the role sees.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import secrets
from time import perf_counter
from typing import Any, Mapping


ROLES = frozenset({"planner", "executor", "diagnoser", "summarizer"})
PURPOSES = frozenset({"plan", "quote", "code", "fix", "diagnose", "review", "summarize"})
EFFORTS = frozenset({"low", "medium", "high"})  # ledger/1 enum

# Two ways Qwen-family OpenAI-compatible gateways take the "do not think" switch.
# Both go through the SDK's extra_body, so they never touch the Responses fields.
THINKING_SWITCH_STYLES = ("enable_thinking", "chat_template_kwargs")
DEFAULT_THINKING_SWITCH_STYLE = THINKING_SWITCH_STYLES[0]
ROLE_CONTROL_KEYS = frozenset({"thinking", "max_output_tokens"})

# Roles that run once, at the end of a task. They are held back by their own
# worst case rather than the full per-call reserve; see ModelClient._reserve_for.
WRAPUP_ROLES = frozenset({"summarizer"})
# Deliberately pessimistic request-size estimate (mixed Chinese/JSON prompts
# measure 2.5-4 bytes per token; digit-heavy JSON gets close to 1).
_INPUT_BYTES_PER_TOKEN = 1.5
_REQUEST_OVERHEAD_TOKENS = 256


class ModelCallError(RuntimeError):
    """A model request failed, was refused, or returned unusable output."""


class BudgetReserveError(ModelCallError):
    """A new model call would consume the reserved task budget."""


@dataclass(frozen=True)
class TokenUsage:
    """Disjoint billing buckets expected by ledger/1.

    Responses API input_tokens includes cached_tokens, while output_tokens
    includes reasoning_tokens. Subtract the nested categories here so the
    four ledger price buckets never double count the same token.
    """

    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning: int = 0


@dataclass(frozen=True)
class ModelResult:
    data: dict[str, Any]
    usage: TokenUsage
    seconds: float
    cost: float
    prompt_hash: str
    response_id: str | None


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def token_usage_from_response(response: Any) -> TokenUsage:
    """Convert inclusive Responses API counters to disjoint ledger buckets."""
    usage = _field(response, "usage")
    if usage is None:
        raise ModelCallError("Model returned no token usage")
    input_total = _field(usage, "input_tokens")
    output_total = _field(usage, "output_tokens")
    cached = _field(_field(usage, "input_tokens_details", {}), "cached_tokens", 0) or 0
    reasoning = _field(_field(usage, "output_tokens_details", {}), "reasoning_tokens", 0) or 0
    if (any(type(value) is not int or value < 0 for value in
            (input_total, output_total, cached, reasoning))
            or cached > input_total or reasoning > output_total):
        raise ModelCallError("Invalid or inconsistent token usage returned by model")
    return TokenUsage(input_total - cached, cached, output_total - reasoning, reasoning)


def _has_refusal(response: Any) -> bool:
    for item in _field(response, "output", []) or []:
        for content in _field(item, "content", []) or []:
            if _field(content, "type") == "refusal" or _field(content, "refusal"):
                return True
    return False


def _ledger_usage(usage: TokenUsage) -> Any:
    """Use C's exact Usage class when its ledger module is installed."""
    try:
        usage_type = getattr(importlib.import_module("ledger"), "Usage")
    except (ImportError, AttributeError):
        return usage
    return usage_type(
        input=usage.input,
        cached_input=usage.cached_input,
        output=usage.output,
        reasoning=usage.reasoning,
    )


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate the small JSON Schema subset used by the four role contracts.

    Some Responses-compatible gateways accept json_schema metadata without
    enforcing it. Reject a shape mismatch instead of silently treating a
    malformed planner response as an empty plan.
    """
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise ModelCallError(f"{path} must be an object")
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - value.keys()
        if missing:
            raise ModelCallError(f"{path} is missing {sorted(missing)}")
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ModelCallError(f"{path} has unexpected keys {sorted(value.keys() - properties.keys())}")
        for key, child in properties.items():
            if key in value:
                _validate_schema(value[key], child, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ModelCallError(f"{path} must be an array")
        for index, item in enumerate(value):
            _validate_schema(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise ModelCallError(f"{path} must be a string")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ModelCallError(f"{path} must be a finite number")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelCallError(f"{path} must be an integer")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ModelCallError(f"{path} must be a boolean")
    else:
        raise ValueError(f"Unsupported role schema type: {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ModelCallError(f"{path} is not in the allowed enum")


def _parse_json_text(output_text: str) -> Any:
    """Allow a single fenced JSON object from gateways ignoring format hints."""
    text = output_text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n"):-len("\n```")]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[len("```\n"):-len("\n```")]
    return json.loads(text)


def check_thinking_switch_style(style: Any) -> str:
    if not isinstance(style, str) or style not in THINKING_SWITCH_STYLES:
        raise ValueError(f"thinking_switch_style must be one of {list(THINKING_SWITCH_STYLES)}")
    return style


def thinking_off_extra_body(style: str) -> dict[str, Any]:
    """The extra request fields that switch thinking off for the chosen style."""
    if check_thinking_switch_style(style) == "enable_thinking":
        return {"enable_thinking": False}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _check_output_limit(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_role_controls(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate agent/1 ``role_controls``: per-role ``thinking`` and ``max_output_tokens``.

    A missing key means "behave as before" for every role.
    """
    raw = config.get("role_controls")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("role_controls must be an object keyed by role")
    controls: dict[str, dict[str, Any]] = {}
    for role, spec in raw.items():
        if role not in ROLES:
            raise ValueError(f"role_controls names an unknown role: {role!r}")
        if not isinstance(spec, Mapping) or set(spec) - ROLE_CONTROL_KEYS:
            raise ValueError(f"role_controls.{role} may only set {sorted(ROLE_CONTROL_KEYS)}")
        entry: dict[str, Any] = {}
        if "thinking" in spec:
            if not isinstance(spec["thinking"], bool):
                raise ValueError(f"role_controls.{role}.thinking must be a boolean")
            entry["thinking"] = spec["thinking"]
        if "max_output_tokens" in spec:
            entry["max_output_tokens"] = _check_output_limit(
                spec["max_output_tokens"], f"role_controls.{role}.max_output_tokens")
        controls[role] = entry
    return controls


class ModelClient:
    """One-model, many-effort JSON client, injectable for deterministic tests.

    ledger must expose the C-owned Ledger.llm_call interface. On transport or
    parse errors we still book any usage returned, then raise ModelCallError.
    """

    def __init__(
        self,
        model: str,
        ledger: Any | None,
        *,
        client: Any | None = None,
        max_output_tokens: int = 30000,
        budget_limit: float | None = None,
        reserve_cost: float = 0.0,
        audit_dir: Path | None = None,
        clock: Any = perf_counter,
        thinking_switch_style: str = DEFAULT_THINKING_SWITCH_STYLE,
    ) -> None:
        if not model:
            raise ValueError("A model name must come from the run manifest or caller")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        check_thinking_switch_style(thinking_switch_style)
        if (isinstance(reserve_cost, bool) or not isinstance(reserve_cost, (int, float))
                or not math.isfinite(reserve_cost) or reserve_cost < 0):
            raise ValueError("reserve_cost must be nonnegative and finite")
        if budget_limit is not None and (isinstance(budget_limit, bool)
                or not isinstance(budget_limit, (int, float)) or not math.isfinite(budget_limit)
                or budget_limit <= 0 or reserve_cost > budget_limit):
            raise ValueError("budget_limit must be positive and cover the reserve")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install research-agent-abc[llm] to make live model calls") from exc
            # Budget accounting cannot see usage from hidden SDK retries.
            # Keep each recorded call to one gateway attempt and cap stalls.
            client = OpenAI(timeout=180.0, max_retries=0)
        self.model = model
        self.ledger = ledger
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.budget_limit = float(budget_limit) if budget_limit is not None else None
        self.reserve_cost = float(reserve_cost)
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        self.clock = clock
        self.thinking_switch_style = thinking_switch_style

    def _reserve_for(self, role: str, instructions: str, user_input: str,
                     schema: Mapping[str, Any], output_cap: int,
                     wrapup_reserve: bool = False) -> float:
        """Cost held back before dispatch; the call is refused if it would not fit.

        Every role is normally held to the full ``reserve_cost`` (30% of the task
        budget in the token study). That starved the summarizer, which only runs
        after the experiments have spent most of the budget. When the caller opts
        in (agent/1 ``summarizer_reserve: "worst_case"``), wrap-up roles are
        instead held to their own worst case: a pessimistic estimate of the
        request's input tokens plus the output cap, never more than
        ``reserve_cost``. The call still cannot take the task past
        ``budget_limit``, but it is no longer refused for headroom it does not use.

        The estimate is in tokens, so it only lowers the reserve where one cost
        unit is one token (the TOKENS/TEST token studies). For any other currency
        the min() keeps the configured reserve, i.e. the old behaviour.
        """
        if role not in WRAPUP_ROLES or not wrapup_reserve:
            return self.reserve_cost
        request = (instructions + user_input
                   + json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        input_tokens = math.ceil(len(request.encode("utf-8")) / _INPUT_BYTES_PER_TOKEN)
        return min(self.reserve_cost, float(input_tokens + _REQUEST_OVERHEAD_TOKENS + output_cap))

    def call_json(
        self,
        *,
        role: str,
        purpose: str,
        effort: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        schema: Mapping[str, Any],
        node_id: str | None = None,
        thinking: bool | None = None,
        max_output_tokens: int | None = None,
        thinking_switch_style: str | None = None,
        wrapup_reserve: bool = False,
    ) -> ModelResult:
        """One stateless JSON call.

        thinking=False asks the gateway not to think (extra_body switch, style
        from thinking_switch_style or the client default); None and True leave
        the request exactly as before. max_output_tokens is a per-call limit and
        can only lower the client-wide limit. wrapup_reserve=True lets a wrap-up
        role reserve only its own worst case (see _reserve_for); False keeps the
        full reserve_cost gate.
        """
        if role not in ROLES or purpose not in PURPOSES or effort not in EFFORTS:
            raise ValueError("role, purpose, or effort is not in ledger/1")
        if not system_prompt.strip():
            raise ValueError("Role instructions cannot be empty")
        if thinking is not None and not isinstance(thinking, bool):
            raise ValueError("thinking must be True, False or None")
        style = check_thinking_switch_style(
            self.thinking_switch_style if thinking_switch_style is None else thinking_switch_style)
        output_cap = self.max_output_tokens
        if max_output_tokens is not None:
            output_cap = min(output_cap, _check_output_limit(max_output_tokens, "max_output_tokens"))
        user_input = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        schema_name = f"{role}_{purpose}"
        instructions = (system_prompt + "\n\nOutput JSON contract: Return exactly one JSON object "
                        "matching this schema. Use these exact field names, include every "
                        "required field, do not add other fields, and use JSON numbers "
                        "rather than numeric strings. No Markdown or prose.\n"
                        + json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if self.budget_limit is not None:
            if self.ledger is None or not hasattr(self.ledger, "spent"):
                raise ValueError("A budgeted model call requires a spending ledger")
            spent = float(self.ledger.spent())
            if not math.isfinite(spent) or spent < 0:
                raise ValueError("Ledger reported invalid spend")
            if spent + self._reserve_for(role, instructions, user_input, schema, output_cap,
                                       wrapup_reserve) > self.budget_limit:
                raise BudgetReserveError("Not enough budget remains for a reserved model call")
        hashed = {"instructions": instructions, "input": user_input, "schema": schema, "effort": effort}
        if thinking is False:
            hashed["thinking"] = False  # absent otherwise, so default hashes stay as they were
        prompt_hash = hashlib.sha256(
            json.dumps(
                hashed,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": user_input,
            "reasoning": {"effort": effort},
            "text": {"format": {"type": "json_schema", "name": schema_name, "schema": dict(schema), "strict": True}},
            "max_output_tokens": output_cap,
        }
        if thinking is False:
            request["extra_body"] = thinking_off_extra_body(style)
        started = self.clock()
        response: Any | None = None
        usage = TokenUsage()
        usage_status = "unknown"
        data: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            response = self.client.responses.create(**request)
            usage = token_usage_from_response(response)
            usage_status = "reported"
            status = _field(response, "status", "completed")
            if status != "completed":
                reason = _field(_field(response, "incomplete_details", {}), "reason", status)
                raise ModelCallError(f"Model response {status}: {reason}")
            if _has_refusal(response):
                raise ModelCallError("Model refused the structured response")
            output_text = _field(response, "output_text", "")
            if not output_text:
                raise ModelCallError("Model returned no JSON text")
            parsed = _parse_json_text(output_text)
            if not isinstance(parsed, dict):
                raise ModelCallError("Model JSON result must be an object")
            _validate_schema(parsed, schema)
            data = parsed
        except Exception as exc:
            error = exc
        seconds = max(0.0, float(self.clock() - started))
        cost = 0.0
        if self.ledger is not None:
            cost = self.ledger.llm_call(
                node_id=node_id,
                role=role,
                purpose=purpose,
                effort=effort,
                model=self.model,
                usage=_ledger_usage(usage),
                usage_status=usage_status,
                seconds=seconds,
                prompt_hash=prompt_hash,
                ok=error is None,
                error=str(error) if error is not None else None,
            )
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = self.audit_dir / f"response-{secrets.token_hex(8)}.json"
            audit_path.write_text(json.dumps({
                "schema": "model_response_audit/1", "role": role, "purpose": purpose,
                "node_id": node_id, "model": self.model, "prompt_hash": prompt_hash,
                "response_id": _field(response, "id"),
                "status": _field(response, "status"),
                "output_text": _field(response, "output_text"),
                "thinking": thinking, "max_output_tokens": output_cap,
                "tokens": vars(usage), "cost": cost,
                "usage_status": usage_status,
                "error": str(error) if error is not None else None,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if error is not None:
            raise ModelCallError(str(error)) from error
        assert data is not None and response is not None
        return ModelResult(
            data=data,
            usage=usage,
            seconds=seconds,
            cost=float(cost),
            prompt_hash=prompt_hash,
            response_id=_field(response, "id"),
        )
