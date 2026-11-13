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
from time import perf_counter
from typing import Any, Mapping


ROLES = frozenset({"planner", "executor", "diagnoser", "summarizer"})
PURPOSES = frozenset({"plan", "quote", "code", "fix", "diagnose", "review", "summarize"})
EFFORTS = frozenset({"low", "medium", "high"})  # ledger/1 enum


class ModelCallError(RuntimeError):
    """A model request failed, was refused, or returned unusable output."""


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
        return TokenUsage()
    input_total = int(_field(usage, "input_tokens", 0) or 0)
    output_total = int(_field(usage, "output_tokens", 0) or 0)
    cached = int(_field(_field(usage, "input_tokens_details", {}), "cached_tokens", 0) or 0)
    reasoning = int(_field(_field(usage, "output_tokens_details", {}), "reasoning_tokens", 0) or 0)
    if min(input_total, output_total, cached, reasoning) < 0 or cached > input_total or reasoning > output_total:
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
        clock: Any = perf_counter,
    ) -> None:
        if not model:
            raise ValueError("A model name must come from the run manifest or caller")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install research-agent-a[llm] to make live model calls") from exc
            client = OpenAI()
        self.model = model
        self.ledger = ledger
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.clock = clock

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
    ) -> ModelResult:
        if role not in ROLES or purpose not in PURPOSES or effort not in EFFORTS:
            raise ValueError("role, purpose, or effort is not in ledger/1")
        if not system_prompt.strip():
            raise ValueError("Role instructions cannot be empty")
        user_input = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        schema_name = f"{role}_{purpose}"
        prompt_hash = hashlib.sha256(
            json.dumps(
                {"instructions": system_prompt, "input": user_input, "schema": schema, "effort": effort},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        started = self.clock()
        response: Any | None = None
        usage = TokenUsage()
        data: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_input,
                reasoning={"effort": effort},
                text={"format": {"type": "json_schema", "name": schema_name, "schema": dict(schema), "strict": True}},
                max_output_tokens=self.max_output_tokens,
            )
            usage = token_usage_from_response(response)
            status = _field(response, "status", "completed")
            if status != "completed":
                reason = _field(_field(response, "incomplete_details", {}), "reason", status)
                raise ModelCallError(f"Model response {status}: {reason}")
            if _has_refusal(response):
                raise ModelCallError("Model refused the structured response")
            output_text = _field(response, "output_text", "")
            if not output_text:
                raise ModelCallError("Model returned no JSON text")
            parsed = json.loads(output_text)
            if not isinstance(parsed, dict):
                raise ModelCallError("Model JSON result must be an object")
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
                seconds=seconds,
                prompt_hash=prompt_hash,
                ok=error is None,
                error=str(error) if error is not None else None,
            )
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
