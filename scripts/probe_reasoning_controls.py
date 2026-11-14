"""Probe whether the gateway honours the reasoning controls the agent can set.

One tiny JSON task is sent under five settings, N times each: reasoning effort
low / medium / high, and thinking switched off in the two ways Qwen-family
OpenAI-compatible gateways accept (``extra_body={"enable_thinking": false}`` and
``extra_body={"chat_template_kwargs": {"enable_thinking": false}}``). For every
setting the table shows how many calls succeeded and the average token buckets, so
one glance tells whether effort changes anything and which off-switch works.

Requests go through research_agent.model.ModelClient, the same code the agent
uses, so a switch that works here works there.

The API key and the gateway address are read only from the environment variables
OPENAI_API_KEY and OPENAI_BASE_URL. Neither is printed, logged or written to a
file, and error text is scrubbed of both before it is shown. --dry-run uses a
built-in fake client: no network, no environment needed. The default is
5 settings x 3 calls = 15 calls; a larger plan needs --max-total-calls.

    python -m scripts.probe_reasoning_controls --dry-run
    OPENAI_API_KEY=... OPENAI_BASE_URL=... python -m scripts.probe_reasoning_controls
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, TextIO
from urllib.parse import urlsplit

from research_agent.model import ModelCallError, ModelClient

MODEL = "qwen3.5-27b"
DEFAULT_CALLS_PER_SETTING = 3
DEFAULT_MAX_TOTAL_CALLS = 15
DEFAULT_MAX_OUTPUT_TOKENS = 2048
ENV_KEY = "OPENAI_API_KEY"
ENV_URL = "OPENAI_BASE_URL"

PROMPT = ("You are a probe for a gateway test. Return one JSON object whose integer "
          "field answer is the sum of the integers a and b.")
PAYLOAD = {"a": 17, "b": 25}
EXPECTED_ANSWER = 42
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Setting:
    name: str
    effort: str
    thinking: bool | None  # False switches thinking off; None leaves the request as it is
    style: str | None = None


SETTINGS = (
    Setting("effort=low", "low", None),
    Setting("effort=medium", "medium", None),
    Setting("effort=high", "high", None),
    Setting("off: enable_thinking", "low", False, "enable_thinking"),
    Setting("off: chat_template_kwargs", "low", False, "chat_template_kwargs"),
)


class _UsageLedger:
    """Stands in for the run ledger: keeps what ModelClient books for every call."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def llm_call(self, **kwargs: Any) -> float:
        self.records.append(kwargs)
        return 0.0


class DryRunClient:
    """A fake Responses client for --dry-run and tests; its numbers are not measurements."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        thinks = "extra_body" not in request
        reasoning = 1500 + 7 * (len(self.requests) % 3) if thinks else 0
        return SimpleNamespace(
            id=f"dry-{len(self.requests)}", status="completed", output=[],
            output_text=json.dumps({"answer": EXPECTED_ANSWER}),
            usage=SimpleNamespace(
                input_tokens=190, input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens=reasoning + 12,
                output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning)))


def _scrub(text: str, secrets: Sequence[str]) -> str:
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, "<hidden>")
    return text


def _secrets(environ: Mapping[str, str]) -> list[str]:
    """Everything that must never be shown: the key, the address, its host and any userinfo."""
    url = environ.get(ENV_URL, "")
    try:
        parts = urlsplit(url)
        pieces = [parts.netloc, parts.hostname or "", parts.username or "", parts.password or ""]
    except ValueError:  # a malformed address still must not be echoed
        pieces = []
    return [item for item in [environ.get(ENV_KEY, ""), url, *pieces] if item]


def _client_from_env(environ: Mapping[str, str]) -> Any:
    missing = [name for name in (ENV_KEY, ENV_URL) if not environ.get(name)]
    if missing:  # names only, never values
        raise SystemExit(f"probe_reasoning_controls: set {' and '.join(missing)} in the environment, or use --dry-run")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("probe_reasoning_controls: the openai package is not installed") from exc
    # One gateway attempt per call keeps the counts honest: no hidden SDK retries.
    return OpenAI(api_key=environ[ENV_KEY], base_url=environ[ENV_URL], timeout=180.0, max_retries=0)


def _bucket(usage: Any, name: str) -> int:
    return int(getattr(usage, name, 0) or 0)


def run_probe(
    client: Any, *, model: str = MODEL, calls_per_setting: int = DEFAULT_CALLS_PER_SETTING,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, settings: Sequence[Setting] = SETTINGS,
    secrets: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Run every setting calls_per_setting times; return one summary row per setting."""
    ledger = _UsageLedger()
    model_client = ModelClient(model, ledger, client=client, max_output_tokens=max_output_tokens)
    rows = []
    for setting in settings:
        ok = correct = 0
        usages: list[tuple[int, int, int]] = []
        errors: list[str] = []
        for _ in range(calls_per_setting):
            booked = len(ledger.records)
            try:
                result = model_client.call_json(
                    role="diagnoser", purpose="diagnose", effort=setting.effort,
                    system_prompt=PROMPT, payload=PAYLOAD, schema=SCHEMA,
                    thinking=setting.thinking, thinking_switch_style=setting.style)
                ok += 1
                correct += int(result.data.get("answer") == EXPECTED_ANSWER)
            except ModelCallError as exc:
                errors.append(_scrub(f"{type(exc).__name__}: {exc}", secrets)[:160])
            if len(ledger.records) > booked:  # a failed call still books what the gateway reported
                usage = ledger.records[-1]["usage"]
                seen = (_bucket(usage, "input") + _bucket(usage, "cached_input"),
                        _bucket(usage, "output"), _bucket(usage, "reasoning"))
                # A request that never got an answer books all zeros. That is "no data",
                # not "zero reasoning"; any real reply carries at least its prompt tokens.
                if seen[0] > 0:
                    usages.append(seen)
        count = len(usages) or 1
        rows.append({
            "setting": setting.name, "calls": calls_per_setting, "ok": ok, "correct": correct,
            "with_usage": len(usages),
            "avg_input": sum(item[0] for item in usages) / count if usages else None,
            "avg_output": sum(item[1] for item in usages) / count if usages else None,
            "avg_reasoning": sum(item[2] for item in usages) / count if usages else None,
            "errors": sorted(set(errors)),
        })
    return rows


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def format_report(rows: Sequence[Mapping[str, Any]], *, model: str, dry_run: bool) -> str:
    calls = sum(row["calls"] for row in rows)
    lines = [f"Reasoning-control probe  model={model}  settings={len(rows)}  total calls={calls}"]
    if dry_run:
        lines.append("DRY RUN: fake client, no network. The numbers below are not measurements.")
    lines.append("avg_input counts cached tokens; avg_output excludes reasoning tokens.")
    header = ("setting", "ok", "correct", "avg_input", "avg_output", "avg_reasoning")
    body = [(row["setting"], f"{row['ok']}/{row['calls']}", f"{row['correct']}/{row['calls']}",
             _cell(row["avg_input"]), _cell(row["avg_output"]), _cell(row["avg_reasoning"]))
            for row in rows]
    widths = [max(len(str(line[i])) for line in [header, *body]) for i in range(len(header))]
    for line in [header, *body]:
        lines.append("  ".join(str(item).ljust(widths[i]) for i, item in enumerate(line)).rstrip())
    baseline = next((row for row in rows if row["setting"] == "effort=low"), None)
    switches = [row for row in rows if row["setting"].startswith("off:")]
    if baseline and switches:
        lines.append("")
        for row in switches:
            if row["avg_reasoning"] is None:
                verdict = "no usable reply, see errors"
            elif not baseline["avg_reasoning"]:
                verdict = "cannot compare: effort=low reported no reasoning tokens"
            else:
                share = row["avg_reasoning"] / baseline["avg_reasoning"]
                verdict = f"reasoning is {share:.0%} of effort=low -> " + (
                    "switch works" if share <= 0.2 else "no clear effect")
                if row["ok"] < row["calls"]:
                    verdict += f" ({row['calls'] - row['ok']} of {row['calls']} calls failed)"
            lines.append(f"{row['setting']}: {verdict}")
    problems = [(row["setting"], message) for row in rows for message in row["errors"]]
    if problems:
        lines.append("")
        lines.append("errors:")
        lines.extend(f"  {name}: {message}" for name, message in problems)
    return "\n".join(lines)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: Sequence[str] | None = None, *, client: Any | None = None,
         environ: Mapping[str, str] | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--calls-per-setting", type=_positive, default=DEFAULT_CALLS_PER_SETTING)
    parser.add_argument("--max-total-calls", type=_positive, default=DEFAULT_MAX_TOTAL_CALLS,
                        help="refuse to start when settings x calls exceeds this (default 15)")
    parser.add_argument("--max-output-tokens", type=_positive, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--dry-run", action="store_true",
                        help="use the built-in fake client: no network, no credentials")
    parser.add_argument("--json-out", type=Path,
                        help="also write the summary rows (no credentials) to this file")
    args = parser.parse_args(argv)
    out = stdout or sys.stdout
    environ = os.environ if environ is None else environ
    planned = len(SETTINGS) * args.calls_per_setting
    if planned > args.max_total_calls:
        print(f"probe_reasoning_controls: {len(SETTINGS)} settings x {args.calls_per_setting} calls = {planned} "
              f"exceeds --max-total-calls {args.max_total_calls}; nothing was sent", file=sys.stderr)
        return 2
    if args.dry_run:
        client = DryRunClient()
    elif client is None:
        client = _client_from_env(environ)
    rows = run_probe(client, model=args.model, calls_per_setting=args.calls_per_setting,
                     max_output_tokens=args.max_output_tokens, secrets=_secrets(environ))
    print(format_report(rows, model=args.model, dry_run=args.dry_run), file=out)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(
            {"schema": "reasoning_probe/1", "model": args.model, "dry_run": args.dry_run,
             "calls_per_setting": args.calls_per_setting, "rows": rows},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["ok"] == row["calls"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
