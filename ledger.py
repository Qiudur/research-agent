"""C-owned append-only ledger and exclusive-token price accounting (ledger/1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


KINDS = frozenset({
    "task_start", "quote", "llm_call", "experiment", "score",
    "review", "library", "stop", "task_end", "run_start", "run_end",
})
TOKEN_BUCKETS = ("input", "cached_input", "output", "reasoning")
_TASK_ID = re.compile(r"^(?:tab|spd|quant)-[a-z0-9]+(?:-[a-z0-9]+)*$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return number


def _sealed_quant_input(root: Path, manifest: dict, name: str, schema: str) -> None:
    """Check a quant manifest's project-local, immutable JSON input."""
    descriptor = manifest.get(name)
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ValueError(f"Quant run must identify {name}")
    relative = Path(descriptor["path"])
    expected = str(descriptor.get("sha256", "")).lower()
    if relative.is_absolute() or not _HASH.fullmatch(expected):
        raise ValueError(f"Quant run {name} needs a relative path and SHA-256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Quant run {name} escapes project root")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Quant run {name} changed after sealing")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"Quant run {name} must have schema {schema}")


def _validate_quant_manifest(manifest: dict, root: Path) -> None:
    tasks = manifest.get("tasks")
    if (not isinstance(tasks, list) or not tasks
            or any(not isinstance(item, str) or not item.startswith("quant-")
                   or not _TASK_ID.fullmatch(item) for item in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError("Quant run needs unique quant-* tasks")
    if manifest.get("phase") != "dev":
        raise ValueError("Quant run phase must be dev")
    if not isinstance(manifest.get("model"), str) or not manifest["model"].strip():
        raise ValueError("Quant run needs a model")
    commit = manifest.get("code_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError("Quant run needs a code commit SHA")
    library = manifest.get("library")
    if (not isinstance(library, dict) or not isinstance(library.get("root"), str)
            or not library["root"] or not isinstance(library.get("start_version"), str)
            or not re.fullmatch(r"lib-(0|[1-9][0-9]*)", library["start_version"])
            or not isinstance(library.get("frozen"), bool)):
        raise ValueError("Quant run needs sealed library settings")
    library_root = (root / library["root"]).resolve()
    if Path(library["root"]).is_absolute() or not library_root.is_relative_to(root):
        raise ValueError("Quant run library root escapes project")
    hard_caps = manifest.get("hard_caps")
    if (not isinstance(hard_caps, dict) or isinstance(hard_caps.get("budget_ratio"), bool)
            or not isinstance(hard_caps.get("budget_ratio"), (int, float))
            or not 0 < hard_caps["budget_ratio"] < 1):
        raise ValueError("Quant run needs a valid hard cap ratio")
    _sealed_quant_input(root, manifest, "dataset", "quant_dataset/1")
    _sealed_quant_input(root, manifest, "agent_config", "agent/1")


@dataclass(frozen=True)
class Usage:
    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning: int = 0

    def __post_init__(self) -> None:
        for key in TOKEN_BUCKETS:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Usage.{key} must be a nonnegative integer")

    def __eq__(self, other: object) -> bool:
        """Remain compatible with A's equivalent TokenUsage value object."""
        if not all(hasattr(other, key) for key in TOKEN_BUCKETS):
            return False
        return all(getattr(self, key) == getattr(other, key) for key in TOKEN_BUCKETS)


def price_usage(prices: dict, model: str, usage: Usage) -> float:
    """Price four disjoint token buckets without writing a ledger event."""
    unit = prices.get("unit_tokens")
    if isinstance(unit, bool) or not isinstance(unit, int) or unit <= 0:
        raise ValueError("unit_tokens must be a positive integer")
    tokens = {key: getattr(usage, key) for key in TOKEN_BUCKETS}
    for key, value in tokens.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid {key} token count")
    models = prices.get("models")
    rates = models.get(model) if isinstance(models, dict) else None
    if not isinstance(rates, dict):
        raise ValueError(f"Price table has no rates for model {model!r}")
    priced: dict[str, float] = {}
    for key in TOKEN_BUCKETS:
        if rates.get(key) is None:
            raise ValueError(f"Missing {key} price for model {model!r}")
        priced[key] = _nonnegative(rates[key], f"{model}.{key} price")
    return sum(tokens[key] * priced[key] for key in TOKEN_BUCKETS) / unit


class Ledger:
    """One task or run-level ledger. Sequence numbers are local to its file."""

    def __init__(self, run_dir: Path, task_id: str | None, manifest: dict, prices: dict):
        self.run_dir = run_dir
        self.run_id = run_dir.name
        self.task_id = task_id
        self.manifest = manifest
        self.prices = prices
        self.path = (run_dir / "ledger.jsonl" if task_id is None
                     else run_dir / "tasks" / task_id / "ledger.jsonl")

    @classmethod
    def open(cls, run_dir: Path, task_id: str | None = None) -> "Ledger":
        run_dir = Path(run_dir).expanduser().resolve()
        if task_id is not None and not _TASK_ID.fullmatch(task_id):
            raise ValueError("Invalid task_id")
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = manifest.get("schema")
        if schema not in {"run/1", "quant_run/1"} or manifest.get("run_id") != run_dir.name:
            raise ValueError("Run manifest schema or run_id mismatch")
        project_root = run_dir.parent.parent
        if schema == "quant_run/1":
            _validate_quant_manifest(manifest, project_root)
        elif any(isinstance(item, str) and item.startswith("quant-")
                 for item in manifest.get("tasks", [])):
            raise ValueError("Quant tasks require a quant_run/1 manifest")
        if task_id is not None and task_id not in manifest.get("tasks", []):
            raise ValueError("Task is absent from run manifest")
        descriptor = manifest.get("prices")
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
            raise ValueError("Run manifest must identify a price file")
        expected_hash = str(descriptor.get("sha256", "")).lower()
        if not _HASH.fullmatch(expected_hash):
            raise ValueError("Run manifest has no sealed price hash")
        price_path = Path(descriptor["path"]).expanduser()
        if schema == "quant_run/1" and price_path.is_absolute():
            raise ValueError("Quant run prices need a project-relative path")
        price_path = (price_path if price_path.is_absolute() else project_root / price_path).resolve()
        if not price_path.is_relative_to(project_root):
            raise ValueError("Price file escapes project root")
        raw = price_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError("Price file changed after manifest was sealed")
        prices = json.loads(raw)
        if prices.get("schema") != "prices/1":
            raise ValueError("Unsupported price schema")
        if not isinstance(prices.get("unit_tokens"), int) or prices["unit_tokens"] <= 0:
            raise ValueError("unit_tokens must be a positive integer")
        if not isinstance(prices.get("models"), dict):
            raise ValueError("Price table must contain models")
        return cls(run_dir, task_id, manifest, prices)

    def _events(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                lines = handle.readlines()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        events = [json.loads(line) for line in lines if line.strip()]
        for seq, event in enumerate(events, 1):
            if (event.get("schema") != "ledger/1" or event.get("seq") != seq
                    or event.get("run_id") != self.run_id or event.get("task_id") != self.task_id):
                raise ValueError(f"Corrupt ledger sequence or identity in {self.path}")
        return events

    def emit(self, kind: str, *, node_id: str | None = None, **fields: Any) -> None:
        if kind not in KINDS:
            raise ValueError(f"Unknown ledger/1 event kind: {kind}")
        if self.task_id is None and kind not in {"run_start", "run_end"}:
            raise ValueError("Run-level ledger accepts only run_start/run_end")
        if self.task_id is not None and kind in {"run_start", "run_end"}:
            raise ValueError("Run lifecycle events belong in the run-level ledger")
        if "schema" in fields or "seq" in fields or "ts" in fields or "run_id" in fields or "task_id" in fields or "kind" in fields:
            raise ValueError("Ledger common fields are assigned by emit()")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                previous = [json.loads(line) for line in handle if line.strip()]
                if previous and (previous[-1].get("schema") != "ledger/1"
                                 or previous[-1].get("run_id") != self.run_id
                                 or previous[-1].get("task_id") != self.task_id
                                 or previous[-1].get("seq") != len(previous)):
                    raise ValueError("Cannot append to a corrupt ledger")
                event = {"schema": "ledger/1", "seq": len(previous) + 1,
                         "ts": _utc_now(), "run_id": self.run_id,
                         "task_id": self.task_id, "node_id": node_id,
                         "kind": kind, **fields}
                handle.seek(0, 2)
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                import os
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def llm_call(
        self, *, node_id: str | None, role: str, purpose: str, effort: str, model: str,
        usage: Usage, seconds: float, prompt_hash: str, ok: bool = True,
        error: str | None = None, usage_status: str = "reported",
    ) -> float:
        if model != self.manifest.get("model"):
            raise ValueError("Model differs from sealed run manifest")
        if role not in {"planner", "executor", "diagnoser", "summarizer"}:
            raise ValueError("Invalid model role")
        if purpose not in {"plan", "quote", "code", "fix", "diagnose", "review", "summarize"}:
            raise ValueError("Invalid model purpose")
        if effort not in {"low", "medium", "high"}:
            raise ValueError("Invalid model effort")
        if not _HASH.fullmatch(prompt_hash.lower()):
            raise ValueError("prompt_hash must be SHA256")
        if usage_status not in {"reported", "unknown"}:
            raise ValueError("usage_status must be reported or unknown")
        duration = _nonnegative(seconds, "seconds")
        tokens = {key: getattr(usage, key) for key in TOKEN_BUCKETS}
        cost = price_usage(self.prices, model, usage)
        self.emit(
            "llm_call", node_id=node_id, role=role, purpose=purpose, effort=effort,
            model=model, tokens=tokens, cost=cost, seconds=duration,
            prompt_hash=prompt_hash.lower(), ok=bool(ok), error=error,
            usage_status=usage_status,
        )
        return cost

    def spent(self) -> float:
        return sum(float(event["cost"]) for event in self._events() if event["kind"] == "llm_call")

    def elapsed(self) -> dict[str, float]:
        events = self._events()
        llm = sum(float(event["seconds"]) for event in events if event["kind"] == "llm_call")
        run = sum(float(event["seconds_run"]) for event in events if event["kind"] == "experiment")
        queue = sum(float(event["seconds_queue"]) for event in events if event["kind"] == "score")
        first = next((event["ts"] for event in events if event["kind"] in {"task_start", "run_start"}), None)
        started = first or self.manifest.get("started_ts")
        wall = 0.0
        if isinstance(started, str):
            wall = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds())
        return {"llm": llm, "run": run, "queue": queue, "wall": wall}


__all__ = ["Ledger", "Usage", "price_usage"]
