"""C-only holdout, task_end and run_end finalization."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterator

from ledger import Ledger


_EVAL_LOCK = RLock()


def assert_not_in_eval_window() -> None:
    with _EVAL_LOCK:
        if os.environ.get("RA_EVAL") == "1":
            raise ValueError("RA_EVAL must be absent during A's main loop")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


_SAFE_SCORER_MESSAGES = frozenset({
    "Scoring timed out", "Candidate exceeded wall time",
    "Scoring exceeded memory", "Candidate exceeded memory limit",
    "Sandbox worker returned no JSON", "Scoring worker failed",
})
_FALLBACK_SCORER_MESSAGES = {
    "timeout": "Scoring timed out",
    "oom": "Scoring exceeded memory",
    "exception": "Scoring worker failed",
    "missing_score": "Holdout scorer returned no score",
    "unknown": "Holdout scoring failed",
}


def _holdout_failure_metadata(result: dict, best_node: str) -> dict:
    """Allowlist scorer diagnostics without retaining hidden labels or predictions."""
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    kind = error.get("kind")
    if not isinstance(kind, str) or kind not in _FALLBACK_SCORER_MESSAGES:
        kind = "missing_score" if result.get("ok") else "unknown"
    message = error.get("message")
    if not isinstance(message, str) or message not in _SAFE_SCORER_MESSAGES:
        message = _FALLBACK_SCORER_MESSAGES[kind]
    metadata = {
        "schema": "holdout_failure/1",
        "ts": _utc_now(),
        "best_node": best_node,
        "reason": "not_ok" if not result.get("ok") else "missing_score",
        "error": {"kind": kind, "message": message},
    }
    seconds = result.get("seconds")
    if isinstance(seconds, dict):
        safe_seconds = {}
        for key in ("run", "queue"):
            value = seconds.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)
                if math.isfinite(value) and value >= 0:
                    safe_seconds[key] = value
        if safe_seconds:
            metadata["seconds"] = safe_seconds
    seed = result.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**64:
        metadata["seed"] = seed
    return metadata


@contextmanager
def _eval_window() -> Iterator[None]:
    with _EVAL_LOCK:
        previous = os.environ.get("RA_EVAL")
        os.environ["RA_EVAL"] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("RA_EVAL", None)
            else:
                os.environ["RA_EVAL"] = previous


def start_run(run_dir: Path) -> None:
    """Start one run-level ledger after create_manifest()."""
    ledger = Ledger.open(run_dir, None)
    if ledger._events():
        raise ValueError("Run-level ledger already started")
    ledger.emit("run_start")


def finish_task(run_dir: Path, task_id: str, harness: Any) -> dict:
    """Evaluate holdout exactly once, then record task_end with quote errors."""
    run_dir = Path(run_dir).expanduser().resolve()
    ledger = Ledger.open(run_dir, task_id)
    events = ledger._events()
    if any(event["kind"] == "task_end" for event in events):
        raise ValueError("task_end is already recorded")
    stops = [event for event in events if event["kind"] == "stop"]
    if len(stops) != 1:
        raise ValueError("Exactly one A stop event is required before holdout")
    best_node = stops[0].get("best_node")
    if not isinstance(best_node, str) or not re.fullmatch(r"n[0-9]{4,}", best_node):
        raise ValueError("Stop event has no valid best node")
    node_dir = run_dir / "tasks" / task_id / "nodes" / best_node
    if not (node_dir / "solution.py").is_file():
        raise FileNotFoundError(node_dir / "solution.py")
    task_dir = run_dir / "tasks" / task_id
    attempt = task_dir / "holdout_attempt.json"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    with attempt.open("x", encoding="utf-8") as handle:
        json.dump({"schema": "holdout_attempt/1", "ts": _utc_now(), "best_node": best_node}, handle)
        handle.write("\n")
    scored_holdout = getattr(harness, "score_holdout", None)
    if callable(scored_holdout):
        result = scored_holdout(task_id, node_dir)
    else:
        with _eval_window():
            result = harness.score(task_id, node_dir, "holdout")
    if not isinstance(result, dict) or result.get("schema") != "score/1" or result.get("split") != "holdout":
        raise ValueError("B returned an invalid holdout score")
    if not result.get("ok") or result.get("score") is None:
        _write_json(task_dir / "holdout_failure.json", _holdout_failure_metadata(result, best_node))
        raise RuntimeError("Holdout failed; attempt is recorded and must not be retried")
    _write_json(task_dir / "holdout_score.json", result)
    elapsed = ledger.elapsed()
    holdout_seconds = result.get("seconds", {})
    seconds = {
        "llm": elapsed["llm"],
        "run": elapsed["run"] + float(holdout_seconds.get("run", 0)),
        "queue": elapsed["queue"] + float(holdout_seconds.get("queue", 0)),
        "wall": elapsed["wall"],
    }
    total_cost = ledger.spent()
    quotes = [event for event in events if event["kind"] == "quote"]
    if len(quotes) > 1:
        raise ValueError("Multiple quotes found in task ledger")
    if quotes:
        forecast = quotes[0]["predicted"]
        quote_error = {
            "cost": total_cost - float(forecast["cost"]),
            "seconds": seconds["wall"] - float(forecast["seconds"]),
            "score": float(result["score"]) - float(forecast["score"]),
        }
    else:
        quote_error = {"cost": None, "seconds": None, "score": None}
    ledger.emit("task_end", holdout_score=result["score"], best_node=best_node,
                total_cost=total_cost, seconds=seconds, quote_error=quote_error)
    return {"score": result, "task_end": ledger._events()[-1]}


def finish_run(run_dir: Path) -> dict:
    """Close a run after every listed task has one task_end event."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_ledger = Ledger.open(run_dir, None)
    events = run_ledger._events()
    if len([event for event in events if event["kind"] == "run_start"]) != 1:
        raise ValueError("Run must have one run_start event")
    if any(event["kind"] == "run_end" for event in events):
        raise ValueError("Run is already ended")
    task_costs = []
    for task_id in run_ledger.manifest["tasks"]:
        task_events = Ledger.open(run_dir, task_id)._events()
        endings = [event for event in task_events if event["kind"] == "task_end"]
        if len(endings) != 1:
            raise ValueError(f"Task {task_id} lacks exactly one task_end event")
        task_costs.append(float(endings[0]["total_cost"]))
    total_cost = sum(task_costs)
    run_ledger.emit("run_end", total_cost=total_cost)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ended_ts"] = _utc_now()
    _write_json(manifest_path, manifest)
    return {"run_id": run_dir.name, "total_cost": total_cost, "ended_ts": manifest["ended_ts"]}


__all__ = ["finish_run", "finish_task", "start_run"]
