"""C-only holdout, task_end and run_end finalization.

The one-shot holdout is scored on a private copy of the best node's solution.py,
never on the node directory itself: the scorer sandbox may write into whatever
directory it is given, and the node directory is sealed evidence. Any holdout
failure (exception, missing or altered file, invalid B result) is recorded in
holdout_score.json and task_end instead of crashing the run; the one-shot
attempt marker still forbids a retry.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from threading import RLock
from typing import Any, Iterator, NamedTuple

from ledger import Ledger


_EVAL_LOCK = RLock()
HOLDOUT_COPY_DIR = "holdout_candidate"
BASELINE_NODE = "n0000"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FLAG = re.compile(r"[a-z_]{1,40}")


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HoldoutIntegrityError(RuntimeError):
    """The sealed solution could not be scored as sealed; the message is fixed text."""


_SAFE_SCORER_MESSAGES = frozenset({
    "Scoring timed out", "Candidate exceeded wall time",
    "Scoring exceeded memory", "Candidate exceeded memory limit",
    "Sandbox worker returned no JSON", "Scoring worker failed",
})
# Exception texts raised by B's trusted parent-side code; none embed hidden data.
_SAFE_HARNESS_MESSAGES = frozenset({
    "Node must contain solution.py", "Node directory must exist outside task packages",
    "RA_EVAL=1 is required for holdout", "Split must be dev or holdout",
    "macOS sandbox-exec is required; refusing unisolated candidate execution",
    "Missing or invalid hidden labels", "Forbidden feature present in hidden split",
    "AUC requires both classes", "AUC inputs differ in size or are empty",
    "Limits must be positive",
})
_FALLBACK_SCORER_MESSAGES = {
    "timeout": "Scoring timed out",
    "oom": "Scoring exceeded memory",
    "exception": "Scoring worker failed",
    "missing_score": "Holdout scorer returned no score",
    "unknown": "Holdout scoring failed",
}


def _safe_seconds(seconds: Any) -> dict:
    safe = {}
    if isinstance(seconds, dict):
        for key in ("run", "queue"):
            value = seconds.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value)
                if math.isfinite(value) and value >= 0:
                    safe[key] = value
    return safe


def _failure_record(best_node: str, reason: str, error: dict, *,
                    seconds: Any = None, seed: Any = None) -> dict:
    metadata = {"schema": "holdout_failure/1", "ts": _utc_now(), "best_node": best_node,
                "reason": reason, "error": error}
    safe_seconds = _safe_seconds(seconds)
    if safe_seconds:
        metadata["seconds"] = safe_seconds
    if isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**64:
        metadata["seed"] = seed
    return metadata


def _holdout_failure_metadata(result: dict, best_node: str) -> dict:
    """Allowlist scorer diagnostics without retaining hidden labels or predictions."""
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    kind = error.get("kind")
    if not isinstance(kind, str) or kind not in _FALLBACK_SCORER_MESSAGES:
        kind = "missing_score" if result.get("ok") else "unknown"
    message = error.get("message")
    if not isinstance(message, str) or message not in _SAFE_SCORER_MESSAGES:
        message = _FALLBACK_SCORER_MESSAGES[kind]
    return _failure_record(best_node, "not_ok" if not result.get("ok") else "missing_score",
                           {"kind": kind, "message": message},
                           seconds=result.get("seconds"), seed=result.get("seed"))


def _exception_error(exc: BaseException) -> dict:
    """Record what failed without copying text that could carry hidden data."""
    if isinstance(exc, HoldoutIntegrityError):
        return {"kind": "integrity", "message": str(exc)}
    message = "Holdout scoring raised an exception"
    if str(exc) in _SAFE_HARNESS_MESSAGES:
        message = str(exc)
    elif isinstance(exc, OSError):
        text = " ".join(str(part) for part in (exc.strerror, exc.filename) if part)
        if text:
            message = text[:300]  # file-system errors name paths only
    return {"kind": "harness_exception", "type": type(exc).__name__, "message": message}


def _failure_score(metadata: dict, cheat_flags: Any = ()) -> dict:
    """A score/1-shaped failure record: score is null, never a number."""
    seconds = metadata.get("seconds", {})
    flags = ([flag for flag in cheat_flags if isinstance(flag, str) and _FLAG.fullmatch(flag)][:10]
             if isinstance(cheat_flags, (list, tuple)) else [])
    return {"schema": "score/1", "ok": False, "split": "holdout", "score": None,
            "raw": None, "spread": 0.0, "correct": False, "cheat_flags": flags,
            "error": metadata["error"],
            "seconds": {"run": seconds.get("run", 0.0), "queue": seconds.get("queue", 0.0)},
            "seed": metadata.get("seed")}


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


class _Holdout(NamedTuple):
    result: dict            # score/1 shaped; score is None when failed
    failure: dict | None    # holdout_failure/1 metadata, None on success
    integrity: dict         # holdout_integrity/1 evidence about what was scored


def _sealed_hash(task_dir: Path, node_id: str) -> str | None:
    """SHA-256 sealed in replay_context.json; None only when no seal exists yet."""
    path = task_dir / "replay_context.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["nodes"][node_id]["solution_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        raise HoldoutIntegrityError("Replay seal does not record this node's solution") from None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise HoldoutIntegrityError("Replay seal has no solution hash for this node")
    return value


def _run_holdout(harness: Any, task_id: str, scoring_dir: Path) -> Any:
    scored_holdout = getattr(harness, "score_holdout", None)
    if callable(scored_holdout):
        return scored_holdout(task_id, scoring_dir)
    with _eval_window():
        return harness.score(task_id, scoring_dir, "holdout")


def _accept(result: Any, node_id: str) -> tuple[dict, dict | None]:
    """Turn B's return value into (record to save, failure metadata or None)."""
    if (not isinstance(result, dict) or result.get("schema") != "score/1"
            or result.get("split") != "holdout"):
        metadata = _failure_record(node_id, "invalid_result", {
            "kind": "invalid_result", "message": "B returned an invalid holdout score"})
        return _failure_score(metadata), metadata
    score = result.get("score")
    unusable = (not result.get("ok") or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(float(score)))
    if unusable:
        metadata = _holdout_failure_metadata(result, node_id)
        return _failure_score(metadata, result.get("cheat_flags", ())), metadata
    return result, None


def _score_node(harness: Any, task_id: str, task_dir: Path, node_id: str) -> _Holdout:
    """Score one node's solution on a private copy; Exceptions become failure records."""
    source = task_dir / "nodes" / node_id / "solution.py"
    integrity = {"schema": "holdout_integrity/1", "node": node_id, "solution_sha256": None,
                 "sealed_sha256": None, "copy_path": None, "status": "not_started"}
    try:
        sealed = _sealed_hash(task_dir, node_id)
        integrity["sealed_sha256"] = sealed
        if not source.is_file():
            raise HoldoutIntegrityError("Node solution.py is missing before holdout scoring")
        before = _sha256(source)
        integrity["solution_sha256"] = before
        if sealed is not None and before != sealed:
            raise HoldoutIntegrityError("Node solution.py differs from the sealed record")
        copy_dir = task_dir / HOLDOUT_COPY_DIR / node_id
        copy_dir.mkdir(parents=True, exist_ok=False)
        integrity["copy_path"] = str(copy_dir)
        shutil.copyfile(source, copy_dir / "solution.py")
        if _sha256(copy_dir / "solution.py") != before:
            raise HoldoutIntegrityError("Private holdout copy differs from the original")
        integrity["status"] = "scoring"
        returned = _run_holdout(harness, task_id, copy_dir)
        if not source.is_file() or _sha256(source) != before:
            raise HoldoutIntegrityError("Node solution.py changed or vanished during holdout scoring")
        if not (copy_dir / "solution.py").is_file() or _sha256(copy_dir / "solution.py") != before:
            raise HoldoutIntegrityError("Private holdout copy changed during holdout scoring")
        result, failure = _accept(returned, node_id)
        integrity["status"] = "ok" if failure is None else "scored_failed"
        return _Holdout(result, failure, integrity)
    except Exception as exc:  # noqa: BLE001 - recorded, never allowed to crash the run
        error = _exception_error(exc)
        metadata = _failure_record(
            node_id, "integrity" if error["kind"] == "integrity" else "exception", error)
        integrity["status"] = "error: " + error["kind"]
        return _Holdout(_failure_score(metadata), metadata, integrity)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def finish_task(run_dir: Path, task_id: str, harness: Any) -> dict:
    """Evaluate holdout exactly once, then record task_end with quote errors.

    A failed holdout still ends the task: holdout_score.json records the failure
    (score null), task_end carries holdout_failed=true, and the attempt marker
    keeps the holdout from being retried. The baseline solution is scored once as
    a reference (baseline_holdout.json); that is not a second attempt.
    """
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
    quotes = [event for event in events if event["kind"] == "quote"]
    if len(quotes) > 1:
        raise ValueError("Multiple quotes found in task ledger")
    task_dir = run_dir / "tasks" / task_id
    attempt = task_dir / "holdout_attempt.json"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    with attempt.open("x", encoding="utf-8") as handle:
        json.dump({"schema": "holdout_attempt/1", "ts": _utc_now(), "best_node": best_node}, handle)
        handle.write("\n")
    best = _score_node(harness, task_id, task_dir, best_node)
    result = best.result
    failed = best.failure is not None
    _write_json(task_dir / "holdout_score.json", result)
    _write_json(task_dir / "holdout_integrity.json", best.integrity)
    if failed:
        _write_json(task_dir / "holdout_failure.json", best.failure)
    elapsed = ledger.elapsed()
    holdout_seconds = _safe_seconds(result.get("seconds"))
    seconds = {
        "llm": elapsed["llm"],
        "run": elapsed["run"] + holdout_seconds.get("run", 0.0),
        "queue": elapsed["queue"] + holdout_seconds.get("queue", 0.0),
        "wall": elapsed["wall"],
    }
    total_cost = ledger.spent()
    if quotes:
        forecast = quotes[0]["predicted"]
        quote_error = {
            "cost": total_cost - float(forecast["cost"]),
            "seconds": seconds["wall"] - float(forecast["seconds"]),
            "score": None if failed else float(result["score"]) - float(forecast["score"]),
        }
    else:
        quote_error = {"cost": None, "seconds": None, "score": None}
    # Baseline reference on the same hidden data. When the agent kept the baseline,
    # its result is the best node's result (lift is exactly zero, no extra scoring).
    reused = best_node == BASELINE_NODE
    baseline = best if reused else _score_node(harness, task_id, task_dir, BASELINE_NODE)
    baseline_failed = baseline.failure is not None
    _write_json(task_dir / "baseline_holdout.json", {
        "schema": "baseline_holdout/1", "ts": _utc_now(), "node": BASELINE_NODE,
        "reused_best_result": reused, "failed": baseline_failed, "result": baseline.result,
        "failure": baseline.failure, "integrity": baseline.integrity})
    best_raw = None if failed else _finite_number(result.get("raw"))
    baseline_raw = None if baseline_failed else _finite_number(baseline.result.get("raw"))
    lift = best_raw - baseline_raw if best_raw is not None and baseline_raw is not None else None
    ledger.emit("task_end", holdout_score=None if failed else result["score"], best_node=best_node,
                holdout_failed=failed, holdout_raw=best_raw,
                baseline_holdout_score=None if baseline_failed else baseline.result["score"],
                baseline_holdout_raw=baseline_raw, baseline_holdout_failed=baseline_failed,
                holdout_raw_lift=lift,
                total_cost=total_cost, seconds=seconds, quote_error=quote_error)
    return {"score": result, "baseline": baseline.result, "task_end": ledger._events()[-1]}


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


__all__ = ["HoldoutIntegrityError", "finish_run", "finish_task", "start_run"]
