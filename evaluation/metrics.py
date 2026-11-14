"""Offline efficiency metrics rebuilt from a task's ledger.jsonl and tree.jsonl.

Everything here is read-only and works on already finished (or partly
finished) task directories, so it can be applied to old practice evidence as
well as to new A/B runs. Tokens are the exact sum of the four disjoint ledger
token buckets; the ledger's priced ``cost`` is kept separately because it only
equals tokens under the one-unit-per-token internal price tables.

Terminology used by the metrics:

* main loop: every model call except the summarizer, which only runs after the
  loop has ended;
* node completion: the point in ledger order at which a node is finished, see
  ``node_completions``;
* reference: the dev score 1.0 (the trusted reference solution), the same
  threshold at which the scheduler stops with reason ``ceiling``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

BUCKETS = ("input", "cached_input", "output", "reasoning")
ROLES = ("planner", "executor", "diagnoser", "summarizer")
REFERENCE_SCORE = 1.0
COMPLETION_MODES = ("experiment_score", "node_events")
_EPS = 1e-12


def scenario_of(task_id: str) -> str:
    """Return ``tab`` or ``spd`` from the published task id prefix."""
    if task_id.startswith("tab-"):
        return "tab"
    if task_id.startswith("spd-"):
        return "spd"
    raise ValueError(f"Task id must start with tab- or spd-: {task_id!r}")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file; a missing file is an error, blank lines are not."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{number}")
        rows.append(row)
    return rows


def load_events(task_dir: Path) -> list[dict]:
    """Task ledger events in file (= sequence) order."""
    return read_jsonl(Path(task_dir) / "ledger.jsonl")


def load_nodes(task_dir: Path) -> dict[str, dict]:
    """Final snapshot of every node; later tree.jsonl lines replace earlier ones."""
    nodes: dict[str, dict] = {}
    path = Path(task_dir) / "tree.jsonl"
    if path.exists():
        for row in read_jsonl(path):
            nodes[row["node_id"]] = row
    return nodes


def event_tokens(event: Mapping[str, Any]) -> int:
    """Total tokens of one llm_call event (all four buckets)."""
    tokens = event.get("tokens") or {}
    return sum(int(tokens.get(bucket, 0) or 0) for bucket in BUCKETS)


def _dev_score(node: Mapping[str, Any]) -> float | None:
    score = node.get("score")
    if isinstance(score, Mapping):
        value = score.get("dev")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _completion_indices(events: list[dict], nodes: Mapping[str, dict],
                        completion: str) -> dict[str, int]:
    if completion not in COMPLETION_MODES:
        raise ValueError(f"completion must be one of {COMPLETION_MODES}")
    kinds = {"experiment", "score"} if completion == "experiment_score" else None
    last: dict[str, int] = {}
    for index, event in enumerate(events):
        node_id = event.get("node_id")
        if node_id and node_id != "n0000" and node_id in nodes and (
                kinds is None or event.get("kind") in kinds):
            last[node_id] = index
    return last


def node_completions(events: list[dict], nodes: Mapping[str, dict], *,
                     completion: str = "experiment_score") -> list[dict]:
    """Completed non-root nodes in ledger order with cumulative spend at completion.

    ``completion="experiment_score"`` takes the node's last experiment/score
    event as its completion point (nodes that never reached a run are not
    counted). ``"node_events"`` takes the node's last event of any kind, which
    also covers trailing diagnoser calls and nodes whose executor failed before
    any run. Each row carries the tokens spent by the main loop (everything but
    the summarizer) up to and including that point and the best accepted dev
    score so far (the baseline counts from the start).
    """
    last = _completion_indices(events, nodes, completion)
    at_index: dict[int, str] = {index: node_id for node_id, index in last.items()}
    root = nodes.get("n0000")
    best = _dev_score(root) if root is not None else None
    main = 0
    total = 0
    rows: list[dict] = []
    for index, event in enumerate(events):
        if event.get("kind") == "llm_call":
            tokens = event_tokens(event)
            total += tokens
            if event.get("role") != "summarizer":
                main += tokens
        node_id = at_index.get(index)
        if node_id is None:
            continue
        node = nodes[node_id]
        score = _dev_score(node)
        accepted = node.get("status") == "accepted"
        if accepted and score is not None and (best is None or score > best):
            best = score
        rows.append({
            "node_id": node_id, "order": len(rows) + 1, "seq": event.get("seq"),
            "status": node.get("status"), "accepted": accepted, "dev_score": score,
            "main_tokens": main, "total_tokens": total, "best_dev_so_far": best,
        })
    return rows


def _final_best(nodes: Mapping[str, dict]) -> tuple[str | None, float | None]:
    """Best accepted node by dev score, mirroring ExperimentTree.best_node."""
    best_id, best_score = None, None
    for node_id, node in nodes.items():
        score = _dev_score(node)
        if node.get("status") == "accepted" and score is not None and (
                best_score is None or score > best_score):
            best_id, best_score = node_id, score
    return best_id, best_score


def _read_json_if(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def task_metrics(task_dir: Path, *, completion: str = "experiment_score") -> dict:
    """Efficiency and outcome metrics of one task directory.

    ``task_dir`` is ``run_dir/tasks/<task_id>``. Fields that cannot be derived
    (no holdout yet, a field an older ledger does not have) are ``None``.
    """
    task_dir = Path(task_dir)
    events = load_events(task_dir)
    nodes = load_nodes(task_dir)
    task_id = next((event["task_id"] for event in events if event.get("task_id")), task_dir.name)
    llm = [event for event in events if event.get("kind") == "llm_call"]
    by_role = {role: sum(event_tokens(e) for e in llm if e.get("role") == role) for role in ROLES}
    buckets = {bucket: sum(int((e.get("tokens") or {}).get(bucket, 0) or 0) for e in llm)
               for bucket in BUCKETS}
    total_tokens = sum(event_tokens(e) for e in llm)
    main_tokens = sum(event_tokens(e) for e in llm if e.get("role") != "summarizer")
    stops = [event for event in events if event.get("kind") == "stop"]
    endings = [event for event in events if event.get("kind") == "task_end"]
    stop = stops[-1] if stops else {}
    ending = endings[-1] if endings else {}
    completions = node_completions(events, nodes, completion=completion)
    root = nodes.get("n0000")
    baseline_dev = _dev_score(root) if root is not None else None
    tree_best_id, tree_best = _final_best(nodes)
    best_id = stop.get("best_node") if stop.get("best_node") in nodes else tree_best_id
    best_dev = _dev_score(nodes[best_id]) if best_id in nodes else tree_best

    tokens_to_reference = None
    if baseline_dev is not None and baseline_dev >= REFERENCE_SCORE - _EPS:
        tokens_to_reference = 0
    else:
        for row in completions:
            if row["best_dev_so_far"] is not None and row["best_dev_so_far"] >= REFERENCE_SCORE - _EPS:
                tokens_to_reference = row["main_tokens"]
                break
    if best_id is None or best_id == "n0000":
        tokens_to_best = 0
    else:
        done = next((row for row in completions if row["node_id"] == best_id), None)
        tokens_to_best = done["main_tokens"] if done is not None else None
    holdout = _read_json_if(task_dir / "holdout_score.json") or {}
    failed = ending.get("holdout_failed")
    if failed is None:
        failed = (task_dir / "holdout_failure.json").exists()
    non_root = [node for node_id, node in nodes.items() if node_id != "n0000"]
    return {
        "schema": "task_metrics/1", "task_id": task_id, "scenario": scenario_of(task_id),
        "run_id": next((event["run_id"] for event in events if event.get("run_id")), None),
        "stop_reason": stop.get("reason"), "best_node": best_id,
        "baseline_dev_score": baseline_dev, "best_dev_score": best_dev,
        "completion_mode": completion,
        "experiments": len(non_root), "completed_nodes": len(completions),
        "accepted_nodes": sum(1 for node in non_root if node.get("status") == "accepted"),
        "model_calls": len(llm),
        "failed_calls": sum(1 for event in llm if event.get("ok") is False),
        "total_tokens": total_tokens, "main_loop_tokens": main_tokens,
        "tokens_by_role": by_role, "token_buckets": buckets,
        "total_cost": sum(float(e.get("cost") or 0.0) for e in llm),
        "tokens_to_reference": tokens_to_reference,
        "reached_reference": tokens_to_reference is not None,
        "tokens_to_best": tokens_to_best,
        "tokens_after_best": (main_tokens - tokens_to_best) if tokens_to_best is not None else None,
        "has_task_end": bool(endings),
        "holdout_failed": bool(failed),
        "holdout_score": _number(holdout.get("score", ending.get("holdout_score"))),
        "holdout_raw": _number(holdout.get("raw")),
        "holdout_raw_lift": _number(ending.get("holdout_raw_lift")),
        "baseline_holdout_score": _number(ending.get("baseline_holdout_score")),
        "baseline_holdout_raw": _number(ending.get("baseline_holdout_raw")),
    }


def run_metrics(run_dir: Path, *, completion: str = "experiment_score") -> dict:
    """Metrics for every task directory of one run (manifest order when present)."""
    run_dir = Path(run_dir)
    manifest = _read_json_if(run_dir / "manifest.json") or {}
    tasks_root = run_dir / "tasks"
    names = manifest.get("tasks")
    if not isinstance(names, list):
        names = sorted(path.name for path in tasks_root.iterdir() if path.is_dir()) \
            if tasks_root.is_dir() else []
    tasks = [task_metrics(tasks_root / name, completion=completion)
             for name in names if (tasks_root / name / "ledger.jsonl").is_file()]
    return {"schema": "run_metrics/1", "run_id": manifest.get("run_id", run_dir.name),
            "phase": manifest.get("phase"), "config": manifest.get("config"),
            "rep": manifest.get("rep"), "tasks": tasks}


def find_run_dirs(roots: Iterable[Path]) -> list[Path]:
    """Run directories (those holding a run/1 manifest) below the given paths."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        for current, dirnames, filenames in os.walk(root):
            dirnames.sort()
            path = Path(current)
            if "manifest.json" in filenames:
                manifest = _read_json_if(path / "manifest.json")
                if manifest and manifest.get("schema") == "run/1":
                    dirnames[:] = []  # never descend into a run's own files
                    resolved = path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(path)
    return found


__all__ = ["BUCKETS", "COMPLETION_MODES", "REFERENCE_SCORE", "ROLES", "event_tokens",
           "find_run_dirs", "load_events", "load_nodes", "node_completions",
           "read_jsonl", "run_metrics", "scenario_of", "task_metrics"]
