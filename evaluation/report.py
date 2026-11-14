"""Evidence-linked summaries from A's tree and C's append-only ledgers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ledger import Ledger
from research_agent.replay import _verify_replay_context
from token_accounting import is_exact_token_accounting


def _jsonl(path: Path, schema: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) or row.get("schema") != schema for row in rows):
        raise ValueError(f"Unsupported schema in {path}")
    return rows


def _latest_nodes(tree_path: Path) -> dict[str, dict]:
    nodes = {}
    for row in _jsonl(tree_path, "node/1"):
        nodes[row["node_id"]] = row
    return nodes


def extract_summary(run_dir: Path, task_id: str, *, max_nodes: int = 40) -> dict:
    """Compact public evidence for a summarizer; never open B's hidden data."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    run_dir = Path(run_dir).expanduser().resolve()
    ledger = Ledger.open(run_dir, task_id)
    task_dir = run_dir / "tasks" / task_id
    nodes = list(_latest_nodes(task_dir / "tree.jsonl").values())
    compact = [{
        "node_id": item["node_id"], "parent": item["parent"],
        "hypothesis": item["hypothesis"], "status": item["status"],
        "score": item["score"], "cost": item["cost"],
        "failure": item["failure"], "cites": item["cites"],
    } for item in nodes if item["node_id"] != "n0000"][-max_nodes:]
    return {"schema": "task_summary_input/1", "task_id": task_id,
            "experiments": compact, "spent": ledger.spent(),
            "elapsed": ledger.elapsed(),
            "source": {"tree": str(task_dir / "tree.jsonl"),
                       "ledger": str(ledger.path)}}


def _check_raw_lift(ending: dict) -> float | None:
    """Return holdout_raw_lift after checking it equals best raw minus baseline raw."""
    lift = ending.get("holdout_raw_lift")
    if lift is None:
        return None
    parts = [ending.get(key) for key in ("holdout_raw", "baseline_holdout_raw")]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in [lift, *parts]):
        raise ValueError("Holdout raw lift needs numeric raw measurements")
    if not math.isclose(float(lift), float(parts[0]) - float(parts[1]), rel_tol=0, abs_tol=1e-9):
        raise ValueError("Holdout raw lift differs from best raw minus baseline raw")
    return float(lift)


def task_report(run_dir: Path, task_id: str) -> dict:
    """Validate and summarize one completed task with ledger-row citations."""
    run_dir = Path(run_dir).expanduser().resolve()
    ledger = Ledger.open(run_dir, task_id)
    events = ledger._events()
    task_dir = run_dir / "tasks" / task_id
    endings = [event for event in events if event["kind"] == "task_end"]
    stops = [event for event in events if event["kind"] == "stop"]
    if len(endings) != 1 or len(stops) != 1:
        raise ValueError("Completed task requires one stop and one task_end")
    ending, stop = endings[0], stops[0]
    if ending["best_node"] != stop["best_node"]:
        raise ValueError("Best node differs between stop and task_end")
    holdout = json.loads((task_dir / "holdout_score.json").read_text(encoding="utf-8"))
    if holdout.get("schema") != "score/1" or holdout.get("split") != "holdout":
        raise ValueError("Invalid holdout score file")
    holdout_failed = ending.get("holdout_failed") is True
    if holdout_failed:
        # A failed one-shot holdout is evidence, not a score: both records say null.
        if ending.get("holdout_score") is not None or holdout.get("score") is not None \
                or holdout.get("ok") is not False:
            raise ValueError("Failed holdout must record a null score in task_end and holdout_score.json")
    else:
        if ending.get("holdout_score") is None or holdout.get("score") is None:
            raise ValueError("Holdout has no score and task_end lacks a holdout_failed marker")
        if not math.isclose(float(ending["holdout_score"]), float(holdout["score"]), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Holdout score differs from task_end")
    raw_lift = _check_raw_lift(ending)
    model_events = [event for event in events if event["kind"] == "llm_call"]
    billed = sum(float(event["cost"]) for event in model_events)
    if not math.isclose(billed, float(ending["total_cost"]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Task cost differs from sum of model calls")
    nodes = _latest_nodes(task_dir / "tree.jsonl")
    best = nodes.get(ending["best_node"])
    if best is None or best["status"] != "accepted":
        raise ValueError("Task best node is not an accepted tree node")
    tokens = {key: sum(int(event["tokens"][key]) for event in model_events)
              for key in ("input", "cached_input", "output", "reasoning")}
    starts = [event for event in events if event["kind"] == "task_start"]
    budget = starts[0].get("budget") if len(starts) == 1 else None
    budget_valid = type(budget) in (int, float) and math.isfinite(budget) and budget > 0
    # Overspending is a protocol violation to report, not a reason to lose the run.
    budget_violation = bool(budget_valid and billed > budget)
    budget_excess = float(billed - budget) if budget_violation else 0.0
    if (ledger.manifest["phase"] != "dev"
            and is_exact_token_accounting(ledger.prices, ledger.manifest["model"])):
        if (not budget_valid
                or not math.isclose(billed, sum(tokens.values()), rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError("Formal token task has an invalid budget or ledger units differ")
        library_root = run_dir.parent.parent / ledger.manifest["library"]["root"]
        replay = _verify_replay_context(run_dir=run_dir, task_id=task_id,
                                         node=best, library_root=library_root)
        if (type(replay.get("max_output_tokens")) is not int
                or not 1 <= replay["max_output_tokens"] <= 2048):
            raise ValueError("Formal token replay exceeds the call output limit")
    role_cost = {role: sum(float(event["cost"]) for event in model_events if event["role"] == role)
                 for role in ("planner", "executor", "diagnoser", "summarizer")}
    quotes = [event for event in events if event["kind"] == "quote"]
    if len(quotes) > 1:
        raise ValueError("Task has more than one quote")
    return {
        "schema": "task_report/1", "run_id": run_dir.name, "task_id": task_id,
        "best_node": ending["best_node"], "dev_score": stop["dev_score"],
        "holdout_score": None if holdout_failed else ending["holdout_score"],
        "holdout_failed": holdout_failed,
        "holdout_error": holdout.get("error") if holdout_failed else None,
        # Same-data reference: AUC difference (tabular) or speedup-ratio difference.
        "holdout_raw": ending.get("holdout_raw"),
        "baseline_holdout_score": ending.get("baseline_holdout_score"),
        "baseline_holdout_raw": ending.get("baseline_holdout_raw"),
        "baseline_holdout_failed": ending.get("baseline_holdout_failed") is True,
        "holdout_raw_lift": raw_lift,
        "budget": float(budget) if budget_valid else None,
        "budget_violation": budget_violation, "budget_excess": budget_excess,
        "total_cost": ending["total_cost"],
        "currency": ledger.prices["currency"], "seconds": ending["seconds"],
        "quote": quotes[0]["predicted"] if quotes else None,
        "quote_error": ending["quote_error"],
        "model_calls": len(model_events), "tokens": tokens, "cost_by_role": role_cost,
        "experiments": len([node for node in nodes.values() if node["node_id"] != "n0000"]),
        "accepted_nodes": [node["node_id"] for node in nodes.values() if node["node_id"] != "n0000" and node["status"] == "accepted"],
        "evidence": {
            "ledger_path": str(ledger.path), "task_end_seq": ending["seq"],
            "stop_seq": stop["seq"], "quote_seq": quotes[0]["seq"] if quotes else None,
            "model_call_seqs": [event["seq"] for event in model_events],
            "tree_path": str(task_dir / "tree.jsonl"),
            "holdout_path": str(task_dir / "holdout_score.json"),
            "baseline_holdout_path": str(task_dir / "baseline_holdout.json"),
        },
    }


def run_report(run_dir: Path) -> dict:
    """Build a run-wide report only after run_end and all task reports validate."""
    run_dir = Path(run_dir).expanduser().resolve()
    ledger = Ledger.open(run_dir, None)
    run_ends = [event for event in ledger._events() if event["kind"] == "run_end"]
    if len(run_ends) != 1:
        raise ValueError("Run has no unique run_end")
    tasks = [task_report(run_dir, task_id) for task_id in ledger.manifest["tasks"]]
    total = sum(float(task["total_cost"]) for task in tasks)
    if not math.isclose(total, float(run_ends[0]["total_cost"]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Run total differs from task ledgers")
    violations = [task for task in tasks if task.get("budget_violation")]
    failures = [task for task in tasks if task.get("holdout_failed")]
    report = {"schema": "run_report/1", "run_id": run_dir.name,
              "phase": ledger.manifest["phase"], "config": ledger.manifest["config"],
              "rep": ledger.manifest["rep"], "currency": ledger.prices["currency"],
              "exact_token_accounting": is_exact_token_accounting(ledger.prices, ledger.manifest["model"]),
              "total_cost": total,
              "budget_violations": len(violations),
              "budget_violation_task_ids": [task["task_id"] for task in violations],
              "holdout_failures": len(failures),
              "holdout_failed_task_ids": [task["task_id"] for task in failures],
              "tasks": tasks,
              "evidence": {"run_ledger_path": str(ledger.path), "run_end_seq": run_ends[0]["seq"]}}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["extract_summary", "run_report", "task_report"]
