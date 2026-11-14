"""Derive one bounded route candidate from completed, practice-only Qwen evidence."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

from ledger import Ledger
from research_agent.routing import validate_policy


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nodes(path: Path) -> dict[str, dict]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("schema") != "node/1":
                raise ValueError("Baseline tree has an unsupported node schema")
            latest[row["node_id"]] = row
    return latest


def calibrate_policy(
    baseline_campaign: Path, *, default_model: str,
    planner_model: str, executor_model: str,
) -> dict:
    """Use visible practice reports, ledgers and trees; never hidden task data."""
    campaign = Path(baseline_campaign).expanduser().resolve(strict=True)
    summary_path = campaign / "summary.json"
    report_path = campaign / "project/runs/default-study/self_research_report.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (summary.get("schema") != "token_parallel_default_summary/1"
            or summary.get("run_count") != 3 or report.get("schema") != "self_research/1"
            or report.get("practice_tasks") is None or len(report["practice_tasks"]) != 10
            or len(report.get("reports", [])) != 3):
        raise ValueError("Calibration requires completed Qwen-only ten-practice, three-repetition evidence")
    task_ids = set(report["practice_tasks"])
    if len(task_ids) != 10:
        raise ValueError("Baseline practice tasks are duplicated")
    rows = []
    baseline_commits = set()
    exposure = {"executor_failures": 0, "plateau_sequences": 0,
                "score_drops": 0, "failed_nodes": 0}
    seen = set()
    total = 0.0
    for run in report["reports"]:
        run_dir = campaign / "project/runs" / run["run_id"]
        if not run_dir.resolve().is_relative_to(campaign) or not run_dir.is_dir():
            raise ValueError("Baseline run path escapes the campaign")
        if {task["task_id"] for task in run["tasks"]} != task_ids:
            raise ValueError("Each baseline repetition must contain exactly the ten practice tasks")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "run/1" or manifest.get("model") != default_model:
            raise ValueError("Baseline run does not seal the expected Qwen model")
        baseline_commits.add(manifest.get("code_commit"))
        for task in run["tasks"]:
            task_id = task["task_id"]
            if (run["run_id"], task_id) in seen:
                raise ValueError("Duplicate baseline task result")
            seen.add((run["run_id"], task_id))
            ledger = Ledger.open(run_dir, task_id)
            events = ledger._events()
            if not any(event["kind"] == "task_end" for event in events):
                raise ValueError("Baseline task is incomplete")
            if abs(ledger.spent() - float(task["total_cost"])) > 1e-8:
                raise ValueError("Baseline task cost does not reconcile")
            tree = _nodes(run_dir / "tasks" / task_id / "tree.jsonl")
            ordered = [node for node in tree.values() if node["node_id"] != "n0000"]
            failures = sum(event["kind"] == "experiment" and
                           (event.get("exit_code") != 0 or event.get("timed_out") or event.get("oom"))
                           for event in events)
            failures += sum(event["kind"] == "llm_call" and event["role"] == "executor" and not event["ok"]
                            for event in events)
            plateau = 0
            streak = 0
            for node in ordered:
                if node["status"] in {"rejected", "failed"}:
                    streak += 1
                    if streak == 2:
                        plateau += 1
                else:
                    streak = 0
            drops = 0
            for node in ordered:
                if node.get("parent") not in tree or node.get("score") is None:
                    continue
                parent_score = tree[node["parent"]].get("score")
                if parent_score and node["score"]["dev"] < parent_score["dev"] - 0.1:
                    drops += 1
            failed_nodes = sum(node["status"] == "failed" for node in ordered)
            exposure["executor_failures"] += failures
            exposure["plateau_sequences"] += plateau
            exposure["score_drops"] += drops
            exposure["failed_nodes"] += failed_nodes
            rows.append({
                "run_id": run["run_id"], "task_id": task_id,
                "holdout_score": task["holdout_score"], "task_tokens": task["total_cost"],
                "executor_failures": failures, "plateau_sequences": plateau,
                "score_drops": drops, "failed_nodes": failed_nodes,
                "ledger_sha256": _sha(ledger.path),
                "tree_sha256": _sha(run_dir / "tasks" / task_id / "tree.jsonl"),
            })
            total += float(task["total_cost"])
    if len(seen) != 30 or abs(total - float(summary["total_task_tokens"])) > 1e-8:
        raise ValueError("Baseline summary does not reconcile to thirty task ledgers")
    if len(baseline_commits) != 1 or not next(iter(baseline_commits)):
        raise ValueError("Baseline repetitions do not share a frozen code commit")
    eligible = []
    if exposure["plateau_sequences"] >= 2:
        eligible.append("planner")
    if exposure["executor_failures"] >= 2:
        eligible.append("executor")
    if exposure["score_drops"] + exposure["failed_nodes"] >= 2:
        eligible.append("diagnoser")
    policy = {
        "schema": "route_policy/1",
        "default_model": default_model,
        "role_models": {"planner": planner_model, "executor": executor_model,
                        "diagnoser": executor_model},
        "eligible_roles": eligible,
        "thresholds": {
            "executor_failures": 2 if exposure["executor_failures"] >= 2 else 3,
            "plateau_nodes": 2 if exposure["plateau_sequences"] >= 2 else 3,
            "score_drop": 0.1 if exposure["score_drops"] >= 2 else 0.2,
        },
        "remote_call_cap": 2 if len(eligible) >= 2 else 1,
        "calibration": {
            "schema": "route_calibration/1",
            "algorithm": "bounded_practice_exposure_v1",
            "baseline_campaign": str(campaign),
            "baseline_summary_sha256": _sha(summary_path),
            "baseline_report_sha256": _sha(report_path),
            "baseline_code_commit": next(iter(baseline_commits)),
            "runs": 3, "tasks_per_run": 10,
            "total_task_tokens": total,
            "mean_holdout_score": mean(float(row["holdout_score"]) for row in rows),
            "exposure": exposure,
            "task_evidence": rows,
            "proposal_model_calls": 0,
            "proposal_tokens": 0,
        },
    }
    validate_policy(policy)
    return policy


__all__ = ["calibrate_policy"]
