"""Exclusion-aware inputs for admission and comparison.

A task whose one-shot holdout failed carries ``holdout_score: null`` and
``holdout_failed: true`` in its task report. Paired comparison needs the same
task list in every run, so such a task is dropped from *every* run of the
compared group and listed in the output. Nothing is imputed, and admission is
only decided when all required repetitions exist; otherwise the claim is
``incomplete``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .admission import admit_configuration


REPS = (1, 2, 3)
INCOMPLETE_RULE = ("Admission needs three completed paired repetitions of both "
                   "configurations; otherwise no claim is made.")


def _unscored(task: Mapping) -> bool:
    return bool(task.get("holdout_failed")) or task.get("holdout_score") is None


def comparable_reports(reports: Sequence[Mapping]) -> tuple[list[Mapping], list[dict]]:
    """Return reports without unscored tasks plus one exclusion record per task.

    With no failures the original report objects are returned untouched, so a
    complete study produces byte-identical comparisons to the legacy path.
    """
    reports = list(reports)  # iterated twice below
    failed: dict[str, list[dict]] = {}
    for report in reports:
        for task in report.get("tasks", []):
            if _unscored(task):
                failed.setdefault(task["task_id"], []).append({
                    "run_id": report.get("run_id"), "config": report.get("config"),
                    "rep": report.get("rep")})
    if not failed:
        return list(reports), []
    filtered = []
    for report in reports:
        kept = [task for task in report.get("tasks", []) if task["task_id"] not in failed]
        filtered.append({
            **report, "tasks": kept,
            # Cost is compared on the same remaining tasks as the scores.
            "total_cost": sum(float(task["total_cost"]) for task in kept),
            "original_total_cost": report.get("total_cost"),
            "excluded_task_ids": sorted(failed),
        })
    excluded = [{"task_id": task_id, "reason": "holdout_failed", "failed_in": runs}
                for task_id, runs in failed.items()]
    return filtered, excluded


def _incomplete(baseline: str, candidate: str, missing: list[dict], excluded: list[dict],
                reason: str) -> dict:
    return {"schema": "admission/1", "claim": "incomplete", "baseline": baseline,
            "candidate": candidate, "score_not_lower_each_rep": None,
            "cost_reduced_on_average": None, "comparison": None,
            "missing_runs": missing, "excluded_tasks": excluded, "reason": reason,
            "rule": INCOMPLETE_RULE}


def admit_complete(reports: Sequence[Mapping], *, baseline: str = "default_lib",
                   candidate: str = "tuned_lib") -> dict:
    """Admission that is computed only from a full set of paired repetitions."""
    present = {(report.get("config"), report.get("rep")) for report in reports}
    missing = [{"config": config, "rep": rep} for config in (baseline, candidate)
               for rep in REPS if (config, rep) not in present]
    if missing:
        return _incomplete(baseline, candidate, missing, [],
                           "required validation runs did not complete")
    comparable, excluded = comparable_reports(reports)
    if not all(report["tasks"] for report in comparable):
        return _incomplete(baseline, candidate, [], excluded,
                           "no task has a scored holdout in every run")
    admission = admit_configuration(comparable, baseline=baseline, candidate=candidate)
    if excluded:
        admission = {**admission, "excluded_tasks": excluded,
                     "comparison": {**admission["comparison"], "excluded_tasks": excluded}}
    return admission


__all__ = ["admit_complete", "comparable_reports"]
