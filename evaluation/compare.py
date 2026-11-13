"""Three-repeat comparison rule from the original evaluation plan."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import Mapping


def compare_configurations(
    reports: list[Mapping], *, baseline: str = "naive", candidate: str = "default_lib",
) -> dict:
    """Compare paired repetitions only when each run covers identical tasks."""
    grouped: dict[str, dict[int, Mapping]] = defaultdict(dict)
    phase = None
    task_ids = None
    for report in reports:
        if report.get("schema") != "run_report/1":
            raise ValueError("Only audited run_report/1 data may be compared")
        if phase is None:
            phase = report["phase"]
        if report["phase"] != phase:
            raise ValueError("All compared runs must have the same phase")
        ids = tuple(task["task_id"] for task in report["tasks"])
        if task_ids is None:
            task_ids = ids
        if ids != task_ids or not ids:
            raise ValueError("All compared runs must cover the same ordered tasks")
        config = report["config"]
        rep = report["rep"]
        if rep in grouped[config]:
            raise ValueError(f"Duplicate {config} repetition {rep}")
        grouped[config][rep] = report
    if set(grouped.get(baseline, {})) != {1, 2, 3} or set(grouped.get(candidate, {})) != {1, 2, 3}:
        raise ValueError("Both configurations need exactly reps 1, 2, 3")
    base = [grouped[baseline][rep] for rep in (1, 2, 3)]
    trial = [grouped[candidate][rep] for rep in (1, 2, 3)]
    scores_base = [mean(float(task["holdout_score"]) for task in report["tasks"]) for report in base]
    scores_trial = [mean(float(task["holdout_score"]) for task in report["tasks"]) for report in trial]
    costs_base = [float(report["total_cost"]) for report in base]
    costs_trial = [float(report["total_cost"]) for report in trial]
    differences = [trial_score - base_score for trial_score, base_score in zip(scores_trial, scores_base)]
    mean_gap = mean(differences)
    noise = max(stdev(scores_base), stdev(scores_trial))
    better = all(value > 0 for value in differences) and mean_gap > noise
    return {
        "schema": "comparison/1", "phase": phase, "tasks": list(task_ids),
        "baseline": baseline, "candidate": candidate,
        "baseline_scores": scores_base, "candidate_scores": scores_trial,
        "paired_differences": differences, "mean_score_gap": mean_gap,
        "within_config_noise": noise, "direction_consistent": all(value > 0 for value in differences),
        "claim": "better" if better else "no_clear_difference",
        "baseline_mean_cost": mean(costs_base), "candidate_mean_cost": mean(costs_trial),
        "evidence_run_ids": {baseline: [report["run_id"] for report in base],
                             candidate: [report["run_id"] for report in trial]},
        "rule": "Three paired repetitions must all improve; mean score gap must exceed both configurations' sample standard deviations.",
    }


__all__ = ["compare_configurations"]
