"""Validation-set admission gate from the frozen evaluation plan.

The gate is deliberately separate from score-only comparison: a tuned
configuration must keep scores from falling across all three paired runs and
reduce mean cost before it can replace the default.
"""

from __future__ import annotations

from typing import Mapping

from .compare import compare_configurations


def admit_configuration(
    reports: list[Mapping], *, baseline: str = "default", candidate: str = "default_lib",
) -> dict:
    """Return an auditable admission decision for exactly three paired reps."""
    comparison = compare_configurations(reports, baseline=baseline, candidate=candidate)
    score_not_lower = all(
        candidate_score >= baseline_score
        for candidate_score, baseline_score in zip(
            comparison["candidate_scores"], comparison["baseline_scores"]
        )
    )
    cost_reduced = comparison["candidate_mean_cost"] < comparison["baseline_mean_cost"]
    return {
        "schema": "admission/1",
        "claim": "admitted" if score_not_lower and cost_reduced else "rejected",
        "baseline": baseline,
        "candidate": candidate,
        "score_not_lower_each_rep": score_not_lower,
        "cost_reduced_on_average": cost_reduced,
        "comparison": comparison,
        "rule": "Three paired repetitions; candidate score may not be lower in any repetition and its mean cost must be lower.",
    }


__all__ = ["admit_configuration"]
