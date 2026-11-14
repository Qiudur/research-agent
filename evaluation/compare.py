"""Comparison rules: the original three-repeat rule and the paired A/B analysis."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean, stdev
from typing import Mapping, Sequence


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


# ---------------------------------------------------------------------------
# Paired (task, repetition) comparison used by the A/B experiments.
#
# Normalized tabular scores divide by (ceiling - baseline), which is tiny for a
# few tasks, so one task can swing a ten-task mean by hundreds of points. This
# analysis therefore measures tabular quality as the raw AUC lift, clips the
# speedup score, and pairs every (task, repetition) of two configurations.

QUALITY_CLIP = (-1.0, 2.0)
BOOTSTRAP_SEED = 20261115
BOOTSTRAP_RESAMPLES = 10000
_TIE = 1e-12
_SCENARIOS = {"tab": "tab", "tabular": "tab", "spd": "spd", "speedup": "spd"}


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def truncate_score(value, low: float = QUALITY_CLIP[0], high: float = QUALITY_CLIP[1]):
    """Clip a score to [low, high]; non-numeric or non-finite input gives None."""
    number = _finite(value)
    return None if number is None else min(high, max(low, number))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED, confidence: float = 0.95):
    """Percentile bootstrap interval of the mean; None below two observations."""
    if len(values) < 2:
        return None
    if not 0 < confidence < 1 or n_boot < 100:
        raise ValueError("Need 0 < confidence < 1 and at least 100 resamples")
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(rng.choices(values, k=count)) / count for _ in range(n_boot))
    tail = (1 - confidence) / 2
    return (_percentile(means, tail), _percentile(means, 1 - tail))


def sign_test(values: Sequence[float], *, tolerance: float = _TIE) -> dict:
    """Exact two-sided sign test on candidate-minus-baseline differences."""
    higher = sum(1 for value in values if value > tolerance)
    lower = sum(1 for value in values if value < -tolerance)
    ties = len(values) - higher - lower
    trials = higher + lower
    if trials == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(trials, k) for k in range(min(higher, lower) + 1))
        p_value = min(1.0, 2 * tail / 2 ** trials)
    return {"candidate_higher": higher, "candidate_lower": lower, "ties": ties,
            "p_two_sided": p_value}


def summarize_differences(values: Sequence[float], *, n_boot: int = BOOTSTRAP_RESAMPLES,
                          seed: int = BOOTSTRAP_SEED, confidence: float = 0.95) -> dict:
    """Count, mean, bootstrap interval and sign test of paired differences."""
    interval = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed, confidence=confidence)
    return {"n": len(values), "mean": mean(values) if values else None,
            "ci95": list(interval) if interval else None,
            "sign_test": sign_test(values)}


def _quality_verdict(stats: Mapping, margin: float) -> str:
    interval = stats["ci95"]
    if interval is None:
        return "insufficient_data"
    low, high = interval
    if low > 0:
        return "improved"
    if low >= -margin:
        return "not_worse"
    if high < 0:
        return "worse"
    return "inconclusive"


def _cost_verdict(stats: Mapping) -> str:
    interval = stats["ci95"]
    if interval is None:
        return "insufficient_data"
    if interval[1] < 0:
        return "cheaper"
    if interval[0] > 0:
        return "costlier"
    return "no_clear_difference"


def _scenario_code(value) -> str:
    try:
        return _SCENARIOS[value]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown scenario {value!r}; use tab/tabular or spd/speedup") from None


def _pair_quality(base: Mapping, cand: Mapping, scenario: str):
    """(basis, baseline quality, candidate quality) or (None, None, None)."""
    if scenario == "spd":
        low, high = truncate_score(base.get("holdout_score")), truncate_score(cand.get("holdout_score"))
        return ("holdout_score_truncated", low, high) if None not in (low, high) else (None, None, None)
    lifts = _finite(base.get("holdout_raw_lift")), _finite(cand.get("holdout_raw_lift"))
    if None not in lifts:
        return ("holdout_raw_lift", *lifts)
    raws = _finite(base.get("holdout_raw")), _finite(cand.get("holdout_raw"))
    if None not in raws:
        return ("holdout_raw", *raws)
    return (None, None, None)


def _margin(quality_margin, scenario: str) -> float:
    value = quality_margin.get(scenario, 0.0) if isinstance(quality_margin, Mapping) else quality_margin
    value = _finite(value)
    if value is None or value < 0:
        raise ValueError("quality_margin must be a nonnegative number")
    return value


def compare_paired_tasks(
    rows: Sequence[Mapping], *, baseline: str, candidate: str,
    quality_margin: float | Mapping[str, float] = 0.0,
    n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Pair two configurations by (task, repetition) and judge cost and quality.

    Each row describes one finished task of one repetition: config, rep,
    task_id, scenario (tab/tabular or spd/speedup), total_tokens, holdout_score,
    holdout_raw and holdout_raw_lift (missing or None values are allowed).

    Quality is the tabular holdout_raw_lift (raw AUC difference when either
    lift is missing) and the speedup holdout score clipped to [-1, 2]. Cost is
    total tokens per task. Differences are candidate minus baseline, so a
    negative cost difference is a saving. Intervals are percentile bootstrap
    intervals of the mean with a fixed seed.

    The verdict is judged per scenario because the two quality scales differ:
    "not_worse" needs the interval's lower bound to be at least minus the
    margin (non-inferiority; default margin 0), "worse" needs the whole interval
    below zero. "cheaper_and_not_worse" requires the pooled token-saving
    interval to lie below zero and no scenario to be worse or inconclusive.
    The pooled quality statistics mix scales and are context only.
    """
    if baseline == candidate:
        raise ValueError("Baseline and candidate must be different configurations")
    index: dict[tuple, Mapping] = {}
    for row in rows:
        config = row.get("config")
        if config not in (baseline, candidate):
            continue
        key = (config, row["task_id"], int(row["rep"]))
        if key in index:
            raise ValueError(f"Duplicate row for {key}")
        index[key] = row
    base_keys = {key[1:] for key in index if key[0] == baseline}
    cand_keys = {key[1:] for key in index if key[0] == candidate}
    shared = sorted(base_keys & cand_keys)
    unpaired = [{"config": baseline if key in base_keys else candidate,
                 "task_id": key[0], "rep": key[1]}
                for key in sorted(base_keys ^ cand_keys)]
    pairs = []
    for task_id, rep in shared:
        base, cand = index[(baseline, task_id, rep)], index[(candidate, task_id, rep)]
        scenario = _scenario_code(base.get("scenario"))
        if _scenario_code(cand.get("scenario")) != scenario:
            raise ValueError(f"Scenario differs between configurations for {task_id}")
        basis, base_quality, cand_quality = _pair_quality(base, cand, scenario)
        base_tokens, cand_tokens = _finite(base.get("total_tokens")), _finite(cand.get("total_tokens"))
        have_cost = None not in (base_tokens, cand_tokens)
        pairs.append({
            "task_id": task_id, "rep": rep, "scenario": scenario,
            "quality_basis": basis,
            "baseline_quality": base_quality, "candidate_quality": cand_quality,
            "quality_diff": None if basis is None else cand_quality - base_quality,
            "baseline_tokens": base_tokens, "candidate_tokens": cand_tokens,
            "token_diff": cand_tokens - base_tokens if have_cost else None,
            "token_ratio": cand_tokens / base_tokens if have_cost and base_tokens > 0 else None,
            "baseline_holdout_score_truncated": truncate_score(base.get("holdout_score")),
            "candidate_holdout_score_truncated": truncate_score(cand.get("holdout_score")),
        })

    def stats(values):
        return summarize_differences(values, n_boot=n_boot, seed=seed)

    scenarios = sorted({pair["scenario"] for pair in pairs})
    quality_by_scenario, cost_by_scenario = {}, {}
    for scenario in scenarios:
        in_scenario = [p for p in pairs if p["scenario"] == scenario]
        margin = _margin(quality_margin, scenario)
        entry = stats([p["quality_diff"] for p in in_scenario if p["quality_diff"] is not None])
        entry.update(margin=margin, verdict=_quality_verdict(entry, margin),
                     bases=sorted({p["quality_basis"] for p in in_scenario if p["quality_basis"]}))
        quality_by_scenario[scenario] = entry
        cost_entry = stats([p["token_diff"] for p in in_scenario if p["token_diff"] is not None])
        cost_entry["verdict"] = _cost_verdict(cost_entry)
        cost_by_scenario[scenario] = cost_entry
    costed = [p for p in pairs if p["token_diff"] is not None]
    cost_overall = stats([p["token_diff"] for p in costed])
    cost_overall["verdict"] = _cost_verdict(cost_overall)
    if costed:
        base_mean = mean(p["baseline_tokens"] for p in costed)
        cost_overall.update(
            mean_baseline_tokens=base_mean,
            mean_candidate_tokens=mean(p["candidate_tokens"] for p in costed),
            relative_change=(cost_overall["mean"] / base_mean) if base_mean > 0 else None)
    quality_overall = stats([p["quality_diff"] for p in pairs if p["quality_diff"] is not None])
    quality_overall["note"] = "Pooled over scenarios with different scales; context only."

    verdicts = [entry["verdict"] for entry in quality_by_scenario.values()]
    if not verdicts:
        quality_verdict = "insufficient_data"
    elif "worse" in verdicts:
        quality_verdict = "worse"
    elif any(v in ("inconclusive", "insufficient_data") for v in verdicts):
        quality_verdict = "inconclusive"
    else:
        quality_verdict = "improved" if all(v == "improved" for v in verdicts) else "not_worse"
    cost_verdict = cost_overall["verdict"]
    if cost_verdict == "cheaper":
        headline = {"improved": "cheaper_and_not_worse", "not_worse": "cheaper_and_not_worse",
                    "worse": "cheaper_but_worse"}.get(quality_verdict, "cheaper_quality_inconclusive")
    else:
        headline = {"costlier": "costlier", "no_clear_difference": "no_clear_saving"}.get(
            cost_verdict, "insufficient_data")

    def truncated(config: str) -> dict:
        key = "baseline_holdout_score_truncated" if config == baseline else "candidate_holdout_score_truncated"
        out = {}
        for scope in ("overall", *scenarios):
            values = [p[key] for p in pairs if p[key] is not None and scope in ("overall", p["scenario"])]
            out[scope] = {"n": len(values), "mean": mean(values) if values else None}
        return out

    per_task = []
    for task_id in sorted({p["task_id"] for p in pairs}):
        group = [p for p in pairs if p["task_id"] == task_id]
        qualities = [p["quality_diff"] for p in group if p["quality_diff"] is not None]
        tokens = [p for p in group if p["token_diff"] is not None]
        per_task.append({
            "task_id": task_id, "scenario": group[0]["scenario"], "pairs": len(group),
            "mean_quality_diff": mean(qualities) if qualities else None,
            "mean_token_diff": mean(p["token_diff"] for p in tokens) if tokens else None,
            "mean_baseline_tokens": mean(p["baseline_tokens"] for p in tokens) if tokens else None,
            "mean_candidate_tokens": mean(p["candidate_tokens"] for p in tokens) if tokens else None,
        })
    return {
        "schema": "paired_comparison/1", "baseline": baseline, "candidate": candidate,
        "definitions": {
            "quality": "tab: holdout_raw_lift (raw AUC difference when a lift is missing); "
                       f"spd: holdout score clipped to {list(QUALITY_CLIP)}",
            "cost": "total tokens per task", "difference": "candidate minus baseline",
            "interval": f"percentile bootstrap of the mean, {n_boot} resamples, seed {seed}",
            "sign_test": "exact two-sided, ties excluded",
            "independence": "(task, repetition) pairs are resampled as independent; repetitions of "
                            "one task share its difficulty, so intervals are optimistic when only "
                            "a few tasks are used",
        },
        "n_pairs": len(pairs), "pairs": pairs, "unpaired": unpaired, "per_task": per_task,
        "quality": {"overall": quality_overall, "by_scenario": quality_by_scenario,
                    "verdict": quality_verdict},
        "cost": {"overall": cost_overall, "by_scenario": cost_by_scenario,
                 "verdict": cost_verdict},
        "truncated_mean_holdout_score": {baseline: truncated(baseline),
                                         candidate: truncated(candidate),
                                         "clip": list(QUALITY_CLIP)},
        "conclusion": headline,
        "claim_supported": headline == "cheaper_and_not_worse",
    }


__all__ = ["BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "QUALITY_CLIP", "bootstrap_mean_ci",
           "compare_configurations", "compare_paired_tasks", "sign_test",
           "summarize_differences", "truncate_score"]
