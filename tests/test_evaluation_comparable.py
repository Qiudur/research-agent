"""Admission and comparison inputs with failed holdouts and missing runs."""

from __future__ import annotations

import copy
import unittest

from evaluation.admission import admit_configuration
from evaluation.comparable import admit_complete, comparable_reports


def report(config: str, rep: int, scores: dict, *, cost_each: float = 1.0) -> dict:
    tasks = [{"task_id": task_id, "holdout_score": score, "holdout_failed": score is None,
              "total_cost": cost_each * (index + 1)}
             for index, (task_id, score) in enumerate(scores.items())]
    return {"schema": "run_report/1", "run_id": f"run-{config}-{rep}", "phase": "dev",
            "config": config, "rep": rep, "tasks": tasks,
            "total_cost": sum(task["total_cost"] for task in tasks)}


def paired(base_scores, tuned_scores, *, tuned_cost: float = 0.5) -> list[dict]:
    reports = []
    for rep, (base, tuned) in enumerate(zip(base_scores, tuned_scores), 1):
        reports.append(report("default_lib", rep, base))
        reports.append(report("tuned_lib", rep, tuned, cost_each=tuned_cost))
    return reports


GOOD = [{"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.6}, {"a": 0.7, "b": 0.7}]
BETTER = [{"a": 0.7, "b": 0.7}, {"a": 0.8, "b": 0.8}, {"a": 0.9, "b": 0.9}]


class ComparableTests(unittest.TestCase):
    def test_complete_reports_pass_through_unchanged(self):
        reports = paired(GOOD, BETTER)
        kept, excluded = comparable_reports(reports)
        self.assertEqual(excluded, [])
        self.assertTrue(all(left is right for left, right in zip(kept, reports)))
        # Identical to the legacy path, so recorded admissions still verify.
        self.assertEqual(admit_complete(reports), admit_configuration(
            reports, baseline="default_lib", candidate="tuned_lib"))
        self.assertEqual(admit_complete(reports)["claim"], "admitted")

    def test_failed_task_is_dropped_from_every_run_and_listed(self):
        tuned = copy.deepcopy(BETTER)
        tuned[1]["b"] = None  # one holdout failed in one tuned run
        reports = paired(GOOD, tuned)
        original = copy.deepcopy(reports)
        kept, excluded = comparable_reports(reports)
        self.assertEqual(reports, original)  # inputs are never mutated
        self.assertEqual([item["task_id"] for item in excluded], ["b"])
        self.assertEqual(excluded[0]["reason"], "holdout_failed")
        self.assertEqual(excluded[0]["failed_in"],
                         [{"run_id": "run-tuned_lib-2", "config": "tuned_lib", "rep": 2}])
        for item in kept:
            self.assertEqual([task["task_id"] for task in item["tasks"]], ["a"])
            self.assertEqual(item["excluded_task_ids"], ["b"])
            # Cost is compared over the same remaining tasks as the scores.
            self.assertEqual(item["total_cost"], item["tasks"][0]["total_cost"])
        admission = admit_complete(reports)
        self.assertEqual(admission["claim"], "admitted")
        self.assertEqual(admission["excluded_tasks"], excluded)
        self.assertEqual(admission["comparison"]["excluded_tasks"], excluded)
        self.assertEqual(admission["comparison"]["tasks"], ["a"])
        # The failed task is excluded, not averaged in as zero.
        self.assertEqual(admission["comparison"]["candidate_scores"], [0.7, 0.8, 0.9])
        self.assertEqual(admission["comparison"]["baseline_scores"], [0.5, 0.6, 0.7])
        self.assertAlmostEqual(admission["comparison"]["candidate_mean_cost"], 0.5)

    def test_a_missing_repetition_makes_the_claim_incomplete_not_an_error(self):
        reports = [item for item in paired(GOOD, BETTER)
                   if (item["config"], item["rep"]) != ("tuned_lib", 3)]
        admission = admit_complete(reports)
        self.assertEqual(admission["schema"], "admission/1")
        self.assertEqual(admission["claim"], "incomplete")
        self.assertEqual(admission["missing_runs"], [{"config": "tuned_lib", "rep": 3}])
        self.assertIsNone(admission["comparison"])
        self.assertIsNone(admission["score_not_lower_each_rep"])
        self.assertEqual(admit_complete([])["claim"], "incomplete")
        self.assertEqual(len(admit_complete([])["missing_runs"]), 6)

    def test_nothing_left_to_compare_is_incomplete(self):
        failed = [{"a": None}, {"a": None}, {"a": None}]
        admission = admit_complete(paired([{"a": 0.5}, {"a": 0.6}, {"a": 0.7}], failed))
        self.assertEqual(admission["claim"], "incomplete")
        self.assertIn("no task", admission["reason"])
        self.assertEqual([item["task_id"] for item in admission["excluded_tasks"]], ["a"])

    def test_rejection_still_works_with_exclusions(self):
        worse = copy.deepcopy(GOOD)
        worse[0]["b"] = None
        admission = admit_complete(paired(BETTER, worse))
        self.assertEqual(admission["claim"], "rejected")
        self.assertEqual([item["task_id"] for item in admission["excluded_tasks"]], ["b"])


if __name__ == "__main__":
    unittest.main()
