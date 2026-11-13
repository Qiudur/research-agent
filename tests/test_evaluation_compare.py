import unittest

from evaluation.compare import compare_configurations


def report(config, rep, score, cost=1):
    return {"schema": "run_report/1", "run_id": f"run-{config}-{rep}",
            "phase": "validation", "config": config, "rep": rep,
            "tasks": [{"task_id": "spd-demo", "holdout_score": score}],
            "total_cost": cost}


class ComparisonTests(unittest.TestCase):
    def test_three_consistent_gains_above_noise_can_be_called_better(self):
        rows = [report("naive", i, score) for i, score in enumerate((0.1, 0.11, 0.09), 1)]
        rows += [report("default_lib", i, score, cost=2) for i, score in enumerate((0.4, 0.39, 0.41), 1)]
        result = compare_configurations(rows)
        self.assertEqual(result["claim"], "better")
        self.assertEqual(result["candidate_mean_cost"], 2)
        self.assertEqual(result["evidence_run_ids"]["naive"], ["run-naive-1", "run-naive-2", "run-naive-3"])

    def test_mixed_directions_report_no_clear_difference(self):
        rows = [report("naive", i, 0.2) for i in (1, 2, 3)]
        rows += [report("default_lib", i, score) for i, score in enumerate((0.3, 0.1, 0.4), 1)]
        self.assertEqual(compare_configurations(rows)["claim"], "no_clear_difference")

    def test_requires_three_reps(self):
        with self.assertRaisesRegex(ValueError, "exactly reps"):
            compare_configurations([report("naive", 1, 0.1), report("default_lib", 1, 0.2)])


if __name__ == "__main__":
    unittest.main()
