import unittest

from evaluation.admission import admit_configuration


def report(config, rep, score, cost):
    return {"schema": "run_report/1", "run_id": f"run-{config}-{rep}",
            "phase": "validation", "config": config, "rep": rep,
            "tasks": [{"task_id": "spd-demo", "holdout_score": score}],
            "total_cost": cost}


class AdmissionTests(unittest.TestCase):
    def test_admits_non_degrading_lower_cost_configuration(self):
        rows = [report("default", rep, score, 1.0) for rep, score in enumerate((.2, .21, .19), 1)]
        rows += [report("default_lib", rep, score, .7) for rep, score in enumerate((.2, .22, .19), 1)]
        decision = admit_configuration(rows)
        self.assertEqual(decision["claim"], "admitted")
        self.assertTrue(decision["score_not_lower_each_rep"])
        self.assertTrue(decision["cost_reduced_on_average"])

    def test_rejects_any_score_regression_even_when_cheaper(self):
        rows = [report("default", rep, .2, 1.0) for rep in (1, 2, 3)]
        rows += [report("default_lib", rep, score, .1) for rep, score in enumerate((.3, .19, .3), 1)]
        decision = admit_configuration(rows)
        self.assertEqual(decision["claim"], "rejected")
        self.assertFalse(decision["score_not_lower_each_rep"])

    def test_rejects_higher_mean_cost(self):
        rows = [report("default", rep, .2, 1.0) for rep in (1, 2, 3)]
        rows += [report("default_lib", rep, .2, 1.1) for rep in (1, 2, 3)]
        decision = admit_configuration(rows)
        self.assertEqual(decision["claim"], "rejected")
        self.assertFalse(decision["cost_reduced_on_average"])


if __name__ == "__main__":
    unittest.main()
