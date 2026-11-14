import math
import unittest

from evaluation.compare import (bootstrap_mean_ci, compare_configurations,
                                compare_paired_tasks, sign_test, truncate_score)


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


def row(config, rep, task, tokens, *, score=None, raw=None, lift=None):
    return {"config": config, "rep": rep, "task_id": task,
            "scenario": "tab" if task.startswith("tab-") else "spd",
            "total_tokens": tokens, "holdout_score": score,
            "holdout_raw": raw, "holdout_raw_lift": lift}


def study(token_scale=0.85, tab_lift_delta=0.0, spd_delta=0.0, noisy=False):
    """Three tasks x three reps; the candidate spends token_scale of the baseline."""
    rows = []
    for rep in (1, 2, 3):
        for task, base_tokens in (("tab-a", 30000), ("tab-b", 24000), ("spd-c", 36000)):
            base_tokens += rep * 700
            wiggle = ((-1) ** rep) * 0.01 if noisy else 0.0
            rows.append(row("base", rep, task, base_tokens, score=1.0 + rep * 0.01,
                            raw=0.80 + rep * 0.001, lift=0.02 + rep * 0.001))
            rows.append(row("cand", rep, task, base_tokens * token_scale,
                            score=1.0 + rep * 0.01 + spd_delta + wiggle,
                            raw=0.80 + rep * 0.001 + tab_lift_delta + wiggle / 10,
                            lift=0.02 + rep * 0.001 + tab_lift_delta + wiggle / 10))
    return rows


class StatisticsTests(unittest.TestCase):
    def test_truncation_bounds_outliers_and_rejects_non_numbers(self):
        self.assertEqual(truncate_score(-341.0), -1.0)
        self.assertEqual(truncate_score(7), 2.0)
        self.assertEqual(truncate_score(0.4), 0.4)
        for bad in (None, True, float("nan"), float("inf"), "1"):
            self.assertIsNone(truncate_score(bad))

    def test_sign_test_is_exact_and_ignores_ties(self):
        self.assertAlmostEqual(sign_test([1, 2, 3, -1])["p_two_sided"], 0.625)
        self.assertAlmostEqual(sign_test([1] * 5)["p_two_sided"], 2 / 32)
        result = sign_test([0.0, 0.0, 1e-15, -2.0])
        self.assertEqual((result["candidate_higher"], result["candidate_lower"], result["ties"]),
                         (0, 1, 3))
        self.assertEqual(sign_test([0.0, 0.0])["p_two_sided"], 1.0)

    def test_bootstrap_interval_is_reproducible_and_brackets_the_mean(self):
        values = [-3.0, -1.0, -2.0, -4.0, -2.5]
        first = bootstrap_mean_ci(values)
        self.assertEqual(first, bootstrap_mean_ci(values))
        self.assertLess(first[0], sum(values) / len(values))
        self.assertGreater(first[1], sum(values) / len(values))
        self.assertNotEqual(first, bootstrap_mean_ci(values, seed=1))
        self.assertIsNone(bootstrap_mean_ci([1.0]))
        self.assertEqual(bootstrap_mean_ci([2.0, 2.0, 2.0]), (2.0, 2.0))


class PairedComparisonTests(unittest.TestCase):
    def test_clear_saving_with_unchanged_quality_supports_the_claim(self):
        result = compare_paired_tasks(study(0.85), baseline="base", candidate="cand")
        self.assertEqual(result["schema"], "paired_comparison/1")
        self.assertEqual(result["n_pairs"], 9)
        self.assertEqual(result["cost"]["verdict"], "cheaper")
        self.assertLess(result["cost"]["overall"]["ci95"][1], 0)
        self.assertAlmostEqual(result["cost"]["overall"]["relative_change"], -0.15)
        self.assertEqual(result["cost"]["overall"]["sign_test"]["candidate_lower"], 9)
        self.assertEqual(result["quality"]["verdict"], "not_worse")
        self.assertEqual(result["conclusion"], "cheaper_and_not_worse")
        self.assertTrue(result["claim_supported"])
        self.assertEqual(result["quality"]["by_scenario"]["tab"]["bases"], ["holdout_raw_lift"])
        self.assertEqual(result["quality"]["by_scenario"]["tab"]["n"], 6)
        self.assertEqual(result["quality"]["by_scenario"]["spd"]["n"], 3)

    def test_quality_drop_blocks_the_claim_even_when_cheaper(self):
        result = compare_paired_tasks(study(0.85, tab_lift_delta=-0.01), baseline="base",
                                      candidate="cand")
        self.assertEqual(result["quality"]["by_scenario"]["tab"]["verdict"], "worse")
        self.assertEqual(result["conclusion"], "cheaper_but_worse")
        self.assertFalse(result["claim_supported"])

    def test_noisy_quality_is_inconclusive_until_a_margin_covers_it(self):
        strict = compare_paired_tasks(study(0.85, noisy=True), baseline="base", candidate="cand")
        self.assertEqual(strict["conclusion"], "cheaper_quality_inconclusive")
        self.assertFalse(strict["claim_supported"])
        wide = compare_paired_tasks(study(0.85, noisy=True), baseline="base", candidate="cand",
                                    quality_margin={"tab": 0.02, "spd": 0.1})
        self.assertEqual(wide["conclusion"], "cheaper_and_not_worse")
        self.assertEqual(wide["quality"]["by_scenario"]["spd"]["margin"], 0.1)

    def test_no_saving_or_extra_cost_is_never_a_win(self):
        flat = compare_paired_tasks(study(1.0), baseline="base", candidate="cand")
        self.assertEqual(flat["conclusion"], "no_clear_saving")
        costly = compare_paired_tasks(study(1.2), baseline="base", candidate="cand")
        self.assertEqual(costly["conclusion"], "costlier")

    def test_tabular_falls_back_to_raw_auc_and_speedup_scores_are_clipped(self):
        rows = [row("base", 1, "tab-a", 100, raw=0.80, lift=None),
                row("cand", 1, "tab-a", 90, raw=0.83, lift=None),
                row("base", 2, "tab-a", 100, raw=0.80, lift=0.01),
                row("cand", 2, "tab-a", 90, raw=0.90, lift=0.03),
                row("base", 1, "spd-c", 100, score=-50.0),
                row("cand", 1, "spd-c", 90, score=9.0)]
        pairs = {(p["task_id"], p["rep"]): p
                 for p in compare_paired_tasks(rows, baseline="base", candidate="cand")["pairs"]}
        self.assertEqual(pairs[("tab-a", 1)]["quality_basis"], "holdout_raw")
        self.assertAlmostEqual(pairs[("tab-a", 1)]["quality_diff"], 0.03)
        self.assertEqual(pairs[("tab-a", 2)]["quality_basis"], "holdout_raw_lift")
        self.assertAlmostEqual(pairs[("tab-a", 2)]["quality_diff"], 0.02)
        self.assertEqual(pairs[("spd-c", 1)]["quality_diff"], 3.0)  # clip(9)=2 minus clip(-50)=-1
        self.assertAlmostEqual(pairs[("spd-c", 1)]["token_ratio"], 0.9)

    def test_one_extreme_score_does_not_dominate_truncated_means(self):
        rows = [row("base", 1, "tab-a", 100, score=-341.0, raw=0.8, lift=0.0),
                row("cand", 1, "tab-a", 90, score=0.5, raw=0.8, lift=0.0),
                row("base", 1, "spd-c", 100, score=1.0),
                row("cand", 1, "spd-c", 90, score=1.0)]
        result = compare_paired_tasks(rows, baseline="base", candidate="cand")
        truncated = result["truncated_mean_holdout_score"]
        self.assertAlmostEqual(truncated["base"]["overall"]["mean"], 0.0)
        self.assertAlmostEqual(truncated["cand"]["overall"]["mean"], 0.75)
        self.assertEqual(truncated["base"]["tab"]["n"], 1)
        self.assertEqual(result["quality"]["by_scenario"]["tab"]["verdict"], "insufficient_data")

    def test_unpaired_rows_are_listed_and_missing_quality_still_counts_cost(self):
        rows = [r for r in study(0.85)
                if not (r["config"] == "cand" and r["rep"] == 3 and r["task_id"] == "tab-b")]
        for item in rows:
            if item["config"] == "cand" and item["rep"] == 1 and item["task_id"] == "spd-c":
                item["holdout_score"] = None  # e.g. a failed holdout
        result = compare_paired_tasks(rows, baseline="base", candidate="cand")
        self.assertEqual(result["unpaired"], [{"config": "base", "task_id": "tab-b", "rep": 3}])
        self.assertEqual(result["n_pairs"], 8)
        self.assertEqual(result["cost"]["overall"]["n"], 8)
        self.assertEqual(result["quality"]["by_scenario"]["spd"]["n"], 2)

    def test_invalid_inputs_fail_loudly(self):
        rows = study()
        with self.assertRaisesRegex(ValueError, "different"):
            compare_paired_tasks(rows, baseline="base", candidate="base")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare_paired_tasks(rows + [rows[0]], baseline="base", candidate="cand")
        broken = [dict(r, scenario="spd") if r["config"] == "cand" and r["task_id"] == "tab-a"
                  else r for r in rows]
        with self.assertRaisesRegex(ValueError, "Scenario differs"):
            compare_paired_tasks(broken, baseline="base", candidate="cand")
        with self.assertRaisesRegex(ValueError, "Unknown scenario"):
            compare_paired_tasks([dict(r, scenario="x") for r in rows], baseline="base",
                                 candidate="cand")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            compare_paired_tasks(rows, baseline="base", candidate="cand", quality_margin=-1)

    def test_empty_overlap_reports_insufficient_data_without_crashing(self):
        result = compare_paired_tasks([row("base", 1, "tab-a", 5)], baseline="base",
                                      candidate="cand")
        self.assertEqual(result["n_pairs"], 0)
        self.assertEqual(result["conclusion"], "insufficient_data")
        self.assertEqual(result["unpaired"], [{"config": "base", "task_id": "tab-a", "rep": 1}])

    def test_results_are_json_serializable_and_reproducible(self):
        import json

        first = compare_paired_tasks(study(0.9), baseline="base", candidate="cand")
        second = compare_paired_tasks(study(0.9), baseline="base", candidate="cand")
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertTrue(all(math.isfinite(p["token_diff"]) for p in first["pairs"]))


if __name__ == "__main__":
    unittest.main()
