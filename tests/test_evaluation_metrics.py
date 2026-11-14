from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.metrics import (find_run_dirs, node_completions, load_events, load_nodes,
                                run_metrics, scenario_of, task_metrics)
from tests.ledger_fixtures import TaskWriter, write_manifest


def speedup_task(run_dir: Path, *, reach: bool = True) -> Path:
    writer = TaskWriter(run_dir, "spd-demo")
    writer.plan(1000)
    writer.node("n0001", "rejected", 0.2, 2000)
    writer.node("n0002", "accepted", 0.6, 3000, diag=500)
    writer.plan(800)
    writer.node("n0003", "accepted", 1.05 if reach else 0.7, 2500)
    writer.node("n0004", "rejected", 0.9, 1500)
    writer.summarize(700)
    writer.stop("ceiling", "n0003")
    return writer.finish(holdout={"score": 1.0, "raw": 3.0})


class MetricsTests(unittest.TestCase):
    def test_scenario_prefix(self):
        self.assertEqual(scenario_of("tab-german-credit"), "tab")
        self.assertEqual(scenario_of("spd-distance-matrix"), "spd")
        with self.assertRaises(ValueError):
            scenario_of("other-task")

    def test_tokens_to_reference_best_roles_and_stop_reason(self):
        with TemporaryDirectory() as temp:
            metrics = task_metrics(speedup_task(Path(temp) / "run-a"))
        # planner 1000 + n0001 2000 + n0002 3000 + its diagnoser 500 + planner 800 + n0003 2500
        self.assertEqual(metrics["tokens_to_reference"], 9800)
        self.assertEqual(metrics["tokens_to_best"], 9800)
        self.assertTrue(metrics["reached_reference"])
        self.assertEqual(metrics["main_loop_tokens"], 11300)
        self.assertEqual(metrics["total_tokens"], 12000)
        self.assertEqual(metrics["tokens_after_best"], 1500)
        self.assertEqual(metrics["tokens_by_role"], {
            "planner": 1800, "executor": 9000, "diagnoser": 500, "summarizer": 700})
        self.assertEqual(sum(metrics["token_buckets"].values()), 12000)
        self.assertEqual(metrics["stop_reason"], "ceiling")
        self.assertEqual((metrics["experiments"], metrics["accepted_nodes"],
                          metrics["completed_nodes"]), (4, 2, 4))
        self.assertEqual(metrics["scenario"], "spd")
        self.assertEqual(metrics["best_node"], "n0003")
        self.assertAlmostEqual(metrics["best_dev_score"], 1.05)
        self.assertEqual(metrics["total_cost"], 12000.0)

    def test_reference_not_reached_is_null_and_best_is_tracked(self):
        with TemporaryDirectory() as temp:
            metrics = task_metrics(speedup_task(Path(temp) / "run-a", reach=False))
        self.assertIsNone(metrics["tokens_to_reference"])
        self.assertFalse(metrics["reached_reference"])
        self.assertEqual(metrics["tokens_to_best"], 9800)

    def test_no_improvement_means_zero_tokens_to_best(self):
        with TemporaryDirectory() as temp:
            writer = TaskWriter(Path(temp) / "run-a", "tab-demo")
            writer.plan(900).node("n0001", "rejected", 0.1, 1200).node(
                "n0002", "failed", None, 1100, scored=False)
            writer.stop("planner", "n0000")
            metrics = task_metrics(writer.finish())
        self.assertEqual(metrics["best_node"], "n0000")
        self.assertEqual(metrics["tokens_to_best"], 0)
        self.assertEqual(metrics["tokens_after_best"], metrics["main_loop_tokens"])
        self.assertIsNone(metrics["tokens_to_reference"])
        self.assertEqual(metrics["completed_nodes"], 2)  # a failed run still completes its node

    def test_completion_modes_differ_on_trailing_diagnosis_and_unrun_nodes(self):
        with TemporaryDirectory() as temp:
            writer = TaskWriter(Path(temp) / "run-a", "tab-demo")
            writer.plan(1000)
            writer.node("n0001", "rejected", 0.1, 2000, diag=400)
            writer.node("n0002", "failed", None, 1000, scored=False)
            writer.node("n0003", "failed", None, 600, ran=False)
            writer.stop("planner")
            task_dir = writer.finish()
            events, nodes = load_events(task_dir), load_nodes(task_dir)
            narrow = node_completions(events, nodes, completion="experiment_score")
            wide = node_completions(events, nodes, completion="node_events")
        # n0002 has an experiment event (failed run) but no score event: still completed.
        self.assertEqual([row["node_id"] for row in narrow], ["n0001", "n0002"])
        self.assertEqual([row["main_tokens"] for row in narrow], [3000, 4400])
        # n0003 never ran, and n0001's diagnoser call trails its score event.
        self.assertEqual([row["node_id"] for row in wide], ["n0001", "n0002", "n0003"])
        self.assertEqual([row["main_tokens"] for row in wide], [3400, 4400, 5000])
        self.assertEqual([row["status"] for row in wide], ["rejected", "failed", "failed"])
        with self.assertRaises(ValueError):
            node_completions(events, nodes, completion="bogus")

    def test_tabular_holdout_fields_and_missing_lift(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "run-a"
            writer = TaskWriter(run, "tab-demo")
            writer.plan(500).node("n0001", "accepted", 0.4, 700).stop("planner", "n0001")
            metrics = task_metrics(writer.finish(
                holdout={"score": 0.31, "raw": 0.8123},
                task_end={"holdout_raw_lift": 0.0151, "baseline_holdout_raw": 0.7972,
                          "baseline_holdout_score": 0.2, "holdout_failed": False}))
            self.assertAlmostEqual(metrics["holdout_raw"], 0.8123)
            self.assertAlmostEqual(metrics["holdout_raw_lift"], 0.0151)
            self.assertAlmostEqual(metrics["baseline_holdout_raw"], 0.7972)
            self.assertFalse(metrics["holdout_failed"])
            old = TaskWriter(Path(temp) / "run-b", "tab-demo")
            old.plan(500).node("n0001", "accepted", 0.4, 700).stop("planner", "n0001")
            legacy = task_metrics(old.finish(holdout={"score": 0.31, "raw": 0.8123}))
        self.assertIsNone(legacy["holdout_raw_lift"])
        self.assertIsNone(legacy["baseline_holdout_raw"])
        self.assertAlmostEqual(legacy["holdout_raw"], 0.8123)
        self.assertTrue(legacy["has_task_end"])

    def test_failed_holdout_is_flagged_without_scores(self):
        with TemporaryDirectory() as temp:
            writer = TaskWriter(Path(temp) / "run-a", "tab-demo")
            writer.plan(500).stop("planner")
            task_dir = writer.finish()
            (task_dir / "holdout_failure.json").write_text("{}")
            metrics = task_metrics(task_dir)
        self.assertTrue(metrics["holdout_failed"])
        self.assertIsNone(metrics["holdout_score"])
        self.assertFalse(metrics["has_task_end"])

    def test_new_stop_loss_reason_is_reported_verbatim(self):
        with TemporaryDirectory() as temp:
            writer = TaskWriter(Path(temp) / "run-a", "tab-demo")
            writer.plan(500).node("n0001", "rejected", 0.1, 700).stop("stop_loss")
            self.assertEqual(task_metrics(writer.finish())["stop_reason"], "stop_loss")

    def test_run_metrics_follow_manifest_order_and_find_nested_runs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "campaign/project/runs/run-20261115-000000-aaaa"
            write_manifest(run, phase="practice", tasks=["tab-demo", "spd-demo"])
            TaskWriter(run, "spd-demo").plan(100).stop("planner").finish()
            TaskWriter(run, "tab-demo").plan(200).stop("planner").finish()
            (root / "campaign/notes").mkdir()
            report = run_metrics(run)
            found = find_run_dirs([root])
        self.assertEqual([task["task_id"] for task in report["tasks"]], ["tab-demo", "spd-demo"])
        self.assertEqual(report["phase"], "practice")
        self.assertEqual([path.name for path in found], [run.name])


if __name__ == "__main__":
    unittest.main()
