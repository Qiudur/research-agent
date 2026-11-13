"""Contract tests for A's append-only tree and rules-only scheduler."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research_agent.scheduler import ScheduleError, Scheduler
from research_agent.tree import ExperimentTree, TreeError


SETTINGS = {
    "schema": "agent/1",
    "plateau_nodes": 5,
    "behind_schedule": {"spent_ratio": 0.5, "progress_ratio": 0.5},
    "explore_every": 4,
}
CAPS = {"schema": "hard_caps/1", "budget_ratio": 0.9, "reserve_for_wrapup": 0.1}


def score(dev: float, *, spread: float = 0.01, cheat_flags: list[str] | None = None, correct: bool = True) -> dict:
    return {
        "schema": "score/1", "ok": True, "split": "dev", "score": dev,
        "raw": dev, "spread": spread, "correct": correct,
        "cheat_flags": cheat_flags or [], "error": None,
        "seconds": {"run": 1.0, "queue": 0.0}, "seed": 1,
    }


class ExperimentTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run-20261113-220000-abcd"
        self.tree = ExperimentTree.open(self.run_dir, "tab-demo")
        self.tree.create_root(score(0.0), "lib-0")

    def enqueue(self, parent: str = "n0000", origin: str = "plan") -> str:
        node = self.tree.enqueue(
            parent=parent, hypothesis="Try encoding", rationale="High cardinality",
            origin=origin, cites=["exp-0001"],
            estimate={"gain": 0.1, "cost": 0.2, "seconds": 30},
            priority=0.5, library_version="lib-0",
        )
        return node["node_id"]

    def test_root_snapshot_and_replay(self) -> None:
        node_id = self.enqueue()
        self.tree.start_attempt(node_id)
        self.tree.start_attempt(node_id)
        self.tree.succeed(node_id, score(0.04))
        result = self.tree.judge(node_id, score(0.04))
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(self.tree.best_node()["node_id"], node_id)
        lines = self.tree.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 6)
        self.assertTrue(all(json.loads(line)["schema"] == "node/1" for line in lines))
        self.assertEqual(ExperimentTree.open(self.run_dir, "tab-demo").get(node_id), result)
        self.assertTrue((self.tree.task_dir / result["code_dir"]).is_dir())

    def test_reject_noise_cheat_and_incorrect_results(self) -> None:
        for result in (score(0.005), score(0.5, cheat_flags=["network"]), score(0.5, correct=False)):
            node_id = self.enqueue()
            self.tree.start_attempt(node_id)
            self.tree.succeed(node_id)
            self.assertEqual(self.tree.judge(node_id, result)["status"], "rejected")
        self.assertEqual(self.tree.best_node()["node_id"], "n0000")

    def test_failure_is_persisted_and_invalid_transitions_rejected(self) -> None:
        node_id = self.enqueue()
        self.tree.start_attempt(node_id)
        failed = self.tree.fail(node_id, "timeout", "Too slow")
        self.assertEqual(failed["failure"], {"kind": "timeout", "summary": "Too slow"})
        self.assertEqual(ExperimentTree.open(self.run_dir, "tab-demo").get(node_id), failed)
        with self.assertRaises(TreeError):
            self.tree.start_attempt(node_id)
        with self.assertRaises(TreeError):
            self.tree.patch(node_id, parent="n0000")

    def test_holdout_score_never_enters_tree(self) -> None:
        node_id = self.enqueue()
        self.tree.start_attempt(node_id)
        self.tree.succeed(node_id)
        result = score(0.3)
        result["split"] = "holdout"
        with self.assertRaises(TreeError):
            self.tree.judge(node_id, result)

    def test_unknown_schema_fails_closed(self) -> None:
        with self.tree.path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema":"node/2"}\n')
        with self.assertRaises(TreeError):
            ExperimentTree.open(self.run_dir, "tab-demo")


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = Scheduler(SETTINGS, CAPS, budget=10.0, wall_seconds=100.0)

    def test_hard_gates_and_reserve(self) -> None:
        self.assertEqual(self.scheduler.spend_cap, 9.0)
        self.assertFalse(self.scheduler.can_expand(spent=8, elapsed_wall=20, estimated_cost=1.1))
        self.assertTrue(self.scheduler.can_expand(spent=8, elapsed_wall=20, estimated_cost=1.0))
        self.assertEqual(self.scheduler.stop_reason(spent=9, elapsed_wall=20, best_score=0.5), "budget_cap")
        self.assertEqual(self.scheduler.stop_reason(spent=8, elapsed_wall=100, best_score=0.5), "time")
        self.assertEqual(self.scheduler.stop_reason(spent=8, elapsed_wall=20, best_score=1.0), "ceiling")
        with self.assertRaises(ScheduleError):
            Scheduler(SETTINGS, CAPS, budget=0, wall_seconds=100)

    def test_review_triggers(self) -> None:
        self.assertEqual(self.scheduler.review_trigger(non_improving_nodes=5, spent=1, best_score=0.1), "plateau")
        self.assertEqual(self.scheduler.review_trigger(
            non_improving_nodes=2, spent=5, best_score=0.2,
            expected_score_at_spend=0.6,
        ), "behind_schedule")
        self.assertIsNone(self.scheduler.review_trigger(
            non_improving_nodes=2, spent=4.9, best_score=0.2,
            expected_score_at_spend=0.6,
        ))

    def test_rank_history_and_select_exploration(self) -> None:
        estimate = {"gain": 0.1, "cost": 0.2}
        self.assertAlmostEqual(Scheduler.priority_for(estimate), 0.5)
        adjusted = Scheduler.priority_for(
            estimate,
            cited_effects=[{"uses": 4, "mean_gain": 0.02, "mean_cost": 0.4}],
            past_predictions=[{"gain": 0.1, "actual_gain": 0.02}],
        )
        self.assertLess(adjusted, 0.5)
        pending = [
            {"node_id": "n0001", "status": "queued", "parent": "n0000", "origin": "plan", "priority": 2},
            {"node_id": "n0002", "status": "queued", "parent": "n0000", "origin": "explore", "priority": 1},
        ]
        self.assertEqual(self.scheduler.choose_next(pending, expanded_count=0, best_node_id="n0000")["node_id"], "n0001")
        self.assertEqual(self.scheduler.choose_next(pending, expanded_count=3, best_node_id="n0000")["node_id"], "n0002")
        accepted = [
            {"node_id": "n0000", "status": "accepted", "score": {"dev": 0}},
            {"node_id": "n0003", "status": "accepted", "score": {"dev": 0.2}},
        ]
        self.assertEqual(self.scheduler.choose_parent(accepted, expanded_count=0), "n0003")
        self.assertEqual(self.scheduler.choose_parent(accepted, expanded_count=3), "n0000")


if __name__ == "__main__":
    unittest.main()
