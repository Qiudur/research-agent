from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from library import Library, LibraryError, LibraryFrozen
from library.skills import (demote_skill, load_skill_verified, promote_candidates,
                            promote_to_skill, verified_wins)
from research_agent.tree import ExperimentTree


def score(value: float) -> dict:
    return {"schema": "score/1", "split": "dev", "ok": True, "score": value,
            "raw": value, "spread": 0.01, "correct": True, "cheat_flags": []}


class SkillLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name) / "runs"
        self.lib = Library.open(self.runs / "run-master" / "library")
        [self.exp_id] = self.lib.write_experiences([{
            "scope": "domain", "scenario": "tabular",
            "triggers": ["high_cardinality_categorical"],
            "conditions": "more than 100 categories",
            "action": "try target encoding", "negative": False,
        }], source={"run_id": "run-master", "task_id": "tab-origin", "node_ids": ["n0001"]})

    def use(self, run_id: str, task_id: str, *, candidate_score: float = 0.5,
            claimed_win: bool = True, exp_id: str | None = None) -> None:
        exp_id = exp_id or self.exp_id
        tree = ExperimentTree.open(self.runs / run_id, task_id)
        tree.create_root(score(0.2), self.lib.version)
        node = tree.enqueue(parent="n0000", hypothesis="Target encode", rationale="test",
                            origin="plan", cites=[exp_id],
                            estimate={"gain": 0.2, "cost": 0.1, "seconds": 1.0},
                            priority=2.0, library_version=self.lib.version)
        tree.start_attempt(node["node_id"])
        tree.succeed(node["node_id"], score(candidate_score))
        tree.judge(node["node_id"], score(candidate_score))
        self.lib.record_use(exp_id, run_id=run_id, task_id=task_id,
                            node_id=node["node_id"], accepted=claimed_win,
                            gain=candidate_score - 0.2, cost=0.1)

    def ready(self) -> None:
        self.use("run-one", "tab-one")
        self.use("run-two", "tab-two")

    def test_requires_two_distinct_verified_task_wins(self) -> None:
        self.assertEqual([], promote_candidates(self.lib))
        self.use("run-one", "tab-one")
        self.use("run-two", "tab-one")
        self.assertEqual(0.7, round(self.lib.get(self.exp_id)["credibility"], 1))
        self.assertEqual([], promote_candidates(self.lib))
        with self.assertRaises(LibraryError):
            promote_to_skill(self.lib, [self.exp_id], "# Steps", "Brief")
        self.assertFalse((self.lib.root / "skills/skl-0001.md").exists())

    def test_rejected_node_cannot_be_laundered_as_win(self) -> None:
        self.use("run-one", "tab-one")
        self.use("run-two", "tab-two", candidate_score=0.2, claimed_win=True)
        self.assertEqual(["tab-one"], [item["task_id"] for item in verified_wins(self.lib, self.exp_id)])
        self.assertEqual([], promote_candidates(self.lib))

    def test_promotion_is_versioned_and_body_is_loaded_on_demand(self) -> None:
        self.ready()
        self.assertEqual([[self.exp_id]], promote_candidates(self.lib))
        before = self.lib.version
        body = "# Target encoding\n1. Check leakage.\n2. Fit folds.\n"
        skill_id = promote_to_skill(self.lib, [self.exp_id], body, "Target encode with folds")
        entry = self.lib.get(skill_id)
        self.assertEqual("skl-0001", skill_id)
        self.assertEqual("skill", entry["kind"])
        self.assertEqual([self.exp_id], entry["from_experiences"])
        self.assertEqual({"tab-one", "tab-two"}, {item["task_id"] for item in entry["verified_wins"]})
        self.assertEqual(body, load_skill_verified(self.lib, skill_id))
        brief = self.lib.brief(self.lib.query("tabular", ["high_cardinality_categorical"], kinds=("skill",)))
        self.assertIn("Target encode with folds", brief)
        self.assertNotIn("Check leakage", brief)
        self.assertEqual([], [item for item in Library.open(self.lib.root, version=before).entries if item["kind"] == "skill"])
        change = self.lib.changelog()[-1]
        self.assertEqual("promote", change["op"])
        self.assertEqual({"tab-one", "tab-two"}, {item["task_id"] for item in change["evidence"]})
        self.assertEqual([], promote_candidates(self.lib))

    def test_body_tamper_and_frozen_library_are_rejected(self) -> None:
        self.ready()
        frozen = Library.open(self.lib.root, frozen=True)
        with self.assertRaises(LibraryFrozen):
            promote_to_skill(frozen, [self.exp_id], "# Steps", "Brief")
        skill_id = promote_to_skill(self.lib, [self.exp_id], "# Steps", "Brief")
        (self.lib.root / self.lib.get(skill_id)["body_path"]).write_text("changed", encoding="utf-8")
        with self.assertRaises(LibraryError):
            load_skill_verified(self.lib, skill_id)

    def test_manual_demotion_then_loss_based_retirement(self) -> None:
        self.ready()
        skill_id = promote_to_skill(self.lib, [self.exp_id], "# Steps", "Brief")
        demote_skill(self.lib, skill_id, reason="Failed on validation",
                     evidence=[{"run_id": "run-validation", "task_id": "tab-new", "node_ids": ["n0001"]}])
        self.assertEqual("demoted", self.lib.get(skill_id)["status"])
        self.assertEqual([], self.lib.query("tabular", ["high_cardinality_categorical"], kinds=("skill",)))
        with self.assertRaises(LibraryError):
            load_skill_verified(self.lib, skill_id)
        for index in range(3):
            self.lib.record_use(skill_id, run_id="run-validation", task_id="tab-new",
                                node_id=f"n{index+1:04d}", accepted=False, gain=0, cost=0.1)
        self.assertEqual("retired", self.lib.get(skill_id)["status"])
        self.assertEqual("retire", self.lib.changelog()[-1]["op"])

    def test_skill_cap_retires_lowest_credibility(self) -> None:
        self.ready()
        self.lib.rules["max_skills"] = 1
        first = promote_to_skill(self.lib, [self.exp_id], "# First", "First skill")
        [second_exp] = self.lib.write_experiences([{
            "scope": "domain", "scenario": "tabular",
            "triggers": ["high_cardinality_categorical"],
            "conditions": "more than 100 categories",
            "action": "try frequency encoding", "negative": False,
        }], source={"run_id": "run-master", "task_id": "tab-origin", "node_ids": ["n0002"]})
        self.use("run-three", "tab-three", exp_id=second_exp)
        self.use("run-four", "tab-four", exp_id=second_exp)
        second = promote_to_skill(self.lib, [second_exp], "# Second", "Second skill")
        self.assertEqual("retired", self.lib.get(first)["status"])
        self.assertEqual("active", self.lib.get(second)["status"])
        self.assertEqual(1, len(self.lib.query("tabular", ["high_cardinality_categorical"], kinds=("skill",))))


if __name__ == "__main__":
    unittest.main()
