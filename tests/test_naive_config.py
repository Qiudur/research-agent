from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from research_agent.runner import TaskRunner


ROOT = Path(__file__).resolve().parents[1]


class _Library:
    version = "lib-0"

    def query(self, *args, **kwargs):
        return []

    def brief(self, entries):
        return ""


class _Tree:
    def all_nodes(self):
        return [{"node_id": f"n{i:04d}", "hypothesis": str(i), "status": "accepted",
                 "score": {"dev": float(i)}, "cost": 0.0, "failure": None} for i in range(12)]

    def best_node(self):
        return {"node_id": "n0011", "score": {"dev": 11.0}}


class NaiveConfigTests(unittest.TestCase):
    def test_generalist_uses_full_history_and_high_effort(self):
        config = json.loads((ROOT / "config/naive.json").read_text())
        self.assertEqual(set(config["effort"].values()), {"high"})
        self.assertEqual(len(set(config["prompts"].values())), 1)
        self.assertEqual(config["history_mode"], "full")
        runner = object.__new__(TaskRunner)
        runner.config = config
        runner.context = {"scenario": "speedup", "features": []}
        runner.library = _Library()
        runner.tree = _Tree()
        runner.scheduler = SimpleNamespace(budget=5.0, budget_remaining=lambda spent: 4.0)
        runner.allocations = {}
        runner.last_review_note = ""
        runner._spent = lambda: 1.0
        runner._emit = lambda *args, **kwargs: None
        self.assertEqual(len(runner._planner_payload()["recent_experiments"]), 12)
        runner.config = {**config, "history_mode": "compact"}
        self.assertEqual(len(runner._planner_payload()["recent_experiments"]), 8)


if __name__ == "__main__":
    unittest.main()
