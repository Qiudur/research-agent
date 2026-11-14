from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import harness
from scripts.run_token_self_research import (PRICE_NAME, PROJECT, TASKS,
                                             _project_override, prepare_project)


class TokenSelfResearchPilotTests(unittest.TestCase):
    def test_prepare_copies_only_practice_tasks_with_exact_token_units(self):
        with TemporaryDirectory() as temp:
            campaign = Path(temp) / "campaign"
            project = prepare_project(PROJECT, campaign)
            bank = json.loads((project / "tasks/bank.json").read_text())
            prices = json.loads((project / "config" / PRICE_NAME).read_text())
            protocol = json.loads((campaign / "protocol.json").read_text())
            self.assertEqual({entry["task_id"] for entry in bank["tasks"]}, set(TASKS))
            self.assertTrue(all(entry["tier"] == "practice" for entry in bank["tasks"]))
            self.assertEqual(prices["currency"], "TEST")
            self.assertTrue(prices["test_only"])
            self.assertEqual(prices["unit_tokens"], 1)
            self.assertEqual(set(prices["models"][protocol["model"]].values()), {1})
            self.assertEqual(protocol["budget_tokens_per_task"], 40_000)
            self.assertEqual(protocol["reserve_tokens_per_call"], 12_000)
            with _project_override(project):
                self.assertEqual({harness.load_task(task_id)["scenario"] for task_id in TASKS},
                                 {"speedup", "tabular"})
            for task_id in TASKS:
                original = json.loads((PROJECT / "tasks" / task_id / "task.json").read_text())
                self.assertIsNone(original["budget"]["money"])
                copied = json.loads((project / "tasks" / task_id / "task.json").read_text())
                self.assertEqual(protocol["wall_seconds_per_task"][task_id],
                                 copied["budget"]["wall_seconds"])

    def test_prepare_transfer_pair_keeps_original_task_budgets(self):
        task_ids = ("spd-distance-matrix", "tab-adult-income")
        with TemporaryDirectory() as temp:
            campaign = Path(temp) / "campaign"
            project = prepare_project(PROJECT, campaign, task_ids=task_ids,
                                      proposal_rounds=0)
            protocol = json.loads((campaign / "protocol.json").read_text())
            bank = json.loads((project / "tasks/bank.json").read_text())
            self.assertEqual(protocol["tasks"], list(task_ids))
            self.assertEqual(protocol["proposal_rounds"], 0)
            self.assertEqual({row["task_id"] for row in bank["tasks"]}, set(task_ids))
            for task_id in task_ids:
                original = json.loads((PROJECT / "tasks" / task_id / "task.json").read_text())
                copied = json.loads((project / "tasks" / task_id / "task.json").read_text())
                self.assertEqual(copied["budget"], original["budget"])
                self.assertEqual(protocol["wall_seconds_per_task"][task_id], 1800)


if __name__ == "__main__":
    unittest.main()
