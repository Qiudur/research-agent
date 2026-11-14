from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.run_token_pilot import (BUDGET_TOKENS, TASKS, WALL_SECONDS,
                                     prepare_task_copies)


PROJECT = Path(__file__).resolve().parents[1]


class TokenPilotTests(unittest.TestCase):
    def test_copies_practice_tasks_without_changing_official_budget(self):
        original = {task_id: (PROJECT / "tasks" / task_id / "task.json").read_bytes()
                    for task_id in TASKS}
        with TemporaryDirectory() as temp:
            campaign = Path(temp) / "campaign"
            copied = prepare_task_copies(PROJECT, campaign, TASKS)
            for task_id in TASKS:
                task = json.loads((copied / task_id / "task.json").read_text())
                self.assertEqual(task["budget"],
                                 {"money": BUDGET_TOKENS, "wall_seconds": WALL_SECONDS})
                self.assertEqual((PROJECT / "tasks" / task_id / "task.json").read_bytes(),
                                 original[task_id])
            inputs = json.loads((campaign / "pilot_inputs.json").read_text())
            self.assertEqual(inputs["label"], "DEV TOKEN MEASUREMENT - NOT MONEY")
            self.assertEqual(len(json.loads((copied / "bank.json").read_text())["tasks"]), 2)


if __name__ == "__main__":
    unittest.main()
