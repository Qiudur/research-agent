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

    def test_optional_german_credit_is_practice_only(self):
        with TemporaryDirectory() as temp:
            copied = prepare_task_copies(PROJECT, Path(temp) / "campaign",
                                         ("tab-german-credit",))
            bank = json.loads((copied / "bank.json").read_text())
            self.assertEqual([row["task_id"] for row in bank["tasks"]],
                             ["tab-german-credit"])

    def test_naive_control_preserves_one_generic_prompt_and_high_effort(self):
        path = PROJECT / "config/naive_token_pilot.json"
        config = json.loads(path.read_text())
        self.assertEqual(set(config["effort"].values()), {"high"})
        self.assertEqual(set(config["prompts"].values()),
                         {"config/prompts/naive_token_pilot.md"})
        self.assertEqual(config["query_k"], 0)
        with TemporaryDirectory() as temp:
            campaign = Path(temp) / "campaign"
            prepare_task_copies(PROJECT, campaign, ("spd-membership-filter",), path)
            inputs = json.loads((campaign / "pilot_inputs.json").read_text())
            self.assertEqual(inputs["agent_config_path"],
                             "config/naive_token_pilot.json")


if __name__ == "__main__":
    unittest.main()
