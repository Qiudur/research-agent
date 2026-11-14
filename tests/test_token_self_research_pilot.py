from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import harness
from scripts import run_token_self_research as pilot
from scripts.run_token_self_research import (PRICE_NAME, PROJECT, TASKS,
                                             _project_override, prepare_project)


class TokenSelfResearchPilotTests(unittest.TestCase):
    def _mocked_default_summary(self, root: Path, *, fail: bool) -> dict:
        def prepare(_source, campaign, **_kwargs):
            project = campaign / "project"
            project.mkdir(parents=True)
            pilot._write(campaign / "protocol.json", {"label": "TEST ONLY"})
            return project

        fake_result = {"ranking": [{"candidate": "default"}],
                       "reports": [{"total_cost": 12.0}]}
        outcome = RuntimeError("evaluation failed") if fail else fake_result
        with (patch.object(pilot, "PROJECT", root),
              patch.object(pilot, "prepare_project", side_effect=prepare),
              patch.object(pilot.subprocess, "check_output", side_effect=["", "a" * 40]),
              patch.object(pilot, "evaluate_candidates", side_effect=[outcome]),
              patch.object(pilot, "_utc_now", side_effect=[
                  "2026-11-14T01:00:00.000Z", "2026-11-14T01:00:01.500Z"]),
              patch.object(pilot, "monotonic", side_effect=[100.0, 101.5]),
              patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret-never-record"}),
              patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **_: object())})):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    pilot.main(["--default-only"])
            else:
                self.assertEqual(pilot.main(["--default-only"]), 0)
        campaigns = list((root / "runs").iterdir())
        self.assertEqual(len(campaigns), 1)
        raw = (campaigns[0] / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn("test-secret-never-record", raw)
        return json.loads(raw)

    def test_parallel_default_summary_records_successful_wall_time(self):
        with TemporaryDirectory() as temp:
            summary = self._mocked_default_summary(Path(temp), fail=False)
            self.assertEqual(summary["run_count"], 1)
            self.assertEqual(summary["total_task_tokens"], 12.0)
            self.assertEqual(summary["started_utc"], "2026-11-14T01:00:00.000Z")
            self.assertEqual(summary["ended_utc"], "2026-11-14T01:00:01.500Z")
            self.assertEqual(summary["elapsed_seconds"], 1.5)

    def test_parallel_default_summary_records_failed_wall_time(self):
        with TemporaryDirectory() as temp:
            summary = self._mocked_default_summary(Path(temp), fail=True)
            self.assertEqual(summary["error"], "evaluation failed")
            self.assertEqual(summary["started_utc"], "2026-11-14T01:00:00.000Z")
            self.assertEqual(summary["ended_utc"], "2026-11-14T01:00:01.500Z")
            self.assertEqual(summary["elapsed_seconds"], 1.5)

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
