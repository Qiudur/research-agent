from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.report import run_report, task_report


class FormalTokenReportTests(unittest.TestCase):
    def report_fixture(self, root: Path, *, cost: int, budget: int):
        run = root / "runs/run-20261114-120000-abcd"
        task_id = "spd-example"
        task_dir = run / "tasks" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "holdout_score.json").write_text(json.dumps({
            "schema": "score/1", "split": "holdout", "score": 0.1,
        }), encoding="utf-8")
        events = [
            {"kind": "task_start", "budget": budget},
            {"kind": "llm_call", "role": "planner", "cost": cost,
             "tokens": {"input": cost, "cached_input": 0, "output": 0, "reasoning": 0},
             "seq": 2},
            {"kind": "stop", "best_node": "n0000", "dev_score": 0.0, "seq": 3},
            {"kind": "task_end", "best_node": "n0000", "holdout_score": 0.1,
             "total_cost": cost, "seconds": {"llm": 1, "run": 0,
                                              "queue": 0, "wall": 1},
             "quote_error": {"cost": None, "seconds": None, "score": None},
             "seq": 4},
        ]
        ledger = SimpleNamespace(
            manifest={"phase": "exam", "model": "qwen3.5-27b",
                      "library": {"root": f"runs/{run.name}/library"}},
            prices={"currency": "TOKENS", "unit_tokens": 1,
                    "models": {"qwen3.5-27b": {key: 1 for key in
                               ("input", "cached_input", "output", "reasoning")}}},
            path=task_dir / "ledger.jsonl", _events=lambda: events,
        )
        node = {"node_id": "n0000", "status": "accepted", "library_version": "lib-0"}
        return run, task_id, ledger, node

    def test_over_budget_real_tokens_are_reported_not_fatal(self):
        with TemporaryDirectory() as temp:
            run, task_id, ledger, node = self.report_fixture(Path(temp),
                                                              cost=40_001, budget=40_000)
            with patch("evaluation.report.Ledger.open", return_value=ledger):
                with patch("evaluation.report._latest_nodes", return_value={"n0000": node}):
                    with patch("evaluation.report._verify_replay_context",
                               return_value={"max_output_tokens": 2048}) as verify:
                        report = task_report(run, task_id)
                        # The sealed replay is still verified: only the budget rule relaxed.
                        verify.assert_called_once()
            self.assertTrue(report["budget_violation"])
            self.assertEqual(report["budget_excess"], 1.0)
            self.assertEqual(report["budget"], 40_000.0)
            self.assertEqual(report["total_cost"], 40_001)

    def test_within_budget_is_not_a_violation(self):
        with TemporaryDirectory() as temp:
            for cost in (12_000, 40_000):
                run, task_id, ledger, node = self.report_fixture(Path(temp) / str(cost),
                                                                  cost=cost, budget=40_000)
                with patch("evaluation.report.Ledger.open", return_value=ledger):
                    with patch("evaluation.report._latest_nodes", return_value={"n0000": node}):
                        with patch("evaluation.report._verify_replay_context",
                                   return_value={"max_output_tokens": 2048}):
                            report = task_report(run, task_id)
                self.assertFalse(report["budget_violation"])
                self.assertEqual(report["budget_excess"], 0.0)

    def test_missing_budget_or_unit_mismatch_still_blocks_formal_report(self):
        with TemporaryDirectory() as temp:
            run, task_id, ledger, node = self.report_fixture(Path(temp), cost=100, budget=40_000)
            ledger._events()[0]["budget"] = None
            with patch("evaluation.report.Ledger.open", return_value=ledger):
                with patch("evaluation.report._latest_nodes", return_value={"n0000": node}):
                    with patch("evaluation.report._verify_replay_context") as verify:
                        with self.assertRaisesRegex(ValueError, "invalid budget"):
                            task_report(run, task_id)
                        verify.assert_not_called()

    def test_valid_budget_still_requires_a_sealed_replay(self):
        with TemporaryDirectory() as temp:
            run, task_id, ledger, node = self.report_fixture(Path(temp),
                                                              cost=12_000, budget=40_000)
            with patch("evaluation.report.Ledger.open", return_value=ledger):
                with patch("evaluation.report._latest_nodes", return_value={"n0000": node}):
                    with patch("evaluation.report._verify_replay_context",
                               return_value={"max_output_tokens": 2048}) as verify:
                        report = task_report(run, task_id)
                        self.assertEqual(report["total_cost"], 12_000)
                        verify.assert_called_once()
                    with patch("evaluation.report._verify_replay_context",
                               side_effect=ValueError("Replay source changed")):
                        with self.assertRaisesRegex(ValueError, "Replay source changed"):
                            task_report(run, task_id)

    def test_unknown_call_usage_marks_report_cost_incomplete(self):
        with TemporaryDirectory() as temp:
            run, task_id, ledger, node = self.report_fixture(Path(temp), cost=12_000,
                                                              budget=40_000)
            ledger._events()[1]["usage_status"] = "unknown"
            with patch("evaluation.report.Ledger.open", return_value=ledger):
                with patch("evaluation.report._latest_nodes", return_value={"n0000": node}):
                    with patch("evaluation.report._verify_replay_context",
                               return_value={"max_output_tokens": 2048}):
                        task = task_report(run, task_id)
            self.assertFalse(task["cost_complete"])
            self.assertEqual(task["unknown_usage_call_seqs"], [2])
            run_ledger = SimpleNamespace(
                manifest={"phase": "self_research", "config": "default_lib", "rep": 1,
                          "model": "qwen3.5-27b", "tasks": [task_id]},
                prices=ledger.prices,
                _events=lambda: [{"kind": "run_end", "seq": 3, "total_cost": 12_000}],
                path=run / "ledger.jsonl",
            )
            with patch("evaluation.report.Ledger.open", return_value=run_ledger):
                with patch("evaluation.report.task_report", return_value=task):
                    report = run_report(run)
            self.assertFalse(report["cost_complete"])
            self.assertFalse(report["exact_token_accounting"])


if __name__ == "__main__":
    unittest.main()
