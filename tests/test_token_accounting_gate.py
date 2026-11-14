from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.batch import execute_batch
from evaluation.report import run_report
from evaluation.self_research import auto_evaluate_candidates, evaluate_candidates
from token_accounting import is_exact_token_accounting, require_exact_token_accounting


class TokenAccountingGateTests(unittest.TestCase):
    def test_run_report_derives_exact_flag_from_sealed_ledger_prices(self):
        with TemporaryDirectory() as temp:
            run_dir = Path(temp) / "runs/run-example"
            run_dir.mkdir(parents=True)
            manifest = {"phase": "exam", "config": "default", "rep": 1,
                        "tasks": ["spd-example"], "model": "model"}
            for currency, unit, output_rate, expected in (("TOKENS", 1, 1, True),
                                                          ("TOKENS", 1, 2, False),
                                                          ("CNY", 1, 1, False)):
                prices = {"currency": currency, "unit_tokens": unit,
                          "models": {"model": {"input": 1, "cached_input": 1,
                                               "output": output_rate, "reasoning": 1}}}
                ledger = SimpleNamespace(manifest=manifest, prices=prices, path=run_dir / "ledger.jsonl",
                                         _events=lambda: [{"kind": "run_end", "total_cost": 0, "seq": 1}])
                with patch("evaluation.report.Ledger.open", return_value=ledger):
                    with patch("evaluation.report.task_report", return_value={"total_cost": 0}):
                        report = run_report(run_dir)
                self.assertIs(report["exact_token_accounting"], expected)

    def test_exact_accounting_rejects_weighted_or_boolean_rates(self):
        prices = {"currency": "TOKENS", "unit_tokens": 1,
                  "models": {"model": {"input": 1, "cached_input": 1,
                                       "output": 1.0, "reasoning": 1}}}
        self.assertTrue(is_exact_token_accounting(prices, "model"))
        require_exact_token_accounting(prices, "model")
        for unit, rate in ((1000, 1), (1, 2), (1, True)):
            wrong = json.loads(json.dumps(prices))
            wrong["unit_tokens"] = unit
            wrong["models"]["model"]["output"] = rate
            self.assertFalse(is_exact_token_accounting(wrong, "model"))
            with self.assertRaisesRegex(ValueError, "unit_tokens=1"):
                require_exact_token_accounting(wrong, "model")
        prices["currency"] = "CNY"
        self.assertFalse(is_exact_token_accounting(prices, "model"))
        require_exact_token_accounting(prices, "model")

    def test_formal_batch_and_self_research_reject_weighted_tokens_before_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tasks").mkdir()
            (root / "config").mkdir()
            (root / "tasks/bank.json").write_text(json.dumps({
                "schema": "bank/1", "version": "bank-v1", "tasks": [],
            }), encoding="utf-8")
            baseline = root / "tasks/spd-practice-a/workspace/original.py"
            baseline.parent.mkdir(parents=True)
            baseline.write_text("pass" + chr(10), encoding="utf-8")
            config = root / "config/agent.json"
            config.write_text("{}", encoding="utf-8")
            prices_path = root / "config/prices.json"
            for unit, output_rate in ((1000, 1), (1, 2)):
                prices_path.write_text(json.dumps({
                    "schema": "prices/1", "currency": "TOKENS", "test_only": False,
                    "unit_tokens": unit, "models": {"model": {
                        "input": 1, "cached_input": 1, "output": output_rate, "reasoning": 1}},
                }), encoding="utf-8")
                with patch("evaluation.batch._bank_tasks", return_value={
                        "practice": ["spd-practice-a"], "validation": [], "exam": []}):
                    with self.assertRaisesRegex(ValueError, "unit_tokens=1"):
                        execute_batch(project_root=root, batch_dir=root / "batches/bad",
                                      harness=object(), baseline_solutions={"spd-practice-a": baseline},
                                      agent_configs={}, prices_path=prices_path, model="model",
                                      code_commit="a" * 40, run_validation=False)
                with patch("evaluation.self_research._practice_tasks", return_value=["spd-practice-a"]):
                    with patch("evaluation.self_research._candidate_files", return_value=({}, {})):
                        with self.assertRaisesRegex(ValueError, "unit_tokens=1"):
                            evaluate_candidates(project_root=root, study_dir=root / "batches/research-bad",
                                                harness=object(), candidates={},
                                                baseline_solutions={"spd-practice-a": baseline},
                                                prices_path=prices_path, model="model", code_commit="a" * 40)
                with patch("evaluation.self_research._candidate_files", return_value=({}, {})):
                    with self.assertRaisesRegex(ValueError, "unit_tokens=1"):
                        auto_evaluate_candidates(project_root=root, study_dir=root / "batches/auto-bad",
                                                 harness=object(), default_config=config,
                                                 baseline_solutions={"spd-practice-a": baseline},
                                                 prices_path=prices_path, model="model", code_commit="a" * 40)
                self.assertFalse((root / "batches").exists())


if __name__ == "__main__":
    unittest.main()
