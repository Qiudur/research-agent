from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.cli import baseline_solutions, formal_preflight, main


SHA = "a" * 40


class EvaluationCliTests(unittest.TestCase):
    def test_qwen_token_study_requires_authorized_gateway(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.setup_project(root)
            (root / "config").mkdir()
            prices = root / "config/prices.json"
            prices.write_text(json.dumps({
                "schema": "prices/1", "currency": "TOKENS", "unit_tokens": 1,
                "models": {"qwen3.5-27b": {key: 1 for key in
                           ("input", "cached_input", "output", "reasoning")}},
            }), encoding="utf-8")
            args = dict(project_root=root, model="qwen3.5-27b", prices_path=prices,
                        code_commit=SHA, default_config=root / "config/missing.json")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only",
                                      "OPENAI_BASE_URL": "https://wrong.invalid/v1"}):
                result = formal_preflight(**args)
                self.assertTrue(any("OPENAI_BASE_URL" in item for item in result["blockers"]))
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only",
                                      "OPENAI_BASE_URL": "https://ai-gateway.ailab.jiuan.com/v1"}):
                result = formal_preflight(**args)
                self.assertFalse(any("OPENAI_BASE_URL" in item for item in result["blockers"]))

    def test_formal_preflight_marks_valid_token_accounting_as_internal(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.setup_project(root)
            (root / "config").mkdir()
            path = root / "config/prices.json"
            prices = {"schema": "prices/1", "currency": "TOKENS", "unit_tokens": 1,
                      "test_only": False, "models": {"real-model": {
                          "input": 1, "cached_input": 1, "output": 1, "reasoning": 1}}}
            cases = (("TOKENS", False, 1, 1, "internal_token_study"),
                     ("TOKENS", False, 1000, 1, None),
                     ("TOKENS", False, 1, 2, None),
                     ("TOKENS", True, 1, 1, None),
                     ("CNY", False, 1, 1, None))
            for currency, test_only, unit, output_rate, expected in cases:
                prices.update(currency=currency, test_only=test_only, unit_tokens=unit)
                prices["models"]["real-model"]["output"] = output_rate
                path.write_text(json.dumps(prices), encoding="utf-8")
                report = formal_preflight(project_root=root, model="real-model", prices_path=path,
                                          code_commit=SHA, default_config=root / "config/missing.json")
                self.assertEqual(report["accounting_scope"], expected)
                if expected:
                    self.assertIn("非现金节省", report["accounting_note"])
                else:
                    self.assertIsNone(report["accounting_note"])
                if currency == "TOKENS" and not test_only and expected is None:
                    self.assertTrue(any("unit_tokens=1" in blocker for blocker in report["blockers"]))

    def setup_project(self, root: Path):
        rows = []
        for task_id, tier in (("spd-practice-a", "practice"),
                              ("spd-validation-a", "validation"),
                              ("spd-exam-a", "exam")):
            rows.append({"task_id": task_id, "tier": tier})
            path = root / "tasks" / task_id / "workspace/original.py"
            path.parent.mkdir(parents=True)
            path.write_text("def solve(): return 1" + chr(10))
        bank = {"schema": "bank/1", "version": "bank-v1", "tasks": rows}
        (root / "tasks/bank.json").write_text(json.dumps(bank))
        return bank

    def common(self, root: Path):
        return ["--project-root", str(root), "--model", "real-model",
                "--prices", "config/prices.json", "--code-commit", SHA]

    def test_baseline_discovery_limits_auto_research_to_practice(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            bank = self.setup_project(root)
            found = baseline_solutions(root, bank, practice_only=True)
            self.assertEqual(list(found), ["spd-practice-a"])
            self.assertEqual(len(baseline_solutions(root, bank, practice_only=False)), 3)
            (root / "tasks/spd-validation-a/workspace/original.py").unlink()
            self.assertEqual(list(baseline_solutions(root, bank, practice_only=True)),
                             ["spd-practice-a"])
            with self.assertRaises(FileNotFoundError):
                baseline_solutions(root, bank, practice_only=False)

    def test_preflight_requires_explicit_formal_inputs_and_exits_one_when_blocked(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.setup_project(root)
            with self.assertRaises(SystemExit) as context:
                main(["preflight", *self.common(root)])
            self.assertEqual(context.exception.code, 2)
            output = StringIO()
            with redirect_stdout(output):
                code = main(["preflight", *self.common(root),
                             "--tuned-config", "config/tuned.json"])
            self.assertEqual(code, 1)
            report = json.loads(output.getvalue())
            self.assertEqual(report["schema"], "formal_preflight/1")
            self.assertFalse(report["ready"])
            self.assertTrue(report["blockers"])
            self.assertFalse((root / "batches").exists())

    @patch("evaluation.cli.formal_preflight", return_value={"schema": "formal_preflight/1", "ready": True})
    def test_auto_research_dispatch_has_no_test_overrides(self, readiness):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.setup_project(root)
            fake = {"selection": {"candidate": "auto-01"},
                    "total_research_cost": 3.5, "currency": "CNY"}
            output = StringIO()
            with patch("evaluation.cli.auto_evaluate_candidates", return_value=fake) as execute:
                with redirect_stdout(output):
                    code = main(["auto-research", *self.common(root),
                                 "--study-dir", "batches/auto-1", "--rounds", "2",
                                 "--max-output-tokens", "512"])
            self.assertEqual(code, 0)
            kwargs = execute.call_args.kwargs
            self.assertEqual(kwargs["rounds"], 2)
            self.assertEqual(kwargs["max_output_tokens"], 512)
            self.assertEqual(list(kwargs["baseline_solutions"]), ["spd-practice-a"])
            self.assertNotIn("test_only_budget", kwargs)
            self.assertNotIn("proposal_client", kwargs)
            self.assertEqual(json.loads(output.getvalue())["selection"]["candidate"], "auto-01")

    @patch("evaluation.cli.formal_preflight", return_value={"schema": "formal_preflight/1", "ready": True})
    def test_batch_and_exam_dispatch(self, readiness):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.setup_project(root)
            fake = {"admission": {"claim": "admitted"},
                    "reports": {"practice": [{}] * 12, "validation": [{}] * 6,
                                "exam": [{}] * 12}, "tuned_exam_skipped": False}
            output = StringIO()
            with patch("evaluation.cli.execute_batch", return_value=fake) as execute:
                with redirect_stdout(output):
                    code = main(["batch", *self.common(root),
                                 "--tuned-config", "batches/selected/agent.json",
                                 "--batch-dir", "batches/batch-1", "--exam"])
            self.assertEqual(code, 0)
            kwargs = execute.call_args.kwargs
            self.assertTrue(kwargs["run_exam"])
            self.assertEqual(len(kwargs["baseline_solutions"]), 3)
            self.assertEqual(kwargs["agent_configs"]["naive"], root / "config/naive.json")
            self.assertEqual(kwargs["agent_configs"]["tuned_lib"],
                             root / "batches/selected/agent.json")
            self.assertEqual(json.loads(output.getvalue())["reports"]["exam"], 12)

    def test_figures_dispatch_and_runtime_exit_code(self):
        with patch("evaluation.cli.figures.main", return_value=0) as render:
            code = main(["figures", "--batch-dir", "batches/batch-1",
                         "--auto-study-dir", "batches/auto-1",
                         "--output", "batches/batch-1/study.html"])
        self.assertEqual(code, 0)
        self.assertIn("--auto-study-dir", render.call_args.args[0])
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.setup_project(root)
            error = StringIO()
            with patch("evaluation.cli.formal_preflight", return_value={"ready": True}):
                with patch("evaluation.cli.execute_batch", side_effect=ValueError("exam already attempted")):
                    with redirect_stderr(error):
                        code = main(["batch", *self.common(root),
                                     "--tuned-config", "config/tuned.json",
                                     "--batch-dir", "batches/batch-1"])
            self.assertEqual(code, 2)
            self.assertIn("exam already attempted", error.getvalue())

    def test_cwd_project_root_and_harness_environment_are_scoped(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.setup_project(root)
            (root / "config").mkdir()
            arguments = ["--model", "real-model", "--prices", "config/prices.json",
                         "--code-commit", SHA]
            expected = {"RA_PROJECT_ROOT": str(root), "RA_TASKS_ROOT": str(root / "tasks")}

            def check_environment(**kwargs):
                self.assertEqual(kwargs["project_root"], root)
                self.assertEqual({name: os.environ.get(name) for name in expected}, expected)

            def ready(**kwargs):
                check_environment(**kwargs)
                return {"schema": "formal_preflight/1", "ready": True}

            def research(**kwargs):
                check_environment(**kwargs)
                self.assertEqual(kwargs["harness"]._tasks_root(), root / "tasks")
                return {"selection": {"candidate": "auto-01"},
                        "total_research_cost": 1.0, "currency": "CNY"}

            def batch(**kwargs):
                check_environment(**kwargs)
                self.assertEqual(kwargs["harness"]._tasks_root(), root / "tasks")
                return {"admission": {}, "reports": {"practice": [], "validation": [], "exam": []},
                        "tuned_exam_skipped": True}

            old_cwd = Path.cwd()
            with patch.dict(os.environ, {"RA_PROJECT_ROOT": "old-project",
                                      "RA_TASKS_ROOT": "old-tasks"}, clear=False):
                try:
                    os.chdir(root)
                    with patch("evaluation.cli.formal_preflight", side_effect=ready):
                        with redirect_stdout(StringIO()):
                            self.assertEqual(main(["preflight", *arguments,
                                                   "--tuned-config", "config/tuned.json"]), 0)
                        with patch("evaluation.cli.auto_evaluate_candidates", side_effect=research):
                            with redirect_stdout(StringIO()):
                                self.assertEqual(main(["auto-research", *arguments,
                                                       "--study-dir", "batches/auto-1"]), 0)
                        with patch("evaluation.cli.execute_batch", side_effect=batch):
                            with redirect_stdout(StringIO()):
                                self.assertEqual(main(["batch", *arguments,
                                                       "--tuned-config", "config/tuned.json",
                                                       "--batch-dir", "batches/batch-1"]), 0)
                        with patch("evaluation.cli.execute_batch", side_effect=ValueError("expected failure")):
                            with redirect_stderr(StringIO()):
                                self.assertEqual(main(["batch", *arguments,
                                                       "--tuned-config", "config/tuned.json",
                                                       "--batch-dir", "batches/batch-1"]), 2)
                    self.assertEqual(os.environ["RA_PROJECT_ROOT"], "old-project")
                    self.assertEqual(os.environ["RA_TASKS_ROOT"], "old-tasks")
                finally:
                    os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
