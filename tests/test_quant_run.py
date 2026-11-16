"""End-to-end dev research: real sandbox, no key, no sealed scoring."""

from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ledger import Ledger, Usage
from quant_research.data import load_dataset
from quant_research.demo_data import write_demo_dataset
from quant_research.finalize import finalize_holdout
from quant_research.finalize_cli import harness_from_run
from research_agent.model import ModelCallError, UnknownUsageError
from quant_research.run import ROOT, QuantRunFailed, ScriptedRoles, main, run_quant_dev


class LedgerFixtureRoles(ScriptedRoles):
    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.run_dir = run_dir

    def plan(self, payload: dict, *, node_id: str | None = None) -> dict:
        if self.plans == 0:
            Ledger.open(self.run_dir, "quant-demo").llm_call(
                node_id=None, role="planner", purpose="plan", effort="medium",
                model="qwen3.5-27b", usage=Usage(input=80, cached_input=20,
                                                 output=40, reasoning=10),
                seconds=0.01, prompt_hash="a" * 64)
        return super().plan(payload, node_id=node_id)


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "requires macOS sandbox-exec")
class QuantDevRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = self.root / "config"
        config.mkdir()
        for name in ("quant_agent.json", "prices_quant_dev.json",
                     "hard_caps.json", "triggers.json"):
            shutil.copyfile(ROOT / "config" / name, config / name)
        shutil.copytree(ROOT / "config/prompts", config / "prompts")
        (self.root / "runs").mkdir()
        self.run_dir = self.root / "runs" / "run-test-quant"
        output = StringIO()
        redirect = redirect_stdout(output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_scripted_run_freezes_selection_and_retains_negative_candidate(self) -> None:
        with patch("quant_research.run._code_commit", return_value=("a" * 40, False)):
            report_path = run_quant_dev(project_root=self.root, run_dir=self.run_dir)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        selection = json.loads((self.run_dir / "selection.json").read_text(encoding="utf-8"))
        task_dir = self.run_dir / "tasks/quant-demo"
        events = [json.loads(line) for line in (task_dir / "ledger.jsonl").read_text().splitlines()]
        snapshots = [json.loads(line) for line in (task_dir / "tree.jsonl").read_text().splitlines()]
        latest = {node["node_id"]: node for node in snapshots}
        self.assertEqual(manifest["schema"], "quant_run/1")
        self.assertEqual(manifest["mode"], "scripted")
        self.assertEqual(manifest["model"], "test-scripted")
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(report["resource_cost"], 0)
        self.assertFalse(report["holdout_scored"])
        self.assertFalse(selection["holdout_scored"])
        self.assertEqual(latest["n0001"]["status"], "rejected")
        self.assertEqual(report["negative_or_failed_nodes"], ["n0001"])
        self.assertEqual(selection["dev_score"], latest[report["selected_node"]]["score"]["dev"])
        self.assertEqual([event["kind"] for event in events].count("score"), 1)
        self.assertFalse((self.run_dir / "final").exists())
        self.assertEqual(set(selection["node_artifacts"]), set(latest))
        self.assertEqual(hashlib.sha256((task_dir / "tree.jsonl").read_bytes()).hexdigest(),
                         selection["tree_sha256"])
        for key in ("candidate", "dev_evidence", "baseline_evidence"):
            descriptor = selection[key]
            path = self.root / descriptor["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), descriptor["sha256"])
        for artifacts in selection["node_artifacts"].values():
            for descriptor in artifacts.values():
                if descriptor is not None:
                    path = self.root / descriptor["path"]
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), descriptor["sha256"])
        with patch("quant_research.run._code_commit", return_value=("a" * 40, False)):
            with self.assertRaisesRegex(ValueError, "fresh run"):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir)

    def test_qwen_mode_requires_key_before_creating_run(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir, mode="qwen",
                               allow_synthetic_paid_demo=True)
        self.assertFalse(self.run_dir.exists())

    def test_injected_model_fixture_keeps_four_bucket_ledger(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-a-live-key"}), \
                patch("quant_research.run._code_commit", return_value=("b" * 40, False)):
            report_path = run_quant_dev(
                project_root=self.root, run_dir=self.run_dir, mode="qwen",
                roles=LedgerFixtureRoles(self.run_dir), allow_synthetic_paid_demo=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["model_calls"], 1)
        self.assertTrue(report["usage_complete"])
        self.assertEqual(report["total_token_buckets"], {
            "input": 80, "cached_input": 20, "output": 40, "reasoning": 10})
        self.assertEqual(report["resource_cost"], 150)
        self.assertNotIn("test-not-a-live-key", report_path.read_text(encoding="utf-8"))

    def test_user_dataset_is_snapshotted_before_research(self) -> None:
        supplied = write_demo_dataset(self.root / "provided")
        panel = load_dataset(supplied)
        boundaries = (panel.dates[50], panel.dates[55],
                      panel.dates[95], panel.dates[100])
        with patch("quant_research.run._code_commit", return_value=("c" * 40, False)):
            report_path = run_quant_dev(
                project_root=self.root, run_dir=self.run_dir,
                dataset_manifest=supplied, boundaries=boundaries)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIn("USER-SUPPLIED", report["label"])
        copied = self.run_dir / "private/manifest.json"
        self.assertEqual(load_dataset(copied).csv_sha256, panel.csv_sha256)
        supplied.unlink()
        self.assertEqual(load_dataset(copied).csv_sha256, panel.csv_sha256)

    def assert_failed_run(self, *, usage_complete: bool = True) -> dict:
        report = json.loads((self.run_dir / "dev_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["selection"])
        self.assertFalse(report["holdout_scored"])
        self.assertFalse((self.run_dir / "selection.json").exists())
        self.assertFalse((self.run_dir / "final").exists())
        self.assertEqual(report["usage_complete"], usage_complete)
        task_events = Ledger(self.run_dir, "quant-demo", {}, {})._events()
        run_events = Ledger(self.run_dir, None, {}, {})._events()
        self.assertEqual([e["kind"] for e in task_events].count("task_end"), 1)
        self.assertEqual([e["kind"] for e in run_events].count("run_end"), 1)
        self.assertEqual(run_events[-1]["status"], "failed")
        self.assertEqual(task_events[-1]["status"], "failed")
        return report

    def test_qwen_requires_explicit_data_choice_before_key_or_run(self) -> None:
        with patch("quant_research.run._code_commit") as git, \
                patch("quant_research.run._execute_quant_dev") as execute:
            with self.assertRaisesRegex(ValueError, "dataset manifest"):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir, mode="qwen")
        git.assert_not_called()
        execute.assert_not_called()
        self.assertFalse(self.run_dir.exists())

    def test_user_data_requires_four_dates_in_every_mode(self) -> None:
        supplied = write_demo_dataset(self.root / "provided")
        for mode in ("scripted", "qwen"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "four explicit"):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir,
                               dataset_manifest=supplied, mode=mode)
        self.assertFalse(self.run_dir.exists())

    def test_preflight_needs_no_key_git_or_network_and_creates_no_run(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}), \
                patch("quant_research.run._code_commit") as git, \
                patch("quant_research.run._execute_quant_dev") as execute:
            preview = run_quant_dev(project_root=self.root, run_dir=self.run_dir,
                                    mode="qwen", allow_synthetic_paid_demo=True,
                                    preflight_only=True)
        git.assert_not_called()
        execute.assert_not_called()
        self.assertFalse(self.run_dir.exists())
        self.assertFalse(preview["will_call_model"])
        self.assertTrue(preview["preflight_only"])
        self.assertEqual(preview["data_type"], "synthetic")
        self.assertEqual(preview["train"]["sessions"], 51)
        self.assertEqual(preview["dev"]["sessions"], 41)
        self.assertEqual(preview["holdout"]["sessions"], 20)
        self.assertEqual(preview["budget_tokens"], 40000)

    def test_baseline_failure_writes_failed_terminal_and_report(self) -> None:
        with patch("quant_research.run._code_commit", return_value=("d" * 40, False)), \
                patch("quant_research.run.QuantHarness.score", side_effect=RuntimeError("PRIVATE-MARKER")):
            with self.assertRaises(QuantRunFailed):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir)
        self.assert_failed_run()
        self.assertNotIn("PRIVATE-MARKER", (self.run_dir / "dev_report.json").read_text())

    def test_role_initialization_failure_is_recorded(self) -> None:
        with patch("quant_research.run._code_commit", return_value=("d" * 40, False)), \
                patch("quant_research.run.ScriptedRoles", side_effect=RuntimeError("PRIVATE-MARKER")):
            with self.assertRaises(QuantRunFailed):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir)
        self.assert_failed_run()

    def test_model_error_cannot_generate_selection(self) -> None:
        class ErrorRoles(ScriptedRoles):
            def plan(self, *args, **kwargs):
                raise ModelCallError("PRIVATE-MARKER")
        with patch("quant_research.run._code_commit", return_value=("e" * 40, False)):
            with self.assertRaises(QuantRunFailed):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir, roles=ErrorRoles())
        report = self.assert_failed_run()
        self.assertEqual(report["failure"]["kind"], "agent_error")

    def test_unknown_usage_is_failed_and_keeps_partial_cost(self) -> None:
        directory = self.run_dir
        class UnknownRoles(ScriptedRoles):
            def plan(self, *args, **kwargs):
                ledger = Ledger.open(directory, "quant-demo")
                ledger.llm_call(node_id=None, role="planner", purpose="plan", effort="medium",
                                model="test-scripted", usage=Usage(input=11), seconds=0.01,
                                prompt_hash="a" * 64)
                ledger.llm_call(node_id=None, role="planner", purpose="plan", effort="medium",
                                model="test-scripted", usage=Usage(), seconds=0.01,
                                prompt_hash="b" * 64, ok=False, usage_status="unknown")
                raise UnknownUsageError("PRIVATE-MARKER")
        with patch("quant_research.run._code_commit", return_value=("e" * 40, False)):
            with self.assertRaises(QuantRunFailed):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir, roles=UnknownRoles())
        report = self.assert_failed_run(usage_complete=False)
        self.assertEqual(report["failure"]["kind"], "unknown_usage")
        self.assertEqual(report["reported_resource_cost"], 11)
        self.assertEqual(report["unknown_usage_calls"], 1)
        self.assertIsNone(report["total_token_buckets"])
        self.assertIsNone(report["resource_cost"])

    def test_changed_prompt_stops_before_next_role_and_prevents_selection(self) -> None:
        path = self.root / "config/prompts/quant_executor.md"
        class ChangingRoles(ScriptedRoles):
            def plan(self, *args, **kwargs):
                path.write_text("Changed while planner was running")
                return super().plan(*args, **kwargs)
            def execute(self, *args, **kwargs):
                raise AssertionError("Next role must not run")
        with patch("quant_research.run._code_commit", return_value=("e" * 40, False)):
            with self.assertRaises(QuantRunFailed):
                run_quant_dev(project_root=self.root, run_dir=self.run_dir, roles=ChangingRoles())
        report = self.assert_failed_run()
        self.assertEqual(report["failure"]["kind"], "input_changed")

    def test_full_new_protocol_reaches_one_sealed_attempt(self) -> None:
        with patch("quant_research.run._code_commit", return_value=("f" * 40, False)):
            run_quant_dev(project_root=self.root, run_dir=self.run_dir)
        harness = harness_from_run(self.run_dir)
        report = finalize_holdout(harness, self.run_dir / "selection.json", self.run_dir / "final")
        self.assertEqual(report["status"], "scored", report.get("failure"))
        with self.assertRaises(FileExistsError):
            finalize_holdout(harness, self.run_dir / "selection.json", self.run_dir / "final")


class QuantPreflightCliTests(unittest.TestCase):
    def test_cli_preflight_is_json_and_needs_no_credentials(self) -> None:
        with TemporaryDirectory() as temporary, \
                patch.dict("os.environ", {"OPENAI_API_KEY": ""}), \
                patch("quant_research.run._code_commit") as git, \
                patch("quant_research.run._execute_quant_dev") as execute, \
                redirect_stdout(StringIO()) as output:
            root = Path(temporary)
            status = main(["--project-root", str(root), "--mode", "qwen",
                           "--allow-synthetic-paid-demo", "--preflight-only"])
            self.assertEqual(status, 0)
            preview = json.loads(output.getvalue())
            self.assertEqual(preview["mode"], "qwen")
            self.assertFalse(preview["will_call_model"])
            self.assertEqual(preview["holdout"]["sessions"], 20)
            self.assertFalse((root / "runs").exists())
        git.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
