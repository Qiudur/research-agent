"""End-to-end dev research: real sandbox, no key, no sealed scoring."""

from __future__ import annotations

import hashlib
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
from quant_research.run import ROOT, ScriptedRoles, run_quant_dev


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
        (self.root / "runs").mkdir()
        self.run_dir = self.root / "runs" / "run-test-quant"

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
                run_quant_dev(project_root=self.root, run_dir=self.run_dir, mode="qwen")
        self.assertFalse(self.run_dir.exists())

    def test_injected_model_fixture_keeps_four_bucket_ledger(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-a-live-key"}), \
                patch("quant_research.run._code_commit", return_value=("b" * 40, False)):
            report_path = run_quant_dev(
                project_root=self.root, run_dir=self.run_dir, mode="qwen",
                roles=LedgerFixtureRoles(self.run_dir))
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


if __name__ == "__main__":
    unittest.main()
