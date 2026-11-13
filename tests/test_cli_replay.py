from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from library import Library
from research_agent import cli
from research_agent.replay import replay_node, sha256_file
from tests.test_runner import FakeHarness, FakeLedger, FakeRoles, idea, score


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class CliReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config_dir = self.root / "config"
        (config_dir / "prompts").mkdir(parents=True)
        for name in ("agent.json", "hard_caps.json", "triggers.json"):
            shutil.copyfile(SOURCE_ROOT / "config" / name, config_dir / name)
        for role in ("planner", "executor", "diagnoser", "summarizer"):
            shutil.copyfile(SOURCE_ROOT / "config/prompts" / f"{role}.md", config_dir / "prompts" / f"{role}.md")
        prices = {
            "schema": "prices/1", "currency": "CNY", "unit_tokens": 1_000_000,
            "models": {"test-model": {"input": 1.0, "cached_input": 0.5, "output": 2.0, "reasoning": 2.0}},
        }
        (config_dir / "prices.json").write_text(json.dumps(prices), encoding="utf-8")
        self.baseline = self.root / "baseline.json"
        self.baseline.write_text(json.dumps(score(0.1)), encoding="utf-8")
        self.run_dir = self.root / "runs/run-test"
        self.run_dir.mkdir(parents=True)
        self.library_root = self.run_dir / "library"
        Library.open(self.library_root)
        self.harness = FakeHarness(self.root)
        self.ledger = FakeLedger()
        self.roles = FakeRoles(self.ledger, [idea("good")], draft=False)
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest = {
            "schema": "run/1", "run_id": "run-test", "phase": "practice",
            "config": "default", "rep": 1, "bank_version": "bank-v1", "tasks": ["tab-sample"],
            "library": {"root": "runs/run-test/library", "start_version": "lib-0", "frozen": False},
            "agent_config": {"path": "config/agent.json", "sha256": sha256_file(config_dir / "agent.json")},
            "prices": {"path": "config/prices.json", "sha256": sha256_file(config_dir / "prices.json")},
            "hard_caps": {"budget_ratio": 0.9}, "code_commit": "test", "model": "test-model",
            "started_ts": "2026-11-13T00:00:00.000Z", "ended_ts": None,
        }
        self._save_manifest()

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def _run_args(self, **changes):
        data = {
            "project_root": self.root, "run_dir": self.run_dir, "task_id": "tab-sample",
            "baseline_json": self.baseline, "config": None, "model": None,
            "max_output_tokens": 30000,
        }
        data.update(changes)
        return argparse.Namespace(**data)

    def _execute_fake_run(self):
        with (
            patch.object(cli, "_ledger", return_value=self.ledger),
            patch.object(cli, "_harness", return_value=self.harness),
            patch.object(cli, "ModelClient", return_value=object()),
            patch.object(cli, "RoleAgents", return_value=self.roles),
        ):
            return cli._run(self._run_args())

    def test_cli_seals_inputs_and_replay_uses_historical_library(self):
        outcome = self._execute_fake_run()
        self.assertEqual(outcome["best_node"], "n0001")
        context = json.loads((self.run_dir / "tasks/tab-sample/replay_context.json").read_text(encoding="utf-8"))
        self.assertEqual(context["sources"]["baseline"]["sha256"], sha256_file(self.baseline))
        self.assertEqual(context["nodes"]["n0001"]["library_version"], "lib-0")
        report = replay_node(
            run_dir=self.run_dir, task_id="tab-sample", node_id="n0001",
            harness=self.harness, library_root=self.library_root,
            require_provenance=True,
        )
        self.assertTrue(report["ok"])
        self.assertTrue(report["integrity_verified"])
        self.assertEqual(report["observation"]["current"]["seed"], 12)

    def test_modified_solution_is_rejected_before_rescoring(self):
        self._execute_fake_run()
        solution = self.run_dir / "tasks/tab-sample/nodes/n0001/solution.py"
        solution.write_text("changed but still scores the same", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "solution.py changed"):
            replay_node(run_dir=self.run_dir, task_id="tab-sample", node_id="n0001",
                        harness=self.harness, library_root=self.library_root)

    def test_changed_baseline_or_library_snapshot_is_rejected(self):
        self._execute_fake_run()
        self.baseline.write_text(self.baseline.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Replay source changed: baseline"):
            replay_node(run_dir=self.run_dir, task_id="tab-sample", node_id="n0001",
                        harness=self.harness, library_root=self.library_root)
        self.baseline.write_text(json.dumps(score(0.1)), encoding="utf-8")
        snapshot = self.library_root / "snapshots/lib-0.json"
        snapshot.write_text(snapshot.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Historical library snapshot changed"):
            replay_node(run_dir=self.run_dir, task_id="tab-sample", node_id="n0001",
                        harness=self.harness, library_root=self.library_root)

    def test_manifest_hash_placeholder_and_mismatch_rejected(self):
        self.manifest["agent_config"]["sha256"] = "..."
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "placeholder"):
            cli._run(self._run_args())
        self.manifest["agent_config"]["sha256"] = "0" * 64
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "does not match"):
            cli._run(self._run_args())
        self.manifest["agent_config"]["sha256"] = sha256_file(self.root / "config/agent.json")
        self.manifest["prices"]["sha256"] = "..."
        self._save_manifest()
        with self.assertRaisesRegex(ValueError, "placeholder"):
            cli._run(self._run_args())

    def test_unrecorded_config_or_model_override_rejected(self):
        with self.assertRaisesRegex(ValueError, "--config may only repeat"):
            cli._run(self._run_args(config=self.root / "config/triggers.json"))
        with self.assertRaisesRegex(ValueError, "--model may only repeat"):
            cli._run(self._run_args(model="different-model"))

    def test_tuned_manifest_config_controls_library_rules_and_replay(self):
        tuned = json.loads((self.root / "config/agent.json").read_text(encoding="utf-8"))
        tuned["library_rules"]["init_credibility"] = 0.73
        tuned_path = self.root / "config/tuned.json"
        tuned_path.write_text(json.dumps(tuned), encoding="utf-8")
        self.manifest["agent_config"] = {"path": "config/tuned.json", "sha256": sha256_file(tuned_path)}
        self._save_manifest()
        self.roles = FakeRoles(self.ledger, [idea("good")], draft=True)
        wrong = str(self.root / "config/hard_caps.json")
        with patch.dict(os.environ, {"RA_AGENT_CONFIG": wrong, "RA_TRIGGERS_CONFIG": wrong}):
            self._execute_fake_run()
            report = replay_node(run_dir=self.run_dir, task_id="tab-sample", node_id="n0001",
                                 harness=self.harness, library_root=self.library_root)
        self.assertTrue(report["integrity_verified"])
        library = Library.open(self.library_root)
        self.assertEqual(library.entries[0]["credibility"], 0.73)

    def test_historical_library_must_contain_cited_entry(self):
        self._execute_fake_run()
        task_dir = self.run_dir / "tasks/tab-sample"
        tree_path = task_dir / "tree.jsonl"
        rows = [json.loads(line) for line in tree_path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["node_id"] == "n0001":
                row["cites"] = ["exp-9999"]
        tree_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        context_path = task_dir / "replay_context.json"
        context = json.loads(context_path.read_text(encoding="utf-8"))
        context["tree_sha256"] = sha256_file(tree_path)
        context_path.write_text(json.dumps(context), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "absent from its historical library"):
            replay_node(run_dir=self.run_dir, task_id="tab-sample", node_id="n0001",
                        harness=self.harness, library_root=self.library_root)

    def test_manifest_backed_replay_needs_sealed_context(self):
        with self.assertRaisesRegex(ValueError, "Sealed replay context"):
            replay_node(run_dir=self.run_dir, task_id="tab-sample", node_id="n0000",
                        harness=self.harness, library_root=self.library_root)


if __name__ == "__main__":
    unittest.main()
