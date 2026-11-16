"""Dev-only quant task contract and real sandboxed candidate execution."""

from __future__ import annotations

import json
import errno
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from quant_research.demo_data import write_demo_dataset
from quant_research.data import load_dataset
from quant_research.harness import QuantHarness, _child_env


BASELINE = "def compute_signal(history):\n    return {symbol: 0.0 for symbol in history}\n"
MOMENTUM = (
    "def compute_signal(history):\n"
    "    return {symbol: (prices[-1] / prices[-6] - 1.0) if len(prices) > 5 else 0.0\n"
    "            for symbol, prices in history.items()}\n"
)


class _HarnessFixture:
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task_root = self.root / "task" / "quant-demo"
        workspace = self.task_root / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "original.py").write_text(BASELINE, encoding="utf-8")
        (workspace / "profile.json").write_text(json.dumps({
            "schema": "profile/1", "features": ["time_series"],
            "summary": {"frequency": "daily"},
        }), encoding="utf-8")
        self.manifest = write_demo_dataset(self.root / "private")
        self.harness = self.make_harness()
        self.node = self.root / "runs" / "run-test" / "tasks" / "quant-demo" / "nodes" / "n0000"
        self.node.mkdir(parents=True)
        self.solution = self.node / "solution.py"
        self.solution.write_text(BASELINE, encoding="utf-8")

    def make_harness(self, **overrides):
        panel = load_dataset(self.manifest)
        args = dict(
            train_end=panel.dates[50], dev_start=panel.dates[52],
            dev_end=panel.dates[56], holdout_start=panel.dates[58],
            budget_money=1000.0, wall_seconds=60, top_k=1,
            fee_bps=2.0, slippage_bps=3.0, run_seconds=10, memory_mb=512,
        )
        args.update(overrides)
        return QuantHarness(self.task_root, self.manifest, **args)


class QuantHarnessTests(_HarnessFixture, unittest.TestCase):
    def test_task_contract_and_ast_validation_do_not_run_candidate(self):
        task = self.harness.load_task("quant-demo")
        self.assertEqual(task["scenario"], "quant")
        self.assertEqual(task["workspace"].name, "workspace")
        self.assertEqual(task["budget"]["money"], 1000.0)
        self.assertEqual(task["metric"]["name"], "net_excess_return")
        self.assertEqual(task["features"], ["time_series"])
        self.solution.write_text(
            "def compute_signal(history):\n    raise RuntimeError('only score executes')\n",
            encoding="utf-8")
        parsed = self.harness.run_script("quant-demo", self.node, ["solution.py"])
        self.assertEqual(parsed["exit_code"], 0)
        self.solution.write_text("import socket\n" + BASELINE, encoding="utf-8")
        blocked = self.harness.run_script("quant-demo", self.node, ["solution.py"])
        self.assertNotEqual(blocked["exit_code"], 0)
        self.assertIn("compute_signal", blocked["stderr_tail"])
        with self.assertRaises(KeyError):
            self.harness.load_task("quant-other")
        with self.assertRaises(PermissionError):
            self.harness.score("quant-demo", self.node, "holdout")

    def test_bad_splits_and_workspace_data_are_rejected(self):
        panel = self.harness.panel
        with self.assertRaisesRegex(ValueError, "gap"):
            self.make_harness(dev_start=panel.dates[51])
        with self.assertRaisesRegex(ValueError, "visible workspace"):
            QuantHarness(
                self.task_root, self.task_root / "workspace" / "profile.json",
                train_end=panel.dates[50], dev_start=panel.dates[52],
                dev_end=panel.dates[56], holdout_start=panel.dates[58],
                budget_money=1000.0, wall_seconds=60,
            )
        profile_path = self.task_root / "workspace" / "profile.json"
        profile_path.write_text(json.dumps({"schema": "profile/1",
                                            "features": [["time_series"]]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid trigger features"):
            self.make_harness()

    def test_child_env_never_inherits_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "not-for-child",
                                  "PYTHONPATH": "/private/data"}):
            env = _child_env(self.node)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["HOME"], str(self.node))

    def test_missing_sandbox_has_no_fallback(self):
        with patch("quant_research.harness.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "refusing unisolated"):
                self.harness.score("quant-demo", self.node, "dev")


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "requires macOS sandbox-exec")
class QuantHarnessSandboxTests(_HarnessFixture, unittest.TestCase):
    def test_dev_score_is_reproducible_and_json_ready(self):
        self.solution.write_text(MOMENTUM, encoding="utf-8")
        first = self.harness.score("quant-demo", self.node, "dev")
        second = self.harness.score("quant-demo", self.node, "dev")
        self.assertTrue(first["ok"], first.get("error"))
        self.assertTrue(first["correct"])
        self.assertEqual(first["split"], "dev")
        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["quant_metrics"], second["quant_metrics"])
        self.assertEqual(first["quant_metrics"]["dataset_sha256"], self.harness.panel.csv_sha256)
        self.assertEqual(first["quant_metrics"]["dev_end"], self.harness.splits.dev[-1].isoformat())
        json.dumps(first, allow_nan=False)

    def test_candidate_file_and_network_attempts_are_denied(self):
        secret = self.root / "private" / "secret.txt"
        secret.write_text("PRIVATE-MARKER", encoding="utf-8")
        self.solution.write_text(
            "def compute_signal(history):\n"
            f"    open({str(secret)!r}).read()\n"
            "    return {symbol: 0.0 for symbol in history}\n",
            encoding="utf-8")
        # AST parsing does not execute a candidate; the sandboxed score does.
        self.assertEqual(self.harness.run_script("quant-demo", self.node, ["solution.py"])["exit_code"], 0)
        failed = self.harness.score("quant-demo", self.node, "dev")
        self.assertFalse(failed["ok"])
        self.assertNotIn("PRIVATE-MARKER", json.dumps(failed))
        self.solution.write_text("import socket\n" + BASELINE, encoding="utf-8")
        self.assertNotEqual(self.harness.run_script("quant-demo", self.node, ["solution.py"])["exit_code"], 0)

    def test_os_sandbox_denies_private_read_even_outside_candidate_validator(self):
        secret = self.root / "private" / "secret.txt"
        secret.write_text("PRIVATE-MARKER", encoding="utf-8")
        profile = self.harness._sandbox_profile(self.solution)
        result = subprocess.run(
            [shutil.which("sandbox-exec"), "-p", profile, sys.executable,
             "-I", "-S", "-c", f"open({str(secret)!r}).read()"],
            cwd=self.node, env=_child_env(self.node), capture_output=True, text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("PRIVATE-MARKER", result.stdout + result.stderr)

    def test_os_sandbox_denies_network_even_outside_candidate_validator(self):
        profile = self.harness._sandbox_profile(self.solution)
        result = subprocess.run(
            [shutil.which("sandbox-exec"), "-p", profile, sys.executable,
             "-I", "-S", "-c",
             "import socket; s = socket.socket(); print(s.connect_ex(('127.0.0.1', 1)))"],
            cwd=self.node, env=_child_env(self.node), capture_output=True, text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(int(result.stdout.strip()), {errno.EPERM, errno.EACCES})


if __name__ == "__main__":
    unittest.main()
