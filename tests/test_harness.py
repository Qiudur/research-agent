"""B-owned contract and sandbox tests for the production task harness."""

from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import harness


ROOT = Path(__file__).resolve().parents[1]
TASK = "spd-distance-matrix"


class HarnessContractTests(unittest.TestCase):
    def test_explicit_project_root_is_validated_and_scopes_task_profiles(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "config").mkdir()
            (root / "tasks").mkdir()
            shutil.copyfile(ROOT / "config/triggers.json", root / "config/triggers.json")
            shutil.copyfile(ROOT / "tasks/bank.json", root / "tasks/bank.json")
            shutil.copytree(ROOT / "tasks/spd-prefix-sum", root / "tasks/spd-prefix-sum")
            with patch.dict(os.environ, {"RA_PROJECT_ROOT": str(root)}, clear=False):
                old_tasks = os.environ.pop("RA_TASKS_ROOT", None)
                try:
                    self.assertEqual(harness._tasks_root(), root / "tasks")
                    self.assertEqual(harness.load_task("spd-prefix-sum")["workspace"],
                                     root / "tasks/spd-prefix-sum/workspace")
                finally:
                    if old_tasks is not None:
                        os.environ["RA_TASKS_ROOT"] = old_tasks
            with patch.dict(os.environ, {"RA_PROJECT_ROOT": "relative/path"}, clear=False):
                with self.assertRaisesRegex(ValueError, "absolute path"):
                    harness._project_root()
            with patch.dict(os.environ, {"RA_PROJECT_ROOT": str(root / "missing")}, clear=False):
                with self.assertRaisesRegex(ValueError, "tasks/bank.json"):
                    harness._project_root()

    def test_task_loading_and_bank_filters(self):
        task = harness.load_task(TASK)
        self.assertEqual(task["schema"], "task/1")
        self.assertEqual(task["scenario"], "speedup")
        self.assertEqual(task["workspace"].name, "workspace")
        self.assertIn("nested_python_loop", task["features"])
        self.assertEqual(set(harness.list_tasks("bank-v1", tier="practice", scenario="speedup")),
                         {TASK, "spd-rolling-mean", "spd-max-drawdown", "spd-edit-distance",
                          "spd-membership-filter"})
        self.assertEqual(len(harness.list_tasks("bank-v1", tier="validation", scenario="speedup")), 2)
        self.assertEqual(len(harness.list_tasks("bank-v1", tier="exam", scenario="speedup")), 5)

    def test_speedup_bank_measurements_have_replayable_source(self):
        bank = json.loads((ROOT / "tasks/bank.json").read_text(encoding="utf-8"))
        calibration = json.loads((ROOT / "docs/B-提速校准.json").read_text(encoding="utf-8"))
        measurements = {record["task_id"]: record for record in calibration["records"]}
        self.assertEqual(len(measurements), 12)
        self.assertFalse(bank["frozen"])
        for entry in bank["tasks"]:
            if entry["task_id"] not in measurements:
                continue
            measured = measurements[entry["task_id"]]
            self.assertEqual(len(measured["seeds"]), 3)
            self.assertEqual(entry["baseline_raw"], statistics.median(measured["baseline_raw"]))
            self.assertEqual(entry["ceiling_raw"], statistics.median(measured["ceiling_raw"]))
            self.assertIsNone(entry["naive_cost"])
        self.assertEqual(next(item["trap"] for item in bank["tasks"]
                              if item["task_id"] == "spd-already-vectorized"), "stop_loss")

    def test_run_script_accepts_node_script_and_argv(self):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            shutil.copyfile(ROOT / "tasks" / TASK / "workspace" / "original.py", node / "solution.py")
            (node / "inspect.py").write_text("import sys\nprint(sys.argv[1:])\n", encoding="utf-8")
            result = harness.run_script(TASK, node, ["inspect.py", "--probe"])
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("['--probe']", result["stdout_tail"])

    def test_holdout_requires_evaluation_gate(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(PermissionError):
                harness.score(TASK, Path(temp), "holdout", seed=7)

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                         "B's real scorer requires macOS sandbox-exec")
    def test_tabular_dispatch_returns_real_score_contract(self):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            (node / "solution.py").write_text(
                "def fit_predict(train_csv, test_rows):\n    return [0.5] * len(test_rows)\n",
                encoding="utf-8")
            result = harness.score("tab-bank-marketing", node, "dev", seed=17)
            self.assertEqual(result["schema"], "score/1")
            self.assertEqual(result["split"], "dev")
            self.assertTrue(result["ok"])
            self.assertTrue(result["correct"])
            self.assertAlmostEqual(result["raw"], 0.5)
            self.assertEqual(result["seed"], 17)

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                         "B's real scorer requires macOS sandbox-exec")
    def test_dev_score_has_contract_and_rejects_wrong_output(self):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            shutil.copyfile(ROOT / "tasks" / TASK / "workspace" / "original.py", node / "solution.py")
            result = harness.score(TASK, node, "dev", seed=17)
            self.assertEqual(result["schema"], "score/1")
            self.assertEqual(result["split"], "dev")
            self.assertTrue(result["ok"])
            self.assertTrue(result["correct"])
            self.assertIsInstance(result["score"], float)
            (node / "solution.py").write_text(
                "def distance_matrix(points):\n    return [[0.0] * len(points) for _ in points]\n",
                encoding="utf-8",
            )
            bad = harness.score(TASK, node, "dev", seed=18)
            self.assertTrue(bad["ok"])
            self.assertFalse(bad["correct"])
            self.assertIn("behavior_change", bad["cheat_flags"])


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "B's sandbox tests require macOS sandbox-exec")
class HarnessSandboxTests(unittest.TestCase):
    def _run(self, source: str):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            (node / "solution.py").write_text(source, encoding="utf-8")
            yield harness.run_script(TASK, node, ["solution.py"])

    def test_reads_hidden_is_flagged(self):
        hidden = (ROOT / "tasks" / TASK / "hidden" / "reference.py").resolve()
        source = f"from pathlib import Path\nPath({str(hidden)!r}).read_text()\n"
        result = next(self._run(source))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("reads_hidden", result["stderr_tail"])

    def test_private_home_files_are_not_readable(self):
        with TemporaryDirectory(dir=Path.home()) as secret_dir:
            secret = Path(secret_dir) / "sandbox-secret-test.txt"
            secret.write_text("private-home-sentinel", encoding="utf-8")
            source = f"from pathlib import Path\nprint(Path({str(secret)!r}).read_text())\n"
            result = next(self._run(source))
            self.assertNotEqual(result["exit_code"], 0)
            self.assertNotIn("private-home-sentinel", result["stdout_tail"])

    def test_writes_outside_node_is_flagged(self):
        source = "from pathlib import Path\nPath('/tmp/research-agent-harness-escape').write_text('x')\n"
        result = next(self._run(source))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("writes_outside_node", result["stderr_tail"])

    def test_network_is_flagged(self):
        source = "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=1)\n"
        result = next(self._run(source))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("network", result["stderr_tail"])

    def test_memory_limit_is_enforced(self):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            (node / "solution.py").write_text("blob = bytearray(100 * 1024 * 1024)\n", encoding="utf-8")
            result = harness.run_script(TASK, node, ["solution.py"], harness.Limits(5, 32))
            self.assertTrue(result["oom"])

    @unittest.skipUnless((ROOT / ".venv/bin/python").is_file(),
                         "project Python virtual environment is unavailable")
    def test_project_venv_scores_and_still_blocks_hidden_files(self):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            task_id = "spd-prefix-sum"
            shutil.copyfile(ROOT / "tasks" / task_id / "workspace/original.py",
                            node / "solution.py")
            hidden = ROOT / "tasks" / task_id / "hidden/reference.py"
            (node / "probe.py").write_text(
                f"from pathlib import Path\nPath({str(hidden)!r}).read_text()\n",
                encoding="utf-8")
            program = (
                "import harness, json, sys\n"
                "from pathlib import Path\n"
                "node = Path(sys.argv[1])\n"
                "score = harness.score('spd-prefix-sum', node, 'dev', seed=91731)\n"
                "probe = harness.run_script('spd-prefix-sum', node, ['probe.py'])\n"
                "print(json.dumps({'ok': score['ok'], 'correct': score['correct'], "
                "'probe_exit': probe['exit_code'], 'probe_stderr': probe['stderr_tail']}))\n"
            )
            proc = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", program, str(node)],
                                  cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout.splitlines()[-1])
            self.assertTrue(result["ok"])
            self.assertTrue(result["correct"])
            self.assertNotEqual(result["probe_exit"], 0)
            self.assertIn("reads_hidden", result["probe_stderr"])


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "B's real scorer requires macOS sandbox-exec")
class AdditionalSpeedupTests(unittest.TestCase):
    TASKS = ("spd-rolling-mean", "spd-max-drawdown", "spd-edit-distance",
             "spd-membership-filter", "spd-prefix-sum", "spd-dedupe-order",
             "spd-group-counts", "spd-string-build", "spd-top-frequency",
             "spd-weighted-groups", "spd-already-vectorized")

    def test_new_task_baselines_and_wrong_candidates(self):
        for task_id in self.TASKS:
            with self.subTest(task_id=task_id), TemporaryDirectory() as temp:
                node = Path(temp)
                task_root = ROOT / "tasks" / task_id
                shutil.copyfile(task_root / "workspace/original.py", node / "solution.py")
                baseline = harness.score(task_id, node, "dev", seed=739)
                self.assertTrue(baseline["ok"], baseline)
                self.assertTrue(baseline["correct"])
                self.assertGreater(baseline["score"], 0)
                function = harness.load_task(task_id)["deliverable"]["function"]
                (node / "solution.py").write_text(
                    f"def {function}(*args):\n    return 0\n", encoding="utf-8")
                bad = harness.score(task_id, node, "dev", seed=740)
                self.assertTrue(bad["ok"], bad)
                self.assertFalse(bad["correct"])
                self.assertEqual(bad["score"], 0)
                self.assertIn("behavior_change", bad["cheat_flags"])

    def test_reference_candidates_score_on_all_new_workloads(self):
        for task_id in self.TASKS:
            with self.subTest(task_id=task_id), TemporaryDirectory() as temp:
                node = Path(temp)
                shutil.copyfile(ROOT / "tasks" / task_id / "hidden/reference.py",
                                node / "solution.py")
                result = harness.score(task_id, node, "dev", seed=20261114)
                self.assertTrue(result["ok"], result)
                self.assertTrue(result["correct"])
                self.assertGreater(result["score"], 0)

    def test_reference_candidate_passes_holdout_gate_and_new_seed(self):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            task_id = "spd-edit-distance"
            shutil.copyfile(ROOT / "tasks" / task_id / "hidden/reference.py", node / "solution.py")
            with self.assertRaises(PermissionError):
                harness.score(task_id, node, "holdout", seed=841)
            previous = os.environ.get("RA_EVAL")
            os.environ["RA_EVAL"] = "1"
            try:
                result = harness.score(task_id, node, "holdout", seed=841)
            finally:
                if previous is None:
                    os.environ.pop("RA_EVAL", None)
                else:
                    os.environ["RA_EVAL"] = previous
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["correct"])
            self.assertEqual(result["split"], "holdout")
            self.assertEqual(result["seed"], 841)
            self.assertGreater(result["raw"], 1.0)


if __name__ == "__main__":
    unittest.main()
