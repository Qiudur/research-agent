"""B-owned contract and sandbox tests for the production task harness."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

import harness


ROOT = Path(__file__).resolve().parents[1]
TASK = "spd-distance-matrix"


class HarnessContractTests(unittest.TestCase):
    def test_task_loading_and_bank_filters(self):
        task = harness.load_task(TASK)
        self.assertEqual(task["schema"], "task/1")
        self.assertEqual(task["scenario"], "speedup")
        self.assertEqual(task["workspace"].name, "workspace")
        self.assertIn("nested_python_loop", task["features"])
        self.assertEqual(harness.list_tasks("bank-v1", tier="practice", scenario="speedup"), [TASK])

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


if __name__ == "__main__":
    unittest.main()
