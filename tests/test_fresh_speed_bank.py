"""Dev-only checks for five candidate bank-v2 speedup tasks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

import harness
from harness_worker import _case, _edge_cases, _equal


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("pair_sum_count", "range_counts", "sliding_minimum",
         "running_median", "inversion_count")


def _function(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"fixture_{name}_{path.parent.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


class FreshSpeedBankTests(unittest.TestCase):
    def test_new_tasks_do_not_reuse_bank_v1(self):
        old = json.loads((ROOT / "tasks/bank.json").read_text(encoding="utf-8"))
        used = {item["task_id"] for item in old["tasks"]}
        for kind in KINDS:
            task_id = "spd-" + kind.replace("_", "-")
            self.assertNotIn(task_id, used)
            self.assertEqual(harness.load_task(task_id)["benchmark"]["kind"], kind)

    def test_reference_agrees_on_edges_and_fresh_dev_cases(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                root = ROOT / "tasks" / ("spd-" + kind.replace("_", "-"))
                original = _function(root / "workspace/original.py", kind)
                reference = _function(root / "hidden/reference.py", kind)
                cases = _edge_cases(kind, 739) + [_case(kind, 740, 47), _case(kind, 741, 119)]
                for case in cases:
                    self.assertTrue(_equal(kind, reference(*case), original(*case)))

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                         "Dev scorer requires macOS sandbox-exec")
    def test_real_dev_sandbox_baseline_reference_and_bad_candidate(self):
        for kind in KINDS:
            with self.subTest(kind=kind), TemporaryDirectory() as temp:
                task_id = "spd-" + kind.replace("_", "-")
                root = ROOT / "tasks" / task_id
                node = Path(temp)
                scores = {}
                for name, path in (("baseline", root / "workspace/original.py"),
                                   ("reference", root / "hidden/reference.py")):
                    shutil.copyfile(path, node / "solution.py")
                    scores[name] = harness.score(task_id, node, "dev", seed=739)
                    self.assertTrue(scores[name]["ok"], scores[name])
                    self.assertTrue(scores[name]["correct"], scores[name])
                self.assertGreater(scores["reference"]["raw"], 4.0)
                self.assertLess(scores["baseline"]["raw"], 2.0)
                (node / "solution.py").write_text(f"def {kind}(*args):\n    return 0\n",
                                                 encoding="utf-8")
                bad = harness.score(task_id, node, "dev", seed=740)
                self.assertTrue(bad["ok"], bad)
                self.assertFalse(bad["correct"], bad)
                self.assertEqual(bad["score"], 0)


if __name__ == "__main__":
    unittest.main()
