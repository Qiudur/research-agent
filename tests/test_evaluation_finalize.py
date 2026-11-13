from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from evaluation.finalize import finish_run, finish_task, start_run
from evaluation.manifest import create_manifest
from evaluation.report import extract_summary, run_report, task_report
from ledger import Ledger
from library import Library


PROJECT = Path(__file__).resolve().parents[1]
TASK = "spd-distance-matrix"


def setup(root: Path) -> Path:
    config = root / "config"
    config.mkdir()
    for name in ("agent.json", "hard_caps.json"):
        shutil.copyfile(PROJECT / "config" / name, config / name)
    shutil.copyfile(PROJECT / "tests/fixtures/prices_test_only.json", config / "prices_test_only.json")
    run = root / "runs/run-20261114-120000-abcd"
    create_manifest(project_root=root, run_dir=run, phase="dev", config="default", rep=1,
                    bank_version="bank-v1", tasks=[TASK], library_root=run / "library",
                    library_start_version="lib-0", library_frozen=False, model="test-scripted",
                    agent_config_path=config / "agent.json", prices_path=config / "prices_test_only.json",
                    code_commit="a" * 40)
    start_run(run)
    node = run / "tasks" / TASK / "nodes" / "n0000"
    node.mkdir(parents=True)
    (node / "solution.py").write_text("def f(): pass\n")
    ledger = Ledger.open(run, TASK)
    ledger.emit("task_start", budget=1, wall_seconds=60)
    ledger.emit("quote", predicted={"cost": 0.1, "seconds": 2, "score": 0.5}, basis={"source": "test"})
    ledger.emit("stop", reason="planner", best_node="n0000", dev_score=0)
    return run


class FakeHarness:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def score(self, task_id, node_dir, split):
        self.calls += 1
        assert os.environ.get("RA_EVAL") == "1"
        assert task_id == TASK and node_dir.name == "n0000" and split == "holdout"
        if self.fail:
            raise RuntimeError("holdout backend failed")
        return {"schema": "score/1", "ok": True, "split": "holdout", "score": 0.7,
                "raw": 2.0, "spread": 0.1, "correct": True, "cheat_flags": [], "error": None,
                "seconds": {"run": 0.2, "queue": 0.1}, "seed": 1234}


class FinalizeTests(unittest.TestCase):
    def test_one_holdout_then_task_end_and_run_end(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            harness = FakeHarness()
            self.assertNotIn("RA_EVAL", os.environ)
            result = finish_task(run, TASK, harness)
            self.assertNotIn("RA_EVAL", os.environ)
            self.assertEqual(harness.calls, 1)
            self.assertEqual(result["task_end"]["holdout_score"], 0.7)
            self.assertEqual(result["task_end"]["total_cost"], 0)
            self.assertAlmostEqual(result["task_end"]["quote_error"]["score"], 0.2)
            with self.assertRaisesRegex(ValueError, "already recorded"):
                finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, 1)
            finished = finish_run(run)
            self.assertEqual(finished["total_cost"], 0)
            self.assertIsNotNone(json.loads((run / "manifest.json").read_text())["ended_ts"])
            self.assertEqual([event["kind"] for event in Ledger.open(run, None)._events()], ["run_start", "run_end"])

    def test_failed_holdout_cannot_be_retried(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            harness = FakeHarness(fail=True)
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                finish_task(run, TASK, harness)
            self.assertNotIn("RA_EVAL", os.environ)
            self.assertTrue((run / "tasks" / TASK / "holdout_attempt.json").exists())
            with self.assertRaises(FileExistsError):
                finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, 1)

    def test_exam_marked_library_remains_unchanged(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["phase"] = "exam"
            manifest["library"]["frozen"] = True
            manifest_path.write_text(json.dumps(manifest))  # TEST fixture only, not a formal exam
            library = Library.open(run / "library")
            self.assertEqual(library.version, "lib-0")
            finish_task(run, TASK, FakeHarness())
            self.assertEqual(Library.open(run / "library").version, "lib-0")

    def test_report_reconciles_cost_and_cites_ledger_rows(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            node = run / "tasks" / TASK / "tree.jsonl"
            root_node = {"schema": "node/1", "node_id": "n0000", "parent": None,
                         "hypothesis": "Baseline", "status": "accepted",
                         "score": {"dev": 0}, "cost": 0, "failure": None, "cites": []}
            node.write_text(json.dumps(root_node) + "\n")
            finish_task(run, TASK, FakeHarness())
            finish_run(run)
            report = run_report(run)
            self.assertEqual(report["tasks"][0]["evidence"]["task_end_seq"], 4)
            self.assertEqual(report["evidence"]["run_end_seq"], 2)
            self.assertEqual(report["tasks"][0]["model_calls"], 0)
            self.assertEqual(task_report(run, TASK)["holdout_score"], 0.7)
            self.assertEqual(extract_summary(run, TASK)["experiments"], [])
            self.assertTrue((run / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
