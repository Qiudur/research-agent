from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import harness
from evaluation.finalize import finish_run, finish_task, start_run
from evaluation.manifest import create_manifest
from evaluation.report import extract_summary, run_report, task_report
from ledger import Ledger
from library import Library


PROJECT = Path(__file__).resolve().parents[1]
TASK = "spd-distance-matrix"


def setup(root: Path, *, best_node: str = "n0000") -> Path:
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
    node = run / "tasks" / TASK / "nodes" / best_node
    node.mkdir(parents=True)
    (node / "solution.py").write_text("def f(): pass\n")
    ledger = Ledger.open(run, TASK)
    ledger.emit("task_start", budget=1, wall_seconds=60)
    ledger.emit("quote", predicted={"cost": 0.1, "seconds": 2, "score": 0.5}, basis={"source": "test"})
    ledger.emit("stop", reason="planner", best_node=best_node, dev_score=0)
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


class ReturnedHarness:
    def __init__(self, result: dict):
        self.result = result
        self.calls = 0

    def score(self, task_id, node_dir, split):
        self.calls += 1
        assert os.environ.get("RA_EVAL") == "1"
        assert task_id == TASK and node_dir.name == "n0000" and split == "holdout"
        return self.result


class FinalizeTests(unittest.TestCase):
    def test_bank_v2_candidate_exception_redacts_persisted_holdout_score(self):
        secret = "SENSITIVE_HIDDEN_ROW_AND_PREDICTION_12345"
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run = setup(root, best_node="n0001")
            bank = root / "tasks/bank.json"
            bank.parent.mkdir(exist_ok=True)
            bank.write_text(json.dumps({"schema": "bank/1", "version": "bank-v2"}))
            task = {"scenario": "speedup", "benchmark": {"holdout_size": 2}}
            payload = {"ok": False, "cheat_flags": ["network", secret],
                       "error": {"kind": "exception", "message": secret, "trace_tail": secret}}
            meta = {"timed_out": False, "oom": False, "exit_code": 1, "stderr_tail": secret}
            with patch.object(harness, "_tasks_root", return_value=bank.parent), \
                    patch.object(harness, "_project_root", return_value=root), \
                    patch.object(harness, "load_task", return_value=task), \
                    patch.object(harness, "_execute", return_value=(payload, meta)) as worker:
                result = finish_task(run, TASK, harness)
            worker.assert_called_once()
            saved = (run / "tasks" / TASK / "holdout_score.json").read_text()
            self.assertNotIn(secret, saved)
            self.assertEqual(json.loads(saved)["error"],
                             {"kind": "exception", "message": "Candidate execution failed"})
            self.assertEqual(result["task_end"]["holdout_score"], 0.0)

    def test_zero_candidate_holdout_reconciles_and_retains_failure_kind(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp), best_node="n0001")
            task_dir = run / "tasks" / TASK
            nodes = [
                {"schema": "node/1", "node_id": "n0000", "parent": None,
                 "hypothesis": "Baseline", "status": "accepted",
                 "score": {"dev": 0}, "cost": 0, "failure": None, "cites": []},
                {"schema": "node/1", "node_id": "n0001", "parent": "n0000",
                 "hypothesis": "Candidate", "status": "accepted",
                 "score": {"dev": 0.3}, "cost": 0, "failure": None, "cites": []},
            ]
            (task_dir / "tree.jsonl").write_text("".join(json.dumps(node) + "\n" for node in nodes))

            class ZeroHarness:
                def __init__(self):
                    self.calls = 0

                def score(self, task_id, node_dir, split):
                    self.calls += 1
                    assert os.environ.get("RA_EVAL") == "1"
                    assert node_dir.name == "n0001" and split == "holdout"
                    return {"schema": "score/1", "ok": True, "split": "holdout",
                            "score": 0.0, "raw": 0.0, "spread": 0.0, "correct": False,
                            "cheat_flags": [], "error": {"kind": "oom", "message": "candidate OOM"},
                            "seconds": {"run": 0.2, "queue": 0.0}, "seed": 13}

            scorer = ZeroHarness()
            result = finish_task(run, TASK, scorer)
            self.assertEqual(scorer.calls, 1)
            self.assertEqual(result["task_end"]["holdout_score"], 0.0)
            self.assertEqual(json.loads((task_dir / "holdout_score.json").read_text())["error"]["kind"], "oom")
            with self.assertRaisesRegex(ValueError, "already recorded"):
                finish_task(run, TASK, scorer)
            self.assertEqual(scorer.calls, 1)
            finish_run(run)
            self.assertEqual(run_report(run)["tasks"][0]["holdout_score"], 0.0)

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

    def test_failed_score_records_safe_metadata_without_retry(self):
        cases = [
            ("not_ok", {
                "schema": "score/1", "split": "holdout", "ok": False, "score": None,
                "error": {"kind": "timeout", "message": "Scoring timed out",
                          "trace_tail": "trace-secret"},
                "seconds": {"run": 302.0, "queue": 4.0, "labels": ["hidden-label-value"]},
                "seed": 1234, "labels": ["hidden-label-value"],
                "predictions": ["prediction-secret"],
            }, "timeout", "Scoring timed out"),
            ("missing_score", {
                "schema": "score/1", "split": "holdout", "ok": True, "score": None,
                "error": None, "seconds": {"run": float("inf"), "queue": -3.0},
                "seed": True, "labels": ["hidden-label-value"],
                "predictions": ["prediction-secret"],
            }, "missing_score", "Holdout scorer returned no score"),
            ("untrusted_message", {
                "schema": "score/1", "split": "holdout", "ok": False, "score": None,
                "error": {"kind": "exception",
                          "message": "hidden-label-value prediction-secret",
                          "trace_tail": "trace-secret"},
                "seconds": {"run": 1.5, "queue": 0}, "seed": 99,
            }, "exception", "Scoring worker failed"),
        ]
        for name, result, expected_kind, expected_message in cases:
            with self.subTest(name=name), TemporaryDirectory() as temp:
                run = setup(Path(temp))
                harness = ReturnedHarness(result)
                with self.assertRaisesRegex(RuntimeError, "must not be retried"):
                    finish_task(run, TASK, harness)
                task_dir = run / "tasks" / TASK
                failure = json.loads((task_dir / "holdout_failure.json").read_text())
                self.assertEqual(failure["schema"], "holdout_failure/1")
                self.assertEqual(failure["best_node"], "n0000")
                self.assertEqual(failure["reason"],
                                 "missing_score" if name == "missing_score" else "not_ok")
                self.assertEqual(failure["error"],
                                 {"kind": expected_kind, "message": expected_message})
                if name == "missing_score":
                    self.assertNotIn("seconds", failure)
                    self.assertNotIn("seed", failure)
                else:
                    self.assertIn("seconds", failure)
                    self.assertIn("seed", failure)
                contents = (task_dir / "holdout_failure.json").read_text()
                for secret in ("hidden-label-value", "prediction-secret", "trace-secret"):
                    self.assertNotIn(secret, contents)
                self.assertFalse((task_dir / "holdout_failure.json.tmp").exists())
                self.assertFalse((task_dir / "holdout_score.json").exists())
                self.assertFalse(any(event["kind"] == "task_end"
                                     for event in Ledger.open(run, TASK)._events()))
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
