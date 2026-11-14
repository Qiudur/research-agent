"""Pure fixtures for bank-v2 candidate sandbox failure accounting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import harness
import tabular_harness


TASK = "spd-fixture"


class BankV2FailureScoringTests(unittest.TestCase):
    def _fixture(self, root: Path, version: str, node_name: str) -> tuple[Path, Path]:
        tasks = root / "tasks"
        tasks.mkdir()
        (tasks / "bank.json").write_text(json.dumps({"schema": "bank/1", "version": version}))
        node = root / "nodes" / node_name
        node.mkdir(parents=True)
        (node / "solution.py").write_text("def solve(): pass\n")
        return tasks, node

    def _speed(self, tasks: Path, node: Path, split: str, *, kind: str, host_error: bool = False) -> dict:
        meta = {"timed_out": kind == "timeout", "oom": kind == "oom",
                "exit_code": 1, "stderr_tail": "sandbox traceback"}
        payload = {"ok": False, "cheat_flags": [], "error":
                   {"kind": kind, "message": "candidate failed", "trace_tail": "trace"}}
        task = {"scenario": "speedup", "benchmark": {"dev_size": 2, "holdout_size": 2}}
        with patch.object(harness, "_tasks_root", return_value=tasks), \
                patch.object(harness, "_project_root", return_value=tasks.parent), \
                patch.object(harness, "load_task", return_value=task), \
                patch.object(harness, "_execute", side_effect=RuntimeError("host failed") if host_error else None,
                             return_value=(payload, meta)), \
                patch.dict(os.environ, {"RA_EVAL": "1"}):
            return harness.score(TASK, node, split, seed=13)

    def test_speedup_known_failures_score_zero_for_dev_and_holdout(self):
        for split in ("dev", "holdout"):
            for kind in ("timeout", "oom", "exception"):
                with self.subTest(split=split, kind=kind), TemporaryDirectory() as temp:
                    tasks, node = self._fixture(Path(temp), "bank-v2", "n0001")
                    result = self._speed(tasks, node, split, kind=kind)
                    self.assertEqual((result["schema"], result["ok"], result["score"], result["raw"], result["correct"]),
                                     ("score/1", True, 0.0, 0.0, False))
                    self.assertEqual(result["error"]["kind"], kind)
                    self.assertEqual(result["error"]["trace_tail"], "trace" if kind == "exception" else
                                     ("sandbox traceback" if kind == "oom" else ""))

    def test_speedup_baseline_old_bank_unknown_error_and_host_failure_abort(self):
        for version, node_name, kind in (("bank-v2", "n0000", "exception"),
                                         ("bank-v1", "n0001", "exception"),
                                         ("bank-v2", "n0001", "worker_protocol")):
            with self.subTest(version=version, node=node_name, kind=kind), TemporaryDirectory() as temp:
                tasks, node = self._fixture(Path(temp), version, node_name)
                result = self._speed(tasks, node, "dev", kind=kind)
                self.assertFalse(result["ok"])
                self.assertIsNone(result["score"])
                self.assertEqual(result["error"]["kind"], kind)
        with TemporaryDirectory() as temp:
            tasks, node = self._fixture(Path(temp), "bank-v2", "n0001")
            with self.assertRaisesRegex(RuntimeError, "host failed"):
                self._speed(tasks, node, "dev", kind="exception", host_error=True)
            (tasks / "bank.json").write_text(json.dumps({"schema": "invalid/1", "version": "bank-v2"}))
            with self.assertRaisesRegex(ValueError, "Expected bank/1"):
                self._speed(tasks, node, "dev", kind="exception")

    def test_tabular_known_failures_score_zero_once_for_dev_and_holdout(self):
        task = {"metric": {"baseline_raw": 0.5, "ceiling_raw": 0.9}}
        for split in ("dev", "holdout"):
            for kind in ("timeout", "oom", "exception"):
                with self.subTest(split=split, kind=kind), TemporaryDirectory() as temp:
                    tasks, node = self._fixture(Path(temp), "bank-v2", "n0001")
                    failure = {"ok": False, "raw": None, "correct": False, "cheat_flags": [],
                               "error": {"kind": kind, "message": "candidate failed", "trace_tail": "trace"},
                               "seconds": 0.2, "seed": 13}
                    with patch.object(harness, "_tasks_root", return_value=tasks), \
                            patch.object(harness, "load_task", return_value=task), \
                            patch.object(tabular_harness, "raw_score", return_value=failure) as worker, \
                            patch.dict(os.environ, {"RA_EVAL": "1"}):
                        result = tabular_harness.score(TASK, node, split, seed=13)
                    worker.assert_called_once_with(TASK, node, split, 13)
                    self.assertEqual((result["schema"], result["ok"], result["score"], result["raw"], result["correct"]),
                                     ("score/1", True, 0.0, 0.0, False))
                    self.assertEqual(result["error"], failure["error"])

    def test_tabular_baseline_old_bank_unknown_error_and_host_failure_abort(self):
        task = {"metric": {"baseline_raw": 0.5, "ceiling_raw": 0.9}}
        for version, node_name, kind in (("bank-v2", "n0000", "exception"),
                                         ("bank-v1", "n0001", "exception"),
                                         ("bank-v2", "n0001", "worker_protocol")):
            with self.subTest(version=version, node=node_name, kind=kind), TemporaryDirectory() as temp:
                tasks, node = self._fixture(Path(temp), version, node_name)
                failure = {"ok": False, "correct": False, "cheat_flags": [],
                           "error": {"kind": kind, "message": "failed"}, "seconds": 0.2}
                with patch.object(harness, "_tasks_root", return_value=tasks), \
                        patch.object(harness, "load_task", return_value=task), \
                        patch.object(tabular_harness, "raw_score", return_value=failure):
                    result = tabular_harness.score(TASK, node, "dev", seed=13)
                self.assertFalse(result["ok"])
                self.assertIsNone(result["score"])
        with TemporaryDirectory() as temp:
            tasks, node = self._fixture(Path(temp), "bank-v2", "n0001")
            with patch.object(harness, "_tasks_root", return_value=tasks), \
                    patch.object(harness, "load_task", return_value=task), \
                    patch.object(tabular_harness, "raw_score", side_effect=RuntimeError("host failed")):
                with self.assertRaisesRegex(RuntimeError, "host failed"):
                    tabular_harness.score(TASK, node, "dev", seed=13)


if __name__ == "__main__":
    unittest.main()
