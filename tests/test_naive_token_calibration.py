from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

import harness
from scripts import calibrate_naive_tokens as calibration
from scripts.calibrate_naive_tokens import (BUDGET_TOKENS, DevOnlyHarness,
                                            _bank_ids, _run_one, _verify_inputs,
                                            prepare_project)


PROJECT = Path(__file__).resolve().parents[1]


class FakeHarness:
    def __init__(self):
        self.splits = []

    def load_task(self, task_id):
        return harness.load_task(task_id)

    def score(self, task_id, node_dir, split, **kwargs):
        self.splits.append(split)
        return {"schema": "score/1", "ok": True, "split": split,
                "score": 0.5, "raw": 0.5, "spread": 0.0,
                "correct": True, "cheat_flags": [], "error": None,
                "seconds": {"run": 0.0, "queue": 0.0}, "seed": 1}


class NoWorkRoles:
    def plan(self, payload, **kwargs):
        return {"hypotheses": []}

    def summarize(self, payload, **kwargs):
        return {"drafts": [], "narrowings": []}


class NaiveTokenCalibrationTests(unittest.TestCase):
    def _source_with_tasks(self, temp: str, task_ids: tuple[str, ...]) -> Path:
        source = Path(temp) / "source"
        shutil.copytree(PROJECT / "config", source / "config")
        (source / "tasks").mkdir()
        for task_id in task_ids:
            shutil.copytree(PROJECT / "tasks" / task_id, source / "tasks" / task_id)
        bank = json.loads((PROJECT / "tasks/bank.json").read_text())
        bank["tasks"] = [entry for entry in bank["tasks"] if entry["task_id"] in task_ids]
        (source / "tasks/bank.json").write_text(json.dumps(bank))
        return source

    def test_bank_selection_covers_all_24_tasks(self):
        entries, chosen = _bank_ids(PROJECT, None)
        self.assertEqual(len(chosen), 24)
        self.assertEqual(len(entries), 24)
        self.assertEqual({row["tier"] for row in entries},
                         {"practice", "validation", "exam"})
        with self.assertRaises(ValueError):
            _bank_ids(PROJECT, ("spd-membership-filter", "spd-membership-filter"))

    def test_scenario_selects_twelve_and_excludes_explicit_task(self):
        for scenario, prefix in (("speedup", "spd-"), ("tabular", "tab-")):
            captured = []

            def capture(_source, _campaign, **kwargs):
                captured.extend(kwargs["task_ids"])
                raise RuntimeError("selection captured")

            with (patch.object(calibration.subprocess, "check_output",
                               side_effect=["", "a" * 40]),
                  patch.object(calibration, "prepare_project", side_effect=capture),
                  patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"})):
                with self.assertRaisesRegex(RuntimeError, "selection captured"):
                    calibration.main(["--scenario", scenario])
            self.assertEqual(len(captured), 12)
            self.assertTrue(all(task.startswith(prefix) for task in captured))
        with self.assertRaises(SystemExit):
            calibration.main(["--scenario", "speedup",
                              "--task", "spd-membership-filter"])

    def test_copy_excludes_tabular_holdout_and_preserves_source(self):
        selected = ("spd-membership-filter", "tab-breast-cancer-diagnostic",
                    "tab-bank-marketing")
        before = {task: (PROJECT / "tasks" / task / "task.json").read_bytes()
                  for task in selected}
        with TemporaryDirectory() as temp:
            campaign = Path(temp) / "campaign"
            project = prepare_project(PROJECT, campaign, task_ids=selected, repetitions=1)
            protocol = json.loads((campaign / "protocol.json").read_text())
            self.assertEqual(protocol["tasks"], list(selected))
            self.assertEqual(protocol["repetitions"], 1)
            self.assertEqual(protocol["budget_tokens_per_task"], BUDGET_TOKENS)
            self.assertEqual(len(protocol["copied_sha256"]), len(set(protocol["copied_sha256"])))
            for tabular in ("tab-breast-cancer-diagnostic", "tab-bank-marketing"):
                self.assertTrue((project / "tasks" / tabular / "hidden/dev.csv").is_file())
                self.assertFalse((project / "tasks" / tabular / "hidden/holdout.csv").exists())
            self.assertTrue((project / "tasks/spd-membership-filter/hidden/reference.py").is_file())
            self.assertEqual([p.name for p in
                              (project / "tasks/tab-bank-marketing/hidden").iterdir()], ["dev.csv"])
            for task in selected:
                self.assertEqual((PROJECT / "tasks" / task / "task.json").read_bytes(), before[task])
                copied = json.loads((project / "tasks" / task / "task.json").read_text())
                self.assertEqual(copied["budget"]["money"], BUDGET_TOKENS)
            _verify_inputs(project, protocol)
            (project / "config/naive_token_pilot.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "input changed"):
                _verify_inputs(project, protocol)

    def test_dev_only_guard_rejects_holdout_and_run_records_no_attempt(self):
        fake = FakeHarness()
        guard = DevOnlyHarness(fake)
        with self.assertRaises(PermissionError):
            guard.score("spd-already-vectorized", Path("/tmp/none"), "holdout")
        self.assertEqual(fake.splits, [])
        with TemporaryDirectory() as temp:
            campaign = Path(temp) / "campaign"
            project = prepare_project(PROJECT, campaign,
                                      task_ids=("spd-already-vectorized",), repetitions=1)
            row = _run_one(project, "spd-already-vectorized", 1, "a" * 40,
                           roles_factory=lambda **_: NoWorkRoles(), base_harness=fake)
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["total_tokens"], 0)
            self.assertEqual(row["stop_reason"], "planner")
            self.assertEqual(fake.splits, ["dev"])
            run = Path(row["run_dir"])
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["phase"], "dev")
            self.assertFalse((run / "tasks/spd-already-vectorized/holdout_attempt.json").exists())
            self.assertFalse((run / "tasks/spd-already-vectorized/holdout_score.json").exists())
            self.assertTrue(Path(row["replay_context_path"]).is_file())

    def test_failure_records_run_path_without_leaking_key(self):
        task_id = "spd-membership-filter"
        with TemporaryDirectory() as temp:
            source = self._source_with_tasks(temp, (task_id,))
            with (patch.object(calibration, "PROJECT", source),
                  patch.object(calibration.subprocess, "check_output",
                               side_effect=["", "a" * 40]),
                  patch.object(calibration, "_run_one",
                               side_effect=RuntimeError("gateway rejected test-secret")),
                  patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"})):
                result = calibration.main(["--task", task_id, "--repetitions", "1"])
            self.assertEqual(result, 1)
            campaigns = list((source / "runs").glob("naive-token-calibration-*"))
            self.assertEqual(len(campaigns), 1)
            raw = (campaigns[0] / "summary.json").read_text()
            self.assertNotIn("test-secret", raw)
            attempt = json.loads(raw)["attempts"][0]
            self.assertEqual(attempt["status"], "failed")
            self.assertIn("run_dir", attempt)
            self.assertEqual(attempt["error"], "gateway rejected [REDACTED]")

    def test_parallel_repetitions_keep_task_barrier_and_rep_order(self):
        chosen = ("spd-membership-filter", "tab-bank-marketing")
        barriers = {task: threading.Barrier(3) for task in chosen}
        lock = threading.Lock()
        active = set()
        peak = 0
        mixed_tasks = False
        wrappers = set()

        def fake_run(project, task_id, rep, code_commit, **kwargs):
            nonlocal peak, mixed_tasks
            self.assertEqual(os.environ["RA_PROJECT_ROOT"], str(project))
            self.assertEqual(os.environ["RA_TASKS_ROOT"], str(project / "tasks"))
            with lock:
                mixed_tasks |= any(other_task != task_id for other_task, _ in active)
                active.add((task_id, rep))
                peak = max(peak, len(active))
                wrappers.add(id(kwargs["base_harness"]))
            barriers[task_id].wait(timeout=5)
            time.sleep((4 - rep) * 0.01)
            with lock:
                active.remove((task_id, rep))
            return {"status": "completed", "total_tokens": rep}

        with TemporaryDirectory() as temp:
            source = self._source_with_tasks(temp, chosen)
            with (patch.object(calibration, "PROJECT", source),
                  patch.object(calibration.subprocess, "check_output",
                               side_effect=["", "a" * 40]),
                  patch.object(calibration, "_run_one", side_effect=fake_run),
                  patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"})):
                result = calibration.main(["--task", chosen[0], "--task", chosen[1],
                                           "--repetitions", "3", "--workers", "3"])
            self.assertEqual(result, 0)
            campaign = next((source / "runs").glob("naive-token-calibration-*"))
            protocol = json.loads((campaign / "protocol.json").read_text())
            attempts = json.loads((campaign / "summary.json").read_text())["attempts"]
            self.assertEqual(protocol["max_workers"], 3)
            self.assertEqual([(row["task_id"], row["rep"]) for row in attempts],
                             [(task, rep) for task in chosen for rep in (1, 2, 3)])
            self.assertFalse(mixed_tasks)
            self.assertEqual(peak, 3)
            self.assertEqual(len(wrappers), 1)
            self.assertTrue(all(row["status"] == "completed" for row in attempts))

    def test_dev_harness_serializes_parallel_scores_and_denies_holdout(self):
        class SlowBase:
            def __init__(self):
                self.active = 0
                self.peak = 0

            def score(self, *args, **kwargs):
                self.active += 1
                self.peak = max(self.peak, self.active)
                time.sleep(0.01)
                self.active -= 1
                return {}

        from concurrent.futures import ThreadPoolExecutor
        base = SlowBase()
        guarded = DevOnlyHarness(base, threading.Lock())
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(guarded.score, "spd-demo", Path("/tmp/none"), "dev")
                       for _ in range(3)]
            for future in futures:
                future.result()
        self.assertEqual(base.peak, 1)
        with self.assertRaises(PermissionError):
            guarded.score_holdout("spd-demo", Path("/tmp/none"))


if __name__ == "__main__":
    unittest.main()
