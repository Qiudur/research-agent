"""C batch schedule uses real manifests, ledgers and audited run reports."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import threading
from tempfile import TemporaryDirectory
import time
import unittest

from evaluation.batch import _bank_tasks, _verify_code_freeze, execute_batch
from evaluation.figures import build_study, build_study_from_batch, render_html
from ledger import Ledger, Usage
from library import Library


PROJECT = Path(__file__).resolve().parents[1]
TASKS = ("spd-practice-a", "spd-practice-b", "spd-validation-a", "spd-exam-a")
TIERS = {task_id: task_id.split("-")[1] for task_id in TASKS}


class BatchTests(unittest.TestCase):
    def setup_project(self, root: Path) -> dict[str, Path]:
        (root / "config").mkdir()
        for name in ("agent.json", "hard_caps.json", "triggers.json"):
            shutil.copyfile(PROJECT / "config" / name, root / "config" / name)
        shutil.copyfile(PROJECT / "tests/fixtures/prices_test_only.json",
                        root / "config/prices_test_only.json")
        bank = {"schema": "bank/1", "version": "bank-v1", "frozen": False,
                "tasks": [{"task_id": task_id, "tier": TIERS[task_id]}
                          for task_id in TASKS]}
        (root / "tasks").mkdir()
        (root / "tasks/bank.json").write_text(json.dumps(bank), encoding="utf-8")
        baselines = {}
        for task_id in TASKS:
            workspace = root / "tasks" / task_id / "workspace"
            workspace.mkdir(parents=True)
            path = workspace / "original.py"
            path.write_text("def solve(): return 1" + chr(10), encoding="utf-8")
            baselines[task_id] = path
        return baselines

    def fake_executor(self, calls: list[tuple], *, tuned_score: float = 0.61,
                      emit_quote: bool = False,
                      quote_history_seen: list[tuple[str, int]] | None = None):
        def execute(*, run_dir, task_id, harness, baseline_solution,
                    test_only_budget, **kwargs):
            if quote_history_seen is not None:
                quote_history_seen.append((task_id, len(kwargs["quote_history"])))
            self.assertEqual(test_only_budget, 1.0)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            tier = TIERS[task_id]
            self.assertEqual(manifest["phase"], "dev")
            self.assertEqual(Path(baseline_solution).name, "original.py")
            config, rep = manifest["config"], manifest["rep"]
            calls.append((tier, config, rep, task_id, manifest["library"]["start_version"]))
            library_root = run_dir / "library"
            library = Library.open(library_root, frozen=manifest["library"]["frozen"])
            if tier == "practice" and task_id == "spd-practice-a":
                library.write_experiences([{
                    "scope": "method", "scenario": "speedup",
                    "triggers": ["nested_python_loop"], "conditions": "nested loop",
                    "action": "vectorize", "negative": False,
                }], source={"run_id": run_dir.name, "task_id": task_id,
                            "node_ids": ["n0000"]})
            if tier != "practice":
                self.assertTrue(library.frozen)
                self.assertEqual(library.version, "lib-1" if config.endswith("_lib") else "lib-0")
            task_dir = run_dir / "tasks" / task_id
            node = task_dir / "nodes/n0000"
            node.mkdir(parents=True)
            shutil.copyfile(baseline_solution, node / "solution.py")
            ledger = Ledger.open(run_dir, task_id)
            ledger.emit("task_start", budget=1, wall_seconds=60)
            tokens = 100 if config == "tuned_lib" else 200
            ledger.llm_call(node_id=None, role="planner", purpose="plan", effort="high",
                            model="test-scripted", usage=Usage(input=tokens),
                            seconds=0.1, prompt_hash="a" * 64)
            if emit_quote:
                ledger.emit("quote", predicted={"cost": 0.02, "seconds": 1.0, "score": 0.6},
                            basis={"source": "test_fixture"})
            value = tuned_score if config == "tuned_lib" else 0.60
            root_node = {"schema": "node/1", "node_id": "n0000", "parent": None,
                         "hypothesis": "Baseline", "status": "accepted",
                         "score": {"dev": value}, "cost": 0, "failure": None, "cites": []}
            (task_dir / "tree.jsonl").write_text(json.dumps(root_node) + chr(10), encoding="utf-8")
            ledger.emit("stop", reason="planner", best_node="n0000", dev_score=value)
            holdout = {"schema": "score/1", "split": "holdout", "score": value,
                       "ok": True, "correct": True, "raw": value, "spread": 0.0,
                       "cheat_flags": [], "error": None, "seed": 1,
                       "seconds": {"run": 0.0, "queue": 0.0}}
            (task_dir / "holdout_score.json").write_text(json.dumps(holdout), encoding="utf-8")
            ledger.emit("task_end", best_node="n0000", holdout_score=value,
                        total_cost=ledger.spent(), seconds={"llm": 0.1, "run": 0,
                                                             "queue": 0, "wall": 1},
                        quote_error={"cost": None, "seconds": None, "score": None})
            return {}
        return execute

    def run_batch(self, root: Path, baselines: dict[str, Path], executor, *,
                  run_validation=True, run_exam=True, max_workers=1, harness=None):
        configs = {name: root / "config/agent.json"
                   for name in ("naive", "default", "default_lib", "tuned_lib")}
        return execute_batch(
            project_root=root, batch_dir=root / "batches/first",
            harness=harness if harness is not None else object(),
            baseline_solutions=baselines, agent_configs=configs,
            prices_path=root / "config/prices_test_only.json", model="test-scripted",
            code_commit="a" * 40, run_validation=run_validation,
            run_exam=run_exam, max_workers=max_workers, test_only_budget=1.0,
            task_executor=executor,
        )

    def test_parallel_runs_preserve_order_and_serialize_sandbox_calls(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            calls: list[tuple] = []
            ordinary = self.fake_executor(calls)
            first_wave = threading.Barrier(4)
            counter_lock = threading.Lock()
            active = 0
            peak = 0

            class ProbeHarness:
                def score(self):
                    nonlocal active, peak
                    with counter_lock:
                        active += 1
                        peak = max(peak, active)
                    time.sleep(0.01)
                    with counter_lock:
                        active -= 1

            def executor(**kwargs):
                manifest = json.loads((kwargs["run_dir"] / "manifest.json").read_text())
                if kwargs["task_id"] == "spd-practice-a" and manifest["rep"] == 1:
                    first_wave.wait(timeout=10)
                kwargs["harness"].score()
                return ordinary(**kwargs)

            result = self.run_batch(root, baselines, executor, max_workers=4,
                                    harness=ProbeHarness())
            self.assertEqual(peak, 1)
            self.assertEqual([len(result["reports"][tier]) for tier in
                              ("practice", "validation", "exam")], [12, 6, 12])
            self.assertEqual([(row["rep"], row["config"]) for row in result["runs"][:4]],
                             [(1, name) for name in
                              ("naive", "default", "default_lib", "tuned_lib")])
            self.assertEqual(result["admission"]["claim"], "admitted")

    def test_three_repetitions_and_frozen_snapshots_make_audited_reports(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            calls: list[tuple] = []
            result = self.run_batch(root, baselines, self.fake_executor(calls))
            self.assertEqual(result["label"], "TEST ONLY")
            self.assertEqual(result["admission"]["claim"], "admitted")
            self.assertEqual([len(result["reports"][tier]) for tier in ("practice", "validation", "exam")],
                             [12, 6, 12])
            self.assertEqual(len(calls), 42)
            self.assertEqual([x[3] for x in calls[:2]], ["spd-practice-a", "spd-practice-b"])
            self.assertTrue(all(row["manifest_phase"] == "dev" and row["test_only"]
                                for row in result["runs"]))
            self.assertEqual({row["library_start_version"] for row in result["runs"]
                              if row["logical_phase"] == "validation"}, {"lib-1"})
            self.assertTrue(all(report["schema"] == "run_report/1"
                                and report["evidence"]["run_end_seq"] == 2
                                for rows in result["reports"].values() for report in rows))
            study = build_study(result["reports"]["exam"])
            self.assertTrue(study["test_only"])
            self.assertEqual(len(study["task_rows"]), 12)
            self.assertTrue((root / "batches/first/exam_attempt.json").is_file())
            with self.assertRaises(FileExistsError):
                self.run_batch(root, baselines, self.fake_executor([]))

    def test_rejected_tuning_never_reaches_exam(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            calls: list[tuple] = []
            result = self.run_batch(root, baselines, self.fake_executor(calls, tuned_score=0.59))
            self.assertEqual(result["admission"]["claim"], "rejected")
            self.assertTrue(result["tuned_exam_skipped"])
            self.assertEqual(len(result["reports"]["exam"]), 9)
            self.assertNotIn("tuned_lib", {r["config"] for r in result["reports"]["exam"]})
            study = build_study_from_batch(root / "batches/first")
            self.assertEqual(set(study["configurations"]), {"naive", "default", "default_lib"})
            self.assertNotIn("self_research", study["comparisons"])
            self.assertFalse(study["breakeven"]["eligible"])
            self.assertIsNone(study["breakeven"]["tasks"])
            self.assertIn("未通过验证准入", render_html(study))

    def test_practice_only_does_not_touch_validation_or_exam(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            calls: list[tuple] = []
            result = self.run_batch(root, baselines, self.fake_executor(calls),
                                    run_validation=False, run_exam=False)
            self.assertEqual([len(result["reports"][tier]) for tier in
                              ("practice", "validation", "exam")], [12, 0, 0])
            self.assertIsNone(result["admission"])
            self.assertFalse((root / "batches/first/exam_attempt.json").exists())

    def test_practice_quote_history_stays_within_scenario_and_run(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            seen: list[tuple[str, int]] = []
            self.run_batch(root, baselines,
                           self.fake_executor([], emit_quote=True, quote_history_seen=seen),
                           run_validation=False, run_exam=False)
            self.assertEqual(seen.count(("spd-practice-a", 0)), 12)
            self.assertEqual(seen.count(("spd-practice-b", 1)), 12)

    def test_frozen_library_file_mutation_is_detected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            ordinary = self.fake_executor([])

            def tamper(**kwargs):
                ordinary(**kwargs)
                if TIERS[kwargs["task_id"]] == "validation":
                    (kwargs["run_dir"] / "library/tampered.txt").write_text("x")

            with self.assertRaisesRegex(ValueError, "Frozen library changed"):
                self.run_batch(root, baselines, tamper, run_exam=False)
            self.assertFalse((root / "batches/first/exam_attempt.json").exists())

    def test_missing_formal_inputs_fail_before_any_run(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            with self.assertRaisesRegex(ValueError, "Formal task|frozen bank"):
                execute_batch(
                    project_root=root, batch_dir=root / "batches/first", harness=object(),
                    baseline_solutions=baselines,
                    agent_configs={"default_lib": root / "config/agent.json"},
                    prices_path=root / "config/prices_test_only.json", model="test-scripted",
                    code_commit="a" * 40, configs=("default_lib",),
                )
            self.assertFalse((root / "batches/first").exists())

    def test_formal_stop_loss_accepts_measured_no_headroom(self):
        rows = []
        for prefix in ("tab", "spd"):
            for tier, number in (("practice", 5), ("validation", 2), ("exam", 5)):
                for index in range(number):
                    trap = ("leakage" if prefix == "tab" else "cheat_bait") if tier == "exam" and index == 1 else None
                    if prefix == "spd" and tier == "exam" and index == 0:
                        trap = "stop_loss"
                    rows.append({"task_id": f"{prefix}-{tier}-{index}", "tier": tier,
                                 "difficulty": "medium", "trap": trap,
                                 "score_formula": "gap" if prefix == "tab" else "ratio",
                                 "baseline_raw": 1.01177 if trap == "stop_loss" else 1.0,
                                 "ceiling_raw": 0.99599 if trap == "stop_loss" else 2.0,
                                 "naive_cost": 0.01})
        bank = {"schema": "bank/1", "version": "bank-v1", "frozen": True, "tasks": rows}
        by_id = {row["task_id"]: dict(row) for row in rows}

        class Harness:
            def load_task(self, task_id):
                return {"scenario": "tabular" if task_id.startswith("tab-") else "speedup",
                        "budget": {"money": 1.0},
                        "metric": {"baseline_raw": by_id[task_id]["baseline_raw"],
                                   "ceiling_raw": by_id[task_id]["ceiling_raw"]}}

        self.assertEqual(len(_bank_tasks(bank, Harness(), formal=True)["exam"]), 10)
        tab_row = next(row for row in rows if row["task_id"].startswith("tab-"))
        tab_row["ceiling_raw"] += 0.01
        with self.assertRaisesRegex(ValueError, "differs from bank"):
            _bank_tasks(bank, Harness(), formal=True)
        tab_row["ceiling_raw"] -= 0.01
        stop_loss = next(row for row in rows if row["trap"] == "stop_loss")
        stop_loss["trap"] = "cheat_bait"
        with self.assertRaisesRegex(ValueError, "invalid measurements"):
            _bank_tasks(bank, Harness(), formal=True)

    def test_formal_code_freeze_checks_head_and_dirty_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "tracked.py").write_text("value = 1" + chr(10))
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                            "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)
            head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            _verify_code_freeze(root, head)
            with self.assertRaisesRegex(ValueError, "does not match"):
                _verify_code_freeze(root, "a" * 40)
            (root / "untracked.py").write_text("value = 2" + chr(10))
            with self.assertRaisesRegex(ValueError, "clean Git worktree"):
                _verify_code_freeze(root, head)


if __name__ == "__main__":
    unittest.main()
