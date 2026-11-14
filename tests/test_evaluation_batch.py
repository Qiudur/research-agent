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
                      quote_history_seen: list[tuple[str, int]] | None = None,
                      fail_holdout=None, over_budget_config: str | None = None):
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
            if config == over_budget_config:
                tokens = 2_000_000  # TEST price: cost 2.0 against the budget of 1.0
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
            failed = bool(fail_holdout and fail_holdout(config, rep, task_id))
            holdout = {"schema": "score/1", "split": "holdout", "score": None if failed else value,
                       "ok": not failed, "correct": not failed, "raw": None if failed else value,
                       "spread": 0.0, "cheat_flags": [],
                       "error": ({"kind": "exception", "message": "Scoring worker failed"}
                                 if failed else None), "seed": 1,
                       "seconds": {"run": 0.0, "queue": 0.0}}
            (task_dir / "holdout_score.json").write_text(json.dumps(holdout), encoding="utf-8")
            ledger.emit("task_end", best_node="n0000",
                        holdout_score=None if failed else value, holdout_failed=failed,
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
            first_wave = threading.Barrier(8)
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
                if kwargs["task_id"] == "spd-practice-a" and manifest["rep"] in (1, 2):
                    first_wave.wait(timeout=10)
                kwargs["harness"].score()
                return ordinary(**kwargs)

            result = self.run_batch(root, baselines, executor, max_workers=8,
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

            # Detected per run: every tampered validation run is a recorded failure
            # and the (incomplete) admission cannot be claimed from them.
            result = self.run_batch(root, baselines, tamper, run_exam=False)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(len(result["failed_runs"]), 6)
            self.assertTrue(all("Frozen library changed" in row["error"]
                                for row in result["failed_runs"]))
            self.assertEqual(result["admission"]["claim"], "incomplete")
            self.assertFalse((root / "batches/first/exam_attempt.json").exists())

    def run_failing(self, root, baselines, fail_if, *, ordinary=None, **kwargs):
        ordinary = ordinary or self.fake_executor([])

        def executor(**call):
            manifest = json.loads((call["run_dir"] / "manifest.json").read_text())
            if fail_if(TIERS[call["task_id"]], manifest["config"], manifest["rep"]):
                raise RuntimeError("solution.py vanished during holdout scoring")
            return ordinary(**call)

        return self.run_batch(root, baselines, executor, **kwargs)

    def test_one_failed_run_is_isolated_and_the_report_is_still_written(self):
        for workers in (1, 4):
            with self.subTest(workers=workers), TemporaryDirectory() as temp:
                root = Path(temp)
                baselines = self.setup_project(root)
                result = self.run_failing(
                    root, baselines,
                    lambda tier, config, rep: (tier, config, rep) == ("validation", "tuned_lib", 2),
                    max_workers=workers)
                written = json.loads((root / "batches/first/batch_report.json").read_text())
                self.assertEqual(written["status"], "partial")
                self.assertEqual(result["status"], "partial")
                [failure] = result["failed_runs"]
                self.assertEqual((failure["logical_phase"], failure["config"], failure["rep"]),
                                 ("validation", "tuned_lib", 2))
                self.assertIn("vanished", failure["error"])
                self.assertEqual(failure["error_type"], "RuntimeError")
                self.assertTrue(failure["run_id"].startswith("run-"))
                self.assertTrue((root / "runs" / failure["run_id"]).is_dir())  # evidence kept
                self.assertEqual([len(result["reports"][tier]) for tier in
                                  ("practice", "validation", "exam")], [12, 5, 9])
                # Admission needs all six validation runs; otherwise no claim is made.
                self.assertEqual(result["admission"]["claim"], "incomplete")
                self.assertEqual(result["admission"]["missing_runs"],
                                 [{"config": "tuned_lib", "rep": 2}])
                self.assertIsNone(result["admission"]["comparison"])
                self.assertTrue(result["tuned_exam_skipped"])
                self.assertNotIn("tuned_lib", {r["config"] for r in result["reports"]["exam"]})
                self.assertEqual(result["skipped_runs"], [])

    def test_practice_failure_skips_only_runs_that_need_its_library(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            result = self.run_failing(
                root, baselines,
                lambda tier, config, rep: (tier, config, rep) == ("practice", "default_lib", 1))
            self.assertEqual(result["status"], "partial")
            self.assertEqual([(row["logical_phase"], row["config"], row["rep"])
                              for row in result["failed_runs"]], [("practice", "default_lib", 1)])
            self.assertEqual({(row["logical_phase"], row["config"], row["rep"])
                              for row in result["skipped_runs"]},
                             {("validation", "default_lib", 1), ("exam", "default_lib", 1)})
            self.assertTrue(all("practice run" in row["reason"] for row in result["skipped_runs"]))
            self.assertEqual(result["admission"]["claim"], "incomplete")
            exam = {(r["config"], r["rep"]) for r in result["reports"]["exam"]}
            self.assertIn(("naive", 1), exam)       # independent lines keep going
            self.assertIn(("default", 1), exam)
            self.assertNotIn(("default_lib", 1), exam)

    def test_failed_untrained_practice_run_does_not_block_admission(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            result = self.run_failing(
                root, baselines,
                lambda tier, config, rep: (tier, config, rep) == ("practice", "naive", 3))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["admission"]["claim"], "admitted")
            self.assertEqual([len(result["reports"][tier]) for tier in
                              ("practice", "validation", "exam")], [11, 6, 12])
            self.assertEqual(result["skipped_runs"], [])

    def test_interrupt_still_writes_batch_report_before_propagating(self):
        class Interrupt(BaseException):
            pass

        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            ordinary = self.fake_executor([])

            def executor(**call):
                if TIERS[call["task_id"]] == "validation":
                    raise Interrupt("operator stop")
                return ordinary(**call)

            with self.assertRaises(Interrupt):
                self.run_batch(root, baselines, executor)
            written = json.loads((root / "batches/first/batch_report.json").read_text())
            self.assertEqual(written["status"], "partial")
            self.assertIn("Interrupt", written["aborted"])
            self.assertEqual(len(written["reports"]["practice"]), 12)
            self.assertTrue(written["tuned_exam_skipped"])

    def test_failed_holdout_is_listed_and_never_scored_as_zero(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            executor = self.fake_executor(
                [], fail_holdout=lambda config, rep, task: (
                    (config, rep, task) == ("tuned_lib", 1, "spd-validation-a")))
            result = self.run_batch(root, baselines, executor)
            self.assertEqual(result["status"], "complete")  # every run finished
            [failure] = result["holdout_failures"]
            self.assertEqual((failure["logical_phase"], failure["config"], failure["rep"],
                              failure["task_id"]), ("validation", "tuned_lib", 1, "spd-validation-a"))
            self.assertEqual(failure["error"]["kind"], "exception")
            # The only validation task is excluded from every run: no admission claim.
            admission = result["admission"]
            self.assertEqual(admission["claim"], "incomplete")
            self.assertEqual([item["task_id"] for item in admission["excluded_tasks"]],
                             ["spd-validation-a"])
            self.assertEqual(admission["excluded_tasks"][0]["failed_in"][0]["config"], "tuned_lib")
            self.assertTrue(result["tuned_exam_skipped"])
            row = next(r for r in result["reports"]["validation"]
                       if (r["config"], r["rep"]) == ("tuned_lib", 1))
            self.assertIsNone(row["tasks"][0]["holdout_score"])
            self.assertEqual(row["holdout_failures"], 1)

    def test_budget_violation_is_reported_in_the_batch_instead_of_stopping_it(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            result = self.run_batch(root, baselines,
                                    self.fake_executor([], over_budget_config="naive"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["budget_violations"]), 9)  # 3 reps x (2 practice + 1 exam)
            self.assertTrue(all(row["config"] == "naive" for row in result["budget_violations"]))
            self.assertAlmostEqual(result["budget_violations"][0]["excess"], 1.0)
            naive_exam = next(r for r in result["reports"]["exam"] if r["config"] == "naive")
            self.assertEqual(naive_exam["budget_violations"], 1)

    def test_complete_batch_reports_complete_status_with_no_failures(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baselines = self.setup_project(root)
            result = self.run_batch(root, baselines, self.fake_executor([]))
            self.assertEqual((result["status"], result["failed_runs"], result["skipped_runs"],
                              result["holdout_failures"], result["budget_violations"],
                              result["aborted"]), ("complete", [], [], [], [], None))
            self.assertEqual(result["admission"]["claim"], "admitted")

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
                    trap = "leakage" if prefix == "tab" and tier == "exam" and index == 1 else None
                    if prefix == "spd" and tier == "exam" and index == 0:
                        trap = "stop_loss"
                    rows.append({"task_id": f"{prefix}-{tier}-{index}", "tier": tier,
                                 "difficulty": "medium", "trap": trap,
                                 "score_formula": "gap" if prefix == "tab" else "ratio",
                                 "baseline_raw": 1.01177 if trap == "stop_loss" else 1.0,
                                 "ceiling_raw": 0.99599 if trap == "stop_loss" else 2.0,
                                 "naive_cost": 0.01})
        bank = {"schema": "bank/1", "version": "bank-v2", "frozen": True, "tasks": rows}
        by_id = {row["task_id"]: dict(row) for row in rows}

        class Harness:
            def load_task(self, task_id):
                return {"scenario": "tabular" if task_id.startswith("tab-") else "speedup",
                        "budget": {"money": 1.0},
                        "metric": {"baseline_raw": by_id[task_id]["baseline_raw"],
                                   "ceiling_raw": by_id[task_id]["ceiling_raw"]}}

        self.assertEqual(len(_bank_tasks(bank, Harness(), formal=True)["exam"]), 10)
        leakage = next(row for row in rows if row["trap"] == "leakage")
        leakage["trap"] = None
        with self.assertRaisesRegex(ValueError, "exam trap: tabular"):
            _bank_tasks(bank, Harness(), formal=True)
        leakage["trap"] = "leakage"
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
