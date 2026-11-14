from __future__ import annotations

import hashlib
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
from ledger import Ledger, Usage
from library import Library


PROJECT = Path(__file__).resolve().parents[1]
TASK = "spd-distance-matrix"


def setup(root: Path, *, best_node: str = "n0000", best: str | None = None) -> Path:
    if best is not None:
        best_node = best
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
    for node_id in dict.fromkeys(["n0000", best_node]):
        node = run / "tasks" / TASK / "nodes" / node_id
        node.mkdir(parents=True)
        (node / "solution.py").write_text(f"def f(): return '{node_id}'\n")
        (node / "score.json").write_text("{}\n")
    ledger = Ledger.open(run, TASK)
    ledger.emit("task_start", budget=1, wall_seconds=60)
    ledger.emit("quote", predicted={"cost": 0.1, "seconds": 2, "score": 0.5}, basis={"source": "test"})
    ledger.emit("stop", reason="planner", best_node=best_node, dev_score=0)
    return run


def write_tree(run: Path, *node_ids: str) -> None:
    rows = [{"schema": "node/1", "node_id": node_id, "parent": None if node_id == "n0000" else "n0000",
             "hypothesis": "Baseline", "status": "accepted", "score": {"dev": 0},
             "cost": 0, "failure": None, "cites": []} for node_id in node_ids]
    (run / "tasks" / TASK / "tree.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def seal(run: Path, hashes: dict[str, str] | None = None) -> None:
    """Write the node hashes the way write_replay_context seals them."""
    nodes = {}
    for node in (run / "tasks" / TASK / "nodes").iterdir():
        digest = hashlib.sha256((node / "solution.py").read_bytes()).hexdigest()
        nodes[node.name] = {"solution_sha256": (hashes or {}).get(node.name, digest)}
    (run / "tasks" / TASK / "replay_context.json").write_text(
        json.dumps({"schema": "a_replay_context/1", "nodes": nodes}))


def saved(run: Path, name: str) -> dict:
    return json.loads((run / "tasks" / TASK / name).read_text())


def snapshot(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


GOOD = {"schema": "score/1", "ok": True, "split": "holdout", "score": 0.7,
        "raw": 2.0, "spread": 0.1, "correct": True, "cheat_flags": [], "error": None,
        "seconds": {"run": 0.2, "queue": 0.1}, "seed": 1234}


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
        return dict(GOOD)


class ReturnedHarness:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def score(self, task_id, node_dir, split):
        self.calls += 1
        assert os.environ.get("RA_EVAL") == "1"
        assert task_id == TASK and node_dir.name == "n0000" and split == "holdout"
        return self.result


class PerNodeHarness:
    """Scores each node differently; a node listed in ``fail`` raises."""

    def __init__(self, results: dict[str, tuple[float, float]], fail: tuple[str, ...] = ()):
        self.results = results
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def score(self, task_id, node_dir, split):
        assert os.environ.get("RA_EVAL") == "1" and split == "holdout"
        self.calls.append((node_dir.name, node_dir.parent.name))
        if node_dir.name in self.fail:
            raise RuntimeError("scorer exploded")
        score, raw = self.results[node_dir.name]
        return {**GOOD, "score": score, "raw": raw}


class MeddlingHarness:
    """Simulates a candidate run that touches files while it is being scored."""

    def __init__(self, action):
        self.action = action
        self.calls = 0
        self.scored_dirs: list[Path] = []

    def score(self, task_id, node_dir, split):
        self.calls += 1
        self.scored_dirs.append(Path(node_dir))
        self.action(Path(node_dir))
        return dict(GOOD)


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
            self.assertEqual(worker.call_count, 2)
            self.assertEqual([call.args[1].name for call in worker.call_args_list],
                             ["n0001", "n0000"])
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
                    assert node_dir.name in {"n0001", "n0000"} and split == "holdout"
                    if node_dir.name == "n0000":
                        return {**GOOD, "score": 0.1, "raw": 0.1}
                    return {"schema": "score/1", "ok": True, "split": "holdout",
                            "score": 0.0, "raw": 0.0, "spread": 0.0, "correct": False,
                            "cheat_flags": [], "error": {"kind": "oom", "message": "candidate OOM"},
                            "seconds": {"run": 0.2, "queue": 0.0}, "seed": 13}

            scorer = ZeroHarness()
            result = finish_task(run, TASK, scorer)
            self.assertEqual(scorer.calls, 2)
            self.assertEqual(result["task_end"]["holdout_score"], 0.0)
            self.assertEqual(json.loads((task_dir / "holdout_score.json").read_text())["error"]["kind"], "oom")
            with self.assertRaisesRegex(ValueError, "already recorded"):
                finish_task(run, TASK, scorer)
            self.assertEqual(scorer.calls, 2)
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
            self.assertFalse(result["task_end"]["holdout_failed"])
            self.assertEqual(result["task_end"]["total_cost"], 0)
            self.assertAlmostEqual(result["task_end"]["quote_error"]["score"], 0.2)
            with self.assertRaisesRegex(ValueError, "already recorded"):
                finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, 1)
            finished = finish_run(run)
            self.assertEqual(finished["total_cost"], 0)
            self.assertIsNotNone(json.loads((run / "manifest.json").read_text())["ended_ts"])
            self.assertEqual([event["kind"] for event in Ledger.open(run, None)._events()], ["run_start", "run_end"])

    def test_harness_exception_is_recorded_not_raised_and_not_retried(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            harness = FakeHarness(fail=True)
            result = finish_task(run, TASK, harness)  # the run must not crash
            self.assertNotIn("RA_EVAL", os.environ)
            end = result["task_end"]
            self.assertIs(end["holdout_failed"], True)
            self.assertIsNone(end["holdout_score"])
            self.assertIsNone(end["quote_error"]["score"])
            self.assertTrue((run / "tasks" / TASK / "holdout_attempt.json").exists())
            record = saved(run, "holdout_score.json")
            self.assertEqual((record["schema"], record["split"], record["ok"], record["score"]),
                             ("score/1", "holdout", False, None))
            self.assertEqual((record["error"]["kind"], record["error"]["type"]),
                             ("harness_exception", "RuntimeError"))
            self.assertNotIn("backend failed", json.dumps(record))  # free text is not retained
            self.assertEqual(saved(run, "holdout_failure.json")["reason"], "exception")
            with self.assertRaisesRegex(ValueError, "already recorded"):
                finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, 1)
            self.assertEqual(finish_run(run)["total_cost"], 0)

    def test_harness_exception_text_is_allowlisted(self):
        cases = [
            (PermissionError("RA_EVAL=1 is required for holdout"), "PermissionError",
             "RA_EVAL=1 is required for holdout"),
            (FileNotFoundError(2, "No such file or directory", "/runs/x/nodes/n0003/solution.py"),
             "FileNotFoundError", "No such file or directory /runs/x/nodes/n0003/solution.py"),
            (ValueError("hidden-label-value prediction-secret"), "ValueError",
             "Holdout scoring raised an exception"),
        ]
        for exc, expected_type, expected_message in cases:
            with self.subTest(kind=expected_type), TemporaryDirectory() as temp:
                run = setup(Path(temp))

                class Raising:
                    def score(self, task_id, node_dir, split):
                        raise exc

                finish_task(run, TASK, Raising())
                error = saved(run, "holdout_score.json")["error"]
                self.assertEqual(error, {"kind": "harness_exception", "type": expected_type,
                                         "message": expected_message})
                self.assertNotIn("hidden-label-value",
                                 (run / "tasks" / TASK / "holdout_failure.json").read_text())

    def test_attempt_marker_without_task_end_is_never_retried(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            (run / "tasks" / TASK / "holdout_attempt.json").write_text("{}\n")
            harness = FakeHarness()
            with self.assertRaises(FileExistsError):
                finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, 0)

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
                outcome = finish_task(run, TASK, harness)
                self.assertTrue(outcome["task_end"]["holdout_failed"])
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
                record = saved(run, "holdout_score.json")
                self.assertIsNone(record["score"])
                self.assertEqual(record["error"], failure["error"])
                for filename in ("holdout_failure.json", "holdout_score.json",
                                 "holdout_integrity.json", "baseline_holdout.json"):
                    contents = (task_dir / filename).read_text()
                    for secret in ("hidden-label-value", "prediction-secret", "trace-secret"):
                        self.assertNotIn(secret, contents)
                self.assertFalse((task_dir / "holdout_failure.json.tmp").exists())
                self.assertEqual(sum(event["kind"] == "task_end"
                                     for event in Ledger.open(run, TASK)._events()), 1)
                with self.assertRaisesRegex(ValueError, "already recorded"):
                    finish_task(run, TASK, harness)
                self.assertEqual(harness.calls, 1)

    def test_invalid_scorer_return_is_a_recorded_failure(self):
        for name, value in (("none", None), ("wrong_schema", {"schema": "other", "split": "holdout"}),
                            ("wrong_split", {**GOOD, "split": "dev"}),
                            ("nan_score", {**GOOD, "score": float("nan")}),
                            ("bool_score", {**GOOD, "score": True})):
            with self.subTest(name=name), TemporaryDirectory() as temp:
                run = setup(Path(temp))
                outcome = finish_task(run, TASK, ReturnedHarness(value))
                self.assertTrue(outcome["task_end"]["holdout_failed"])
                self.assertIsNone(saved(run, "holdout_score.json")["score"])
                if name in {"none", "wrong_schema", "wrong_split"}:
                    self.assertEqual(saved(run, "holdout_failure.json")["error"]["kind"],
                                     "invalid_result")

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
            write_tree(run, "n0000")
            finish_task(run, TASK, FakeHarness())
            finish_run(run)
            report = run_report(run)
            self.assertEqual(report["tasks"][0]["evidence"]["task_end_seq"], 4)
            self.assertEqual(report["evidence"]["run_end_seq"], 2)
            self.assertEqual(report["tasks"][0]["model_calls"], 0)
            self.assertEqual(task_report(run, TASK)["holdout_score"], 0.7)
            self.assertEqual(extract_summary(run, TASK)["experiments"], [])
            self.assertTrue((run / "report.json").exists())
            self.assertEqual((report["holdout_failures"], report["budget_violations"]), (0, 0))


class PrivateCopyTests(unittest.TestCase):
    def test_holdout_scores_a_private_copy_and_leaves_the_node_untouched(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            node = run / "tasks" / TASK / "nodes" / "n0000"
            before = snapshot(node)
            harness = MeddlingHarness(lambda scored: (scored / "scratch.tmp").write_text("junk"))
            finish_task(run, TASK, harness)
            copy = (run / "tasks" / TASK / "holdout_candidate" / "n0000").resolve()
            self.assertEqual(harness.scored_dirs, [copy])
            self.assertNotEqual(copy, node)
            self.assertEqual((copy / "solution.py").read_bytes(), (node / "solution.py").read_bytes())
            self.assertEqual(snapshot(node), before)  # no scratch file reached the sealed node
            self.assertTrue((copy / "scratch.tmp").exists())
            integrity = saved(run, "holdout_integrity.json")
            self.assertEqual((integrity["status"], integrity["node"]), ("ok", "n0000"))
            self.assertEqual(integrity["solution_sha256"], before["solution.py"])

    def test_matching_seal_is_accepted_and_mismatch_blocks_scoring(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            seal(run)
            harness = FakeHarness()
            outcome = finish_task(run, TASK, harness)
            self.assertFalse(outcome["task_end"]["holdout_failed"])
            self.assertEqual(saved(run, "holdout_integrity.json")["sealed_sha256"],
                             saved(run, "holdout_integrity.json")["solution_sha256"])
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            seal(run, {"n0000": "0" * 64})
            node = run / "tasks" / TASK / "nodes" / "n0000"
            before = snapshot(node)
            harness = FakeHarness()
            outcome = finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, 0)  # never scored what was not sealed
            self.assertTrue(outcome["task_end"]["holdout_failed"])
            self.assertEqual(saved(run, "holdout_score.json")["error"]["kind"], "integrity")
            self.assertEqual(saved(run, "holdout_failure.json")["reason"], "integrity")
            self.assertEqual(snapshot(node), before)
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            (run / "tasks" / TASK / "replay_context.json").write_text("not json")
            harness = FakeHarness()
            outcome = finish_task(run, TASK, harness)
            self.assertEqual((harness.calls, outcome["task_end"]["holdout_failed"]), (0, True))

    def test_missing_solution_is_recorded_not_raised(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            (run / "tasks" / TASK / "nodes" / "n0000" / "solution.py").unlink()
            harness = FakeHarness()
            outcome = finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, 0)
            self.assertTrue(outcome["task_end"]["holdout_failed"])
            self.assertTrue((run / "tasks" / TASK / "holdout_attempt.json").exists())
            self.assertIn("missing", saved(run, "holdout_score.json")["error"]["message"])
            self.assertEqual(saved(run, "holdout_integrity.json")["status"], "error: integrity")

    def test_changes_during_scoring_are_recorded_in_the_result(self):
        node_name = "n0000"

        def delete_original(scored):
            (scored.parents[1] / "nodes" / node_name / "solution.py").unlink()

        def edit_original(scored):
            with (scored.parents[1] / "nodes" / node_name / "solution.py").open("a") as handle:
                handle.write("# edited\n")

        def edit_copy(scored):
            (scored / "solution.py").write_text("def f(): return 'tampered'\n")

        for name, action, fragment in (
                ("original deleted", delete_original, "vanished"),
                ("original edited", edit_original, "changed"),
                ("copy edited", edit_copy, "copy changed")):
            with self.subTest(name=name), TemporaryDirectory() as temp:
                run = setup(Path(temp))
                harness = MeddlingHarness(action)
                outcome = finish_task(run, TASK, harness)  # no crash
                self.assertEqual(harness.calls, 1)
                self.assertTrue(outcome["task_end"]["holdout_failed"])
                self.assertIsNone(outcome["task_end"]["holdout_score"])
                record = saved(run, "holdout_score.json")
                self.assertIsNone(record["score"])
                self.assertEqual(record["error"]["kind"], "integrity")
                self.assertIn(fragment, record["error"]["message"])
                finish_run(run)  # the run still closes


class BaselineReferenceTests(unittest.TestCase):
    def test_baseline_is_scored_on_the_same_holdout_without_a_second_attempt(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp), best="n0001")
            write_tree(run, "n0000", "n0001")
            harness = PerNodeHarness({"n0001": (0.9, 0.93), "n0000": (0.5, 0.88)})
            outcome = finish_task(run, TASK, harness)
            self.assertEqual(harness.calls, [("n0001", "holdout_candidate"),
                                             ("n0000", "holdout_candidate")])
            end = outcome["task_end"]
            self.assertEqual((end["holdout_score"], end["holdout_raw"]), (0.9, 0.93))
            self.assertEqual((end["baseline_holdout_score"], end["baseline_holdout_raw"]), (0.5, 0.88))
            self.assertAlmostEqual(end["holdout_raw_lift"], 0.05)
            self.assertFalse(end["baseline_holdout_failed"])
            baseline = saved(run, "baseline_holdout.json")
            self.assertEqual((baseline["schema"], baseline["node"], baseline["reused_best_result"]),
                             ("baseline_holdout/1", "n0000", False))
            self.assertEqual(baseline["result"]["score"], 0.5)
            # The baseline reference is not an attempt: one marker, naming the best node.
            attempts = list((run / "tasks" / TASK).glob("holdout_attempt*.json"))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(saved(run, "holdout_attempt.json")["best_node"], "n0001")
            report = task_report(run, TASK)
            self.assertAlmostEqual(report["holdout_raw_lift"], 0.05)
            self.assertEqual((report["holdout_raw"], report["baseline_holdout_raw"]), (0.93, 0.88))
            self.assertEqual(report["baseline_holdout_score"], 0.5)

    def test_kept_baseline_reuses_its_own_result_with_zero_lift(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            harness = PerNodeHarness({"n0000": (0.4, 0.88)})
            end = finish_task(run, TASK, harness)["task_end"]
            self.assertEqual(len(harness.calls), 1)
            self.assertEqual((end["baseline_holdout_score"], end["holdout_raw_lift"]), (0.4, 0.0))
            self.assertTrue(saved(run, "baseline_holdout.json")["reused_best_result"])

    def test_baseline_failure_never_changes_the_best_nodes_result(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp), best="n0001")
            write_tree(run, "n0000", "n0001")
            harness = PerNodeHarness({"n0001": (0.9, 0.93)}, fail=("n0000",))
            end = finish_task(run, TASK, harness)["task_end"]
            self.assertFalse(end["holdout_failed"])
            self.assertEqual(end["holdout_score"], 0.9)
            self.assertTrue(end["baseline_holdout_failed"])
            self.assertIsNone(end["baseline_holdout_score"])
            self.assertIsNone(end["holdout_raw_lift"])
            self.assertTrue(saved(run, "baseline_holdout.json")["failed"])
            report = task_report(run, TASK)
            self.assertTrue(report["baseline_holdout_failed"])
            self.assertEqual(report["holdout_score"], 0.9)

    def test_failed_best_holdout_has_no_lift_but_baseline_is_still_recorded(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp), best="n0001")
            harness = PerNodeHarness({"n0000": (0.5, 0.88)}, fail=("n0001",))
            end = finish_task(run, TASK, harness)["task_end"]
            self.assertTrue(end["holdout_failed"])
            self.assertEqual(end["baseline_holdout_raw"], 0.88)
            self.assertIsNone(end["holdout_raw_lift"])

    def test_inconsistent_raw_lift_is_rejected_by_the_report(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp), best="n0001")
            write_tree(run, "n0000", "n0001")
            finish_task(run, TASK, PerNodeHarness({"n0001": (0.9, 0.93), "n0000": (0.5, 0.88)}))
            ledger = run / "tasks" / TASK / "ledger.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["holdout_raw_lift"] = 0.5
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "raw lift"):
                task_report(run, TASK)


class FailureReportingTests(unittest.TestCase):
    def test_failed_holdout_reports_a_null_score_and_run_summary(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            write_tree(run, "n0000")
            finish_task(run, TASK, FakeHarness(fail=True))
            finish_run(run)
            task = task_report(run, TASK)
            self.assertIsNone(task["holdout_score"])
            self.assertIs(task["holdout_failed"], True)
            self.assertEqual(task["holdout_error"]["kind"], "harness_exception")
            report = run_report(run)
            self.assertEqual(report["holdout_failures"], 1)
            self.assertEqual(report["holdout_failed_task_ids"], [TASK])
            self.assertEqual(report["tasks"][0]["total_cost"], 0)

    def test_report_rejects_inconsistent_failure_records(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            write_tree(run, "n0000")
            finish_task(run, TASK, FakeHarness(fail=True))
            path = run / "tasks" / TASK / "holdout_score.json"
            record = json.loads(path.read_text())
            path.write_text(json.dumps({**record, "ok": True, "score": 0.7}))
            with self.assertRaisesRegex(ValueError, "null score"):
                task_report(run, TASK)
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            write_tree(run, "n0000")
            finish_task(run, TASK, FakeHarness())
            ledger = run / "tasks" / TASK / "ledger.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[-1]["holdout_score"] = None  # a null score without the failure marker
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "holdout_failed marker"):
                task_report(run, TASK)

    def test_budget_overspend_is_reported_and_counted_per_run(self):
        with TemporaryDirectory() as temp:
            run = setup(Path(temp))
            write_tree(run, "n0000")
            Ledger.open(run, TASK).llm_call(
                node_id=None, role="planner", purpose="plan", effort="high",
                model="test-scripted", usage=Usage(input=2_000_000), seconds=0.1,
                prompt_hash="a" * 64)  # TEST price: cost 2.0 against a budget of 1
            finish_task(run, TASK, FakeHarness())
            finish_run(run)
            task = task_report(run, TASK)
            self.assertTrue(task["budget_violation"])
            self.assertAlmostEqual(task["budget_excess"], 1.0)
            self.assertEqual(task["budget"], 1.0)
            report = run_report(run)
            self.assertEqual((report["budget_violations"], report["budget_violation_task_ids"]),
                             (1, [TASK]))


if __name__ == "__main__":
    unittest.main()
