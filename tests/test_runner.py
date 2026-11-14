from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from library import Library
from research_agent.io import read_jsonl
from research_agent.model import BudgetReserveError
from research_agent.replay import replay_node
from research_agent.runner import TaskRunner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/agent.json").read_text(encoding="utf-8"))
CAPS = json.loads((ROOT / "config/hard_caps.json").read_text(encoding="utf-8"))


def score(value: float, *, spread: float = 0.01, correct: bool = True, flags: list[str] | None = None) -> dict:
    return {
        "schema": "score/1", "ok": True, "split": "dev", "score": value,
        "raw": value, "spread": spread, "correct": correct,
        "cheat_flags": flags or [], "error": None,
        "seconds": {"run": 0.2, "queue": 0.1}, "seed": 12,
    }


class FakeLedger:
    def __init__(self):
        self.events = []
        self.money = 0.0
        self.llm_seconds = 0.0

    def emit(self, kind, *, node_id=None, **values):
        self.events.append({"kind": kind, "node_id": node_id, **values})

    def spent(self):
        return self.money

    def _events(self):
        return list(self.events)

    def elapsed(self):
        return {"llm": self.llm_seconds, "run": 0.0, "queue": 0.0, "wall": 0.0}

    def charge(self):
        self.money += 0.01
        self.llm_seconds += 0.1


class FakeHarness:
    def __init__(self, root: Path, *, money=10.0, failures=False):
        workspace = root / "tasks" / "tab-sample" / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "profile.json").write_text(json.dumps({"schema": "profile/1", "features": ["missing_heavy"], "summary": {"rows": 4}}), encoding="utf-8")
        (workspace / "train.csv").write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
        self.view = {
            "task_id": "tab-sample", "scenario": "tabular", "title": "sample",
            "description": "Predict target", "metric": {"name": "auc", "higher_is_better": True},
            "deliverable": {"file": "solution.py", "function": "fit_predict", "signature": "fit_predict(train,test,target)"},
            "budget": {"money": money, "wall_seconds": 60},
            "workspace": workspace, "features": ["missing_heavy"],
        }
        self.failures = failures

    def load_task(self, task_id):
        assert task_id == "tab-sample"
        return self.view

    def run_script(self, task_id, node_dir, argv):
        assert task_id == "tab-sample" and argv == ["solution.py"]
        code = (node_dir / "solution.py").read_text(encoding="utf-8")
        failed = self.failures and code.startswith("bad")
        return {"exit_code": 1 if failed else 0, "timed_out": False, "oom": False, "seconds": 0.1, "stdout_tail": "", "stderr_tail": "bad candidate" if failed else ""}

    def score(self, task_id, node_dir, split, seed=None):
        assert task_id == "tab-sample" and split == "dev"
        assert seed is None or seed == 12
        code = (node_dir / "solution.py").read_text(encoding="utf-8")
        if "weak" in code:
            return score(0.41, spread=0.05)
        return score(0.4)


class FakeRoles:
    def __init__(self, ledger, hypotheses, *, draft=True, fail_first=False):
        self.ledger = ledger
        self.hypotheses = hypotheses
        self.draft = draft
        self.fail_first = fail_first
        self.plans = 0
        self.diagnoses = 0
        self.calls = []

    def plan(self, payload, *, node_id=None):
        self.ledger.charge()
        self.plans += 1
        return {
            "hypotheses": self.hypotheses if self.plans == 1 else [],
            "predicted": {"cost": 1.0, "seconds": 10, "score": 0.5},
            "budget_allocation": [{"direction": "features", "amount": 1.0}],
            "expected_progress": [{"spent_ratio": 0.5, "score": 0.5}],
        }

    def execute(self, payload, *, fix=False, node_id=None):
        self.ledger.charge()
        self.calls.append((node_id, fix))
        if self.fail_first and payload["hypothesis"] == "broken":
            return {"code": "bad code", "notes": ""}
        return {"code": payload["hypothesis"], "notes": ""}

    def diagnose(self, payload, *, node_id=None):
        self.ledger.charge()
        self.diagnoses += 1
        return {"hypothesis": "alternate", "rationale": "Fix the failure", "advice": ""}

    def review(self, payload, *, node_id=None):
        self.ledger.charge()
        return {"decision": "continue", "budget_moves": [], "note": "Keep testing"}

    def summarize(self, payload, *, node_id=None):
        self.ledger.charge()
        if not self.draft:
            return {"drafts": [], "narrowings": []}
        node_ids = [node["id"] for node in payload["experiments"] if node["status"] in {"accepted", "rejected", "failed"}]
        return {"drafts": [{"scope": "domain", "scenario": "tabular", "triggers": ["missing_heavy"], "conditions": "", "action": "Check missing values", "negative": False, "node_ids": node_ids}], "narrowings": []}


def idea(title, gain=0.2, cost=0.2):
    return {"hypothesis": title, "rationale": "test", "estimate": {"gain": gain, "cost": cost, "seconds": 1.0}, "cites": []}


class RunnerTests(unittest.TestCase):
    def make_runner(self, root: Path, harness, roles, ledger, library):
        return TaskRunner(
            run_dir=root / "runs" / "run-test", task_id="tab-sample",
            harness=harness, ledger=ledger, roles=roles, library=library,
            baseline_score=score(0.1), config=CONFIG, hard_caps=CAPS,
        )

    def test_full_loop_accepts_and_rejects_then_writes_experience(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root)
            library = Library.open(root / "runs/run-test/library")
            roles = FakeRoles(ledger, [idea("good", 0.3), idea("weak", 0.1)])
            result = self.make_runner(root, harness, roles, ledger, library).run()
            self.assertEqual(result.reason, "planner")
            self.assertEqual(result.best_node, "n0001")
            self.assertEqual(result.experiments, 2)
            self.assertEqual(result.library_version, "lib-1")
            nodes = read_jsonl(root / "runs/run-test/tasks/tab-sample/tree.jsonl", "node/1")
            last = {node["node_id"]: node for node in nodes}
            self.assertEqual([last[f"n{i:04d}"]["status"] for i in range(3)], ["accepted", "accepted", "rejected"])
            self.assertEqual(last["n0001"]["attempts"], 1)
            self.assertEqual(last["n0002"]["parent"], "n0001")
            self.assertEqual(last["n0001"]["calls"], [{"role": "executor", "effort": "low", "n": 1}])
            self.assertTrue((root / "runs/run-test/tasks/tab-sample/nodes/n0001/parent.diff").exists())
            self.assertEqual([event["kind"] for event in ledger.events if event["kind"] in {"task_start", "stop", "experiment", "score"}].count("experiment"), 2)

    def test_opening_plan_is_available_before_first_experiment(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root)
            library = Library.open(root / "runs/run-test/library")
            roles = FakeRoles(ledger, [idea("good")], draft=False)
            seen = []

            def on_initial_plan(plan):
                self.assertFalse(any(event["kind"] == "experiment" for event in ledger.events))
                seen.append(plan)

            runner = TaskRunner(
                run_dir=root / "runs/run-test", task_id="tab-sample", harness=harness,
                ledger=ledger, roles=roles, library=library, baseline_score=score(0.1),
                config=CONFIG, hard_caps=CAPS, on_initial_plan=on_initial_plan,
            )
            runner.run()
            self.assertEqual(len(seen), 1)
            opening = json.loads((root / "runs/run-test/tasks/tab-sample/opening_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(opening["predicted"], seen[0]["predicted"])

    def test_three_failed_attempts_call_diagnoser_and_keep_failure(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root, failures=True)
            library = Library.open(root / "runs/run-test/library")
            roles = FakeRoles(ledger, [idea("broken")], fail_first=True, draft=False)
            result = self.make_runner(root, harness, roles, ledger, library).run()
            self.assertEqual(result.best_node, "n0002")
            self.assertEqual(roles.diagnoses, 1)
            latest = {node["node_id"]: node for node in read_jsonl(root / "runs/run-test/tasks/tab-sample/tree.jsonl", "node/1")}
            self.assertEqual(latest["n0001"]["status"], "failed")
            self.assertEqual(latest["n0001"]["attempts"], 3)
            self.assertEqual(latest["n0002"]["origin"], "diagnose")
            self.assertEqual(latest["n0002"]["status"], "accepted")

    def test_estimate_is_blocked_by_budget_gate(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root, money=1)
            library = Library.open(root / "runs/run-test/library")
            roles = FakeRoles(ledger, [idea("unaffordable", cost=0.95)], draft=False)
            result = self.make_runner(root, harness, roles, ledger, library).run()
            self.assertEqual(result.reason, "budget_cap")
            self.assertEqual(result.experiments, 0)
            self.assertEqual(result.best_node, "n0000")
            self.assertFalse(any(e["kind"] == "experiment" for e in ledger.events))

    def test_reserve_during_executor_closes_node_and_stops(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root)
            library = Library.open(root / "runs/run-test/library")

            class BlockedExecutor(FakeRoles):
                def execute(self, payload, *, fix=False, node_id=None):
                    raise BudgetReserveError("reserve reached")

            roles = BlockedExecutor(ledger, [idea("good")], draft=False)
            result = self.make_runner(root, harness, roles, ledger, library).run()
            self.assertEqual(result.reason, "budget_cap")
            self.assertEqual(result.best_node, "n0000")
            self.assertEqual(result.experiments, 0)
            latest = {node["node_id"]: node for node in read_jsonl(
                root / "runs/run-test/tasks/tab-sample/tree.jsonl", "node/1")}
            self.assertEqual(latest["n0001"]["status"], "failed")
            self.assertEqual(latest["n0001"]["failure"]["kind"], "budget")
            self.assertEqual(latest["n0001"]["calls"], [])
            self.assertEqual([event["kind"] for event in ledger.events].count("stop"), 1)

    def test_exam_frozen_library_logs_use_without_mutation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            lib_root = root / "runs/run-test/library"
            editable = Library.open(lib_root)
            entry_id = editable.write_experiences(
                [{"scope": "domain", "scenario": "tabular", "triggers": ["missing_heavy"], "conditions": "", "action": "Try feature", "negative": False}],
                source={"run_id": "older", "task_id": "tab-old", "node_ids": ["n0001"]},
            )[0]
            frozen = Library.open(lib_root, frozen=True)
            ledger = FakeLedger()
            harness = FakeHarness(root)
            hypothesis = idea("good")
            hypothesis["cites"] = [entry_id]
            roles = FakeRoles(ledger, [hypothesis])
            result = self.make_runner(root, harness, roles, ledger, frozen).run()
            self.assertEqual(result.library_version, "lib-1")
            self.assertEqual(Library.open(lib_root).get(entry_id)["effect"]["uses"], 0)
            self.assertTrue(any(e["kind"] == "library" and e["op"] == "use" for e in ledger.events))
            drafts = read_jsonl(root / "runs/run-test/tasks/tab-sample/experience_drafts.jsonl", "experience_drafts/1")
            self.assertEqual(drafts[0]["drafts"][0]["action"], "Check missing values")

    def test_missing_real_budget_fails_before_events(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root, money=None)
            library = Library.open(root / "runs/run-test/library")
            with self.assertRaisesRegex(ValueError, "budget.money"):
                self.make_runner(root, harness, FakeRoles(ledger, []), ledger, library)
            self.assertFalse(ledger.events)

    def test_replay_restores_node_library_version_and_score_seed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root)
            library_root = root / "runs/run-test/library"
            library = Library.open(library_root)
            roles = FakeRoles(ledger, [idea("good")])
            self.make_runner(root, harness, roles, ledger, library).run()
            report = replay_node(
                run_dir=root / "runs/run-test", task_id="tab-sample",
                node_id="n0001", harness=harness, library_root=library_root,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["library_version"], "lib-0")
            self.assertEqual(report["observation"]["current"]["seed"], 12)
            self.assertEqual(len(list((root / "runs/run-test/replays").glob("*.json"))), 1)

    def test_replay_failed_node_uses_saved_code(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FakeHarness(root, failures=True)
            library_root = root / "runs/run-test/library"
            library = Library.open(library_root)
            roles = FakeRoles(ledger, [idea("broken")], fail_first=True, draft=False)
            self.make_runner(root, harness, roles, ledger, library).run()
            report = replay_node(
                run_dir=root / "runs/run-test", task_id="tab-sample",
                node_id="n0001", harness=harness, library_root=library_root,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["observation"]["type"], "run")

    def test_review_switch_discards_old_proposals_and_replans(self):
        class SwitchRoles(FakeRoles):
            def plan(self, payload, *, node_id=None):
                self.ledger.charge()
                self.plans += 1
                ideas = [idea("bad1", 0.4), idea("stale", 0.1)] if self.plans == 1 else [idea("fresh", 0.2)] if self.plans == 2 else []
                return {"hypotheses": ideas, "predicted": {"cost": 1, "seconds": 10, "score": 0.5}, "budget_allocation": [], "expected_progress": []}

            def review(self, payload, *, node_id=None):
                self.ledger.charge()
                return {"decision": "switch", "budget_moves": [], "note": "Try a new direction"}

        class SwitchHarness(FakeHarness):
            def score(self, task_id, node_dir, split, seed=None):
                code = (node_dir / "solution.py").read_text(encoding="utf-8")
                return score(0.1 if "bad1" in code else 0.5 if "fresh" in code else 0.4)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = SwitchHarness(root)
            library = Library.open(root / "runs/run-test/library")
            roles = SwitchRoles(ledger, [])
            settings = {**CONFIG, "plateau_nodes": 1}
            runner = TaskRunner(
                run_dir=root / "runs/run-test", task_id="tab-sample", harness=harness,
                ledger=ledger, roles=roles, library=library, baseline_score=score(0.1),
                config=settings, hard_caps=CAPS,
            )
            result = runner.run()
            self.assertEqual(result.experiments, 2)
            self.assertEqual(roles.plans, 3)
            hypotheses = [node["hypothesis"] for node in runner.tree.all_nodes()]
            self.assertEqual(hypotheses, ["Baseline", "bad1", "fresh"])
            self.assertEqual(runner.tree.best_node()["hypothesis"], "fresh")

    def test_visible_workspace_symlink_cannot_read_hidden(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            harness = FakeHarness(root)
            workspace = harness.view["workspace"]
            (workspace / "profile.json").unlink()
            hidden = root / "tasks/tab-sample/hidden"
            hidden.mkdir()
            (hidden / "profile.json").write_text('{"schema":"profile/1","features":[]}', encoding="utf-8")
            (workspace / "profile.json").symlink_to(hidden / "profile.json")
            ledger = FakeLedger()
            library = Library.open(root / "runs/run-test/library")
            with self.assertRaisesRegex(ValueError, "outside the visible workspace"):
                self.make_runner(root, harness, FakeRoles(ledger, []), ledger, library)

    def test_retired_citation_does_not_abort_other_planned_experiments(self):
        class FlatHarness(FakeHarness):
            def score(self, task_id, node_dir, split, seed=None):
                return score(0.1)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            library = Library.open(root / "runs/run-test/library")
            entry_id = library.write_experiences(
                [{"scope": "domain", "scenario": "tabular", "triggers": ["missing_heavy"], "conditions": "", "action": "Try feature", "negative": False}],
                source={"run_id": "old", "task_id": "old", "node_ids": ["n0001"]},
            )[0]
            hypotheses = [idea(f"failed-{index}", gain=0.3 - index * 0.01) for index in range(6)]
            for hypothesis in hypotheses:
                hypothesis["cites"] = [entry_id]
            ledger = FakeLedger()
            harness = FlatHarness(root)
            roles = FakeRoles(ledger, hypotheses, draft=False)
            result = self.make_runner(root, harness, roles, ledger, library).run()
            self.assertEqual(result.experiments, 6)
            self.assertEqual(library.get(entry_id)["status"], "retired")
            self.assertEqual(result.reason, "planner")

    def test_suspicious_ceiling_score_still_calls_diagnoser(self):
        class SpikeHarness(FakeHarness):
            def score(self, task_id, node_dir, split, seed=None):
                return score(1.3)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = SpikeHarness(root)
            library = Library.open(root / "runs/run-test/library")
            roles = FakeRoles(ledger, [idea("spike")], draft=False)
            result = self.make_runner(root, harness, roles, ledger, library).run()
            self.assertEqual(result.reason, "ceiling")
            self.assertEqual(roles.diagnoses, 1)

    def test_task_end_narrows_failed_cited_experience_with_evidence(self):
        class FlatHarness(FakeHarness):
            def score(self, task_id, node_dir, split, seed=None):
                return score(0.1)

        class NarrowRoles(FakeRoles):
            def summarize(self, payload, *, node_id=None):
                self.ledger.charge()
                return {"drafts": [], "narrowings": [{
                    "entry_id": entry_id,
                    "conditions": "categorical data and more than 10000 rows",
                    "reason": "Cited experiment failed on a four-row task",
                }]}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            library = Library.open(root / "runs/run-test/library")
            entry_id = library.write_experiences(
                [{"scope": "domain", "scenario": "tabular", "triggers": ["missing_heavy"], "conditions": "categorical data", "action": "Encode category", "negative": False}],
                source={"run_id": "old", "task_id": "old", "node_ids": ["n0001"]},
            )[0]
            hypothesis = idea("failed")
            hypothesis["cites"] = [entry_id]
            ledger = FakeLedger()
            roles = NarrowRoles(ledger, [hypothesis], draft=False)
            result = self.make_runner(root, FlatHarness(root), roles, ledger, library).run()
            self.assertEqual(result.reason, "planner")
            self.assertEqual(library.get(entry_id)["conditions"], "categorical data and more than 10000 rows")
            change = library.changelog()[-1]
            self.assertEqual(change["op"], "narrow")
            self.assertEqual(change["evidence"][0]["node_ids"], ["n0001"])


if __name__ == "__main__":
    unittest.main()
