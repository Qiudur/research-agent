"""Repeatable end-to-end boundary checks for the A-owned task runner."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from library import Library
from research_agent.runner import TaskRunner
from tests.test_runner import CAPS, CONFIG, FakeHarness, FakeLedger, FakeRoles, idea, score


def make_runner(root, harness, ledger, roles, *, settings=None, clock=None):
    kwargs = {
        "run_dir": root / "runs/run-edge", "task_id": "tab-sample",
        "harness": harness, "ledger": ledger, "roles": roles,
        "library": Library.open(root / "runs/run-edge/library"),
        "baseline_score": score(0.1),
        "config": settings or CONFIG, "hard_caps": CAPS,
    }
    if clock is not None:
        kwargs["clock"] = clock
    return TaskRunner(**kwargs)


class RunnerBoundaryTests(unittest.TestCase):
    def test_reallocate_replans_and_never_executes_stale_candidate(self):
        class ReallocRoles(FakeRoles):
            def plan(self, payload, *, node_id=None):
                self.ledger.charge()
                self.plans += 1
                hypotheses = (
                    [idea("no-gain", gain=0.5), idea("stale", gain=0.1)]
                    if self.plans == 1 else [idea("fresh", gain=0.3)] if self.plans == 2 else []
                )
                return {
                    "hypotheses": hypotheses,
                    "predicted": {"cost": 1.0, "seconds": 10, "score": 0.5},
                    "budget_allocation": [{"direction": "features", "amount": 0.7}, {"direction": "algorithms", "amount": 0.3}],
                    "expected_progress": [],
                }

            def review(self, payload, *, node_id=None):
                self.ledger.charge()
                self.reviews = getattr(self, "reviews", 0) + 1
                if self.reviews > 1:
                    return {"decision": "continue", "budget_moves": [], "note": "Continue"}
                return {
                    "decision": "reallocate",
                    "budget_moves": [{"from": "features", "to": "algorithms", "amount": 0.4}],
                    "note": "Move budget to the stronger direction",
                }

        class ReallocHarness(FakeHarness):
            def score(self, task_id, node_dir, split, seed=None):
                code = (node_dir / "solution.py").read_text(encoding="utf-8")
                return score(0.1 if code.strip() == "no-gain" else 0.4)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = ReallocRoles(ledger, [], draft=False)
            runner = make_runner(
                root, ReallocHarness(root), ledger, roles,
                settings={**CONFIG, "plateau_nodes": 1},
            )
            outcome = runner.run()
            self.assertEqual(outcome.reason, "planner")
            self.assertEqual(outcome.experiments, 2)
            self.assertEqual(roles.plans, 3)
            self.assertEqual([n["hypothesis"] for n in runner.tree.all_nodes()], ["Baseline", "no-gain", "fresh"])
            self.assertEqual([n["hypothesis"] for n in runner.tree.all_nodes() if n["origin"] != "root"], ["no-gain", "fresh"])
            self.assertAlmostEqual(runner.allocations["features"], 0.3)
            self.assertAlmostEqual(runner.allocations["algorithms"], 0.7)
            self.assertEqual(
                [(event["decision"], event["budget_moves"]) for event in ledger.events if event["kind"] == "review"],
                [("reallocate", [{"from": "features", "to": "algorithms", "amount": 0.4}])],
            )

    def test_budget_cap_reserves_wrap_up_and_stops_expansion(self):
        class BudgetRoles(FakeRoles):
            def __init__(self, ledger):
                super().__init__(ledger, [], draft=False)
                self.summaries = 0

            def plan(self, payload, *, node_id=None):
                self.ledger.money += 0.05
                self.plans += 1
                return {"hypotheses": [idea("one", gain=0.2, cost=0.1), idea("two", gain=0.1, cost=0.1)]}

            def execute(self, payload, *, fix=False, node_id=None):
                self.ledger.money += 0.86
                self.calls.append((node_id, fix))
                return {"code": payload["hypothesis"], "notes": ""}

            def summarize(self, payload, *, node_id=None):
                self.ledger.money += 0.04
                self.summaries += 1
                return {"drafts": [], "narrowings": []}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = BudgetRoles(ledger)
            runner = make_runner(root, FakeHarness(root, money=1.0), ledger, roles)
            outcome = runner.run()
            self.assertEqual(outcome.reason, "budget_cap")
            self.assertEqual(outcome.experiments, 1)
            self.assertEqual(roles.summaries, 1)
            self.assertEqual(len(roles.calls), 1)
            self.assertEqual(len([event for event in ledger.events if event["kind"] == "experiment"]), 1)
            self.assertLess(outcome.spent, 1.0)
            self.assertGreaterEqual(outcome.spent, 0.9)

    def test_wall_time_stop_skips_summary_when_window_exhausted(self):
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

        class TimeRoles(FakeRoles):
            def __init__(self, ledger, clock):
                super().__init__(ledger, [idea("one", gain=0.2), idea("two", gain=0.1)], draft=False)
                self.clock = clock
                self.summaries = 0

            def execute(self, payload, *, fix=False, node_id=None):
                result = super().execute(payload, fix=fix, node_id=node_id)
                self.clock.now = 60.0
                return result

            def summarize(self, payload, *, node_id=None):
                self.summaries += 1
                return {"drafts": [], "narrowings": []}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            clock = Clock()
            ledger = FakeLedger()
            roles = TimeRoles(ledger, clock)
            runner = make_runner(root, FakeHarness(root), ledger, roles, clock=clock)
            outcome = runner.run()
            self.assertEqual(outcome.reason, "time")
            self.assertEqual(outcome.experiments, 1)
            self.assertEqual(len(roles.calls), 1)
            self.assertEqual(roles.summaries, 0)
            self.assertEqual(runner.tree.best_node()["hypothesis"], "one")

    def test_failed_dev_scoring_is_fixed_within_same_node(self):
        class FlakyScoreHarness(FakeHarness):
            def __init__(self, root):
                super().__init__(root)
                self.score_calls = 0

            def score(self, task_id, node_dir, split, seed=None):
                self.score_calls += 1
                if self.score_calls == 1:
                    failed = score(0.0)
                    failed.update({"ok": False, "score": None, "error": {"kind": "exception", "message": "temporary scoring error", "trace_tail": ""}})
                    return failed
                return score(0.4)

        class FixRoles(FakeRoles):
            def execute(self, payload, *, fix=False, node_id=None):
                self.ledger.charge()
                self.calls.append((node_id, fix, payload["previous_error"]))
                return {"code": "fixed" if fix else "initial", "notes": ""}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            harness = FlakyScoreHarness(root)
            roles = FixRoles(ledger, [idea("try-scoring")], draft=False)
            runner = make_runner(root, harness, ledger, roles)
            outcome = runner.run()
            node = runner.tree.get("n0001")
            self.assertEqual(outcome.experiments, 1)
            self.assertEqual(node["status"], "accepted")
            self.assertEqual(node["attempts"], 2)
            self.assertEqual(harness.score_calls, 2)
            self.assertEqual([call[1] for call in roles.calls], [False, True])
            self.assertIn("temporary scoring error", roles.calls[1][2])
            self.assertEqual(len([event for event in ledger.events if event["kind"] == "score"]), 2)

    def test_cheat_flag_rejects_candidate_without_retry(self):
        class FlagHarness(FakeHarness):
            def score(self, task_id, node_dir, split, seed=None):
                return score(0.6, flags=["reads_hidden"])

        class FlagRoles(FakeRoles):
            def diagnose(self, payload, *, node_id=None):
                self.ledger.charge()
                self.diagnoses += 1
                return {"hypothesis": "", "rationale": "", "advice": ""}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = FlagRoles(ledger, [idea("cheating")], draft=False)
            runner = make_runner(root, FlagHarness(root), ledger, roles)
            outcome = runner.run()
            node = runner.tree.get("n0001")
            self.assertEqual(outcome.best_node, "n0000")
            self.assertEqual(outcome.experiments, 1)
            self.assertEqual(node["status"], "rejected")
            self.assertEqual(node["score"]["cheat_flags"], ["reads_hidden"])
            self.assertEqual(node["attempts"], 1)
            self.assertEqual(roles.diagnoses, 1)
            self.assertEqual(len(roles.calls), 1)


if __name__ == "__main__":
    unittest.main()
