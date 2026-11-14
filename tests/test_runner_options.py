"""TaskRunner behavior under the optional agent/1 switches (fake roles and harness only).

Each switch is absent from config/agent.json; every test also pins the
original behavior when the switch is not set.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from library import Library
from research_agent.runner import FAILURE_TAIL_CHARS, TaskRunner, failure_tail
from tests.test_runner import CAPS, CONFIG, FakeLedger, FakeRoles, idea, score


class OptionHarness:
    """Scriptable harness for either scenario; the code text (the hypothesis) picks the outcome."""

    def __init__(self, root: Path, task_id: str = "tab-sample", *, money: float = 10.0,
                 outcomes: dict | None = None, default_score: float = 0.1):
        self.task_id = task_id
        speedup = task_id.startswith("spd-")
        workspace = root / "tasks" / task_id / "workspace"
        workspace.mkdir(parents=True)
        features = ["nested_python_loop"] if speedup else ["missing_heavy"]
        (workspace / "profile.json").write_text(
            json.dumps({"schema": "profile/1", "features": features, "summary": {}}), encoding="utf-8")
        (workspace / ("original.py" if speedup else "train.csv")).write_text(
            "def f(x):\n    return x\n" if speedup else "feature,target\n1,0\n2,1\n", encoding="utf-8")
        self.view = {
            "task_id": task_id, "scenario": "speedup" if speedup else "tabular", "title": "sample",
            "description": "sample task", "metric": {"name": "score", "higher_is_better": True},
            "deliverable": {"file": "solution.py", "function": "f", "signature": "f(x)"},
            "budget": {"money": money, "wall_seconds": 60},
            "workspace": workspace, "features": features,
        }
        self.outcomes = outcomes or {}
        self.default_score = default_score

    def load_task(self, task_id):
        assert task_id == self.task_id
        return self.view

    def _spec(self, node_dir: Path) -> dict:
        return self.outcomes.get((node_dir / "solution.py").read_text(encoding="utf-8").strip(), {})

    def run_script(self, task_id, node_dir, argv):
        return {"exit_code": 0, "timed_out": False, "oom": False, "seconds": 0.1,
                "stdout_tail": "", "stderr_tail": "", **self._spec(node_dir).get("run", {})}

    def score(self, task_id, node_dir, split, seed=None):
        return self._spec(node_dir).get("score", score(self.default_score))


class RecordingRoles(FakeRoles):
    """FakeRoles that keeps every payload and can return diagnoser advice."""

    def __init__(self, *args, advice: str = "", drop_allocation: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.advice = advice
        self.drop_allocation = drop_allocation
        self.summaries = 0
        self.plan_payloads, self.execute_payloads = [], []
        self.diagnose_payloads, self.review_payloads = [], []

    def plan(self, payload, *, node_id=None):
        self.plan_payloads.append(payload)
        result = super().plan(payload, node_id=node_id)
        if self.drop_allocation:
            result = {key: value for key, value in result.items() if key != "budget_allocation"}
        return result

    def execute(self, payload, *, fix=False, node_id=None):
        self.execute_payloads.append((fix, payload))
        return super().execute(payload, fix=fix, node_id=node_id)

    def diagnose(self, payload, *, node_id=None):
        self.diagnose_payloads.append(payload)
        return {**super().diagnose(payload, node_id=node_id), "advice": self.advice}

    def review(self, payload, *, node_id=None):
        self.review_payloads.append(payload)
        return super().review(payload, node_id=node_id)

    def summarize(self, payload, *, node_id=None):
        self.summaries += 1
        return super().summarize(payload, node_id=node_id)


def run_task(root: Path, harness, roles, ledger, *, settings=None, baseline=0.1, **kwargs):
    runner = TaskRunner(
        run_dir=root / "runs/run-test", task_id=harness.task_id, harness=harness, ledger=ledger, roles=roles, library=Library.open(root / "runs/run-test/library"),
        baseline_score=score(baseline), config={**CONFIG, **(settings or {})}, hard_caps=CAPS, **kwargs,
    )
    return runner, runner.run()


def statuses(runner) -> list[str]:
    return [node["status"] for node in runner.tree.all_nodes() if node["origin"] != "root"]


def flat_ideas(count: int):
    return [idea(f"flat-{index}", gain=0.9 - 0.1 * index) for index in range(count)]


class StopLossRunnerTests(unittest.TestCase):
    def test_first_n_no_gain_stops_a_tabular_task_and_still_summarizes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(5), draft=False)
            runner, outcome = run_task(root, OptionHarness(root), roles, ledger,
                                       settings={"stop_loss": {"tab": {"first_n_no_gain": 2}}})
            self.assertEqual(outcome.reason, "stop_loss")
            self.assertEqual(outcome.experiments, 2)
            self.assertEqual(outcome.best_node, "n0000")
            self.assertEqual(statuses(runner), ["rejected", "rejected"])
            self.assertEqual(roles.plans, 1)
            self.assertEqual(roles.summaries, 1)
            stops = [event for event in ledger.events if event["kind"] == "stop"]
            self.assertEqual([event["reason"] for event in stops], ["stop_loss"])

    def test_without_the_switch_the_same_task_runs_every_hypothesis(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            runner, outcome = run_task(root, OptionHarness(root), RecordingRoles(ledger, flat_ideas(5), draft=False), ledger)
            self.assertEqual((outcome.reason, outcome.experiments), ("planner", 5))
            # Another scenario's rule does not apply to a tabular task either.
            other = Path(temp) / "other"
            ledger = FakeLedger()
            runner, outcome = run_task(other, OptionHarness(other), RecordingRoles(ledger, flat_ideas(3), draft=False),
                                       ledger, settings={"stop_loss": {"spd": {"first_n_no_gain": 1}}})
            self.assertEqual((outcome.reason, outcome.experiments), ("planner", 3))

    def test_speedup_tasks_use_the_spd_rule(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(4), draft=False)
            runner, outcome = run_task(
                root, OptionHarness(root, "spd-sample"), roles, ledger,
                settings={"stop_loss": {"spd": {"first_n_no_gain": 1}, "tab": {"first_n_no_gain": 4}}})
            self.assertEqual((outcome.reason, outcome.experiments), ("stop_loss", 1))

    def test_consecutive_no_gain_counts_from_the_last_accepted_node(self):
        outcomes = {"good": {"score": score(0.5)}}
        ideas = [idea("flat-a", 0.9), idea("good", 0.8), idea("flat-b", 0.7), idea("flat-c", 0.6)]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            runner, outcome = run_task(
                root, OptionHarness(root, outcomes=outcomes), RecordingRoles(ledger, ideas, draft=False), ledger,
                settings={"stop_loss": {"tab": {"consecutive_no_gain": 2}}})
            self.assertEqual(statuses(runner), ["rejected", "accepted", "rejected", "rejected"])
            self.assertEqual((outcome.reason, outcome.best_node), ("stop_loss", "n0002"))

    def test_failed_nodes_count_and_the_doomed_diagnosis_is_skipped(self):
        outcomes = {"broken": {"run": {"exit_code": 1, "stderr_tail": "boom"}}}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("broken")], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, outcomes=outcomes), roles, ledger,
                                       settings={"stop_loss": {"tab": {"first_n_no_gain": 1}}})
            self.assertEqual((outcome.reason, outcome.experiments), ("stop_loss", 1))
            self.assertEqual(statuses(runner), ["failed"])
            self.assertEqual(runner.tree.get("n0001")["attempts"], 3)
            self.assertEqual(roles.diagnoses, 0)  # the diagnoser's new hypothesis would never run
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("broken")], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, outcomes=outcomes), roles, ledger)
            self.assertEqual(roles.diagnoses, 1)  # unchanged without the switch

    def test_hard_gate_wins_over_stop_loss(self):
        class BigSpender(RecordingRoles):
            def execute(self, payload, *, fix=False, node_id=None):
                self.ledger.money += 0.9  # money 1.0 caps spending at 0.9
                return super().execute(payload, fix=fix, node_id=node_id)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = BigSpender(ledger, flat_ideas(3), draft=False)
            runner, outcome = run_task(root, OptionHarness(root, money=1.0), roles, ledger,
                                       settings={"stop_loss": {"tab": {"first_n_no_gain": 1}}})
            self.assertEqual(statuses(runner), ["rejected"])  # stop-loss was met as well
            self.assertEqual((outcome.reason, outcome.experiments), ("budget_cap", 1))


class CeilingRunnerTests(unittest.TestCase):
    def test_baseline_at_the_ceiling_stops_at_once_by_default(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(2), draft=False)
            runner, outcome = run_task(root, OptionHarness(root, default_score=1.04), roles, ledger, baseline=1.05)
            self.assertEqual((outcome.reason, outcome.experiments, roles.plans), ("ceiling", 0, 0))

    def test_baseline_at_the_ceiling_still_lets_the_planner_open(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(2), draft=False)
            runner, outcome = run_task(root, OptionHarness(root, default_score=1.04), roles, ledger,
                                       baseline=1.05, settings={"ceiling_excludes_baseline": True})
            self.assertEqual(roles.plans, 2)
            self.assertEqual((outcome.reason, outcome.experiments, outcome.best_node), ("planner", 2, "n0000"))

    def test_an_accepted_node_at_the_ceiling_still_ends_the_task(self):
        outcomes = {"better": {"score": score(1.2)}}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("better", 0.9), idea("flat", 0.5)], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, outcomes=outcomes, default_score=1.04), roles, ledger,
                                       baseline=1.05, settings={"ceiling_excludes_baseline": True})
            self.assertEqual((outcome.reason, outcome.experiments, outcome.best_node), ("ceiling", 1, "n0001"))

    def test_stop_loss_ends_a_task_whose_baseline_cannot_be_beaten(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(5), draft=False)
            runner, outcome = run_task(
                root, OptionHarness(root, "spd-sample", default_score=1.04), roles, ledger, baseline=1.05,
                settings={"ceiling_excludes_baseline": True, "stop_loss": {"spd": {"first_n_no_gain": 2}}})
            self.assertEqual((outcome.reason, outcome.experiments, outcome.best_node), ("stop_loss", 2, "n0000"))


class LeanPlannerRunnerTests(unittest.TestCase):
    def test_lean_planner_payloads_drop_allocations_and_carry_the_plan_size(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(2), draft=False)
            runner, outcome = run_task(
                root, OptionHarness(root), roles, ledger,
                settings={"planner_mode": "lean", "hypotheses_per_plan": 3, "plateau_nodes": 1})
            self.assertEqual(outcome.experiments, 2)
            for payload in roles.plan_payloads:
                self.assertEqual(payload["hypotheses_per_plan"], 3)
                self.assertNotIn("allocations", payload)
            self.assertTrue(roles.review_payloads)
            for payload in roles.review_payloads:
                self.assertNotIn("allocations", payload)

    def test_standard_planner_payloads_are_unchanged(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(2), draft=False)
            runner, outcome = run_task(root, OptionHarness(root), roles, ledger, settings={"plateau_nodes": 1})
            for payload in roles.plan_payloads + roles.review_payloads:
                self.assertIn("allocations", payload)
            for payload in roles.plan_payloads:
                self.assertNotIn("hypotheses_per_plan", payload)
            self.assertEqual(set(roles.plan_payloads[0]), {
                "task", "ability_brief", "available_entry_ids", "best", "recent_experiments",
                "budget", "allocations", "last_review"})

    def test_plan_without_a_budget_allocation_runs_normally(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, flat_ideas(2), draft=False, drop_allocation=True)
            runner, outcome = run_task(root, OptionHarness(root), roles, ledger, settings={"planner_mode": "lean"})
            self.assertEqual((outcome.reason, outcome.experiments), ("planner", 2))
            self.assertEqual(runner.allocations, {})
            opening = json.loads((root / "runs/run-test/tasks/tab-sample/opening_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(opening["budget_allocation"], [])


class FailureFeedbackRunnerTests(unittest.TestCase):
    WRONG = {"wrong": {"score": score(0.0, correct=False, flags=["behavior_change"])}}

    def run_wrong(self, root: Path, *, advice: str = "", settings=None, outcomes=None):
        ledger = FakeLedger()
        roles = RecordingRoles(ledger, [idea("wrong")], advice=advice, draft=False)
        runner, outcome = run_task(root, OptionHarness(root, outcomes=outcomes or self.WRONG), roles, ledger,
                                   settings=settings)
        return runner, roles

    def test_fix_and_diagnose_calls_receive_the_last_failure(self):
        with TemporaryDirectory() as temp:
            runner, roles = self.run_wrong(Path(temp), advice="Keep the original tie order.",
                                           settings={"failure_feedback": True})
            fixes = [payload for fix, payload in roles.execute_payloads if fix]
            self.assertEqual(len(fixes), 2)
            self.assertNotIn("last_failure", roles.execute_payloads[0][1])  # the first call has no history
            expected = {"phase": "score", "category": "score", "cheat_flags": ["behavior_change"],
                        "correct": False, "stderr_tail": ""}
            for payload in fixes:
                self.assertEqual(payload["last_failure"], expected)
                self.assertEqual(payload["previous_error"], "Correctness check failed")
            self.assertEqual(roles.diagnose_payloads[0]["last_failure"], expected)

    def test_diagnoser_advice_joins_the_new_hypothesis_rationale(self):
        with TemporaryDirectory() as temp:
            runner, roles = self.run_wrong(Path(temp), advice="Keep the original tie order.",
                                           settings={"failure_feedback": True})
            rationale = runner.tree.get("n0002")["rationale"]
            self.assertEqual(runner.tree.get("n0002")["origin"], "diagnose")
            self.assertIn("Fix the failure", rationale)
            self.assertIn("Advice: Keep the original tie order.", rationale)

    def test_without_the_switch_nothing_changes(self):
        with TemporaryDirectory() as temp:
            runner, roles = self.run_wrong(Path(temp), advice="Keep the original tie order.")
            for fix, payload in roles.execute_payloads:
                self.assertNotIn("last_failure", payload)
            self.assertNotIn("last_failure", roles.diagnose_payloads[0])
            self.assertEqual(runner.tree.get("n0002")["rationale"], "Fix the failure")
            self.assertEqual(runner.last_failure, {})

    def test_failure_categories_follow_the_phase_that_failed(self):
        slow = {"timed_out": True, "exit_code": -1}
        cases = {
            "crash": ({"run": {"exit_code": 1, "stderr_tail": "ValueError: boom"}}, "run", "exception"),
            "slow": ({"run": slow}, "run", "timeout"),
            "huge": ({"run": {"exit_code": -9, "oom": True}}, "run", "oom"),
            "scorer-timeout": ({"score": {**score(0.0), "ok": False, "score": None,
                                          "error": {"kind": "timeout", "message": "Scoring timed out"}}},
                               "score", "timeout"),
            "scorer-crash": ({"score": {**score(0.0), "ok": False, "score": None,
                                        "error": {"kind": "exception", "message": "x"}}}, "score", "exception"),
            "scorer-odd": ({"score": {**score(0.0), "ok": False, "score": None, "error": None}}, "score", "score"),
        }
        for title, (spec, phase, category) in cases.items():
            with self.subTest(title=title):
                with TemporaryDirectory() as temp:
                    root = Path(temp)
                    ledger = FakeLedger()
                    roles = RecordingRoles(ledger, [idea(title)], draft=False)
                    run_task(root, OptionHarness(root, outcomes={title: spec}), roles, ledger,
                             settings={"failure_feedback": True})
                    failure = roles.execute_payloads[1][1]["last_failure"]
                    self.assertEqual((failure["phase"], failure["category"]), (phase, category))

    def test_executor_output_failure_is_reported_too(self):
        class EmptyCode(RecordingRoles):
            def execute(self, payload, *, fix=False, node_id=None):
                self.execute_payloads.append((fix, payload))
                return {"code": "   ", "notes": ""}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = EmptyCode(ledger, [idea("silent")], draft=False)
            run_task(root, OptionHarness(root), roles, ledger, settings={"failure_feedback": True})
            failure = roles.execute_payloads[1][1]["last_failure"]
            self.assertEqual((failure["phase"], failure["category"], failure["stderr_tail"]), ("executor", "exception", ""))

    def test_stderr_tail_is_truncated_and_never_shows_private_paths_or_scorer_text(self):
        noise = "x" * 2000
        stderr = (f"{noise}\nTraceback (most recent call last):\n"
                  "PermissionError: [Errno 1] /work/tasks/spd-x/hidden/reference.py\n"
                  "OSError: cannot open /work/tasks/tab-x/hidden/holdout.csv\n"
                  "TypeError: hidden_layer_sizes must be a tuple\nValueError: boom")
        outcomes = {"crash": {"run": {"exit_code": 1, "stderr_tail": stderr}},
                    "dirty": {"score": {**score(0.0), "ok": False, "score": None,
                                        "error": {"kind": "exception", "message": "SECRET dev rows",
                                                  "trace_tail": "SECRET trace"}}}}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("crash")], draft=False)
            run_task(root, OptionHarness(root, outcomes=outcomes), roles, ledger, settings={"failure_feedback": True})
            tail = roles.execute_payloads[1][1]["last_failure"]["stderr_tail"]
            self.assertLessEqual(len(tail), FAILURE_TAIL_CHARS)
            self.assertTrue(tail.endswith("ValueError: boom"))
            self.assertIn("hidden_layer_sizes", tail)  # an ordinary identifier is not private
            for private in ("hidden/", "reference.py", "holdout"):
                self.assertNotIn(private, tail)
            self.assertIn("[line removed", tail)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("dirty")], draft=False)
            run_task(root, OptionHarness(root, outcomes=outcomes), roles, ledger, settings={"failure_feedback": True})
            self.assertNotIn("SECRET", json.dumps(roles.execute_payloads[1][1]["last_failure"]))
            self.assertNotIn("SECRET", json.dumps(roles.diagnose_payloads[0]["last_failure"]))

    def test_failure_tail_helper(self):
        self.assertEqual(failure_tail(None), "")
        self.assertEqual(failure_tail("a\x00b\x07c\n"), "abc")
        tail = failure_tail("\n".join(str(i) for i in range(500)), 20)
        self.assertLessEqual(len(tail), 20)
        self.assertTrue(tail.endswith("498\n499"))
        self.assertEqual(len(failure_tail("y" * 5000)), FAILURE_TAIL_CHARS)

    def test_a_later_success_clears_the_recorded_failure(self):
        class FixesItself(RecordingRoles):
            def execute(self, payload, *, fix=False, node_id=None):
                self.execute_payloads.append((fix, payload))
                return {"code": "fixed" if fix else "wrong", "notes": ""}

        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = FixesItself(ledger, [idea("wrong")], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, outcomes={**self.WRONG, "fixed": {"score": score(0.5)}}),
                                       roles, ledger, settings={"failure_feedback": True})
            self.assertEqual(runner.tree.get("n0001")["attempts"], 2)
            self.assertEqual(runner.tree.get("n0001")["status"], "accepted")
            self.assertEqual(runner.last_failure, {})
            self.assertEqual(roles.diagnoses, 0)


class CostPriorRunnerTests(unittest.TestCase):
    def test_prior_replaces_the_planner_cost_when_checking_fit(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("cheap-in-truth", cost=0.95)], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, money=1.0), roles, ledger)
            self.assertEqual((outcome.reason, outcome.experiments), ("budget_cap", 0))  # as before
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("cheap-in-truth", cost=0.95)], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, money=1.0), roles, ledger,
                                       settings={"cost_priors": {"tab": 0.1}})
            self.assertEqual((outcome.reason, outcome.experiments), ("planner", 1))
            self.assertEqual(runner.tree.get("n0001")["estimate"]["cost"], 0.95)  # the planner's value is kept
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("looks-cheap", cost=0.1)], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, money=1.0), roles, ledger,
                                       settings={"cost_priors": {"tab": 5.0}})
            self.assertEqual((outcome.reason, outcome.experiments), ("budget_cap", 0))

    def test_prior_only_applies_to_its_own_scenario(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            roles = RecordingRoles(ledger, [idea("cheap-in-truth", cost=0.95)], draft=False)
            runner, outcome = run_task(root, OptionHarness(root, money=1.0), roles, ledger,
                                       settings={"cost_priors": {"spd": 0.1}})
            self.assertEqual((outcome.reason, outcome.experiments), ("budget_cap", 0))

    def test_prior_changes_the_order_hypotheses_are_tried_in(self):
        ideas = [idea("lean-win", gain=0.1, cost=0.05), idea("big-win", gain=0.3, cost=0.5)]

        def first_tried(settings):
            with TemporaryDirectory() as temp:
                root = Path(temp)
                ledger = FakeLedger()
                roles = RecordingRoles(ledger, ideas, draft=False)
                runner, _ = run_task(root, OptionHarness(root), roles, ledger, settings=settings)
                return [node["hypothesis"] for node in runner.tree.all_nodes()][1]

        self.assertEqual(first_tried(None), "lean-win")  # 0.1/0.05 beats 0.3/0.5
        self.assertEqual(first_tried({"cost_priors": {"tab": 0.2}}), "big-win")  # 0.1/0.2 loses to 0.3/0.2


if __name__ == "__main__":
    unittest.main()
