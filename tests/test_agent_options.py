"""Optional agent/1 switches: scheduler rules, lean planner schema, validation, stop labels.

Each switch is absent from config/agent.json; absent must mean the original behavior.
No real model, harness or hidden data is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from dashboard.export import load_run
from ledger import Ledger
from research_agent.model import ModelCallError, ModelClient, _validate_schema
from research_agent.roles import PLAN_SCHEMA, PLAN_SCHEMA_LEAN, SCHEMAS, RoleAgents
from research_agent.scheduler import ScheduleError, Scheduler, scenario_key, validate_options
from scripts.summarize_formal_batch import STOP_LABELS, render_markdown, summarize
from tests.test_ledger import fixture as ledger_fixture
from tests.test_model_roles import FakeClient, FakeLedger, response
from tests.test_summarize_formal_batch import build as build_formal_batch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/agent.json").read_text(encoding="utf-8"))
SETTINGS = {
    "schema": "agent/1",
    "plateau_nodes": 5,
    "behind_schedule": {"spent_ratio": 0.5, "progress_ratio": 0.5},
    "explore_every": 4,
}
CAPS = {"schema": "hard_caps/1", "budget_ratio": 0.9, "reserve_for_wrapup": 0.1}


def scheduler(task_id: str = "tab-demo", **options) -> Scheduler:
    return Scheduler({**SETTINGS, **options}, CAPS, budget=10.0, wall_seconds=100.0, task_id=task_id)


class SchedulerOptionTests(unittest.TestCase):
    def test_without_options_the_original_rules_apply(self):
        plain = scheduler("spd-demo")
        self.assertEqual(plain.stop_reason(spent=1, elapsed_wall=1, best_score=1.0, best_node_id="n0000"), "ceiling")
        self.assertIsNone(plain.stop_loss_reason(["rejected"] * 20))
        self.assertIsNone(plain.cost_prior)
        self.assertEqual(plain.estimated_cost(0.7), 0.7)
        self.assertTrue(plain.can_expand(spent=8, elapsed_wall=20, estimated_cost=1.0))
        self.assertFalse(plain.can_expand(spent=8, elapsed_wall=20, estimated_cost=1.1))

    def test_scenario_comes_from_the_task_id_prefix(self):
        self.assertEqual(scenario_key("spd-weighted-groups"), "spd")
        self.assertEqual(scenario_key("tab-bank-marketing"), "tab")
        for other in ("", None, "abc", "spdx-1", "tabular"):
            self.assertIsNone(scenario_key(other))
        self.assertEqual(scheduler("spd-demo").scenario, "spd")
        self.assertIsNone(scheduler("").scenario)

    def test_ceiling_excludes_baseline_only_when_enabled(self):
        enabled = scheduler(ceiling_excludes_baseline=True)
        high = {"spent": 1, "elapsed_wall": 1, "best_score": 1.4}
        self.assertIsNone(enabled.stop_reason(**high, best_node_id="n0000"))
        self.assertEqual(enabled.stop_reason(**high, best_node_id="n0003"), "ceiling")
        self.assertEqual(enabled.stop_reason(**high), "ceiling")  # unknown best node: original rule
        self.assertIsNone(enabled.stop_reason(spent=1, elapsed_wall=1, best_score=0.99, best_node_id="n0003"))
        # The hard budget and time gates are not affected by the baseline exception.
        self.assertEqual(enabled.stop_reason(spent=9, elapsed_wall=1, best_score=1.4, best_node_id="n0000"), "budget_cap")
        self.assertEqual(enabled.stop_reason(spent=1, elapsed_wall=100, best_score=1.4, best_node_id="n0000"), "time")
        disabled = scheduler(ceiling_excludes_baseline=False)
        self.assertEqual(disabled.stop_reason(**high, best_node_id="n0000"), "ceiling")

    def test_stop_loss_rules_are_per_scenario(self):
        rules = {"spd": {"first_n_no_gain": 2}, "tab": {"consecutive_no_gain": 3}}
        spd, tab = scheduler("spd-demo", stop_loss=rules), scheduler("tab-demo", stop_loss=rules)
        self.assertIsNone(spd.stop_loss_reason([]))
        self.assertIsNone(spd.stop_loss_reason(["rejected"]))
        self.assertEqual(spd.stop_loss_reason(["rejected", "failed"]), "stop_loss")
        # One accepted node among the first N lifts first_n_no_gain for good.
        self.assertIsNone(spd.stop_loss_reason(["rejected", "accepted", "rejected", "rejected", "rejected"]))
        self.assertIsNone(tab.stop_loss_reason(["rejected", "failed"]))  # tab has no first_n rule
        self.assertEqual(tab.stop_loss_reason(["rejected", "failed", "rejected"]), "stop_loss")
        self.assertIsNone(tab.stop_loss_reason(["rejected", "rejected", "accepted", "rejected", "rejected"]))
        self.assertEqual(tab.stop_loss_reason(["accepted", "rejected", "failed", "rejected"]), "stop_loss")
        # Unknown prefixes and a missing task id never match a rule.
        self.assertIsNone(scheduler("", stop_loss=rules).stop_loss_reason(["rejected"] * 9))

    def test_first_n_and_consecutive_rules_stop_independently(self):
        both = scheduler("tab-demo", stop_loss={"tab": {"first_n_no_gain": 3, "consecutive_no_gain": 2}})
        self.assertIsNone(both.stop_loss_reason(["rejected"]))
        self.assertEqual(both.stop_loss_reason(["rejected", "rejected"]), "stop_loss")  # streak from the start
        self.assertEqual(both.stop_loss_reason(["accepted", "rejected", "rejected"]), "stop_loss")
        self.assertIsNone(both.stop_loss_reason(["accepted", "rejected", "accepted", "rejected"]))
        only_first = scheduler("tab-demo", stop_loss={"tab": {"first_n_no_gain": 3, "consecutive_no_gain": None}})
        self.assertIsNone(only_first.stop_loss_reason(["rejected", "rejected"]))
        self.assertEqual(only_first.stop_loss_reason(["rejected", "failed", "rejected"]), "stop_loss")

    def test_cost_priors_replace_the_planner_estimate_for_fit_and_rank(self):
        prior = scheduler("spd-demo", cost_priors={"spd": 2.5, "tab": 8})
        self.assertEqual(prior.cost_prior, 2.5)
        self.assertEqual(prior.estimated_cost(0.1), 2.5)
        self.assertFalse(prior.can_expand(spent=7, elapsed_wall=1, estimated_cost=0.1))  # 7 + 2.5 > 9
        self.assertTrue(prior.can_expand(spent=6, elapsed_wall=1, estimated_cost=100))  # the guess is ignored
        self.assertEqual(scheduler("tab-demo", cost_priors={"spd": 2.5}).cost_prior, None)
        estimate = {"gain": 0.1, "cost": 0.2}
        self.assertAlmostEqual(Scheduler.priority_for(estimate), 0.5)
        self.assertAlmostEqual(Scheduler.priority_for(estimate, cost_prior=0.5), 0.2)
        with self.assertRaises(ScheduleError):  # an invalid planner estimate is still rejected
            Scheduler.priority_for({"gain": 0.1, "cost": 0}, cost_prior=1.0)
        with self.assertRaises(ScheduleError):
            Scheduler.priority_for(estimate, cost_prior=0)

    def test_invalid_option_values_are_rejected_with_type_and_range_checks(self):
        invalid = [
            ("stop_loss", []), ("stop_loss", {"web": {}}), ("stop_loss", {"spd": 3}),
            ("stop_loss", {"spd": {"first_n": 2}}),
            ("stop_loss", {"spd": {"first_n_no_gain": 0}}),
            ("stop_loss", {"spd": {"first_n_no_gain": True}}),
            ("stop_loss", {"tab": {"consecutive_no_gain": 2.5}}),
            ("stop_loss", {"tab": {"consecutive_no_gain": "3"}}),
            ("stop_loss", {"tab": {"consecutive_no_gain": 101}}),
            ("ceiling_excludes_baseline", "yes"), ("ceiling_excludes_baseline", 1),
            ("failure_feedback", "true"), ("failure_feedback", 0),
            ("planner_mode", "tiny"), ("planner_mode", 3),
            ("hypotheses_per_plan", 0), ("hypotheses_per_plan", 21),
            ("hypotheses_per_plan", 2.0), ("hypotheses_per_plan", True),
            ("cost_priors", []), ("cost_priors", {"web": 1}), ("cost_priors", {"spd": 0}),
            ("cost_priors", {"spd": -5}), ("cost_priors", {"tab": "8000"}),
            ("cost_priors", {"tab": True}), ("cost_priors", {"tab": float("inf")}),
            ("cost_priors", {"tab": float("nan")}),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ScheduleError):
                    scheduler(**{key: value})
                with self.assertRaises(ValueError):
                    validate_options({**SETTINGS, key: value})

    def test_valid_options_and_nulls_are_accepted(self):
        scheduler(
            stop_loss={"spd": {"first_n_no_gain": 2, "consecutive_no_gain": None}, "tab": {}},
            ceiling_excludes_baseline=False, planner_mode="lean", hypotheses_per_plan=4,
            failure_feedback=True, cost_priors={"spd": 2500, "tab": 8000.5},
        )
        scheduler(stop_loss=None, ceiling_excludes_baseline=None, planner_mode=None,
                  hypotheses_per_plan=None, failure_feedback=None, cost_priors=None)

    def test_shipped_configs_stay_valid_and_do_not_carry_the_new_keys(self):
        for path in sorted((ROOT / "config").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != "agent/1":
                continue
            with self.subTest(config=path.name):
                validate_options(data)
        for key in ("stop_loss", "ceiling_excludes_baseline", "planner_mode", "hypotheses_per_plan",
                    "failure_feedback", "cost_priors"):
            self.assertNotIn(key, CONFIG)


class RecordingModel:
    def __init__(self):
        self.calls = []

    def call_json(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data={})


class LeanPlannerTests(unittest.TestCase):
    def test_lean_schema_no_longer_requires_the_allocation(self):
        self.assertIn("budget_allocation", PLAN_SCHEMA["required"])
        self.assertNotIn("budget_allocation", PLAN_SCHEMA_LEAN["required"])
        self.assertEqual(set(PLAN_SCHEMA["required"]) - set(PLAN_SCHEMA_LEAN["required"]), {"budget_allocation"})
        self.assertEqual(PLAN_SCHEMA_LEAN["properties"], PLAN_SCHEMA["properties"])
        self.assertIs(PLAN_SCHEMA_LEAN["additionalProperties"], False)
        plan = {"hypotheses": [], "predicted": {"cost": 1, "seconds": 2, "score": 0.1}, "expected_progress": []}
        _validate_schema(plan, PLAN_SCHEMA_LEAN)  # omitted
        _validate_schema({**plan, "budget_allocation": []}, PLAN_SCHEMA_LEAN)  # empty
        _validate_schema({**plan, "budget_allocation": [{"direction": "x", "amount": 1}]}, PLAN_SCHEMA_LEAN)
        with self.assertRaises(ModelCallError):  # a filled allocation is still checked
            _validate_schema({**plan, "budget_allocation": "none"}, PLAN_SCHEMA_LEAN)
        with self.assertRaises(ModelCallError):  # the standard schema still demands it
            _validate_schema(plan, PLAN_SCHEMA)

    def test_model_client_path_accepts_a_lean_plan_without_allocation(self):
        plan = {"hypotheses": [], "predicted": {"cost": 1, "seconds": 2, "score": 0.1}, "expected_progress": []}
        for mode in ("lean", "standard"):
            with self.subTest(mode=mode):
                fake = FakeClient(response(plan))
                model = ModelClient("chosen-at-runtime", FakeLedger(), client=fake, clock=lambda: 1.0)
                roles = RoleAgents(model, {**CONFIG, "planner_mode": mode}, project_root=ROOT)
                if mode == "standard":
                    with self.assertRaisesRegex(ModelCallError, "budget_allocation"):
                        roles.plan({"state": 1})
                    continue
                self.assertEqual(roles.plan({"state": 1}), plan)
                sent = fake.responses.requests[0]["text"]["format"]["schema"]
                self.assertNotIn("budget_allocation", sent["required"])

    def test_planner_mode_selects_the_plan_schema_only(self):
        for mode, expected in ((None, SCHEMAS["plan"]), ("standard", SCHEMAS["plan"]), ("lean", PLAN_SCHEMA_LEAN)):
            with self.subTest(mode=mode):
                model = RecordingModel()
                config = {**CONFIG, **({} if mode is None else {"planner_mode": mode})}
                roles = RoleAgents(model, config, project_root=ROOT)
                roles.plan({"state": 1})
                roles.review({"state": 2})
                roles.diagnose({"state": 3})
                self.assertIs(model.calls[0]["schema"], expected)
                self.assertIs(model.calls[1]["schema"], SCHEMAS["review"])
                self.assertIs(model.calls[2]["schema"], SCHEMAS["diagnose"])

    def test_role_agents_reject_invalid_options_before_any_call(self):
        for key, value in (("planner_mode", "tiny"), ("failure_feedback", "yes"), ("hypotheses_per_plan", 0)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    RoleAgents(RecordingModel(), {**CONFIG, key: value}, project_root=ROOT)

    def test_lean_prompts_load_and_state_their_rules(self):
        config = {**CONFIG, "planner_mode": "lean", "hypotheses_per_plan": 3, "failure_feedback": True,
                  "prompts": {**CONFIG["prompts"], "planner": "config/prompts/planner_lean.md",
                              "executor": "config/prompts/executor_lean.md"}}
        roles = RoleAgents(RecordingModel(), config, project_root=ROOT)
        planner, executor = roles.prompts["planner"], roles.prompts["executor"]
        standard = (ROOT / "config/prompts/planner.md").read_text(encoding="utf-8")
        self.assertLess(len(planner), len(standard))
        self.assertIn("hypotheses_per_plan", planner)
        self.assertIn("不要输出 budget_allocation", planner)
        self.assertNotIn("安排方向", planner)  # the budget allocation duty is gone
        for rule in ("subprocess", "multiprocessing", "n_jobs", "联网", "空输入", "并列", "behavior_change",
                     "last_failure", "cheat_flags", "完整"):
            self.assertIn(rule, executor)
        original = (ROOT / "config/prompts/executor.md").read_text(encoding="utf-8")
        for rule in ("solution.py", "JSON schema", "hidden"):
            self.assertIn(rule, original)
            self.assertIn(rule, executor)


class StopLossReasonConsumerTests(unittest.TestCase):
    """Every reader of the stop reason must take the new value in stride."""

    def test_ledger_records_and_returns_the_new_reason(self):
        with TemporaryDirectory() as temp:
            run = ledger_fixture(Path(temp))
            ledger = Ledger.open(run, "spd-demo")
            ledger.emit("task_start", budget=1.0, wall_seconds=60)
            ledger.emit("stop", reason="stop_loss", best_node="n0000", dev_score=0.0)
            stop = [event for event in Ledger.open(run, "spd-demo")._events() if event["kind"] == "stop"]
            self.assertEqual([event["reason"] for event in stop], ["stop_loss"])

    def test_batch_summary_counts_and_labels_the_new_reason(self):
        def tweak(phase, config, rep, task_id, row):
            if (phase, config, rep, task_id) == ("exam", "default_lib", 2, "tab-exam-0"):
                row.update(stop="stop_loss")

        self.assertEqual(STOP_LABELS["stop_loss"], "止损停止")
        with TemporaryDirectory() as temp:
            fixture = build_formal_batch(Path(temp), tweak=tweak)
            summary = summarize(fixture.batch_dir)
            run = next(item for item in summary["runs"]
                       if (item["phase"], item["config"], item["rep"]) == ("exam", "default_lib", 2))
            self.assertEqual(run["stop_reasons"], {"ceiling": 2, "stop_loss": 1})
            self.assertEqual(summary["warnings"], [])
            markdown = render_markdown(summary)
            self.assertIn("达到评分上限 2、止损停止 1", markdown)
            self.assertIn("stop_loss=止损停止", markdown)

    def test_dashboard_timeline_passes_the_new_reason_through(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "runs" / "run-test"
            task = run / "tasks" / "spd-example"
            task.mkdir(parents=True)
            events = [{"schema": "ledger/1", "seq": 1, "kind": "task_start", "budget": 2.0},
                      {"schema": "ledger/1", "seq": 2, "kind": "stop", "reason": "stop_loss",
                       "best_node": "n0000", "dev_score": 0.2}]
            nodes = [{"schema": "node/1", "node_id": "n0000", "parent": None, "status": "accepted",
                      "hypothesis": "Baseline", "score": {"dev": 0.2, "raw": 1.0}}]
            (task / "ledger.jsonl").write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")
            (task / "tree.jsonl").write_text("".join(json.dumps(item) + "\n" for item in nodes), encoding="utf-8")
            timeline = load_run(run, project_root=Path(temp))["tasks"][0]["timeline"]
            self.assertEqual([row["reason"] for row in timeline if row["kind"] == "stop"], ["stop_loss"])


if __name__ == "__main__":
    unittest.main()
