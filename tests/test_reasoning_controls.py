"""Per-role thinking switch and output limits, plus the summarizer's reserve rule.

Everything runs against fake transports; no gateway is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from library import Library
from research_agent.model import (BudgetReserveError, ModelClient, TokenUsage,
                                  parse_role_controls)
from research_agent.roles import RoleAgents, SCHEMAS
from research_agent.runner import TaskRunner
from tests.test_model_roles import FakeClient, FakeLedger, response
from tests.test_runner import CAPS, CONFIG, FakeHarness, idea, score

ROOT = Path(__file__).resolve().parents[1]
AGENT_CONFIG = json.loads((ROOT / "config/agent.json").read_text(encoding="utf-8"))
DEFAULT_REQUEST_KEYS = {"model", "instructions", "input", "reasoning", "text", "max_output_tokens"}


def make_client(**kwargs):
    ledger = FakeLedger()
    fake = FakeClient(response({"code": "pass", "notes": ""}, reasoning_tokens=0))
    return ModelClient("chosen-at-runtime", ledger, client=fake, clock=lambda: 1.0, **kwargs), fake, ledger


def call(client, **kwargs):
    return client.call_json(role="executor", purpose="code", effort="low",
                            system_prompt="Code", payload={}, schema=SCHEMAS["code"], **kwargs)


class ThinkingSwitchTests(unittest.TestCase):
    def test_default_request_is_unchanged(self):
        client, fake, _ = make_client()
        call(client)
        request = fake.responses.requests[0]
        self.assertEqual(set(request), DEFAULT_REQUEST_KEYS)
        self.assertEqual(request["max_output_tokens"], 30000)
        self.assertEqual(request["reasoning"], {"effort": "low"})

    def test_thinking_none_or_true_adds_nothing(self):
        client, fake, _ = make_client()
        call(client, thinking=None)
        call(client, thinking=True, thinking_switch_style="chat_template_kwargs")
        self.assertEqual([set(r) for r in fake.responses.requests], [DEFAULT_REQUEST_KEYS] * 2)

    def test_thinking_off_uses_enable_thinking_by_default(self):
        client, fake, _ = make_client()
        call(client, thinking=False)
        request = fake.responses.requests[0]
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(set(request), DEFAULT_REQUEST_KEYS | {"extra_body"})

    def test_chat_template_style_from_call_or_client(self):
        expected = {"chat_template_kwargs": {"enable_thinking": False}}
        client, fake, _ = make_client()
        call(client, thinking=False, thinking_switch_style="chat_template_kwargs")
        self.assertEqual(fake.responses.requests[0]["extra_body"], expected)
        client, fake, _ = make_client(thinking_switch_style="chat_template_kwargs")
        call(client, thinking=False)
        self.assertEqual(fake.responses.requests[0]["extra_body"], expected)
        call(client, thinking=False, thinking_switch_style="enable_thinking")
        self.assertEqual(fake.responses.requests[1]["extra_body"], {"enable_thinking": False})

    def test_thinking_off_is_a_different_prompt_but_default_hash_is_stable(self):
        client, _, ledger = make_client()
        call(client)
        call(client, thinking=True)
        call(client, thinking=False)
        hashes = [item["prompt_hash"] for item in ledger.calls]
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])

    def test_thinking_off_still_books_all_four_buckets(self):
        client, _, ledger = make_client()
        call(client, thinking=False)
        self.assertEqual(ledger.calls[0]["usage"], TokenUsage(75, 25, 80, 0))
        self.assertTrue(ledger.calls[0]["ok"])
        self.assertEqual(ledger.calls[0]["effort"], "low")

    def test_invalid_controls_are_rejected_before_any_request(self):
        client, fake, ledger = make_client()
        for kwargs in ({"thinking": "no"}, {"thinking": 0}, {"thinking_switch_style": "nope"},
                       {"max_output_tokens": 0}, {"max_output_tokens": True},
                       {"max_output_tokens": 1.5}, {"max_output_tokens": "9"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                call(client, **kwargs)
        with self.assertRaises(ValueError):
            make_client(thinking_switch_style="nope")
        self.assertEqual(fake.responses.requests, [])
        self.assertEqual(ledger.calls, [])

    def test_audit_file_records_the_controls(self):
        with TemporaryDirectory() as temp:
            client, _, _ = make_client(audit_dir=Path(temp), max_output_tokens=2048)
            call(client, thinking=False, max_output_tokens=500)
            saved = json.loads(next(Path(temp).glob("response-*.json")).read_text(encoding="utf-8"))
            self.assertIs(saved["thinking"], False)
            self.assertEqual(saved["max_output_tokens"], 500)


class OutputLimitTests(unittest.TestCase):
    def test_role_limit_can_only_lower_the_global_limit(self):
        client, fake, _ = make_client(max_output_tokens=2048)
        call(client)
        call(client, max_output_tokens=500)
        call(client, max_output_tokens=5000)
        self.assertEqual([r["max_output_tokens"] for r in fake.responses.requests], [2048, 500, 2048])


class SummarizerReserveTests(unittest.TestCase):
    """The token study holds every call to a 12,000-token reserve (30% of 40,000)."""

    def summarize(self, ledger, fake, **kwargs):
        kwargs.setdefault("wrapup_reserve", True)  # agent/1 summarizer_reserve: "worst_case"
        client = ModelClient("m", ledger, client=fake, clock=lambda: 1.0, max_output_tokens=2048,
                             budget_limit=40_000, reserve_cost=12_000)
        return client.call_json(role="summarizer", purpose="summarize", effort="low",
                                system_prompt="Summarize", payload=kwargs.pop("payload", {}),
                                schema=SCHEMAS["summarize"], **kwargs)

    def plain_client(self, ledger, fake, **kwargs):
        return ModelClient("m", ledger, client=fake, clock=lambda: 1.0,
                           budget_limit=kwargs.pop("budget_limit", 40_000),
                           reserve_cost=kwargs.pop("reserve_cost", 12_000), **kwargs)

    def test_other_roles_still_need_the_full_reserve(self):
        ledger, fake = FakeLedger(), FakeClient(response({"code": "pass", "notes": ""}))
        ledger.money = 30_000
        client = self.plain_client(ledger, fake)
        with self.assertRaises(BudgetReserveError):
            call(client)
        self.assertEqual(fake.responses.requests, [])

    def test_without_opt_in_the_summarizer_keeps_the_full_reserve(self):
        ledger, fake = FakeLedger(), FakeClient(response({"drafts": [], "narrowings": []}))
        ledger.money = 30_000
        with self.assertRaises(BudgetReserveError):
            self.summarize(ledger, fake, payload={"experiments": []}, wrapup_reserve=False)
        self.assertEqual(fake.responses.requests, [])

    def test_summarizer_runs_after_the_experiments_spent_past_28000(self):
        ledger, fake = FakeLedger(), FakeClient(response({"drafts": [], "narrowings": []}))
        ledger.money = 33_000
        self.summarize(ledger, fake, payload={"experiments": []})
        self.assertEqual(len(fake.responses.requests), 1)
        self.assertEqual(len(ledger.calls), 1)

    def test_summarizer_is_still_bound_by_the_task_budget(self):
        ledger, fake = FakeLedger(), FakeClient(response({"drafts": [], "narrowings": []}))
        for money in (39_000, 39_990, 40_000, 40_001):
            ledger.money = money
            with self.subTest(spent=money), self.assertRaises(BudgetReserveError):
                self.summarize(ledger, fake, payload={"experiments": []})
        self.assertEqual(fake.responses.requests, [])
        self.assertEqual(ledger.calls, [])

    def test_worst_case_counts_input_size_and_output_cap(self):
        ledger, fake = FakeLedger(), FakeClient(response({"drafts": [], "narrowings": []}))
        ledger.money = 37_500
        with self.assertRaises(BudgetReserveError):  # a 2,048-token reply could run past the budget
            self.summarize(ledger, fake, payload={"experiments": []})
        self.summarize(ledger, fake, payload={"experiments": []}, max_output_tokens=300)
        self.assertEqual(fake.responses.requests[-1]["max_output_tokens"], 300)
        ledger.money = 36_000  # 4,000 tokens left: a small request fits, a 6 kB one does not
        self.summarize(ledger, fake, payload={"experiments": []}, max_output_tokens=300)
        with self.assertRaises(BudgetReserveError):
            self.summarize(ledger, fake, payload={"experiments": ["x" * 60] * 100}, max_output_tokens=300)
        self.assertEqual(len(fake.responses.requests), 2)

    def test_summarizer_reserve_never_exceeds_the_configured_reserve(self):
        ledger, fake = FakeLedger(), FakeClient(response({"drafts": [], "narrowings": []}))
        client = self.plain_client(ledger, fake, budget_limit=5.0, reserve_cost=1.5)  # not token units
        ledger.money = 3.6
        with self.assertRaises(BudgetReserveError):
            client.call_json(role="summarizer", purpose="summarize", effort="low",
                             system_prompt="Summarize", payload={}, schema=SCHEMAS["summarize"])
        ledger.money = 3.4
        client.call_json(role="summarizer", purpose="summarize", effort="low",
                         system_prompt="Summarize", payload={}, schema=SCHEMAS["summarize"])
        self.assertEqual(len(fake.responses.requests), 1)


class RoleControlConfigTests(unittest.TestCase):
    def config(self, **extra):
        return {**AGENT_CONFIG, **extra}

    def test_missing_keys_mean_no_new_arguments_on_any_call(self):
        class Recorder:
            def __init__(self):
                self.calls = []

            def call_json(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(data={})

        model = Recorder()
        roles = RoleAgents(model, self.config(), project_root=ROOT)
        roles.plan({}), roles.execute({}), roles.diagnose({}), roles.review({}), roles.summarize({})
        for kwargs in model.calls:
            self.assertEqual(set(kwargs), {"role", "purpose", "effort", "system_prompt",
                                           "payload", "schema", "node_id"})

    def test_controls_reach_only_the_roles_that_set_them(self):
        class Recorder:
            def __init__(self):
                self.calls = {}

            def call_json(self, **kwargs):
                self.calls[kwargs["role"]] = kwargs
                return SimpleNamespace(data={})

        model = Recorder()
        roles = RoleAgents(model, self.config(
            thinking_switch_style="chat_template_kwargs",
            role_controls={"summarizer": {"thinking": False, "max_output_tokens": 900},
                           "executor": {"max_output_tokens": 700},
                           "planner": {"thinking": True}}), project_root=ROOT)
        roles.plan({}), roles.execute({}), roles.diagnose({}), roles.summarize({})
        extras = {role: {key: value for key, value in kwargs.items()
                         if key in {"thinking", "max_output_tokens", "thinking_switch_style"}}
                  for role, kwargs in model.calls.items()}
        self.assertEqual(extras["summarizer"], {"thinking": False, "max_output_tokens": 900,
                                                "thinking_switch_style": "chat_template_kwargs"})
        self.assertEqual(extras["executor"], {"max_output_tokens": 700})
        self.assertEqual(extras["planner"], {"thinking": True})
        self.assertEqual(extras["diagnoser"], {})

    def test_style_defaults_to_enable_thinking(self):
        class Recorder:
            def call_json(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(data={})

        model = Recorder()
        RoleAgents(model, self.config(role_controls={"summarizer": {"thinking": False}}),
                   project_root=ROOT).summarize({})
        self.assertEqual(model.kwargs["thinking_switch_style"], "enable_thinking")

    def test_invalid_settings_fail_when_roles_are_built(self):
        bad = [
            {"role_controls": []},
            {"role_controls": {"manager": {"thinking": False}}},
            {"role_controls": {"planner": "off"}},
            {"role_controls": {"planner": {"effort": "low"}}},
            {"role_controls": {"planner": {"thinking": "false"}}},
            {"role_controls": {"planner": {"max_output_tokens": 0}}},
            {"role_controls": {"planner": {"max_output_tokens": True}}},
            {"role_controls": {"planner": {"max_output_tokens": "500"}}},
            {"thinking_switch_style": "kwargs"},
            {"experience_writer": "scripted"},
            {"experience_writer": None},
        ]
        for extra in bad:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                RoleAgents(object(), self.config(**extra), project_root=ROOT)
        self.assertEqual(parse_role_controls({}), {})

    def test_through_the_real_client_the_summarizer_thinks_less_and_writes_less(self):
        fake = FakeClient(response({"drafts": [], "narrowings": []}, reasoning_tokens=0))
        ledger = FakeLedger()
        model = ModelClient("m", ledger, client=fake, clock=lambda: 1.0, max_output_tokens=2048)
        roles = RoleAgents(model, self.config(
            role_controls={"summarizer": {"thinking": False, "max_output_tokens": 900}}),
            project_root=ROOT)
        roles.summarize({"experiments": []})
        request = fake.responses.requests[0]
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertEqual(request["max_output_tokens"], 900)
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertEqual(ledger.calls[0]["role"], "summarizer")


class TokenLedger:
    """Books provider-reported tokens as cost, like the TOKENS price table."""

    def __init__(self):
        self.events, self.money, self.llm_seconds = [], 0.0, 0.0

    def emit(self, kind, *, node_id=None, **values):
        self.events.append({"kind": kind, "node_id": node_id, **values})

    def spent(self):
        return self.money

    def elapsed(self):
        return {"llm": self.llm_seconds, "run": 0.0, "queue": 0.0, "wall": 0.0}

    def llm_call(self, **kwargs):
        usage = kwargs["usage"]
        cost = float(usage.input + usage.cached_input + usage.output + usage.reasoning)
        self.money += cost
        self.events.append({"kind": "llm_call", "role": kwargs["role"], "cost": cost, "ok": kwargs["ok"]})
        return cost


class ScriptedGateway:
    """One fake transport answering every role; tokens per call are fixed per role."""

    PLAN_TOKENS = (3_000, 3_000)
    EXECUTOR_TOKENS = 12_000
    SUMMARIZER_TOKENS = 2_500

    def __init__(self):
        self.requests = []
        self.plans = 0
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        name = request["text"]["format"]["name"]
        if name == "planner_plan":
            tokens = self.PLAN_TOKENS[min(self.plans, len(self.PLAN_TOKENS) - 1)]
            self.plans += 1
            data = {"hypotheses": [idea("good", 0.3), idea("also good", 0.2)] if self.plans == 1 else [],
                    "predicted": {"cost": 1.0, "seconds": 10, "score": 0.5},
                    "budget_allocation": [], "expected_progress": []}
        elif name == "executor_code":
            tokens = self.EXECUTOR_TOKENS
            data = {"code": f"solution {len(self.requests)}", "notes": ""}
        else:
            tokens = self.SUMMARIZER_TOKENS
            data = {"drafts": [{"scope": "domain", "scenario": "tabular", "triggers": ["missing_heavy"],
                                "conditions": "", "action": "Check missing values", "negative": False,
                                "node_ids": ["n0001"]}], "narrowings": []}
        return SimpleNamespace(
            id=f"r{len(self.requests)}", status="completed", output=[], output_text=json.dumps(data),
            usage=SimpleNamespace(
                input_tokens=tokens - 100, input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens=100, output_tokens_details=SimpleNamespace(reasoning_tokens=0)))


class SummarizerAfterBudgetSpentTests(unittest.TestCase):
    """The reported bug: past 28,000 tokens the gate refused the summarizer, so no experience was written."""

    WORST_CASE = {**CONFIG, "summarizer_reserve": "worst_case"}

    def run_task(self, root, gateway, config=WORST_CASE):
        ledger = TokenLedger()
        model = ModelClient("m", ledger, client=gateway, clock=lambda: 1.0, max_output_tokens=2048,
                            budget_limit=40_000, reserve_cost=12_000)
        roles = RoleAgents(model, config, project_root=ROOT)
        library = Library.open(root / "runs/run-test/library")
        outcome = TaskRunner(run_dir=root / "runs/run-test", task_id="tab-sample",
                             harness=FakeHarness(root, money=40_000), ledger=ledger, roles=roles,
                             library=library, baseline_score=score(0.1), config=config,
                             hard_caps=CAPS).run()
        return outcome, ledger, library

    def test_summarizer_still_runs_and_writes_after_spending_30000(self):
        with TemporaryDirectory() as temp:
            gateway = ScriptedGateway()
            outcome, ledger, library = self.run_task(Path(temp), gateway)
            names = [request["text"]["format"]["name"] for request in gateway.requests]
            self.assertEqual(names.count("executor_code"), 2)
            self.assertEqual(names[-1], "summarizer_summarize")
            self.assertEqual(outcome.reason, "planner")
            before_summary = sum(e["cost"] for e in ledger.events
                                 if e["kind"] == "llm_call" and e["role"] != "summarizer")
            self.assertEqual(before_summary, 30_000)  # old gate: 30,000 + 12,000 > 40,000
            self.assertEqual(len(library.entries), 1)
            self.assertEqual(ledger.money, 32_500)
            self.assertTrue(any(e["kind"] == "library" and e["op"] == "write" for e in ledger.events))

    def test_default_config_keeps_the_old_gate(self):
        with TemporaryDirectory() as temp:
            gateway = ScriptedGateway()
            outcome, ledger, library = self.run_task(Path(temp), gateway, config=CONFIG)
            self.assertNotIn("summarizer_summarize",
                             [request["text"]["format"]["name"] for request in gateway.requests])
            self.assertEqual(outcome.reason, "budget_cap")
            self.assertEqual(ledger.money, 30_000)
            self.assertEqual(len(library.entries), 0)

    def test_unknown_summarizer_reserve_is_rejected(self):
        with self.assertRaises(ValueError):
            RoleAgents(object(), {**CONFIG, "summarizer_reserve": "bogus"}, project_root=ROOT)

    def test_a_summary_that_cannot_fit_is_skipped_and_the_task_stays_in_budget(self):
        class LateExpensivePlan(ScriptedGateway):
            PLAN_TOKENS = (3_000, 11_000)  # leaves 2,000 tokens: not enough for a summary

        with TemporaryDirectory() as temp:
            gateway = LateExpensivePlan()
            outcome, ledger, library = self.run_task(Path(temp), gateway)
            self.assertEqual(ledger.money, 38_000)
            self.assertNotIn("summarizer_summarize",
                             [request["text"]["format"]["name"] for request in gateway.requests])
            self.assertEqual(outcome.reason, "budget_cap")
            self.assertEqual(len(library.entries), 0)


if __name__ == "__main__":
    unittest.main()
