from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_agent.model import BudgetReserveError, ModelCallError, ModelClient, TokenUsage, token_usage_from_response
from research_agent.model import _validate_schema
from research_agent.roles import RoleAgents, SCHEMAS


ROOT = Path(__file__).resolve().parents[1]


class FakeLedger:
    def __init__(self):
        self.calls = []
        self.money = 0.0

    def spent(self):
        return self.money

    def llm_call(self, **kwargs):
        self.calls.append(kwargs)
        return 0.125


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.responses = FakeResponses(response, error)


def response(data, *, status="completed", input_tokens=100, cached_tokens=25, output_tokens=80, reasoning_tokens=30):
    return SimpleNamespace(
        id="resp-test",
        status=status,
        output_text=json.dumps(data, ensure_ascii=False),
        output=[],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            output_tokens=output_tokens,
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


class ModelTests(unittest.TestCase):
    def test_default_transport_disables_unmetered_sdk_retries(self):
        with patch("openai.OpenAI") as constructor:
            ModelClient("chosen-at-runtime", None)
        constructor.assert_called_once_with(timeout=180.0, max_retries=0)

    def test_integer_schema_accepts_int_and_rejects_bool_or_float(self):
        _validate_schema(4, {"type": "integer"})
        for invalid in (True, 4.0):
            with self.assertRaises(ModelCallError):
                _validate_schema(invalid, {"type": "integer"})

    def test_usage_is_disjoint_for_ledger_pricing(self):
        self.assertEqual(token_usage_from_response(response({})), TokenUsage(75, 25, 50, 30))
        with self.assertRaises(ModelCallError):
            token_usage_from_response(response({}, input_tokens=10, cached_tokens=11))

    def test_responses_request_and_success_are_logged(self):
        ledger = FakeLedger()
        fake = FakeClient(response({"code": "def solve(): pass", "notes": "ok"}))
        times = iter([10.0, 12.25])
        client = ModelClient("chosen-at-runtime", ledger, client=fake, clock=lambda: next(times))
        result = client.call_json(
            role="executor",
            purpose="code",
            effort="low",
            system_prompt="Fixed executor instructions",
            payload={"task": "task-specific state"},
            schema=SCHEMAS["code"],
            node_id="n0001",
        )
        request = fake.responses.requests[0]
        self.assertEqual(request["model"], "chosen-at-runtime")
        self.assertTrue(request["instructions"].startswith("Fixed executor instructions"))
        self.assertIn('"code"', request["instructions"])
        self.assertEqual(json.loads(request["input"]), {"task": "task-specific state"})
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertEqual(result.data["code"], "def solve(): pass")
        self.assertEqual(result.seconds, 2.25)
        self.assertEqual(result.cost, 0.125)
        self.assertEqual(ledger.calls[0]["usage"], TokenUsage(75, 25, 50, 30))
        self.assertEqual(ledger.calls[0]["node_id"], "n0001")
        self.assertTrue(ledger.calls[0]["ok"])
        self.assertEqual(len(result.prompt_hash), 64)

    def test_model_failure_is_logged_as_failed_call(self):
        ledger = FakeLedger()
        fake = FakeClient(error=RuntimeError("transport unavailable"))
        times = iter([0.0, 0.4])
        client = ModelClient("chosen-at-runtime", ledger, client=fake, clock=lambda: next(times))
        with self.assertRaisesRegex(ModelCallError, "transport unavailable"):
            client.call_json(
                role="planner", purpose="plan", effort="high", system_prompt="Plan",
                payload={}, schema=SCHEMAS["plan"],
            )
        self.assertEqual(ledger.calls[0]["usage"], TokenUsage())
        self.assertFalse(ledger.calls[0]["ok"])
        self.assertIn("transport unavailable", ledger.calls[0]["error"])

    def test_incomplete_response_books_returned_tokens(self):
        ledger = FakeLedger()
        fake = FakeClient(response({}, status="incomplete"))
        times = iter([0.0, 0.2])
        client = ModelClient("chosen-at-runtime", ledger, client=fake, clock=lambda: next(times))
        with self.assertRaises(ModelCallError):
            client.call_json(
                role="planner", purpose="review", effort="high", system_prompt="Review",
                payload={}, schema=SCHEMAS["review"],
            )
        self.assertEqual(ledger.calls[0]["usage"], TokenUsage(75, 25, 50, 30))
        self.assertFalse(ledger.calls[0]["ok"])

    def test_reserve_blocks_dispatch_without_booking_a_call(self):
        ledger = FakeLedger()
        ledger.money = 36_001
        fake = FakeClient(response({"code": "pass"}))
        client = ModelClient("chosen-at-runtime", ledger, client=fake,
                             budget_limit=40_000, reserve_cost=5_000)
        with self.assertRaises(BudgetReserveError):
            client.call_json(role="executor", purpose="code", effort="low",
                             system_prompt="Code", payload={}, schema=SCHEMAS["code"])
        self.assertEqual(fake.responses.requests, [])
        self.assertEqual(ledger.calls, [])

    def test_reserve_allows_call_at_exact_boundary(self):
        ledger = FakeLedger()
        ledger.money = 35_000
        fake = FakeClient(response({"code": "pass", "notes": ""}))
        client = ModelClient("chosen-at-runtime", ledger, client=fake,
                             budget_limit=40_000, reserve_cost=5_000,
                             clock=lambda: 1.0)
        client.call_json(role="executor", purpose="code", effort="low",
                         system_prompt="Code", payload={}, schema=SCHEMAS["code"])
        self.assertEqual(len(fake.responses.requests), 1)
        self.assertEqual(len(ledger.calls), 1)

    def test_invalid_json_is_booked_and_visible_in_response_audit(self):
        ledger = FakeLedger()
        fake = FakeClient(response({}))
        fake.responses.response.output_text = "not-json"
        with TemporaryDirectory() as temp:
            client = ModelClient("chosen-at-runtime", ledger, client=fake,
                                 audit_dir=Path(temp), clock=lambda: 1.0)
            with self.assertRaises(ModelCallError):
                client.call_json(role="planner", purpose="plan", effort="low",
                                 system_prompt="Plan", payload={}, schema=SCHEMAS["plan"])
            audits = list(Path(temp).glob("response-*.json"))
            self.assertEqual(len(audits), 1)
            saved = json.loads(audits[0].read_text())
            self.assertEqual(saved["output_text"], "not-json")
            self.assertIsNotNone(saved["error"])
            self.assertEqual(len(ledger.calls), 1)

    def test_schema_shape_mismatch_is_not_treated_as_empty_plan(self):
        ledger = FakeLedger()
        fake = FakeClient(response({"hypotheses": [], "budget_plan": {}}))
        client = ModelClient("chosen-at-runtime", ledger, client=fake,
                             clock=lambda: 1.0)
        with self.assertRaisesRegex(ModelCallError, "missing"):
            client.call_json(role="planner", purpose="plan", effort="high",
                             system_prompt="Plan", payload={}, schema=SCHEMAS["plan"])
        self.assertFalse(ledger.calls[0]["ok"])

    def test_single_json_code_fence_is_accepted_but_extra_prose_is_not(self):
        ledger = FakeLedger()
        fake = FakeClient(response({"code": "pass", "notes": "ok"}))
        fake.responses.response.output_text = '```json\n{"code":"pass","notes":"ok"}\n```'
        client = ModelClient("chosen-at-runtime", ledger, client=fake,
                             clock=lambda: 1.0)
        result = client.call_json(role="executor", purpose="code", effort="low",
                                  system_prompt="Code", payload={}, schema=SCHEMAS["code"])
        self.assertEqual(result.data["code"], "pass")
        fake.responses.response.output_text += '\nAdditional prose'
        with self.assertRaises(ModelCallError):
            client.call_json(role="executor", purpose="code", effort="low",
                             system_prompt="Code", payload={}, schema=SCHEMAS["code"])


class RolesTests(unittest.TestCase):
    def test_prompt_path_cannot_escape_project_root(self):
        config = json.loads((ROOT / "config/agent.json").read_text(encoding="utf-8"))
        config["prompts"]["planner"] = "../outside-prompt.md"

        class UnusedModel:
            pass

        with self.assertRaisesRegex(ValueError, "escapes project root"):
            RoleAgents(UnusedModel(), config, project_root=ROOT)

    def test_all_roles_use_configured_effort_and_stable_keys(self):
        config = json.loads((ROOT / "config/agent.json").read_text(encoding="utf-8"))

        class RecordingModel:
            def __init__(self):
                self.calls = []

            def call_json(self, **kwargs):
                self.calls.append(kwargs)
                values = {
                    "plan": {"hypotheses": [], "predicted": {"cost": 1, "seconds": 2, "score": 0.1}},
                    "code": {"code": "def f(): pass", "notes": ""},
                    "fix": {"code": "def f(): return 1", "notes": "fixed"},
                    "diagnose": {"hypothesis": "bad input", "rationale": "error", "advice": "validate"},
                    "review": {"decision": "continue", "budget_moves": [], "note": "gain"},
                    "summarize": {"drafts": [], "narrowings": []},
                }
                return SimpleNamespace(data=values[kwargs["purpose"]])

        fake = RecordingModel()
        roles = RoleAgents(fake, config, project_root=ROOT)
        self.assertIn("predicted", roles.plan({"state": 1}))
        self.assertIn("code", roles.execute({"state": 2}))
        self.assertEqual(roles.execute({"error": "x"}, fix=True)["notes"], "fixed")
        self.assertIn("advice", roles.diagnose({"error": "x"}))
        self.assertEqual(roles.review({"state": 3})["decision"], "continue")
        summary = roles.summarize({"state": 4})
        self.assertEqual(summary["drafts"], [])
        self.assertEqual(summary["narrowings"], [])
        self.assertEqual([call["effort"] for call in fake.calls],
                         ["high", "low", "low", "high", "high", "medium"])
        self.assertTrue(all(call["system_prompt"].strip() for call in fake.calls))
        self.assertTrue(all(call["schema"]["additionalProperties"] is False for call in fake.calls))
        summary_schema = fake.calls[-1]["schema"]
        self.assertIn("narrowings", summary_schema["required"])
        self.assertIn("node_ids", summary_schema["properties"]["drafts"]["items"]["required"])
        self.assertEqual(
            summary_schema["properties"]["narrowings"]["items"]["required"],
            ["entry_id", "conditions", "reason"],
        )


if __name__ == "__main__":
    unittest.main()
