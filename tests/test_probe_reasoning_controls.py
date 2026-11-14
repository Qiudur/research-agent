"""The reasoning-control probe, always against a fake client."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import probe_reasoning_controls as probe

KEY = "sk-test-secret-0123456789"
URL = "https://gateway.example.internal:8443/v1"
ENV = {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": URL}


def reply(*, reasoning=0, text='{"answer": 42}', input_tokens=200, output_extra=10):
    return SimpleNamespace(
        id="r", status="completed", output=[], output_text=text,
        usage=SimpleNamespace(
            input_tokens=input_tokens, input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens=reasoning + output_extra,
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning)))


class ScriptedClient:
    """Answers each request with whatever the handler returns or raises."""

    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        return self.handler(request, len(self.requests))


def run_main(argv, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        code = probe.main(argv, stdout=out, **kwargs)
    return code, out.getvalue(), err.getvalue()


class DryRunTests(unittest.TestCase):
    def test_dry_run_needs_no_network_and_no_environment(self):
        with patch("openai.OpenAI", side_effect=AssertionError("dry run must not build a client")):
            code, out, err = run_main(["--dry-run"], environ={})
        self.assertEqual((code, err), (0, ""))
        for name in ("effort=low", "effort=medium", "effort=high", "off: enable_thinking",
                     "off: chat_template_kwargs"):
            self.assertIn(name, out)
        self.assertIn("DRY RUN", out)
        for column in ("ok", "avg_input", "avg_output", "avg_reasoning"):
            self.assertIn(column, out)
        self.assertIn("total calls=15", out)

    def test_dry_run_reports_each_switch_as_working(self):
        _, out, _ = run_main(["--dry-run"], environ={})
        self.assertEqual(out.count("switch works"), 2)


class CallPlanTests(unittest.TestCase):
    def test_default_plan_is_fifteen_calls_in_five_groups_of_three(self):
        client = ScriptedClient(lambda request, n: reply(reasoning=900 if "extra_body" not in request else 0))
        code, out, _ = run_main([], client=client, environ={})
        self.assertEqual(code, 0)
        requests = client.requests
        self.assertEqual(len(requests), 15)
        self.assertEqual([r["reasoning"]["effort"] for r in requests],
                         ["low"] * 3 + ["medium"] * 3 + ["high"] * 3 + ["low"] * 6)
        self.assertTrue(all("extra_body" not in r for r in requests[:9]))
        self.assertEqual([r["extra_body"] for r in requests[9:12]], [{"enable_thinking": False}] * 3)
        self.assertEqual([r["extra_body"] for r in requests[12:]],
                         [{"chat_template_kwargs": {"enable_thinking": False}}] * 3)
        self.assertTrue(all(r["model"] == probe.MODEL for r in requests))
        self.assertTrue(all(json.loads(r["input"]) == probe.PAYLOAD for r in requests))

    def test_a_bigger_plan_is_refused_before_anything_is_sent(self):
        client = ScriptedClient(lambda request, n: reply())
        code, out, err = run_main(["--calls-per-setting", "5"], client=client, environ={})
        self.assertEqual((code, out), (2, ""))
        self.assertIn("25", err)
        self.assertIn("--max-total-calls", err)
        self.assertEqual(client.requests, [])

    def test_a_bigger_plan_runs_when_the_limit_is_raised_explicitly(self):
        client = ScriptedClient(lambda request, n: reply())
        code, _, _ = run_main(["--calls-per-setting", "5", "--max-total-calls", "25"],
                              client=client, environ={})
        self.assertEqual((code, len(client.requests)), (0, 25))

    def test_the_default_limit_is_fifteen_calls(self):
        self.assertEqual(len(probe.SETTINGS) * probe.DEFAULT_CALLS_PER_SETTING, 15)
        self.assertEqual(probe.DEFAULT_MAX_TOTAL_CALLS, 15)

    def test_output_limit_is_passed_through(self):
        client = ScriptedClient(lambda request, n: reply())
        run_main(["--max-output-tokens", "300", "--calls-per-setting", "1"], client=client, environ={})
        self.assertEqual({r["max_output_tokens"] for r in client.requests}, {300})


class SummaryTests(unittest.TestCase):
    def test_averages_per_setting_and_verdict_for_a_switch_that_does_nothing(self):
        def handler(request, n):
            effort_group = (n - 1) // 3
            return reply(reasoning=[1000, 2000, 3000, 3000, 0][effort_group] + 100 * ((n - 1) % 3 == 0),
                         input_tokens=200 + effort_group)

        client = ScriptedClient(handler)
        code, out, _ = run_main([], client=client, environ={})
        rows = {row["setting"]: row for row in probe.run_probe(
            ScriptedClient(handler), calls_per_setting=3)}
        self.assertEqual(code, 0)
        self.assertAlmostEqual(rows["effort=low"]["avg_reasoning"], (1100 + 1000 + 1000) / 3)
        self.assertAlmostEqual(rows["effort=high"]["avg_reasoning"], (3100 + 3000 + 3000) / 3)
        self.assertEqual(rows["effort=medium"]["avg_input"], 201)
        self.assertEqual(rows["effort=low"]["avg_output"], 10)
        self.assertIn("off: enable_thinking: reasoning is", out)
        self.assertIn("no clear effect", out)  # still thinks as much as effort=low
        self.assertIn("switch works", out)  # the second switch answered with zero reasoning

    def test_failed_calls_are_counted_and_still_book_their_usage(self):
        def handler(request, n):
            return reply(reasoning=50, text="not json" if n % 3 == 0 else '{"answer": 42}')

        rows = probe.run_probe(ScriptedClient(handler), calls_per_setting=3)
        first = rows[0]
        self.assertEqual((first["ok"], first["correct"], first["with_usage"]), (2, 2, 3))
        self.assertEqual(first["avg_reasoning"], 50)
        self.assertEqual(len(first["errors"]), 1)
        code, out, _ = run_main([], client=ScriptedClient(handler), environ={})
        self.assertEqual(code, 1)
        self.assertIn("errors:", out)

    def test_a_switch_the_gateway_rejects_is_not_reported_as_working(self):
        def handler(request, n):
            if "extra_body" in request:
                raise RuntimeError("400 unknown field")
            return reply(reasoning=1500)

        code, out, _ = run_main([], client=ScriptedClient(handler), environ={})
        self.assertEqual(code, 1)
        self.assertNotIn("switch works", out)
        self.assertEqual(out.count("no usable reply"), 2)

    def test_a_half_failing_switch_says_so(self):
        def handler(request, n):
            if "extra_body" in request and n % 2:
                raise RuntimeError("500")
            return reply(reasoning=0 if "extra_body" in request else 1500)

        _, out, _ = run_main(["--calls-per-setting", "2"], client=ScriptedClient(handler), environ={})
        self.assertIn("switch works", out)
        self.assertIn("calls failed", out)

    def test_wrong_answers_are_visible_separately_from_call_success(self):
        rows = probe.run_probe(ScriptedClient(lambda r, n: reply(text='{"answer": 41}')), calls_per_setting=1)
        self.assertEqual((rows[0]["ok"], rows[0]["correct"]), (1, 0))

    def test_calls_without_any_usage_do_not_crash_the_table(self):
        def handler(request, n):
            raise RuntimeError("connection reset")

        code, out, _ = run_main(["--calls-per-setting", "1"], client=ScriptedClient(handler), environ={})
        self.assertEqual(code, 1)
        self.assertIn("n/a", out)
        self.assertIn("connection reset", out)


class SecretHandlingTests(unittest.TestCase):
    def test_key_and_address_never_reach_stdout_stderr_or_the_json_file(self):
        def handler(request, n):
            raise RuntimeError(f"cannot reach {URL} with key {KEY} at host gateway.example.internal:8443")

        with TemporaryDirectory() as temp:
            target = Path(temp) / "probe.json"
            code, out, err = run_main(["--calls-per-setting", "1", "--json-out", str(target)],
                                      client=ScriptedClient(handler), environ=ENV)
            saved = target.read_text(encoding="utf-8")
        self.assertEqual(code, 1)
        for shown in (out, err, saved):
            for secret in (KEY, URL, "gateway.example.internal", "8443"):
                self.assertNotIn(secret, shown)
        self.assertIn("<hidden>", out)
        self.assertEqual(json.loads(saved)["schema"], "reasoning_probe/1")
        self.assertEqual(len(json.loads(saved)["rows"]), 5)

    def test_bare_host_and_url_credentials_are_scrubbed_too(self):
        env = {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": "https://svc-user:p4ssw0rd@gw.example.internal:8443/v1"}

        def handler(request, n):
            raise RuntimeError("dns failure for gw.example.internal (login svc-user, password p4ssw0rd)")

        code, out, err = run_main(["--calls-per-setting", "1"], client=ScriptedClient(handler), environ=env)
        self.assertEqual(code, 1)
        for secret in (KEY, "gw.example.internal", "svc-user", "p4ssw0rd", "8443"):
            self.assertNotIn(secret, out + err)
        self.assertEqual(probe._secrets({"OPENAI_BASE_URL": "http://[bad"}), ["http://[bad"])

    def test_live_mode_names_the_missing_variables_but_never_a_value(self):
        for env, expected, absent in (({}, ("OPENAI_API_KEY", "OPENAI_BASE_URL"), ()),
                                      ({"OPENAI_API_KEY": KEY}, ("OPENAI_BASE_URL",), (KEY,)),
                                      ({"OPENAI_BASE_URL": URL}, ("OPENAI_API_KEY",), (URL,))):
            with self.subTest(env=sorted(env)), patch("openai.OpenAI", side_effect=AssertionError):
                with self.assertRaises(SystemExit) as caught:
                    run_main([], environ=env)
            message = str(caught.exception)
            for name in expected:
                self.assertIn(name, message)
            for value in absent:
                self.assertNotIn(value, message)

    def test_live_mode_reads_key_and_address_from_the_environment_only(self):
        built = []

        def fake_openai(**kwargs):
            built.append(kwargs)
            return probe.DryRunClient()

        with patch("openai.OpenAI", side_effect=fake_openai):
            code, out, err = run_main(["--calls-per-setting", "1"], environ=ENV)
        self.assertEqual(code, 0)
        self.assertEqual(built, [{"api_key": KEY, "base_url": URL, "timeout": 180.0, "max_retries": 0}])
        self.assertNotIn("DRY RUN", out)
        for secret in (KEY, URL, "gateway.example.internal"):
            self.assertNotIn(secret, out + err)

    def test_the_script_source_has_no_hardcoded_address_or_key(self):
        source = Path(probe.__file__).read_text(encoding="utf-8")
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)
        self.assertNotIn("sk-", source)


if __name__ == "__main__":
    unittest.main()
