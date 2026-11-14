from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dashboard.export import load_run, render_html


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.run = self.project / "runs" / "run-test"
        self.task = self.run / "tasks" / "spd-example"
        self.task.mkdir(parents=True)
        self.events = [
            {"schema": "ledger/1", "seq": 1, "kind": "task_start", "budget": 2.0},
            {"schema": "ledger/1", "seq": 2, "kind": "score", "split": "dev", "score": 0.8, "node_id": "n0001"},
        ]
        self.nodes = [
            {"schema": "node/1", "node_id": "n0000", "parent": None, "status": "accepted",
             "hypothesis": "Baseline", "score": {"dev": 0.2, "raw": 1.0}},
            {"schema": "node/1", "node_id": "n0001", "parent": "n0000", "status": "running",
             "hypothesis": "Try change", "score": None},
            {"schema": "node/1", "node_id": "n0001", "parent": "n0000", "status": "accepted",
             "hypothesis": "Try change", "score": {"dev": 0.8, "raw": 2.0}},
        ]
        write_jsonl(self.task / "ledger.jsonl", self.events)
        write_jsonl(self.task / "tree.jsonl", self.nodes)

    def add_manifest_and_prices(self) -> None:
        prices = self.project / "fixtures" / "prices.json"
        write_json(prices, {"schema": "prices/1", "currency": "TEST", "test_only": True,
                            "models": {"test-model": {"input": 1, "cached_input": 1,
                                                      "output": 1, "reasoning": 1}}})
        manifest = {"schema": "run/1", "run_id": "run-test", "phase": "dev",
                    "config": "default", "rep": 1, "model": "test-model",
                    "prices": {"path": "fixtures/prices.json",
                               "sha256": hashlib.sha256(prices.read_bytes()).hexdigest()},
                    "tasks": ["spd-example"], "library": {"root": "runs/run-test/library"}}
        write_json(self.run / "manifest.json", manifest)

    def test_missing_manifest_never_invents_money(self) -> None:
        result = load_run(self.run, project_root=self.project)
        task = result["tasks"][0]
        self.assertEqual(task["best"]["node_id"], "n0001")
        self.assertEqual(len(task["nodes"]), 2)
        self.assertIsNone(task["cost"])
        self.assertIsNone(task["budget"])
        self.assertFalse(result["run"]["price"]["verified"])

    def test_verified_test_prices_and_event_cost_are_labeled(self) -> None:
        self.add_manifest_and_prices()
        self.events.insert(1, {"schema": "ledger/1", "seq": 2, "kind": "llm_call",
                               "cost": 0.25, "seconds": 1.5,
                               "tokens": {"input": 30, "cached_input": 70, "output": 4, "reasoning": 2}})
        self.events[-1]["seq"] = 3
        self.events.append({"schema": "ledger/1", "seq": 4, "kind": "task_end",
                            "total_cost": 0.25, "holdout_score": 0.75,
                            "seconds": {"llm": 1.5, "run": 2.0, "queue": 0.1, "wall": 3.8}})
        write_jsonl(self.task / "ledger.jsonl", self.events)
        result = load_run(self.run, project_root=self.project)
        task = result["tasks"][0]
        self.assertTrue(result["run"]["price"]["test_only"])
        self.assertEqual(task["cost"], 0.25)
        self.assertEqual(task["budget"], 2.0)
        self.assertAlmostEqual(task["cache_rate"], 0.7)
        self.assertEqual(task["seconds"]["wall"], 3.8)
        self.assertEqual(task["series"][-1]["spent"], 0.25)

    def test_token_study_and_cash_prices_have_distinct_labels(self) -> None:
        self.add_manifest_and_prices()
        prices_path = self.project / "config/prices.json"
        manifest_path = self.run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for currency, test_only, internal in (("TOKENS", False, True),
                                              ("TOKENS", True, False), ("CNY", False, False)):
            write_json(prices_path, {"schema": "prices/1", "currency": currency,
                                     "test_only": test_only, "models": {"test-model": {
                                         "input": 1, "cached_input": 1, "output": 1, "reasoning": 1}}})
            manifest["prices"] = {"path": "config/prices.json",
                                  "sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest()}
            write_json(manifest_path, manifest)
            info = load_run(self.run, project_root=self.project)["run"]["price"]
            self.assertTrue(info["verified"])
            self.assertEqual(info["internal_token_study"], internal)
            if currency == "TOKENS":
                self.assertIn("非现金节省、非官方比赛成绩", info["message"])
                self.assertIn("token", info["accounting_note"])
            else:
                self.assertIsNone(info["accounting_note"])
            if test_only:
                self.assertIn("仅供联调", info["message"])
        html = render_html(load_run(self.run, project_root=self.project))
        self.assertIn("isTokens ? 'token 消耗' : '花费'", html)

    def test_missing_cost_evidence_remains_unknown_even_with_prices(self) -> None:
        self.add_manifest_and_prices()
        result = load_run(self.run, project_root=self.project)
        self.assertTrue(result["run"]["price"]["verified"])
        self.assertIsNone(result["tasks"][0]["cost"])

    def test_price_hash_change_disables_cost_display(self) -> None:
        self.add_manifest_and_prices()
        self.events.append({"schema": "ledger/1", "seq": 3, "kind": "task_end",
                            "total_cost": 0.0})
        write_jsonl(self.task / "ledger.jsonl", self.events)
        prices = self.project / "fixtures" / "prices.json"
        prices.write_text(prices.read_text() + " ", encoding="utf-8")
        result = load_run(self.run, project_root=self.project)
        self.assertIsNone(result["tasks"][0]["cost"])

    def test_html_escapes_embedded_script_breakout(self) -> None:
        self.nodes[-1]["hypothesis"] = "</script><script>alert(1)</script>"
        write_jsonl(self.task / "tree.jsonl", self.nodes)
        html = render_html(load_run(self.run, project_root=self.project))
        self.assertNotIn("</script><script>alert(1)</script>", html)
        self.assertIn("\\u003c/script", html)


if __name__ == "__main__":
    unittest.main()
