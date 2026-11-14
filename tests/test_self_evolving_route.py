from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from evaluation.finalize import finish_run, start_run
from evaluation.manifest import create_manifest
from evaluation.orchestrate import execute_task
from evaluation.report import run_report
from ledger import Ledger, Usage
from pilot.harness import TASK_ID as PILOT_TASK, WORKSPACE
from research_agent.model import ModelClient, ModelResult, TokenUsage
from research_agent.roles import RoleAgents
from research_agent.routing import RoutingClient, validate_policy
from tests.test_abc_e2e import FakeResponses
from tests.test_evaluation_orchestrate import HoldoutPilot


PROJECT = Path(__file__).resolve().parents[1]
TASK_ID = "spd-distance-matrix"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _policy() -> dict:
    return {
        "schema": "route_policy/1", "default_model": "qwen3.5-27b",
        "role_models": {"planner": "gpt-6-astra", "executor": "claude-opus-5-5",
                        "diagnoser": "claude-opus-5-5"},
        "eligible_roles": ["planner", "executor", "diagnoser"],
        "thresholds": {"executor_failures": 2, "plateau_nodes": 2, "score_drop": 0.1},
        "remote_call_cap": 2,
        "calibration": {"schema": "route_calibration/1", "algorithm": "test"},
    }


def _fixture(root: Path) -> tuple[Path, dict]:
    config = root / "config"
    config.mkdir()
    shutil.copyfile(PROJECT / "config/agent.json", config / "agent.json")
    shutil.copyfile(PROJECT / "config/hard_caps.json", config / "hard_caps.json")
    shutil.copyfile(PROJECT / "config/triggers.json", config / "triggers.json")
    (config / "prompts").mkdir()
    for source in (PROJECT / "config/prompts").iterdir():
        shutil.copyfile(source, config / "prompts" / source.name)
    policy = _policy()
    _write(config / "route.json", policy)
    prices = {"schema": "prices/1", "currency": "TEST", "test_only": True,
              "unit_tokens": 1,
              "models": {model: {bucket: 1 for bucket in ("input", "cached_input", "output", "reasoning")}
                         for model in {policy["default_model"], *policy["role_models"].values()}}}
    _write(config / "prices.json", prices)
    run = root / "runs/run-20261114-120000-abcd"
    create_manifest(project_root=root, run_dir=run, phase="dev", config="self_evolving", rep=1,
                    bank_version="bank-v1", tasks=[TASK_ID], library_root=run / "library",
                    library_start_version="lib-0", library_frozen=False,
                    model="qwen3.5-27b", agent_config_path=config / "agent.json",
                    prices_path=config / "prices.json", routing_config_path=config / "route.json",
                    code_commit="a" * 40)
    return run, policy


class _ScriptedModel:
    def __init__(self, name: str, ledger: Ledger, *, fail: bool = False):
        self.name = name
        self.ledger = ledger
        self.fail = fail
        self.calls = 0

    def call_json(self, **kwargs):
        self.calls += 1
        self.ledger.llm_call(node_id=kwargs.get("node_id"), role=kwargs["role"],
                             purpose=kwargs["purpose"], effort=kwargs["effort"],
                             model=self.name, usage=Usage(input=2, output=1), seconds=0.1,
                             prompt_hash="a" * 64, ok=not self.fail,
                             error="simulated negative result" if self.fail else None)
        if self.fail:
            raise RuntimeError("simulated negative result")
        return ModelResult({"ok": True}, TokenUsage(input=2, output=1), 0.1, 3.0,
                           "a" * 64, None)


class SelfEvolvingRouteTests(unittest.TestCase):
    def test_routed_run_reaches_holdout_report_and_replay_context(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run, policy = _fixture(root)
            manifest = json.loads((run / "manifest.json").read_text())
            ledger = Ledger.open(run, PILOT_TASK)
            transport = SimpleNamespace(responses=FakeResponses())
            clients = {model: ModelClient(model, ledger, client=transport,
                                          max_output_tokens=2048,
                                          budget_limit=40_000, reserve_cost=12_000)
                       for model in manifest["models"]}
            config = json.loads((root / "config/agent.json").read_text())
            roles = RoleAgents(RoutingClient(clients, ledger, policy), config, project_root=root)
            start_run(run)
            pilot = HoldoutPilot(size=180, repetitions=2, measurements_path=run / "measurements.jsonl")
            execute_task(run_dir=run, task_id=PILOT_TASK, harness=pilot,
                         baseline_solution=WORKSPACE / "original.py", roles=roles,
                         test_only_budget=40_000, max_output_tokens=2048)
            finish_run(run)
            report = run_report(run)
            task = report["tasks"][0]
            self.assertGreater(task["model_calls"], 0)
            self.assertEqual(task["routing"]["remote_calls"], 0)
            self.assertEqual(len(task["routing"]["route_decision_seqs"]), task["model_calls"])
            context = json.loads((run / "tasks" / PILOT_TASK / "replay_context.json").read_text())
            self.assertIn("routing", context["sources"])
            self.assertEqual(context["manifest_fields"]["schema"], "run/2")

    def test_sealed_multimodel_ledger_routes_and_bills_failed_calls(self):
        with TemporaryDirectory() as temp:
            run, policy = _fixture(Path(temp))
            ledger = Ledger.open(run, TASK_ID)
            clients = {model: _ScriptedModel(model, ledger, fail=model == "claude-opus-5-5")
                       for model in {policy["default_model"], *policy["role_models"].values()}}
            route = RoutingClient(clients, ledger, policy)

            def call(role, purpose, payload):
                return route.call_json(role=role, purpose=purpose, effort="low",
                                       system_prompt="test", payload=payload,
                                       schema={}, node_id="n0001")

            call("planner", "plan", {"recent_experiments": []})
            with self.assertRaisesRegex(RuntimeError, "negative result"):
                call("executor", "fix", {"attempt": 3, "previous_error": "bad code"})
            call("planner", "plan", {"recent_experiments": [
                {"status": "rejected"}, {"status": "failed"}]})
            call("executor", "fix", {"attempt": 3, "previous_error": "bad code"})
            events = ledger._events()
            models = [event["model"] for event in events if event["kind"] == "llm_call"]
            self.assertEqual(models, ["qwen3.5-27b", "claude-opus-5-5", "gpt-6-astra", "qwen3.5-27b"])
            self.assertEqual(ledger.spent(), 12.0)
            decisions = [event for event in events if event["kind"] == "route_decision"]
            self.assertEqual(len(decisions), 4)
            self.assertEqual(decisions[-1]["reason"], "remote_cap_exhausted")
            self.assertFalse(decisions[1]["ok"])
            self.assertIsNotNone(decisions[1]["llm_seq"])
            self.assertEqual(sum(event["model"] != "qwen3.5-27b" for event in events
                                 if event["kind"] == "llm_call"), 2)
            start_run(run)
            tree_path = run / "tasks" / TASK_ID / "tree.jsonl"
            tree_path.parent.mkdir(parents=True, exist_ok=True)
            tree_path.write_text(json.dumps({"schema": "node/1", "node_id": "n0000",
                                             "status": "accepted"}) + "\n")
            _write(run / "tasks" / TASK_ID / "holdout_score.json",
                   {"schema": "score/1", "split": "holdout", "score": 0.5})
            ledger.emit("stop", best_node="n0000", dev_score=0.5)
            ledger.emit("task_end", best_node="n0000", holdout_score=0.5,
                        total_cost=ledger.spent(), seconds={"wall": 1},
                        quote_error={"cost": None, "seconds": None, "score": None})
            finish_run(run)
            report = run_report(run)["tasks"][0]
            self.assertEqual(report["routing"]["remote_calls"], 2)
            self.assertEqual(report["routing"]["failed_remote_calls"], 1)
            self.assertEqual(report["routing"]["remote_tokens"], 6)

    def test_route_policy_is_sealed_and_run_one_stays_single_model(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run, policy = _fixture(root)
            ledger = Ledger.open(run, TASK_ID)
            with self.assertRaisesRegex(ValueError, "sealed run manifest"):
                ledger.llm_call(node_id=None, role="planner", purpose="plan", effort="low",
                                model="unlisted", usage=Usage(), seconds=0, prompt_hash="a" * 64)
            policy["remote_call_cap"] = 3
            with self.assertRaisesRegex(ValueError, "cap"):
                validate_policy(policy)
            _write(root / "config/route.json", policy)
            with self.assertRaisesRegex(ValueError, "policy changed"):
                Ledger.open(run, TASK_ID)


if __name__ == "__main__":
    unittest.main()
