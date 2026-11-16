"""Quant task IDs and scenarios plug into A's existing audited primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ledger import Ledger
from library import Library
from library.skills import _TASK
from research_agent.roles import SCHEMAS
from research_agent.runner import TaskRunner
from research_agent.scheduler import Scheduler, scenario_key
from research_agent.tree import ExperimentTree, TreeError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/agent.json").read_text(encoding="utf-8"))
CAPS = json.loads((ROOT / "config/hard_caps.json").read_text(encoding="utf-8"))


class FakeHarness:
    def __init__(self, root: Path):
        workspace = root / "tasks/quant-demo/workspace"
        workspace.mkdir(parents=True)
        (workspace / "original.py").write_text(
            "def compute_signal(history):\n    return {symbol: 0.0 for symbol in history}\n",
            encoding="utf-8",
        )
        (workspace / "profile.json").write_text(json.dumps({
            "schema": "profile/1", "features": ["time_series"],
            "summary": {"dates": 7, "symbols": 2},
        }), encoding="utf-8")
        self.task = {
            "task_id": "quant-demo", "scenario": "quant", "title": "demo",
            "description": "Rank assets using only available history",
            "metric": {"name": "net_return", "higher_is_better": True},
            "deliverable": {"file": "solution.py", "function": "compute_signal",
                            "signature": "compute_signal(history)"},
            "budget": {"money": 10.0, "wall_seconds": 60.0},
            "workspace": workspace,
        }

    def load_task(self, task_id: str) -> dict:
        assert task_id == "quant-demo"
        return self.task

    def run_script(self, *args, **kwargs):
        raise AssertionError("No candidate should be run for an empty plan")


class FakeLedger:
    def __init__(self):
        self.events = []

    def spent(self) -> float:
        return 0.0

    def elapsed(self) -> dict[str, float]:
        return {"llm": 0.0, "run": 0.0, "queue": 0.0, "wall": 0.0}

    def emit(self, kind, *, node_id=None, **values):
        self.events.append({"kind": kind, "node_id": node_id, **values})


class FakeRoles:
    def __init__(self):
        self.planner_task = None

    def plan(self, payload, *, node_id=None):
        self.planner_task = payload["task"]
        return {"hypotheses": [], "predicted": {"cost": 1.0, "seconds": 1.0, "score": 0.1},
                "budget_allocation": [], "expected_progress": []}

    def summarize(self, payload, *, node_id=None):
        return {"drafts": [], "narrowings": []}


class QuantCoreCompatibilityTests(unittest.TestCase):
    def test_tree_scheduler_and_role_contract_accept_quant_without_changing_old_keys(self):
        with TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run-test"
            self.assertEqual(ExperimentTree.open(run_dir, "quant-demo").task_id, "quant-demo")
            for invalid in ("quant-../escape", "quant-", "quant-demo-", "../quant-demo"):
                with self.subTest(invalid=invalid), self.assertRaises(TreeError):
                    ExperimentTree.open(run_dir, invalid)
        self.assertEqual(tuple(scenario_key(item) for item in
                               ("spd-demo", "tab-demo", "quant-demo")),
                         ("spd", "tab", "quant"))
        scheduler = Scheduler({**CONFIG, "cost_priors": {"quant": 2.0}}, CAPS,
                              budget=10.0, wall_seconds=60.0, task_id="quant-demo")
        self.assertEqual(scheduler.cost_prior, 2.0)
        self.assertIn("quant", SCHEMAS["summarize"]["properties"]["drafts"]["items"]["properties"]["scenario"]["enum"])
        self.assertIsNotNone(_TASK.fullmatch("quant-demo"))

    def test_library_accepts_quant_trigger_and_queries_only_matching_scenario(self):
        with TemporaryDirectory() as temp:
            lib = Library.open(Path(temp) / "library")
            self.assertEqual(lib.query("quant", ["time_series"]), [])
            ids = lib.write_experiences([{"scope": "domain", "scenario": "quant",
                                          "triggers": ["time_series"], "conditions": "",
                                          "action": "Test a lagged signal", "negative": False}],
                                        source={"run_id": "run-test", "task_id": "quant-demo",
                                                "node_ids": ["n0001"]})
            self.assertEqual([item["id"] for item in lib.query("quant", ["time_series"])], ids)
            self.assertEqual(lib.query("speedup", ["nested_python_loop"]), [])
            with self.assertRaisesRegex(ValueError, "unknown trigger"):
                lib.query("quant", ["missing_heavy"])

    def test_runner_preserves_quant_baseline_and_planner_context(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            harness = FakeHarness(root)
            ledger = FakeLedger()
            roles = FakeRoles()
            lib = Library.open(root / "runs/run-test/library")
            runner = TaskRunner(run_dir=root / "runs/run-test", task_id="quant-demo",
                                harness=harness, ledger=ledger, roles=roles, library=lib,
                                baseline_score={"schema": "score/1", "split": "dev",
                                                "ok": True, "score": 0.1, "raw": 0.1},
                                config=CONFIG, hard_caps=CAPS)
            outcome = runner.run()
            self.assertEqual((outcome.reason, outcome.best_node), ("planner", "n0000"))
            context = roles.planner_task
            self.assertEqual((context["scenario"], context["visible_sample_file"]),
                             ("quant", "original.py"))
            self.assertIn("def compute_signal", context["visible_sample"])
            root_solution = root / "runs/run-test/tasks/quant-demo/nodes/n0000/solution.py"
            self.assertEqual(root_solution.read_text(encoding="utf-8"),
                             (harness.task["workspace"] / "original.py").read_text(encoding="utf-8"))
            self.assertEqual([row["kind"] for row in ledger.events].count("stop"), 1)

    def test_ledger_can_open_sealed_quant_task(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config"
            config.mkdir()
            inputs = {
                "dataset": ("config/dataset.json", {"schema": "quant_dataset/1"}),
                "agent_config": ("config/quant_agent.json", {"schema": "agent/1"}),
                "prices": ("config/prices.json", {"schema": "prices/1",
                    "currency": "TEST", "unit_tokens": 1, "models": {}}),
            }
            descriptors = {}
            for name, (relative, value) in inputs.items():
                path = root / relative
                path.write_text(json.dumps(value), encoding="utf-8")
                descriptors[name] = {"path": relative,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            run_dir = root / "runs/run-test"
            run_dir.mkdir(parents=True)
            manifest_path = run_dir / "manifest.json"
            manifest = {
                "schema": "quant_run/1", "run_id": "run-test",
                "tasks": ["quant-demo"], "phase": "dev",
                "model": "test-scripted", "code_commit": "a" * 40,
                "library": {"root": "runs/run-test/library", "start_version": "lib-0",
                            "frozen": False},
                "hard_caps": {"budget_ratio": 0.9},
                **descriptors,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(Ledger.open(run_dir, "quant-demo").task_id, "quant-demo")
            manifest["prices"]["path"] = str(root / "config/prices.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "project-relative path"):
                Ledger.open(run_dir, "quant-demo")
            manifest["prices"]["path"] = "config/prices.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "config/dataset.json").write_text(
                json.dumps({"schema": "quant_dataset/1", "changed": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset changed after sealing"):
                Ledger.open(run_dir, "quant-demo")
            manifest["schema"] = "run/1"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "require a quant_run/1"):
                Ledger.open(run_dir, "quant-demo")

    def test_quant_runner_rejects_non_dev_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            harness = FakeHarness(root)
            run_dir = root / "runs/run-test"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps({
                "schema": "quant_run/1", "run_id": "run-test",
                "tasks": ["quant-demo"], "phase": "final",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only uses dev data"):
                TaskRunner(run_dir=run_dir, task_id="quant-demo", harness=harness,
                           ledger=FakeLedger(), roles=FakeRoles(),
                           library=Library.open(run_dir / "library"),
                           baseline_score={"schema": "score/1", "split": "dev",
                                           "ok": True, "score": 0.1},
                           config=CONFIG, hard_caps=CAPS)


if __name__ == "__main__":
    unittest.main()
