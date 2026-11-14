"""TEST-only A+B+C batch smoke with the actual macOS sandbox scorer.

The candidate code and role responses are scripted fixtures. Scores are real
timings; token prices, token usage, and decisions are not autonomous evidence.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.batch import execute_batch
from evaluation.figures import build_study_from_batch, render_html
import harness
from ledger import Ledger, Usage
from pilot.roles import ScriptedRoles


PROJECT = Path(__file__).resolve().parents[1]
TASKS = ("spd-membership-filter", "spd-prefix-sum", "spd-group-counts")
TIERS = {"spd-membership-filter": "practice", "spd-prefix-sum": "validation",
         "spd-group-counts": "exam"}


class RecordingHarness:
    def __init__(self):
        self.scores = defaultdict(int)

    def score(self, task_id, node_dir, split, seed=None):
        if split == "holdout":
            assert os.environ.get("RA_EVAL") == "1"
        else:
            assert os.environ.get("RA_EVAL") != "1"
        run_id = Path(node_dir).resolve().parents[3].name
        self.scores[(run_id, task_id, split)] += 1
        return harness.score(task_id, node_dir, split, seed=seed)

    def __getattr__(self, name):
        return getattr(harness, name)


class CostedScriptedRoles(ScriptedRoles):
    """Known code fixture with synthetic token usage booked in the real ledger."""

    def __init__(self, candidate_path, *, config, run_dir, task_id, effort):
        super().__init__(candidate_path)
        self.config = config
        self.task_id = task_id
        self.effort = effort
        self.ledger = Ledger.open(run_dir, task_id)
        self.tokens = 1000 if config == "tuned_lib" else 2400

    def _book(self, role, purpose, node_id):
        self.ledger.llm_call(
            node_id=node_id, role=role, purpose=purpose,
            effort=self.effort[role], model="test-scripted",
            usage=Usage(input=self.tokens), seconds=0.01,
            prompt_hash="a" * 64,
        )

    def plan(self, payload, *, node_id=None):
        self._book("planner", "plan", node_id)
        self.plans += 1
        if self.plans > 1:
            hypotheses = []
        else:
            hypotheses = [{
                "hypothesis": "Try the known reference candidate (scripted fixture)",
                "rationale": "Exercise A and the real B scorer; no model chose this code",
                "estimate": {"gain": 0.2, "cost": 0.01, "seconds": 5.0},
                "cites": [],
            }]
        return {"hypotheses": hypotheses,
                "predicted": {"cost": 0.01, "seconds": 5.0, "score": 0.8},
                "budget_allocation": [], "expected_progress": []}

    def execute(self, payload, *, fix=False, node_id=None):
        self._book("executor", "fix" if fix else "code", node_id)
        return super().execute(payload, fix=fix, node_id=node_id)

    def diagnose(self, payload, *, node_id=None):
        self._book("diagnoser", "diagnose", node_id)
        return super().diagnose(payload, node_id=node_id)

    def review(self, payload, *, node_id=None):
        self._book("planner", "review", node_id)
        return super().review(payload, node_id=node_id)

    def summarize(self, payload, *, node_id=None):
        self._book("summarizer", "summarize", node_id)
        if self.config not in {"default_lib", "tuned_lib"}:
            return {"drafts": [], "narrowings": []}
        return {"drafts": [{
            "scope": "method", "scenario": "speedup",
            "triggers": list(payload["features"]),
            "conditions": f"Scripted fixture for {self.task_id}",
            "action": "Known reference candidate from the test fixture",
            "negative": self.config == "default_lib",
            "node_ids": ["n0001"],
        }], "narrowings": []}


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "requires B's real macOS sandbox scorer")
class RealBatchSmokeTests(unittest.TestCase):
    def test_scripted_four_config_batch_reaches_real_b_and_c(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tasks").mkdir()
            bank = json.loads((PROJECT / "tasks/bank.json").read_text(encoding="utf-8"))
            bank["tasks"] = [entry for entry in bank["tasks"] if entry["task_id"] in TASKS]
            self.assertEqual({entry["task_id"]: entry["tier"] for entry in bank["tasks"]}, TIERS)
            (root / "tasks/bank.json").write_text(json.dumps(bank) + "\n", encoding="utf-8")
            baselines = {}
            for task_id in TASKS:
                shutil.copytree(PROJECT / "tasks" / task_id, root / "tasks" / task_id)
                baselines[task_id] = root / "tasks" / task_id / "workspace/original.py"
            (root / "config/prompts").mkdir(parents=True)
            for name in ("agent.json", "naive.json", "hard_caps.json", "triggers.json"):
                shutil.copyfile(PROJECT / "config" / name, root / "config" / name)
            for source in (PROJECT / "config/prompts").iterdir():
                if source.is_file():
                    shutil.copyfile(source, root / "config/prompts" / source.name)
            shutil.copyfile(PROJECT / "tests/fixtures/prices_test_only.json",
                            root / "config/prices_test_only.json")
            configs = {name: root / "config" / ("naive.json" if name == "naive" else "agent.json")
                       for name in ("naive", "default", "default_lib", "tuned_lib")}
            scored = RecordingHarness()

            def roles_factory(config, rep, task_id, run_dir):
                part = "hidden/reference.py" if config == "tuned_lib" else "workspace/original.py"
                candidate = PROJECT / "tasks" / task_id / part
                effort = json.loads(configs[config].read_text(encoding="utf-8"))["effort"]
                return CostedScriptedRoles(candidate, config=config, run_dir=run_dir,
                                           task_id=task_id, effort=effort)

            batch_dir = root / "batches/smoke"
            with patch.dict(os.environ, {"RA_TASKS_ROOT": str(root / "tasks")}, clear=False):
                batch = execute_batch(
                    project_root=root, batch_dir=batch_dir, harness=scored,
                    baseline_solutions=baselines, agent_configs=configs,
                    prices_path=root / "config/prices_test_only.json",
                    model="test-scripted", code_commit="a" * 40,
                    test_only_budget=1.0, run_validation=True, run_exam=True,
                    max_workers=4,
                    roles_factory=roles_factory,
                )
                study = build_study_from_batch(batch_dir)
            self.assertEqual(batch["label"], "TEST ONLY")
            self.assertEqual(batch["admission"]["claim"], "admitted")
            self.assertEqual([len(batch["reports"][tier]) for tier in
                              ("practice", "validation", "exam")], [12, 6, 12])
            self.assertEqual(len(batch["runs"]), 30)
            self.assertEqual(json.loads((batch_dir / "plan.json").read_text())["max_workers"], 4)
            self.assertEqual(len(scored.scores), 60)
            self.assertEqual(sum(split == "holdout" for _, _, split in scored.scores), 30)
            self.assertNotIn("RA_EVAL", os.environ)
            self.assertTrue(study["test_only"])
            self.assertEqual(len(study["task_rows"]), 12)
            self.assertIn("TEST 联调数据", render_html(study))
            self.assertGreater(study["configurations"]["default_lib"]["cost_mean"],
                               study["configurations"]["tuned_lib"]["cost_mean"])
            for entry in batch["runs"]:
                run_dir = root / "runs" / entry["run_id"]
                manifest = json.loads((run_dir / "manifest.json").read_text())
                self.assertEqual(manifest["phase"], "dev")
                self.assertEqual(manifest["library"]["frozen"],
                                 entry["logical_phase"] != "practice")
                task_id = TASKS[("practice", "validation", "exam").index(entry["logical_phase"])]
                events = Ledger.open(run_dir, task_id)._events()
                kinds = [event["kind"] for event in events]
                self.assertEqual(kinds.count("quote"), 1)
                self.assertLess(kinds.index("quote"), kinds.index("experiment"))
                self.assertEqual(kinds.count("task_end"), 1)
                self.assertTrue((run_dir / "tasks" / task_id / "holdout_attempt.json").exists())
                # One holdout attempt on the best node, plus one baseline reference
                # score on the same hidden data when the best node is not the baseline.
                best = next(event for event in events if event["kind"] == "stop")["best_node"]
                self.assertEqual(scored.scores[(entry["run_id"], task_id, "holdout")],
                                 1 if best == "n0000" else 2)
                ending = next(event for event in events if event["kind"] == "task_end")
                self.assertFalse(ending["holdout_failed"])
                self.assertIsNotNone(ending["holdout_raw_lift"])
                self.assertEqual(ending["baseline_holdout_raw"] is not None, True)
                baseline = json.loads((run_dir / "tasks" / task_id / "baseline_holdout.json").read_text())
                self.assertEqual(baseline["reused_best_result"], best == "n0000")
                self.assertTrue((run_dir / "tasks" / task_id / "test_controls.json").exists())
                self.assertTrue(all(event["model"] == "test-scripted" and event["cost"] > 0
                                    for event in events if event["kind"] == "llm_call"))


if __name__ == "__main__":
    unittest.main()
