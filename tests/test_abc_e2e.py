"""A+B+C integration with real B scoring and TEST-only model responses/prices.

This test is intentionally slower than unit tests. It does not call a paid model
or claim real model cost or autonomous research quality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.finalize import finish_run, start_run
from evaluation.manifest import create_manifest
from evaluation import orchestrate
from evaluation.report import run_report
import harness
from ledger import Ledger
from research_agent.model import ModelClient
from research_agent.replay import replay_node


ROOT = Path(__file__).resolve().parents[1]
TASK = "spd-distance-matrix"


class FakeResponses:
    def __init__(self):
        self.calls: list[str] = []

    def create(self, **kwargs):
        purpose = kwargs["text"]["format"]["name"]
        self.calls.append(purpose)
        if purpose == "planner_plan":
            hypotheses = [] if self.calls.count(purpose) > 1 else [{
                "hypothesis": "Vectorize two-dimensional distances",
                "rationale": "The public profile shows nested Python loops",
                "estimate": {"gain": 0.2, "cost": 0.01, "seconds": 20.0},
                "cites": [],
            }]
            data = {"hypotheses": hypotheses,
                    "predicted": {"cost": 0.01, "seconds": 20.0, "score": 0.8},
                    "budget_allocation": [], "expected_progress": []}
        elif purpose == "executor_code":
            data = {"code": (ROOT / "pilot/candidate.py").read_text(encoding="utf-8"),
                    "notes": "TEST ONLY: known candidate"}
        elif purpose == "diagnoser_diagnose":
            data = {"hypothesis": "", "rationale": "No new hypothesis in fixture", "advice": ""}
        elif purpose == "planner_review":
            data = {"decision": "continue", "budget_moves": [], "note": "Fixture review"}
        elif purpose == "summarizer_summarize":
            draft = {"scope": "domain", "scenario": "speedup",
                     "conditions": "pairwise distance task",
                     "action": "Vectorize distance calculation",
                     "negative": False, "node_ids": ["n0001"]}
            data = {"drafts": [{**draft, "triggers": ["n0001"]},
                               {**draft, "triggers": ["nested_python_loop"]}],
                    "narrowings": []}
        else:
            raise AssertionError(f"Unexpected model purpose: {purpose}")
        return SimpleNamespace(
            id=f"fixture-{len(self.calls)}", status="completed", output=[],
            output_text=json.dumps(data),
            usage={"input_tokens": 100, "output_tokens": 50,
                   "input_tokens_details": {"cached_tokens": 20},
                   "output_tokens_details": {"reasoning_tokens": 10}},
        )


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "B's real sandbox scorer requires macOS sandbox-exec")
class FullStackTests(unittest.TestCase):
    def test_real_scorer_and_model_accounting_reach_single_holdout(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            config = project / "config"
            (config / "prompts").mkdir(parents=True)
            for name in ("agent.json", "hard_caps.json", "triggers.json"):
                shutil.copyfile(ROOT / "config" / name, config / name)
            for name in ("planner", "executor", "diagnoser", "summarizer"):
                shutil.copyfile(ROOT / "config/prompts" / f"{name}.md",
                                config / "prompts" / f"{name}.md")
            shutil.copyfile(ROOT / "tests/fixtures/prices_test_only.json",
                            config / "prices_test_only.json")
            tasks = project / "tasks"
            shutil.copytree(ROOT / "tasks", tasks)
            task_json = tasks / TASK / "task.json"
            self.assertIsNone(json.loads(task_json.read_text())["budget"]["money"])
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            run = project / "runs/run-20261114-120000-abcd"
            create_manifest(
                project_root=project, run_dir=run, phase="dev", config="default", rep=1,
                bank_version="bank-v1", tasks=[TASK], library_root=run / "library",
                library_start_version="lib-0", library_frozen=False, model="test-scripted",
                agent_config_path=config / "agent.json",
                prices_path=config / "prices_test_only.json", code_commit=commit,
            )
            fake = FakeResponses()
            client = SimpleNamespace(responses=fake)
            with patch.dict(os.environ, {"RA_TASKS_ROOT": str(tasks)}, clear=False):
                start_run(run)
                with patch.object(orchestrate, "ModelClient",
                                  side_effect=lambda model, ledger, **kwargs: ModelClient(
                                      model, ledger, client=client, **kwargs)):
                    result = orchestrate.execute_task(
                        run_dir=run, task_id=TASK, harness=harness,
                        baseline_solution=tasks / TASK / "workspace/original.py",
                        test_only_budget=1.0,
                    )
                finish_run(run)
                report = run_report(run)
                replay = replay_node(
                    run_dir=run, task_id=TASK, node_id=result["outcome"]["best_node"],
                    harness=harness, library_root=run / "library", require_provenance=True,
                )
            self.assertEqual(result["outcome"]["best_node"], "n0001")
            self.assertTrue(result["holdout"]["correct"])
            self.assertEqual(report["schema"], "run_report/1")
            self.assertEqual(report["tasks"][0]["task_id"], TASK)
            self.assertEqual(report["currency"], "TEST")
            self.assertTrue(replay["integrity_verified"])
            self.assertEqual(report["tasks"][0]["best_node"], "n0001")
            rejections = (run / "tasks" / TASK / "experience_rejections.jsonl").read_text().splitlines()
            self.assertEqual(len(rejections), 1)
            self.assertEqual(json.loads(rejections[0])["reason"], "invalid_triggers")
            self.assertEqual(len(json.loads((run / "library/state.json").read_text())["entries"]), 1)
            events = Ledger.open(run, TASK)._events()
            kinds = [item["kind"] for item in events]
            self.assertLess(kinds.index("quote"), kinds.index("experiment"))
            self.assertEqual(kinds.count("task_end"), 1)
            llm_events = [item for item in events if item["kind"] == "llm_call"]
            self.assertEqual(len(llm_events), len(fake.calls))
            self.assertTrue(llm_events)
            self.assertTrue(all(item["tokens"] == {"input": 80, "cached_input": 20,
                                                     "output": 40, "reasoning": 10}
                                for item in llm_events))
            expected_cost = len(llm_events) * 150 / 1_000_000
            self.assertAlmostEqual(result["task_end"]["total_cost"], expected_cost)
            self.assertAlmostEqual(Ledger.open(run, TASK).spent(), expected_cost)
            self.assertIsNone(json.loads(task_json.read_text())["budget"]["money"])


if __name__ == "__main__":
    unittest.main()
