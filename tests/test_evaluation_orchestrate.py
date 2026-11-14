from __future__ import annotations

import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from evaluation.finalize import finish_run, start_run
from evaluation.manifest import create_manifest
from evaluation.orchestrate import execute_task
from evaluation.report import run_report
from ledger import Ledger
from pilot.harness import PilotHarness, TASK_ID, WORKSPACE
from pilot.roles import ScriptedRoles
from research_agent.model import ModelCallError


PROJECT = Path(__file__).resolve().parents[1]


class HoldoutPilot(PilotHarness):
    def score(self, task_id, node_dir, split, seed=None):
        if split == "holdout":
            assert os.environ.get("RA_EVAL") == "1"
            result = super().score(task_id, node_dir, "dev", seed=20269999)
            return {**result, "split": "holdout"}
        assert os.environ.get("RA_EVAL") != "1"
        return super().score(task_id, node_dir, split, seed=seed)


class OrchestrateTests(unittest.TestCase):
    def test_model_json_error_still_reaches_one_holdout_and_report(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config"
            (config / "prompts").mkdir(parents=True)
            for name in ("agent.json", "hard_caps.json", "triggers.json"):
                shutil.copyfile(PROJECT / "config" / name, config / name)
            for source in (PROJECT / "config/prompts").iterdir():
                shutil.copyfile(source, config / "prompts" / source.name)
            shutil.copyfile(PROJECT / "tests/fixtures/prices_test_only.json",
                            config / "prices_test_only.json")
            run = root / "runs/run-20261114-120000-abcd"
            create_manifest(project_root=root, run_dir=run, phase="dev", config="default", rep=1,
                            bank_version="bank-v1", tasks=[TASK_ID], library_root=run / "library",
                            library_start_version="lib-0", library_frozen=False, model="test-scripted",
                            agent_config_path=config / "agent.json", prices_path=config / "prices_test_only.json",
                            code_commit="a" * 40)
            start_run(run)

            class BadPlanner:
                def plan(self, payload, *, node_id=None):
                    raise ModelCallError("malformed JSON")

            harness = HoldoutPilot(size=180, repetitions=2,
                                   measurements_path=run / "measurements.jsonl")
            result = execute_task(run_dir=run, task_id=TASK_ID, harness=harness,
                                  baseline_solution=WORKSPACE / "original.py",
                                  roles=BadPlanner(), test_only_budget=1.0)
            self.assertEqual(result["outcome"]["reason"], "error")
            self.assertEqual(result["outcome"]["best_node"], "n0000")
            self.assertTrue(result["holdout"]["ok"])
            finish_run(run)
            report = run_report(run)
            self.assertEqual(len(report["tasks"]), 1)
            self.assertTrue((run / "tasks" / TASK_ID / "replay_context.json").is_file())
            self.assertTrue((run / "tasks" / TASK_ID / "holdout_attempt.json").is_file())

    def test_real_timing_a_c_scripted_loop_quotes_before_experiment(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config"
            (config / "prompts").mkdir(parents=True)
            for name in ("agent.json", "hard_caps.json", "triggers.json"):
                shutil.copyfile(PROJECT / "config" / name, config / name)
            for source in (PROJECT / "config/prompts").iterdir():
                shutil.copyfile(source, config / "prompts" / source.name)
            shutil.copyfile(PROJECT / "tests/fixtures/prices_test_only.json", config / "prices_test_only.json")
            run = root / "runs/run-20261114-120000-abcd"
            create_manifest(project_root=root, run_dir=run, phase="dev", config="default", rep=1,
                            bank_version="bank-v1", tasks=[TASK_ID], library_root=run / "library",
                            library_start_version="lib-0", library_frozen=False, model="test-scripted",
                            agent_config_path=config / "agent.json", prices_path=config / "prices_test_only.json",
                            code_commit="a" * 40)
            start_run(run)
            harness = HoldoutPilot(size=750, repetitions=3, measurements_path=run / "measurements.jsonl")
            result = execute_task(run_dir=run, task_id=TASK_ID, harness=harness,
                                  baseline_solution=WORKSPACE / "original.py",
                                  roles=ScriptedRoles(PROJECT / "pilot/candidate.py"),
                                  test_only_budget=1.0)
            self.assertEqual(result["outcome"]["best_node"], "n0001")
            self.assertTrue(result["holdout"]["correct"])
            events = Ledger.open(run, TASK_ID)._events()
            quote = next(i for i, event in enumerate(events) if event["kind"] == "quote")
            experiment = next(i for i, event in enumerate(events) if event["kind"] == "experiment")
            self.assertLess(quote, experiment)
            self.assertNotIn("RA_EVAL", os.environ)
            self.assertTrue((run / "tasks" / TASK_ID / "replay_context.json").exists())
            finish_run(run)
            report = run_report(run)
            self.assertEqual(report["currency"], "TEST")
            self.assertEqual(report["total_cost"], 0)
            self.assertTrue((run / "tasks" / TASK_ID / "test_controls.json").exists())


if __name__ == "__main__":
    unittest.main()
