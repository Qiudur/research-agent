"""config/agent_optimized.json turns on the opt-in switches; default configs stay unchanged."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from library import Library, selected_settings
from research_agent.roles import RoleAgents
from research_agent.scheduler import validate_options

ROOT = Path(__file__).resolve().parents[1]
NEW_KEYS = {
    "stop_loss", "ceiling_excludes_baseline", "planner_mode", "hypotheses_per_plan",
    "failure_feedback", "cost_priors", "role_controls", "thinking_switch_style",
    "experience_writer", "summarizer_reserve", "library_brief",
}


def load(name: str) -> dict:
    return json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))


class OptimizedConfigTests(unittest.TestCase):
    def test_optimized_config_validates_and_loads_lean_prompts(self):
        config = load("agent_optimized.json")
        validate_options(config)
        roles = RoleAgents(object(), config, project_root=ROOT)
        self.assertEqual(roles.planner_mode, "lean")
        self.assertEqual(roles.summarizer_reserve, "worst_case")
        self.assertEqual(roles.prompts["planner"],
                         (ROOT / "config/prompts/planner_lean.md").read_text(encoding="utf-8"))
        self.assertEqual(roles.prompts["executor"],
                         (ROOT / "config/prompts/executor_lean.md").read_text(encoding="utf-8"))

    def test_library_accepts_the_optimized_brief_block(self):
        with TemporaryDirectory() as temp, selected_settings(
                ROOT / "config/agent_optimized.json", ROOT / "config/triggers.json"):
            Library.open(Path(temp) / "library")

    def test_default_and_naive_configs_carry_none_of_the_new_switches(self):
        for name in ("agent.json", "naive.json"):
            with self.subTest(config=name):
                self.assertEqual(NEW_KEYS & set(load(name)), set())


if __name__ == "__main__":
    unittest.main()
