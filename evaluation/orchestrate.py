"""Run one A task through B and C, then seal and finalize its evidence."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from ledger import Ledger
from library import Library
from research_agent.io import read_json
from research_agent.model import ModelClient
from research_agent.replay import selected_library_settings, write_replay_context
from research_agent.roles import RoleAgents
from research_agent.runner import TaskRunner
from token_accounting import require_exact_token_accounting

from .finalize import assert_not_in_eval_window, finish_task
from .quote import make_quote_callback


def _internal_token_limits(prices: dict, model: str, task: dict) -> dict | None:
    """Match the measured token study's per-call budget controls."""
    if prices.get("currency") != "TOKENS":
        return None
    require_exact_token_accounting(prices, model)
    budget = task.get("budget", {}).get("money")
    if (isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or budget <= 0):
        raise ValueError("Internal token study needs a positive task budget")
    return {"max_output_tokens": 2048, "budget_limit": float(budget),
            "reserve_cost": float(budget) * 0.3}


class _TestBudgetHarness:
    """Test-only TaskView budget injection; B's source task stays untouched."""

    def __init__(self, harness: Any, budget: float):
        self.harness = harness
        self.budget = budget

    def load_task(self, task_id: str) -> dict:
        task = copy.deepcopy(self.harness.load_task(task_id))
        task["budget"]["money"] = self.budget
        return task

    def __getattr__(self, name: str) -> Any:
        return getattr(self.harness, name)


def execute_task(
    *, run_dir: Path, task_id: str, harness: Any, baseline_solution: Path,
    roles: Any | None = None, test_only_budget: float | None = None,
    max_output_tokens: int = 30000,
    quote_history: Iterable[Mapping[str, Any]] | None = None,
) -> dict:
    """Run dev scoring, A's loop, sealed replay, and one C-only holdout."""
    run_dir = Path(run_dir).expanduser().resolve()
    project_root = run_dir.parent.parent
    manifest = read_json(run_dir / "manifest.json", "run/1")
    if task_id not in manifest["tasks"]:
        raise ValueError("Task is absent from run manifest")
    assert_not_in_eval_window()
    ledger = Ledger.open(run_dir, task_id)
    scored_harness = harness
    test_controls: Path | None = None
    if test_only_budget is not None:
        if manifest["phase"] != "dev" or ledger.prices.get("currency") != "TEST":
            raise ValueError("Test-only budget override requires a dev manifest and TEST prices")
        if isinstance(test_only_budget, bool) or not isinstance(test_only_budget, (int, float)) or not math.isfinite(test_only_budget) or test_only_budget <= 0:
            raise ValueError("Test-only budget must be positive and finite")
        scored_harness = _TestBudgetHarness(harness, float(test_only_budget))
        test_controls = run_dir / "tasks" / task_id / "test_controls.json"
        test_controls.parent.mkdir(parents=True, exist_ok=True)
        test_controls.write_text(json.dumps({
            "schema": "test_controls/1", "label": "TEST ONLY - no real monetary budget",
            "budget": float(test_only_budget), "currency": "TEST",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    token_limits = _internal_token_limits(ledger.prices, manifest["model"],
                                          scored_harness.load_task(task_id))
    if token_limits is not None:
        max_output_tokens = min(max_output_tokens, token_limits["max_output_tokens"])
    baseline_dir = run_dir / "tasks" / task_id / "nodes" / "n0000"
    baseline_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(baseline_solution), baseline_dir / "solution.py")
    baseline = harness.score(task_id, baseline_dir, "dev")
    if not isinstance(baseline, dict) or baseline.get("schema") != "score/1" or not baseline.get("ok") or not baseline.get("correct"):
        raise RuntimeError("B baseline dev score is not successful")
    config_path = (project_root / manifest["agent_config"]["path"]).resolve()
    prices_path = (project_root / manifest["prices"]["path"]).resolve()
    hard_caps_path = project_root / "config/hard_caps.json"
    triggers_path = project_root / "config/triggers.json"
    config = read_json(config_path, "agent/1")
    hard_caps = read_json(hard_caps_path, "hard_caps/1")
    library_root = (project_root / manifest["library"]["root"]).resolve()
    with selected_library_settings(config_path, triggers_path):
        library = Library.open(library_root, frozen=bool(manifest["library"]["frozen"]))
        if roles is None:
            model_kwargs = {"max_output_tokens": max_output_tokens}
            if token_limits is not None:
                model_kwargs.update(budget_limit=token_limits["budget_limit"],
                                    reserve_cost=token_limits["reserve_cost"])
            roles = RoleAgents(ModelClient(manifest["model"], ledger, **model_kwargs),
                               config, project_root=project_root)
        outcome = TaskRunner(
            run_dir=run_dir, task_id=task_id, harness=scored_harness, ledger=ledger,
            roles=roles, library=library, baseline_score=baseline, config=config,
            hard_caps=hard_caps, on_initial_plan=make_quote_callback(ledger, quote_history),
        ).run()
    source_files = {
        "baseline": baseline_dir / "score.json", "agent_config": config_path,
        "prices": prices_path, "hard_caps": hard_caps_path, "triggers": triggers_path,
    }
    for role, relative in config["prompts"].items():
        source_files[f"prompt_{role}"] = (project_root / relative).resolve()
    if test_controls is not None:
        source_files["test_controls"] = test_controls
    opening_plan = run_dir / "tasks" / task_id / "opening_plan.json"
    if opening_plan.exists():
        source_files["opening_plan"] = opening_plan
    write_replay_context(run_dir=run_dir, task_id=task_id, library_root=library_root,
                         source_files=source_files, model=manifest["model"],
                         max_output_tokens=max_output_tokens)
    holdout = finish_task(run_dir, task_id, harness)
    return {"outcome": vars(outcome), "holdout": holdout["score"],
            "task_end": holdout["task_end"]}


__all__ = ["execute_task"]
