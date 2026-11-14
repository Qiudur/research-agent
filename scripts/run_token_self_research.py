"""Run C's automatic self-research with real model calls on two practice tasks.

This is a dev-only, copied-project experiment. TEST units are exactly the
provider-reported four-bucket token counts (one unit per token), not money or
an official validation/exam result. The API key is read only from the process
environment and never written to the copied project or campaign summary.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Iterator

import harness
from evaluation.orchestrate import execute_task
from evaluation.self_research import auto_evaluate_candidates
from ledger import Ledger
from research_agent.model import ModelClient
from research_agent.roles import RoleAgents

from .run_token_pilot import (BASE_URL, BUDGET_TOKENS, MAX_OUTPUT_TOKENS, MODEL,
                              PROJECT, RESERVE_TOKENS, _stamp, _write)

TASKS = ("spd-membership-filter", "tab-german-credit")
PRICE_NAME = "prices_real_token_test.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_project(source: Path, campaign: Path) -> Path:
    """Copy only practice inputs; never alter the official bank or tasks."""
    source = source.resolve()
    project = campaign / "project"
    project.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "config", project / "config")
    (project / "tasks").mkdir()
    bank = json.loads((source / "tasks/bank.json").read_text(encoding="utf-8"))
    selected = [entry for entry in bank["tasks"] if entry["task_id"] in TASKS]
    if len(selected) != len(TASKS) or any(entry["tier"] != "practice" for entry in selected):
        raise ValueError("The token self-research pilot requires its two practice tasks")
    bank["tasks"] = selected
    _write(project / "tasks/bank.json", bank)
    source_tasks = {}
    for task_id in TASKS:
        shutil.copytree(source / "tasks" / task_id, project / "tasks" / task_id)
        original = source / "tasks" / task_id / "task.json"
        copied = project / "tasks" / task_id / "task.json"
        source_tasks[task_id] = {"original_task_sha256": _sha(original),
                                 "copied_task_sha256": _sha(copied)}
    prices = json.loads((source / "config/prices_token_pilot.json").read_text(encoding="utf-8"))
    if prices.get("currency") != "TOKENS" or prices.get("unit_tokens") != 1:
        raise ValueError("Expected the four-bucket one-token price table")
    if set(prices["models"][MODEL].values()) != {1}:
        raise ValueError("All four token buckets must cost one unit")
    prices["currency"] = "TEST"
    prices["test_only"] = True
    _write(project / "config" / PRICE_NAME, prices)
    _write(campaign / "protocol.json", {
        "schema": "token_self_research_protocol/1",
        "label": "REAL MODEL DEV; TEST UNIT = ONE REPORTED TOKEN; NOT MONEY",
        "model": MODEL, "tasks": list(TASKS), "proposal_rounds": 1,
        "repetitions_per_candidate": 3,
        "budget_tokens_per_task": BUDGET_TOKENS,
        "reserve_tokens_per_call": RESERVE_TOKENS,
        "max_output_tokens_requested": MAX_OUTPUT_TOKENS,
        "prices_sha256": _sha(project / "config" / PRICE_NAME),
        "default_config_sha256": _sha(project / "config/agent_token_pilot_v2.json"),
        "source_tasks": source_tasks,
    })
    return project


@contextmanager
def _project_override(project: Path) -> Iterator[None]:
    values = {"RA_PROJECT_ROOT": str(project), "RA_TASKS_ROOT": str(project / "tasks")}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be injected in the process environment")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                               cwd=PROJECT, text=True).strip():
        raise SystemExit("Commit source changes before the token self-research pilot")
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=PROJECT, text=True).strip()
    campaign = PROJECT / "runs" / f"token-self-research-{_stamp()}-{secrets.token_hex(2)}"
    project = prepare_project(PROJECT, campaign)
    protocol = json.loads((campaign / "protocol.json").read_text(encoding="utf-8"))
    protocol["code_commit"] = code_commit
    _write(campaign / "protocol.json", protocol)
    from openai import OpenAI
    transport = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL,
                       timeout=180.0, max_retries=0)

    def roles_factory(name: str, rep: int, task_id: str, run_dir: Path) -> RoleAgents:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config_path = project / manifest["agent_config"]["path"]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ledger = Ledger.open(run_dir, task_id)
        model = ModelClient(MODEL, ledger, client=transport,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                            budget_limit=BUDGET_TOKENS, reserve_cost=RESERVE_TOKENS,
                            audit_dir=run_dir / "tasks" / task_id / "model_responses")
        return RoleAgents(model, config, project_root=project)

    def token_execute(**kwargs):
        return execute_task(max_output_tokens=MAX_OUTPUT_TOKENS, **kwargs)

    print(f"Starting real-model dev self-research: {campaign}", flush=True)
    try:
        with _project_override(project):
            result = auto_evaluate_candidates(
                project_root=project, study_dir=project / "runs/auto-study",
                harness=harness, default_config=project / "config/agent_token_pilot_v2.json",
                baseline_solutions={task_id: project / "tasks" / task_id / "workspace/original.py"
                                    for task_id in TASKS},
                prices_path=project / "config" / PRICE_NAME, model=MODEL,
                code_commit=code_commit, rounds=1,
                max_output_tokens=MAX_OUTPUT_TOKENS, test_only_budget=BUDGET_TOKENS,
                proposal_client=transport, roles_factory=roles_factory,
                task_executor=token_execute,
            )
        summary = {
            "schema": "token_self_research_summary/1",
            "label": protocol["label"], "campaign_dir": str(campaign),
            "study_report": str(project / "runs/auto-study/auto_self_research_report.json"),
            "proposal_attempts": result["proposal_attempts"],
            "ranking": result["ranking"], "selection": result["selection"],
            "total_task_tokens": result["total_task_cost"],
            "total_proposal_tokens": result["total_proposal_cost"],
            "total_research_tokens": result["total_research_cost"],
            "run_count": len(result["reports"]),
        }
        _write(campaign / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        _write(campaign / "summary.json", {
            "schema": "token_self_research_summary/1",
            "label": protocol["label"], "campaign_dir": str(campaign),
            "error": str(exc),
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
