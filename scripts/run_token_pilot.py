"""Run two real-model practice tasks as a dev-only token-metered pilot.

OPENAI_API_KEY is read only from the environment. No key or request payload is
stored by this script; the normal run manifest, ledger and replay evidence are.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Iterator

import harness
from evaluation.finalize import finish_run, start_run
from evaluation.manifest import create_manifest
from evaluation.orchestrate import execute_task
from evaluation.report import run_report
from ledger import Ledger
from research_agent.model import ModelClient
from research_agent.roles import RoleAgents


PROJECT = Path(__file__).resolve().parents[1]
TASKS = ("spd-membership-filter", "tab-mushroom-edibility")
ALLOWED_TASKS = TASKS + ("tab-german-credit",)
MODEL = "qwen3.5-27b"
BUDGET_TOKENS = 40_000
RESERVE_TOKENS = 5_000
WALL_SECONDS = 900
MAX_OUTPUT_TOKENS = 2048
BASE_URL = "https://ai-gateway.ailab.jiuan.com/v1"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def prepare_task_copies(project_root: Path, campaign_dir: Path, task_ids: tuple[str, ...],
                        agent_config_path: Path | None = None) -> Path:
    """Keep the official bank and task.json files untouched."""
    copied = campaign_dir / "copied_tasks"
    copied.mkdir(parents=True, exist_ok=False)
    bank = json.loads((project_root / "tasks/bank.json").read_text(encoding="utf-8"))
    entries = {entry["task_id"]: entry for entry in bank["tasks"]}
    if any(task_id not in ALLOWED_TASKS or entries[task_id]["tier"] != "practice" for task_id in task_ids):
        raise ValueError("The token pilot accepts only the selected practice tasks")
    bank["tasks"] = [entries[task_id] for task_id in task_ids]
    _write(copied / "bank.json", bank)
    sources = {}
    for task_id in task_ids:
        source = project_root / "tasks" / task_id
        target = copied / task_id
        shutil.copytree(source, target)
        task_file = target / "task.json"
        task = json.loads(task_file.read_text(encoding="utf-8"))
        sources[task_id] = {"original_task_sha256": _sha(source / "task.json")}
        task["budget"] = {"money": BUDGET_TOKENS, "wall_seconds": WALL_SECONDS}
        _write(task_file, task)
        sources[task_id]["copied_task_sha256"] = _sha(task_file)
    agent_config_path = agent_config_path or project_root / "config/agent_token_pilot.json"
    _write(campaign_dir / "pilot_inputs.json", {
        "schema": "token_pilot_inputs/1", "label": "DEV TOKEN MEASUREMENT - NOT MONEY",
        "model": MODEL, "tasks": list(task_ids), "budget_tokens_per_task": BUDGET_TOKENS,
        "reserve_tokens_per_call": RESERVE_TOKENS,
        "max_output_tokens_requested": MAX_OUTPUT_TOKENS,
        "wall_seconds_per_task": WALL_SECONDS,
        "prices_path": "config/prices_token_pilot.json",
        "prices_sha256": _sha(project_root / "config/prices_token_pilot.json"),
        "agent_config_path": str(agent_config_path.relative_to(project_root)),
        "agent_config_sha256": _sha(agent_config_path),
        "source_tasks": sources,
    })
    return copied


@contextmanager
def _tasks_override(path: Path) -> Iterator[None]:
    previous = os.environ.get("RA_TASKS_ROOT")
    os.environ["RA_TASKS_ROOT"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RA_TASKS_ROOT", None)
        else:
            os.environ["RA_TASKS_ROOT"] = previous


def _run_one(project_root: Path, run_dir: Path, task_id: str,
             copied_tasks: Path, base_url: str, config_path: Path,
             library_source: Path | None) -> dict:
    from openai import OpenAI

    prices_path = project_root / "config/prices_token_pilot.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    start_version = "lib-0"
    if library_source is not None:
        shutil.copytree(library_source, run_dir / "library")
        state = json.loads((run_dir / "library/state.json").read_text(encoding="utf-8"))
        if state.get("schema") != "library/1":
            raise ValueError("Library source is not a library/1 state")
        start_version = state["version"]
    create_manifest(
        project_root=project_root, run_dir=run_dir, phase="dev", config="default", rep=1,
        bank_version="bank-v1", tasks=[task_id], library_root=run_dir / "library",
        library_start_version=start_version, library_frozen=False, model=MODEL,
        agent_config_path=config_path, prices_path=prices_path, code_commit=code_commit,
    )
    start_run(run_dir)
    ledger = Ledger.open(run_dir, task_id)
    transport = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url,
                       timeout=180.0, max_retries=0)
    model = ModelClient(MODEL, ledger, client=transport,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        budget_limit=BUDGET_TOKENS, reserve_cost=RESERVE_TOKENS,
                        audit_dir=run_dir / "tasks" / task_id / "model_responses")
    roles = RoleAgents(model, config, project_root=project_root)
    with _tasks_override(copied_tasks):
        execute_task(run_dir=run_dir, task_id=task_id, harness=harness,
                     baseline_solution=copied_tasks / task_id / "workspace/original.py",
                     roles=roles, max_output_tokens=MAX_OUTPUT_TOKENS)
    finish_run(run_dir)
    report = run_report(run_dir)
    task = report["tasks"][0]
    events = ledger._events()
    stop = next(event for event in events if event["kind"] == "stop")
    return {
        "task_id": task_id, "run_dir": str(run_dir), "stop_reason": stop["reason"],
        "code_commit": code_commit,
        "library_start_version": start_version,
        "library_end_version": json.loads((run_dir / "library/state.json").read_text())["version"],
        "baseline_dev_score": json.loads((run_dir / "tasks" / task_id / "nodes/n0000/score.json").read_text())["score"],
        "best_node": task["best_node"], "dev_score": task["dev_score"],
        "holdout_score": task["holdout_score"], "model_calls": task["model_calls"],
        "token_buckets": task["tokens"], "total_tokens": task["total_cost"],
        "experiments": task["experiments"], "report_path": str(run_dir / "report.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--task", choices=ALLOWED_TASKS, action="append")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--library-source", type=Path,
                        help="Existing run/library directory for a dev-only reuse check")
    parser.add_argument("--agent-config", type=Path,
                        default=Path("config/agent_token_pilot.json"))
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be injected in the process environment")
    root = args.project_root.resolve()
    worktree_clean = not subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root, text=True).strip()
    config_path = (root / args.agent_config).resolve()
    if not config_path.is_relative_to(root / "config") or not config_path.is_file():
        raise SystemExit("--agent-config must be an existing project config file")
    library_source = None
    if args.library_source is not None:
        library_source = (root / args.library_source).resolve()
        if (not library_source.is_relative_to(root / "runs")
                or not (library_source / "state.json").is_file()):
            raise SystemExit("--library-source must be an existing run library")
    task_ids = tuple(dict.fromkeys(args.task or TASKS))
    campaign = root / "runs" / f"token-pilot-{_stamp()}-{secrets.token_hex(2)}"
    copied = prepare_task_copies(root, campaign, task_ids, config_path)
    summary = {"schema": "token_pilot_report/1", "label": "DEV TOKEN MEASUREMENT - NOT MONEY",
               "campaign_dir": str(campaign), "model": MODEL, "base_url": args.base_url,
               "agent_config_path": str(config_path.relative_to(root)),
               "worktree_clean_at_start": worktree_clean,
               "library_source": str(library_source) if library_source else None,
               "library_source_state_sha256": _sha(library_source / "state.json")
               if library_source else None,
               "budget_tokens_per_task": BUDGET_TOKENS,
               "reserve_tokens_per_call": RESERVE_TOKENS,
               "attempts": [], "tasks": [], "failed": []}
    _write(campaign / "summary.json", summary)
    for task_id in task_ids:
        run_dir = root / "runs" / f"run-{_stamp()}-{secrets.token_hex(2)}"
        summary["attempts"].append({"task_id": task_id, "run_dir": str(run_dir)})
        _write(campaign / "summary.json", summary)
        print(f"Starting {task_id}: {run_dir}", flush=True)
        try:
            row = _run_one(root, run_dir, task_id, copied, args.base_url,
                           config_path, library_source)
            summary["tasks"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        except Exception as exc:
            summary["failed"].append({"task_id": task_id, "error": str(exc)})
            _write(campaign / "summary.json", summary)
            raise
        _write(campaign / "summary.json", summary)
    print(f"Pilot evidence: {campaign / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
