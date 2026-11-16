"""Run C's automatic self-research with real model calls on two practice tasks.

This is a dev-only, copied-project experiment. TEST units are exactly the
provider-reported four-bucket token counts (one unit per token), not money or
an official validation/exam result. The API key is read only from the process
environment and never written to the copied project or campaign summary.
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
from threading import local
from time import monotonic
from typing import Iterator
from urllib.parse import urlsplit

import harness
from evaluation.orchestrate import execute_task
from evaluation.self_research import auto_evaluate_candidates, evaluate_candidates
from ledger import Ledger
from research_agent.model import ModelClient
from research_agent.roles import RoleAgents

from .run_token_pilot import (BASE_URL, BUDGET_TOKENS, MAX_OUTPUT_TOKENS, MODEL,
                              PROJECT, RESERVE_TOKENS, _stamp, _write)

TASKS = ("spd-membership-filter", "tab-german-credit")
PRICE_NAME = "prices_real_token_test.json"


def all_practice_task_ids(source: Path) -> tuple[str, ...]:
    """Select exactly the five speedup and five tabular practice tasks in bank order."""
    bank = json.loads((source / "tasks/bank.json").read_text(encoding="utf-8"))
    task_ids = tuple(entry["task_id"] for entry in bank["tasks"]
                     if entry["tier"] == "practice")
    if len(task_ids) != 10 or len(set(task_ids)) != 10:
        raise ValueError("The practice study requires ten distinct practice tasks")
    scenarios = [json.loads((source / "tasks" / task_id / "task.json").read_text(encoding="utf-8"))["scenario"]
                 for task_id in task_ids]
    if scenarios.count("speedup") != 5 or scenarios.count("tabular") != 5:
        raise ValueError("The practice study requires five tasks in each scenario")
    return task_ids


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timing(started_utc: str, started_tick: float) -> dict[str, str | float]:
    return {"started_utc": started_utc, "ended_utc": _utc_now(),
            "elapsed_seconds": max(0.0, monotonic() - started_tick)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_project(source: Path, campaign: Path, *,
                    task_ids: tuple[str, ...] = TASKS, proposal_rounds: int = 1,
                    max_workers: int = 3) -> Path:
    """Copy only practice inputs; never alter the official bank or tasks."""
    source = source.resolve()
    project = campaign / "project"
    project.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "config", project / "config")
    (project / "tasks").mkdir()
    bank = json.loads((source / "tasks/bank.json").read_text(encoding="utf-8"))
    selected = [entry for entry in bank["tasks"] if entry["task_id"] in task_ids]
    if (not task_ids or len(set(task_ids)) != len(task_ids) or len(selected) != len(task_ids)
            or any(entry["tier"] != "practice" for entry in selected)):
        raise ValueError("The token study requires distinct practice tasks only")
    bank["tasks"] = selected
    _write(project / "tasks/bank.json", bank)
    source_tasks = {}
    wall_seconds_per_task = {}
    for task_id in task_ids:
        shutil.copytree(source / "tasks" / task_id, project / "tasks" / task_id)
        original = source / "tasks" / task_id / "task.json"
        copied = project / "tasks" / task_id / "task.json"
        source_tasks[task_id] = {"original_task_sha256": _sha(original),
                                 "copied_task_sha256": _sha(copied)}
        copied_task = json.loads(copied.read_text(encoding="utf-8"))
        wall_seconds_per_task[task_id] = copied_task["budget"]["wall_seconds"]
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
        "model": MODEL, "tasks": list(task_ids), "proposal_rounds": proposal_rounds,
        "repetitions_per_candidate": 3,
        "max_workers": max_workers,
        "budget_tokens_per_task": BUDGET_TOKENS,
        "wall_seconds_per_task": wall_seconds_per_task,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-only", action="store_true",
                        help="Three default repetitions; no proposal or candidate")
    parser.add_argument("--all-practice", action="store_true",
                        help="Use all ten practice tasks instead of the two-task pilot")
    parser.add_argument("--base-url",
                        help="OpenAI-compatible Responses API base URL")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    if args.all_practice and not args.base_url:
        raise SystemExit("--all-practice requires an explicit --base-url")
    if args.all_practice and not args.default_only:
        raise SystemExit("--all-practice currently requires --default-only; do not start automatic proposals")
    base_url = args.base_url or BASE_URL
    parsed_url = urlsplit(base_url)
    if (parsed_url.scheme != "https" or not parsed_url.netloc or parsed_url.username
            or parsed_url.password or parsed_url.query or parsed_url.fragment):
        raise SystemExit("--base-url must be an HTTPS URL without credentials or query parameters")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be injected in the process environment")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                               cwd=PROJECT, text=True).strip():
        raise SystemExit("Commit source changes before the token self-research pilot")
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=PROJECT, text=True).strip()
    task_ids = all_practice_task_ids(PROJECT) if args.all_practice else TASKS
    if args.all_practice:
        prefix = "token-practice10-default" if args.default_only else "token-practice10-self-research"
    else:
        prefix = "token-parallel-default" if args.default_only else "token-self-research"
    campaign = PROJECT / "runs" / f"{prefix}-{_stamp()}-{secrets.token_hex(2)}"
    project = prepare_project(PROJECT, campaign,
                              task_ids=task_ids,
                              proposal_rounds=0 if args.default_only else 1,
                              max_workers=args.workers)
    protocol = json.loads((campaign / "protocol.json").read_text(encoding="utf-8"))
    protocol["code_commit"] = code_commit
    protocol["base_url"] = base_url
    protocol["study_type"] = "parallel_default_benchmark" if args.default_only else "auto_self_research"
    _write(campaign / "protocol.json", protocol)
    from openai import OpenAI
    worker_state = local()

    def task_transport():
        if not hasattr(worker_state, "client"):
            worker_state.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                         base_url=base_url, timeout=180.0, max_retries=0)
        return worker_state.client

    def roles_factory(name: str, rep: int, task_id: str, run_dir: Path) -> RoleAgents:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config_path = project / manifest["agent_config"]["path"]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ledger = Ledger.open(run_dir, task_id)
        model = ModelClient(MODEL, ledger, client=task_transport(),
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                            budget_limit=BUDGET_TOKENS, reserve_cost=RESERVE_TOKENS,
                            audit_dir=run_dir / "tasks" / task_id / "model_responses")
        return RoleAgents(model, config, project_root=project)

    def token_execute(**kwargs):
        return execute_task(max_output_tokens=MAX_OUTPUT_TOKENS, **kwargs)

    started_utc = _utc_now()
    started_tick = monotonic()
    print(f"Starting real-model dev {protocol['study_type']}: {campaign}", flush=True)
    try:
        with _project_override(project):
            common = dict(project_root=project, harness=harness,
                          baseline_solutions={task_id: project / "tasks" / task_id / "workspace/original.py"
                                              for task_id in task_ids},
                          prices_path=project / "config" / PRICE_NAME, model=MODEL,
                          code_commit=code_commit, test_only_budget=BUDGET_TOKENS,
                          max_workers=args.workers, roles_factory=roles_factory,
                          task_executor=token_execute)
            if args.default_only:
                result = evaluate_candidates(
                    study_dir=project / "runs/default-study",
                    candidates={"default": project / "config/agent_token_pilot_v2.json"},
                    **common)
            else:
                proposal_transport = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                            base_url=base_url, timeout=180.0, max_retries=0)
                result = auto_evaluate_candidates(
                    study_dir=project / "runs/auto-study",
                    default_config=project / "config/agent_token_pilot_v2.json",
                    rounds=1, max_output_tokens=MAX_OUTPUT_TOKENS,
                    proposal_client=proposal_transport, **common)
        if args.default_only:
            summary = {"schema": "token_parallel_default_summary/1",
                       "label": protocol["label"], "campaign_dir": str(campaign),
                       "study_report": str(project / "runs/default-study/self_research_report.json"),
                       "ranking": result["ranking"],
                       "total_task_tokens": sum(report["total_cost"] for report in result["reports"]),
                       "run_count": len(result["reports"])}
        else:
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
        summary.update(_timing(started_utc, started_tick))
        _write(campaign / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        summary = {
            "schema": "token_parallel_default_summary/1" if args.default_only else "token_self_research_summary/1",
            "label": protocol["label"], "campaign_dir": str(campaign),
            "error": str(exc),
            **_timing(started_utc, started_tick),
        }
        report_path = (project / "runs/default-study/self_research_report.json" if args.default_only
                       else project / "runs/auto-study/auto_self_research_report.json")
        if report_path.is_file():
            summary["study_report"] = str(report_path)
            try:
                partial = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as report_exc:
                summary["study_report_read_error"] = str(report_exc)
            else:
                if partial.get("status") == "partial":
                    summary.update({"status": "partial",
                                    "known_task_tokens": partial.get("known_task_cost"),
                                    "known_proposal_tokens": partial.get("known_proposal_cost"),
                                    "known_total_tokens": partial.get(
                                        "known_total_cost", partial.get("known_task_cost")),
                                    "known_total_tokens_is_lower_bound": True,
                                    "unknown_task_usage_calls": partial.get(
                                        "unknown_task_usage_calls", partial.get("unknown_usage_calls", [])),
                                    "unknown_proposal_usage_seqs": partial.get(
                                        "unknown_proposal_usage_seqs", [])})
        _write(campaign / "summary.json", summary)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
