"""Dev-only transfer test of a previously proposed configuration on fresh practice tasks.

Both variants run three times on the same two practice tasks. TEST units are
provider-reported four-bucket tokens, not money. No validation/exam is opened.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

import harness
from evaluation.orchestrate import execute_task
from evaluation.self_research import evaluate_candidates, verify_auto_selected
from ledger import Ledger
from research_agent.model import ModelClient
from research_agent.roles import RoleAgents

from .run_token_pilot import (BASE_URL, BUDGET_TOKENS, MAX_OUTPUT_TOKENS, MODEL,
                              PROJECT, RESERVE_TOKENS, _stamp, _write)
from .run_token_self_research import (PRICE_NAME, _project_override, _sha,
                                      prepare_project)

TRANSFER_TASKS = ("spd-distance-matrix", "tab-adult-income")


def prepare_candidate(project: Path, source_campaign: Path) -> tuple[Path, dict]:
    source_campaign = source_campaign.resolve()
    if not source_campaign.is_relative_to((PROJECT / "runs").resolve()):
        raise ValueError("Source campaign must be inside project runs/")
    source_project = source_campaign / "project"
    report = json.loads((source_project / "runs/auto-study/auto_self_research_report.json")
                        .read_text(encoding="utf-8"))
    if report.get("label") != "TEST ONLY" or report.get("currency") != "TEST":
        raise ValueError("Transfer source must be a dev TEST-only self-research run")
    candidate = verify_auto_selected(report).resolve()
    if not candidate.is_relative_to(source_project.resolve()):
        raise ValueError("Candidate path escaped the source campaign")
    relative = candidate.relative_to(source_project.resolve())
    copied = project / relative
    copied.parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate.parent, copied.parent)
    copied_files = {}
    for original in candidate.parent.iterdir():
        if not original.is_file():
            raise ValueError("Candidate directory contains a non-file")
        digest = _sha(original)
        if _sha(copied.parent / original.name) != digest:
            raise ValueError("Candidate changed during transfer copy")
        copied_files[original.name] = digest
    return copied, {"source_campaign": str(source_campaign),
                    "source_candidate": report["selection"]["candidate"],
                    "source_candidate_sha256": _sha(candidate),
                    "source_candidate_files_sha256": copied_files,
                    "source_research_tokens": report["total_research_cost"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be injected in the process environment")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                               cwd=PROJECT, text=True).strip():
        raise SystemExit("Commit source changes before the token transfer test")
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=PROJECT, text=True).strip()
    campaign = PROJECT / "runs" / f"token-transfer-{_stamp()}-{secrets.token_hex(2)}"
    project = prepare_project(PROJECT, campaign, task_ids=TRANSFER_TASKS,
                              proposal_rounds=0)
    candidate, source = prepare_candidate(project, args.source_campaign)
    protocol_path = campaign / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({"study_type": "cross_task_transfer", "code_commit": code_commit,
                     **source})
    _write(protocol_path, protocol)
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

    print(f"Starting real-model dev transfer test: {campaign}", flush=True)
    try:
        with _project_override(project):
            result = evaluate_candidates(
                project_root=project, study_dir=project / "runs/transfer-study",
                harness=harness,
                candidates={"default": project / "config/agent_token_pilot_v2.json",
                            "transfer_auto_01": candidate},
                baseline_solutions={task_id: project / "tasks" / task_id / "workspace/original.py"
                                    for task_id in TRANSFER_TASKS},
                prices_path=project / "config" / PRICE_NAME, model=MODEL,
                code_commit=code_commit, test_only_budget=BUDGET_TOKENS,
                roles_factory=roles_factory, task_executor=token_execute,
            )
        summary = {"schema": "token_transfer_summary/1",
                   "label": "REAL MODEL DEV; TEST UNIT = ONE REPORTED TOKEN; NOT MONEY",
                   "campaign_dir": str(campaign),
                   "study_report": str(project / "runs/transfer-study/self_research_report.json"),
                   "ranking": result["ranking"], "selection": result["selection"],
                   "total_task_tokens": sum(report["total_cost"] for report in result["reports"]),
                   "run_count": len(result["reports"])}
        _write(campaign / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        _write(campaign / "summary.json", {"schema": "token_transfer_summary/1",
                                            "campaign_dir": str(campaign), "error": str(exc)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
