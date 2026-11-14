"""Prepare and run one evidence-calibrated, practice-only multi-model candidate.

Qwen is an API proxy for a possible local model. TEST cost is one unit per
provider-reported token for every model, not money or local GPU cost.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from getpass import getpass
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from statistics import mean, median
from threading import Lock, local
from time import monotonic
from urllib.parse import urlsplit

import harness
from evaluation.finalize import finish_run, start_run
from evaluation.manifest import create_manifest
from evaluation.orchestrate import execute_task
from evaluation.report import run_report
from evaluation.route_calibration import calibrate_policy
from evaluation.self_research import _SerializedHarness
from ledger import Ledger
from library import Library
from research_agent.model import ModelClient
from research_agent.roles import RoleAgents
from research_agent.routing import RoutingClient, validate_policy

from .run_token_pilot import BUDGET_TOKENS, MAX_OUTPUT_TOKENS, MODEL, PROJECT, RESERVE_TOKENS, _stamp, _write
from .run_token_self_research import _project_override, all_practice_task_ids, prepare_project


DEFAULT_BASE_URL = "https://ai-gateway.theunifiedcare.com/v1"
DEFAULT_BASELINE = PROJECT.parent / "research-agent-team-yi/runs/token-practice10-default-20261114-091613-02ab"
PLANNER_MODEL = "gpt-6-astra"
EXECUTOR_MODEL = "claude-opus-5-5"
PRICE_NAME = "prices_real_token_test.json"
POLICY_NAME = "route_policy_self_evolving.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _source_commit() -> str:
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                               cwd=PROJECT, text=True).strip():
        raise ValueError("Commit source changes before preparing a routed campaign")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()


def prepare_campaign(baseline: Path, base_url: str, workers: int) -> Path:
    commit = _source_commit()
    task_ids = all_practice_task_ids(PROJECT)
    campaign = PROJECT / "runs" / f"self-evolving-practice10-{_stamp()}-{secrets.token_hex(2)}"
    project = prepare_project(PROJECT, campaign, task_ids=task_ids,
                              proposal_rounds=0, max_workers=workers)
    policy = calibrate_policy(baseline, default_model=MODEL,
                              planner_model=PLANNER_MODEL, executor_model=EXECUTOR_MODEL)
    policy_path = project / "config" / POLICY_NAME
    _write(policy_path, policy)
    prices_path = project / "config" / PRICE_NAME
    prices = json.loads(prices_path.read_text(encoding="utf-8"))
    for remote in (PLANNER_MODEL, EXECUTOR_MODEL):
        prices["models"][remote] = {key: 1 for key in ("input", "cached_input", "output", "reasoning")}
    _write(prices_path, prices)
    protocol_path = campaign / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({
        "schema": "self_evolving_protocol/1", "study_type": "one_self_evolving_candidate",
        "label": "REAL MODEL DEV; TEST UNIT = ONE REPORTED TOKEN; NOT MONEY",
        "code_commit": commit, "base_url": base_url,
        "default_model": MODEL, "remote_models": [PLANNER_MODEL, EXECUTOR_MODEL],
        "routing_policy_sha256": _sha(policy_path),
        "prices_sha256": _sha(prices_path),
        "baseline_summary_sha256": policy["calibration"]["baseline_summary_sha256"],
        "baseline_report_sha256": policy["calibration"]["baseline_report_sha256"],
        "baseline_code_commit": policy["calibration"]["baseline_code_commit"],
        "quality_gate": {"task_mean_tolerance": 0.1, "max_regressed_tasks": 2,
                         "scenario_median_tolerance": 0.1},
    })
    _write(protocol_path, protocol)
    _write(campaign / "summary.json", {
        "schema": "self_evolving_campaign/1", "status": "prepared",
        "campaign_dir": str(campaign), "baseline_campaign": str(Path(baseline).resolve()),
        "policy_path": str(policy_path), "code_commit": commit,
        "task_count": len(task_ids), "repetitions": 3,
        "proposal_model_calls": 0, "proposal_tokens": 0,
    })
    return campaign


def _validated_campaign(campaign: Path) -> tuple[Path, dict, dict]:
    campaign = Path(campaign).expanduser().resolve(strict=True)
    if campaign.parent != PROJECT / "runs":
        raise ValueError("Campaign must be inside this worktree's runs directory")
    protocol = json.loads((campaign / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != "self_evolving_protocol/1" or protocol.get("code_commit") != _source_commit():
        raise ValueError("Campaign code commit or protocol differs from the checkout")
    project = campaign / "project"
    policy_path = project / "config" / POLICY_NAME
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy(policy)
    if (_sha(policy_path) != protocol["routing_policy_sha256"]
            or _sha(project / "config" / PRICE_NAME) != protocol["prices_sha256"]):
        raise ValueError("Sealed routing policy or price table changed")
    baseline = Path(policy["calibration"]["baseline_campaign"])
    if (_sha(baseline / "summary.json") != protocol["baseline_summary_sha256"]
            or _sha(baseline / "project/runs/default-study/self_research_report.json") != protocol["baseline_report_sha256"]):
        raise ValueError("Frozen baseline summary or report changed")
    summary = json.loads((campaign / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "prepared":
        raise ValueError("Campaign was already started; prepare a new campaign")
    if list(all_practice_task_ids(project)) != protocol["tasks"]:
        raise ValueError("Copied practice bank differs from protocol")
    return project, protocol, policy


def _preflight(campaign: Path, client: object, policy: dict) -> dict:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    rows = []
    for model in sorted({policy["default_model"], *policy["role_models"].values()}):
        audit = campaign / "preflight" / model
        model_client = ModelClient(model, None, client=client, max_output_tokens=1024, audit_dir=audit)
        try:
            result = model_client.call_json(role="planner", purpose="plan", effort="low",
                                            system_prompt="Reply with JSON ok=true.",
                                            payload={"probe": "respond in the requested JSON shape"},
                                            schema=schema)
            row = {"model": model, "ok": result.data.get("ok") is True,
                   "tokens": vars(result.usage), "seconds": result.seconds,
                   "response_id": result.response_id}
        except Exception as exc:
            files = list(audit.glob("response-*.json")) if audit.exists() else []
            captured = json.loads(files[-1].read_text(encoding="utf-8")) if files else {}
            row = {"model": model, "ok": False, "tokens": captured.get("tokens", {}),
                   "seconds": None, "error": str(exc)[:300]}
        rows.append(row)
    result = {"schema": "route_preflight/1", "ts": _now(),
              "purpose": "Responses JSON contract and model availability",
              "models": rows,
              "total_reported_tokens": sum(sum(row["tokens"].values()) for row in rows)}
    _write(campaign / "preflight.json", result)
    return result


def _comparison(baseline: Path, reports: list[dict], protocol: dict) -> dict:
    baseline_report = json.loads((baseline / "project/runs/default-study/self_research_report.json").read_text(encoding="utf-8"))
    baseline_rows: dict[str, list[float]] = {}
    candidate_rows: dict[str, list[float]] = {}
    candidate_costs: dict[str, list[float]] = {}
    baseline_costs: dict[str, list[float]] = {}
    remote_calls: dict[str, list[int]] = {}
    remote_tokens: dict[str, list[int]] = {}
    failed_remote_calls: dict[str, list[int]] = {}
    for report in baseline_report["reports"]:
        for task in report["tasks"]:
            baseline_rows.setdefault(task["task_id"], []).append(float(task["holdout_score"]))
            baseline_costs.setdefault(task["task_id"], []).append(float(task["total_cost"]))
    for report in reports:
        for task in report["tasks"]:
            task_id = task["task_id"]
            candidate_rows.setdefault(task_id, []).append(float(task["holdout_score"]))
            candidate_costs.setdefault(task_id, []).append(float(task["total_cost"]))
            remote_calls.setdefault(task_id, []).append(task["routing"]["remote_calls"])
            remote_tokens.setdefault(task_id, []).append(task["routing"]["remote_tokens"])
            failed_remote_calls.setdefault(task_id, []).append(task["routing"]["failed_remote_calls"])
    tasks = []
    for task_id in protocol["tasks"]:
        baseline_scores = baseline_rows[task_id]
        candidate_scores = candidate_rows[task_id]
        if len(baseline_scores) != 3 or len(candidate_scores) != 3:
            raise ValueError("Comparison requires three results per practice task")
        tasks.append({"task_id": task_id, "scenario": "speedup" if task_id.startswith("spd-") else "tabular",
                      "baseline_scores": baseline_scores, "candidate_scores": candidate_scores,
                      "mean_score_delta": mean(candidate_scores) - mean(baseline_scores),
                      "baseline_tokens": baseline_costs[task_id],
                      "candidate_tokens": candidate_costs[task_id],
                      "candidate_remote_calls": remote_calls[task_id],
                      "candidate_remote_tokens": remote_tokens[task_id],
                      "candidate_failed_remote_calls": failed_remote_calls[task_id]})
    scenarios = {}
    for scenario in ("speedup", "tabular"):
        group = [row for row in tasks if row["scenario"] == scenario]
        scenarios[scenario] = {
            "baseline_median_task_mean": median(mean(row["baseline_scores"]) for row in group),
            "candidate_median_task_mean": median(mean(row["candidate_scores"]) for row in group),
        }
    gate = protocol["quality_gate"]
    regressed = [row["task_id"] for row in tasks if row["mean_score_delta"] < -gate["task_mean_tolerance"]]
    scenario_ok = all(row["candidate_median_task_mean"] >= row["baseline_median_task_mean"] - gate["scenario_median_tolerance"]
                      for row in scenarios.values())
    return {"schema": "route_practice_comparison/1", "label": "EXPLORATORY; SAME PRACTICE TASKS",
            "baseline_code_commit": protocol["baseline_code_commit"], "candidate_code_commit": protocol["code_commit"],
            "quality_gate": gate, "quality_gate_passed": len(regressed) <= gate["max_regressed_tasks"] and scenario_ok,
            "regressed_tasks": regressed, "scenarios": scenarios, "tasks": tasks}


def run_campaign(campaign: Path) -> dict:
    campaign = Path(campaign).resolve()
    project, protocol, policy = _validated_campaign(campaign)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in your visible terminal before running live models")
    from openai import OpenAI
    base_url = protocol["base_url"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url,
                    timeout=180.0, max_retries=0)
    started = _now()
    tick = monotonic()
    _write(campaign / "summary.json", {"schema": "self_evolving_campaign/1",
                                      "status": "running", "campaign_dir": str(campaign),
                                      "started_utc": started})
    try:
        preflight = _preflight(campaign, client, policy)
        if not all(row["ok"] for row in preflight["models"]):
            raise RuntimeError("Model preflight failed; inspect preflight.json before running tasks")
        task_ids = tuple(protocol["tasks"])
        worker_state = local()

        def task_client():
            if not hasattr(worker_state, "client"):
                worker_state.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url,
                                             timeout=180.0, max_retries=0)
            return worker_state.client

        def one_rep(rep: int) -> dict:
            run_dir = project / "runs" / f"run-{_stamp()}-{secrets.token_hex(2)}"
            run_dir.mkdir(parents=True, exist_ok=False)
            library_root = run_dir / "library"
            library = Library.open(library_root)
            create_manifest(
                project_root=project, run_dir=run_dir, phase="dev", config="self_evolving",
                rep=rep, bank_version=json.loads((project / "tasks/bank.json").read_text())["version"],
                tasks=list(task_ids), library_root=library_root,
                library_start_version=library.version, library_frozen=False,
                model=MODEL, agent_config_path=project / "config/agent_token_pilot_v2.json",
                prices_path=project / "config" / PRICE_NAME,
                routing_config_path=project / "config" / POLICY_NAME,
                code_commit=protocol["code_commit"],
            )
            start_run(run_dir)
            for task_id in task_ids:
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                config = json.loads((project / manifest["agent_config"]["path"]).read_text(encoding="utf-8"))
                ledger = Ledger.open(run_dir, task_id)
                models = {model: ModelClient(model, ledger, client=task_client(),
                                             max_output_tokens=MAX_OUTPUT_TOKENS,
                                             budget_limit=BUDGET_TOKENS, reserve_cost=RESERVE_TOKENS,
                                             audit_dir=run_dir / "tasks" / task_id / "model_responses")
                          for model in manifest["models"]}
                roles = RoleAgents(RoutingClient(models, ledger, policy), config, project_root=project)
                execute_task(run_dir=run_dir, task_id=task_id, harness=task_harness,
                             baseline_solution=project / "tasks" / task_id / "workspace/original.py",
                             roles=roles, test_only_budget=BUDGET_TOKENS,
                             max_output_tokens=MAX_OUTPUT_TOKENS)
            finish_run(run_dir)
            return run_report(run_dir)

        task_harness = (_SerializedHarness(harness, Lock()) if protocol["max_workers"] > 1 else harness)
        with _project_override(project):
            if protocol["max_workers"] == 1:
                reports = [one_rep(rep) for rep in (1, 2, 3)]
            else:
                with ThreadPoolExecutor(max_workers=protocol["max_workers"]) as pool:
                    futures = [pool.submit(one_rep, rep) for rep in (1, 2, 3)]
                    reports = [future.result() for future in futures]
        comparison = _comparison(Path(policy["calibration"]["baseline_campaign"]), reports, protocol)
        _write(campaign / "comparison.json", comparison)
        result = {"schema": "self_evolving_campaign/1", "status": "completed",
                  "campaign_dir": str(campaign), "started_utc": started, "ended_utc": _now(),
                  "elapsed_seconds": monotonic() - tick,
                  "run_ids": [report["run_id"] for report in reports],
                  "total_task_tokens": sum(report["total_cost"] for report in reports),
                  "remote_calls": sum(task["routing"]["remote_calls"] for report in reports for task in report["tasks"]),
                  "remote_tokens": sum(task["routing"]["remote_tokens"] for report in reports for task in report["tasks"]),
                  "preflight_tokens": preflight["total_reported_tokens"],
                  "total_research_tokens": sum(report["total_cost"] for report in reports) + preflight["total_reported_tokens"],
                  "quality_gate_passed": comparison["quality_gate_passed"],
                  "comparison_path": str(campaign / "comparison.json")}
        _write(campaign / "summary.json", result)
        return result
    except Exception as exc:
        _write(campaign / "summary.json", {"schema": "self_evolving_campaign/1",
                                          "status": "failed", "campaign_dir": str(campaign),
                                          "started_utc": started, "ended_utc": _now(),
                                          "elapsed_seconds": monotonic() - tick,
                                          "error": str(exc)[:500]})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prompt-key", action="store_true",
                        help="Ask for the gateway key in this visible terminal, ignoring inherited keys")
    parser.add_argument("--campaign", type=Path, help="Prepared campaign directory to run")
    parser.add_argument("--baseline-campaign", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    parsed = urlsplit(args.base_url)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        parser.error("--base-url must be HTTPS without credentials or query parameters")
    if args.campaign and args.prepare_only:
        parser.error("--campaign and --prepare-only cannot be combined")
    if args.prompt_key and args.prepare_only:
        parser.error("--prompt-key is only for a live run")
    campaign = args.campaign or prepare_campaign(args.baseline_campaign, args.base_url, args.workers)
    if args.prepare_only:
        print(json.dumps(json.loads((campaign / "summary.json").read_text()), ensure_ascii=False))
        return 0
    if args.prompt_key and not sys.stdin.isatty():
        parser.error("--prompt-key requires a visible interactive terminal")
    if args.prompt_key or (not os.environ.get("OPENAI_API_KEY") and sys.stdin.isatty()):
        os.environ["OPENAI_API_KEY"] = getpass("新网关 API key（输入不回显）：")
    print(f"Starting self-evolving practice campaign: {campaign}", flush=True)
    print(json.dumps(run_campaign(campaign), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
