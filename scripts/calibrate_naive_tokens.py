"""Measure naive-agent token use on every bank task without opening holdout.

This is a development calibration, not a formal validation/exam run. Each
attempt gets a fresh ability library and only the task's dev scoring inputs.
The command writes per-attempt evidence under ignored runs/. Failed tasks can
be selected for a new campaign; they are never silently retried.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from threading import Lock
from typing import Any, Callable, Iterator

import harness
from evaluation.manifest import create_manifest
from evaluation.quote import make_quote_callback
from ledger import Ledger, TOKEN_BUCKETS
from library import Library, selected_settings
from research_agent.model import ModelClient
from research_agent.replay import write_replay_context
from research_agent.roles import RoleAgents
from research_agent.runner import TaskRunner

from .run_token_pilot import (BASE_URL, BUDGET_TOKENS, MAX_OUTPUT_TOKENS,
                              MODEL, PROJECT, RESERVE_TOKENS, _sha, _stamp, _write)

CONFIG_NAME = "naive_token_pilot.json"
PRICE_NAME = "prices_token_pilot.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class DevOnlyHarness:
    """An explicit guard against ever requesting a holdout score."""

    def __init__(self, base: Any, lock: Lock | None = None):
        self.base = base
        self.lock = lock

    def _serialized(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.lock is None:
            return getattr(self.base, method)(*args, **kwargs)
        with self.lock:
            return getattr(self.base, method)(*args, **kwargs)

    def score(self, task_id: str, node_dir: Path, split: str, **kwargs: Any) -> dict:
        if split != "dev":
            raise PermissionError("Naive cost calibration may only score dev")
        return self._serialized("score", task_id, node_dir, split, **kwargs)

    def score_holdout(self, *args: Any, **kwargs: Any) -> dict:
        raise PermissionError("Naive cost calibration may not open holdout")

    def run_script(self, *args: Any, **kwargs: Any) -> Any:
        return self._serialized("run_script", *args, **kwargs)

    def profile_code(self, *args: Any, **kwargs: Any) -> Any:
        return self._serialized("profile_code", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@contextmanager
def _project_override(project: Path) -> Iterator[None]:
    keys = {"RA_PROJECT_ROOT": str(project), "RA_TASKS_ROOT": str(project / "tasks")}
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _bank_ids(source: Path, selected: tuple[str, ...] | None) -> tuple[list[dict], tuple[str, ...]]:
    bank = json.loads((source / "tasks/bank.json").read_text(encoding="utf-8"))
    if bank.get("schema") != "bank/1" or not isinstance(bank.get("tasks"), list):
        raise ValueError("Source bank must be bank/1")
    entries = {entry["task_id"]: entry for entry in bank["tasks"]}
    if len(entries) != len(bank["tasks"]):
        raise ValueError("Source bank has duplicate task IDs")
    chosen = selected or tuple(entries)
    if not chosen or len(set(chosen)) != len(chosen) or any(item not in entries for item in chosen):
        raise ValueError("Select unique task IDs from the source bank")
    return [entries[item] for item in chosen], chosen


def prepare_project(source: Path, campaign: Path, *,
                    task_ids: tuple[str, ...] | None = None, repetitions: int = 3,
                    max_workers: int = 1) -> Path:
    """Copy only dev scoring inputs; never copy tabular holdout.csv."""
    if type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError("Repetitions must be 1, 2, or 3")
    if type(max_workers) is not int or not 1 <= max_workers <= 3:
        raise ValueError("Workers must be 1, 2, or 3")
    source = Path(source).resolve()
    entries, chosen = _bank_ids(source, task_ids)
    project = campaign / "project"
    project.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "config", project / "config")
    (project / "tasks").mkdir()
    bank = json.loads((source / "tasks/bank.json").read_text(encoding="utf-8"))
    bank["tasks"] = entries
    bank["frozen"] = False
    _write(project / "tasks/bank.json", bank)
    source_hashes: dict[str, str] = {}
    copied_hashes: dict[str, str] = {}
    source_hashes["tasks/bank.json"] = _sha(source / "tasks/bank.json")
    copied_hashes["tasks/bank.json"] = _sha(project / "tasks/bank.json")
    for task_id in chosen:
        original = source / "tasks" / task_id
        copied = project / "tasks" / task_id
        copied.mkdir()
        shutil.copytree(original / "workspace", copied / "workspace")
        hidden = copied / "hidden"
        hidden.mkdir()
        task = json.loads((original / "task.json").read_text(encoding="utf-8"))
        if task.get("task_id") != task_id or task.get("scenario") not in {"speedup", "tabular"}:
            raise ValueError(f"Invalid task package: {task_id}")
        dev_source = original / "hidden" / ("dev.csv" if task["scenario"] == "tabular" else "reference.py")
        if not dev_source.is_file():
            raise FileNotFoundError(dev_source)
        shutil.copyfile(dev_source, hidden / dev_source.name)
        source_hashes[str((original / "task.json").relative_to(source))] = _sha(original / "task.json")
        source_hashes[str(dev_source.relative_to(source))] = _sha(dev_source)
        task["budget"]["money"] = BUDGET_TOKENS
        _write(copied / "task.json", task)
        for path in sorted(copied.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(project))
                copied_hashes[relative] = _sha(path)
                if path.is_relative_to(copied / "workspace"):
                    source_hashes[relative] = _sha(original / path.relative_to(copied))
        if (hidden / "holdout.csv").exists():
            raise AssertionError("Holdout data must not enter a calibration copy")
    for relative in ("config/naive_token_pilot.json", "config/prompts/naive_token_pilot.md",
                     "config/prices_token_pilot.json", "config/hard_caps.json",
                     "config/triggers.json"):
        copied_hashes[relative] = _sha(project / relative)
        source_hashes[relative] = _sha(source / relative)
    prices = json.loads((project / "config" / PRICE_NAME).read_text(encoding="utf-8"))
    if (prices.get("currency") != "TOKENS" or not prices.get("test_only")
            or prices.get("unit_tokens") != 1
            or set(prices.get("models", {}).get(MODEL, {}).values()) != {1}):
        raise ValueError("Naive calibration needs TEST-only one-unit-per-token prices")
    config = json.loads((project / "config" / CONFIG_NAME).read_text(encoding="utf-8"))
    if (config.get("history_mode") != "full"
            or set(config.get("effort", {}).values()) != {"high"}
            or len(set(config.get("prompts", {}).values())) != 1
            or config.get("query_k") != 0):
        raise ValueError("Naive config must have one high-effort prompt and no retrieval")
    _write(campaign / "protocol.json", {
        "schema": "naive_token_calibration_protocol/1",
        "label": "DEV ONLY; TEST TOKEN UNITS; NO HOLDOUT OR MONEY",
        "source_root": str(source), "model": MODEL, "base_url": BASE_URL,
        "bank_version": bank["version"],
        "tasks": list(chosen), "repetitions": repetitions,
        "max_workers": max_workers, "sandbox_calls_serialized": max_workers > 1,
        "budget_tokens_per_task": BUDGET_TOKENS,
        "reserve_tokens_per_call": RESERVE_TOKENS,
        "max_output_tokens_requested": MAX_OUTPUT_TOKENS,
        "source_sha256": source_hashes, "copied_sha256": copied_hashes,
    })
    return project


def _verify_inputs(project: Path, protocol: dict) -> None:
    for relative, expected in protocol["copied_sha256"].items():
        path = (project / relative).resolve()
        if not path.is_relative_to(project.resolve()) or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Calibration input changed: {relative}")


def _run_one(project: Path, task_id: str, rep: int, code_commit: str,
             *, run_dir: Path | None = None,
             roles_factory: Callable[..., Any] | None = None,
             base_harness: Any = harness) -> dict:
    """Run A on dev only and return metered cost, never a holdout score."""
    from openai import OpenAI

    run_dir = run_dir or project / "runs" / f"run-{_stamp()}-{secrets.token_hex(2)}"
    config_path = project / "config" / CONFIG_NAME
    prices_path = project / "config" / PRICE_NAME
    library_root = run_dir / "library"
    bank = json.loads((project / "tasks/bank.json").read_text(encoding="utf-8"))
    create_manifest(project_root=project, run_dir=run_dir, phase="dev",
                    config="naive", rep=rep, bank_version=bank["version"],
                    tasks=[task_id], library_root=library_root,
                    library_start_version="lib-0", library_frozen=False, model=MODEL,
                    agent_config_path=config_path, prices_path=prices_path,
                    code_commit=code_commit)
    scored = (base_harness if isinstance(base_harness, DevOnlyHarness)
              else DevOnlyHarness(base_harness))
    baseline_dir = run_dir / "tasks" / task_id / "nodes/n0000"
    baseline_dir.mkdir(parents=True)
    shutil.copyfile(project / "tasks" / task_id / "workspace/original.py",
                    baseline_dir / "solution.py")
    with _project_override(project):
        baseline = scored.score(task_id, baseline_dir, "dev")
        if (baseline.get("schema") != "score/1" or not baseline.get("ok")
                or not baseline.get("correct") or baseline.get("split") != "dev"):
            raise RuntimeError("Naive calibration baseline dev score failed")
        ledger = Ledger.open(run_dir, task_id)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hard_caps = json.loads((project / "config/hard_caps.json").read_text(encoding="utf-8"))
        with selected_settings(config_path, project / "config/triggers.json"):
            library = Library.open(library_root)
            if roles_factory is None:
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL,
                                timeout=180.0, max_retries=0)
                model = ModelClient(MODEL, ledger, client=client,
                                    max_output_tokens=MAX_OUTPUT_TOKENS,
                                    budget_limit=BUDGET_TOKENS, reserve_cost=RESERVE_TOKENS,
                                    audit_dir=run_dir / "tasks" / task_id / "model_responses")
                roles = RoleAgents(model, config, project_root=project)
            else:
                roles = roles_factory(ledger=ledger, run_dir=run_dir, task_id=task_id)
            outcome = TaskRunner(
                run_dir=run_dir, task_id=task_id, harness=scored, ledger=ledger,
                roles=roles, library=library, baseline_score=baseline, config=config,
                hard_caps=hard_caps, on_initial_plan=make_quote_callback(ledger),
            ).run()
        sources = {
            "baseline": baseline_dir / "score.json", "agent_config": config_path,
            "prices": prices_path, "hard_caps": project / "config/hard_caps.json",
            "triggers": project / "config/triggers.json",
        }
        for role, relative in config["prompts"].items():
            sources[f"prompt_{role}"] = project / relative
        opening = run_dir / "tasks" / task_id / "opening_plan.json"
        if opening.is_file():
            sources["opening_plan"] = opening
        write_replay_context(run_dir=run_dir, task_id=task_id,
                             library_root=library_root, source_files=sources,
                             model=MODEL, max_output_tokens=MAX_OUTPUT_TOKENS)
    events = ledger._events()
    token_buckets = {bucket: sum(event["tokens"][bucket] for event in events
                                 if event["kind"] == "llm_call") for bucket in TOKEN_BUCKETS}
    total = sum(token_buckets.values())
    if total != ledger.spent() or total != outcome.spent:
        raise ValueError("Calibration token ledger does not reconcile")
    if (run_dir / "tasks" / task_id / "holdout_attempt.json").exists():
        raise AssertionError("Calibration unexpectedly opened holdout")
    return {"status": "completed" if outcome.reason != "error" and total <= BUDGET_TOKENS
            else "incomplete", "budget_respected": total <= BUDGET_TOKENS,
            "task_id": task_id, "rep": rep, "run_dir": str(run_dir),
            "stop_reason": outcome.reason, "dev_score": outcome.dev_score,
            "best_node": outcome.best_node, "total_tokens": total,
            "token_buckets": token_buckets, "model_calls": sum(
                event["kind"] == "llm_call" for event in events),
            "ledger_path": str(ledger.path), "manifest_path": str(run_dir / "manifest.json"),
            "replay_context_path": str(run_dir / "tasks" / task_id / "replay_context.json")}


def _attempt_one(project: Path, task_id: str, rep: int, code_commit: str,
                 run_dir: Path, scored: DevOnlyHarness) -> dict:
    try:
        return _run_one(project, task_id, rep, code_commit, run_dir=run_dir,
                        base_harness=scored)
    except Exception as exc:
        message = str(exc)
        secret = os.environ.get("OPENAI_API_KEY")
        if secret:
            message = message.replace(secret, "[REDACTED]")
        spent = None
        if (run_dir / "manifest.json").is_file():
            try:
                spent = Ledger.open(run_dir, task_id).spent()
            except (ValueError, OSError):
                pass
        return {"status": "failed", "error": message, "recorded_tokens": spent}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", action="append",
                           help="Bank task ID; repeat to select tasks")
    selection.add_argument("--scenario", choices=("speedup", "tabular"),
                           help="Select all 12 tasks of one scenario")
    parser.add_argument("--repetitions", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2, 3),
                        help="Independent repetitions per task; sandbox calls stay serialized")
    args = parser.parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be injected in the process environment")
    if os.environ.get("RA_EVAL") == "1":
        raise SystemExit("RA_EVAL must be absent during dev calibration")
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                               cwd=PROJECT, text=True).strip():
        raise SystemExit("Commit source changes before naive token calibration")
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=PROJECT, text=True).strip()
    campaign = PROJECT / "runs" / f"naive-token-calibration-{_stamp()}-{secrets.token_hex(2)}"
    selected = tuple(args.task) if args.task else None
    if args.scenario:
        entries, _ = _bank_ids(PROJECT, None)
        prefix = "spd-" if args.scenario == "speedup" else "tab-"
        selected = tuple(entry["task_id"] for entry in entries
                         if entry["task_id"].startswith(prefix))
        if len(selected) != 12:
            raise SystemExit(f"Expected 12 {args.scenario} tasks, found {len(selected)}")
    project = prepare_project(PROJECT, campaign, task_ids=selected,
                              repetitions=args.repetitions, max_workers=args.workers)
    protocol = json.loads((campaign / "protocol.json").read_text(encoding="utf-8"))
    protocol["code_commit"] = code_commit
    _write(campaign / "protocol.json", protocol)
    summary = {"schema": "naive_token_calibration_summary/1", "label": protocol["label"],
               "campaign_dir": str(campaign), "started_utc": _utc_now(),
               "attempts": [], "ended_utc": None}
    _write(campaign / "summary.json", summary)
    scored = DevOnlyHarness(harness, Lock() if args.workers > 1 else None)
    with _project_override(project):
        for task_id in protocol["tasks"]:
            _verify_inputs(project, protocol)
            jobs = []
            for rep in range(1, protocol["repetitions"] + 1):
                for _ in range(100):
                    run_dir = project / "runs" / f"run-{_stamp()}-{secrets.token_hex(2)}"
                    try:
                        run_dir.mkdir(parents=True, exist_ok=False)
                    except FileExistsError:
                        continue
                    break
                else:
                    raise RuntimeError("Could not reserve an unused calibration run ID")
                attempt = {"task_id": task_id, "rep": rep, "run_dir": str(run_dir),
                           "status": "started"}
                jobs.append((rep, run_dir, attempt))
                summary["attempts"].append(attempt)
            _write(campaign / "summary.json", summary)
            if args.workers == 1:
                for rep, run_dir, attempt in jobs:
                    attempt.update(_attempt_one(project, task_id, rep, code_commit, run_dir, scored))
                    _write(campaign / "summary.json", summary)
                    print(json.dumps(attempt, ensure_ascii=False), flush=True)
            else:
                with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
                    futures = {pool.submit(_attempt_one, project, task_id, rep,
                                           code_commit, run_dir, scored): attempt
                               for rep, run_dir, attempt in jobs}
                    for future in as_completed(futures):
                        attempt = futures[future]
                        attempt.update(future.result())
                        _write(campaign / "summary.json", summary)
                        print(json.dumps(attempt, ensure_ascii=False), flush=True)
    summary["ended_utc"] = _utc_now()
    _write(campaign / "summary.json", summary)
    print(f"Calibration evidence: {campaign / 'summary.json'}", flush=True)
    return 0 if all(row["status"] == "completed" for row in summary["attempts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
