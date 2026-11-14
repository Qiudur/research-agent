"""Dev A/B experiment: run several agent configs on a few practice tasks, paired by repetition.

This is dev research, not a formal exam: practice tasks only, copied into an
ignored runs/ campaign directory, TEST price table (one unit = one reported
token), 40,000 tokens per task with the same per-call reserve and output cap as
the token studies. Every (configuration, repetition) is one run that works
through the tasks in order and ends with the single holdout of each task. The
first --configs entry is the baseline the analysis compares the others with.

OPENAI_API_KEY and the gateway URL (OPENAI_BASE_URL, else the repository's
default gateway) are read only from the environment and are never printed or
written to disk. --dry-run replaces the model and the sandboxed scorers by
deterministic synthetic stand-ins so the whole pipeline can be exercised
offline; its numbers mean nothing.

    python -m scripts.run_ab_experiment --configs baseline=config/agent_token_pilot_v2.json stoploss=config/agent_ab_stoploss.json --reps 3 --workers 3
    python -m scripts.analyze_ab runs/ab-<stamp>-<id>
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
from threading import Lock, local
from time import monotonic
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import harness
from evaluation.batch import _SerializedHarness, _agent_signature, _run_path
from evaluation.finalize import finish_run, start_run
from evaluation.manifest import create_manifest
from evaluation.orchestrate import execute_task
from evaluation.report import run_report
from library import Library
from ledger import Ledger
from research_agent.model import ModelClient
from research_agent.roles import RoleAgents

from .run_token_pilot import (BASE_URL, BUDGET_TOKENS, MAX_OUTPUT_TOKENS, MODEL, PROJECT,
                              RESERVE_TOKENS, _stamp, _write)
from .run_token_self_research import (PRICE_NAME, _project_override, _timing, _utc_now,
                                      prepare_project)

DEFAULT_TASKS = ("spd-distance-matrix", "tab-german-credit", "tab-online-shoppers")
MAX_REPS = 3  # run manifests only allow repetitions 1-3
MAX_WORKERS = 4
LABEL = re.compile(r"^[a-z][a-z0-9_-]*$")
STUDY_LABEL = "DEV A/B RESEARCH - NOT AN EXAM; TEST UNIT = ONE REPORTED TOKEN; NOT MONEY"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redact(text: str) -> str:
    """Never let the API key reach a summary through an exception message."""
    key = os.environ.get("OPENAI_API_KEY")
    return text.replace(key, "[redacted]")[:500] if key else text[:500]


# --------------------------------------------------------------------------
# Offline stand-ins for --dry-run.

class DryRunResponses:
    """Scripted Responses API: a plan of three ideas, then two, then an empty plan."""

    PLAN_SIZES = (3, 2)

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.calls: Counter = Counter()
        self.ideas = 0
        self.codes = 0

    def create(self, **kwargs) -> Any:
        name = kwargs["text"]["format"]["name"]
        self.calls[name] += 1
        if name == "planner_plan":
            round_index = self.calls[name] - 1
            size = self.PLAN_SIZES[round_index] if round_index < len(self.PLAN_SIZES) else 0
            hypotheses = []
            for _ in range(size):
                self.ideas += 1
                hypotheses.append({"hypothesis": f"Dry-run idea {self.ideas} for {self.task_id}",
                                   "rationale": "Synthetic", "cites": [],
                                   "estimate": {"gain": 0.2, "cost": 3000.0, "seconds": 5.0}})
            data = {"hypotheses": hypotheses, "budget_allocation": [], "expected_progress": [],
                    "predicted": {"cost": 20000.0, "seconds": 60.0, "score": 0.8}}
            usage = (2400, 500)
        elif name in ("executor_code", "executor_fix"):
            self.codes += 1
            data = {"code": f"def solve(*args, **kwargs):\n    return {self.codes}\n",
                    "notes": "dry run"}
            usage = (1800, 700)
        elif name == "diagnoser_diagnose":
            data = {"hypothesis": "", "rationale": "dry run", "advice": ""}
            usage = (900, 200)
        elif name == "planner_review":
            data = {"decision": "continue", "budget_moves": [], "note": "dry run"}
            usage = (1200, 200)
        elif name == "summarizer_summarize":
            data = {"drafts": [], "narrowings": []}
            usage = (1500, 300)
        else:
            raise AssertionError(f"Unexpected dry-run purpose: {name}")
        total = sum(self.calls.values())
        return SimpleNamespace(
            id=f"dry-{self.task_id}-{total}", status="completed", output=[],
            output_text=json.dumps(data),
            usage={"input_tokens": usage[0] + total, "output_tokens": usage[1],
                   "input_tokens_details": {"cached_tokens": usage[0] // 4},
                   "output_tokens_details": {"reasoning_tokens": usage[1] // 5}})


class DryRunHarness:
    """Scorer stand-in: package metadata is real, scores are a hash of the code."""

    def load_task(self, task_id: str) -> dict:
        import harness as task_packages  # the real loader, even if this module's name is patched

        return task_packages.load_task(task_id)

    def run_script(self, task_id: str, node_dir: Path, argv: list, limits=None) -> dict:
        try:
            ast.parse((Path(node_dir) / "solution.py").read_text(encoding="utf-8"))
            exit_code, error = 0, ""
        except SyntaxError as exc:
            exit_code, error = 1, str(exc)
        return {"exit_code": exit_code, "timed_out": False, "oom": False, "seconds": 0.0,
                "peak_mb": 0, "stdout_tail": "", "stderr_tail": error}

    def score(self, task_id: str, node_dir: Path, split: str, seed: int | None = None) -> dict:
        if split not in ("dev", "holdout"):
            raise ValueError("Split must be dev or holdout")
        if split == "holdout" and os.environ.get("RA_EVAL") != "1":
            raise PermissionError("RA_EVAL=1 is required for holdout")
        node = Path(node_dir)
        code = (node / "solution.py").read_text(encoding="utf-8")
        digest = hashlib.sha256(f"{task_id}|{split}|{code}".encode()).hexdigest()
        value = 0.1 if node.name == "n0000" else 0.1 + 1.1 * int(digest[:8], 16) / 0xFFFFFFFF
        raw = 0.7 + 0.1 * value if task_id.startswith("tab-") else 1.0 + 2.0 * value
        return {"schema": "score/1", "ok": True, "split": split, "score": value, "raw": raw,
                "spread": 0.01, "correct": True, "cheat_flags": [], "error": None,
                "seconds": {"run": 0.0, "queue": 0.0}, "seed": int(digest[8:16], 16)}


# --------------------------------------------------------------------------
# Campaign preparation.

def parse_config_specs(project_root: Path, specs: list[str]) -> dict[str, Path]:
    """label=path pairs; each config is an agent/1 file inside the project with distinct content.

    Everything is checked before the campaign directory is created.
    """
    configs: dict[str, Path] = {}
    signatures: dict[str, str] = {}
    for spec in specs:
        label, separator, raw = spec.partition("=")
        if not separator or not LABEL.fullmatch(label):
            raise SystemExit(f"--configs needs label=path with a label like 'baseline': {spec!r}")
        if label in configs:
            raise SystemExit(f"Duplicate configuration label: {label}")
        path = Path(raw)
        path = (path if path.is_absolute() else project_root / path).resolve()
        if not path.is_relative_to(project_root) or not path.is_file():
            raise SystemExit(f"Configuration for {label} must be an existing file inside the project")
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("schema") != "agent/1" or not isinstance(config.get("prompts"), dict):
            raise SystemExit(f"{label} is not an agent/1 configuration")
        for relative in config["prompts"].values():
            prompt = (project_root / relative).resolve()
            if not prompt.is_relative_to(project_root) or not prompt.is_file():
                raise SystemExit(f"{label} points at a missing or external prompt: {relative}")
        try:
            signature = _agent_signature(project_root, config)
        except ValueError as exc:
            raise SystemExit(f"{label}: {exc}") from None
        if signature in signatures:
            raise SystemExit(f"{label} has the same settings and prompts as {signatures[signature]}; "
                             "an A/B comparison needs different configurations")
        signatures[signature] = label
        configs[label] = path
    return configs


def install_configs(project_root: Path, project: Path, configs: dict[str, Path]) -> dict[str, dict]:
    """Copy each config (and any prompt outside the copied config/) into the campaign project."""
    installed = {}
    for label, source in configs.items():
        config = json.loads(source.read_text(encoding="utf-8"))
        for relative in config["prompts"].values():
            prompt = (project_root / relative).resolve()
            target = project / prompt.relative_to(project_root)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(prompt, target)
        copied = project / "config" / f"ab_{label}.json"
        shutil.copyfile(source, copied)
        installed[label] = {"source": str(source.relative_to(project_root)),
                            "sha256": _sha(copied), "path": str(copied.relative_to(project))}
    return installed


def check_tasks(project_root: Path, task_ids: list[str]) -> None:
    bank = json.loads((project_root / "tasks/bank.json").read_text(encoding="utf-8"))
    tiers = {entry["task_id"]: entry["tier"] for entry in bank["tasks"]}
    for task_id in task_ids:
        if task_id not in tiers:
            raise SystemExit(f"Unknown task: {task_id}")
        if tiers[task_id] != "practice":
            raise SystemExit(f"{task_id} is a {tiers[task_id]} task; A/B research uses practice tasks only")
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("Tasks must be distinct")


# --------------------------------------------------------------------------
# One run = one (configuration, repetition).

def execute_run(*, project: Path, task_ids: list[str], label: str, rep: int, installed: dict,
                code_commit: str, bank_version: str, dry_run: bool,
                guarded: Any, transport: Any) -> dict:
    record = {"label": label, "rep": rep, "tasks": {}, "status": "running"}
    for _ in range(100):
        run_dir = _run_path(project)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Could not reserve an unused run ID")
    record["run_id"] = run_dir.name
    library_root = run_dir / "library"
    library = Library.open(library_root)
    create_manifest(
        project_root=project, run_dir=run_dir, phase="dev", config="default_lib", rep=rep,
        bank_version=bank_version, tasks=task_ids, library_root=library_root,
        library_start_version=library.version, library_frozen=False, model=MODEL,
        agent_config_path=project / installed[label]["path"],
        prices_path=project / "config" / PRICE_NAME, code_commit=code_commit)
    _write(run_dir / "ab_run.json", {
        "schema": "ab_run/1", "label": label, "rep": rep, "tasks": task_ids,
        "config": installed[label], "dry_run": dry_run})
    started = monotonic()
    start_run(run_dir)
    config = json.loads((project / installed[label]["path"]).read_text(encoding="utf-8"))
    for task_id in task_ids:
        try:
            ledger = Ledger.open(run_dir, task_id)
            client = SimpleNamespace(responses=DryRunResponses(task_id)) if dry_run else transport()
            model = ModelClient(MODEL, ledger, client=client, max_output_tokens=MAX_OUTPUT_TOKENS,
                                budget_limit=BUDGET_TOKENS, reserve_cost=RESERVE_TOKENS,
                                audit_dir=run_dir / "tasks" / task_id / "model_responses")
            roles = RoleAgents(model, config, project_root=project)
            execute_task(run_dir=run_dir, task_id=task_id, harness=guarded,
                         baseline_solution=project / "tasks" / task_id / "workspace/original.py",
                         roles=roles, test_only_budget=BUDGET_TOKENS,
                         max_output_tokens=MAX_OUTPUT_TOKENS)
            record["tasks"][task_id] = {"status": "completed"}
        except Exception as exc:  # keep the other tasks and runs going; evidence stays on disk
            record["tasks"][task_id] = {"status": "failed", "error": _redact(str(exc))}
    if all(item["status"] == "completed" for item in record["tasks"].values()):
        finish_run(run_dir)
        run_report(run_dir)
        record["status"] = "completed"
    else:
        record["status"] = "partial"
    record["seconds"] = monotonic() - started
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS),
                        help="practice task ids, run in this order")
    parser.add_argument("--configs", nargs="+", action="extend", required=True, metavar="LABEL=PATH",
                        help="agent/1 configs inside the project, several after one flag or the "
                             "flag repeated; the first is the baseline")
    parser.add_argument("--reps", type=int, choices=range(1, MAX_REPS + 1), default=3)
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=3,
                        help="runs in flight at once; sandbox scoring stays serialized")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--out", type=Path, help="new campaign directory under <project-root>/runs/")
    parser.add_argument("--dry-run", action="store_true",
                        help="synthetic model and scorers: no key, network or sandbox needed")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    task_ids = list(args.tasks)
    check_tasks(root, task_ids)
    configs = parse_config_specs(root, args.configs)
    out = (args.out or root / "runs" / f"ab-{_stamp()}-{secrets.token_hex(2)}")
    out = (out if out.is_absolute() else Path.cwd() / out).resolve()
    if not out.is_relative_to(root / "runs") or out.exists():
        raise SystemExit("--out must be a new directory under <project-root>/runs/")
    base_url = None
    if not args.dry_run:
        base_url = os.environ.get("OPENAI_BASE_URL") or BASE_URL  # read now, never recorded
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise SystemExit("The gateway URL (OPENAI_BASE_URL) must be an HTTPS URL without "
                             "credentials or query parameters")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY must be injected in the process environment")
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                            cwd=PROJECT, capture_output=True, text=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True)
    if commit.returncode != 0:
        raise SystemExit("The code checkout must be a Git repository (its HEAD is recorded)")
    code_commit = commit.stdout.strip()

    project = prepare_project(root, out, task_ids=tuple(task_ids), proposal_rounds=0,
                              max_workers=args.workers)
    installed = install_configs(root, project, configs)
    bank_version = json.loads((project / "tasks/bank.json").read_text(encoding="utf-8"))["version"]
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    for key in ("proposal_rounds", "default_config_sha256", "repetitions_per_candidate"):
        protocol.pop(key, None)
    protocol.update({
        "schema": "ab_protocol/1", "label": STUDY_LABEL, "study_type": "ab_experiment",
        "dry_run": args.dry_run, "tasks": task_ids, "reps": args.reps, "workers": args.workers,
        "configs": installed, "baseline": next(iter(configs)), "code_commit": code_commit,
        "worktree_clean_at_start": status.returncode == 0 and not status.stdout.strip(),
        "gateway": ("none (dry run)" if args.dry_run else
                    "OPENAI_BASE_URL" if os.environ.get("OPENAI_BASE_URL") else "repository default"),
        "reserve_tokens_per_call": RESERVE_TOKENS,
        "budget_tokens_per_task": BUDGET_TOKENS, "max_output_tokens_requested": MAX_OUTPUT_TOKENS})
    _write(out / "protocol.json", protocol)
    summary = {"schema": "ab_summary/1", "label": STUDY_LABEL, "campaign_dir": str(out),
               "runs": [], "started_utc": _utc_now()}
    _write(out / "summary.json", summary)

    state, lock = local(), Lock()
    serialized = _SerializedHarness(DryRunHarness() if args.dry_run else harness, Lock())

    def transport():
        if not hasattr(state, "client"):
            from openai import OpenAI
            state.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url,
                                  timeout=180.0, max_retries=0)
        return state.client

    def job(label: str, rep: int) -> dict:
        print(f"Starting {label} rep {rep}", flush=True)
        try:
            record = execute_run(project=project, task_ids=task_ids, label=label, rep=rep,
                                 installed=installed, code_commit=code_commit,
                                 bank_version=bank_version, dry_run=args.dry_run,
                                 guarded=serialized, transport=transport)
        except Exception as exc:
            record = {"label": label, "rep": rep, "status": "failed", "error": _redact(str(exc))}
        with lock:
            summary["runs"].append(record)
            _write(out / "summary.json", summary)
        print(f"Finished {label} rep {rep}: {record['status']}", flush=True)
        return record

    started_tick = monotonic()
    started_utc = summary["started_utc"]
    jobs = [(label, rep) for rep in range(1, args.reps + 1) for label in configs]
    with _project_override(project), ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda item: job(*item), jobs))
    summary.update(_timing(started_utc, started_tick))
    _write(out / "summary.json", summary)
    failed = [r for r in records if r["status"] != "completed"]
    print(f"Campaign: {out}")
    print(f"Runs completed: {len(records) - len(failed)}/{len(records)}")
    print(f"Analyze with: python -m scripts.analyze_ab {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
