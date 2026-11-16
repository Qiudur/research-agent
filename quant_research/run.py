"""Audited development-only quant Agent run; never scores sealed holdout."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

from ledger import Ledger
from library import Library
from research_agent.artifacts import node_directory
from research_agent.io import atomic_json, read_json
from research_agent.model import ModelClient
from research_agent.replay import selected_library_settings, sha256_file
from research_agent.roles import RoleAgents
from research_agent.runner import TaskRunner
from research_agent.tree import ExperimentTree

from .data import load_dataset
from .demo_data import write_demo_dataset
from .harness import QuantHarness
from .provenance import capture_inputs, verify_inputs
from .splits import make_splits

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "quant-demo"
BASE_URL = "https://ai-gateway.theunifiedcare.com/v1"
ORIGINAL = (
    "def compute_signal(history):\n"
    "    return {symbol: (prices[-1] / prices[-6] - 1.0) if len(prices) > 5 else 0.0\n"
    "            for symbol, prices in history.items()}\n"
)
SCRIPTED = (
    "def compute_signal(history):\n"
    "    return {symbol: (prices[-1] / prices[-11] - 1.0) if len(prices) > 10 else 0.0\n"
    "            for symbol, prices in history.items()}\n"
)

SUCCESS_REASONS = frozenset({"planner", "budget_cap", "time", "ceiling", "stop_loss"})


class QuantRunFailed(RuntimeError):
    """A run retained its evidence but is not eligible for sealed evaluation."""


class DevAdmissionError(RuntimeError):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(kind)


class ScriptedRoles:
    """One auditable, key-free integration fixture, never a model result."""

    def __init__(self) -> None:
        self.plans = 0

    def plan(self, payload: dict, *, node_id: str | None = None) -> dict:
        self.plans += 1
        hypothesis = ([{
            "hypothesis": "Compare ten-session trailing momentum to the five-session original",
            "rationale": "A longer historical window may reduce short-term noise",
            "estimate": {"gain": 0.001, "cost": 1.0, "seconds": 5.0},
            "cites": [],
        }] if self.plans == 1 else [])
        return {"hypotheses": hypothesis,
                "predicted": {"cost": 1.0, "seconds": 5.0,
                              "score": float(payload["best"]["score"]) + 0.001},
                "budget_allocation": [], "expected_progress": []}

    def execute(self, payload: dict, *, fix: bool = False,
                node_id: str | None = None) -> dict:
        return {"code": SCRIPTED, "notes": "Deterministic fixture, no model call"}

    def diagnose(self, payload: dict, *, node_id: str | None = None) -> dict:
        return {"hypothesis": "", "rationale": "", "advice": ""}

    def review(self, payload: dict, *, node_id: str | None = None) -> dict:
        return {"decision": "continue", "budget_moves": [], "note": ""}

    def summarize(self, payload: dict, *, node_id: str | None = None) -> dict:
        return {"drafts": [], "narrowings": []}


def _descriptor(project_root: Path, path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(project_root):
        raise ValueError("Sealed input must be a project-local file")
    return {"path": str(path.relative_to(project_root)), "sha256": sha256_file(path)}


def _snapshot_dataset(source: Path, target: Path) -> Path:
    """Copy validated source bytes into the private run area."""
    load_dataset(source)
    source = Path(source).resolve(strict=True)
    data = read_json(source, "quant_dataset/1")
    target.mkdir(parents=True, exist_ok=False)
    source_csv = (source.parent / data["csv"]).resolve(strict=True)
    shutil.copyfile(source_csv, target / "prices.csv")
    data["csv"] = "prices.csv"
    manifest = target / "manifest.json"
    atomic_json(manifest, data)
    load_dataset(manifest)
    return manifest


def _split_dates(panel: Any, boundaries: tuple[date, date, date, date] | None) -> tuple[date, date, date, date]:
    if boundaries is None:
        if len(panel.dates) < 105:
            raise ValueError("Choose explicit split dates for datasets shorter than 105 sessions")
        boundaries = (panel.dates[50], panel.dates[55],
                      panel.dates[95], panel.dates[100])
    make_splits(panel.dates, train_end=boundaries[0], dev_start=boundaries[1],
                dev_end=boundaries[2], holdout_start=boundaries[3],
                label_horizon=2)
    return boundaries


def _code_commit(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=project_root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=project_root, text=True).strip())
    return commit, dirty


def _usage_summary(events: list[dict]) -> dict:
    calls = [event for event in events if event["kind"] == "llm_call"]
    complete = all(event.get("usage_status") == "reported" for event in calls)
    buckets = {key: sum(event["tokens"][key] for event in calls
                        if event.get("usage_status") == "reported")
               for key in ("input", "cached_input", "output", "reasoning")}
    cost = math.fsum(event["cost"] for event in calls
                     if event.get("usage_status") == "reported")
    return {
        "model_calls": len(calls), "usage_complete": complete,
        "unknown_usage_calls": sum(event.get("usage_status") != "reported" for event in calls),
        "reported_token_buckets": buckets,
        "total_token_buckets": buckets if complete else None,
        "reported_resource_cost": cost, "resource_cost": cost if complete else None,
        "currency": "TOKENS",
    }


def _failure_report(run_dir: Path, exc: BaseException) -> None:
    """Persist a failed terminal state without copying provider errors or secrets."""
    kind = (exc.kind if isinstance(exc, DevAdmissionError) else
            "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "execution_error")
    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, DevAdmissionError):
            kind = cause.kind
            break
        cause = cause.__cause__
    failure = {"kind": kind, "type": type(exc).__name__,
               "message": "Development run failed; sealed evaluation is blocked"}
    events: list[dict] = []
    # Use already-created ledger files even if a changed input prevents Ledger.open().
    if (run_dir / "manifest.json").is_file():
        task_ledger = Ledger(run_dir, TASK_ID, {}, {})
        run_ledger = Ledger(run_dir, None, {}, {})
        events = task_ledger._events()
        usage = _usage_summary(events)
        if not any(event["kind"] == "task_end" for event in events):
            task_ledger.emit("task_end", status="failed", best_node=None,
                             total_cost=task_ledger.spent(), failure=failure)
        if not any(event["kind"] == "run_end" and event.get("status") == "failed"
                   for event in run_ledger._events()):
            run_ledger.emit("run_end", status="failed", tasks_completed=0,
                            total_cost=task_ledger.spent(), failure=failure)
    else:
        usage = _usage_summary(events)
    (run_dir / "selection.json").unlink(missing_ok=True)
    atomic_json(run_dir / "dev_report.json", {
        "schema": "quant_dev_report/1", "run_id": run_dir.name, "status": "failed",
        "selection": None, "selected_node": None, "holdout_scored": False,
        "failure": failure, "evidence": {"task": "tasks/quant-demo/",
                                             "run_ledger": "ledger.jsonl"}, **usage,
    })


def _verify_run_inputs(project_root: Path, inputs: dict) -> None:
    try:
        verify_inputs(project_root, inputs)
    except (ValueError, OSError) as exc:
        raise DevAdmissionError("input_changed") from exc


class _CheckedRoles:
    """Detect changed prompts and settings before dispatching another model call."""

    def __init__(self, roles: Any, project_root: Path, inputs: dict) -> None:
        self.roles, self.project_root, self.inputs = roles, project_root, inputs

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.roles, name)
        if not callable(method):
            return method
        def checked(*args: Any, **kwargs: Any) -> Any:
            _verify_run_inputs(self.project_root, self.inputs)
            result = method(*args, **kwargs)
            _verify_run_inputs(self.project_root, self.inputs)
            return result
        return checked


def _preview(panel: Any, boundaries: tuple[date, date, date, date], *,
             mode: str, synthetic: bool, budget_tokens: int, wall_seconds: int,
             max_output_tokens: int, top_k: int, fee_bps: float, slippage_bps: float) -> dict:
    split = make_splits(panel.dates, train_end=boundaries[0], dev_start=boundaries[1],
                        dev_end=boundaries[2], holdout_start=boundaries[3], label_horizon=2)
    def window(dates: tuple[date, ...]) -> dict:
        return {"start": dates[0].isoformat(), "end": dates[-1].isoformat(),
                "sessions": len(dates)}
    return {"schema": "quant_preflight/1", "mode": mode,
            "data_type": "synthetic" if synthetic else "user_supplied",
            "dataset_sha256": panel.csv_sha256, "symbols": len(panel.symbols),
            "sessions": len(panel.dates), "train": window(split.train),
            "dev": window(split.dev), "holdout": window(split.holdout),
            "excluded_sessions": len(split.excluded), "label_horizon": 2,
            "budget_tokens": budget_tokens, "wall_seconds": wall_seconds,
            "max_output_tokens": max_output_tokens, "top_k": top_k,
            "fee_bps": fee_bps, "slippage_bps": slippage_bps,
            "will_call_model": mode == "qwen"}


def run_quant_dev(
    *, project_root: Path = ROOT, run_dir: Path, dataset_manifest: Path | None = None,
    boundaries: tuple[date, date, date, date] | None = None, mode: str = "scripted",
    base_url: str = BASE_URL, model_name: str = "qwen3.5-27b",
    budget_tokens: int = 40000, wall_seconds: int = 900,
    max_output_tokens: int = 6000, top_k: int = 1,
    fee_bps: float = 2.0, slippage_bps: float = 3.0,
    roles: Any | None = None, allow_synthetic_paid_demo: bool = False,
    preflight_only: bool = False,
) -> Path | dict:
    """Validate the data choice, then run dev research or preview without model calls."""
    project_root = Path(project_root).expanduser().resolve(strict=True)
    run_dir = Path(run_dir).expanduser().resolve()
    if run_dir.parent != project_root / "runs" or run_dir.exists():
        raise ValueError("Use a fresh run directory directly under project runs/")
    if mode not in {"scripted", "qwen"}:
        raise ValueError("mode must be scripted or qwen")
    if type(budget_tokens) is not int or budget_tokens < 1 or type(wall_seconds) is not int or wall_seconds < 1:
        raise ValueError("Budget and wall time must be positive integers")
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if type(allow_synthetic_paid_demo) is not bool or type(preflight_only) is not bool:
        raise ValueError("Preview and synthetic demo flags must be booleans")
    if dataset_manifest is not None and boundaries is None:
        raise ValueError("User-supplied data requires all four explicit split dates")
    if dataset_manifest is not None and allow_synthetic_paid_demo:
        raise ValueError("Synthetic demo flag cannot be used with a dataset manifest")
    if mode == "qwen" and dataset_manifest is None and not allow_synthetic_paid_demo:
        raise ValueError("Qwen requires a dataset manifest or explicit --allow-synthetic-paid-demo")
    if boundaries is not None and (len(boundaries) != 4 or any(type(day) is not date for day in boundaries)):
        raise ValueError("Pass all four split dates together")
    if mode == "qwen":
        url = urlsplit(base_url)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("Live model base URL must use HTTPS without embedded credentials or query parameters")
        if model_name != "qwen3.5-27b":
            raise ValueError("Qwen mode requires qwen3.5-27b")
    if dataset_manifest is None:
        with TemporaryDirectory(prefix="quant-preview-") as temporary:
            panel = load_dataset(write_demo_dataset(Path(temporary) / "data"))
    else:
        panel = load_dataset(dataset_manifest)
    boundaries = _split_dates(panel, boundaries)
    if type(top_k) is not int or not 1 <= top_k <= len(panel.symbols):
        raise ValueError("top_k must be within the fixed universe")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in (fee_bps, slippage_bps)):
        raise ValueError("Trading costs must be finite and nonnegative")
    preview = _preview(panel, boundaries, mode=mode, synthetic=dataset_manifest is None,
                        budget_tokens=budget_tokens, wall_seconds=wall_seconds,
                        max_output_tokens=max_output_tokens, top_k=top_k,
                        fee_bps=fee_bps, slippage_bps=slippage_bps)
    if preflight_only:
        return {**preview, "will_call_model": False, "preflight_only": True}
    print("Quant preflight: " + json.dumps(preview, ensure_ascii=False), flush=True)
    if mode == "qwen" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Qwen mode needs OPENAI_API_KEY in the environment")
    commit, dirty = _code_commit(project_root)
    if mode == "qwen" and dirty:
        raise ValueError("Commit or discard local source changes before a live model run")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _execute_quant_dev(
            project_root=project_root, run_dir=run_dir, dataset_manifest=dataset_manifest,
            boundaries=boundaries, mode=mode, base_url=base_url, model_name=model_name,
            budget_tokens=budget_tokens, wall_seconds=wall_seconds,
            max_output_tokens=max_output_tokens, top_k=top_k, fee_bps=fee_bps,
            slippage_bps=slippage_bps, roles=roles, commit=commit, dirty=dirty,
            expected_dataset_sha256=panel.csv_sha256)
    except BaseException as exc:
        try:
            _failure_report(run_dir, exc)
        finally:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        raise QuantRunFailed(f"Development run failed; see {run_dir / 'dev_report.json'}") from exc


def _execute_quant_dev(
    *, project_root: Path = ROOT, run_dir: Path, dataset_manifest: Path | None = None,
    boundaries: tuple[date, date, date, date] | None = None, mode: str = "scripted",
    base_url: str = BASE_URL, model_name: str = "qwen3.5-27b",
    budget_tokens: int = 40000, wall_seconds: int = 900,
    max_output_tokens: int = 6000, top_k: int = 1,
    fee_bps: float = 2.0, slippage_bps: float = 3.0,
    roles: Any | None = None, commit: str, dirty: bool,
    expected_dataset_sha256: str,
) -> Path:
    """Run one dev-only experiment and freeze the best dev candidate."""
    config_path = project_root / "config/quant_agent.json"
    prices_path = project_root / "config/prices_quant_dev.json"
    hard_caps_path = project_root / "config/hard_caps.json"
    triggers_path = project_root / "config/triggers.json"
    config = read_json(config_path, "agent/1")
    hard_caps = read_json(hard_caps_path, "hard_caps/1")
    prices = read_json(prices_path, "prices/1")
    if prices.get("currency") != "TOKENS" or prices.get("test_only") is not True:
        raise ValueError("Quant MVP requires the test-only token price table")
    if model_name not in prices.get("models", {}):
        raise ValueError("No declared token rates for selected model")
    private = run_dir / "private"
    manifest_path = (write_demo_dataset(private) if dataset_manifest is None
                     else _snapshot_dataset(dataset_manifest, private))
    panel = load_dataset(manifest_path)
    if panel.csv_sha256 != expected_dataset_sha256:
        raise DevAdmissionError("input_changed")
    train_end, dev_start, dev_end, holdout_start = _split_dates(panel, boundaries)
    task_root = run_dir / "task_package" / TASK_ID
    workspace = task_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "original.py").write_text(ORIGINAL, encoding="utf-8")
    atomic_json(workspace / "profile.json", {
        "schema": "profile/1", "features": ["time_series"],
        "summary": {"frequency": "daily", "universe_size": len(panel.symbols),
                    "baseline_signal": "five-session trailing momentum",
                    "comparison": "equal weight"},
    })
    inputs = capture_inputs(
        project_root, config, agent_config=config_path, prices=prices_path,
        hard_caps=hard_caps_path, triggers=triggers_path, task_workspace=workspace)
    hard_caps = read_json(hard_caps_path, "hard_caps/1")
    _verify_run_inputs(project_root, inputs)
    library_path = run_dir / "library"
    manifest = {
        "schema": "quant_run/1", "run_id": run_dir.name, "tasks": [TASK_ID],
        "phase": "dev", "model": model_name if mode == "qwen" else "test-scripted",
        "code_commit": commit, "working_tree_dirty": dirty,
        "mode": mode, "started_ts": datetime.now(timezone.utc).isoformat(),
        "library": {"root": str(library_path.relative_to(project_root)),
                    "start_version": "lib-0", "frozen": False},
        "hard_caps": {"budget_ratio": hard_caps["budget_ratio"]},
        "dataset": _descriptor(project_root, manifest_path),
        "agent_config": _descriptor(project_root, config_path),
        "prices": _descriptor(project_root, prices_path),
        "inputs": inputs,
        "model_settings": {"max_output_tokens": max_output_tokens,
                           "base_url": base_url if mode == "qwen" else None},
        "split": {"train_end": train_end.isoformat(), "dev_start": dev_start.isoformat(),
                  "dev_end": dev_end.isoformat(), "holdout_start": holdout_start.isoformat(),
                  "label_horizon": 2},
        "execution": {"top_k": top_k, "fee_bps": fee_bps,
                       "slippage_bps": slippage_bps},
        "limits": {"budget_tokens": budget_tokens, "wall_seconds": wall_seconds,
                   "run_seconds": 20, "memory_mb": 512},
        "label": "DEVELOPMENT ONLY / SYNTHETIC DATA" if dataset_manifest is None
                 else "DEVELOPMENT ONLY / USER-SUPPLIED DATA",
    }
    atomic_json(run_dir / "manifest.json", manifest)
    harness = QuantHarness(
        task_root, manifest_path, train_end=train_end, dev_start=dev_start,
        dev_end=dev_end, holdout_start=holdout_start,
        budget_money=budget_tokens, wall_seconds=wall_seconds, top_k=top_k,
        fee_bps=fee_bps, slippage_bps=slippage_bps, label_horizon=2,
        run_seconds=manifest["limits"]["run_seconds"],
        memory_mb=manifest["limits"]["memory_mb"],
    )
    task_ledger = Ledger.open(run_dir, TASK_ID)
    run_ledger = Ledger.open(run_dir)
    run_ledger.emit("run_start", phase="dev", mode=mode)
    baseline_dir = node_directory(run_dir, TASK_ID, "n0000")
    baseline_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(workspace / "original.py", baseline_dir / "solution.py")
    baseline = harness.score(TASK_ID, baseline_dir, "dev")
    if not baseline.get("ok") or not baseline.get("correct"):
        raise RuntimeError("Original signal failed development scoring")
    with selected_library_settings(config_path, triggers_path):
        _verify_run_inputs(project_root, inputs)
        library = Library.open(library_path)
        if roles is None:
            if mode == "scripted":
                roles = ScriptedRoles()
            else:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                base_url=base_url, timeout=180.0, max_retries=0)
                roles = RoleAgents(ModelClient(
                    model_name, task_ledger, client=client,
                    max_output_tokens=max_output_tokens,
                    budget_limit=budget_tokens * float(hard_caps["budget_ratio"]),
                    reserve_cost=min(12000.0, budget_tokens * 0.3),
                    audit_dir=run_dir / "model_audit",
                ), config, project_root=project_root)
        from evaluation.quote import make_quote_callback
        _verify_run_inputs(project_root, inputs)
        outcome = TaskRunner(
            run_dir=run_dir, task_id=TASK_ID, harness=harness, ledger=task_ledger,
            roles=_CheckedRoles(roles, project_root, inputs), library=library,
            baseline_score=baseline, config=config,
            hard_caps=hard_caps, on_initial_plan=make_quote_callback(task_ledger),
        ).run()
    _verify_run_inputs(project_root, inputs)
    if (sha256_file(manifest_path) != manifest["dataset"]["sha256"]
            or load_dataset(manifest_path).csv_sha256 != panel.csv_sha256):
        raise DevAdmissionError("input_changed")
    usage = _usage_summary(task_ledger._events())
    if not usage["usage_complete"]:
        raise DevAdmissionError("unknown_usage")
    if outcome.reason not in SUCCESS_REASONS:
        raise DevAdmissionError("agent_error")
    tree = ExperimentTree.open(run_dir, TASK_ID)
    best = tree.best_node()
    chosen = node_directory(run_dir, TASK_ID, best["node_id"])
    score_path = chosen / "score.json"
    candidate_path = chosen / "solution.py"
    if not score_path.is_file() or not candidate_path.is_file():
        raise RuntimeError("Selected node lacks immutable score or candidate code")
    selected_score = read_json(score_path, "score/1")
    task_ledger.emit("task_end", best_node=outcome.best_node,
                     dev_score=outcome.dev_score, total_cost=outcome.spent,
                     status=outcome.reason)
    run_ledger.emit("run_end", status="dev_complete",
                    tasks_completed=1, total_cost=outcome.spent)
    node_artifacts = {}
    for node in tree.all_nodes():
        directory = node_directory(run_dir, TASK_ID, node["node_id"])
        solution = directory / "solution.py"
        score = directory / "score.json"
        node_artifacts[node["node_id"]] = {
            "solution": _descriptor(project_root, solution) if solution.is_file() else None,
            "score": _descriptor(project_root, score) if score.is_file() else None,
        }
    selection = {
        "schema": "quant_dev_selection/1", "run_id": run_dir.name,
        "task_id": TASK_ID, "selected_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "highest accepted development score; ties keep earlier node",
        "candidate": _descriptor(project_root, candidate_path),
        "dev_evidence": _descriptor(project_root, score_path),
        "baseline_evidence": _descriptor(project_root, baseline_dir / "score.json"),
        "dev_score": best["score"]["dev"],
        "dev_baseline_score": tree.get("n0000")["score"]["dev"],
        "dataset_manifest_sha256": manifest["dataset"]["sha256"],
        "agent_config_sha256": manifest["agent_config"]["sha256"],
        "prices_sha256": manifest["prices"]["sha256"],
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "tree_sha256": sha256_file(tree.path),
        "node_artifacts": node_artifacts,
        "task_ledger_sha256": sha256_file(task_ledger.path),
        "run_ledger_sha256": sha256_file(run_ledger.path),
        "code_commit": commit, "holdout_scored": False,
    }
    atomic_json(run_dir / "selection.json", selection)
    events = task_ledger._events()
    atomic_json(run_dir / "dev_report.json", {
        "schema": "quant_dev_report/1", "label": manifest["label"], "status": "dev_complete",
        "run_id": run_dir.name, "selection": "selection.json",
        "selected_node": outcome.best_node, "dev_score": outcome.dev_score,
        "original_dev_score": baseline["score"], "experiments": outcome.experiments,
        "development_backtest": {"selected": selected_score["quant_metrics"]["candidate"],
                                 "original": baseline["quant_metrics"]["candidate"],
                                 "equal_weight": selected_score["quant_metrics"]["equal_weight"]},
        "development_window": manifest["split"],
        "trading_costs": manifest["execution"],
        "dataset": manifest["dataset"],
        "nodes": [{"node_id": node["node_id"], "status": node["status"],
                   "dev_score": node["score"]["dev"] if node["score"] else None,
                   "attempts": node["attempts"]} for node in tree.all_nodes()],
        "negative_or_failed_nodes": [n["node_id"] for n in tree.all_nodes()
                                     if n["status"] in {"rejected", "failed"}],
        **_usage_summary(events),
        "holdout_scored": False,
        "warning": "Development evidence only; no out-of-sample or live returns",
    })
    return run_dir / "dev_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Repository checkout containing config/ and runs/")
    parser.add_argument("--mode", choices=("scripted", "qwen"), default="scripted")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--allow-synthetic-paid-demo", action="store_true",
                        help="Explicitly permit Qwen calls on synthetic demo prices")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Inspect data windows and budgets without a run or model call")
    for name in ("train-end", "dev-start", "dev-end", "holdout-start"):
        parser.add_argument("--" + name, type=date.fromisoformat)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--budget-tokens", type=int, default=40000)
    parser.add_argument("--wall-seconds", type=int, default=900)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    args = parser.parse_args(argv)
    choices = (args.train_end, args.dev_start, args.dev_end, args.holdout_start)
    if any(value is not None for value in choices) and not all(value is not None for value in choices):
        parser.error("Pass all four split dates together")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    project_root = args.project_root.expanduser().resolve()
    run_dir = args.run_dir or project_root / "runs" / f"run-{stamp}-{secrets.token_hex(2)}"
    try:
        report = run_quant_dev(project_root=project_root, run_dir=run_dir,
                               dataset_manifest=args.dataset_manifest,
                               boundaries=choices if all(choices) else None,
                               mode=args.mode, base_url=args.base_url,
                               budget_tokens=args.budget_tokens,
                               wall_seconds=args.wall_seconds,
                               max_output_tokens=args.max_output_tokens,
                               top_k=args.top_k, fee_bps=args.fee_bps,
                               slippage_bps=args.slippage_bps,
                               allow_synthetic_paid_demo=args.allow_synthetic_paid_demo,
                               preflight_only=args.preflight_only)
    except Exception as exc:
        print(f"quant development run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2) if isinstance(report, dict) else report)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
