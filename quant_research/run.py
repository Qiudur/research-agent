"""Audited development-only quant Agent run; never scores sealed holdout."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Any

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


def run_quant_dev(
    *, project_root: Path = ROOT, run_dir: Path, dataset_manifest: Path | None = None,
    boundaries: tuple[date, date, date, date] | None = None, mode: str = "scripted",
    base_url: str = BASE_URL, model_name: str = "qwen3.5-27b",
    budget_tokens: int = 40000, wall_seconds: int = 900,
    max_output_tokens: int = 6000, top_k: int = 1,
    fee_bps: float = 2.0, slippage_bps: float = 3.0,
    roles: Any | None = None,
) -> Path:
    """Run one dev-only experiment and freeze the best dev candidate."""
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
    if mode == "qwen" and (model_name != "qwen3.5-27b" or not os.environ.get("OPENAI_API_KEY")):
        raise ValueError("Qwen mode needs qwen3.5-27b and OPENAI_API_KEY in the environment")
    if mode == "qwen" and not base_url.startswith("https://"):
        raise ValueError("Live model base URL must use HTTPS")
    commit, dirty = _code_commit(project_root)
    if mode == "qwen" and dirty:
        raise ValueError("Commit or discard local source changes before a live model run")
    source_panel = load_dataset(dataset_manifest) if dataset_manifest is not None else None
    if source_panel is not None:
        _split_dates(source_panel, boundaries)
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
    run_dir.mkdir(parents=True, exist_ok=False)
    private = run_dir / "private"
    manifest_path = (write_demo_dataset(private) if dataset_manifest is None
                     else _snapshot_dataset(dataset_manifest, private))
    panel = load_dataset(manifest_path)
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
        outcome = TaskRunner(
            run_dir=run_dir, task_id=TASK_ID, harness=harness, ledger=task_ledger,
            roles=roles, library=library, baseline_score=baseline, config=config,
            hard_caps=hard_caps, on_initial_plan=make_quote_callback(task_ledger),
        ).run()
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
    model_events = [event for event in events if event["kind"] == "llm_call"]
    usage_complete = all(event.get("usage_status") == "reported"
                         for event in model_events)
    reported_buckets = {
        key: sum(event["tokens"][key] for event in model_events
                 if event.get("usage_status") == "reported")
        for key in ("input", "cached_input", "output", "reasoning")
    }
    atomic_json(run_dir / "dev_report.json", {
        "schema": "quant_dev_report/1", "label": manifest["label"],
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
        "model_calls": len(model_events),
        "unknown_usage_calls": sum(event.get("usage_status") != "reported"
                                   for event in model_events),
        "usage_complete": usage_complete,
        "reported_token_buckets": reported_buckets,
        "total_token_buckets": reported_buckets if usage_complete else None,
        "resource_cost": outcome.spent if usage_complete else None,
        "reported_resource_cost": outcome.spent, "currency": "TOKENS",
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
                               slippage_bps=args.slippage_bps)
    except Exception as exc:
        print(f"quant development run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
