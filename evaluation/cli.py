"""Formal-only operator CLI for readiness, self-research and evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import harness
from ledger import Usage, price_usage
from token_accounting import require_exact_token_accounting

from . import figures
from .batch import (_agent_signature, _bank_tasks, _hash, _json, _verify_code_freeze,
                    _within, execute_batch)
from .self_research import _practice_tasks, auto_evaluate_candidates


ROOT = Path(__file__).resolve().parents[1]
QWEN_STUDY_GATEWAYS = frozenset({
    "https://ai-gateway.ailab.jiuan.com/v1",
    "https://ai-gateway.theunifiedcare.com/v1",
})


def _default_project_root() -> Path:
    current = Path.cwd().resolve()
    return current if (current / "config").is_dir() and (current / "tasks").is_dir() else ROOT


@contextmanager
def _project_environment(root: Path):
    names = ("RA_PROJECT_ROOT", "RA_TASKS_ROOT")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["RA_PROJECT_ROOT"] = str(root)
    os.environ["RA_TASKS_ROOT"] = str(root / "tasks")
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def baseline_solutions(root: Path, bank: dict, *, practice_only: bool) -> dict[str, Path]:
    """Discover the documented workspace/original.py for requested bank tiers."""
    solutions = {}
    for entry in bank["tasks"]:
        if not isinstance(entry, dict):
            raise ValueError("Bank task entry is invalid")
        if practice_only and entry.get("tier") != "practice":
            continue
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or task_id in solutions:
            raise ValueError("Duplicate or invalid task ID in bank")
        path = _within(root, root / "tasks" / task_id / "workspace/original.py")
        if not path.is_file():
            raise FileNotFoundError(f"Baseline solution is missing: {path}")
        solutions[task_id] = path
    if not solutions:
        raise ValueError("Bank has no applicable baseline solutions")
    return solutions


def formal_preflight(
    *, project_root: Path, model: str, prices_path: Path, code_commit: str,
    default_config: Path, naive_config: Path | None = None,
    tuned_config: Path | None = None, exam: bool = False,
) -> dict:
    """Read-only fail-closed check; tuned_config selects full four-line mode."""
    root = Path(project_root).expanduser().resolve()
    blockers = []
    bank = None
    bank_version = None
    accounting_scope = None
    accounting_note = None
    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        blockers.append("macOS sandbox-exec is unavailable")
    for module in ("openai", "numpy", "pandas", "lightgbm", "autogluon"):
        if importlib.util.find_spec(module) is None:
            blockers.append(f"Required dependency {module} is unavailable")
    if not os.environ.get("OPENAI_API_KEY"):
        blockers.append("OPENAI_API_KEY is unavailable")
    try:
        _verify_code_freeze(root, code_commit)
    except (ValueError, OSError) as exc:
        blockers.append(str(exc))
    try:
        prices_path = _within(root, Path(prices_path))
        prices = _json(prices_path)
        if prices.get("schema") != "prices/1" or prices.get("currency") in (None, "", "TEST") or prices.get("test_only"):
            raise ValueError("A non-TEST price or token-accounting table is required")
        price_usage(prices, model, Usage(input=1))
        require_exact_token_accounting(prices, model)
        if prices["currency"] == "TOKENS":
            accounting_scope = "internal_token_study"
            accounting_note = "自选 token 预算的内部冻结题库研究；非现金节省、非官方比赛成绩。"
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"Price table: {exc}")
    if (model == "qwen3.5-27b" and accounting_scope == "internal_token_study"
            and os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
            not in QWEN_STUDY_GATEWAYS):
        blockers.append("OPENAI_BASE_URL must point to the authorized Qwen study gateway")
    try:
        bank = _json(root / "tasks/bank.json")
        bank_version = bank.get("version")
        if tuned_config is None:
            _practice_tasks(bank, harness, formal=True)
        else:
            _bank_tasks(bank, harness, formal=True)
        baseline_solutions(root, bank, practice_only=tuned_config is None)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"Task bank: {exc}")
    try:
        default_path = _within(root, Path(default_config))
        default_signature = _agent_signature(root, _json(default_path))
        if tuned_config is not None:
            if naive_config is None:
                raise ValueError("Full four-line evaluation needs a naive config")
            naive_signature = _agent_signature(root, _json(_within(root, Path(naive_config))))
            tuned_signature = _agent_signature(root, _json(_within(root, Path(tuned_config))))
            if naive_signature == default_signature:
                raise ValueError("Naive and default configurations are semantically identical")
            if tuned_signature == default_signature:
                raise ValueError("Tuned and default configurations are semantically identical")
            if bank_version == "bank-v2":
                from scripts.audit_formal_outputs import audit_carry_forward
                audit_carry_forward(root, bank, _hash(prices_path),
                                    candidate_path=_within(root, Path(tuned_config)),
                                    default_path=default_path)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"Agent configurations: {exc}")
    if tuned_config is not None and isinstance(bank_version, str):
        for stage, enabled in (("validation", True), ("exam", exam)):
            if enabled and (root / "runs" / f"{stage}-{bank_version}.attempt.json").exists():
                blockers.append(f"Formal {stage} for {bank_version} was already attempted")
    return {"schema": "formal_preflight/1", "ready": not blockers,
            "project_root": str(root), "model": model,
            "prices_path": str(prices_path), "code_commit": code_commit,
            "bank_version": bank_version, "mode": "batch" if tuned_config is not None else "auto_research",
            "exam_requested": bool(exam), "blockers": blockers,
            "accounting_scope": accounting_scope, "accounting_note": accounting_note}


def _arguments(parser: argparse.ArgumentParser, *, tuned: bool) -> None:
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--default-config", type=Path, default=Path("config/agent.json"))
    if tuned:
        parser.add_argument("--naive-config", type=Path, default=Path("config/naive.json"))
        parser.add_argument("--tuned-config", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-bank Research Agent evaluation operator")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Audit formal four-line readiness")
    _arguments(preflight, tuned=True)
    preflight.add_argument("--exam", action="store_true")
    research = commands.add_parser("auto-research", help="Generate and test candidates on practice only")
    _arguments(research, tuned=False)
    research.add_argument("--study-dir", type=Path, required=True)
    research.add_argument("--rounds", type=int, default=3)
    research.add_argument("--max-output-tokens", type=int, default=2048)
    research.add_argument("--workers", type=int, default=1,
                          help="Independent repetitions in parallel (1-3)")
    batch = commands.add_parser("batch", help="Run four-line practice and one validation; optionally exam")
    _arguments(batch, tuned=True)
    batch.add_argument("--batch-dir", type=Path, required=True)
    batch.add_argument("--exam", action="store_true")
    batch.add_argument("--workers", type=int, default=1,
                       help="Independent runs in parallel (1-8); sandbox calls stay serialized")
    chart = commands.add_parser("figures", help="Render the audited batch study")
    chart.add_argument("--batch-dir", type=Path, required=True)
    chart.add_argument("--auto-study-dir", type=Path)
    chart.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "figures":
        passed = ["--batch-dir", str(args.batch_dir), "--output", str(args.output)]
        if args.auto_study_dir is not None:
            passed += ["--auto-study-dir", str(args.auto_study_dir)]
        try:
            return figures.main(passed)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(json.dumps({"schema": "cli_error/1", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
    root = (Path(args.project_root).expanduser().resolve() if args.project_root is not None
            else _default_project_root())
    try:
        prices = _within(root, args.prices)
        default = _within(root, args.default_config)
        naive = _within(root, args.naive_config) if hasattr(args, "naive_config") else None
        tuned = _within(root, args.tuned_config) if hasattr(args, "tuned_config") else None
    except (ValueError, OSError) as exc:
        print(json.dumps({"schema": "cli_error/1", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    with _project_environment(root):
        readiness = formal_preflight(
            project_root=root, model=args.model, prices_path=prices,
            code_commit=args.code_commit, default_config=default,
            naive_config=naive, tuned_config=tuned,
            exam=getattr(args, "exam", False))
    if args.command == "preflight":
        print(json.dumps(readiness, ensure_ascii=False, indent=2))
        return 0 if readiness["ready"] else 1
    if not readiness["ready"]:
        print(json.dumps(readiness, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    try:
        bank = _json(root / "tasks/bank.json")
        if args.command == "auto-research":
            with _project_environment(root):
                result = auto_evaluate_candidates(
                    project_root=root, study_dir=_within(root, args.study_dir), harness=harness,
                    default_config=default,
                    baseline_solutions=baseline_solutions(root, bank, practice_only=True),
                    prices_path=prices, model=args.model, code_commit=args.code_commit,
                    rounds=args.rounds, max_output_tokens=args.max_output_tokens,
                    max_workers=args.workers)
            output = {"schema": "cli_result/1", "command": args.command,
                      "study_dir": str(_within(root, args.study_dir)),
                      "selection": result["selection"],
                      "total_research_cost": result["total_research_cost"],
                      "currency": result["currency"],
                      "accounting_scope": readiness.get("accounting_scope"),
                      "accounting_note": readiness.get("accounting_note")}
        else:
            with _project_environment(root):
                result = execute_batch(
                    project_root=root, batch_dir=_within(root, args.batch_dir), harness=harness,
                    baseline_solutions=baseline_solutions(root, bank, practice_only=False),
                    agent_configs={"naive": naive, "default": default,
                                   "default_lib": default, "tuned_lib": tuned},
                    prices_path=prices, model=args.model, code_commit=args.code_commit,
                    run_exam=args.exam, max_workers=args.workers)
            output = {"schema": "cli_result/1", "command": args.command,
                      "batch_dir": str(_within(root, args.batch_dir)),
                      "status": result.get("status", "complete"),
                      "admission": result["admission"],
                      "reports": {tier: len(rows) for tier, rows in result["reports"].items()},
                      "failed_runs": result.get("failed_runs", []),
                      "skipped_runs": len(result.get("skipped_runs", [])),
                      "holdout_failures": len(result.get("holdout_failures", [])),
                      "budget_violations": len(result.get("budget_violations", [])),
                      "tuned_exam_skipped": result["tuned_exam_skipped"],
                      "accounting_scope": readiness.get("accounting_scope"),
                      "accounting_note": readiness.get("accounting_note")}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        # A partial batch still wrote batch_report.json; exit 3 tells scripts to look.
        return 0 if output.get("status", "complete") == "complete" else 3
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"schema": "cli_error/1", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
