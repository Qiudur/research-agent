"""Three-repeat practice, validation and single-use exam orchestration.

TEST prices force dev manifests; this module records logical phases separately.
Every trained (configuration, repetition) starts with an empty library. Later
phases receive independent frozen copies of its final practice snapshot.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
from threading import Lock
from typing import Any, Callable, Mapping

from library import Library

from .admission import admit_configuration
from .finalize import _eval_window, finish_run, start_run
from .manifest import CONFIGS, create_manifest
from .orchestrate import execute_task
from .report import run_report, task_report


TIERS = ("practice", "validation", "exam")
DEFAULT_CONFIGS = ("naive", "default", "default_lib", "tuned_lib")
TRAINED_CONFIGS = frozenset({"default_lib", "tuned_lib"})


class _SerializedHarness:
    """Let independent model calls overlap without overlapping sandbox work."""

    def __init__(self, base: Any, lock: Lock):
        self._base = base
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def score(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._base.score(*args, **kwargs)

    def score_holdout(self, task_id: str, node_dir: Path) -> Any:
        with self._lock:
            with _eval_window():
                return self._base.score(task_id, node_dir, "holdout")

    def run_script(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._base.run_script(*args, **kwargs)

    def profile_code(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._base.profile_code(*args, **kwargs)


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _library_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(_hash(path)))
    return digest.hexdigest()


def _within(root: Path, path: Path) -> Path:
    result = (path if path.is_absolute() else root / path).expanduser().resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path escapes project root: {path}")
    return result


def _agent_signature(root: Path, config: dict) -> str:
    """Compare actual settings and prompt bytes, not paths or JSON whitespace."""
    prompts = config.get("prompts")
    if not isinstance(prompts, dict) or set(prompts) != {
            "planner", "executor", "diagnoser", "summarizer"}:
        raise ValueError("Agent config must contain exactly four role prompts")
    normalized = {key: value for key, value in config.items() if key != "prompts"}
    normalized["prompts"] = {role: _hash(_within(root, Path(relative)))
                             for role, relative in prompts.items()}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _run_path(root: Path) -> Path:
    for _ in range(100):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = root / "runs" / f"run-{stamp}-{secrets.token_hex(2)}"
        if not path.exists():
            return path
    raise RuntimeError("Could not allocate an unused run ID")


def _verify_code_freeze(root: Path, commit: str) -> None:
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Formal code_commit must be the full 40-character Git SHA")
    try:
        repo = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL).strip()
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Formal execution needs an accessible Git repository") from exc
    if Path(repo).resolve() != root or head != commit:
        raise ValueError("Formal code_commit does not match the project HEAD")
    if dirty:
        raise ValueError("Formal execution needs a clean Git worktree")


def _mark_attempt(root: Path, stage: str, bank_version: str,
                  code_commit: str, batch_dir: Path) -> None:
    path = root / "runs" / f"{stage}-{bank_version}.attempt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "exam_attempt/1" if stage == "exam" else "validation_attempt/1",
                                 "bank_version": bank_version, "code_commit": code_commit,
                                 "batch_dir": str(batch_dir)}) + chr(10))


def _bank_tasks(bank: dict, harness: Any, *, formal: bool) -> dict[str, list[str]]:
    if bank.get("schema") != "bank/1" or not isinstance(bank.get("tasks"), list):
        raise ValueError("Expected bank/1 with tasks")
    if formal and not bank.get("frozen"):
        raise ValueError("Formal batch requires a frozen bank")
    tiers: dict[str, list[str]] = {tier: [] for tier in TIERS}
    seen: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    exam_traps = {"tabular": 0, "speedup": 0}
    for item in bank["tasks"]:
        if not isinstance(item, dict):
            raise ValueError("Bank task entry must be an object")
        task_id, tier = item.get("task_id"), item.get("tier")
        if not isinstance(task_id, str) or task_id in seen or tier not in tiers:
            raise ValueError("Duplicate or invalid bank task/tier")
        seen.add(task_id)
        tiers[tier].append(task_id)  # The bank fixes practice order.
        if formal:
            task = harness.load_task(task_id)
            scenario = task.get("scenario")
            if scenario not in exam_traps:
                raise ValueError(f"Invalid scenario for {task_id}")
            if item.get("difficulty") not in {"easy", "medium", "hard"}:
                raise ValueError(f"Formal task {task_id} lacks measured difficulty")
            if item.get("trap") not in {None, "leakage", "cheat_bait", "stop_loss"}:
                raise ValueError(f"Formal task {task_id} has unknown trap type")
            expected_formula = "gap" if scenario == "tabular" else "ratio"
            if item.get("score_formula") != expected_formula:
                raise ValueError(f"Formal task {task_id} has wrong score formula")
            counts[(scenario, tier)] = counts.get((scenario, tier), 0) + 1
            if tier == "exam" and item.get("trap") not in (None, False, ""):
                exam_traps[scenario] += 1
            budget = task.get("budget", {}).get("money")
            if (isinstance(budget, bool) or not isinstance(budget, (int, float))
                    or not math.isfinite(budget) or budget <= 0):
                raise ValueError(f"Formal task {task_id} needs a positive monetary budget")
            for field in ("baseline_raw", "ceiling_raw", "naive_cost"):
                value = item.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Formal bank task {task_id} lacks measured {field}")
            if scenario == "tabular":
                metric = task.get("metric", {})
                for field in ("baseline_raw", "ceiling_raw"):
                    task_value = metric.get(field)
                    if (isinstance(task_value, bool) or not isinstance(task_value, (int, float))
                            or not math.isfinite(task_value)
                            or not math.isclose(task_value, item[field], rel_tol=1e-12, abs_tol=1e-12)):
                        raise ValueError(f"Tabular metric {field} differs from bank for {task_id}")
            if item["naive_cost"] <= 0:
                raise ValueError(f"Formal bank task {task_id} has invalid measurements")
            if scenario == "speedup" and (item["baseline_raw"] <= 0 or item["ceiling_raw"] <= 0):
                raise ValueError(f"Formal bank task {task_id} has invalid speedup measurements")
            if (item["ceiling_raw"] <= item["baseline_raw"]
                    and not (scenario == "speedup" and item.get("trap") == "stop_loss")):
                raise ValueError(f"Formal bank task {task_id} has invalid measurements")
    if formal:
        for scenario in exam_traps:
            if (counts.get((scenario, "practice"), 0) < 5
                    or counts.get((scenario, "validation"), 0) not in (2, 3)
                    or counts.get((scenario, "exam"), 0) < 5
                    or exam_traps[scenario] < 1):
                raise ValueError(f"Formal bank lacks required tiers or exam trap: {scenario}")
    return tiers


def _prepare_library(destination: Path, source: Path | None) -> str:
    if source is None:
        library = Library.open(destination)
    else:
        shutil.copytree(source, destination)
        library = Library.open(destination, frozen=True)
    return library.version


def execute_batch(
    *, project_root: Path, batch_dir: Path, harness: Any,
    baseline_solutions: Mapping[str, Path], agent_configs: Mapping[str, Path],
    prices_path: Path, model: str, code_commit: str,
    configs: tuple[str, ...] = DEFAULT_CONFIGS,
    run_validation: bool = True, run_exam: bool = False,
    max_workers: int = 1,
    test_only_budget: float | None = None,
    roles_factory: Callable[[str, int, str, Path], Any] | None = None,
    task_executor: Callable[..., dict] = execute_task,
) -> dict:
    """Run fixed-order practice then validation, optionally one-shot exam."""
    root = Path(project_root).expanduser().resolve()
    batch_dir = _within(root, Path(batch_dir))
    if batch_dir.exists():
        raise FileExistsError(f"Batch is one-shot: {batch_dir}")
    if run_exam and not run_validation:
        raise ValueError("Exam requires validation admission first")
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be an integer between 1 and 4")
    if not configs or len(set(configs)) != len(configs) or set(configs) - CONFIGS:
        raise ValueError("Configs must be unique run/1 configuration labels")
    if "tuned_lib" in configs and "default_lib" not in configs:
        raise ValueError("Tuned library requires default_lib validation comparator")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A concrete model is required")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", code_commit):
        raise ValueError("A concrete Git commit is required")
    bank_path = root / "tasks" / "bank.json"
    bank = _json(bank_path)
    bank_version = bank.get("version")
    if not isinstance(bank_version, str):
        raise ValueError("Bank version is missing")
    formal = test_only_budget is None
    tasks = _bank_tasks(bank, harness, formal=formal)
    if (not tasks["practice"] or (run_validation and not tasks["validation"])
            or (run_exam and not tasks["exam"])):
        raise ValueError("Requested phases need nonempty task tiers")
    if isinstance(test_only_budget, bool) or (test_only_budget is not None and
            (not isinstance(test_only_budget, (int, float))
             or not math.isfinite(test_only_budget) or test_only_budget <= 0)):
        raise ValueError("TEST-only budget must be positive and finite")
    prices_path = _within(root, Path(prices_path))
    prices = _json(prices_path)
    if (prices.get("schema") != "prices/1" or
            not isinstance(prices.get("models", {}).get(model), dict)):
        raise ValueError("Selected model has no prices/1 rates")
    rates = prices["models"][model]
    if any(isinstance(rates.get(key), bool) or not isinstance(rates.get(key), (int, float))
           or not math.isfinite(rates[key]) or rates[key] < 0
           for key in ("input", "cached_input", "output", "reasoning")):
        raise ValueError("Selected model has incomplete token prices")
    if formal:
        if prices.get("currency") in (None, "", "TEST") or prices.get("test_only"):
            raise ValueError("Formal execution needs real currency and prices")
        if roles_factory is not None or task_executor is not execute_task:
            raise ValueError("Formal execution may not inject scripted roles or executor")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Formal execution needs OPENAI_API_KEY")
    elif prices.get("currency") != "TEST":
        raise ValueError("TEST-only batch requires TEST prices")
    selected_tasks = tasks["practice"] + (tasks["validation"] if run_validation else [])
    selected_tasks += tasks["exam"] if run_exam else []
    baselines = {task_id: _within(root, Path(baseline_solutions[task_id]))
                 for task_id in selected_tasks}
    for task_id, path in baselines.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline solution for {task_id}: {path}")
    config_paths = {config: _within(root, Path(agent_configs[config])) for config in configs}
    config_data = {}
    for config, path in config_paths.items():
        config_data[config] = _json(path)
        if config_data[config].get("schema") != "agent/1":
            raise ValueError(f"Invalid agent config for {config}")
    if formal and run_exam:
        if set(configs) != set(DEFAULT_CONFIGS):
            raise ValueError("Formal exam requires all four planned configurations")
        signatures = {config: _agent_signature(root, data) for config, data in config_data.items()}
        if signatures["naive"] == signatures["default"]:
            raise ValueError("Formal naive and default configurations must differ")
        if signatures["default"] != signatures["default_lib"]:
            raise ValueError("Default and default_lib must share agent settings")
        if signatures["tuned_lib"] == signatures["default_lib"]:
            raise ValueError("Tuned and default configurations must differ")
    sealed_inputs = {str(path): _hash(path) for path in
                     (bank_path, prices_path, *config_paths.values(), *baselines.values())}
    if formal:
        for path in (root / "config/hard_caps.json", root / "config/triggers.json"):
            sealed_inputs[str(path)] = _hash(path)
        for data in config_data.values():
            for relative in data["prompts"].values():
                path = _within(root, Path(relative))
                if not path.read_text(encoding="utf-8").strip():
                    raise ValueError(f"Empty role prompt: {path}")
                sealed_inputs[str(path)] = _hash(path)
        if not (batch_dir.is_relative_to(root / "batches")
                or batch_dir.is_relative_to(root / "runs")):
            raise ValueError("Formal batch_dir must be under batches/ or runs/")
        _verify_code_freeze(root, code_commit)
        for stage, enabled in (("validation", run_validation), ("exam", run_exam)):
            if enabled and (root / "runs" / f"{stage}-{bank_version}.attempt.json").exists():
                raise FileExistsError(f"This bank version's formal {stage} was already attempted")
    batch_dir.mkdir(parents=True)
    (batch_dir / "plan.json").write_text(json.dumps({
        "schema": "batch_plan/1", "label": "TEST ONLY" if not formal else "FORMAL",
        "bank_version": bank_version, "tasks": tasks, "configs": configs,
        "reps": [1, 2, 3], "run_validation": run_validation,
        "run_exam": run_exam, "max_workers": max_workers,
        "sandbox_calls_serialized": max_workers > 1,
        "sealed_inputs": sealed_inputs,
    }, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    reports: dict[str, list[dict]] = {tier: [] for tier in TIERS}
    runs: list[dict] = []
    practice_libraries: dict[tuple[str, int], Path] = {}
    task_harness = (_SerializedHarness(harness, Lock()) if max_workers > 1 else harness)

    def reserve_run() -> Path:
        for _ in range(100):
            run_dir = _run_path(root)
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return run_dir
        raise RuntimeError("Could not reserve an unused run ID")

    def one_run(tier: str, config: str, rep: int, source: Path | None,
                run_dir: Path) -> tuple[dict, dict, Path]:
        if formal:
            _verify_code_freeze(root, code_commit)
        if any(_hash(Path(path)) != digest for path, digest in sealed_inputs.items()):
            raise ValueError("A sealed batch input changed during execution")
        library_root = run_dir / "library"
        start_version = _prepare_library(library_root, source)
        frozen = tier != "practice"
        frozen_hash = _library_hash(library_root) if frozen else None
        manifest = create_manifest(
            project_root=root, run_dir=run_dir, phase=tier if formal else "dev",
            config=config, rep=rep, bank_version=bank_version, tasks=tasks[tier],
            library_root=library_root, library_start_version=start_version,
            library_frozen=frozen, model=model, agent_config_path=config_paths[config],
            prices_path=prices_path, code_commit=code_commit,
        )
        start_run(run_dir)
        quote_history: dict[str, list[dict]] = {"spd": [], "tab": []}
        for task_id in tasks[tier]:
            scenario = task_id.split("-", 1)[0]
            kwargs = dict(run_dir=run_dir, task_id=task_id, harness=task_harness,
                          baseline_solution=baselines[task_id], test_only_budget=test_only_budget,
                          quote_history=tuple(quote_history[scenario]))
            if roles_factory is not None:
                kwargs["roles"] = roles_factory(config, rep, task_id, run_dir)
            task_executor(**kwargs)
            completed_task = task_report(run_dir, task_id)
            if completed_task["quote"] is not None:
                quote_history[scenario].append({
                    "predicted": completed_task["quote"],
                    "actual": {
                        "cost": completed_task["total_cost"],
                        "seconds": completed_task["seconds"]["wall"],
                        "score": completed_task["holdout_score"],
                    },
                })
        finish_run(run_dir)
        report = run_report(run_dir)  # Reconcile ledger, tree and holdout evidence.
        if frozen and _library_hash(library_root) != frozen_hash:
            raise ValueError("Frozen library changed during validation/exam")
        row = {"logical_phase": tier, "manifest_phase": manifest["phase"],
               "config": config, "rep": rep, "run_id": run_dir.name,
               "report_path": str(run_dir / "report.json"),
               "library_start_version": start_version,
               "test_only": not formal}
        return report, row, library_root

    def run_phase(tier: str, specs: list[tuple[str, int, Path | None]]) -> None:
        jobs = [(config, rep, source, reserve_run()) for config, rep, source in specs]
        if max_workers == 1:
            completed = [one_run(tier, *job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(one_run, tier, *job) for job in jobs]
                completed = [future.result() for future in futures]
        for (config, rep, _), (report, row, library_root) in zip(specs, completed):
            reports[tier].append(report)
            runs.append(row)
            if tier == "practice":
                practice_libraries[(config, rep)] = library_root

    run_phase("practice", [(config, rep, None) for rep in (1, 2, 3) for config in configs])
    if run_validation:
        if formal:
            _mark_attempt(root, "validation", bank_version, code_commit, batch_dir)
        run_phase("validation", [
            (config, rep, practice_libraries[(config, rep)])
            for rep in (1, 2, 3) for config in configs if config in TRAINED_CONFIGS])
    admission = None
    if run_validation and "tuned_lib" in configs:
        admission = admit_configuration(
            reports["validation"], baseline="default_lib", candidate="tuned_lib")
    if run_exam:
        if formal:
            _mark_attempt(root, "exam", bank_version, code_commit, batch_dir)
        with (batch_dir / "exam_attempt.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema": "exam_attempt/1",
                                     "label": "TEST ONLY" if not formal else "FORMAL"}) + chr(10))
        run_phase("exam", [
            (config, rep, practice_libraries[(config, rep)]
             if config in TRAINED_CONFIGS else None)
            for rep in (1, 2, 3) for config in configs
            if config != "tuned_lib" or (admission is not None and admission["claim"] == "admitted")])
    result = {"schema": "batch_report/1", "label": "TEST ONLY" if not formal else "FORMAL",
              "bank_version": bank_version, "runs": runs, "reports": reports,
              "admission": admission,
              "tuned_exam_skipped": bool(run_exam and "tuned_lib" in configs
                                        and admission["claim"] != "admitted")}
    (batch_dir / "batch_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return result


__all__ = ["execute_batch"]
