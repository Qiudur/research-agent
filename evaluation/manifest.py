"""Create a sealed run/1 manifest before A's first task."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


PHASES = frozenset({"dev", "practice", "validation", "exam", "self_research", "demo"})
CONFIGS = frozenset({"naive", "default", "default_lib", "tuned_lib"})
_TASK_ID = re.compile(r"^(tab|spd)-[a-z0-9][a-z0-9-]*$")
_BANK_VERSION = re.compile(r"^bank-v[1-9][0-9]*$")
_LIB_VERSION = re.compile(r"^lib-(0|[1-9][0-9]*)$")


def _within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes project root: {path}")
    return resolved


def _descriptor(root: Path, path: Path, schema: str) -> tuple[dict, dict]:
    actual = _within(root, path if path.is_absolute() else root / path)
    raw = actual.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"Expected {schema} in {actual}")
    return {"path": str(actual.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}, payload


def create_manifest(
    *, project_root: Path, run_dir: Path, phase: str, config: str, rep: int,
    bank_version: str, tasks: list[str], library_root: Path,
    library_start_version: str, library_frozen: bool, model: str,
    agent_config_path: Path, prices_path: Path, code_commit: str,
) -> dict:
    """Write manifest.json once; reject incomplete prices and test rates outside dev."""
    project_root = Path(project_root).expanduser().resolve()
    run_path = Path(run_dir).expanduser()
    library_path = Path(library_root).expanduser()
    run_dir = _within(project_root, run_path if run_path.is_absolute() else project_root / run_path)
    library_root = _within(project_root, library_path if library_path.is_absolute() else project_root / library_path)
    if run_dir.parent != project_root / "runs" or not re.fullmatch(r"run-[0-9]{8}-[0-9]{6}-[a-z0-9]{4}", run_dir.name):
        raise ValueError("run_dir must match runs/run-YYYYMMDD-HHMMSS-xxxx")
    if phase not in PHASES or config not in CONFIGS or isinstance(rep, bool) or rep not in {1, 2, 3}:
        raise ValueError("Invalid phase/config/rep")
    if not _BANK_VERSION.fullmatch(bank_version):
        raise ValueError("Invalid bank version")
    if not tasks or len(set(tasks)) != len(tasks) or any(not _TASK_ID.fullmatch(item) for item in tasks):
        raise ValueError("Tasks must be unique valid task IDs")
    if not _LIB_VERSION.fullmatch(library_start_version):
        raise ValueError("Invalid library start version")
    if phase == "exam" and not library_frozen:
        raise ValueError("Exam library must be frozen")
    if not isinstance(model, str) or not model.strip() or model == "开赛给的模型名":
        raise ValueError("A concrete model name is required")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", code_commit):
        raise ValueError("code_commit must be a Git SHA")
    agent_descriptor, _ = _descriptor(project_root, Path(agent_config_path), "agent/1")
    price_descriptor, prices = _descriptor(project_root, Path(prices_path), "prices/1")
    if prices.get("currency") == "TEST" and phase != "dev":
        raise ValueError("TEST prices are restricted to dev runs")
    if prices.get("test_only") and phase != "dev":
        raise ValueError("Test-only prices are restricted to dev runs")
    if not isinstance(prices.get("currency"), str) or not prices["currency"].strip():
        raise ValueError("Price currency is unset")
    unit = prices.get("unit_tokens")
    rates = prices.get("models", {}).get(model)
    if not isinstance(unit, int) or unit <= 0 or not isinstance(rates, dict):
        raise ValueError("No complete price table for the selected model")
    for bucket in ("input", "cached_input", "output", "reasoning"):
        value = rates.get(bucket)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Model price {bucket} is missing or invalid")
    hard_caps = json.loads((project_root / "config/hard_caps.json").read_text(encoding="utf-8"))
    if hard_caps.get("schema") != "hard_caps/1":
        raise ValueError("Unsupported hard cap schema")
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(run_dir / "manifest.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "run/1", "run_id": run_dir.name, "phase": phase,
        "config": config, "rep": rep, "bank_version": bank_version,
        "tasks": tasks,
        "library": {"root": str(library_root.relative_to(project_root)),
                    "start_version": library_start_version, "frozen": bool(library_frozen)},
        "agent_config": agent_descriptor, "prices": price_descriptor,
        "hard_caps": {"budget_ratio": hard_caps["budget_ratio"]},
        "code_commit": code_commit, "model": model,
        "started_ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "ended_ts": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = ["create_manifest"]
