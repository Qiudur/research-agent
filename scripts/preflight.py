"""Read-only readiness audit for a formal B/C evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

TIERS = ("practice", "validation", "exam")
SCENARIOS = ("speedup", "tabular")


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def audit(project_root: Path, model: str | None = None) -> dict:
    root = Path(project_root).expanduser().resolve()
    blockers = []
    counts = {scenario: {tier: 0 for tier in TIERS} for scenario in SCENARIOS}
    traps = {scenario: 0 for scenario in SCENARIOS}
    bank_path = root / "tasks/bank.json"
    try:
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"schema": "preflight/1", "ready": False, "blockers": [f"Cannot read bank: {exc}"],
                "counts": counts, "exam_traps": traps}
    if bank.get("schema") != "bank/1" or not isinstance(bank.get("tasks"), list):
        blockers.append("tasks/bank.json is not a valid bank/1 task list")
        tasks = []
    else:
        tasks = bank["tasks"]
    if not bank.get("frozen"):
        blockers.append("Bank is not frozen")
    seen = set()
    for entry in tasks:
        task_id = entry.get("task_id") if isinstance(entry, dict) else None
        if not isinstance(task_id, str) or task_id in seen:
            blockers.append("Bank contains a duplicate or invalid task ID")
            continue
        seen.add(task_id)
        path = root / "tasks" / task_id / "task.json"
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"{task_id}: task.json is missing or invalid")
            continue
        scenario, tier = task.get("scenario"), entry.get("tier")
        if scenario not in SCENARIOS or tier not in TIERS:
            blockers.append(f"{task_id}: unsupported scenario or tier")
            continue
        counts[scenario][tier] += 1
        if tier == "exam" and entry.get("trap") in {"leakage", "cheat_bait", "stop_loss"}:
            traps[scenario] += 1
        if task.get("task_id") != task_id:
            blockers.append(f"{task_id}: task ID mismatch")
        if not _positive(task.get("budget", {}).get("money")):
            blockers.append(f"{task_id}: formal monetary budget is unset")
        for field in ("baseline_raw", "ceiling_raw", "naive_cost"):
            value = entry.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                blockers.append(f"{task_id}: measured {field} is missing")
        if entry.get("difficulty") not in {"easy", "medium", "hard"}:
            blockers.append(f"{task_id}: measured difficulty is missing")
        if not (root / "tasks" / task_id / "workspace/original.py").is_file():
            blockers.append(f"{task_id}: baseline solution is missing")
    for scenario in SCENARIOS:
        sample = counts[scenario]
        if sample["practice"] < 5 or sample["validation"] not in {2, 3} or sample["exam"] < 5:
            blockers.append(f"{scenario}: needs >=5 practice, 2-3 validation and >=5 exam tasks")
        if not traps[scenario]:
            blockers.append(f"{scenario}: lacks an exam trap")
    try:
        prices = json.loads((root / "config/prices.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prices = {}
    rates = prices.get("models", {}).get(model) if model else None
    if (prices.get("schema") != "prices/1" or prices.get("currency") in {None, "", "TEST"}
            or prices.get("test_only") or not isinstance(rates, dict)
            or any(not isinstance(rates.get(key), (int, float)) or isinstance(rates[key], bool)
                   or not math.isfinite(rates[key]) or rates[key] < 0
                   for key in ("input", "cached_input", "output", "reasoning"))):
        blockers.append("Real model name and complete non-TEST prices are missing")
    if not os.environ.get("OPENAI_API_KEY"):
        blockers.append("OPENAI_API_KEY is unavailable")
    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        blockers.append("macOS sandbox-exec is unavailable")
    try:
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1",
                                          "--untracked-files=all"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        blockers.append("Project Git checkout is unavailable")
    else:
        if status:
            blockers.append("Git worktree is not frozen and clean")
    return {"schema": "preflight/1", "ready": not blockers, "bank_version": bank.get("version"),
            "model": model, "counts": counts, "exam_traps": traps, "blockers": blockers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit formal research evaluation readiness")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    result = audit(args.project_root, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
