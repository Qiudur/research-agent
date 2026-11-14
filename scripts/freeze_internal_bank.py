"""Freeze the bank from freshly audited, dev-only token-cost campaigns.

Only measured naive costs, measured difficulty, the fixed task budget, and
the bank's frozen flag are changed. This tool never opens holdout inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable

from token_accounting import is_exact_token_accounting

from .aggregate_naive_costs import aggregate


MODEL = "qwen3.5-27b"
BUDGET = 40_000
LEVELS = ("easy", "medium", "hard")
TASK_ID = re.compile(r"(?:spd|tab)-[a-z0-9-]+")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + chr(10)).encode("utf-8")


def atomic_replace(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.freeze-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def prepare_changes(root: Path, proposal: dict[str, Any],
                    audit: dict[str, Any]) -> dict[Path, bytes]:
    """Validate every input before staging the exact files to update."""
    root = root.expanduser().resolve()
    bank_path = root / "tasks/bank.json"
    bank = read_json(bank_path)
    digest = sha(bank_path)
    require(audit.get("schema") == "naive_cost_aggregation_audit/1"
            and audit.get("ready") is True and audit.get("blocked") == []
            and audit.get("bank_proposal") == proposal,
            "Fresh ready campaign audit must match the proposal")
    require(proposal.get("schema") == "naive_cost_bank_proposal/1"
            and proposal.get("label") == "DEV TOKEN RESOURCE PROPOSAL; NOT MONEY OR APPLIED"
            and proposal.get("model") == MODEL and proposal.get("currency") == "TOKENS"
            and type(proposal.get("budget_tokens_per_task")) is int
            and proposal["budget_tokens_per_task"] == BUDGET,
            "Proposal model, token unit, or budget differs")
    require(bank.get("schema") == "bank/1" and bank.get("frozen") is False
            and isinstance(bank.get("version"), str)
            and bank["version"] == proposal.get("bank_version") == audit.get("bank_version")
            and digest == proposal.get("source_bank_sha256") == audit.get("source_bank_sha256"),
            "Source bank version, frozen state, or SHA differs")
    prices = read_json(root / "config/prices.json")
    require(prices.get("test_only") is False
            and is_exact_token_accounting(prices, MODEL),
            "Formal prices must count one unit per raw token")
    rows = bank.get("tasks")
    require(isinstance(rows, list) and len(rows) == 24, "Bank needs exactly 24 tasks")
    ids = [row.get("task_id") if isinstance(row, dict) else None for row in rows]
    require(all(isinstance(task_id, str) and TASK_ID.fullmatch(task_id) for task_id in ids)
            and len(set(ids)) == 24
            and sum(task_id.startswith("spd-") for task_id in ids) == 12
            and sum(task_id.startswith("tab-") for task_id in ids) == 12,
            "Bank IDs or scenario counts differ")
    updates = proposal.get("updates")
    require(isinstance(updates, list) and len(updates) == 24,
            "Proposal needs exactly 24 updates")
    by_id: dict[str, dict[str, Any]] = {}
    for update in updates:
        require(isinstance(update, dict)
                and set(update) == {"task_id", "naive_cost", "difficulty"},
                "Proposal update has missing or unexpected fields")
        task_id, cost = update["task_id"], update["naive_cost"]
        require(isinstance(task_id, str) and TASK_ID.fullmatch(task_id)
                and task_id not in by_id
                and type(cost) in (int, float) and math.isfinite(cost) and cost > 0
                and update["difficulty"] in LEVELS,
                "Proposal update has invalid identity, cost, or difficulty")
        by_id[task_id] = update
    require(set(by_id) == set(ids), "Proposal task IDs differ from bank")
    for prefix in ("spd-", "tab-"):
        counts = Counter(by_id[task_id]["difficulty"] for task_id in ids
                         if task_id.startswith(prefix))
        require(all(counts[level] == 4 for level in LEVELS),
                f"Proposal difficulty distribution differs for {prefix}")
    frozen_bank = deepcopy(bank)
    changes: dict[Path, bytes] = {}
    for row in frozen_bank["tasks"]:
        task_id = row["task_id"]
        require(row.get("naive_cost") is None and row.get("difficulty") is None,
                f"Bank row was already calibrated: {task_id}")
        row["naive_cost"] = by_id[task_id]["naive_cost"]
        row["difficulty"] = by_id[task_id]["difficulty"]
        task_path = root / "tasks" / task_id / "task.json"
        require(task_path.is_file() and not task_path.is_symlink()
                and task_path.resolve().is_relative_to(root),
                f"Missing or unsafe task file: {task_id}")
        task = read_json(task_path)
        budget = task.get("budget")
        scenario = "speedup" if task_id.startswith("spd-") else "tabular"
        require(task.get("task_id") == task_id and task.get("scenario") == scenario
                and isinstance(budget, dict) and budget.get("money") is None,
                f"Task identity or unfilled budget differs: {task_id}")
        task["budget"]["money"] = BUDGET
        changes[task_path] = json_bytes(task)
    frozen_bank["frozen"] = True
    changes[bank_path] = json_bytes(frozen_bank)  # Freeze flag is written last.
    return changes


def freeze_bank(root: Path, proposal_path: Path, speedup_campaign: Path,
                tabular_campaign: Path, *, check_only: bool = False,
                auditor: Callable[..., dict[str, Any]] = aggregate) -> dict[str, Any]:
    root = root.expanduser().resolve()
    proposal = read_json(proposal_path)
    audit = auditor(root, speedup_campaign, tabular_campaign)
    changes = prepare_changes(root, proposal, audit)
    bank_path = root / "tasks/bank.json"
    receipt = {"schema": "internal_bank_freeze_receipt/1", "applied": not check_only,
               "bank_version": proposal["bank_version"], "model": MODEL,
               "budget_tokens_per_task": BUDGET, "updated_tasks": 24,
               "source_bank_sha256": proposal["source_bank_sha256"],
               "frozen_bank_sha256": hashlib.sha256(changes[bank_path]).hexdigest()}
    if check_only:
        return receipt
    originals = {path: path.read_bytes() for path in changes}
    replaced: list[Path] = []
    try:
        for path, data in changes.items():
            atomic_replace(path, data)
            replaced.append(path)
    except OSError:
        for path in reversed(replaced):
            atomic_replace(path, originals[path])
        raise
    require(sha(bank_path) == receipt["frozen_bank_sha256"],
            "Frozen bank SHA differs from staged result")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--speedup-campaign", type=Path, required=True)
    parser.add_argument("--tabular-campaign", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1",
                             "--untracked-files=all"], text=True, capture_output=True)
    if status.returncode != 0 or status.stdout.strip():
        parser.error("Freeze requires a clean Git worktree")
    try:
        receipt = freeze_bank(root, args.proposal, args.speedup_campaign,
                              args.tabular_campaign, check_only=args.check_only)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
