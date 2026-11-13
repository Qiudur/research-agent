"""Persist each experiment's reviewable code, run output, and score."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from .io import atomic_json, field


def node_directory(run_dir: Path, task_id: str, node_id: str) -> Path:
    if not task_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in task_id):
        raise ValueError("Invalid task_id")
    if not node_id.startswith("n") or len(node_id) != 5 or not node_id[1:].isdigit():
        raise ValueError("Invalid node_id")
    return Path(run_dir) / "tasks" / task_id / "nodes" / node_id


def save_solution(node_dir: Path, code: str, parent_dir: Path | None = None) -> Path:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Executor returned empty solution")
    node_dir = Path(node_dir)
    node_dir.mkdir(parents=True, exist_ok=True)
    solution = node_dir / "solution.py"
    solution.write_text(code.rstrip() + "\n", encoding="utf-8")
    previous = ""
    if parent_dir is not None:
        parent_solution = Path(parent_dir) / "solution.py"
        if parent_solution.exists():
            previous = parent_solution.read_text(encoding="utf-8")
    diff = difflib.unified_diff(
        previous.splitlines(keepends=True),
        solution.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile="parent/solution.py",
        tofile="solution.py",
    )
    (node_dir / "parent.diff").write_text("".join(diff), encoding="utf-8")
    return solution


def append_run_log(node_dir: Path, attempt: int, result: Any) -> None:
    node_dir = Path(node_dir)
    node_dir.mkdir(parents=True, exist_ok=True)
    with (node_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== attempt {attempt} ===\n")
        for name in ("exit_code", "timed_out", "oom", "seconds", "stdout_tail", "stderr_tail"):
            handle.write(f"{name}: {field(result, name)}\n")


def save_score(node_dir: Path, result: Any) -> None:
    from .io import fields

    data = fields(result)
    if data.get("schema") != "score/1":
        raise ValueError(f"Unsupported score schema: {data.get('schema')!r}")
    atomic_json(Path(node_dir) / "score.json", data)
