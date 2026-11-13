"""Minimal genuine speedup scorer for one local task.

This is a pilot of B's interface, not a security sandbox or a full task bank.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from research_agent.io import append_jsonl, read_json


ROOT = Path(__file__).resolve().parent
TASK_ID = "spd-distance-matrix"
TASK_ROOT = ROOT / "tasks" / TASK_ID
WORKSPACE = TASK_ROOT / "workspace"
REFERENCE = TASK_ROOT / "hidden" / "reference.py"


class PilotHarness:
    def __init__(self, *, size: int = 2500, repetitions: int = 7, measurements_path: Path | None = None):
        self.size = size
        self.repetitions = repetitions
        self.next_seed = 20261113
        self.measurements_path = measurements_path or ROOT / "measurements.jsonl"

    def load_task(self, task_id: str) -> dict:
        if task_id != TASK_ID:
            raise KeyError(task_id)
        task = read_json(TASK_ROOT / "task.json", "task/1")
        task["workspace"] = WORKSPACE
        task["features"] = read_json(WORKSPACE / "profile.json", "profile/1")["features"]
        return task

    def run_script(self, task_id: str, node_dir: Path, argv: list[str], limits=None) -> dict:
        if task_id != TASK_ID or argv != ["solution.py"]:
            raise ValueError("Pilot only checks its solution.py")
        start = perf_counter()
        try:
            ast.parse((Path(node_dir) / "solution.py").read_text(encoding="utf-8"))
            error = ""
            exit_code = 0
        except Exception as exc:
            error = str(exc)
            exit_code = 1
        return {"exit_code": exit_code, "timed_out": False, "oom": False,
                "seconds": perf_counter() - start, "peak_mb": 0,
                "stdout_tail": "", "stderr_tail": error}

    def score(self, task_id: str, node_dir: Path, split: str, seed: int | None = None) -> dict:
        if task_id != TASK_ID or split != "dev":
            raise ValueError("Pilot supports only its dev score")
        used_seed = self.next_seed if seed is None else seed
        if seed is None:
            self.next_seed += 1
        start = perf_counter()
        command = [sys.executable, "-m", "pilot.worker",
                   "--original", str(WORKSPACE / "original.py"),
                   "--candidate", str(Path(node_dir) / "solution.py"),
                   "--reference", str(REFERENCE),
                   "--seed", str(used_seed), "--size", str(self.size),
                   "--repetitions", str(self.repetitions)]
        try:
            process = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True, timeout=120, check=True)
            measured = json.loads(process.stdout)
        except Exception as exc:
            measured = {"ok": False, "correct": False, "error": {"kind": "exception", "message": str(exc), "trace_tail": ""}}
        run_seconds = perf_counter() - start
        if measured.get("ok") and measured.get("correct"):
            result = {
                "schema": "score/1", "ok": True, "split": "dev",
                "score": measured["normalized_score"], "raw": measured["speedup"],
                "spread": measured["spread"], "correct": True, "cheat_flags": [],
                "error": None, "seconds": {"run": run_seconds, "queue": 0.0}, "seed": used_seed,
            }
            append_jsonl(self.measurements_path, {
                "schema": "pilot_measurement/1", "task_id": task_id, "seed": used_seed,
                "node_dir": str(Path(node_dir).resolve()), "size": self.size,
                "repetitions": self.repetitions, "medians": measured["medians"],
                "timings": measured["timings"], "raw_speedup": measured["speedup"],
                "normalized_score": measured["normalized_score"],
            })
        elif measured.get("ok"):
            result = {
                "schema": "score/1", "ok": True, "split": "dev", "score": 0.0,
                "raw": 0.0, "spread": 0.0, "correct": False,
                "cheat_flags": ["behavior_change"], "error": None,
                "seconds": {"run": run_seconds, "queue": 0.0}, "seed": used_seed,
            }
        else:
            result = {
                "schema": "score/1", "ok": False, "split": "dev", "score": None,
                "raw": None, "spread": 0.0, "correct": False, "cheat_flags": [],
                "error": measured.get("error"), "seconds": {"run": run_seconds, "queue": 0.0}, "seed": used_seed,
            }
        return result
