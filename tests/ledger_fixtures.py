"""Synthetic ledger/tree builders shared by the metrics, replay and A/B tests.

They write the same files a real task leaves behind (``ledger.jsonl``,
``tree.jsonl``, ``holdout_score.json`` and a run ``manifest.json``) but only
the fields the offline analysis reads, so no model, sandbox or price table is
needed.
"""

from __future__ import annotations

import json
from pathlib import Path


def _tokens(total: int) -> dict:
    inputs = total * 3 // 4
    return {"input": inputs, "cached_input": 0, "output": total - inputs, "reasoning": 0}


class TaskWriter:
    """Append ledger events and tree snapshots in the order a real runner would."""

    def __init__(self, run_dir: Path, task_id: str, *, baseline_dev: float = 0.1):
        self.run_id = Path(run_dir).name
        self.task_id = task_id
        self.task_dir = Path(run_dir) / "tasks" / task_id
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.snapshots: list[dict] = []
        self.emit("task_start", budget=40000, wall_seconds=1800)
        self.snapshot("n0000", "accepted", baseline_dev, hypothesis="Baseline")

    def emit(self, kind: str, node_id: str | None = None, **fields) -> None:
        self.events.append({"schema": "ledger/1", "seq": len(self.events) + 1,
                            "ts": "2026-11-15T00:00:00.000Z", "run_id": self.run_id,
                            "task_id": self.task_id, "node_id": node_id, "kind": kind, **fields})

    def snapshot(self, node_id: str, status: str, dev: float | None, **extra) -> None:
        score = None if dev is None else {"dev": dev, "raw": dev, "spread": 0.0,
                                          "correct": True, "cheat_flags": []}
        self.snapshots.append({"schema": "node/1", "run_id": self.run_id, "task_id": self.task_id,
                               "node_id": node_id, "parent": None if node_id == "n0000" else "n0000",
                               "status": status, "score": score, **extra})

    def llm(self, role: str, purpose: str, tokens: int, node_id: str | None = None,
            ok: bool = True) -> None:
        self.emit("llm_call", node_id, role=role, purpose=purpose, effort="low",
                  model="test-scripted", tokens=_tokens(tokens), cost=float(tokens),
                  seconds=0.1, prompt_hash="a" * 64, ok=ok, error=None if ok else "bad")

    def plan(self, tokens: int) -> "TaskWriter":
        self.llm("planner", "plan", tokens)
        return self

    def review(self, tokens: int) -> "TaskWriter":
        self.llm("planner", "review", tokens)
        return self

    def node(self, node_id: str, status: str, dev: float | None, tokens: int, *,
             diag: int = 0, scored: bool = True, ran: bool = True) -> "TaskWriter":
        """One finished node: queued/running/final snapshots plus its ledger events.

        ``scored=False`` is a run that failed before scoring; ``ran=False`` is an
        executor failure that never produced a runnable solution.
        """
        self.snapshot(node_id, "queued", None)
        self.snapshot(node_id, "running", None)
        self.llm("executor", "code", tokens, node_id, ok=ran)
        if ran:
            self.emit("experiment", node_id, attempt=1, seconds_run=0.1,
                      exit_code=0 if scored else 1, timed_out=False, oom=False)
        if ran and scored:
            self.emit("score", node_id, split="dev", score=dev, raw=dev, spread=0.0, correct=True,
                      cheat_flags=[], seconds_queue=0.0)
        if diag:
            self.llm("diagnoser", "diagnose", diag, node_id)
        self.snapshot(node_id, status, dev if scored else None)
        return self

    def summarize(self, tokens: int) -> "TaskWriter":
        self.llm("summarizer", "summarize", tokens)
        return self

    def stop(self, reason: str, best_node: str = "n0000", dev: float | None = None) -> "TaskWriter":
        best = next((s for s in reversed(self.snapshots)
                     if s["node_id"] == best_node and s["score"]), None)
        self.emit("stop", reason=reason, best_node=best_node,
                  dev_score=dev if dev is not None else (best["score"]["dev"] if best else None))
        return self

    def finish(self, *, holdout: dict | None = None, task_end: dict | None = None) -> Path:
        if holdout is not None:
            (self.task_dir / "holdout_score.json").write_text(json.dumps({
                "schema": "score/1", "split": "holdout", "ok": True, "correct": True,
                "spread": 0.0, "cheat_flags": [], "error": None, **holdout}))
            self.emit("task_end", best_node="n0000", holdout_score=holdout.get("score"),
                      total_cost=sum(e["cost"] for e in self.events if e["kind"] == "llm_call"),
                      **(task_end or {}))
        (self.task_dir / "ledger.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in self.events), encoding="utf-8")
        (self.task_dir / "tree.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in self.snapshots), encoding="utf-8")
        return self.task_dir


def write_manifest(run_dir: Path, *, phase: str = "practice", config: str = "default_lib",
                   rep: int = 1, tasks: list[str] | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "schema": "run/1", "run_id": run_dir.name, "phase": phase, "config": config,
        "rep": rep, "tasks": tasks or []}), encoding="utf-8")
    return run_dir
