"""Pilot event recorder; no model calls and no simulated monetary charges."""

from __future__ import annotations

from pathlib import Path

from research_agent.io import append_jsonl, utc_now


class PilotLedger:
    def __init__(self, run_dir: Path, task_id: str):
        self.run_dir = Path(run_dir)
        self.task_id = task_id
        self.path = self.run_dir / "tasks" / task_id / "ledger.jsonl"
        self.sequence = 0
        self.events: list[dict] = []

    def emit(self, kind: str, *, node_id: str | None = None, **fields) -> None:
        self.sequence += 1
        event = {
            "schema": "ledger/1", "seq": self.sequence, "ts": utc_now(),
            "run_id": self.run_dir.name, "task_id": self.task_id,
            "node_id": node_id, "kind": kind, **fields,
        }
        append_jsonl(self.path, event)
        self.events.append(event)

    def llm_call(self, **fields):
        raise RuntimeError("This pilot does not have a live model or real price table")

    def spent(self) -> float:
        return 0.0

    def elapsed(self) -> dict[str, float]:
        return {
            "llm": 0.0,
            "run": sum(float(e.get("seconds_run", 0)) for e in self.events if e["kind"] == "experiment"),
            "queue": sum(float(e.get("seconds_queue", 0)) for e in self.events if e["kind"] == "score"),
            "wall": 0.0,
        }
