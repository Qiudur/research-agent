"""Append-only experiment tree using the published node/1 format."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA = "node/1"
STATUSES = {"queued", "running", "succeeded", "accepted", "rejected", "failed"}
TRANSITIONS = {
    "queued": {"running"},
    "running": {"succeeded", "failed"},
    "succeeded": {"accepted", "rejected"},
    "accepted": set(),
    "rejected": set(),
    "failed": set(),
}
ORIGINS = {"root", "plan", "explore", "diagnose", "draft"}
REQUIRED_FIELDS = {
    "schema", "run_id", "task_id", "node_id", "parent", "created_ts",
    "updated_ts", "hypothesis", "rationale", "origin", "cites", "estimate",
    "priority", "status", "attempts", "failure", "score", "cost",
    "seconds", "library_version", "calls", "code_dir",
}
IMMUTABLE_FIELDS = {
    "schema", "run_id", "task_id", "node_id", "parent", "created_ts",
    "hypothesis", "rationale", "origin", "cites", "estimate", "priority",
    "library_version", "code_dir",
}


class TreeError(ValueError):
    """The persisted tree or requested transition is invalid."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump()
    if callable(getattr(value, "_asdict", None)):
        return dict(value._asdict())
    if hasattr(value, "__dict__"):
        return vars(value).copy()
    raise TreeError(f"Expected a mapping-like result, got {type(value).__name__}")


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TreeError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise TreeError(f"{name} must be finite")
    return number


def _compact_score(result: Any) -> dict[str, Any]:
    source = _mapping(result)
    if source.get("schema", "score/1") != "score/1":
        raise TreeError("Unsupported score schema")
    if source.get("split", "dev") != "dev":
        raise TreeError("Experiment nodes may only contain dev scores")
    if "dev" in source:
        dev = source["dev"]
    else:
        dev = source.get("score")
    if dev is None:
        raise TreeError("A successful dev score must be numeric")
    spread = _finite(source.get("spread", 0), "score.spread")
    if spread < 0:
        raise TreeError("score.spread cannot be negative")
    return {
        "dev": _finite(dev, "score.dev"),
        "raw": source.get("raw"),
        "spread": spread,
        "correct": bool(source.get("correct", True)),
        "cheat_flags": list(source.get("cheat_flags") or []),
    }


class ExperimentTree:
    """One task's node snapshots; the last snapshot of each id is authoritative."""

    def __init__(self, run_dir: Path, task_id: str):
        if not re.fullmatch(r"(?:tab|spd)-[a-z0-9]+(?:-[a-z0-9]+)*", task_id):
            raise TreeError("Task id must follow the published tab-/spd- format")
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.task_id = task_id
        self.task_dir = self.run_dir / "tasks" / task_id
        self.path = self.task_dir / "tree.jsonl"
        self.nodes: dict[str, dict[str, Any]] = {}
        self.reload()

    @classmethod
    def open(cls, run_dir: Path, task_id: str) -> "ExperimentTree":
        return cls(run_dir, task_id)

    def reload(self) -> None:
        self.nodes = {}
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    node = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TreeError(f"Invalid JSON at {self.path}:{line_number}") from exc
                if not isinstance(node, dict) or node.get("schema") != SCHEMA:
                    raise TreeError(f"Unknown node schema at {self.path}:{line_number}")
                if REQUIRED_FIELDS - node.keys():
                    raise TreeError(f"Incomplete node at {self.path}:{line_number}")
                if node["run_id"] != self.run_id or node["task_id"] != self.task_id:
                    raise TreeError(f"Node belongs to another run or task at line {line_number}")
                if node["status"] not in STATUSES:
                    raise TreeError(f"Unknown status at line {line_number}")
                self.nodes[node["node_id"]] = node

    def _append(self, node: dict[str, Any]) -> dict[str, Any]:
        if set(node) != REQUIRED_FIELDS:
            raise TreeError(f"node/1 fields mismatch: {set(node) ^ REQUIRED_FIELDS}")
        if node["status"] not in STATUSES or node["origin"] not in ORIGINS:
            raise TreeError("Unknown node status or origin")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(node, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.nodes[node["node_id"]] = deepcopy(node)
        return deepcopy(node)

    def get(self, node_id: str) -> dict[str, Any]:
        try:
            return deepcopy(self.nodes[node_id])
        except KeyError as exc:
            raise TreeError(f"Unknown node {node_id}") from exc

    def all_nodes(self) -> list[dict[str, Any]]:
        return [self.get(node_id) for node_id in sorted(self.nodes)]

    def best_node(self) -> dict[str, Any]:
        accepted = [node for node in self.nodes.values() if node["status"] == "accepted"]
        if not accepted:
            raise TreeError("Create the accepted n0000 root first")
        return deepcopy(max(accepted, key=lambda node: node["score"]["dev"]))

    def create_root(
        self,
        score: Any,
        library_version: str,
        *,
        hypothesis: str = "Baseline",
        code_dir: str = "nodes/n0000",
    ) -> dict[str, Any]:
        if self.nodes:
            raise TreeError("The tree already has nodes")
        path = Path(code_dir)
        if path.is_absolute() or len(path.parts) < 2 or path.parts[:1] != ("nodes",) or ".." in path.parts:
            raise TreeError("Root code_dir must be inside the task's nodes directory")
        now = _timestamp()
        node = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "node_id": "n0000",
            "parent": None,
            "created_ts": now,
            "updated_ts": now,
            "hypothesis": hypothesis,
            "rationale": "Task baseline",
            "origin": "root",
            "cites": [],
            "estimate": {"gain": 0.0, "cost": 0.0, "seconds": 0.0},
            "priority": 0.0,
            "status": "accepted",
            "attempts": 0,
            "failure": None,
            "score": _compact_score(score),
            "cost": 0.0,
            "seconds": {"llm": 0.0, "run": 0.0, "queue": 0.0},
            "library_version": library_version,
            "calls": [],
            "code_dir": code_dir,
        }
        (self.task_dir / code_dir).mkdir(parents=True, exist_ok=True)
        return self._append(node)

    def enqueue(
        self,
        *,
        parent: str,
        hypothesis: str,
        rationale: str,
        origin: str,
        cites: list[str],
        estimate: Mapping[str, float],
        priority: float,
        library_version: str,
    ) -> dict[str, Any]:
        if "n0000" not in self.nodes:
            raise TreeError("Create root before enqueuing nodes")
        if parent not in self.nodes or self.nodes[parent]["status"] != "accepted":
            raise TreeError("Parent must be an accepted node")
        if origin not in ORIGINS - {"root"}:
            raise TreeError("Invalid non-root origin")
        if not hypothesis.strip():
            raise TreeError("Hypothesis cannot be empty")
        required = {"gain", "cost", "seconds"}
        if required - estimate.keys():
            raise TreeError("Estimate needs gain, cost, and seconds")
        clean_estimate = {key: _finite(estimate[key], f"estimate.{key}") for key in required}
        if clean_estimate["cost"] < 0 or clean_estimate["seconds"] < 0:
            raise TreeError("Estimated cost and seconds cannot be negative")
        index = max(int(node_id[1:]) for node_id in self.nodes) + 1
        if index > 9999:
            raise TreeError("node/1 permits at most 10,000 task nodes")
        node_id = f"n{index:04d}"
        now = _timestamp()
        code_dir = f"nodes/{node_id}"
        node = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "node_id": node_id,
            "parent": parent,
            "created_ts": now,
            "updated_ts": now,
            "hypothesis": hypothesis,
            "rationale": rationale,
            "origin": origin,
            "cites": list(cites),
            "estimate": {"gain": clean_estimate["gain"], "cost": clean_estimate["cost"], "seconds": clean_estimate["seconds"]},
            "priority": _finite(priority, "priority"),
            "status": "queued",
            "attempts": 0,
            "failure": None,
            "score": None,
            "cost": 0.0,
            "seconds": {"llm": 0.0, "run": 0.0, "queue": 0.0},
            "library_version": library_version,
            "calls": [],
            "code_dir": code_dir,
        }
        (self.task_dir / code_dir).mkdir(parents=True, exist_ok=True)
        return self._append(node)

    def patch(self, node_id: str, **fields: Any) -> dict[str, Any]:
        old = self.get(node_id)
        if IMMUTABLE_FIELDS & fields.keys():
            raise TreeError("Cannot change immutable node fields")
        if fields.keys() - REQUIRED_FIELDS:
            raise TreeError("Cannot add non-schema node fields")
        next_status = fields.get("status", old["status"])
        if next_status != old["status"] and next_status not in TRANSITIONS[old["status"]]:
            raise TreeError(f"Invalid transition {old['status']} -> {next_status}")
        node = {**old, **deepcopy(fields), "updated_ts": _timestamp()}
        if node["attempts"] < old["attempts"] or node["attempts"] < 0:
            raise TreeError("Attempt count cannot decrease")
        if _finite(node["cost"], "cost") < _finite(old["cost"], "cost"):
            raise TreeError("Cost cannot decrease")
        return self._append(node)

    def start_attempt(self, node_id: str, *, max_attempts: int = 3) -> dict[str, Any]:
        node = self.get(node_id)
        if node["status"] not in {"queued", "running"}:
            raise TreeError("Only queued or running nodes can start an attempt")
        if node["attempts"] >= max_attempts:
            raise TreeError("Attempt limit reached")
        return self.patch(node_id, status="running", attempts=node["attempts"] + 1)

    def succeed(self, node_id: str, score: Any | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {"status": "succeeded"}
        if score is not None:
            fields["score"] = _compact_score(score)
        return self.patch(node_id, **fields)

    def fail(self, node_id: str, kind: str, summary: str) -> dict[str, Any]:
        return self.patch(node_id, status="failed", failure={"kind": kind, "summary": summary})

    def judge(self, node_id: str, score_result: Any) -> dict[str, Any]:
        """Accept only an improvement larger than both dev score spreads."""
        candidate = self.get(node_id)
        if candidate["status"] != "succeeded":
            raise TreeError("Only succeeded nodes can be judged")
        result = _mapping(score_result)
        score = _compact_score(result)
        best = self.best_node()
        improvement = score["dev"] - best["score"]["dev"]
        noise = max(score["spread"], best["score"]["spread"])
        accepted = bool(result.get("ok", True)) and score["correct"] and not score["cheat_flags"] and improvement > noise
        return self.patch(node_id, status="accepted" if accepted else "rejected", score=score)
