"""Versioned experience library shared by the research agent and future skill layer.

The public API follows docs/接口约定.md section 4. Entries are plain dictionaries so
the JSON representation remains the single source of truth for external consumers.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

Entry: TypeAlias = dict[str, Any]

_ENTRY_ID = re.compile(r"^(exp|skl|tool)-\d{4,}$")
_VERSION = re.compile(r"^lib-(0|[1-9]\d*)$")
_KINDS = {"experience", "skill", "tool"}
_SCENARIOS = {"tabular", "speedup", "any"}
_OPS = {"write", "update", "narrow", "promote", "demote", "retire", "merge"}
_ACTORS = {"online", "self_research", "human"}
_STATUSES = {"active", "demoted", "retired"}


class LibraryError(Exception):
    """An invalid or inconsistent library operation."""


class LibraryFrozen(LibraryError):
    """Raised when an operation would change a frozen or historical library."""


class LibraryConflict(LibraryError):
    """Raised when another writer advanced the library after it was opened."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_json(path: Path, schema: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema") != schema:
        raise LibraryError(f"Unexpected schema in {path}: expected {schema}")
    return data


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _settings_path(root: Path, env_name: str, filename: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override).expanduser().resolve()
    for ancestor in (root, *root.parents):
        candidate = ancestor / "config" / filename
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[1] / "config" / filename


def _validate_entry(entry: Entry, triggers: dict[str, Any]) -> None:
    required = {
        "schema", "id", "kind", "scope", "scenario", "triggers", "conditions",
        "action", "negative", "effect", "credibility", "status", "sources",
        "created_version", "updated_version",
    }
    missing = required - entry.keys()
    if missing:
        raise LibraryError(f"Entry missing fields: {sorted(missing)}")
    if entry["schema"] != "entry/1":
        raise LibraryError("Entry schema must be entry/1")
    if not isinstance(entry["id"], str) or not _ENTRY_ID.fullmatch(entry["id"]):
        raise LibraryError("Invalid entry ID")
    if entry["kind"] not in _KINDS or entry["id"].split("-")[0] != {
        "experience": "exp", "skill": "skl", "tool": "tool"
    }[entry["kind"]]:
        raise LibraryError("Entry ID prefix and kind disagree")
    if entry["scope"] not in {"domain", "method"}:
        raise LibraryError("Invalid entry scope")
    if entry["scenario"] not in _SCENARIOS or entry["status"] not in _STATUSES:
        raise LibraryError("Invalid entry scenario or status")
    if not isinstance(entry["triggers"], list) or not all(isinstance(x, str) for x in entry["triggers"]):
        raise LibraryError("Entry triggers must be a string list")
    allowed = set(triggers["tabular"] + triggers["speedup"])
    if entry["scenario"] != "any":
        allowed = set(triggers[entry["scenario"]])
    if any(item not in allowed for item in entry["triggers"]):
        raise LibraryError("Entry contains an unknown or out-of-scenario trigger")
    if not isinstance(entry["conditions"], str) or not isinstance(entry["action"], str):
        raise LibraryError("Entry conditions and action must be strings")
    if not isinstance(entry["negative"], bool):
        raise LibraryError("Entry negative must be boolean")
    if not isinstance(entry["credibility"], (int, float)) or not 0 <= entry["credibility"] <= 1:
        raise LibraryError("Entry credibility must be between 0 and 1")
    effect = entry["effect"]
    if not isinstance(effect, dict) or not {"uses", "wins", "mean_gain", "mean_cost"} <= effect.keys():
        raise LibraryError("Entry effect has missing fields")
    if not isinstance(effect["uses"], int) or not isinstance(effect["wins"], int):
        raise LibraryError("Entry use counts must be integers")
    if not 0 <= effect["wins"] <= effect["uses"]:
        raise LibraryError("Entry use counts are inconsistent")
    if not isinstance(effect["mean_gain"], (int, float)) or not isinstance(effect["mean_cost"], (int, float)):
        raise LibraryError("Entry means must be numeric")
    if effect["mean_cost"] < 0:
        raise LibraryError("Entry mean cost cannot be negative")
    if not isinstance(entry["sources"], list):
        raise LibraryError("Entry sources must be a list")
    for key in ("created_version", "updated_version"):
        if not isinstance(entry[key], str) or not _VERSION.fullmatch(entry[key]):
            raise LibraryError(f"Invalid {key}")
    if entry["kind"] == "skill" and not all(isinstance(entry.get(key), str) for key in ("summary", "body_path")):
        raise LibraryError("Skill entries need summary and body_path")
    if entry["kind"] == "tool" and not all(isinstance(entry.get(key), str) for key in ("module", "function", "cases_path", "from_skill")):
        raise LibraryError("Tool entries need module, function, cases_path and from_skill")


class Library:
    """A single-run library with immutable version snapshots and an append-only log."""

    def __init__(self, root: Path, state: dict[str, Any], frozen: bool = False):
        self.root = root
        self._state = copy.deepcopy(state)
        self.frozen = frozen
        agent_config = _read_json(_settings_path(root, "RA_AGENT_CONFIG", "agent.json"), "agent/1")
        self.rules = agent_config["library_rules"]
        self._triggers = _read_json(_settings_path(root, "RA_TRIGGERS_CONFIG", "triggers.json"), "triggers/1")
        if state.get("schema") != "library/1" or not _VERSION.fullmatch(str(state.get("version"))):
            raise LibraryError("Invalid library state schema or version")
        if not isinstance(state.get("entries"), list) or not isinstance(state.get("metadata"), dict):
            raise LibraryError("Invalid library state content")
        self._entries: dict[str, Entry] = {}
        for entry in state["entries"]:
            _validate_entry(entry, self._triggers)
            if entry["id"] in self._entries:
                raise LibraryError(f"Duplicate entry ID: {entry['id']}")
            self._entries[entry["id"]] = copy.deepcopy(entry)

    @classmethod
    def open(cls, root: Path, version: str | None = None, frozen: bool = False) -> "Library":
        root = Path(root).expanduser().resolve()
        state_path = root / "state.json"
        snapshots = root / "snapshots"
        if not state_path.exists():
            if version is not None or frozen:
                raise FileNotFoundError(f"Library not initialized: {root}")
            state = {"schema": "library/1", "version": "lib-0", "entries": [], "metadata": {"demoted_losses": {}}}
            _atomic_json(snapshots / "lib-0.json", state)
            _atomic_json(state_path, state)
        state = _read_json(state_path, "library/1")
        log = root / "changelog.jsonl"
        if log.exists():
            lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                last = json.loads(lines[-1])
                if last.get("schema") != "libchange/1":
                    raise LibraryError("Invalid library changelog schema")
                log_version = last["version"]
                if int(log_version[4:]) > int(state["version"][4:]):
                    state = _read_json(snapshots / f"{log_version}.json", "library/1")
                    _atomic_json(state_path, state)
                elif int(log_version[4:]) < int(state["version"][4:]):
                    raise LibraryError("Library state is ahead of changelog")
        current_snapshot = _read_json(snapshots / f"{state['version']}.json", "library/1")
        if state != current_snapshot:
            raise LibraryError("Library state differs from its immutable version snapshot")
        if version is not None:
            if not _VERSION.fullmatch(version):
                raise ValueError("Invalid library version")
            state = _read_json(snapshots / f"{version}.json", "library/1")
            if state["version"] != version:
                raise LibraryError("Snapshot version mismatch")
            frozen = True
        return cls(root, state, frozen=frozen)

    @property
    def version(self) -> str:
        return self._state["version"]

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(copy.deepcopy(entry) for entry in self._entries.values())

    def get(self, entry_id: str) -> Entry:
        return copy.deepcopy(self._entries[entry_id])

    def query(
        self, scenario: str, features: list[str], k: int = 8,
        kinds: tuple = ("experience", "skill"),
    ) -> list[Entry]:
        if scenario not in {"tabular", "speedup"}:
            raise ValueError("scenario must be tabular or speedup")
        if not isinstance(features, list) or any(x not in self._triggers[scenario] for x in features):
            raise ValueError("features contain an unknown trigger")
        if k < 0:
            raise ValueError("k must be nonnegative")
        feature_set = set(features)
        candidates = [entry for entry in self._entries.values()
                      if entry["kind"] in kinds and entry["status"] == "active"
                      and entry["scenario"] in (scenario, "any")
                      and (not entry["triggers"] or feature_set.intersection(entry["triggers"]))]
        candidates.sort(key=lambda item: (
            -len(feature_set.intersection(item["triggers"])),
            -float(item["credibility"]),
            -float(item["effect"]["mean_gain"]),
            item["id"],
        ))
        return copy.deepcopy(candidates[:k])

    def brief(self, entries: list[Entry]) -> str:
        lines = []
        for item in entries:
            entry = self.get(item["id"])
            conditions = entry["conditions"].strip().replace("\n", " ")
            if entry["kind"] == "skill":
                detail = entry["summary"].strip().replace("\n", " ")
            else:
                detail = entry["action"].strip().replace("\n", " ")
            prefix = "避免" if entry["negative"] else "可试"
            suffix = f"；适用条件：{conditions}" if conditions else ""
            lines.append(f"[{entry['id']}] {prefix}：{detail}{suffix}")
        return "\n".join(lines)

    def load_skill(self, skill_id: str) -> str:
        entry = self.get(skill_id)
        if entry["kind"] != "skill":
            raise ValueError("Entry is not a skill")
        body_path = (self.root / entry["body_path"]).resolve()
        if not body_path.is_relative_to(self.root) or not body_path.is_file():
            raise LibraryError("Invalid skill body path")
        return body_path.read_text(encoding="utf-8")

    def put(
        self, entry: Entry, *, op: str, reason: str, evidence: list, actor: str,
    ) -> str:
        return self._put_internal(entry, op=op, reason=reason, evidence=evidence, actor=actor)

    def _put_internal(
        self, entry: Entry, *, op: str, reason: str, evidence: list, actor: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if self.frozen:
            raise LibraryFrozen("Library is frozen")
        if op not in _OPS or actor not in _ACTORS or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Invalid library operation, actor, or reason")
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        current = _read_json(self.root / "state.json", "library/1")
        if current["version"] != self.version:
            raise LibraryConflict(f"Library advanced to {current['version']}; reopen before writing")
        candidate = copy.deepcopy(entry)
        entry_id = candidate.get("id")
        old = self._entries.get(entry_id)
        if old is None and op not in {"write", "promote"}:
            raise LibraryError(f"Cannot {op} missing entry {entry_id}")
        if old is not None and op in {"write", "promote"}:
            raise LibraryError(f"Entry already exists: {entry_id}")
        next_version = f"lib-{int(self.version[4:]) + 1}"
        candidate["created_version"] = old["created_version"] if old else next_version
        candidate["updated_version"] = next_version
        _validate_entry(candidate, self._triggers)
        new_state = copy.deepcopy(self._state)
        by_id = {item["id"]: item for item in new_state["entries"]}
        by_id[entry_id] = candidate
        new_state["entries"] = sorted(by_id.values(), key=lambda item: item["id"])
        new_state["version"] = next_version
        if metadata is not None:
            new_state["metadata"] = copy.deepcopy(metadata)
        before = {} if old is None else {key: copy.deepcopy(value) for key, value in old.items()
                                       if candidate.get(key) != value}
        after = {key: copy.deepcopy(value) for key, value in candidate.items()
                 if old is None or old.get(key) != value}
        change = {
            "schema": "libchange/1", "version": next_version, "ts": _utc_now(),
            "op": op, "entry_id": entry_id, "before": before, "after": after,
            "reason": reason, "evidence": copy.deepcopy(evidence), "actor": actor,
        }
        _atomic_json(self.root / "snapshots" / f"{next_version}.json", new_state)
        _append_jsonl(self.root / "changelog.jsonl", change)
        _atomic_json(self.root / "state.json", new_state)
        self._state = new_state
        self._entries[entry_id] = copy.deepcopy(candidate)
        return next_version

    def record_use(
        self, entry_id: str, *, run_id: str, task_id: str, node_id: str,
        accepted: bool, gain: float, cost: float,
    ) -> None:
        if self.frozen:
            raise LibraryFrozen("Frozen libraries record uses in the ledger, not state")
        if not isinstance(accepted, bool) or cost < 0:
            raise ValueError("Invalid outcome or cost")
        entry = self.get(entry_id)
        if entry["status"] == "retired":
            raise LibraryError("Cannot record a new use of a retired entry")
        effect = entry["effect"]
        previous_uses = effect["uses"]
        effect["uses"] += 1
        effect["wins"] += int(accepted)
        effect["mean_gain"] = (effect["mean_gain"] * previous_uses + gain) / effect["uses"]
        effect["mean_cost"] = (effect["mean_cost"] * previous_uses + cost) / effect["uses"]
        delta = self.rules["win_delta"] if accepted else self.rules["loss_delta"]
        entry["credibility"] = min(1.0, max(0.0, entry["credibility"] + delta))
        source = {"run_id": run_id, "task_id": task_id, "node_ids": [node_id]}
        if source not in entry["sources"]:
            entry["sources"].append(source)
        metadata = copy.deepcopy(self._state["metadata"])
        losses = metadata.setdefault("demoted_losses", {})
        if entry["status"] == "demoted":
            losses[entry_id] = 0 if accepted else losses.get(entry_id, 0) + 1
            if accepted and entry["credibility"] >= self.rules["demote_below"]:
                entry["status"] = "active"
            elif losses[entry_id] >= self.rules["retire_after_losses"]:
                entry["status"] = "retired"
        elif entry["status"] == "active" and entry["credibility"] < self.rules["demote_below"]:
            entry["status"] = "demoted"
            losses[entry_id] = 0
        self._put_internal(
            entry, op="retire" if entry["status"] == "retired" else
            "demote" if entry["status"] == "demoted" else "update",
            reason="Outcome of cited experiment", evidence=[source], actor="online", metadata=metadata,
        )

    def write_experiences(self, drafts: list[dict], *, source: dict) -> list[str]:
        if self.frozen:
            raise LibraryFrozen("Library is frozen")
        if not isinstance(drafts, list) or not isinstance(source, dict):
            raise ValueError("drafts must be a list and source a dict")
        if not {"run_id", "task_id", "node_ids"} <= source.keys():
            raise ValueError("source must include run_id, task_id and node_ids")
        ids: list[str] = []
        for draft in drafts:
            if not isinstance(draft, dict):
                raise ValueError("Each experience draft must be a dict")
            existing = next((entry for entry in self._entries.values() if
                             entry["kind"] == "experience" and entry["status"] != "retired" and
                             all(entry[key] == draft.get(key, default) for key, default in (
                                 ("scope", "domain"), ("scenario", None), ("triggers", []),
                                 ("conditions", ""), ("action", ""), ("negative", False),
                             ))), None)
            if existing:
                ids.append(existing["id"])
                if source not in existing["sources"]:
                    merged = copy.deepcopy(existing)
                    merged["sources"].append(copy.deepcopy(source))
                    self.put(merged, op="merge", reason="Exact duplicate experience; merge evidence",
                             evidence=[source], actor="online")
                continue
            number = max((int(entry_id[4:]) for entry_id in self._entries if entry_id.startswith("exp-")), default=0) + 1
            entry: Entry = {
                "schema": "entry/1", "id": f"exp-{number:04d}", "kind": "experience",
                "scope": draft.get("scope", "domain"), "scenario": draft.get("scenario"),
                "triggers": copy.deepcopy(draft.get("triggers", [])),
                "conditions": draft.get("conditions", ""), "action": draft.get("action", ""),
                "negative": draft.get("negative", False),
                "effect": copy.deepcopy(draft.get("effect", {"uses": 0, "wins": 0, "mean_gain": 0.0, "mean_cost": 0.0})),
                "credibility": draft.get("credibility", self.rules["init_credibility"]),
                "status": "active", "sources": [copy.deepcopy(source)],
            }
            self.put(entry, op="write", reason="Task-end summary experience", evidence=[source], actor="online")
            ids.append(entry["id"])
            self._enforce_experience_limit()
        return ids

    def _enforce_experience_limit(self) -> None:
        cap = int(self.rules["max_experiences"])
        active = [entry for entry in self._entries.values()
                  if entry["kind"] == "experience" and entry["status"] == "active"]
        if len(active) > cap:
            victim = min(active, key=lambda item: (item["credibility"], item["id"]))
            retired = copy.deepcopy(victim)
            retired["status"] = "retired"
            self.put(retired, op="retire", reason="Experience cap reached; retire lowest credibility",
                     evidence=[], actor="online")

    def narrow(self, entry_id: str, conditions: str, *, reason: str, evidence: list) -> None:
        if self.frozen:
            raise LibraryFrozen("Library is frozen")
        if not isinstance(conditions, str) or not conditions.strip():
            raise ValueError("conditions must be nonempty")
        entry = self.get(entry_id)
        if entry["conditions"] == conditions:
            return
        entry["conditions"] = conditions
        self.put(entry, op="narrow", reason=reason, evidence=evidence, actor="online")

    def changelog(self, since: str | None = None) -> list[dict]:
        if since is not None and not _VERSION.fullmatch(since):
            raise ValueError("Invalid library version")
        path = self.root / "changelog.jsonl"
        if not path.exists():
            return []
        minimum = int(since[4:]) if since is not None else -1
        changes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            change = json.loads(line)
            if change.get("schema") != "libchange/1":
                raise LibraryError("Unknown library changelog schema")
            if int(change["version"][4:]) > minimum:
                changes.append(change)
        return changes


__all__ = ["Entry", "Library", "LibraryConflict", "LibraryError", "LibraryFrozen"]
