"""Evidence-gated promotion from experiences to on-demand skills.

This is an explicit draft workflow, not autonomous skill generation. A skill
requires two distinct task IDs with accepted, cited experiments that can still
be verified against their saved trees and library changelog.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile

from library import Library, LibraryError, LibraryFrozen
from research_agent.tree import ExperimentTree, TreeError

_RUN = re.compile(r"^run-[a-zA-Z0-9-]+$")
_TASK = re.compile(r"^(?:tab|spd)-[a-z0-9]+(?:-[a-z0-9]+)*$")
_NODE = re.compile(r"^n[0-9]{4}$")


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _saved_win(lib: Library, exp_id: str, source: dict) -> bool:
    run_id, task_id = source.get("run_id"), source.get("task_id")
    node_ids = source.get("node_ids")
    if (not isinstance(run_id, str) or not _RUN.fullmatch(run_id)
            or not isinstance(task_id, str) or not _TASK.fullmatch(task_id)
            or not isinstance(node_ids, list) or len(node_ids) != 1
            or not isinstance(node_ids[0], str) or not _NODE.fullmatch(node_ids[0])):
        return False
    runs_root = lib.root.parent.parent
    run_dir = (runs_root / run_id).resolve()
    if not run_dir.is_relative_to(runs_root.resolve()):
        return False
    try:
        tree = ExperimentTree.open(run_dir, task_id)
        node = tree.get(node_ids[0])
        parent = tree.get(node["parent"])
        score, previous = node["score"], parent["score"]
        historical = Library.open(lib.root, version=node["library_version"])
        historical_entry = historical.get(exp_id)
    except (FileNotFoundError, TreeError, KeyError, TypeError, ValueError, LibraryError):
        return False
    scenario = historical_entry["scenario"]
    if ((scenario == "tabular" and not task_id.startswith("tab-"))
            or (scenario == "speedup" and not task_id.startswith("spd-"))):
        return False
    if (historical_entry["status"] != "active" or node["status"] != "accepted"
            or exp_id not in node["cites"]
            or score is None or previous is None or not score.get("correct")
            or score.get("cheat_flags")):
        return False
    current = _finite(score.get("dev"))
    earlier = _finite(previous.get("dev"))
    spread = _finite(score.get("spread"))
    prior_spread = _finite(previous.get("spread"))
    return (current is not None and earlier is not None and spread is not None
            and prior_spread is not None and current - earlier > max(spread, prior_spread))


def verified_wins(lib: Library, exp_id: str) -> list[dict]:
    """Return independently evidenced accepted uses, excluding write-only sources."""
    entry = lib.get(exp_id)
    if entry["kind"] != "experience":
        raise ValueError("verified_wins expects an experience")
    wins = []
    seen = set()
    for change in lib.changelog():
        if (change.get("entry_id") != exp_id or change.get("actor") != "online"
                or change.get("reason") != "Outcome of cited experiment"):
            continue
        before = change.get("before", {}).get("effect", {})
        after = change.get("after", {}).get("effect", {})
        if (after.get("uses") != before.get("uses", -10) + 1
                or after.get("wins") != before.get("wins", -10) + 1):
            continue
        for source in change.get("evidence", []):
            if not isinstance(source, dict) or source not in entry["sources"]:
                continue
            node_ids = source.get("node_ids") or []
            key = (source.get("run_id"), source.get("task_id"), tuple(node_ids))
            if key not in seen and _saved_win(lib, exp_id, source):
                wins.append(deepcopy(source))
                seen.add(key)
    return wins


def _eligible(lib: Library, exp_id: str) -> tuple[dict, list[dict]] | None:
    entry = lib.get(exp_id)
    if entry["kind"] != "experience" or entry["status"] != "active" or entry["negative"]:
        return None
    threshold = float(lib.rules["skill_min_credibility"])
    minimum = int(lib.rules["skill_min_tasks"])
    if entry["credibility"] + 1e-12 < threshold:
        return None
    wins = verified_wins(lib, exp_id)
    if len({source["task_id"] for source in wins}) < minimum:
        return None
    return entry, wins


def promote_candidates(lib: Library) -> list[list[str]]:
    """List eligible, not-yet-promoted experiences as conservative singletons."""
    promoted = {source_id for item in lib.entries if item["kind"] == "skill"
                for source_id in item.get("from_experiences", [])}
    groups = []
    for entry in sorted(lib.entries, key=lambda item: item["id"]):
        if entry["kind"] == "experience" and entry["id"] not in promoted and _eligible(lib, entry["id"]):
            groups.append([entry["id"]])
    return groups


def promote_to_skill(lib: Library, exp_ids: list[str], body: str, summary: str) -> str:
    """Create a versioned skill from explicitly supplied, qualified drafts."""
    if lib.frozen:
        raise LibraryFrozen("Cannot promote with a frozen library")
    if (not isinstance(exp_ids, list) or not exp_ids or len(exp_ids) != len(set(exp_ids))
            or not all(isinstance(value, str) for value in exp_ids)):
        raise ValueError("exp_ids must be a nonempty list of unique IDs")
    if not isinstance(body, str) or not body.strip() or not isinstance(summary, str) or not summary.strip():
        raise ValueError("Skill body and summary must be nonempty text")
    if len(summary.strip()) > 240:
        raise ValueError("Skill summary must fit in 240 characters")
    already_promoted = {value for item in lib.entries if item["kind"] == "skill"
                        for value in item.get("from_experiences", [])}
    if already_promoted.intersection(exp_ids):
        raise LibraryError("Experience has already been promoted")
    checked = [_eligible(lib, exp_id) for exp_id in exp_ids]
    if any(item is None for item in checked):
        raise LibraryError("Each experience needs verified wins on distinct tasks and sufficient credibility")
    entries = [item[0] for item in checked]
    compatibility = lambda entry: (entry["scope"], entry["scenario"], tuple(entry["triggers"]),
                                   entry["conditions"], entry["action"], entry["negative"])
    if any(compatibility(entry) != compatibility(entries[0]) for entry in entries[1:]):
        raise LibraryError("A skill cannot combine incompatible experiences")
    cap = int(lib.rules["max_skills"])
    if cap < 1:
        raise LibraryError("Skill capacity is zero")
    number = max((int(item["id"][4:]) for item in lib.entries if item["kind"] == "skill"), default=0) + 1
    skill_id = f"skl-{number:04d}"
    body_path = Path("skills") / f"{skill_id}.md"
    target = lib.root / body_path
    if target.exists():
        raise LibraryError("Skill body path already exists")
    wins = [source for _, sources in checked for source in sources]
    evidence = []
    for source in wins:
        if source not in evidence:
            evidence.append(deepcopy(source))
    source_refs = []
    for entry in entries:
        for source in entry["sources"]:
            if source not in source_refs:
                source_refs.append(deepcopy(source))
    effects = [entry["effect"] for entry in entries]
    uses = sum(effect["uses"] for effect in effects)
    skill = {
        **deepcopy(entries[0]), "id": skill_id, "kind": "skill",
        "summary": summary.strip(), "body_path": str(body_path),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "from_experiences": sorted(exp_ids), "verified_wins": evidence,
        "credibility": min(entry["credibility"] for entry in entries),
        "effect": {"uses": uses, "wins": sum(effect["wins"] for effect in effects),
                   "mean_gain": sum(effect["mean_gain"] * effect["uses"] for effect in effects) / uses,
                   "mean_cost": sum(effect["mean_cost"] * effect["uses"] for effect in effects) / uses},
        "sources": source_refs,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{skill_id}.", suffix=".tmp", delete=False) as handle:
            temp = handle.name
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        temp = None
        lib.put(skill, op="promote", reason="Verified accepted uses on distinct tasks",
                evidence=evidence, actor="human")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        if temp is not None:
            Path(temp).unlink(missing_ok=True)
    active = [entry for entry in lib.entries if entry["kind"] == "skill" and entry["status"] == "active"]
    if len(active) > cap:
        victim = min(active, key=lambda entry: (entry["credibility"], entry["id"]))
        retired = deepcopy(victim)
        retired["status"] = "retired"
        lib.put(retired, op="retire", reason="Skill capacity exceeded; retire lowest credibility",
                evidence=evidence, actor="online")
    return skill_id


def load_skill_verified(lib: Library, skill_id: str) -> str:
    """Load the body only after explicit selection, checking its recorded hash."""
    entry = lib.get(skill_id)
    if entry["kind"] != "skill" or entry["status"] != "active":
        raise LibraryError("Skill is unavailable")
    body = lib.load_skill(skill_id)
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != entry.get("body_sha256"):
        raise LibraryError("Skill body differs from the recorded SHA256")
    return body


def demote_skill(lib: Library, skill_id: str, *, reason: str, evidence: list) -> None:
    """Explicitly hide a skill; record_use still controls loss-based retirement."""
    if lib.frozen:
        raise LibraryFrozen("Cannot demote with a frozen library")
    if not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, list) or not evidence:
        raise ValueError("Demotion needs a reason and evidence")
    entry = lib.get(skill_id)
    if entry["kind"] != "skill":
        raise ValueError("Not a skill")
    if entry["status"] != "active":
        return
    entry["status"] = "demoted"
    lib.put(entry, op="demote", reason=reason, evidence=evidence, actor="human")


__all__ = ["promote_candidates", "promote_to_skill", "load_skill_verified",
           "demote_skill", "verified_wins"]
