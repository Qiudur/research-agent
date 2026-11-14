"""Practice-only configuration search with audited score/cost evidence.

This evaluates supplied or automatically proposed agent/1 and prompt
candidates. It never opens validation/exam task packages. There is no overall
research budget, but existing positive per-task budgets still apply formally.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
from threading import Lock
from typing import Any, Callable, Mapping

from library import Library
from ledger import Ledger, Usage, price_usage
from research_agent.model import ModelClient
from token_accounting import require_exact_token_accounting

from .batch import (_SerializedHarness, _agent_signature, _hash, _json, _run_path,
                    _verify_code_freeze, _within)
from .finalize import finish_run, start_run
from .manifest import create_manifest
from .metrics import task_metrics
from .orchestrate import execute_task
from .proposal import ProposalLedger
from .report import run_report


_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_ROLES = ("planner", "executor", "diagnoser", "summarizer")
_FORBIDDEN_NOTE = re.compile(
    r"hidden|holdout|"
    r"\bvalidation\b.{0,24}\b(?:sets?|data|tasks?|splits?|phases?|tiers?|files?)\b|"
    r"\b(?:access|read|load|peek|inspect|query)\b.{0,28}\bvalidation\b|"
    r"\bexam\b|hard_caps|prices[.]json|bank[.]json|"
    r"RA_EVAL|sandbox|ignore previous|bypass|override safety|"
    r"隐藏|验证集|考试|题库|价格表|硬上限|忽略.{0,12}规则|绕过", re.IGNORECASE)
_PROPOSAL_PROMPT = (
    "Tune a research agent using the supplied practice results only. Change "
    "bounded settings and add brief, positive task-facing instructions to "
    "the four existing role prompts. Keep each prompt_append value under "
    "160 characters, and give a one-sentence rationale under 200 characters. "
    "Do not discuss evaluation setup, access to other data, protected files, "
    "or rule changes in any generated text. Use per-task scores, costs, stop "
    "reasons and variability to target observed losses. Preserve the original "
    "prompts and safety rules. Settings state the complete target "
    "configuration; default_settings shows the current values. Reasoning "
    "effort has little effect on this model, so prefer the cost levers: "
    "stop_loss ends a task early (tab: stop after the first N completed "
    "experiments when none was accepted, N 1-4; spd: stop after K completed "
    "experiments in a row without an accepted one, K 2-8), hypotheses_per_plan "
    "caps the ideas per planning call (2-8), planner_mode lean shortens "
    "planning, role_controls can switch thinking off for a role and cap its "
    "max_output_tokens (512-2048), cost_priors seeds per-experiment token "
    "estimates (500-12000), library_brief limits ability-brief credibility "
    "(0.3-0.9) and length (100-600 characters), failure_feedback and "
    "ceiling_excludes_baseline are switches, and experience_writer chooses "
    "model or script. A value of 0 (or unchanged for thinking) leaves that "
    "optional setting unset; only use a lever when practice results show a "
    "loss it can reduce."
)

# Inclusive bounds of every tunable number. The optional levers (everything but
# the first three) additionally accept 0, and thinking accepts "unchanged",
# meaning the setting is not set.
_BOUNDS = {
    "plateau_nodes": (2, 12), "explore_every": (2, 8), "query_k": (1, 12),
    "first_n_no_gain": (1, 4), "consecutive_no_gain": (2, 8),
    "hypotheses_per_plan": (2, 8), "max_output_tokens": (512, 2048),
    "cost_prior": (500, 12000), "min_credibility": (0.3, 0.9), "max_chars": (100, 600),
}
_LEGACY_KEYS = ("plateau_nodes", "explore_every", "query_k", "behind_schedule", "effort")
_NEW_KEYS = ("stop_loss", "ceiling_excludes_baseline", "planner_mode", "hypotheses_per_plan",
             "failure_feedback", "cost_priors", "role_controls", "experience_writer",
             "library_brief")
_PLANNER_MODES = ("standard", "lean")
_EXPERIENCE_WRITERS = ("model", "script")
_THINKING = ("unchanged", "on", "off")
# Value a new key has when an older agent/1 config does not mention it.
_NEW_DEFAULTS = {"ceiling_excludes_baseline": False, "planner_mode": "standard",
                 "failure_feedback": False, "experience_writer": "model"}


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _optional(kind: str, key: str) -> dict:
    low, high = _BOUNDS[key]
    return {"type": kind, "description": f"0 leaves this unset; otherwise {low} to {high}"}


_PROPOSAL_SCHEMA = _object({
    "settings": _object({
        "plateau_nodes": {"type": "integer"},
        "explore_every": {"type": "integer"},
        "query_k": {"type": "integer"},
        "behind_schedule": _object({"spent_ratio": {"type": "number"},
                                    "progress_ratio": {"type": "number"}}),
        "effort": _object({role: {"type": "string", "enum": ["low", "medium", "high"]}
                           for role in _ROLES}),
        "stop_loss": _object({
            "tab": _object({"first_n_no_gain": _optional("integer", "first_n_no_gain")}),
            "spd": _object({"consecutive_no_gain": _optional("integer", "consecutive_no_gain")}),
        }),
        "ceiling_excludes_baseline": {"type": "boolean"},
        "planner_mode": {"type": "string", "enum": list(_PLANNER_MODES)},
        "hypotheses_per_plan": _optional("integer", "hypotheses_per_plan"),
        "failure_feedback": {"type": "boolean"},
        "cost_priors": _object({"spd": _optional("integer", "cost_prior"),
                                "tab": _optional("integer", "cost_prior")}),
        "role_controls": _object({role: _object({
            "thinking": {"type": "string", "enum": list(_THINKING)},
            "max_output_tokens": _optional("integer", "max_output_tokens"),
        }) for role in _ROLES}),
        "experience_writer": {"type": "string", "enum": list(_EXPERIENCE_WRITERS)},
        "library_brief": _object({"min_credibility": _optional("number", "min_credibility"),
                                  "max_chars": _optional("integer", "max_chars")}),
    }),
    "prompt_append": _object({role: {"type": "string",
                                    "description": "Positive task strategy, under 160 characters; no evaluation or access discussion"}
                              for role in _ROLES}),
    "rationale": {"type": "string",
                  "description": "One sentence under 200 characters about observed practice results"},
})


def _practice_tasks(bank: dict, harness: Any, *, formal: bool) -> list[str]:
    if bank.get("schema") != "bank/1" or not isinstance(bank.get("version"), str):
        raise ValueError("Expected versioned bank/1")
    if not isinstance(bank.get("tasks"), list):
        raise ValueError("Bank has no task list")
    if formal and not bank.get("frozen"):
        raise ValueError("Formal self-research needs a frozen bank")
    practice = []
    seen = set()
    counts = {"tabular": 0, "speedup": 0}
    for entry in bank["tasks"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid bank entry")
        if entry.get("tier") != "practice":
            continue  # No validation/exam task packages are opened here.
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or task_id in seen:
            raise ValueError("Duplicate or invalid practice task ID")
        seen.add(task_id)
        practice.append(task_id)
        if formal:
            task = harness.load_task(task_id)
            scenario = task.get("scenario")
            if scenario not in counts:
                raise ValueError(f"Invalid practice scenario for {task_id}")
            counts[scenario] += 1
            money = task.get("budget", {}).get("money")
            if (isinstance(money, bool) or not isinstance(money, (int, float))
                    or not math.isfinite(money) or money <= 0):
                raise ValueError(f"Formal practice task {task_id} needs a positive budget")
    if not practice or (formal and min(counts.values()) < 5):
        raise ValueError("Self-research needs practice tasks in both scenarios")
    return practice


def _candidate_files(root: Path,
                     candidates: Mapping[str, Path]) -> tuple[dict[str, Path], dict[str, dict]]:
    if not candidates or (len(candidates) > 1 and "default" not in candidates):
        raise ValueError("Multi-candidate search must include a default comparator")
    paths, descriptors = {}, {}
    fingerprints = set()
    for name, supplied in candidates.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("Invalid candidate name")
        path = _within(root, Path(supplied))
        if not (path.is_relative_to(root / "config") or path.is_relative_to(root / "batches")
                or path.is_relative_to(root / "runs")):
            raise ValueError("Candidate config must be in config/, batches/ or runs/")
        config = _json(path)
        if config.get("schema") != "agent/1" or not isinstance(config.get("prompts"), dict):
            raise ValueError(f"Invalid agent/1 candidate {name}")
        prompts = {}
        for role in _ROLES:
            relative = config["prompts"].get(role)
            if not isinstance(relative, str):
                raise ValueError(f"Missing {role} prompt in candidate {name}")
            prompt = _within(root, Path(relative))
            if not (prompt.is_relative_to(root / "config/prompts")
                    or prompt.is_relative_to(root / "batches")
                    or prompt.is_relative_to(root / "runs")):
                raise ValueError("Candidate prompt must be in config/prompts/, batches/ or runs/")
            if not prompt.read_text(encoding="utf-8").strip():
                raise ValueError(f"Empty prompt for candidate {name}: {role}")
            prompts[role] = {"path": str(prompt), "sha256": _hash(prompt)}
        descriptor = {"config": {"path": str(path), "sha256": _hash(path)},
                      "prompts": prompts}
        fingerprint = _agent_signature(root, config)
        if fingerprint in fingerprints:
            raise ValueError("Duplicate candidate content would waste a research run")
        fingerprints.add(fingerprint)
        paths[name] = path
        descriptors[name] = descriptor
    return paths, descriptors


def verify_selected(study_report: Mapping) -> Path:
    """Verify the selected candidate files still match their recorded hashes."""
    if study_report.get("schema") != "self_research/1":
        raise ValueError("Expected self_research/1")
    selected = study_report["selection"]
    descriptor = study_report["candidates"][selected["candidate"]]
    files = [descriptor["config"], *descriptor["prompts"].values()]
    for file in files:
        path = Path(file["path"])
        if _hash(path) != file["sha256"]:
            raise ValueError(f"Selected candidate file changed: {path}")
    return Path(descriptor["config"]["path"])


def _proposal_practice_breakdown(root: Path, evaluations: Mapping[str, dict],
                                 ranking: list[dict]) -> list[dict]:
    """Summarize only already-scored practice runs for bounded proposal input."""
    breakdown = []
    recent = ranking if len(ranking) <= 3 else [ranking[0], *ranking[-2:]]
    for candidate in recent:
        name = candidate["candidate"]
        by_task: dict[str, list[dict]] = {}
        for report in evaluations[name]["reports"]:
            run_dir = root / "runs" / report["run_id"]
            for task in report["tasks"]:
                events = Ledger.open(run_dir, task["task_id"])._events()
                stops = [event for event in events if event["kind"] == "stop"]
                starts = [event for event in events if event["kind"] == "task_start"]
                if len(stops) != 1 or len(starts) != 1:
                    raise ValueError("Practice task lacks one start and stop event")
                efficiency = task_metrics(run_dir / "tasks" / task["task_id"])
                by_task.setdefault(task["task_id"], []).append({
                    "tokens_to_reference": efficiency["tokens_to_reference"],
                    "tokens_after_best": efficiency["tokens_after_best"],
                    "score": float(task["holdout_score"]),
                    "cost": float(task["total_cost"]),
                    "experiments": int(task["experiments"]),
                    "stop": stops[0]["reason"],
                    "budget": float(starts[0]["budget"]),
                    "baseline_best": task["best_node"] == "n0000",
                    "cost_by_role": task["cost_by_role"],
                })
        tasks = []
        for task_id, rows in sorted(by_task.items()):
            reached = [row["tokens_to_reference"] for row in rows
                       if row["tokens_to_reference"] is not None]
            after_best = [row["tokens_after_best"] for row in rows
                          if row["tokens_after_best"] is not None]
            tasks.append({
                "reached_reference_runs": len(reached),
                "mean_tokens_to_reference": mean(reached) if reached else None,
                "mean_tokens_after_best": mean(after_best) if after_best else None,
                "task_id": task_id, "runs": len(rows),
                "budget": rows[0]["budget"],
                "mean_holdout_score": mean(row["score"] for row in rows),
                "holdout_score_range": [min(row["score"] for row in rows),
                                           max(row["score"] for row in rows)],
                "mean_cost": mean(row["cost"] for row in rows),
                "cost_range": [min(row["cost"] for row in rows),
                               max(row["cost"] for row in rows)],
                "mean_experiments": mean(row["experiments"] for row in rows),
                "stop_reasons": dict(sorted(Counter(row["stop"] for row in rows).items())),
                "baseline_best_runs": sum(row["baseline_best"] for row in rows),
                "mean_cost_by_role": {role: mean(row["cost_by_role"].get(role, 0.0)
                                                 for row in rows) for role in _ROLES},
            })
        breakdown.append({"candidate": name, "tasks": tasks})
    return breakdown


def _effective_settings(config: Mapping) -> dict:
    """Every tunable setting of an agent/1 config in the proposal's own shape.

    Keys an older config does not mention take their old behavior; optional
    settings that are absent read as 0 / "unchanged", the proposal's way of
    saying "not set". New keys default to the old behavior, so older configs
    and older five-key proposals keep working.
    """
    stop = config.get("stop_loss") or {}
    priors = config.get("cost_priors") or {}
    controls = config.get("role_controls") or {}
    brief = config.get("library_brief") or {}

    def thinking(role: str) -> str:
        value = (controls.get(role) or {}).get("thinking")
        return "unchanged" if value is None else ("on" if value else "off")

    return {
        **{key: deepcopy(config[key]) for key in _LEGACY_KEYS},
        "stop_loss": {
            "tab": {"first_n_no_gain": int((stop.get("tab") or {}).get("first_n_no_gain", 0) or 0)},
            "spd": {"consecutive_no_gain": int((stop.get("spd") or {}).get("consecutive_no_gain", 0) or 0)}},
        "ceiling_excludes_baseline": bool(config.get(
            "ceiling_excludes_baseline", _NEW_DEFAULTS["ceiling_excludes_baseline"])),
        "planner_mode": config.get("planner_mode", _NEW_DEFAULTS["planner_mode"]),
        "hypotheses_per_plan": int(config.get("hypotheses_per_plan", 0) or 0),
        "failure_feedback": bool(config.get("failure_feedback", _NEW_DEFAULTS["failure_feedback"])),
        "cost_priors": {"spd": int(priors.get("spd", 0) or 0), "tab": int(priors.get("tab", 0) or 0)},
        "role_controls": {role: {
            "thinking": thinking(role),
            "max_output_tokens": int((controls.get(role) or {}).get("max_output_tokens", 0) or 0)}
            for role in _ROLES},
        "experience_writer": config.get("experience_writer", _NEW_DEFAULTS["experience_writer"]),
        "library_brief": {"min_credibility": float(brief.get("min_credibility", 0.0) or 0.0),
                          "max_chars": int(brief.get("max_chars", 0) or 0)},
    }


def _exact_keys(value: Any, keys: set, name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Proposal {name} has unauthorized or missing fields")
    return value


def _bounded_int(value: Any, key: str, name: str, *, optional: bool = False) -> int:
    lower, upper = _BOUNDS[key]
    if type(value) is not int or not (lower <= value <= upper or (optional and value == 0)):
        raise ValueError(f"Proposal {name} is outside allowed bounds")
    return value


def _checked_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Proposal {name} must be true or false")
    return value


def _checked_choice(value: Any, choices: tuple, name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"Proposal {name} is not an allowed choice")
    return value


def _validate_new_settings(settings: dict) -> None:
    """Bounds of the optional levers; each present key must be complete."""
    if "stop_loss" in settings:
        stop = _exact_keys(settings["stop_loss"], {"tab", "spd"}, "stop_loss")
        _bounded_int(_exact_keys(stop["tab"], {"first_n_no_gain"}, "stop_loss.tab")["first_n_no_gain"],
                     "first_n_no_gain", "stop_loss.tab.first_n_no_gain", optional=True)
        _bounded_int(_exact_keys(stop["spd"], {"consecutive_no_gain"}, "stop_loss.spd")["consecutive_no_gain"],
                     "consecutive_no_gain", "stop_loss.spd.consecutive_no_gain", optional=True)
    if "ceiling_excludes_baseline" in settings:
        _checked_bool(settings["ceiling_excludes_baseline"], "ceiling_excludes_baseline")
    if "planner_mode" in settings:
        _checked_choice(settings["planner_mode"], _PLANNER_MODES, "planner_mode")
    if "hypotheses_per_plan" in settings:
        _bounded_int(settings["hypotheses_per_plan"], "hypotheses_per_plan",
                     "hypotheses_per_plan", optional=True)
    if "failure_feedback" in settings:
        _checked_bool(settings["failure_feedback"], "failure_feedback")
    if "cost_priors" in settings:
        priors = _exact_keys(settings["cost_priors"], {"spd", "tab"}, "cost_priors")
        for scenario in ("spd", "tab"):
            _bounded_int(priors[scenario], "cost_prior", f"cost_priors.{scenario}", optional=True)
    if "role_controls" in settings:
        controls = _exact_keys(settings["role_controls"], set(_ROLES), "role_controls")
        for role in _ROLES:
            control = _exact_keys(controls[role], {"thinking", "max_output_tokens"},
                                  f"role_controls.{role}")
            _checked_choice(control["thinking"], _THINKING, f"role_controls.{role}.thinking")
            _bounded_int(control["max_output_tokens"], "max_output_tokens",
                         f"role_controls.{role}.max_output_tokens", optional=True)
    if "experience_writer" in settings:
        _checked_choice(settings["experience_writer"], _EXPERIENCE_WRITERS, "experience_writer")
    if "library_brief" in settings:
        brief = _exact_keys(settings["library_brief"], {"min_credibility", "max_chars"}, "library_brief")
        credibility, (lower, upper) = brief["min_credibility"], _BOUNDS["min_credibility"]
        if (isinstance(credibility, bool) or not isinstance(credibility, (int, float))
                or not math.isfinite(credibility) or not (credibility == 0 or lower <= credibility <= upper)):
            raise ValueError("Proposal library_brief.min_credibility is outside allowed bounds")
        _bounded_int(brief["max_chars"], "max_chars", "library_brief.max_chars", optional=True)


def _validate_proposal(data: dict, base_config: dict) -> tuple[dict, dict, str]:
    """Check a proposal and return its complete settings (unset new keys = base)."""
    if not isinstance(data, dict) or set(data) != {"settings", "prompt_append", "rationale"}:
        raise ValueError("Proposal must contain only settings, prompt_append and rationale")
    settings, notes, rationale = data["settings"], data["prompt_append"], data["rationale"]
    if (not isinstance(settings, dict) or not set(_LEGACY_KEYS) <= set(settings)
            or set(settings) - set(_LEGACY_KEYS) - set(_NEW_KEYS)):
        raise ValueError("Proposal settings have unauthorized fields")
    for key in ("plateau_nodes", "explore_every", "query_k"):
        value = settings[key]
        lower, upper = _BOUNDS[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Proposal {key} is outside allowed bounds")
    _validate_new_settings(settings)
    behind = settings["behind_schedule"]
    if not isinstance(behind, dict) or set(behind) != {"spent_ratio", "progress_ratio"}:
        raise ValueError("Proposal behind_schedule has unauthorized fields")
    for key in ("spent_ratio", "progress_ratio"):
        value = behind[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.1 <= value <= 0.9:
            raise ValueError(f"Proposal behind_schedule.{key} is outside allowed bounds")
    effort = settings["effort"]
    if not isinstance(effort, dict) or set(effort) != set(_ROLES) or any(
            value not in {"low", "medium", "high"} for value in effort.values()):
        raise ValueError("Proposal effort contains unauthorized roles or levels")
    if not isinstance(notes, dict) or set(notes) != set(_ROLES):
        raise ValueError("Proposal prompt_append must cover the four roles")
    for role, note in notes.items():
        if (not isinstance(note, str) or len(note) > 800 or chr(0) in note
                or _FORBIDDEN_NOTE.search(note)):
            raise ValueError(f"Unsafe or oversized strategy appendix for {role}")
    if not isinstance(rationale, str) or len(rationale) > 500 or chr(0) in rationale:
        raise ValueError("Proposal rationale is invalid")
    effective = _effective_settings(base_config)
    complete = {**effective, **deepcopy(settings)}  # a legacy proposal leaves new keys unchanged
    if complete == effective and not any(notes.values()):
        raise ValueError("Proposal is a no-op")
    return complete, notes, rationale


def _set_or_drop(config: dict, key: str, value: Any) -> None:
    """Store a non-empty optional setting; an empty one removes the key (old behavior)."""
    if value:
        config[key] = value
    else:
        config.pop(key, None)


def _apply_new_settings(generated: dict, settings: dict, effective: dict) -> None:
    """Write only the new keys the proposal changes, in agent/1 config shape.

    Unset optional values (0 / "unchanged") are never written, so a candidate
    that does not use a lever stays byte-compatible with configs that predate it.
    """
    for key in ("ceiling_excludes_baseline", "planner_mode", "failure_feedback",
                "experience_writer"):
        if settings[key] != effective[key]:
            generated[key] = settings[key]
    if settings["hypotheses_per_plan"] != effective["hypotheses_per_plan"]:
        _set_or_drop(generated, "hypotheses_per_plan", settings["hypotheses_per_plan"])
    if settings["stop_loss"] != effective["stop_loss"]:
        stop = {}
        if settings["stop_loss"]["tab"]["first_n_no_gain"]:
            stop["tab"] = {"first_n_no_gain": settings["stop_loss"]["tab"]["first_n_no_gain"]}
        if settings["stop_loss"]["spd"]["consecutive_no_gain"]:
            stop["spd"] = {"consecutive_no_gain": settings["stop_loss"]["spd"]["consecutive_no_gain"]}
        _set_or_drop(generated, "stop_loss", stop)
    if settings["cost_priors"] != effective["cost_priors"]:
        _set_or_drop(generated, "cost_priors",
                     {name: value for name, value in settings["cost_priors"].items() if value})
    if settings["role_controls"] != effective["role_controls"]:
        controls = {}
        for role, control in settings["role_controls"].items():
            entry = {}
            if control["thinking"] != "unchanged":
                entry["thinking"] = control["thinking"] == "on"
            if control["max_output_tokens"]:
                entry["max_output_tokens"] = control["max_output_tokens"]
            if entry:
                controls[role] = entry
        _set_or_drop(generated, "role_controls", controls)
    if settings["library_brief"] != effective["library_brief"]:
        _set_or_drop(generated, "library_brief",
                     {name: value for name, value in settings["library_brief"].items() if value})


def _materialize_proposal(root: Path, study_dir: Path, round_number: int,
                          base_path: Path, data: dict) -> Path:
    base = _json(base_path)
    settings, notes, _ = _validate_proposal(data, base)
    generated = deepcopy(base)
    for key in _LEGACY_KEYS:
        generated[key] = settings[key]
    _apply_new_settings(generated, settings, _effective_settings(base))
    directory = study_dir / "candidates" / f"auto-{round_number:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    for role in _ROLES:
        source = _within(root, Path(base["prompts"][role]))
        original = source.read_text(encoding="utf-8")
        appendix = notes[role]
        target = directory / f"{role}.md"
        target.write_text(original + chr(10) + "# Bounded research strategy appendix" + chr(10)
                          + appendix + chr(10)
                          + "The original safety and data-isolation rules above remain in force." + chr(10),
                          encoding="utf-8")
        generated["prompts"][role] = str(target.relative_to(root))
    path = directory / "agent.json"
    path.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return path


def evaluate_candidates(
    *, project_root: Path, study_dir: Path, harness: Any,
    candidates: Mapping[str, Path], baseline_solutions: Mapping[str, Path],
    prices_path: Path, model: str, code_commit: str,
    max_workers: int = 1,
    test_only_budget: float | None = None,
    roles_factory: Callable[[str, int, str, Path], Any] | None = None,
    task_executor: Callable[..., dict] = execute_task,
) -> dict:
    """Run each supplied candidate on practice tasks, three independent times.

    The return value is the complete audited research spend plus a selected
    candidate. Validation admission is deliberately outside this function.
    """
    root = Path(project_root).expanduser().resolve()
    study_dir = _within(root, Path(study_dir))
    if study_dir.exists():
        raise FileExistsError(f"Self-research study is one-shot: {study_dir}")
    if type(max_workers) is not int or not 1 <= max_workers <= 3:
        raise ValueError("max_workers must be an integer between 1 and 3")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A concrete model is required")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", code_commit):
        raise ValueError("A concrete Git commit is required")
    formal = test_only_budget is None
    if isinstance(test_only_budget, bool) or (test_only_budget is not None and
            (not isinstance(test_only_budget, (int, float))
             or not math.isfinite(test_only_budget) or test_only_budget <= 0)):
        raise ValueError("TEST-only budget must be positive and finite")
    bank_path = root / "tasks/bank.json"
    bank = _json(bank_path)
    task_ids = _practice_tasks(bank, harness, formal=formal)
    baselines = {task_id: _within(root, Path(baseline_solutions[task_id]))
                 for task_id in task_ids}
    for task_id, path in baselines.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing practice baseline: {path}")
    config_paths, descriptors = _candidate_files(root, candidates)
    prices_path = _within(root, Path(prices_path))
    prices = _json(prices_path)
    if (prices.get("schema") != "prices/1" or not isinstance(prices.get("models"), dict)
            or not isinstance(prices["models"].get(model), dict)):
        raise ValueError("Selected model has no prices/1 rates")
    rates = prices["models"][model]
    if any(isinstance(rates.get(key), bool) or not isinstance(rates.get(key), (int, float))
           or not math.isfinite(rates[key]) or rates[key] < 0
           for key in ("input", "cached_input", "output", "reasoning")):
        raise ValueError("Selected model has incomplete token prices")
    if formal:
        if prices.get("currency") in (None, "", "TEST") or prices.get("test_only"):
            raise ValueError("Formal self-research requires actual model prices")
        require_exact_token_accounting(prices, model)
        if roles_factory is not None or task_executor is not execute_task:
            raise ValueError("Formal self-research may not inject scripted roles or executor")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Formal self-research needs OPENAI_API_KEY")
        if not (study_dir.is_relative_to(root / "batches")
                or study_dir.is_relative_to(root / "runs")):
            raise ValueError("Formal study_dir must be under batches/ or runs/")
        _verify_code_freeze(root, code_commit)
    elif prices.get("currency") != "TEST":
        raise ValueError("TEST-only self-research requires TEST prices")
    protected_paths = {bank_path, prices_path, root / "config/hard_caps.json",
                       root / "config/prices.json"}
    protected = {str(path): _hash(path) for path in protected_paths if path.is_file()}
    inputs = {str(path): _hash(path) for path in baselines.values()}
    for descriptor in descriptors.values():
        files = (descriptor["config"], *descriptor["prompts"].values())
        inputs.update({file["path"]: file["sha256"] for file in files})
    study_dir.mkdir(parents=True)
    (study_dir / "plan.json").write_text(json.dumps({
        "schema": "self_research_plan/1", "label": "FORMAL" if formal else "TEST ONLY",
        "bank_version": bank["version"], "practice_tasks": task_ids,
        "candidate_order": (["default", *(name for name in candidates if name != "default")]
                            if "default" in candidates else list(candidates)),
        "candidate_files": descriptors, "protected_sha256": protected,
        "repetitions": [1, 2, 3], "max_workers": max_workers,
        "sandbox_calls_serialized": max_workers > 1,
        "budget_policy": "positive task budget"
        if formal else "TEST-only budget override",
    }, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    attempts = []
    reports = []
    ordered = (["default", *(name for name in candidates if name != "default")]
               if "default" in candidates else list(candidates))
    task_harness = (_SerializedHarness(harness, Lock()) if max_workers > 1 else harness)

    def reserve_run() -> Path:
        for _ in range(100):
            run_dir = _run_path(root)
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return run_dir
        raise RuntimeError("Could not reserve an unused run ID")

    def one_rep(round_index: int, name: str, rep: int, run_dir: Path) -> tuple[dict, dict]:
        if formal:
            _verify_code_freeze(root, code_commit)
        if any(_hash(Path(path)) != digest for path, digest in {**protected, **inputs}.items()):
            raise ValueError("Self-research input or protected file changed")
        library_root = run_dir / "library"
        library = Library.open(library_root)
        manifest = create_manifest(
            project_root=root, run_dir=run_dir,
            phase="self_research" if formal else "dev",
            config="default_lib" if name == "default" else "tuned_lib",
            rep=rep, bank_version=bank["version"], tasks=task_ids,
            library_root=library_root, library_start_version=library.version,
            library_frozen=False, model=model, agent_config_path=config_paths[name],
            prices_path=prices_path, code_commit=code_commit,
        )
        start_run(run_dir)
        for task_id in task_ids:
            kwargs = dict(run_dir=run_dir, task_id=task_id, harness=task_harness,
                          baseline_solution=baselines[task_id], test_only_budget=test_only_budget)
            if roles_factory is not None:
                kwargs["roles"] = roles_factory(name, rep, task_id, run_dir)
            task_executor(**kwargs)
        finish_run(run_dir)
        report = run_report(run_dir)
        if report.get("holdout_failures"):
            # Candidate ranking needs every practice score; never rank on a partial mean.
            raise ValueError("Self-research repetition has failed holdout scoring: "
                             + ", ".join(report.get("holdout_failed_task_ids", [])))
        if any(_hash(Path(path)) != digest for path, digest in {**protected, **inputs}.items()):
            raise ValueError("Self-research input or protected file changed")
        attempt = {"candidate": name, "round": round_index,
                   "rep": rep, "run_id": run_dir.name,
                   "report_path": str(run_dir / "report.json"),
                   "config_sha256": descriptors[name]["config"]["sha256"],
                   "prompt_sha256": {role: item["sha256"] for role, item in
                                     descriptors[name]["prompts"].items()},
                   "mean_score": mean(float(task["holdout_score"]) for task in report["tasks"]),
                   "total_cost": report["total_cost"],
                   "currency": report["currency"],
                   "manifest_phase": manifest["phase"],
                   "logical_phase": "self_research",
                   "test_only": not formal}
        return report, attempt

    for round_index, name in enumerate(ordered, 1):
        jobs = [(rep, reserve_run()) for rep in (1, 2, 3)]
        if max_workers == 1:
            completed = [one_rep(round_index, name, rep, run_dir) for rep, run_dir in jobs]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(one_rep, round_index, name, rep, run_dir)
                           for rep, run_dir in jobs]
                completed = [future.result() for future in futures]
        for report, attempt in completed:
            reports.append(report)
            attempts.append(attempt)
    ranking = []
    for name in ordered:
        group = [attempt for attempt in attempts if attempt["candidate"] == name]
        mean_score = mean(item["mean_score"] for item in group)
        mean_cost = mean(item["total_cost"] for item in group)
        if not math.isfinite(mean_score) or not math.isfinite(mean_cost) or mean_cost <= 0:
            raise ValueError(f"Candidate {name} lacks a positive audited cost or finite score")
        # A negative normalized score is worse than the reference. Dividing it
        # by a larger cost would perversely reward extra spending.
        ranking.append({"candidate": name, "mean_score": mean_score,
                        "mean_cost": mean_cost,
                        "score_per_currency": max(mean_score, 0.0) / mean_cost,
                        "run_ids": [item["run_id"] for item in group]})
    winner = max(ranking, key=lambda item: (item["score_per_currency"],
                                            item["mean_score"], -item["mean_cost"]))
    result = {"schema": "self_research/1", "label": "FORMAL" if formal else "TEST ONLY",
              "ranking_policy": "max(mean_score, 0) / mean_cost; tie by higher score then lower cost",
              "currency": prices["currency"], "bank_version": bank["version"],
              "practice_tasks": task_ids, "candidates": descriptors,
              "attempts": attempts, "reports": reports, "ranking": ranking,
              "selection": {**winner, "config_path": str(config_paths[winner["candidate"]]),
                            "config_sha256": descriptors[winner["candidate"]]["config"]["sha256"]},
              "next_step": "Run selected config through batch validation admission before exam",
              "budget_policy": "positive task budget" if formal else "TEST-only budget override"}
    (study_dir / "self_research_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return result


def auto_evaluate_candidates(
    *, project_root: Path, study_dir: Path, harness: Any,
    default_config: Path, baseline_solutions: Mapping[str, Path],
    prices_path: Path, model: str, code_commit: str,
    rounds: int = 3, max_output_tokens: int = 2048,
    max_workers: int = 1,
    test_only_budget: float | None = None, proposal_client: Any | None = None,
    roles_factory: Callable[[str, int, str, Path], Any] | None = None,
    task_executor: Callable[..., dict] = execute_task,
) -> dict:
    """Generate bounded candidates and evaluate each on practice only.

    One invocation is limited to eight proposal calls. There is no overall
    research money cap, but each inner A task retains its positive budget and
    the 90 percent hard gate. Every proposal call, including failures, is
    separately token-priced and included in total_research_cost.
    """
    root = Path(project_root).expanduser().resolve()
    study_dir = _within(root, Path(study_dir))
    if study_dir.exists():
        raise FileExistsError(f"Automatic self-research is one-shot: {study_dir}")
    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError("Automatic candidate rounds must be between 1 and 8")
    if type(max_output_tokens) is not int or not 128 <= max_output_tokens <= 4096:
        raise ValueError("Proposal max_output_tokens must be between 128 and 4096")
    if type(max_workers) is not int or not 1 <= max_workers <= 3:
        raise ValueError("max_workers must be an integer between 1 and 3")
    formal = test_only_budget is None
    if formal and proposal_client is not None:
        raise ValueError("Formal proposal generation may not inject a scripted client")
    if not formal and proposal_client is None:
        raise ValueError("TEST-only proposal generation needs an injected client")
    default_config = _within(root, Path(default_config))
    _, default_descriptor = _candidate_files(root, {"default": default_config})
    prices_path = _within(root, Path(prices_path))
    prices = _json(prices_path)
    if prices.get("schema") != "prices/1":
        raise ValueError("Expected prices/1")
    price_usage(prices, model, Usage(input=1))
    if formal and (prices.get("currency") in (None, "", "TEST") or prices.get("test_only")):
        raise ValueError("Formal proposal generation needs actual prices")
    if formal:
        require_exact_token_accounting(prices, model)
    if not formal and prices.get("currency") != "TEST":
        raise ValueError("TEST-only proposal generation needs TEST prices")
    if formal:
        if not (study_dir.is_relative_to(root / "batches")
                or study_dir.is_relative_to(root / "runs")):
            raise ValueError("Formal study_dir must be under batches/ or runs/")
        _verify_code_freeze(root, code_commit)
    bank_path = root / "tasks/bank.json"
    practice = _practice_tasks(_json(bank_path), harness, formal=formal)
    protected_paths = {bank_path, prices_path, root / "config/hard_caps.json",
                       root / "config/prices.json"}
    protected = {str(path): _hash(path) for path in protected_paths if path.is_file()}
    default_files = [default_descriptor["default"]["config"],
                     *default_descriptor["default"]["prompts"].values()]
    default_hashes = {item["path"]: item["sha256"] for item in default_files}
    study_dir.mkdir(parents=True)
    source_prompt_sha256 = hashlib.sha256(_PROPOSAL_PROMPT.encode("utf-8")).hexdigest()
    proposal_ledger = ProposalLedger(
        study_dir / "proposal_ledger.jsonl", prices=prices,
        price_sha256=_hash(prices_path), source_prompt_sha256=source_prompt_sha256,
        model=model)
    (study_dir / "auto_plan.json").write_text(json.dumps({
        "schema": "auto_self_research_plan/1", "label": "FORMAL" if formal else "TEST ONLY",
        "rounds": rounds, "max_output_tokens": max_output_tokens,
        "max_workers": max_workers,
        "practice_tasks": practice, "default_files": default_descriptor["default"],
        "price_sha256": _hash(prices_path),
        "source_prompt_sha256": source_prompt_sha256,
        "budget_policy": "no overall cap; positive inner task budgets"
        if formal else "no overall cap; TEST-only inner task budgets",
    }, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    evaluations: dict[str, dict] = {}
    ranking: list[dict] = []
    proposal_attempts: list[dict] = []
    proposal_fingerprints: set[str] = set()

    def evaluate_one(name: str, path: Path, proposal_cost: float) -> None:
        result = evaluate_candidates(
            project_root=root, study_dir=study_dir / "studies" / name,
            harness=harness, candidates={name: path},
            baseline_solutions=baseline_solutions, prices_path=prices_path,
            model=model, code_commit=code_commit,
            max_workers=max_workers,
            test_only_budget=test_only_budget, roles_factory=roles_factory,
            task_executor=task_executor,
        )
        evaluations[name] = result
        row = result["ranking"][0]
        run_cost = sum(report["total_cost"] for report in result["reports"])
        actual_cost = run_cost + proposal_cost
        if not math.isfinite(actual_cost) or actual_cost <= 0:
            raise ValueError(f"Candidate {name} has no auditable positive cost")
        ranking.append({"candidate": name, "mean_score": row["mean_score"],
                        "mean_task_cost": row["mean_cost"],
                        "proposal_cost": proposal_cost, "research_cost": actual_cost,
                        "score_per_currency": 3 * max(row["mean_score"], 0.0) / actual_cost,
                        "run_ids": row["run_ids"]})

    evaluate_one("default", default_config, 0.0)
    proposal_model = ModelClient(model, proposal_ledger, client=proposal_client,
                                 max_output_tokens=max_output_tokens,
                                 audit_dir=study_dir / "proposal_responses")
    for round_number in range(1, rounds + 1):
        if formal:
            _verify_code_freeze(root, code_commit)
        if any(_hash(Path(path)) != digest for path, digest in {**protected, **default_hashes}.items()):
            raise ValueError("A protected or default proposal source changed")
        proposal_ledger.round = round_number
        prior_events = len(proposal_ledger.events())
        response = None
        payload = {
            "practice_tasks": practice,
            "default_settings": _effective_settings(_json(default_config)),
            "prior_results": ranking,
            "practice_breakdown": _proposal_practice_breakdown(root, evaluations, ranking),
            "instruction": "Improve average practice score per recorded currency unit",
        }
        try:
            response = proposal_model.call_json(
                role="planner", purpose="plan", effort="high",
                system_prompt=_PROPOSAL_PROMPT, payload=payload,
                schema=_PROPOSAL_SCHEMA, node_id=None)
            event = proposal_ledger.events()[-1]
            if len(proposal_ledger.events()) != prior_events + 1:
                raise ValueError("Proposal call did not append exactly one priced event")
            name = f"auto-{round_number:02d}"
            fingerprint = hashlib.sha256(json.dumps(
                {key: response.data[key] for key in ("settings", "prompt_append")},
                ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            if fingerprint in proposal_fingerprints:
                raise ValueError("Duplicate generated candidate proposal")
            candidate_path = _materialize_proposal(root, study_dir, round_number,
                                                   default_config, response.data)
            proposal_fingerprints.add(fingerprint)
            _, descriptor = _candidate_files(root, {name: candidate_path})
            proposal_attempts.append({"round": round_number, "status": "evaluated",
                                      "candidate": name, "ledger_seq": event["seq"],
                                      "cost": event["cost"], "rationale": response.data["rationale"],
                                      "proposal_sha256": fingerprint,
                                      "config_sha256": descriptor[name]["config"]["sha256"]})
            evaluate_one(name, candidate_path, event["cost"])
        except Exception as exc:
            events = proposal_ledger.events()
            event = events[-1] if len(events) > prior_events else None
            if not proposal_attempts or proposal_attempts[-1].get("round") != round_number:
                proposal_attempts.append({"round": round_number, "status": "rejected",
                                          "candidate": None,
                                          "ledger_seq": event["seq"] if event else None,
                                          "cost": event["cost"] if event else 0.0,
                                          "response_json_sha256": hashlib.sha256(json.dumps(
                                              response.data, ensure_ascii=False, sort_keys=True).encode("utf-8")
                                          ).hexdigest() if response is not None else None,
                                          "error": str(exc)})
            else:
                proposal_attempts[-1]["status"] = "evaluation_failed"
                proposal_attempts[-1]["error"] = str(exc)
                raise
        if any(_hash(Path(path)) != digest for path, digest in {**protected, **default_hashes}.items()):
            raise ValueError("A protected or default proposal source changed")
    rank_key = lambda item: (item["score_per_currency"], item["mean_score"],
                             -item["research_cost"])
    overall_winner = max(ranking, key=rank_key)
    generated = [item for item in ranking if item["candidate"] != "default"]
    winner = max(generated, key=rank_key) if generated else None
    all_reports = [report for result in evaluations.values() for report in result["reports"]]
    task_cost = sum(report["total_cost"] for report in all_reports)
    proposal_cost = proposal_ledger.spent()
    selected = evaluations[winner["candidate"]] if winner is not None else None
    result = {"schema": "auto_self_research/1", "label": "FORMAL" if formal else "TEST ONLY",
              "ranking_policy": "max(mean_score, 0) / amortized research cost; tie by higher score then lower cost",
              "currency": prices["currency"], "practice_tasks": practice,
              "proposal_ledger_path": str(proposal_ledger.path),
              "proposal_ledger_sha256": _hash(proposal_ledger.path),
              "proposal_attempts": proposal_attempts, "evaluations": evaluations,
              "reports": all_reports, "ranking": ranking,
              "overall_best_on_practice": overall_winner["candidate"],
              "selection": ({**winner, "config_path": selected["selection"]["config_path"],
                             "config_sha256": selected["selection"]["config_sha256"],
                             "beats_default_on_practice": rank_key(winner) > rank_key(ranking[0])}
                            if winner is not None else None),
              "total_task_cost": task_cost, "total_proposal_cost": proposal_cost,
              "total_research_cost": task_cost + proposal_cost,
              "budget_policy": "no overall cap; positive inner task budgets"
              if formal else "no overall cap; TEST-only inner task budgets",
              "next_step": ("Verify the best generated candidate, then run validation admission once"
                            if winner is not None else
                            "No valid generated candidate; stop before validation and exam")}
    (study_dir / "auto_self_research_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return result


def verify_auto_selected(study_report: Mapping) -> Path:
    if study_report.get("schema") != "auto_self_research/1":
        raise ValueError("Expected auto_self_research/1")
    if not isinstance(study_report.get("selection"), Mapping):
        raise ValueError("Automatic self-research produced no valid tuned candidate")
    name = study_report["selection"]["candidate"]
    return verify_selected(study_report["evaluations"][name])


__all__ = ["evaluate_candidates", "auto_evaluate_candidates",
           "verify_selected", "verify_auto_selected"]
