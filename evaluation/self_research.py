"""Practice-only configuration search with audited score/cost evidence.

This evaluates supplied or automatically proposed agent/1 and prompt
candidates. It never opens validation/exam task packages. There is no overall
research budget, but existing positive per-task budgets still apply formally.
"""

from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any, Callable, Mapping

from library import Library
from ledger import Ledger, Usage, price_usage
from research_agent.model import ModelClient

from .batch import _agent_signature, _hash, _json, _run_path, _verify_code_freeze, _within
from .finalize import finish_run, start_run
from .manifest import create_manifest
from .orchestrate import execute_task
from .proposal import ProposalLedger
from .report import run_report


_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_ROLES = ("planner", "executor", "diagnoser", "summarizer")
_FORBIDDEN_NOTE = re.compile(
    r"hidden|holdout|validation|exam|hard_caps|prices[.]json|bank[.]json|"
    r"RA_EVAL|sandbox|ignore previous|bypass|override safety|"
    r"隐藏|验证集|考试|题库|价格表|硬上限|忽略.{0,12}规则|绕过", re.IGNORECASE)
_PROPOSAL_PROMPT = (
    "You tune a research agent using practice results only. Propose bounded "
    "settings and short strategy appendices. Preserve all existing role prompt "
    "text. Never request hidden data, validation or exam tasks, price changes, "
    "bank changes, hard-cap changes, or removal of safety rules. Use the "
    "per-practice-task score, spend, stop reasons and variability to target "
    "the actual budget losses rather than only the overall average."
)


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_PROPOSAL_SCHEMA = _object({
    "settings": _object({
        "plateau_nodes": {"type": "integer"},
        "explore_every": {"type": "integer"},
        "query_k": {"type": "integer"},
        "behind_schedule": _object({"spent_ratio": {"type": "number"},
                                    "progress_ratio": {"type": "number"}}),
        "effort": _object({role: {"type": "string", "enum": ["low", "medium", "high"]}
                           for role in _ROLES}),
    }),
    "prompt_append": _object({role: {"type": "string"} for role in _ROLES}),
    "rationale": {"type": "string"},
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
                by_task.setdefault(task["task_id"], []).append({
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
            tasks.append({
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


def _validate_proposal(data: dict, base_config: dict) -> tuple[dict, dict, str]:
    if not isinstance(data, dict) or set(data) != {"settings", "prompt_append", "rationale"}:
        raise ValueError("Proposal must contain only settings, prompt_append and rationale")
    settings, notes, rationale = data["settings"], data["prompt_append"], data["rationale"]
    if not isinstance(settings, dict) or set(settings) != {
            "plateau_nodes", "explore_every", "query_k", "behind_schedule", "effort"}:
        raise ValueError("Proposal settings have unauthorized fields")
    for key, lower, upper in (("plateau_nodes", 2, 12), ("explore_every", 2, 8),
                              ("query_k", 1, 12)):
        value = settings[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Proposal {key} is outside allowed bounds")
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
    if all(settings[key] == base_config[key] for key in
           ("plateau_nodes", "explore_every", "query_k", "behind_schedule", "effort")) and not any(notes.values()):
        raise ValueError("Proposal is a no-op")
    return settings, notes, rationale


def _materialize_proposal(root: Path, study_dir: Path, round_number: int,
                          base_path: Path, data: dict) -> Path:
    base = _json(base_path)
    settings, notes, _ = _validate_proposal(data, base)
    generated = deepcopy(base)
    for key, value in settings.items():
        generated[key] = value
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
        "repetitions": [1, 2, 3], "budget_policy": "positive task budget"
        if formal else "TEST-only budget override",
    }, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    attempts = []
    reports = []
    ordered = (["default", *(name for name in candidates if name != "default")]
               if "default" in candidates else list(candidates))
    for round_index, name in enumerate(ordered, 1):
        for rep in (1, 2, 3):
            if formal:
                _verify_code_freeze(root, code_commit)
            if any(_hash(Path(path)) != digest for path, digest in {**protected, **inputs}.items()):
                raise ValueError("Self-research input or protected file changed")
            run_dir = _run_path(root)
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
                kwargs = dict(run_dir=run_dir, task_id=task_id, harness=harness,
                              baseline_solution=baselines[task_id], test_only_budget=test_only_budget)
                if roles_factory is not None:
                    kwargs["roles"] = roles_factory(name, rep, task_id, run_dir)
                task_executor(**kwargs)
            finish_run(run_dir)
            report = run_report(run_dir)
            if any(_hash(Path(path)) != digest for path, digest in {**protected, **inputs}.items()):
                raise ValueError("Self-research input or protected file changed")
            reports.append(report)
            attempts.append({"candidate": name, "round": round_index,
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
                             "test_only": not formal})
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
                                 max_output_tokens=max_output_tokens)
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
            "default_settings": {key: _json(default_config)[key] for key in
                                 ("effort", "plateau_nodes", "explore_every",
                                  "query_k", "behind_schedule")},
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
