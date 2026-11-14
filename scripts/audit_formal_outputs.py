"""Independent, read-only audit of completed internal TOKENS studies.

Reads manifests, saved reports, token ledgers, sealed input files and attempt
markers. It never invokes a scorer, opens task data, or rewrites a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev


BUCKETS = ("input", "cached_input", "output", "reasoning")
ROLES = ("planner", "executor", "diagnoser", "summarizer")
TIERS = ("practice", "validation", "exam")
CONFIGS = ("naive", "default", "default_lib", "tuned_lib")
REPS = (1, 2, 3)
TASK_COUNTS = {"practice": 10, "validation": 4, "exam": 10}
TASK_BUDGET = 40_000
RUN_ID = re.compile(r"run-[0-9]{8}-[0-9]{6}-[a-z0-9]{4}$")
TASK_ID = re.compile(r"(spd|tab)-[a-z0-9][a-z0-9-]*$")
SHA256 = re.compile(r"[0-9a-f]{64}$")
COMMIT = re.compile(r"[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    _require(not path.is_symlink(), f"Refusing symlinked audit input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path, schema: str) -> list[dict]:
    _require(not path.is_symlink(), f"Refusing symlinked audit input: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    _require(all(isinstance(row, dict) and row.get("schema") == schema
                 and row.get("seq") == seq for seq, row in enumerate(rows, 1)),
             f"Invalid {schema} sequence: {path}")
    return rows


def _sha(path: Path) -> str:
    _require(not path.is_symlink(), f"Refusing symlinked audit input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, supplied: str | Path) -> Path:
    _require(isinstance(supplied, (str, Path)), "File path is missing")
    path = Path(supplied)
    path = (path if path.is_absolute() else root / path).resolve()
    _require(path.is_relative_to(root), f"File escapes project root: {path}")
    return path


def _descriptor(root: Path, item: dict) -> Path:
    _require(isinstance(item, dict), "Missing file descriptor")
    path = _inside(root, item.get("path"))
    digest = item.get("sha256")
    _require(isinstance(digest, str) and SHA256.fullmatch(digest)
             and _sha(path) == digest, f"File SHA mismatch: {path}")
    return path


def _number(value: object, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             f"Invalid nonnegative number: {label}")
    return float(value)


def _equal_cost(actual: object, expected: float, label: str) -> None:
    _require(math.isclose(_number(actual, label), expected, rel_tol=0, abs_tol=1e-8),
             f"Cost mismatch: {label}")


def _tokens(value: object, label: str) -> dict[str, int]:
    _require(isinstance(value, dict) and set(value) == set(BUCKETS)
             and all(type(value[key]) is int and value[key] >= 0 for key in BUCKETS),
             f"Invalid four token buckets: {label}")
    return value


def _proposal_data(output: object) -> dict:
    _require(isinstance(output, str), "Proposal response has no JSON text")
    source = output.strip()
    for prefix in ("\x60\x60\x60json\n", "\x60\x60\x60\n"):
        if source.startswith(prefix) and source.endswith("\n\x60\x60\x60"):
            source = source[len(prefix):-4]
            break
    value = json.loads(source)
    _require(isinstance(value, dict), "Proposal response is not a JSON object")
    return value


def _audit_proposal_responses(study_dir: Path, proposals: list[dict], attempts: list[dict]) -> None:
    """Tie priced proposal events and generated fingerprints to saved raw responses."""
    archive = study_dir / "proposal_responses"
    _require(not archive.is_symlink() and archive.resolve() == archive,
             "Proposal response archive path differs")
    files = sorted(archive.glob("response-*.json"))
    _require(len(files) == len(proposals), "Proposal response archive count differs")
    remaining = [_json(path) for path in files]
    for seq, (event, attempt) in enumerate(zip(proposals, attempts), 1):
        matches = [item for item in remaining
                   if item.get("prompt_hash") == event.get("prompt_hash")]
        _require(len(matches) == 1, f"Proposal response hash is missing or ambiguous: {seq}")
        saved = matches[0]
        remaining.remove(saved)
        _require(saved.get("schema") == "model_response_audit/1"
                 and saved.get("model") == event.get("model")
                 and saved.get("role") == event.get("role") == "planner"
                 and saved.get("purpose") == event.get("purpose") == "plan"
                 and saved.get("tokens") == event.get("tokens"),
                 f"Proposal response provenance differs: {seq}")
        _equal_cost(saved.get("cost"), event["cost"], f"proposal response {seq}")
        if attempt["status"] == "evaluated":
            data = _proposal_data(saved.get("output_text"))
            _require(set(data) == {"settings", "prompt_append", "rationale"},
                     f"Proposal response fields differ: {seq}")
            fingerprint = hashlib.sha256(json.dumps(
                {key: data[key] for key in ("settings", "prompt_append")},
                ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            _require(attempt.get("proposal_sha256") == fingerprint,
                     f"Proposal fingerprint SHA differs: {seq}")
        elif attempt.get("response_json_sha256") is not None:
            data = _proposal_data(saved.get("output_text"))
            fingerprint = hashlib.sha256(json.dumps(
                data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            _require(attempt["response_json_sha256"] == fingerprint,
                     f"Rejected proposal response SHA differs: {seq}")


def _candidate(root: Path, descriptor: dict) -> Path:
    _require(isinstance(descriptor, dict) and set(descriptor.get("prompts", {})) == set(ROLES),
             "Candidate lacks four prompt descriptors")
    config_item = descriptor.get("config")
    _require(isinstance(config_item, dict), "Candidate config descriptor is missing")
    path = _inside(root, config_item.get("path"))
    _require(any(path.is_relative_to(root / prefix) for prefix in
                 ("config", "runs", "batches")), "Candidate config is outside config or outputs")
    _descriptor(root, config_item)
    config = _json(path)
    _require(config.get("schema") == "agent/1" and set(config.get("prompts", {})) == set(ROLES),
             "Candidate config is not agent/1 with four prompts")
    for role in ROLES:
        item = descriptor["prompts"][role]
        _require(isinstance(item, dict), f"Candidate {role} descriptor is missing")
        actual = _inside(root, item.get("path"))
        _require(any(actual.is_relative_to(root / prefix) for prefix in
                     ("config", "runs", "batches")),
                 f"Candidate {role} prompt is outside config or outputs")
        _descriptor(root, item)
        _require(actual == _inside(root, config["prompts"][role]),
                 f"Candidate {role} prompt path differs from config")
    return path


def _audit_run(root: Path, run_id: str, *, phase: str, config: str, rep: int,
               task_ids: list[str], bank_version: str, commit: str,
               price_sha: str, selected_config: Path | None = None) -> dict:
    _require(isinstance(run_id, str) and RUN_ID.fullmatch(run_id) is not None,
             f"Invalid run ID: {run_id}")
    run_dir = (root / "runs" / run_id).resolve()
    _require(run_dir.parent == root / "runs", "Run path escapes runs directory")
    manifest = _json(run_dir / "manifest.json")
    report = _json(run_dir / "report.json")
    _require(manifest.get("schema") == "run/1" and manifest.get("run_id") == run_id
             and manifest.get("phase") == phase and manifest.get("config") == config
             and manifest.get("rep") == rep and manifest.get("tasks") == task_ids
             and manifest.get("bank_version") == bank_version
             and manifest.get("code_commit") == commit, f"Run manifest differs: {run_id}")
    _require(report.get("schema") == "run_report/1" and report.get("run_id") == run_id
             and report.get("phase") == phase and report.get("config") == config
             and report.get("rep") == rep and report.get("currency") == "TOKENS"
             and report.get("exact_token_accounting") is True,
             f"Run report identity or accounting differs: {run_id}")
    rates = _json(root / "config/prices.json").get("models", {}).get(manifest.get("model"))
    _require(isinstance(rates, dict) and set(rates) == set(BUCKETS)
             and all(type(rates[key]) in (int, float) and rates[key] == 1 for key in BUCKETS),
             f"Run model lacks four exact token rates: {run_id}")
    agent_item = manifest.get("agent_config")
    _require(isinstance(agent_item, dict), f"Run config descriptor missing: {run_id}")
    config_path = _inside(root, agent_item.get("path"))
    _require(any(config_path.is_relative_to(root / prefix) for prefix in
                 ("config", "runs", "batches")), f"Run config path differs: {run_id}")
    _descriptor(root, agent_item)
    price_item = manifest.get("prices")
    _require(isinstance(price_item, dict)
             and _inside(root, price_item.get("path")) == root / "config/prices.json",
             f"Run price source differs: {run_id}")
    _require(_descriptor(root, price_item) == root / "config/prices.json"
             and price_item["sha256"] == price_sha,
             f"Run price source differs: {run_id}")
    if selected_config is not None:
        _require(config_path == selected_config, f"Selected config path differs: {run_id}")
    tasks = report.get("tasks")
    _require(isinstance(tasks, list) and [item.get("task_id") for item in tasks] == task_ids,
             f"Run task order differs: {run_id}")
    total = 0.0
    for task in tasks:
        task_id = task["task_id"]
        task_dir = (run_dir / "tasks" / task_id).resolve()
        _require(task_dir.parent == run_dir / "tasks", "Task path escapes run directory")
        rows = _jsonl(task_dir / "ledger.jsonl", "ledger/1")
        _require(all(row.get("run_id") == run_id and row.get("task_id") == task_id
                     for row in rows), f"Task ledger identity differs: {run_id}/{task_id}")
        starts = [row for row in rows if row.get("kind") == "task_start"]
        stops = [row for row in rows if row.get("kind") == "stop"]
        ends = [row for row in rows if row.get("kind") == "task_end"]
        calls = [row for row in rows if row.get("kind") == "llm_call"]
        _require(len(starts) == len(stops) == len(ends) == 1,
                 f"Task lacks one start/stop/end: {run_id}/{task_id}")
        _equal_cost(starts[0].get("budget"), TASK_BUDGET, f"{task_id} budget")
        usage = {key: 0 for key in BUCKETS}
        role_cost = {role: 0.0 for role in ROLES}
        for call in calls:
            buckets = _tokens(call.get("tokens"), f"{run_id}/{task_id} call")
            cost = sum(buckets.values())
            _equal_cost(call.get("cost"), cost, f"{run_id}/{task_id} call")
            _require(call.get("role") in ROLES and call.get("model") == manifest.get("model"),
                     f"Call role or model differs: {run_id}/{task_id}")
            for key in BUCKETS:
                usage[key] += buckets[key]
            role_cost[call["role"]] += cost
        cost = float(sum(usage.values()))
        _require(cost <= TASK_BUDGET, f"Task exceeds 40,000 tokens: {run_id}/{task_id}")
        _equal_cost(task.get("total_cost"), cost, f"{run_id}/{task_id} report")
        _equal_cost(ends[0].get("total_cost"), cost, f"{run_id}/{task_id} task_end")
        score = task.get("holdout_score")
        ending_score = ends[0].get("holdout_score")
        _require(type(score) in (int, float) and math.isfinite(score)
                 and type(ending_score) in (int, float) and math.isfinite(ending_score)
                 and math.isclose(score, ending_score, rel_tol=0, abs_tol=1e-12),
                 f"Task saved score differs from task_end: {run_id}/{task_id}")
        saved_score = _json(task_dir / "holdout_score.json")
        result_score = saved_score.get("score")
        _require(saved_score.get("schema") == "score/1"
                 and saved_score.get("split") == "holdout"
                 and type(result_score) in (int, float) and math.isfinite(result_score)
                 and math.isclose(result_score, ending_score, rel_tol=0, abs_tol=1e-12),
                 f"Task holdout score artifact differs: {run_id}/{task_id}")
        _require(task.get("schema") == "task_report/1" and task.get("run_id") == run_id
                 and task.get("currency") == "TOKENS" and task.get("tokens") == usage
                 and task.get("model_calls") == len(calls),
                 f"Task report buckets or identity differ: {run_id}/{task_id}")
        stated_roles = task.get("cost_by_role")
        _require(isinstance(stated_roles, dict) and set(stated_roles) == set(ROLES),
                 f"Task role costs missing: {run_id}/{task_id}")
        for role in ROLES:
            _equal_cost(stated_roles[role], role_cost[role], f"{run_id}/{task_id}/{role}")
        evidence = task.get("evidence", {})
        _require(evidence.get("task_end_seq") == ends[0]["seq"]
                 and evidence.get("stop_seq") == stops[0]["seq"]
                 and evidence.get("model_call_seqs") == [row["seq"] for row in calls]
                 and _inside(root, evidence.get("ledger_path")) == task_dir / "ledger.jsonl"
                 and _inside(root, evidence.get("holdout_path")) == task_dir / "holdout_score.json",
                 f"Task evidence differs: {run_id}/{task_id}")
        attempt = _json(task_dir / "holdout_attempt.json")
        _require(attempt.get("schema") == "holdout_attempt/1",
                 f"Missing one-shot task attempt: {run_id}/{task_id}")
        total += cost
    _equal_cost(report.get("total_cost"), total, f"{run_id} report")
    run_rows = _jsonl(run_dir / "ledger.jsonl", "ledger/1")
    ends = [row for row in run_rows if row.get("kind") == "run_end"]
    _require(len(ends) == 1 and all(row.get("run_id") == run_id for row in run_rows),
             f"Run ledger identity or end differs: {run_id}")
    _equal_cost(ends[0].get("total_cost"), total, f"{run_id} run_end")
    _require(report.get("evidence", {}).get("run_end_seq") == ends[0]["seq"],
             f"Run evidence differs: {run_id}")
    return report


def _audit_study(root: Path, study_dir: Path, tasks: list[str], bank_version: str,
                 commit: str, price_sha: str) -> tuple[Path, dict]:
    plan = _json(study_dir / "auto_plan.json")
    study = _json(study_dir / "auto_self_research_report.json")
    _require(plan.get("schema") == "auto_self_research_plan/1"
             and study.get("schema") == "auto_self_research/1"
             and plan.get("label") == study.get("label") == "FORMAL"
             and study.get("currency") == "TOKENS"
             and plan.get("practice_tasks") == study.get("practice_tasks") == tasks
             and plan.get("price_sha256") == price_sha,
             "Self-research plan or report differs from frozen practice")
    _require(type(plan.get("rounds")) is int and 1 <= plan["rounds"] <= 8
             and plan.get("max_output_tokens") == 2048,
             "Self-research proposal limits differ")
    proposal_path = study_dir / "proposal_ledger.jsonl"
    _require(_inside(root, study.get("proposal_ledger_path")) == proposal_path
             and study.get("proposal_ledger_sha256") == _sha(proposal_path),
             "Proposal ledger SHA or path differs")
    proposals = _jsonl(proposal_path, "proposal_ledger/1")
    attempts = study.get("proposal_attempts")
    _require(isinstance(attempts, list) and len(proposals) == len(attempts) == plan["rounds"],
             "Proposal attempts or rounds differ")
    proposal_models = {row.get("model") for row in proposals}
    _require(len(proposal_models) == 1 and isinstance(next(iter(proposal_models)), str),
             "Proposal calls used inconsistent models")
    model = next(iter(proposal_models))
    proposal_cost = 0.0
    for seq, (row, attempt) in enumerate(zip(proposals, attempts), 1):
        cost = sum(_tokens(row.get("tokens"), f"proposal {seq}").values())
        _equal_cost(row.get("cost"), cost, f"proposal {seq}")
        _equal_cost(attempt.get("cost"), cost, f"proposal attempt {seq}")
        _require(row.get("round") == attempt.get("round") == seq
                 and attempt.get("ledger_seq") == seq
                 and row.get("price_sha256") == price_sha
                 and row.get("source_prompt_sha256") == plan.get("source_prompt_sha256")
                 and attempt.get("status") in {"evaluated", "rejected", "evaluation_failed"},
                 f"Proposal provenance differs: {seq}")
        proposal_cost += cost
    _equal_cost(study.get("total_proposal_cost"), proposal_cost, "study proposals")
    evaluations = study.get("evaluations")
    _require(isinstance(evaluations, dict) and "default" in evaluations,
             "Self-research lacks default comparison")
    evaluated = {row.get("candidate") for row in attempts if row.get("status") == "evaluated"}
    _require(set(evaluations) == {"default", *evaluated},
             "Candidate evaluations differ from priced proposals")
    _require(plan.get("default_files") == evaluations["default"].get("candidates", {}).get("default"),
             "Default study files differ from plan")
    flat = {}
    task_cost = 0.0
    verified_by_candidate = {}
    for name, evaluation in evaluations.items():
        descriptor = evaluation.get("candidates", {}).get(name)
        config_path = _candidate(root, descriptor)
        reps = evaluation.get("attempts")
        embedded = evaluation.get("reports")
        _require(isinstance(reps, list) and isinstance(embedded, list)
                 and len(reps) == len(embedded) == 3
                 and {row.get("rep") for row in reps} == set(REPS),
                 f"Candidate needs three practice runs: {name}")
        by_id = {row.get("run_id"): row for row in embedded}
        _require(len(by_id) == 3, f"Duplicate candidate run: {name}")
        _require([row.get("rep") for row in reps] == list(REPS),
                 f"Candidate repetitions are out of order: {name}")
        verified_by_candidate[name] = []
        for attempt in reps:
            run_id = attempt.get("run_id")
            _require(run_id not in flat and attempt.get("candidate") == name
                     and attempt.get("config_sha256") == descriptor["config"]["sha256"]
                     and attempt.get("prompt_sha256") == {role: descriptor["prompts"][role]["sha256"]
                                                        for role in ROLES},
                     f"Candidate attempt signature differs: {name}")
            report = _audit_run(root, run_id, phase="self_research",
                                config="default_lib" if name == "default" else "tuned_lib",
                                rep=attempt["rep"], task_ids=tasks,
                                bank_version=bank_version, commit=commit,
                                price_sha=price_sha, selected_config=config_path)
            _require(_json(root / "runs" / run_id / "manifest.json").get("model") == model,
                     f"Proposal and task models differ: {run_id}")
            _require(report == by_id.get(run_id)
                     and _inside(root, attempt.get("report_path")) == root / "runs" / run_id / "report.json",
                     f"Candidate saved report differs: {run_id}")
            _equal_cost(attempt.get("total_cost"), report["total_cost"], f"study {run_id}")
            observed_score = mean(task["holdout_score"] for task in report["tasks"])
            stated_score = attempt.get("mean_score")
            _require(type(stated_score) in (int, float) and math.isfinite(stated_score)
                     and math.isclose(stated_score, observed_score, rel_tol=0, abs_tol=1e-12),
                     f"Candidate attempt mean score differs: {run_id}")
            verified_by_candidate[name].append((report, observed_score))
            flat[run_id] = report
            task_cost += report["total_cost"]
    stated = study.get("reports")
    _require(isinstance(stated, list) and len(stated) == len(flat)
             and {row.get("run_id") for row in stated} == set(flat)
             and all(row == flat[row["run_id"]] for row in stated),
             "Self-research flat run list differs")
    _equal_cost(study.get("total_task_cost"), task_cost, "study tasks")
    _equal_cost(study.get("total_research_cost"), task_cost + proposal_cost, "study total")
    ordered_candidates = ["default", *[row["candidate"] for row in attempts
                                       if row.get("status") == "evaluated"]]
    _require(len(ordered_candidates) == len(set(ordered_candidates))
             and set(ordered_candidates) == set(evaluations),
             "Candidate order differs from evaluated proposals")
    cost_by_candidate = {"default": 0.0}
    cost_by_candidate.update({row["candidate"]: float(row["cost"])
                              for row in attempts if row.get("status") == "evaluated"})
    expected_ranking = []
    for name in ordered_candidates:
        verified = verified_by_candidate[name]
        mean_score = mean(score for _, score in verified)
        mean_task_cost = mean(report["total_cost"] for report, _ in verified)
        research_cost = sum(report["total_cost"] for report, _ in verified) + cost_by_candidate[name]
        _require(research_cost > 0, f"Candidate has no positive research cost: {name}")
        expected_ranking.append({
            "candidate": name, "mean_score": mean_score,
            "mean_task_cost": mean_task_cost, "proposal_cost": cost_by_candidate[name],
            "research_cost": research_cost,
            "score_per_currency": 3 * max(mean_score, 0.0) / research_cost,
            "run_ids": [report["run_id"] for report, _ in verified],
        })
    _require(study.get("ranking") == expected_ranking,
             "Self-research ranking differs from audited practice reports")
    _require(len(expected_ranking) > 1, "No selected generated candidate")
    rank_key = lambda row: (row["score_per_currency"], row["mean_score"], -row["research_cost"])
    overall = max(expected_ranking, key=rank_key)
    winner = max(expected_ranking[1:], key=rank_key)
    _require(study.get("overall_best_on_practice") == overall["candidate"],
             "Overall practice winner differs from audited ranking")
    selection = study.get("selection")
    _require(isinstance(selection, dict) and selection.get("candidate") == winner["candidate"],
             "Selected candidate is not the best generated proposal")
    _audit_proposal_responses(study_dir, proposals, attempts)
    chosen = evaluations[selection["candidate"]]["candidates"][selection["candidate"]]
    path = _candidate(root, chosen)
    expected_selection = {**winner, "config_path": str(path),
                          "config_sha256": chosen["config"]["sha256"],
                          "beats_default_on_practice": rank_key(winner) > rank_key(expected_ranking[0])}
    _require(selection == expected_selection, "Selected configuration SHA or ranking differs")
    return path, {"runs": len(flat), "task_cost": task_cost,
                  "proposal_cost": proposal_cost, "total_cost": task_cost + proposal_cost,
                  "selected_config_sha256": chosen["config"]["sha256"], "model": model,
                  "default_config": evaluations["default"]["candidates"]["default"]["config"]}


def _expected_comparison(base: list[dict], tuned: list[dict], tasks: list[str]) -> dict:
    """Reconstruct the saved three-pair validation comparison from reported scores."""
    base_scores = [mean(task["holdout_score"] for task in row["tasks"]) for row in base]
    tuned_scores = [mean(task["holdout_score"] for task in row["tasks"]) for row in tuned]
    differences = [candidate - baseline for baseline, candidate in zip(base_scores, tuned_scores)]
    mean_gap = mean(differences)
    noise = max(stdev(base_scores), stdev(tuned_scores))
    return {
        "schema": "comparison/1", "phase": "validation", "tasks": tasks,
        "baseline": "default_lib", "candidate": "tuned_lib",
        "baseline_scores": base_scores, "candidate_scores": tuned_scores,
        "paired_differences": differences, "mean_score_gap": mean_gap,
        "within_config_noise": noise,
        "direction_consistent": all(value > 0 for value in differences),
        "claim": ("better" if all(value > 0 for value in differences)
                  and mean_gap > noise else "no_clear_difference"),
        "baseline_mean_cost": mean(row["total_cost"] for row in base),
        "candidate_mean_cost": mean(row["total_cost"] for row in tuned),
        "evidence_run_ids": {
            "default_lib": [row["run_id"] for row in base],
            "tuned_lib": [row["run_id"] for row in tuned],
        },
        "rule": "Three paired repetitions must all improve; mean score gap must exceed both configurations' sample standard deviations.",
    }


def audit_formal_outputs(batch_dir: Path, auto_study_dir: Path) -> dict:
    """Audit saved formal outputs only; a completed batch is required."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    root = batch_dir.parent.parent
    _require(batch_dir.parent.name in {"runs", "batches"}
             and batch_dir.is_relative_to(root), "Batch must be under project runs/ or batches/")
    study_dir = Path(auto_study_dir).expanduser().resolve()
    _require(study_dir.parent == root / "runs" or study_dir.parent == root / "batches",
             "Auto study must be under the same project")
    bank = _json(root / "tasks/bank.json")
    plan = _json(batch_dir / "plan.json")
    batch = _json(batch_dir / "batch_report.json")
    _require(bank.get("schema") == "bank/1" and bank.get("frozen") is True
             and plan.get("schema") == "batch_plan/1"
             and batch.get("schema") == "batch_report/1"
             and plan.get("label") == batch.get("label") == "FORMAL"
             and plan.get("bank_version") == batch.get("bank_version") == bank.get("version"),
             "Formal batch or frozen bank identity differs")
    version = bank["version"]
    _require(isinstance(version, str) and re.fullmatch(r"bank-v[1-9][0-9]*", version)
             and isinstance(bank.get("tasks"), list)
             and all(isinstance(item, dict) for item in bank["tasks"]),
             "Frozen bank has invalid version or task entries")
    expected_tasks = {tier: [item["task_id"] for item in bank["tasks"]
                             if item.get("tier") == tier] for tier in TIERS}
    all_task_ids = [task_id for ids in expected_tasks.values() for task_id in ids]
    _require(all(len(expected_tasks[tier]) == TASK_COUNTS[tier] for tier in TIERS)
             and len(set(all_task_ids)) == 24
             and all(TASK_ID.fullmatch(task_id) for task_id in all_task_ids)
             and plan.get("tasks") == expected_tasks
             and set(plan.get("configs", [])) == set(CONFIGS)
             and len(plan["configs"]) == 4 and plan.get("reps") == list(REPS)
             and plan.get("run_validation") is True and plan.get("run_exam") is True,
             "Batch plan lacks expected tasks, configurations or stages")
    prices_path = root / "config/prices.json"
    prices = _json(prices_path)
    _require(prices.get("schema") == "prices/1" and prices.get("currency") == "TOKENS"
             and type(prices.get("unit_tokens")) is int and prices["unit_tokens"] == 1
             and prices.get("test_only") is False,
             "Prices are not formal exact TOKENS")
    price_sha = _sha(prices_path)
    sealed = plan.get("sealed_inputs")
    _require(isinstance(sealed, dict) and str(root / "tasks/bank.json") in sealed
             and str(prices_path) in sealed, "Batch sealed inputs lack bank or prices")
    for filename, digest in sealed.items():
        path = _inside(root, filename)
        _require("hidden" not in {part.lower() for part in path.parts}
                 and "holdout" not in path.name.lower(),
                 f"Refusing private sealed input: {path}")
        _require(isinstance(digest, str) and SHA256.fullmatch(digest)
                 and _sha(path) == digest,
                 f"Batch sealed input changed: {filename}")
    marker_commit = set()
    for tier in ("validation", "exam"):
        marker = _json(root / "runs" / f"{tier}-{version}.attempt.json")
        commit = marker.get("code_commit")
        _require(marker.get("schema") == f"{tier}_attempt/1"
                 and marker.get("bank_version") == version
                 and _inside(root, marker.get("batch_dir")) == batch_dir
                 and isinstance(commit, str) and COMMIT.fullmatch(commit),
                 f"Formal {tier} one-shot marker differs")
        marker_commit.add(commit)
    _require(len(marker_commit) == 1, "Formal markers used different commits")
    commit = marker_commit.pop()
    local_attempt = _json(batch_dir / "exam_attempt.json")
    _require(local_attempt.get("schema") == "exam_attempt/1"
             and local_attempt.get("label") == "FORMAL", "Batch exam attempt marker differs")
    chosen_path, study = _audit_study(root, study_dir, expected_tasks["practice"],
                                      version, commit, price_sha)
    _require(str(chosen_path) in sealed and sealed[str(chosen_path)] ==
             study["selected_config_sha256"],
             "Selected config is not sealed in batch plan")
    default_item = study["default_config"]
    default_path = _inside(root, default_item["path"])
    _require(str(default_path) in sealed and sealed[str(default_path)] ==
             default_item["sha256"], "Default config is not sealed in batch plan")
    # batch_plan/1 seals input files but does not itself map names to paths.
    # This internal protocol fixes the naive comparator at config/naive.json;
    # default comes from auto_plan and tuned from the selected candidate.
    naive_path = root / "config/naive.json"
    _require(str(naive_path) in sealed and sealed[str(naive_path)] == _sha(naive_path),
             "Naive config is not sealed in batch plan")
    admission = batch.get("admission")
    _require(isinstance(admission, dict) and admission.get("schema") == "admission/1"
             and admission.get("baseline") == "default_lib"
             and admission.get("candidate") == "tuned_lib"
             and admission.get("claim") in {"admitted", "rejected"},
             "Batch validation admission differs")
    admitted = admission["claim"] == "admitted"
    _require(batch.get("tuned_exam_skipped") is (not admitted),
             "Tuned exam exclusion differs from admission")
    phase_configs = {"practice": CONFIGS,
                     "validation": ("default_lib", "tuned_lib"),
                     "exam": CONFIGS if admitted else CONFIGS[:3]}
    rows = batch.get("runs")
    embedded = batch.get("reports")
    _require(isinstance(rows, list) and isinstance(embedded, dict)
             and set(embedded) == set(TIERS), "Batch run rows or reports missing")
    expected = {(tier, config, rep) for tier in TIERS
                for config in phase_configs[tier] for rep in REPS}
    _require(len(rows) == len(expected)
             and {(row.get("logical_phase"), row.get("config"), row.get("rep"))
                  for row in rows} == expected
             and len({row.get("run_id") for row in rows}) == len(rows),
             "Batch stage/run count differs")
    report_by_id = {}
    for tier in TIERS:
        _require(isinstance(embedded[tier], list)
                 and len(embedded[tier]) == len(phase_configs[tier]) * 3,
                 f"Batch {tier} report count differs")
        for report in embedded[tier]:
            _require(report.get("run_id") not in report_by_id,
                     "Duplicate embedded batch report")
            report_by_id[report["run_id"]] = report
    audited = {}
    config_sources = {}
    for row in rows:
        tier, config, rep, run_id = (row.get(key) for key in
                                    ("logical_phase", "config", "rep", "run_id"))
        _require(row.get("manifest_phase") == tier and row.get("test_only") is False
                 and _inside(root, row.get("report_path")) == root / "runs" / run_id / "report.json",
                 f"Batch run citation differs: {run_id}")
        report = _audit_run(root, run_id, phase=tier, config=config, rep=rep,
                            task_ids=expected_tasks[tier], bank_version=version,
                            commit=commit, price_sha=price_sha,
                            selected_config=chosen_path if config == "tuned_lib" else None)
        manifest = _json(root / "runs" / run_id / "manifest.json")
        _require(manifest.get("model") == study["model"],
                 f"Batch and self-research models differ: {run_id}")
        agent_item = manifest["agent_config"]
        agent_path = _inside(root, agent_item["path"])
        signature = (str(agent_path), agent_item["sha256"])
        _require(sealed.get(signature[0]) == signature[1],
                 f"Batch config is not sealed by plan: {run_id}")
        if config in ("default", "default_lib"):
            _require(signature == (str(default_path), default_item["sha256"]),
                     f"Default batch config differs from self-research plan: {run_id}")
        elif config == "tuned_lib":
            _require(signature == (str(chosen_path), study["selected_config_sha256"]),
                     f"Tuned batch config differs from selected candidate: {run_id}")
        else:
            _require(signature == (str(naive_path), sealed[str(naive_path)]),
                     f"Naive batch config differs from sealed comparator: {run_id}")
        if config in config_sources:
            _require(config_sources[config] == signature,
                     f"Batch config changed between repetitions or phases: {config}")
        else:
            config_sources[config] = signature
        _require(report == report_by_id.get(run_id), f"Batch embedded report differs: {run_id}")
        audited[(tier, config, rep)] = report
    _require(set(report_by_id) == {row["run_id"] for row in rows},
             "Batch reports contain unexpected runs")
    base = [audited[("validation", "default_lib", rep)] for rep in REPS]
    tuned = [audited[("validation", "tuned_lib", rep)] for rep in REPS]
    score_ok = all(mean(task["holdout_score"] for task in candidate["tasks"])
                   >= mean(task["holdout_score"] for task in reference["tasks"])
                   for reference, candidate in zip(base, tuned))
    cost_ok = mean(row["total_cost"] for row in tuned) < mean(row["total_cost"] for row in base)
    _require(admitted == (score_ok and cost_ok)
             and admission.get("score_not_lower_each_rep") is score_ok
             and admission.get("cost_reduced_on_average") is cost_ok
             and admission.get("comparison") == _expected_comparison(
                 base, tuned, expected_tasks["validation"]),
             "Recorded admission differs from validation evidence")
    return {"schema": "formal_output_audit/1", "ready": True,
            "bank_version": version, "code_commit": commit,
            "currency": "TOKENS", "task_budget": TASK_BUDGET,
            "batch_runs": len(rows), "phase_runs": {tier: len(embedded[tier]) for tier in TIERS},
            "study_runs": study["runs"], "study_task_cost": study["task_cost"],
            "proposal_cost": study["proposal_cost"], "total_research_cost": study["total_cost"],
            "selected_config_sha256": study["selected_config_sha256"],
            "admission": admission["claim"], "over_budget_tasks": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--auto-study-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_formal_outputs(args.batch_dir, args.auto_study_dir)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "formal_output_audit/1", "ready": False,
                          "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
