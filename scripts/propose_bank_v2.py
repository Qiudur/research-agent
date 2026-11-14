"""Audit new dev-only naive campaigns, propose and freeze bank-v2 without scoring.

Never reads holdout data, retries an attempt, or overwrites bank-v1. Frozen
output requires an exact proposal SHA and a separate tabular license review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from . import aggregate_naive_costs as prior

MODEL = "qwen3.5-27b"
TASK_ID = re.compile(r"(?:spd|tab)-[a-z0-9-]+")
LEVELS = ("easy", "medium", "hard")
RUNTIME_FILES = {"harness.py", "harness_worker.py", "tabular_harness.py",
                 "tabular_worker.py", "ledger.py", "token_accounting.py",
                 "scripts/calibrate_naive_tokens.py", "scripts/run_token_pilot.py"}
RUNTIME_PREFIXES = ("evaluation/", "research_agent/", "library/", "config/")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read(path: Path) -> dict[str, Any]:
    return prior.read(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def runtime_drift(root: Path, first: str, target: str) -> dict:
    for commit in (first, target):
        require(isinstance(commit, str) and prior.COMMIT.fullmatch(commit) is not None,
                "Invalid calibration or target commit")
        test = subprocess.run(["git", "-C", str(root), "cat-file", "-e",
                               f"{commit}^{{commit}}"], capture_output=True)
        require(test.returncode == 0, f"Unknown Git commit: {commit}")
    diff = subprocess.run(["git", "-C", str(root), "diff", "--name-only",
                           "--no-renames", first, target], text=True, capture_output=True)
    require(diff.returncode == 0, "Cannot compare code commits")
    changed = sorted(set(diff.stdout.splitlines()))
    risky = [name for name in changed if name in RUNTIME_FILES
             or name.startswith(RUNTIME_PREFIXES)]
    return {"campaign_commit": first, "target_commit": target,
            "runtime_changed": risky, "all_changed": changed}


def verify_target_worktree(root: Path, target: str) -> None:
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          text=True, capture_output=True)
    require(head.returncode == 0 and head.stdout.strip() == target,
            "Target worktree HEAD must equal frozen target commit")
    status = subprocess.run(["git", "-C", str(root), "status",
                             "--porcelain=v1", "--untracked-files=all"],
                            text=True, capture_output=True)
    require(status.returncode == 0 and not status.stdout.strip(),
            "Target worktree must be clean before proposing/freezing")


def audit_campaign(root: Path, campaign: Path, scenario: str, target_commit: str) -> dict:
    campaign = campaign.expanduser().resolve()
    protocol, summary = read(campaign / "protocol.json"), read(campaign / "summary.json")
    label = "DEV ONLY; TEST TOKEN UNITS; NO HOLDOUT OR MONEY"
    require(protocol.get("schema") == "naive_token_calibration_protocol/1"
            and summary.get("schema") == "naive_token_calibration_summary/1"
            and summary.get("ended_utc") and protocol.get("label") == label
            and summary.get("label") == label, f"{scenario}: unfinished/non-dev campaign")
    require(protocol.get("model") == MODEL and protocol.get("bank_version") == "bank-v2"
            and protocol.get("repetitions") == 3 and protocol.get("budget_tokens_per_task") == 40000
            and protocol.get("reserve_tokens_per_call") == 12000
            and protocol.get("max_output_tokens_requested") == 2048,
            f"{scenario}: protocol differs")
    prefix = "spd-" if scenario == "speedup" else "tab-"
    ids = protocol.get("tasks")
    require(isinstance(ids, list) and len(ids) == 5 and len(set(ids)) == 5
            and all(isinstance(x, str) and x.startswith(prefix)
                    and TASK_ID.fullmatch(x) is not None for x in ids),
            f"{scenario}: need five unique task IDs")
    source_root = Path(protocol.get("source_root", "")).expanduser().resolve()
    require(source_root.is_dir() and source_root != root,
            f"{scenario}: calibration source worktree unavailable")
    source, copied = protocol.get("source_sha256"), protocol.get("copied_sha256")
    prior.verify_hashes(source_root, source, f"{scenario} source")
    prior.verify_hashes(campaign / "project", copied, f"{scenario} copy")
    require(all(key in source and key in copied for key in prior.COMMON),
            f"{scenario}: missing input hashes")
    for key in prior.COMMON[1:]:
        require(source[key] == copied[key] and (root / key).is_file()
                and sha(root / key) == source[key],
                f"{scenario}: calibration config drift: {key}")
    source_bank = read(source_root / "tasks/bank.json")
    copy_bank = read(campaign / "project/tasks/bank.json")
    require(source_bank.get("schema") == "bank/1" and source_bank.get("version") == "bank-v2"
            and source_bank.get("frozen") is False
            and isinstance(source_bank.get("tasks"), list)
            and [row.get("task_id") for row in source_bank["tasks"]] == ids
            and copy_bank == source_bank, f"{scenario}: temporary bank mismatch")
    project_tasks = campaign / "project/tasks"
    require({path.name for path in project_tasks.iterdir()} == set(ids) | {"bank.json"}
            and not any(path.name.startswith("holdout") for path in project_tasks.rglob("*")),
            f"{scenario}: copy has extra tasks or holdout data")
    prices = read(campaign / "project/config/prices_token_pilot.json")
    rates = prices.get("models", {}).get(MODEL, {})
    require(prices.get("currency") == "TOKENS" and prices.get("test_only") is True
            and prices.get("unit_tokens") == 1 and isinstance(rates, dict)
            and set(rates) == set(prior.BUCKETS)
            and all(rates[key] == 1 for key in prior.BUCKETS),
            f"{scenario}: price table differs")
    by_id = {row["task_id"]: row for row in source_bank["tasks"]}
    task_hashes, license_evidence = {}, {}
    for task_id in ids:
        entry = by_id[task_id]
        task = read(root / "tasks" / task_id / "task.json")
        require(task.get("task_id") == task_id and task.get("scenario") == scenario
                and task.get("budget", {}).get("money") == 40000
                and entry.get("tier") == "exam"
                and entry.get("score_formula") == ("ratio" if scenario == "speedup" else "gap")
                and positive(entry.get("baseline_raw")) and positive(entry.get("ceiling_raw"))
                and entry["ceiling_raw"] > entry["baseline_raw"],
                f"{task_id}: task or reference differs")
        if scenario == "tabular":
            require(entry["ceiling_raw"] - entry["baseline_raw"] >= 0.01
                    and task.get("metric", {}).get("baseline_raw") == entry["baseline_raw"]
                    and task.get("metric", {}).get("ceiling_raw") == entry["ceiling_raw"],
                    f"{task_id}: dev AUC gap below 0.01 or metric mismatch")
            for relative in ("SOURCE.md", "split_manifest.json",
                             "hidden/bank_v2_reference_measurements.json"):
                require((root / "tasks" / task_id / relative).is_file(),
                        f"{task_id}: missing provenance or dev evidence")
            source_md = root / "tasks" / task_id / "SOURCE.md"
            require(len(source_md.read_text(encoding="utf-8").strip()) >= 40,
                    f"{task_id}: provenance note too short")
            measure = read(root / "tasks" / task_id / "hidden/bank_v2_reference_measurements.json")
            require(measure.get("schema") == "tabular_bank_v2_reference/1"
                    and measure.get("task_id") == task_id
                    and measure.get("split") == "dev"
                    and measure.get("baseline", {}).get("auc") == entry["baseline_raw"]
                    and measure.get("reference", {}).get("auc") == entry["ceiling_raw"]
                    and measure.get("gap_valid") is True
                    and type(measure.get("gap")) in (int, float)
                    and math.isclose(measure["gap"], entry["ceiling_raw"] - entry["baseline_raw"],
                                     rel_tol=0, abs_tol=1e-12)
                    and measure.get("split_manifest_sha256") ==
                    sha(root / "tasks" / task_id / "split_manifest.json"),
                    f"{task_id}: dev reference evidence differs")
            license_evidence[task_id] = sha(source_md)
        hidden = "reference.py" if scenario == "speedup" else "dev.csv"
        prior.verify_hidden_copy(project_tasks / task_id / "hidden", hidden, task_id)
        source_task = read(source_root / "tasks" / task_id / "task.json")
        expected_task = deepcopy(source_task)
        require(isinstance(expected_task.get("budget"), dict),
                f"{task_id}: source budget missing")
        expected_task["budget"]["money"] = 40000
        require(read(project_tasks / task_id / "task.json") == expected_task
                and task == expected_task,
                f"{task_id}: target/copy task differs from allowed budget override")
        workspace = prior.workspace_files(root, task_id)
        require(workspace == prior.workspace_files(campaign / "project", task_id),
                f"{task_id}: workspace file set differs")
        relevant = workspace | {f"tasks/{task_id}/task.json", f"tasks/{task_id}/hidden/{hidden}"}
        hashes = {}
        for relative in relevant:
            require(relative in source and relative in copied and (root / relative).is_file(),
                    f"{task_id}: source/copy hash missing: {relative}")
            if relative == f"tasks/{task_id}/task.json":
                require(sha(root / relative) == copied[relative],
                        f"{task_id}: task budget override drift")
            else:
                require(sha(root / relative) == source[relative],
                        f"{task_id}: source file drift: {relative}")
                require(source[relative] == copied[relative],
                        f"{task_id}: copied file drift: {relative}")
            hashes[relative] = copied[relative] if relative.endswith("/task.json") else source[relative]
        task_hashes[task_id] = dict(sorted(hashes.items()))
    attempts = summary.get("attempts")
    require(isinstance(attempts, list) and len(attempts) == 15,
            f"{scenario}: expected 15 attempts")
    by_key = {}
    for record in attempts:
        require(isinstance(record, dict) and isinstance(record.get("task_id"), str)
                and type(record.get("rep")) is int, f"{scenario}: malformed attempt")
        key = record["task_id"], record["rep"]
        require(key not in by_key, f"{scenario}: duplicate attempt")
        by_key[key] = record
    require(set(by_key) == {(task_id, rep) for task_id in ids for rep in (1, 2, 3)},
            f"{scenario}: missing/extra attempt")
    rows, seen = [], set()
    for task_id in ids:
        evidence = [prior.audit_attempt(campaign, protocol, by_key[(task_id, rep)], task_id, rep)
                    for rep in (1, 2, 3)]
        require(not seen.intersection(item["run_id"] for item in evidence),
                f"{scenario}: duplicate run ID")
        seen.update(item["run_id"] for item in evidence)
        require(all(item["tokens"] <= 40000 and item["holdout_attempts"] == 0
                    for item in evidence), f"{task_id}: budget or holdout violation")
        completed = sum(item["status"] == "completed" for item in evidence)
        total = sum(item["tokens"] for item in evidence)
        rows.append({"task_id": task_id, "completed_repetitions": completed,
                     "attempts": evidence, "naive_cost": total / completed if completed >= 2 else None})
    return {"path": str(campaign), "protocol_sha256": sha(campaign / "protocol.json"),
            "summary_sha256": sha(campaign / "summary.json"),
            "source_bank_sha256": sha(source_root / "tasks/bank.json"),
            "source_root": str(source_root), "code_drift": runtime_drift(root, protocol.get("code_commit"), target_commit),
            "task_ids": ids, "rows": rows, "task_hashes": task_hashes,
            "license_evidence": license_evidence}


def propose(root: Path, speed_campaign: Path, tab_campaign: Path,
            target_commit: str) -> dict:
    root = root.expanduser().resolve()
    verify_target_worktree(root, target_commit)
    old_path = root / "tasks/bank.json"
    old = read(old_path)
    require(old.get("schema") == "bank/1" and old.get("version") == "bank-v1"
            and old.get("frozen") is True and isinstance(old.get("tasks"), list)
            and len(old["tasks"]) == 24, "Expected unchanged frozen bank-v1")
    retained = {"speedup": [], "tabular": []}
    old_exams = {"speedup": 0, "tabular": 0}
    old_ids = set()
    for row in old["tasks"]:
        task_id = row.get("task_id") if isinstance(row, dict) else None
        require(isinstance(task_id, str) and task_id not in old_ids,
                "bank-v1 has duplicate or invalid task IDs")
        old_ids.add(task_id)
        require(task_id.startswith(("spd-", "tab-"))
                and row.get("tier") in ("practice", "validation", "exam"),
                "bank-v1 has unknown scenario or tier")
        scenario = "speedup" if task_id.startswith("spd-") else "tabular"
        if row["tier"] == "exam":
            old_exams[scenario] += 1
        else:
            require(positive(row.get("naive_cost")), f"{task_id}: missing frozen naive_cost")
            retained[scenario].append(deepcopy(row))
    require(all(len(rows) == 7
                and Counter(row["tier"] for row in rows) == {"practice": 5, "validation": 2}
                for rows in retained.values())
            and old_exams == {"speedup": 5, "tabular": 5},
            "Expected 5 practice and 2 validation tasks per scenario")
    audits = {"speedup": audit_campaign(root, speed_campaign, "speedup", target_commit),
              "tabular": audit_campaign(root, tab_campaign, "tabular", target_commit)}
    new_ids = [task_id for scenario in ("speedup", "tabular")
               for task_id in audits[scenario]["task_ids"]]
    run_ids = [item["run_id"] for scenario in ("speedup", "tabular")
               for row in audits[scenario]["rows"] for item in row["attempts"]]
    require(len(run_ids) == len(set(run_ids)), "Cross-campaign run ID collision")
    require(len(new_ids) == 10 and len(set(new_ids)) == 10
            and not old_ids.intersection(new_ids),
            "New exam must contain ten IDs absent from bank-v1")
    blocked, groups = [], {}
    for scenario in ("speedup", "tabular"):
        audit = audits[scenario]
        if audit["code_drift"]["runtime_changed"]:
            blocked.append({"scenario": scenario,
                            "reason": "Runtime code changed after dev cost calibration; remeasure",
                            "paths": audit["code_drift"]["runtime_changed"]})
        measured = {row["task_id"]: row for row in audit["rows"]}
        for task_id in audit["task_ids"]:
            if measured[task_id]["completed_repetitions"] < 2:
                blocked.append({"scenario": scenario, "task_id": task_id,
                                "reason": "Fewer than two completed dev attempts"})
        candidate_bank = read(Path(audit["source_root"]) / "tasks/bank.json")
        new_rows = []
        for entry in candidate_bank["tasks"]:
            row = deepcopy(entry)
            row["naive_cost"] = measured[row["task_id"]]["naive_cost"]
            row["difficulty"] = None
            new_rows.append(row)
        combined = retained[scenario] + new_rows
        require(len(combined) == 12, f"{scenario}: expected twelve combined tasks")
        cost_pairs = [(row["naive_cost"], row["task_id"])
                      for row in combined if positive(row.get("naive_cost"))]
        if len(cost_pairs) == 12:
            levels = {task_id: LEVELS[index // 4]
                      for index, (_, task_id) in enumerate(sorted(cost_pairs))}
            for row in combined:
                row["difficulty"] = levels[row["task_id"]]
        groups[scenario] = combined
    tasks = groups["speedup"] + groups["tabular"]
    require(len(tasks) == 24 and len({row["task_id"] for row in tasks}) == 24,
            "Combined bank must contain 24 distinct tasks")
    return {"schema": "bank_v2_proposal/1", "ready": not blocked, "blocked": blocked,
            "label": "POST-FAILURE NEW EXAM PROPOSAL; NOT AN EXAM RESULT",
            "model": MODEL, "currency": "TOKENS", "budget_tokens_per_task": 40000,
            "source_bank_v1_sha256": sha(old_path), "target_code_commit": target_commit,
            "campaigns": audits, "license_evidence": audits["tabular"]["license_evidence"],
            "tasks": tasks}


def freeze(root: Path, proposal_path: Path, confirm_sha: str,
           license_review_path: Path, output_bank: Path) -> dict:
    root = root.expanduser().resolve()
    proposal_path, license_review_path = proposal_path.resolve(), license_review_path.resolve()
    require(prior.SHA.fullmatch(confirm_sha) is not None and sha(proposal_path) == confirm_sha,
            "Proposal confirmation SHA differs")
    proposal = read(proposal_path)
    require(proposal.get("schema") == "bank_v2_proposal/1" and proposal.get("ready") is True
            and proposal.get("blocked") == [], "Proposal is not ready")
    paths = proposal["campaigns"]
    fresh = propose(root, Path(paths["speedup"]["path"]),
                    Path(paths["tabular"]["path"]), proposal["target_code_commit"])
    require(fresh == proposal, "Proposal inputs changed after review")
    review = read(license_review_path)
    require(review.get("schema") == "bank_v2_license_review/1"
            and review.get("proposal_sha256") == confirm_sha
            and isinstance(review.get("entries"), list),
            "License review does not bind this proposal")
    approved = {}
    for item in review["entries"]:
        require(isinstance(item, dict) and isinstance(item.get("task_id"), str)
                and item["task_id"] not in approved
                and item.get("redistribution_permitted") is True
                and item.get("source_md_sha256") ==
                proposal["license_evidence"].get(item["task_id"]),
                "License review incomplete or stale")
        approved[item["task_id"]] = item
    require(set(approved) == set(proposal["license_evidence"]),
            "License review must cover all five new tabular tasks")
    output_bank = output_bank.expanduser().resolve()
    require(output_bank != root / "tasks/bank.json"
            and not output_bank.is_relative_to(root / "tasks"),
            "Cannot overwrite bank-v1 or task packages")
    bank = {"schema": "bank/1", "version": "bank-v2", "frozen": True,
            "tasks": proposal["tasks"]}
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    with output_bank.open("x", encoding="utf-8") as handle:
        json.dump(bank, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"schema": "bank_v2_freeze_receipt/1", "bank_path": str(output_bank),
            "bank_sha256": sha(output_bank), "proposal_sha256": confirm_sha,
            "source_bank_v1_sha256": proposal["source_bank_v1_sha256"],
            "target_code_commit": proposal["target_code_commit"],
            "task_count": len(bank["tasks"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("propose", help="Audit 15+15 dev attempts")
    draft.add_argument("--project-root", type=Path, required=True)
    draft.add_argument("--speedup-campaign", type=Path, required=True)
    draft.add_argument("--tabular-campaign", type=Path, required=True)
    draft.add_argument("--target-code-commit", required=True)
    draft.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("freeze", help="Require proposal SHA and license review")
    final.add_argument("--project-root", type=Path, required=True)
    final.add_argument("--proposal", type=Path, required=True)
    final.add_argument("--confirm-proposal-sha", required=True)
    final.add_argument("--license-review", type=Path, required=True)
    final.add_argument("--output-bank", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "propose":
            result = propose(args.project_root, args.speedup_campaign,
                             args.tabular_campaign, args.target_code_commit)
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(json.dumps({"ready": result["ready"], "blocked": result["blocked"],
                              "proposal_path": str(output), "proposal_sha256": sha(output)},
                             ensure_ascii=False))
            return 0 if result["ready"] else 1
        receipt = freeze(args.project_root, args.proposal,
                         args.confirm_proposal_sha, args.license_review, args.output_bank)
        print(json.dumps(receipt, ensure_ascii=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
