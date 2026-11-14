"""Audit dev-only naive token campaigns and propose bank values without applying them.

Campaigns can be copied from another machine. Stored absolute run paths are
used only for their run IDs and resolved within the supplied campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

BUCKETS = ("input", "cached_input", "output", "reasoning")
SCENARIOS = {"speedup": "spd-", "tabular": "tab-"}
COMMON = ("tasks/bank.json", "config/naive_token_pilot.json",
          "config/prompts/naive_token_pilot.md", "config/prices_token_pilot.json",
          "config/hard_caps.json", "config/triggers.json")
RUN_ID = re.compile(r"run-\d{8}-\d{6}-[0-9a-f]{4}")
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"Expected object: {path}")
    return data


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def verify_hashes(root: Path, hashes: Any, label: str) -> None:
    require(isinstance(hashes, dict) and hashes, f"{label}: missing SHA map")
    for relative, digest in hashes.items():
        require(isinstance(relative, str) and isinstance(digest, str)
                and SHA.fullmatch(digest) is not None, f"{label}: invalid SHA entry")
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha(path) == digest,
                f"{label}: changed or missing {relative}")


def bank_tasks(root: Path) -> tuple[dict[str, list[str]], str]:
    bank = read(root / "tasks/bank.json")
    require(bank.get("schema") == "bank/1" and isinstance(bank.get("tasks"), list)
            and isinstance(bank.get("version"), str), "Invalid source bank")
    ids: dict[str, list[str]] = {key: [] for key in SCENARIOS}
    for entry in bank["tasks"]:
        require(isinstance(entry, dict) and isinstance(entry.get("task_id"), str),
                "Invalid bank entry")
        task_id = entry["task_id"]
        scenario = next((key for key, prefix in SCENARIOS.items()
                         if task_id.startswith(prefix)), None)
        require(scenario is not None and task_id not in ids[scenario],
                f"Duplicate/unknown bank task: {task_id}")
        task = read(root / "tasks" / task_id / "task.json")
        require(task.get("scenario") == scenario and task.get("task_id") == task_id,
                f"Bank/task mismatch: {task_id}")
        ids[scenario].append(task_id)
    require(len(bank["tasks"]) == 24 and all(len(group) == 12 for group in ids.values()),
            "Expected 12 tasks per scenario")
    return ids, bank["version"]


def load_campaign(root: Path, campaign: Path, scenario: str,
                  ids: list[str], bank_version: str) -> tuple[dict, list[dict]]:
    protocol = read(campaign / "protocol.json")
    summary = read(campaign / "summary.json")
    require(protocol.get("schema") == "naive_token_calibration_protocol/1"
            and summary.get("schema") == "naive_token_calibration_summary/1"
            and summary.get("ended_utc"), f"{scenario}: unfinished/wrong campaign")
    require(protocol.get("label") == "DEV ONLY; TEST TOKEN UNITS; NO HOLDOUT OR MONEY"
            and summary.get("label") == protocol["label"], f"{scenario}: not dev-only")
    require(protocol.get("tasks") == ids and protocol.get("bank_version") == bank_version,
            f"{scenario}: task set or bank version differs")
    require(protocol.get("model") == "qwen3.5-27b" and protocol.get("repetitions") == 3
            and protocol.get("budget_tokens_per_task") == 40000
            and protocol.get("reserve_tokens_per_call") == 12000
            and protocol.get("max_output_tokens_requested") == 2048,
            f"{scenario}: protocol differs from preregistration")
    require(isinstance(protocol.get("code_commit"), str)
            and COMMIT.fullmatch(protocol["code_commit"]) is not None,
            f"{scenario}: invalid code commit")
    source = protocol.get("source_sha256")
    copied = protocol.get("copied_sha256")
    verify_hashes(root, source, f"{scenario} source")
    verify_hashes(campaign / "project", copied, f"{scenario} copy")
    require(all(key in source and key in copied for key in COMMON),
            f"{scenario}: common hashes incomplete")
    copied_bank = read(campaign / "project/tasks/bank.json")
    require(copied_bank.get("schema") == "bank/1"
            and copied_bank.get("frozen") is False
            and isinstance(copied_bank.get("tasks"), list)
            and all(isinstance(entry, dict) for entry in copied_bank["tasks"])
            and [entry.get("task_id") for entry in copied_bank.get("tasks", [])] == ids,
            f"{scenario}: copied bank differs")
    require(not any(path.name.startswith("holdout")
                    for path in (campaign / "project/tasks").rglob("*")),
            f"{scenario}: copied project contains holdout data")
    for task_id in ids:
        hidden = "reference.py" if scenario == "speedup" else "dev.csv"
        require(all(f"tasks/{task_id}/{suffix}" in source
                    for suffix in ("task.json", f"hidden/{hidden}")),
                f"{scenario}: source hashes incomplete for {task_id}")
        hidden_dir = campaign / "project/tasks" / task_id / "hidden"
        require(hidden_dir.is_dir() and
                {path.name for path in hidden_dir.iterdir()} == {hidden},
                f"{scenario}: unexpected hidden data for {task_id}")
        for path in (root / "tasks" / task_id / "workspace").rglob("*"):
            if path.is_file():
                key = str(path.relative_to(root))
                require(key in source and key in copied,
                        f"{scenario}: workspace SHA omitted for {key}")
        require(read(campaign / "project/tasks" / task_id / "task.json")
                .get("budget", {}).get("money") == 40000,
                f"{scenario}: copied budget differs for {task_id}")
    prices = read(campaign / "project/config/prices_token_pilot.json")
    rates = prices.get("models", {}).get(protocol["model"], {})
    require(prices.get("currency") == "TOKENS" and prices.get("test_only") is True
            and prices.get("unit_tokens") == 1 and isinstance(rates, dict)
            and set(rates) == set(BUCKETS)
            and all(rates[bucket] == 1 for bucket in BUCKETS),
            f"{scenario}: prices are not one TEST unit per token")
    attempts = summary.get("attempts")
    require(isinstance(attempts, list) and len(attempts) == 36,
            f"{scenario}: expected 36 attempts")
    return protocol, attempts


def compare_commits(root: Path, first: str, second: str) -> dict[str, Any]:
    if first == second:
        return {"code_commits": [first, second], "docs_only_exception": False,
                "changed_docs": []}
    for commit in (first, second):
        result = subprocess.run(["git", "-C", str(root), "cat-file", "-e",
                                 f"{commit}^{{commit}}"], capture_output=True)
        require(result.returncode == 0, f"Missing local Git commit: {commit}")
    result = subprocess.run(["git", "-C", str(root), "diff", "--name-only",
                             "--no-renames", first, second], text=True,
                            capture_output=True)
    require(result.returncode == 0, "Cannot compare code commits")
    changed = [path for path in result.stdout.splitlines() if path]
    require(changed and all(path.startswith("docs/") for path in changed),
            "Calibration commits differ outside docs/")
    return {"code_commits": [first, second], "docs_only_exception": True,
            "changed_docs": changed}


def audit_attempt(campaign: Path, protocol: dict, record: dict,
                  task_id: str, rep: int) -> dict[str, Any]:
    require(record.get("task_id") == task_id and record.get("rep") == rep,
            f"Wrong attempt identity: {task_id}/{rep}")
    status = record.get("status")
    require(status in {"completed", "incomplete", "failed"},
            f"Unfinished attempt: {task_id}/{rep}")
    raw = record.get("run_dir")
    require(isinstance(raw, str) and RUN_ID.fullmatch(Path(raw).name) is not None,
            f"Invalid run ID: {task_id}/{rep}")
    run_id = Path(raw).name
    run = campaign / "project/runs" / run_id
    require(run.is_dir(), f"Missing run directory: {run_id}")
    manifest = read(run / "manifest.json")
    require(manifest.get("schema") == "run/1" and manifest.get("run_id") == run_id
            and manifest.get("phase") == "dev" and manifest.get("config") == "naive"
            and manifest.get("tasks") == [task_id] and manifest.get("rep") == rep
            and manifest.get("model") == protocol["model"]
            and manifest.get("code_commit") == protocol["code_commit"]
            and manifest.get("bank_version") == protocol["bank_version"]
            and manifest.get("library", {}).get("start_version") == "lib-0"
            and manifest.get("library", {}).get("frozen") is False,
            f"Manifest mismatch: {task_id}/{rep}")
    copied = protocol["copied_sha256"]
    require(manifest.get("agent_config", {}).get("sha256") ==
            copied["config/naive_token_pilot.json"]
            and manifest.get("prices", {}).get("sha256") ==
            copied["config/prices_token_pilot.json"],
            f"Manifest input SHA mismatch: {task_id}/{rep}")
    task_dir = run / "tasks" / task_id
    require(not (task_dir / "holdout_attempt.json").exists()
            and not (task_dir / "holdout_score.json").exists(),
            f"Holdout opened: {task_id}/{rep}")
    ledger_file = task_dir / "ledger.jsonl"
    require(ledger_file.is_file(), f"Missing ledger: {task_id}/{rep}")
    events = [json.loads(line) for line in ledger_file.read_text(encoding="utf-8").splitlines()]
    require(bool(events) and all(isinstance(event, dict)
            and event.get("schema") == "ledger/1"
            and event.get("seq") == index and event.get("run_id") == run_id
            and event.get("task_id") == task_id
            for index, event in enumerate(events, 1)),
            f"Ledger sequence/identity mismatch: {task_id}/{rep}")
    require(all(event.get("split") in (None, "dev") for event in events),
            f"Non-dev score in ledger: {task_id}/{rep}")
    starts = [event for event in events if event.get("kind") == "task_start"]
    stops = [event for event in events if event.get("kind") == "stop"]
    require(len(starts) == 1 and starts[0].get("budget") == 40000,
            f"Task budget mismatch: {task_id}/{rep}")
    if status == "failed":
        require(len(stops) <= 1 and isinstance(record.get("error"), str)
                and bool(record["error"]), f"Failed attempt lacks error: {task_id}/{rep}")
    else:
        require(len(stops) == 1 and
                stops[0].get("reason") == record.get("stop_reason"),
                f"Stop evidence mismatch: {task_id}/{rep}")
        context = task_dir / "replay_context.json"
        require(context.is_file() and read(context).get("schema") == "a_replay_context/1",
                f"Missing replay context: {task_id}/{rep}")
    calls = [event for event in events if event.get("kind") == "llm_call"]
    require(bool(calls), f"No metered model call: {task_id}/{rep}")
    buckets = {bucket: 0 for bucket in BUCKETS}
    total_cost = 0.0
    for call in calls:
        usage = call.get("tokens")
        require(isinstance(usage, dict) and set(usage) == set(BUCKETS)
                and all(type(usage[bucket]) is int and usage[bucket] >= 0
                        for bucket in BUCKETS)
                and call.get("model") == protocol["model"]
                and number(call.get("cost")),
                f"Invalid model usage: {task_id}/{rep}")
        for bucket in BUCKETS:
            buckets[bucket] += usage[bucket]
        total_cost += call["cost"]
    tokens = sum(buckets.values())
    require(abs(tokens - total_cost) < 1e-6,
            f"Four-bucket/price mismatch: {task_id}/{rep}")
    field = "recorded_tokens" if status == "failed" else "total_tokens"
    require(number(record.get(field)) and abs(record[field] - tokens) < 1e-6,
            f"Summary/ledger token mismatch: {task_id}/{rep}")
    if status != "failed":
        require(record.get("token_buckets") == buckets
                and record.get("model_calls") == len(calls)
                and record.get("budget_respected") is (tokens <= 40000),
                f"Summary usage mismatch: {task_id}/{rep}")
        score = record.get("dev_score")
        require(isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score), f"Invalid dev score: {task_id}/{rep}")
    stop_reason = stops[0].get("reason") if stops else None
    require(status != "completed" or
            (tokens <= 40000 and stop_reason != "error"),
            f"Completed status invalid: {task_id}/{rep}")
    require(status != "incomplete" or
            (tokens > 40000 or stop_reason == "error"),
            f"Incomplete status invalid: {task_id}/{rep}")
    return {"task_id": task_id, "rep": rep, "run_id": run_id,
            "status": status, "stop_reason": stop_reason,
            "tokens": tokens, "token_buckets": buckets,
            "dev_score": record.get("dev_score"), "model_calls": len(calls),
            "holdout_attempts": 0}


def aggregate(project_root: Path, speedup_campaign: Path,
              tabular_campaign: Path) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    ids, version = bank_tasks(root)
    paths = {"speedup": speedup_campaign.expanduser().resolve(),
             "tabular": tabular_campaign.expanduser().resolve()}
    loaded = {scenario: load_campaign(root, paths[scenario], scenario,
                                      ids[scenario], version)
              for scenario in SCENARIOS}
    commits = compare_commits(root, loaded["speedup"][0]["code_commit"],
                              loaded["tabular"][0]["code_commit"])
    for key in COMMON:
        require(loaded["speedup"][0]["source_sha256"][key] ==
                loaded["tabular"][0]["source_sha256"][key],
                f"Cross-campaign SHA mismatch: {key}")
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    run_ids: set[str] = set()
    for scenario in SCENARIOS:
        protocol, attempts = loaded[scenario]
        by_key: dict[tuple[str, int], dict] = {}
        for record in attempts:
            require(isinstance(record, dict), f"{scenario}: malformed attempt")
            require(isinstance(record.get("task_id"), str)
                    and type(record.get("rep")) is int,
                    f"{scenario}: malformed task/rep")
            key = record.get("task_id"), record.get("rep")
            require(key not in by_key, f"Duplicate attempt: {key}")
            by_key[key] = record
        expected = {(task_id, rep) for task_id in ids[scenario]
                    for rep in (1, 2, 3)}
        require(set(by_key) == expected, f"{scenario}: missing/extra attempt")
        for task_id in ids[scenario]:
            evidence = [audit_attempt(paths[scenario], protocol,
                                      by_key[(task_id, rep)], task_id, rep)
                        for rep in (1, 2, 3)]
            for item in evidence:
                require(item["run_id"] not in run_ids, "Duplicate run ID")
                run_ids.add(item["run_id"])
            completed = sum(item["status"] == "completed" for item in evidence)
            total = sum(item["tokens"] for item in evidence)
            row = {"task_id": task_id, "scenario": scenario,
                   "completed_repetitions": completed, "attempted_repetitions": 3,
                   "token_costs": [item["tokens"] for item in evidence],
                   "statuses": [item["status"] for item in evidence],
                   "dev_scores": [item["dev_score"] for item in evidence],
                   "naive_cost": total / completed if completed >= 2 else None,
                   "difficulty": None, "attempts": evidence}
            rows.append(row)
            if completed < 2:
                blocked.append({"task_id": task_id,
                                "reason": "Fewer than two completed repetitions"})
    for scenario in SCENARIOS:
        ordered = sorted((row for row in rows if row["scenario"] == scenario
                          and row["naive_cost"] is not None),
                         key=lambda row: (row["naive_cost"], row["task_id"]))
        if len(ordered) == 12:
            for index, row in enumerate(ordered):
                row["difficulty"] = ("easy" if index < 4 else
                                     "medium" if index < 8 else "hard")
    ready = not blocked
    proposal = None
    if ready:
        keyed = {row["task_id"]: row for row in rows}
        proposal = {"schema": "naive_cost_bank_proposal/1",
                    "label": "DEV TOKEN RESOURCE PROPOSAL; NOT MONEY OR APPLIED",
                    "bank_version": version,
                    "source_bank_sha256": sha(root / "tasks/bank.json"),
                    "model": loaded["speedup"][0]["model"],
                    "currency": "TOKENS", "budget_tokens_per_task": 40000,
                    "code_commits": commits["code_commits"],
                    "updates": [{"task_id": task_id,
                                 "naive_cost": keyed[task_id]["naive_cost"],
                                 "difficulty": keyed[task_id]["difficulty"]}
                                for task_id in ids["speedup"] + ids["tabular"]]}
    return {"schema": "naive_cost_aggregation_audit/1",
            "label": "DEV ONLY; TEST TOKEN UNITS; NOT MONEY OR EXAM",
            "ready": ready, "bank_version": version,
            "source_bank_sha256": sha(root / "tasks/bank.json"),
            "campaigns": {name: str(paths[name]) for name in SCENARIOS},
            "commit_check": commits, "blocked": blocked,
            "tasks": rows, "bank_proposal": proposal}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--speedup-campaign", type=Path, required=True)
    parser.add_argument("--tabular-campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="New directory under project runs/; never overwrites")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    try:
        result = aggregate(root, args.speedup_campaign, args.tabular_campaign)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"schema": "naive_cost_aggregation_audit/1", "ready": False,
                  "label": "DEV ONLY; TEST TOKEN UNITS; NOT MONEY OR EXAM",
                  "blocked": [{"reason": str(exc)}], "bank_proposal": None}
    if args.output_dir is not None:
        output = args.output_dir.expanduser().resolve()
        require(output.is_relative_to(root / "runs"),
                "Output must be inside project runs/")
        output.mkdir(parents=True, exist_ok=False)
        (output / "audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if result["bank_proposal"] is not None:
            (output / "bank_proposal.json").write_text(
                json.dumps(result["bank_proposal"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
    print(json.dumps({"ready": result["ready"], "blocked": result["blocked"],
                      "proposal_generated": result["bank_proposal"] is not None},
                     ensure_ascii=False))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
