"""Read-only summary tables for a completed (or still running) formal batch.

Reads only files that already exist in the batch's project tree: plan.json,
batch_report.json, each run's manifest.json / report.json, each task's
ledger.jsonl and holdout_score.json, and tasks/bank.json.  It never calls a
model, scorer or sandbox, never calls run_report() / task_report() (they
rewrite or re-audit saved reports), never opens hidden task data, and refuses
to write inside the batch's own project tree.

All costs are internal TOKENS (the four token buckets all have rate 1).  They
are not cash prices and not official contest scores.

Usage (the output directory must be outside the batch's project tree):
  python scripts/summarize_formal_batch.py --batch-dir <project>/runs/batch-... \
      --output-dir <dir> [--name formal-batch-v2-summary] [--partial] [--research-cost 2094728]
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from statistics import mean
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.admission import admit_configuration  # noqa: E402
from evaluation.compare import compare_configurations  # noqa: E402


SCHEMA = "formal_batch_summary/1"
TIERS = ("practice", "validation", "exam")
CONFIGS = ("naive", "default", "default_lib", "tuned_lib")
CONFIG_LABELS = {"naive": "朴素版", "default": "默认设置／空库",
                 "default_lib": "默认设置／练过的库", "tuned_lib": "研究版／练过的库"}
BUCKETS = ("input", "cached_input", "output", "reasoning")
SECONDS = ("llm", "run", "queue", "wall")
STOP_LABELS = {"ceiling": "达到评分上限", "budget_cap": "预算闸门", "time": "时间上限",
               "planner": "规划者主动停止", "error": "错误"}
TASK_TOKEN_LIMIT = 40_000
SMALL_REFERENCE_GAP = 0.002
EXAM_PAIRS = (("naive", "default", "default_vs_naive"),
              ("default", "default_lib", "default_lib_vs_default"),
              ("default_lib", "tuned_lib", "tuned_lib_vs_default_lib"))
DIAGNOSTIC_NOTE = "仅诊断，不改正式成绩"
COMPARE_RULE_ZH = ("三遍配对分数的差必须全部大于 0，且平均分差必须大于两条线各自样本标准差中较大的一个"
                   "（evaluation/compare.py 的三遍规则）。")
ADMISSION_RULE_ZH = ("三遍配对中候选分数在任一遍都不得低于基线，且候选平均 token 必须更低"
                     "（evaluation/admission.py）。")
UNIT_NOTE = ("内部 TOKENS：input、cached_input、output、reasoning 四桶费率均为 1。"
             "不是现金价格，也不是官方比赛成绩。")


class SummaryError(ValueError):
    """The batch cannot be summarized (missing or malformed saved files)."""


def _issue(message: str, *, strict: bool, warnings: list[str]) -> None:
    if strict:
        raise SummaryError(message)
    warnings.append(message)


class LoadedRun(NamedTuple):
    phase: str
    report: dict
    manifest: dict | None
    run_dir: Path


# ---------------------------------------------------------------- file access

def _load_json(path: Path) -> dict:
    if path.is_symlink():
        raise SummaryError(f"拒绝符号链接输入：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SummaryError(f"不是 JSON 对象：{path}")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink():
        raise SummaryError(f"拒绝符号链接输入：{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger_rows(path: Path, *, strict: bool = False) -> list[dict]:
    """Ignore a half-written last line only while a run is still live."""
    if path.is_symlink():
        raise SummaryError(f"拒绝符号链接输入：{path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    last = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == last and not strict:
                break
            raise SummaryError(f"账本损坏（第 {index + 1} 行不是 JSON）：{path}")
        if not isinstance(row, dict):
            raise SummaryError(f"账本损坏（第 {index + 1} 行不是对象）：{path}")
        rows.append(row)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _seconds_between(start: Any, end: Any) -> float | None:
    try:
        first = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        last = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (last - first).total_seconds()


def _epoch(stamp: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _task_evidence(run_dir: Path, task_id: str, *, strict: bool, warnings: list[str]) -> dict:
    """Read one task's ledger and holdout file straight from disk."""
    tasks_root = (run_dir / "tasks").resolve()
    if tasks_root != run_dir / "tasks":
        raise SummaryError(f"任务目录为符号链接：{run_dir.name}")
    if (run_dir / "tasks" / task_id).is_symlink():
        raise SummaryError(f"任务路径为符号链接：{run_dir.name}/{task_id}")
    task_dir = (run_dir / "tasks" / task_id).resolve()
    if task_dir.parent != tasks_root:
        raise SummaryError(f"任务路径越出 run 目录：{run_dir.name}/{task_id}")
    evidence: dict[str, Any] = {"calls": None, "failed_calls": None, "stop_reasons": None,
                                "raw": None, "raw_saved": False, "saved_score": None}
    ledger_path = task_dir / "ledger.jsonl"
    if ledger_path.is_file():
        rows = _ledger_rows(ledger_path, strict=strict)
        calls = [row for row in rows if row.get("kind") == "llm_call"]
        evidence["calls"] = len(calls)
        evidence["failed_calls"] = sum(1 for row in calls if row.get("ok") is False)
        evidence["stop_reasons"] = [str(row.get("reason") or "未记录")
                                    for row in rows if row.get("kind") == "stop"]
        if strict:
            if any(row.get("schema") != "ledger/1" or row.get("seq") != index
                   or row.get("run_id") != run_dir.name or row.get("task_id") != task_id
                   for index, row in enumerate(rows, 1)):
                raise SummaryError(f"账本身份或序号不一致：{ledger_path}")
            for kind in ("task_start", "stop", "task_end"):
                if sum(row.get("kind") == kind for row in rows) != 1:
                    raise SummaryError(f"账本缺少唯一 {kind}：{ledger_path}")
            buckets = {key: 0 for key in BUCKETS}
            for call in calls:
                tokens = call.get("tokens")
                if (not isinstance(tokens, dict) or set(tokens) != set(BUCKETS)
                        or any(type(tokens[key]) is not int or tokens[key] < 0 for key in BUCKETS)
                        or _number(call.get("cost")) is None
                        or not math.isclose(float(call["cost"]), sum(tokens.values()),
                                            rel_tol=0, abs_tol=1e-8)):
                    raise SummaryError(f"账本调用成本或四桶不一致：{ledger_path}")
                for key in BUCKETS:
                    buckets[key] += tokens[key]
            ending = next(row for row in rows if row.get("kind") == "task_end")
            evidence["ledger_buckets"] = buckets
            evidence["ledger_cost"] = float(sum(buckets.values()))
            evidence["end_cost"] = _number(ending.get("total_cost"))
            evidence["end_score"] = _number(ending.get("holdout_score"))
    elif strict:
        raise SummaryError(f"缺少任务账本：{ledger_path}")
    else:
        warnings.append(f"{run_dir.name}/{task_id}：缺少 ledger.jsonl，停止原因与失败调用未统计")
    holdout_path = task_dir / "holdout_score.json"
    if holdout_path.is_file():
        saved = _load_json(holdout_path)
        if strict and (saved.get("schema") != "score/1" or saved.get("split") != "holdout"
                       or _number(saved.get("score")) is None):
            raise SummaryError(f"holdout_score.json 无效：{holdout_path}")
        raw = _number(saved.get("raw"))
        evidence["raw"], evidence["raw_saved"] = raw, raw is not None
        evidence["saved_score"] = _number(saved.get("score"))
    elif strict:
        raise SummaryError(f"缺少 holdout_score.json：{holdout_path}")
    else:
        warnings.append(f"{run_dir.name}/{task_id}：缺少 holdout_score.json，原始指标记为未保存")
    return evidence


# ------------------------------------------------------------ run summaries

def _summarize_run(loaded: LoadedRun, *, limit: float, strict: bool,
                   warnings: list[str]) -> dict:
    report = loaded.report
    run_id = report["run_id"]
    if (not isinstance(report.get("tasks"), list) or not report["tasks"]
            or any(not isinstance(item, dict) or not isinstance(item.get("task_id"), str)
                   for item in report["tasks"])):
        raise SummaryError(f"{run_id}：报告缺少有效任务列表")
    tasks = []
    for item in report["tasks"]:
        task_id = item["task_id"]
        evidence = _task_evidence(loaded.run_dir, task_id, strict=strict, warnings=warnings)
        cost = _number(item.get("total_cost"))
        score = _number(item.get("holdout_score"))
        stated_buckets = item.get("tokens")
        if (cost is None or score is None or not isinstance(stated_buckets, dict)
                or set(stated_buckets) != set(BUCKETS)
                or any(type(stated_buckets[key]) is not int or stated_buckets[key] < 0
                       for key in BUCKETS)):
            raise SummaryError(f"任务报告成本、分数或四桶无效：{run_id}/{task_id}")
        buckets = {key: stated_buckets[key] for key in BUCKETS}
        if not math.isclose(sum(buckets.values()), cost, rel_tol=0, abs_tol=1e-6):
            _issue(f"{run_id}/{task_id}：四桶之和 {sum(buckets.values()):,} 与任务成本 {cost:,.0f} 不一致",
                   strict=strict, warnings=warnings)
        if evidence["calls"] is not None and evidence["calls"] != item.get("model_calls"):
            _issue(f"{run_id}/{task_id}：账本 llm_call 行数 {evidence['calls']} 与报告 model_calls {item.get('model_calls')} 不一致",
                   strict=strict, warnings=warnings)
        if strict and (evidence["ledger_buckets"] != buckets
                       or not math.isclose(evidence["ledger_cost"], cost, rel_tol=0, abs_tol=1e-8)
                       or evidence["end_cost"] is None
                       or not math.isclose(evidence["end_cost"], cost, rel_tol=0, abs_tol=1e-8)
                       or evidence["end_score"] is None
                       or not math.isclose(evidence["end_score"], score, rel_tol=0, abs_tol=1e-12)):
            raise SummaryError(f"{run_id}/{task_id}：任务报告与账本四桶、成本或分数不一致")
        if strict and evidence["saved_score"] is None:
            raise SummaryError(f"{run_id}/{task_id}：holdout_score.json 缺少有效分数")
        if (evidence["saved_score"] is not None
                and not math.isclose(evidence["saved_score"], score, rel_tol=0, abs_tol=1e-12)):
            _issue(f"{run_id}/{task_id}：holdout_score.json 的分数与报告不一致",
                   strict=strict, warnings=warnings)
        seconds = item.get("seconds") if isinstance(item.get("seconds"), dict) else {}
        tasks.append({
            "task_id": task_id, "score": score, "tokens": cost, "buckets": buckets,
            "raw": evidence["raw"], "raw_saved": evidence["raw_saved"],
            "model_calls": int(item.get("model_calls", 0)),
            "failed_calls": evidence["failed_calls"],
            "stop_reasons": evidence["stop_reasons"],
            "seconds": {key: float(_number(seconds.get(key)) or 0.0) for key in SECONDS},
            "over_limit": cost > limit,
        })
    total = sum(task["tokens"] for task in tasks)
    reported_total = _number(report.get("total_cost"))
    if reported_total is None:
        raise SummaryError(f"{run_id}：报告 total_cost 无效")
    if not math.isclose(total, reported_total, rel_tol=1e-9, abs_tol=1e-6):
        _issue(f"{run_id}：任务成本合计 {total:,.0f} 与报告 total_cost {reported_total:,.0f} 不一致",
               strict=strict, warnings=warnings)
    stops = Counter(reason for task in tasks for reason in (task["stop_reasons"] or []))
    known_failed = [task["failed_calls"] for task in tasks if task["failed_calls"] is not None]
    peak = max(tasks, key=lambda task: task["tokens"])
    manifest = loaded.manifest or {}
    return {
        "phase": loaded.phase, "config": report["config"], "rep": report["rep"], "run_id": run_id,
        "started_ts": manifest.get("started_ts"), "ended_ts": manifest.get("ended_ts"),
        "run_elapsed_seconds": _seconds_between(manifest.get("started_ts"), manifest.get("ended_ts")),
        "task_count": len(tasks), "tokens_total": total,
        "buckets": {key: sum(task["buckets"][key] for task in tasks) for key in BUCKETS},
        "mean_score": mean(task["score"] for task in tasks),
        "seconds": {key: sum(task["seconds"][key] for task in tasks) for key in SECONDS},
        "model_calls": sum(task["model_calls"] for task in tasks),
        "failed_calls": sum(known_failed) if len(known_failed) == len(tasks) else None,
        "stop_reasons": dict(sorted(stops.items())),
        "max_task_tokens": peak["tokens"], "max_task_id": peak["task_id"],
        "over_limit_tasks": sum(1 for task in tasks if task["over_limit"]),
        "tasks": tasks,
    }


def _config_summary(runs: list[dict]) -> dict:
    out = {}
    for config in CONFIGS:
        items = sorted((run for run in runs if run["config"] == config), key=lambda run: run["rep"])
        if not items:
            continue
        scores = [run["mean_score"] for run in items]
        tokens = [run["tokens_total"] for run in items]
        out[config] = {"reps": [run["rep"] for run in items],
                       "run_ids": [run["run_id"] for run in items],
                       "scores": scores, "score_mean": mean(scores),
                       "tokens_by_rep": tokens, "tokens_sum": sum(tokens),
                       "tokens_mean": mean(tokens)}
    return out


def _per_task(runs: list[dict], gaps: dict[str, float], small_gap: float) -> list[dict]:
    rows = []
    if not runs:
        return rows
    for task_id in [task["task_id"] for task in runs[0]["tasks"]]:
        for config in CONFIGS:
            reps = []
            for run in sorted((item for item in runs if item["config"] == config),
                              key=lambda item: item["rep"]):
                task = next((t for t in run["tasks"] if t["task_id"] == task_id), None)
                if task is not None:
                    reps.append({"rep": run["rep"], "run_id": run["run_id"],
                                 "score": task["score"], "raw": task["raw"],
                                 "raw_saved": task["raw_saved"], "tokens": task["tokens"],
                                 "stop_reasons": task["stop_reasons"],
                                 "failed_calls": task["failed_calls"]})
            if reps:
                gap = gaps.get(task_id)
                rows.append({"task_id": task_id, "config": config, "reps": reps,
                             "score_mean": mean(rep["score"] for rep in reps),
                             "tokens_mean": mean(rep["tokens"] for rep in reps),
                             "reference_gap": gap,
                             "small_reference_gap": gap is not None and gap < small_gap})
    return rows


# --------------------------------------------------------- batch loading

def _read_bank(root: Path, plan: dict, warnings: list[str], *, strict: bool) -> tuple[dict, dict]:
    path = root / "tasks" / "bank.json"
    if not path.is_file():
        _issue("缺少 tasks/bank.json：无法标记小参考差表格题", strict=strict, warnings=warnings)
        return {}, {"bank_sha256": None, "bank_sealed_match": None}
    bank = _load_json(path)
    if strict and (bank.get("schema") != "bank/1" or bank.get("frozen") is not True
                   or bank.get("version") != plan.get("bank_version")):
        raise SummaryError("tasks/bank.json 与正式批次计划身份不一致")
    tasks = bank.get("tasks") if isinstance(bank.get("tasks"), list) else []
    digest = _sha256(path)
    sealed = plan.get("sealed_inputs") if isinstance(plan.get("sealed_inputs"), dict) else {}
    sealed_digest = next((value for key, value in sealed.items()
                          if str(key).endswith("/tasks/bank.json")), None)
    match = None if sealed_digest is None else sealed_digest == digest
    if match is not True:
        _issue("tasks/bank.json 缺少封存 SHA-256 或与 plan.json 不一致",
               strict=strict, warnings=warnings)
    return ({item["task_id"]: item for item in tasks if isinstance(item, dict) and "task_id" in item},
            {"bank_sha256": digest, "bank_sealed_match": match})


def _reference_gaps(bank_by_id: dict, task_ids: list[str], small_gap: float,
                    warnings: list[str]) -> tuple[dict[str, float], list[dict]]:
    gaps: dict[str, float] = {}
    table = []
    for task_id in task_ids:
        if not task_id.startswith("tab-"):
            continue
        entry = bank_by_id.get(task_id)
        baseline = _number(entry.get("baseline_raw")) if entry else None
        ceiling = _number(entry.get("ceiling_raw")) if entry else None
        if baseline is None or ceiling is None:
            warnings.append(f"tasks/bank.json 缺少 {task_id} 的 baseline_raw / ceiling_raw")
            continue
        gaps[task_id] = ceiling - baseline
        table.append({"task_id": task_id, "tier": entry.get("tier"),
                      "baseline_raw": baseline, "ceiling_raw": ceiling,
                      "gap": ceiling - baseline, "small_gap": ceiling - baseline < small_gap})
    return gaps, table


def _manifest_for(run_dir: Path, *, strict: bool, warnings: list[str]) -> dict | None:
    path = run_dir / "manifest.json"
    if path.is_file():
        return _load_json(path)
    if strict:
        raise SummaryError(f"缺少 manifest.json：{path}")
    warnings.append(f"{run_dir.name}：缺少 manifest.json")
    return None


def _run_dir(root: Path, run_id: Any) -> Path:
    runs = (root / "runs").resolve()
    run_dir = (runs / str(run_id)).resolve()
    if run_dir.parent != runs:
        raise SummaryError(f"run_id 越出 runs/：{run_id}")
    return run_dir


def _load_complete(root: Path, batch: dict, warnings: list[str]) -> list[LoadedRun]:
    rows, embedded = batch.get("runs"), batch.get("reports")
    if (not isinstance(rows, list) or not isinstance(embedded, dict)
            or set(embedded) != set(TIERS)
            or any(not isinstance(embedded[tier], list) for tier in TIERS)):
        raise SummaryError("batch_report.json 缺少 runs 或 reports")
    row_by_id = {row.get("run_id"): row for row in rows if isinstance(row, dict)}
    if len(row_by_id) != len(rows):
        raise SummaryError("batch_report.json 的 runs 含重复或无效条目")
    loaded: list[LoadedRun] = []
    seen: set[str] = set()
    for tier in TIERS:
        for report in embedded.get(tier) or []:
            if not isinstance(report, dict):
                raise SummaryError(f"batch_report.json 的 {tier} 报告不是对象")
            run_id = report.get("run_id")
            row = row_by_id.get(run_id)
            if (report.get("schema") != "run_report/1" or run_id in seen or row is None
                    or row.get("logical_phase") != tier
                    or row.get("config") != report.get("config")
                    or row.get("rep") != report.get("rep")):
                raise SummaryError(f"batch_report.json 的 runs 与 reports 不一致：{run_id}")
            seen.add(run_id)
            run_dir = _run_dir(root, run_id)
            saved = run_dir / "report.json"
            if not saved.is_file() or _load_json(saved) != report:
                raise SummaryError(f"{run_id}：已保存的 report.json 缺失或与 batch_report.json 内嵌报告不一致")
            manifest = _manifest_for(run_dir, strict=True, warnings=warnings)
            if (manifest.get("schema") != "run/1" or manifest.get("run_id") != run_id
                    or manifest.get("phase") != tier or manifest.get("config") != report.get("config")
                    or manifest.get("rep") != report.get("rep")
                    or manifest.get("tasks") != [task["task_id"] for task in report["tasks"]]):
                raise SummaryError(f"{run_id}：manifest.json 与报告身份或任务顺序不一致")
            loaded.append(LoadedRun(tier, report, manifest, run_dir))
    if set(row_by_id) != seen:
        raise SummaryError("batch_report.json 的 runs 里有未附报告的 run")
    return loaded


def _progress(run_dir: Path, manifest: dict) -> dict:
    tasks = manifest.get("tasks") if isinstance(manifest.get("tasks"), list) else []
    done, spent, current = 0, 0.0, None
    for task_id in tasks:
        path = run_dir / "tasks" / str(task_id) / "ledger.jsonl"
        if not path.is_file():
            continue
        try:
            rows = _ledger_rows(path)
        except SummaryError:
            continue
        spent += sum(_number(row.get("cost")) or 0.0 for row in rows if row.get("kind") == "llm_call")
        if any(row.get("kind") == "task_end" for row in rows):
            done += 1
        else:
            current = task_id
    return {"run_id": run_dir.name, "phase": manifest.get("phase"), "config": manifest.get("config"),
            "rep": manifest.get("rep"), "started_ts": manifest.get("started_ts"),
            "tasks_done": done, "tasks_total": len(tasks), "tokens_so_far": spent,
            "current_task": current}


def _load_partial(root: Path, plan: dict, since: float,
                  warnings: list[str]) -> tuple[list[LoadedRun], list[dict], list[str]]:
    loaded: list[LoadedRun] = []
    running: list[dict] = []
    reserved: list[str] = []
    plan_tasks = plan.get("tasks") if isinstance(plan.get("tasks"), dict) else {}
    configs = plan.get("configs") if isinstance(plan.get("configs"), list) else list(CONFIGS)
    runs_dir = root / "runs"
    for run_dir in sorted(runs_dir.glob("run-*")) if runs_dir.is_dir() else []:
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        try:
            manifest = _load_json(manifest_path) if manifest_path.is_file() else None
        except (json.JSONDecodeError, SummaryError):
            manifest = None  # manifest is being written right now
        if manifest is None:
            if run_dir.stat().st_mtime >= since:
                reserved.append(run_dir.name)
            continue
        started = _epoch(manifest.get("started_ts"))
        phase = manifest.get("phase")
        if (manifest.get("schema") != "run/1" or phase not in TIERS or started is None
                or started < since or manifest.get("bank_version") != plan.get("bank_version")
                or manifest.get("config") not in configs
                or (phase in plan_tasks and manifest.get("tasks") != plan_tasks[phase])):
            continue
        report_path = run_dir / "report.json"
        if not report_path.is_file():
            running.append({**_progress(run_dir, manifest), "status": "进行中"})
            continue
        try:
            report = _load_json(report_path)
        except (json.JSONDecodeError, SummaryError):
            running.append({**_progress(run_dir, manifest), "status": "report.json 正在写入"})
            continue
        if (report.get("schema") != "run_report/1" or report.get("run_id") != run_dir.name
                or report.get("phase") != phase or report.get("config") != manifest.get("config")
                or report.get("rep") != manifest.get("rep")):
            warnings.append(f"{run_dir.name}：report.json 与 manifest 的身份不一致，已跳过")
            continue
        loaded.append(LoadedRun(phase, report, manifest, run_dir))
    return loaded, running, reserved


# ------------------------------------------------------------- comparisons

def _direction(differences: list[float]) -> str:
    if all(value > 0 for value in differences):
        return "all_higher"
    if all(value < 0 for value in differences):
        return "all_lower"
    return "mixed"


def _pair_reports(reports_by: dict, phase: str, baseline: str, candidate: str) -> list[dict] | None:
    base = reports_by.get((phase, baseline), {})
    trial = reports_by.get((phase, candidate), {})
    if set(base) != {1, 2, 3} or set(trial) != {1, 2, 3}:
        return None
    return [base[rep] for rep in (1, 2, 3)] + [trial[rep] for rep in (1, 2, 3)]


def _compare(pair: list[dict], baseline: str, candidate: str) -> dict:
    result = compare_configurations(pair, baseline=baseline, candidate=candidate)
    base_costs = [float(report["total_cost"]) for report in pair[:3]]
    trial_costs = [float(report["total_cost"]) for report in pair[3:]]
    cost_differences = [after - before for before, after in zip(base_costs, trial_costs)]
    base_mean = result["baseline_mean_cost"]
    return {
        "available": True, **result,
        "baseline_costs": base_costs, "candidate_costs": trial_costs,
        "cost_differences": cost_differences,
        "mean_cost_gap": mean(cost_differences),
        "mean_cost_change_pct": (result["candidate_mean_cost"] / base_mean - 1) * 100 if base_mean else None,
        "score_direction": _direction(result["paired_differences"]),
        "cost_direction": _direction(cost_differences),
        "cost_direction_consistent": _direction(cost_differences) != "mixed",
    }


def _exam_comparisons(reports_by: dict, *, tuned_exam_ran: bool) -> dict:
    out = {}
    for baseline, candidate, key in EXAM_PAIRS:
        if candidate == "tuned_lib" and not tuned_exam_ran:
            out[key] = {"available": False, "baseline": baseline, "candidate": candidate,
                        "reason": "研究版未获准入或未实际跑考试，不比较"}
            continue
        pair = _pair_reports(reports_by, "exam", baseline, candidate)
        if pair is None:
            out[key] = {"available": False, "baseline": baseline, "candidate": candidate,
                        "reason": "考试上两条线还没有各自完整的三遍 run"}
        else:
            out[key] = _compare(pair, baseline, candidate)
    return out


def _admission_pairs(reports_by: dict) -> list[dict]:
    base, trial = reports_by.get(("validation", "default_lib"), {}), reports_by.get(("validation", "tuned_lib"), {})
    rows = []
    for rep in (1, 2, 3):
        if rep in base and rep in trial:
            b_score = mean(float(task["holdout_score"]) for task in base[rep]["tasks"])
            t_score = mean(float(task["holdout_score"]) for task in trial[rep]["tasks"])
            b_cost, t_cost = float(base[rep]["total_cost"]), float(trial[rep]["total_cost"])
            rows.append({"rep": rep, "default_lib_run": base[rep]["run_id"], "tuned_lib_run": trial[rep]["run_id"],
                         "default_lib_score": b_score, "tuned_lib_score": t_score,
                         "score_difference": t_score - b_score,
                         "default_lib_tokens": b_cost, "tuned_lib_tokens": t_cost,
                         "token_difference": t_cost - b_cost})
    return rows


def _breakeven(*, complete: bool, admission: dict | None, comparison: dict | None,
               exam_tasks: int, research_cost: float | None) -> dict:
    reasons = []
    if not complete:
        reasons.append("批次尚未完成（没有 batch_report.json）")
    elif admission is None:
        reasons.append("batch_report.json 没有 validation 准入记录")
    elif admission.get("claim") != "admitted":
        reasons.append(f"研究版未通过 validation 准入（claim={admission.get('claim')}），未进入考试")
    elif comparison is None or not comparison.get("available"):
        reasons.append("缺少考试上研究版与 default_lib 的三遍配对")
    else:
        if any(value < 0 for value in comparison["paired_differences"]):
            reasons.append("考试上研究版相对 default_lib 在某一遍分数下降（不满足“分数不降”）")
        if not comparison["candidate_mean_cost"] < comparison["baseline_mean_cost"]:
            reasons.append("考试上研究版平均 token 不低于 default_lib（不满足“平均花费更低”）")
    if research_cost is None:
        reasons.append("没有传入研究总成本（--research-cost）")
    if reasons:
        return {"computed": False, "display": "不计算", "reasons": reasons}
    savings = (comparison["baseline_mean_cost"] - comparison["candidate_mean_cost"]) / exam_tasks
    return {"computed": True, "research_cost_tokens": research_cost,
            "savings_per_task_tokens": savings, "exam_tasks": exam_tasks,
            "breakeven_tasks": math.ceil(research_cost / savings),
            "rule": "准入为 admitted，且考试上研究版三遍分数均不低于 default_lib、平均 token 更低；"
                    "回本点 = ceil(研究总成本 / 每题平均节省 token)，每题节省 = 两线每遍平均 token 之差 / 考试题数。"}


def _filtered(report: dict, excluded: set[str]) -> dict:
    tasks = [task for task in report["tasks"] if task["task_id"] not in excluded]
    return {**report, "tasks": tasks, "total_cost": sum(float(task["total_cost"]) for task in tasks)}


def _diagnostic(loaded: list[LoadedRun], gaps: dict[str, float], small_gap: float,
                *, tuned_exam_ran: bool, allow_admission: bool) -> dict:
    excluded = {task_id for task_id, gap in gaps.items() if gap < small_gap}
    result: dict[str, Any] = {"note": DIAGNOSTIC_NOTE, "threshold": small_gap,
                              "excluded_tasks": [{"task_id": task_id, "gap": gaps[task_id]}
                                                 for task_id in sorted(excluded)],
                              "phases": {}}
    for phase in TIERS:
        members = [item for item in loaded if item.phase == phase]
        present = [task["task_id"] for task in members[0].report["tasks"]] if members else []
        cut = [task_id for task_id in present if task_id in excluded]
        if not cut:
            continue
        kept = [task_id for task_id in present if task_id not in excluded]
        entry: dict[str, Any] = {"excluded_in_phase": cut, "remaining_tasks": kept}
        if kept:
            reports_by: dict = {}
            runs = []
            for item in members:
                view = _filtered(item.report, excluded)
                reports_by.setdefault((phase, view["config"]), {})[view["rep"]] = view
                runs.append({"config": view["config"], "rep": view["rep"], "run_id": view["run_id"],
                             "mean_score": mean(float(task["holdout_score"]) for task in view["tasks"]),
                             "tokens_total": view["total_cost"]})
            runs.sort(key=lambda run: (CONFIGS.index(run["config"]), run["rep"]))
            entry["runs"] = runs
            entry["config_summary"] = _config_summary(runs)
            if phase == "validation" and allow_admission:
                pair = _pair_reports(reports_by, "validation", "default_lib", "tuned_lib")
                entry["admission_replay"] = admit_configuration(
                    pair, baseline="default_lib", candidate="tuned_lib") if pair else None
            if phase == "exam":
                entry["comparisons"] = _exam_comparisons(reports_by, tuned_exam_ran=tuned_exam_ran)
        result["phases"][phase] = entry
    return result


# --------------------------------------------------------------- the summary

def summarize(batch_dir: Path, *, partial: bool = False, research_cost: float | None = None,
              task_token_limit: float = TASK_TOKEN_LIMIT,
              small_gap: float = SMALL_REFERENCE_GAP) -> dict:
    """Return the summary dict; raises SummaryError for unusable input."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    root = batch_dir.parent.parent
    if batch_dir.parent.name not in {"runs", "batches"}:
        raise SummaryError("批次目录必须在项目的 runs/ 或 batches/ 下")
    plan_path = batch_dir / "plan.json"
    if not plan_path.is_file():
        raise SummaryError(f"缺少 plan.json：{plan_path}")
    plan = _load_json(plan_path)
    if plan.get("schema") != "batch_plan/1" or plan.get("label") != "FORMAL":
        raise SummaryError("plan.json 不是 FORMAL batch_plan/1")
    report_path = batch_dir / "batch_report.json"
    complete = report_path.is_file()
    if not complete and not partial:
        raise SummaryError("批次尚未完成：缺少 batch_report.json（正式批次全部跑完才会写出）。"
                           "如只想看已写出 report.json 的 run，请加 --partial。")
    warnings: list[str] = []
    notes: list[str] = []
    batch: dict = {}
    running: list[dict] = []
    reserved: list[str] = []
    if complete:
        batch = _load_json(report_path)
        if batch.get("schema") != "batch_report/1" or batch.get("label") != "FORMAL":
            raise SummaryError("batch_report.json 不是 FORMAL batch_report/1")
        if partial:
            notes.append("批次已写出 batch_report.json，--partial 不起作用，按完整批次汇总。")
        loaded = _load_complete(root, batch, warnings)
    else:
        loaded, running, reserved = _load_partial(root, plan, plan_path.stat().st_mtime, warnings)
    bank_by_id, bank_sources = _read_bank(root, plan, warnings, strict=complete)

    runs = []
    for item in loaded:
        runs.append(_summarize_run(item, limit=task_token_limit, strict=complete, warnings=warnings))
    order = {id(item): (TIERS.index(item.phase), CONFIGS.index(item.report["config"]), item.report["rep"],
                        str((item.manifest or {}).get("started_ts")))
             for item in loaded}
    paired = sorted(zip(loaded, runs), key=lambda pair: order[id(pair[0])])
    loaded, runs = [pair[0] for pair in paired], [pair[1] for pair in paired]

    reports_by: dict = {}
    kept_loaded: list[LoadedRun] = []
    kept_runs: list[dict] = []
    for item, run in zip(loaded, runs):
        slot = reports_by.setdefault((item.phase, run["config"]), {})
        if run["rep"] in slot:
            _issue(f"{item.phase}/{run['config']} 第 {run['rep']} 遍有重复 run（{run['run_id']}）",
                   strict=complete, warnings=warnings)
            continue
        slot[run["rep"]] = item.report
        kept_loaded.append(item)
        kept_runs.append(run)

    plan_tasks = plan.get("tasks") if isinstance(plan.get("tasks"), dict) else {}
    if complete:
        for item in kept_loaded:
            if [task["task_id"] for task in item.report["tasks"]] != plan_tasks.get(item.phase):
                raise SummaryError(f"{item.report['run_id']}：报告任务顺序与 plan.json 不一致")
    task_ids = [task_id for tier in TIERS for task_id in plan_tasks.get(tier, [])]
    gaps, gap_table = _reference_gaps(bank_by_id, task_ids, small_gap, warnings)

    by_phase = {phase: [run for run in kept_runs if run["phase"] == phase] for phase in TIERS}
    config_summary = {phase: _config_summary(by_phase[phase]) for phase in TIERS if by_phase[phase]}
    per_task = {phase: _per_task(by_phase[phase], gaps, small_gap) for phase in TIERS if by_phase[phase]}

    admission = batch.get("admission") if complete else None
    tuned_exam_ran = bool(reports_by.get(("exam", "tuned_lib")))
    recomputed_matches = None
    admission_source = "batch_report.json" if admission is not None else None
    validation_pair = _pair_reports(reports_by, "validation", "default_lib", "tuned_lib")
    if complete:
        if validation_pair is None or not isinstance(admission, dict):
            raise SummaryError("完整批次缺少三遍 validation 配对或准入记录")
        recomputed = admit_configuration(validation_pair, baseline="default_lib", candidate="tuned_lib")
        material_fields = ("schema", "claim", "baseline", "candidate",
                           "score_not_lower_each_rep", "cost_reduced_on_average",
                           "comparison")
        recomputed_matches = (all(admission.get(key) == recomputed[key] for key in material_fields)
                              and ("rule" not in admission or admission["rule"] == recomputed["rule"]))
        if not recomputed_matches:
            raise SummaryError("batch_report.json 的 admission 与按已保存 validation 报告重算的结果不一致")
        admitted = admission["claim"] == "admitted"
        expected = {("practice", config, rep) for config in CONFIGS for rep in (1, 2, 3)}
        expected |= {("validation", config, rep) for config in ("default_lib", "tuned_lib")
                     for rep in (1, 2, 3)}
        expected |= {("exam", config, rep)
                     for config in (CONFIGS if admitted else CONFIGS[:3]) for rep in (1, 2, 3)}
        actual = {(item.phase, item.report["config"], item.report["rep"]) for item in kept_loaded}
        if (len(kept_loaded) != len(expected) or actual != expected
                or batch.get("tuned_exam_skipped") is not (not admitted)
                or tuned_exam_ran is not admitted):
            raise SummaryError("完整批次运行数量或研究版考试准入状态不一致")
    exam_comparisons = _exam_comparisons(reports_by, tuned_exam_ran=tuned_exam_ran)
    exam_ids = [task["task_id"] for task in next(
        (item.report["tasks"] for item in kept_loaded if item.phase == "exam"), [])]

    currencies = sorted({str(item.report.get("currency")) for item in kept_loaded})
    exact = bool(kept_loaded) and all(item.report.get("exact_token_accounting") is True for item in kept_loaded)
    if kept_loaded and not (currencies == ["TOKENS"] and exact):
        _issue(f"存在非 TOKENS 币种或未核验精确计量的报告（币种：{'、'.join(currencies)}），"
               "成本数字不能按内部 token 解读", strict=complete, warnings=warnings)
    expected_runs = {"practice": len(plan.get("configs", [])) * len(plan.get("reps", []))}
    trained = [config for config in plan.get("configs", []) if config in ("default_lib", "tuned_lib")]
    if plan.get("run_validation"):
        expected_runs["validation"] = len(trained) * len(plan.get("reps", []))
    if plan.get("run_exam"):
        base_exam = len([c for c in plan.get("configs", []) if c != "tuned_lib"]) * len(plan.get("reps", []))
        expected_runs["exam"] = [base_exam, base_exam + (len(plan.get("reps", [])) if "tuned_lib" in plan.get("configs", []) else 0)]
    comparison = exam_comparisons["tuned_lib_vs_default_lib"]
    summary = {
        "schema": SCHEMA, "mode": "complete" if complete else "partial", "complete": complete,
        "label": batch.get("label") or plan.get("label"), "bank_version": plan.get("bank_version"),
        "batch_dir": str(batch_dir), "currency": currencies[0] if len(currencies) == 1 else (currencies or None),
        "exact_token_accounting": exact, "unit_note": UNIT_NOTE,
        "thresholds": {"task_token_limit": task_token_limit, "small_reference_gap": small_gap},
        "sources": {"plan_sha256": _sha256(plan_path),
                    "batch_report_sha256": _sha256(report_path) if complete else None,
                    **bank_sources},
        "plan": {"tasks": plan_tasks, "configs": plan.get("configs"), "reps": plan.get("reps"),
                 "max_workers": plan.get("max_workers")},
        "progress": {"expected_runs": expected_runs,
                     "completed_runs": {phase: len(by_phase[phase]) for phase in TIERS},
                     "running": running, "reserved_without_manifest": reserved},
        "runs": kept_runs, "config_summary": config_summary, "per_task": per_task,
        "validation": {"admission": admission, "admission_source": admission_source,
                       "admission_recomputed_matches": recomputed_matches,
                       "pairs": _admission_pairs(reports_by)},
        "exam": {"tuned_exam_skipped": batch.get("tuned_exam_skipped") if complete else None,
                 "comparisons": exam_comparisons},
        "reference_gaps": gap_table,
        "diagnostic_excluding_small_gap": _diagnostic(
            kept_loaded, gaps, small_gap, tuned_exam_ran=tuned_exam_ran, allow_admission=complete),
        "breakeven": _breakeven(complete=complete, admission=admission if complete else None,
                                comparison=comparison, exam_tasks=len(exam_ids) or 1,
                                research_cost=research_cost),
        "notes": notes, "warnings": warnings,
    }
    return summary


# ------------------------------------------------------------------ Markdown

def _int(value: Any) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _avg(value: Any) -> str:
    return "—" if value is None else f"{value:,.1f}"


def _score(value: Any) -> str:
    return "—" if value is None else f"{value:.4f}"


def _raw(value: Any) -> str:
    return "未保存" if value is None else f"{value:.10g}"


def _yes(value: Any) -> str:
    return "是" if value is True else "否" if value is False else "—"


def _triple(values: list[str]) -> str:
    return " / ".join(values)


def _table(headers: list[str], rows: list[list[Any]], right: tuple[int, ...] = ()) -> str:
    sep = "|" + "|".join("---:" if index in right else "---" for index in range(len(headers))) + "|"
    return "\n".join(["| " + " | ".join(headers) + " |", sep,
                      *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)])


def _stops(counts: dict) -> str:
    if not counts:
        return "—"
    return "、".join(f"{STOP_LABELS.get(key, key)} {value}" for key, value in counts.items())


DIRECTION = {"all_higher": "三遍均更高", "all_lower": "三遍均更低", "mixed": "方向不一致"}
CLAIM = {"better": "有明确优势（better）", "no_clear_difference": "无明确差异（no_clear_difference）"}


def _comparison_md(title: str, comp: dict) -> list[str]:
    if not comp.get("available"):
        return [f"### {title}", "", f"不比较：{comp.get('reason')}。", ""]
    rows = []
    for index in range(3):
        rows.append([index + 1, _score(comp["baseline_scores"][index]), _score(comp["candidate_scores"][index]),
                     f"{comp['paired_differences'][index]:+.4f}", _int(comp["baseline_costs"][index]),
                     _int(comp["candidate_costs"][index]), f"{comp['cost_differences'][index]:+,.0f}"])
    pct = comp["mean_cost_change_pct"]
    return [
        f"### {title}（基线 `{comp['baseline']}`，候选 `{comp['candidate']}`）", "",
        _table(["遍次", "基线平均分", "候选平均分", "分差（候选−基线）", "基线 token", "候选 token", "token 差（候选−基线）"],
               rows, right=(0, 1, 2, 3, 4, 5, 6)), "",
        f"- 三遍规则判定：**{CLAIM.get(comp['claim'], comp['claim'])}**；平均分差 {comp['mean_score_gap']:+.4f}，"
        f"两条线组内样本标准差的较大者 {comp['within_config_noise']:.4f}；分数：{DIRECTION[comp['score_direction']]}。",
        f"- 平均 token：基线 {_avg(comp['baseline_mean_cost'])} → 候选 {_avg(comp['candidate_mean_cost'])}"
        f"（差 {comp['mean_cost_gap']:+,.1f}" + (f"，{pct:+.1f}%" if pct is not None else "") + f"）；"
        f"逐遍 token 差：{DIRECTION[comp['cost_direction']]}（方向{'一致' if comp['cost_direction_consistent'] else '不一致'}）。",
        f"- 规则：{COMPARE_RULE_ZH}", ""]


def render_markdown(summary: dict) -> str:
    out: list[str] = []
    partial = summary["mode"] == "partial"
    out += [f"# 正式批次结果汇总{'（进行中，仅已写出 report.json 的 run）' if partial else ''}", "",
            f"- 批次：`{summary['batch_dir']}`；{summary['bank_version']}；标签 {summary['label']}；"
            f"模式：{'进行中（--partial），不是最终结果' if partial else '批次已完成'}。",
            f"- 单位：{summary['unit_note']}",
            f"- 来源 SHA-256：plan.json `{summary['sources']['plan_sha256']}`；"
            f"batch_report.json `{summary['sources']['batch_report_sha256'] or '尚未生成'}`；"
            f"tasks/bank.json `{summary['sources']['bank_sha256']}`"
            + {True: "（与封存值一致）", False: "（与封存值不一致）", None: ""}[summary["sources"]["bank_sealed_match"]] + "。",
            f"- 判定阈值：单题 token 上限 {summary['thresholds']['task_token_limit']:,.0f}；"
            f"表格题小参考差阈值 {summary['thresholds']['small_reference_gap']}（只用于标记与诊断）。",
            "- 配置：" + "；".join(f"`{key}`={label}" for key, label in CONFIG_LABELS.items()) + "。"]
    if summary["label"] != "FORMAL":
        out.append(f"- **注意：标签为 {summary['label']}，不代表正式效果。**")
    out += [f"- 说明：{note}" for note in summary["notes"]]
    out.append("")
    progress = summary["progress"]
    if partial:
        out += ["## 0. 进度", ""]
        rows = []
        for phase in TIERS:
            expected = progress["expected_runs"].get(phase)
            text = f"{expected[0]}–{expected[1]}" if isinstance(expected, list) else ("—" if expected is None else expected)
            rows.append([phase, progress["completed_runs"][phase], text])
        out += [_table(["阶段", "已写出 report.json 的 run", "计划 run 数"], rows, right=(1, 2)), ""]
        if progress["running"]:
            out += [_table(["阶段", "配置", "遍次", "run_id", "状态", "已完成任务", "已记账 token", "当前任务"],
                           [[item["phase"], f"`{item['config']}`", item["rep"], item["run_id"], item["status"],
                             f"{item['tasks_done']} / {item['tasks_total']}", _int(item["tokens_so_far"]),
                             item["current_task"] or "—"] for item in progress["running"]],
                           right=(2, 5, 6)), ""]
        if progress["reserved_without_manifest"]:
            out += [f"另有 {len(progress['reserved_without_manifest'])} 个 run 目录已预留、尚无 manifest（排队等并发槽位）。", ""]
        if not summary["runs"]:
            out += ["**暂无可汇总 run**：plan.json 之后创建的 run 里，还没有任何一个写出 report.json（run 全部任务跑完才会写）。", ""]
    out += ["## 1. 逐遍汇总", ""]
    for phase in TIERS:
        runs = [run for run in summary["runs"] if run["phase"] == phase]
        if not runs:
            continue
        out += [f"### {phase}（{len(runs)} 个 run，每个 {runs[0]['task_count']} 题）", "",
                _table(["配置", "遍次", "run_id", "任务 token 合计", "input", "cached_input", "output", "reasoning",
                        "平均归一化分数", "单题最高 token", f"超 {summary['thresholds']['task_token_limit']:,.0f} 的题数"],
                       [[f"`{run['config']}`", run["rep"], run["run_id"], _int(run["tokens_total"]),
                         *(_int(run["buckets"][key]) for key in BUCKETS), _score(run["mean_score"]),
                         f"{_int(run['max_task_tokens'])}（{run['max_task_id']}）", run["over_limit_tasks"]]
                        for run in runs], right=(1, 3, 4, 5, 6, 7, 8, 9, 10)), "",
                _table(["配置", "遍次", "模型调用", "失败调用", "停止原因", "墙钟合计（秒）", "其中 llm / run / queue", "run 总耗时（秒，manifest 起止）"],
                       [[f"`{run['config']}`", run["rep"], _int(run["model_calls"]), _int(run["failed_calls"]),
                         _stops(run["stop_reasons"]), _avg(run["seconds"]["wall"]),
                         _triple([_avg(run["seconds"][key]) for key in ("llm", "run", "queue")]),
                         _avg(run["run_elapsed_seconds"])] for run in runs], right=(1, 2, 3, 5, 7)), "",
                _table(["配置", "遍数", "三遍任务 token 合计", "每遍平均 token", "各遍平均归一化分数", "三遍平均分数"],
                       [[f"`{config}`", len(item["reps"]), _int(item["tokens_sum"]), _avg(item["tokens_mean"]),
                         _triple([_score(value) for value in item["scores"]]), _score(item["score_mean"])]
                        for config, item in summary["config_summary"][phase].items()], right=(1, 2, 3, 5)), ""]
    validation = summary["validation"]
    out += ["## 2. validation 准入", ""]
    if validation["admission"] is None:
        out += ["尚无准入记录：" + ("批次没有 validation 准入记录。" if not partial else
                "批次尚未生成 batch_report.json；不做正式准入判定。"), ""]
    else:
        admission = validation["admission"]
        out += [f"来源：{validation['admission_source']}。" + ("" if validation["admission_recomputed_matches"] is None else
                f"按已保存 validation 报告重算{'一致' if validation['admission_recomputed_matches'] else '不一致'}。"), "",
                f"- 判定：**{admission.get('claim')}**；三遍分数均不低于 default_lib：{_yes(admission.get('score_not_lower_each_rep'))}；"
                f"平均 token 更低：{_yes(admission.get('cost_reduced_on_average'))}。",
                f"- 规则：{ADMISSION_RULE_ZH}（英文原文见下方 admission 原文的 rule 字段）", ""]
        if validation["pairs"]:
            out += [_table(["遍次", "default_lib 平均分", "tuned_lib 平均分", "分差", "default_lib token", "tuned_lib token", "token 差"],
                           [[pair["rep"], _score(pair["default_lib_score"]), _score(pair["tuned_lib_score"]),
                             f"{pair['score_difference']:+.4f}", _int(pair["default_lib_tokens"]),
                             _int(pair["tuned_lib_tokens"]), f"{pair['token_difference']:+,.0f}"]
                            for pair in validation["pairs"]], right=(0, 1, 2, 3, 4, 5, 6)), ""]
        out += ["admission 原文：", "", "```json", json.dumps(admission, ensure_ascii=False, indent=2), "```", ""]
    out += ["## 3. 逐题结果（validation、exam）", "",
            "停止原因代码：" + "，".join(f"{key}={label}" for key, label in STOP_LABELS.items()) + "。"
            "原始指标取自各 run 已保存的 holdout_score.json 的 raw，没有则写“未保存”。", ""]
    shown = False
    for phase in ("validation", "exam"):
        rows = summary["per_task"].get(phase)
        if not rows:
            continue
        shown = True
        out += [f"### {phase}", "",
                _table(["任务", "配置", "归一化分（遍1 / 遍2 / 遍3）", "三遍均分", "原始指标（遍1 / 遍2 / 遍3）",
                        "token（遍1 / 遍2 / 遍3）", "三遍均 token", "停止（遍1 / 遍2 / 遍3）", "备注"],
                       [[row["task_id"], f"`{row['config']}`",
                         _triple([_score(rep["score"]) for rep in row["reps"]]), _score(row["score_mean"]),
                         _triple([_raw(rep["raw"]) for rep in row["reps"]]),
                         _triple([_int(rep["tokens"]) for rep in row["reps"]]), _avg(row["tokens_mean"]),
                         _triple(["/".join(rep["stop_reasons"]) if rep["stop_reasons"] else "—" for rep in row["reps"]]),
                         ("、".join(filter(None, [
                             f"小参考差 {row['reference_gap']:.6f}（仅标记）" if row["small_reference_gap"] else "",
                             f"失败调用 {sum(rep['failed_calls'] or 0 for rep in row['reps'])}"
                             if sum(rep["failed_calls"] or 0 for rep in row["reps"]) else ""])) or "—")]
                        for row in rows], right=(3, 6)), ""]
    if not shown:
        out += ["暂无 validation / exam 的逐题数据。", ""]
    out += ["## 4. exam 三遍规则对比", ""]
    comparisons = summary["exam"]["comparisons"]
    titles = {"default_vs_naive": "default vs naive", "default_lib_vs_default": "default_lib vs default",
              "tuned_lib_vs_default_lib": "tuned_lib vs default_lib"}
    if not any(item.get("available") for item in comparisons.values()) and partial:
        out += ["考试尚无完整的三遍配对 run，暂不比较。", ""]
    else:
        for key, title in titles.items():
            out += _comparison_md(title, comparisons[key])
    out += ["## 5. 表格题小参考差与诊断视图", "",
            f"参考差 = tasks/bank.json 里 ceiling_raw − baseline_raw。小于 {summary['thresholds']['small_reference_gap']} 的表格题只做标记，"
            f"**以下剔除这些题的视图{DIAGNOSTIC_NOTE}**；正式成绩仍按全部题目计算。", ""]
    if summary["reference_gaps"]:
        out += [_table(["任务", "阶段", "baseline_raw", "ceiling_raw", "参考差", "标记"],
                       [[item["task_id"], item["tier"], f"{item['baseline_raw']:.10g}", f"{item['ceiling_raw']:.10g}",
                         f"{item['gap']:.10g}", "小参考差" if item["small_gap"] else "—"]
                        for item in summary["reference_gaps"]], right=(2, 3, 4)), ""]
    diagnostic = summary["diagnostic_excluding_small_gap"]
    if not diagnostic["phases"]:
        out += ["所有已汇总阶段都没有小参考差表格题，诊断视图与正式视图相同。", ""]
    for phase, entry in diagnostic["phases"].items():
        out += [f"### {phase}：剔除 {'、'.join(entry['excluded_in_phase'])}（{DIAGNOSTIC_NOTE}）", ""]
        if not entry["remaining_tasks"]:
            out += ["剔除后没有剩余任务。", ""]
            continue
        out += [_table(["配置", "遍数", "各遍剔除后平均归一化分数", "三遍平均分数", "每遍平均 token（剔除后）"],
                       [[f"`{config}`", len(item["reps"]), _triple([_score(value) for value in item["scores"]]),
                         _score(item["score_mean"]), _avg(item["tokens_mean"])]
                        for config, item in entry["config_summary"].items()], right=(1, 3, 4)), ""]
        replay = entry.get("admission_replay")
        if replay:
            out += [f"- 剔除后按同一规则重放准入（{DIAGNOSTIC_NOTE}）：{replay['claim']}；"
                    f"三遍分数均不低于 default_lib：{_yes(replay['score_not_lower_each_rep'])}；平均 token 更低：{_yes(replay['cost_reduced_on_average'])}。", ""]
        for key, comp in (entry.get("comparisons") or {}).items():
            if comp.get("available"):
                out += [f"- 剔除后 {titles[key]}（{DIAGNOSTIC_NOTE}）：{CLAIM.get(comp['claim'], comp['claim'])}；"
                        f"平均分差 {comp['mean_score_gap']:+.4f}；平均 token {_avg(comp['baseline_mean_cost'])} → {_avg(comp['candidate_mean_cost'])}。"]
        out.append("")
    breakeven = summary["breakeven"]
    out += ["## 6. 研究版回本点", ""]
    if breakeven["computed"]:
        out += [f"研究总成本 {_int(breakeven['research_cost_tokens'])} token；考试 {breakeven['exam_tasks']} 题口径下，"
                f"研究版相对 default_lib 每题平均少用 {_avg(breakeven['savings_per_task_tokens'])} token；"
                f"**数学回本点 {breakeven['breakeven_tasks']} 题**（内部 token 口径，不是现金节省）。", "",
                f"规则：{breakeven['rule']}", ""]
    else:
        out += ["**不计算。**原因：", "", *[f"- {reason}" for reason in breakeven["reasons"]],
                "", "回本点只在“准入 admitted，且考试上研究版分数不降、平均 token 更低”时计算。", ""]
    out += ["## 7. 数据提醒", ""]
    out += [f"- {item}" for item in summary["warnings"]] or ["无。"]
    return "\n".join(out).rstrip() + "\n"


# ------------------------------------------------------------------------ CLI

def write_outputs(summary: dict, output_dir: Path, name: str, *, allow_inside_project: bool = False) -> tuple[Path, Path]:
    batch_dir = Path(summary["batch_dir"])
    project = batch_dir.parent.parent
    target = Path(output_dir).expanduser().resolve()
    if (target == project or target.is_relative_to(project)) and not allow_inside_project:
        raise SummaryError(f"输出目录在批次所在工作区内，拒绝写入（只读保护）：{target}")
    target.mkdir(parents=True, exist_ok=True)
    json_path, md_path = target / f"{name}.json", target / f"{name}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _cost_argument(text: str) -> float:
    value = float(text.replace(",", "").replace("_", ""))
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("研究总成本必须是非负有限数")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True, help="runs/batch-... 批次目录（只读）")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录；不能在批次所在工作区内")
    parser.add_argument("--name", default="formal-batch-summary", help="输出文件名主干（生成 .json 与 .md）")
    parser.add_argument("--partial", action="store_true",
                        help="批次没有 batch_report.json 时，只汇总已写出 report.json 的 run")
    parser.add_argument("--research-cost", type=_cost_argument,
                        help="研究总成本（内部 token，可带千分位逗号），用于回本点；本轮为 2094728")
    parser.add_argument("--task-token-limit", type=float, default=TASK_TOKEN_LIMIT)
    parser.add_argument("--small-gap", type=float, default=SMALL_REFERENCE_GAP)
    parser.add_argument("--allow-output-inside-project", action="store_true",
                        help="允许输出目录位于批次所在工作区内（默认拒绝）")
    args = parser.parse_args(argv)
    try:
        summary = summarize(args.batch_dir, partial=args.partial, research_cost=args.research_cost,
                            task_token_limit=args.task_token_limit, small_gap=args.small_gap)
        json_path, md_path = write_outputs(summary, args.output_dir, args.name,
                                           allow_inside_project=args.allow_output_inside_project)
    except (SummaryError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    counts = summary["progress"]["completed_runs"]
    print(f"模式：{summary['mode']}；已汇总 run：" + "、".join(f"{phase} {counts[phase]}" for phase in TIERS)
          + ("（暂无可汇总 run）" if not summary["runs"] else "")
          + f"；数据提醒 {len(summary['warnings'])} 条")
    print(f"JSON：{json_path}")
    print(f"Markdown：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
