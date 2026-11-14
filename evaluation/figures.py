"""Audited four-configuration study figures and break-even calculation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean

from ledger import Ledger

from .comparable import admit_complete, comparable_reports
from .compare import compare_configurations
from .proposal import ProposalLedger
from .report import run_report

CONFIGS = ("naive", "default", "default_lib", "tuned_lib")
LABELS = ("朴素版", "默认设置／空库", "默认设置／练过的库", "研究版／练过的库")
COLORS = ("#71819d", "#3769d3", "#169678", "#aa5cc7")


def _verify_formal_attempt_markers(project_root: Path, batch_dir: Path,
                                   bank_version: str) -> str:
    """Require both global one-shot markers to name this exact batch and code."""
    if (not isinstance(bank_version, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", bank_version) is None):
        raise ValueError("Invalid formal bank version")
    commits = set()
    for stage in ("validation", "exam"):
        path = project_root / "runs" / f"{stage}-{bank_version}.attempt.json"
        marker = json.loads(path.read_text(encoding="utf-8"))
        expected_schema = f"{stage}_attempt/1"
        commit = marker.get("code_commit")
        if (marker.get("schema") != expected_schema
                or marker.get("bank_version") != bank_version
                or Path(marker.get("batch_dir", "")).resolve() != batch_dir
                or not isinstance(commit, str)
                or re.fullmatch(r"[0-9a-f]{40}", commit) is None):
            raise ValueError(f"Formal {stage} one-shot marker differs from batch")
        commits.add(commit)
    if len(commits) != 1:
        raise ValueError("Formal validation and exam used different code commits")
    return commits.pop()


def _finite(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite")
    return value


def build_study(reports, *, research_reports=None, research_total_cost=None,
                research_run_ids=None, admission=None):
    """Require three paired runs per eligible config; keep TEST separate.

    A task whose holdout failed is dropped from every run (pairing needs one task
    list) and listed in study["excluded_tasks"]; it is never scored as zero.
    """
    reports, excluded_tasks = comparable_reports(reports)
    grouped = defaultdict(dict)
    phases, currencies, ids = set(), set(), set()
    exact_token_flags = []
    tasks = None
    for report in reports:
        if report.get("schema") != "run_report/1":
            raise ValueError("Study requires run_report/1")
        config, rep, run_id = report.get("config"), report.get("rep"), report.get("run_id")
        if config not in CONFIGS or type(rep) is not int or rep not in {1, 2, 3}:
            raise ValueError("Unexpected configuration or repetition")
        if not isinstance(run_id, str) or run_id in ids or rep in grouped[config]:
            raise ValueError("Duplicate run ID or repetition")
        ids.add(run_id)
        grouped[config][rep] = report
        phases.add(report.get("phase"))
        currencies.add(report.get("currency"))
        exact_token_flags.append(report.get("exact_token_accounting") is True)
        order = tuple(task["task_id"] for task in report["tasks"])
        if not order or len(set(order)) != len(order) or (tasks is not None and order != tasks):
            raise ValueError("Paired runs need identical ordered tasks")
        tasks = order
        _finite(report.get("total_cost"), "run cost", True)
        for task in report["tasks"]:
            _finite(task.get("holdout_score"), "holdout score")
            _finite(task.get("total_cost"), "task cost", True)
            if type(task.get("evidence", {}).get("task_end_seq")) is not int:
                raise ValueError("Missing task_end ledger citation")
            for key in ("llm", "run", "queue", "wall"):
                _finite(task.get("seconds", {}).get(key), "seconds." + key, True)
    active = tuple(config for config in CONFIGS if config in grouped)
    rejected = (isinstance(admission, dict) and admission.get("claim") == "rejected"
                and admission.get("baseline") == "default_lib"
                and admission.get("candidate") == "tuned_lib")
    expected = CONFIGS[:3] if rejected else CONFIGS
    if active != expected or any(set(grouped[c]) != {1, 2, 3} for c in active):
        raise ValueError("Study needs three repetitions of each eligible configuration")
    if len(phases) != 1 or len(currencies) != 1:
        raise ValueError("Do not mix phases or currencies")
    phase, currency = phases.pop(), currencies.pop()
    test_only = phase == "dev" and currency == "TEST"
    if not (phase == "exam" and currency != "TEST") and not test_only:
        raise ValueError("Only exam or explicitly TEST dev runs can be plotted")
    if currency == "TOKENS" and not all(exact_token_flags):
        raise ValueError("TOKENS study needs audited unit_tokens=1 and all four model rates=1")
    if phase == "exam" and not (rejected or (isinstance(admission, dict)
            and admission.get("claim") == "admitted"
            and admission.get("baseline") == "default_lib"
            and admission.get("candidate") == "tuned_lib")):
        raise ValueError("Formal study requires recorded tuning admission")
    stats, rows = {}, []
    for config in active:
        runs = [grouped[config][i] for i in (1, 2, 3)]
        scores = [mean(task["holdout_score"] for task in run["tasks"]) for run in runs]
        costs = [run["total_cost"] for run in runs]
        stats[config] = {"scores": scores, "score_mean": mean(scores), "score_min": min(scores),
                         "score_max": max(scores), "cost_mean": mean(costs),
                         "run_ids": [run["run_id"] for run in runs]}
        for run in runs:
            for task in run["tasks"]:
                rows.append({"config": config, "rep": run["rep"], "run_id": run["run_id"],
                             "task_id": task["task_id"], "score": task["holdout_score"],
                             "cost": task["total_cost"], "seconds": task["seconds"],
                             "quote_error": task.get("quote_error"),
                             "task_end_seq": task["evidence"]["task_end_seq"]})
    comparisons = {}
    pairs = [("naive", "default", "scheduler"), ("default", "default_lib", "library")]
    if not rejected:
        pairs.append(("default_lib", "tuned_lib", "self_research"))
    for baseline, candidate, key in pairs:
        pair = [grouped[c][i] for c in (baseline, candidate) for i in (1, 2, 3)]
        comparisons[key] = compare_configurations(pair, baseline=baseline, candidate=candidate)
    if research_reports and research_total_cost is not None:
        raise ValueError("Research cost may be supplied in only one audited form")
    research_cost, research_ids = None, []
    if research_reports:
        research_cost = 0.0
        research_phase = "dev" if test_only else "self_research"
        for report in research_reports:
            if report.get("schema") != "run_report/1" or report.get("phase") != research_phase:
                raise ValueError("Research cost needs matching self-research run reports")
            if report.get("currency") != currency or report.get("run_id") in ids | set(research_ids):
                raise ValueError("Research run currency or ID mismatch")
            if currency == "TOKENS" and report.get("exact_token_accounting") is not True:
                raise ValueError("TOKENS research report lacks audited exact token accounting")
            research_ids.append(report["run_id"])
            research_cost += _finite(report.get("total_cost"), "research cost", True)
    if research_total_cost is not None:
        research_cost = _finite(research_total_cost, "total research cost", True)
        research_ids = list(research_run_ids or [])
    savings = ((stats["default_lib"]["cost_mean"] - stats["tuned_lib"]["cost_mean"]) / len(tasks)
               if not rejected else None)
    breakeven = math.ceil(research_cost / savings) if research_cost is not None and savings is not None and savings > 0 else None
    return {"schema": "study/1", "phase": phase, "currency": currency, "test_only": test_only,
            "exact_token_accounting": currency == "TOKENS" and all(exact_token_flags),
            "tasks": list(tasks), "configurations": stats, "comparisons": comparisons,
            "excluded_tasks": excluded_tasks, "admission": admission,
            "task_rows": rows, "breakeven": {"research_cost": research_cost,
            "research_run_ids": research_ids, "savings_per_task": savings, "tasks": breakeven,
            "eligible": (not rejected and all(delta >= 0 for delta in
            comparisons["self_research"]["paired_differences"]) and savings > 0)}}


def _auto_research_cost(study_dir: Path, project_root: Path, currency: str, test_only: bool,
                        *, expected_tuned_config: dict | None = None):
    """Reconcile all practice reports and the proposal ledger to the auto study."""
    study_dir = study_dir.expanduser().resolve()
    if not study_dir.is_relative_to(project_root):
        raise ValueError("Auto study path escapes project root")
    report = json.loads((study_dir / "auto_self_research_report.json").read_text(encoding="utf-8"))
    plan = json.loads((study_dir / "auto_plan.json").read_text(encoding="utf-8"))
    label = "TEST ONLY" if test_only else "FORMAL"
    if report.get("schema") != "auto_self_research/1" or plan.get("schema") != "auto_self_research_plan/1":
        raise ValueError("Expected an automatic self-research study")
    if report.get("label") != label or plan.get("label") != label or report.get("currency") != currency:
        raise ValueError("Research study price status does not match exam")
    if expected_tuned_config is not None:
        from .self_research import verify_auto_selected

        selected_path = verify_auto_selected(report).resolve()
        selected = report.get("selection", {})
        if (not selected_path.is_relative_to(project_root)
                or expected_tuned_config.get("path") != str(selected_path.relative_to(project_root))
                or expected_tuned_config.get("sha256") != selected.get("config_sha256")):
            raise ValueError("Exam tuned configuration differs from selected self-research candidate")
    ledger_path = study_dir / "proposal_ledger.jsonl"
    if Path(report.get("proposal_ledger_path", "")).resolve() != ledger_path:
        raise ValueError("Proposal ledger path differs from the study")
    if hashlib.sha256(ledger_path.read_bytes()).hexdigest() != report.get("proposal_ledger_sha256"):
        raise ValueError("Proposal ledger changed after the study")
    run_ids = []
    task_cost = 0.0
    prices = None
    model = None
    for attempt in report.get("reports", []):
        run_id = attempt.get("run_id")
        if not isinstance(run_id, str) or run_id in run_ids:
            raise ValueError("Duplicate self-research run ID")
        run_ids.append(run_id)
        run_dir = (project_root / "runs" / run_id).resolve()
        if run_dir.parent != project_root / "runs":
            raise ValueError("Self-research run path escapes runs/")
        audited = run_report(run_dir)
        if audited["run_id"] != run_id or audited["phase"] != ("dev" if test_only else "self_research"):
            raise ValueError("Self-research run identity or phase changed")
        if currency == "TOKENS" and audited["exact_token_accounting"] is not True:
            raise ValueError("TOKENS self-research run lacks exact token accounting")
        task_cost += audited["total_cost"]
        run_ledger = Ledger.open(run_dir)
        if prices is None:
            prices = run_ledger.prices
            model = run_ledger.manifest["model"]
            if run_ledger.manifest["prices"]["sha256"] != plan["price_sha256"]:
                raise ValueError("Proposal and task prices differ")
        elif run_ledger.manifest["model"] != model or run_ledger.manifest["prices"]["sha256"] != plan["price_sha256"]:
            raise ValueError("Self-research runs used inconsistent prices or models")
    if not run_ids or prices is None:
        raise ValueError("Self-research study has no audited practice runs")
    nested_ids = [row.get("run_id") for study in report.get("evaluations", {}).values()
                  for row in study.get("attempts", [])]
    if sorted(run_ids) != sorted(nested_ids):
        raise ValueError("Self-research run list differs from candidate attempts")
    proposal = ProposalLedger(ledger_path, prices=prices,
                              price_sha256=plan["price_sha256"],
                              source_prompt_sha256=plan["source_prompt_sha256"], model=model)
    proposal_cost = proposal.spent()
    expected = task_cost + proposal_cost
    stated_task = _finite(report.get("total_task_cost"), "stated task cost", True)
    stated_proposal = _finite(report.get("total_proposal_cost"), "stated proposal cost", True)
    stated_total = _finite(report.get("total_research_cost"), "stated research cost", True)
    if (not math.isclose(task_cost, stated_task, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(proposal_cost, stated_proposal, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(expected, stated_total, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Automatic research cost does not reconcile")
    return expected, run_ids


def build_study_from_batch(batch_dir: Path, *, research_reports=None,
                           auto_study_dir: Path | None = None):
    """Re-audit all runs and verify their membership in one sealed batch."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    project_root = batch_dir.parent.parent
    plan = json.loads((batch_dir / "plan.json").read_text(encoding="utf-8"))
    batch = json.loads((batch_dir / "batch_report.json").read_text(encoding="utf-8"))
    if plan.get("schema") != "batch_plan/1" or batch.get("schema") != "batch_report/1":
        raise ValueError("Expected batch_plan/1 and batch_report/1")
    if batch.get("status", "complete") != "complete":  # reports written before status existed are complete
        raise ValueError("Batch is partial (failed or skipped runs); no study is plotted from it")
    if plan.get("label") != batch.get("label") or not plan.get("run_exam"):
        raise ValueError("Batch lacks a completed exam plan")
    if plan.get("bank_version") != batch.get("bank_version"):
        raise ValueError("Batch plan and report bank versions differ")
    formal_commit = (_verify_formal_attempt_markers(project_root, batch_dir,
                     batch["bank_version"]) if batch["label"] == "FORMAL" else None)
    admission = batch.get("admission")
    if (not isinstance(admission, dict) or admission.get("claim") not in {"admitted", "rejected"}
            or bool(batch.get("tuned_exam_skipped")) != (admission["claim"] == "rejected")):
        raise ValueError("Batch has an inconsistent tuning admission or exam exclusion")
    for filename, digest in plan.get("sealed_inputs", {}).items():
        path = Path(filename).resolve()
        if not path.is_relative_to(project_root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Batch source file changed after sealing")
    validation_entries = [row for row in batch.get("runs", [])
                          if row.get("logical_phase") == "validation"]
    if (len(validation_entries) != 6
            or {(row.get("config"), row.get("rep")) for row in validation_entries}
            != {(config, rep) for config in ("default_lib", "tuned_lib") for rep in (1, 2, 3)}):
        raise ValueError("Completed exam needs six distinct paired validation runs")
    validation_reports = []
    tuned_configs = []
    for row in validation_entries:
        run_dir = (project_root / "runs" / row["run_id"]).resolve()
        if run_dir.parent != project_root / "runs" or Path(row["report_path"]).resolve() != run_dir / "report.json":
            raise ValueError("Validation run path escapes the run directory")
        report = run_report(run_dir)
        if report["run_id"] != row["run_id"] or report["config"] != row["config"] or report["rep"] != row["rep"]:
            raise ValueError("Validation run identity differs from audited report")
        if formal_commit is not None:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("phase") != "validation"
                    or manifest.get("bank_version") != batch["bank_version"]
                    or manifest.get("code_commit") != formal_commit):
                raise ValueError("Validation run differs from its one-shot marker")
        if auto_study_dir is not None and row["config"] == "tuned_lib":
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            tuned_configs.append(manifest.get("agent_config"))
        validation_reports.append(report)
    if admit_complete(validation_reports, baseline="default_lib", candidate="tuned_lib") != admission:
        raise ValueError("Recorded tuning admission differs from audited validation runs")
    entries = [row for row in batch.get("runs", []) if row.get("logical_phase") == "exam"]
    expected_runs = 12 if admission["claim"] == "admitted" else 9
    if len(entries) != expected_runs or len({row.get("run_id") for row in entries}) != expected_runs:
        raise ValueError("Completed exam has the wrong number of distinct runs")
    reports = []
    for row in entries:
        run_dir = (project_root / "runs" / row["run_id"]).resolve()
        if run_dir.parent != project_root / "runs" or Path(row["report_path"]).resolve() != run_dir / "report.json":
            raise ValueError("Batch run path escapes the run directory")
        report = run_report(run_dir)
        if report["run_id"] != row["run_id"] or report["config"] != row["config"] or report["rep"] != row["rep"]:
            raise ValueError("Batch run identity differs from audited report")
        if formal_commit is not None:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("phase") != "exam"
                    or manifest.get("bank_version") != batch["bank_version"]
                    or manifest.get("code_commit") != formal_commit):
                raise ValueError("Exam run differs from its one-shot marker")
        if auto_study_dir is not None and row["config"] == "tuned_lib":
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            tuned_configs.append(manifest.get("agent_config"))
        reports.append(report)
    if auto_study_dir is not None and research_reports:
        raise ValueError("Use either an automatic study or explicit research runs")
    auto_cost, auto_runs = (None, [])
    if auto_study_dir is not None:
        expected_tuned = 6 if admission["claim"] == "admitted" else 3
        if len(tuned_configs) != expected_tuned or any(config != tuned_configs[0] for config in tuned_configs):
            raise ValueError("Validation and exam tuned runs used inconsistent configuration files")
        currency = reports[0]["currency"]
        auto_cost, auto_runs = _auto_research_cost(
            Path(auto_study_dir), project_root, currency, batch["label"] == "TEST ONLY",
            expected_tuned_config=tuned_configs[0])
    study = build_study(reports, research_reports=research_reports,
                        research_total_cost=auto_cost, research_run_ids=auto_runs,
                        admission=admission)
    if (study["test_only"] != (batch["label"] == "TEST ONLY")
            or {row["run_id"] for row in entries} !=
            {run_id for value in study["configurations"].values() for run_id in value["run_ids"]}):
        raise ValueError("Batch provenance and study membership disagree")
    study["batch_dir"] = str(batch_dir)
    return study


def render_html(study):
    """Return portable HTML with an SVG range plot and ledger citations."""
    stats = study["configurations"]
    active = [config for config in CONFIGS if config in stats]
    values = [v for c in active for v in stats[c]["scores"]]
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.1)
    lo, hi = lo - span * .15, hi + span * .15
    x = lambda v: 265 + (v - lo) / (hi - lo) * 650
    marks = []
    for i, config in enumerate(active):
        index = CONFIGS.index(config)
        y, s, color = 75 + i * 75, stats[config], COLORS[i]
        marks.append(f'<text x="12" y="{y+5}">{escape(LABELS[index])}</text>')
        marks.append(f'<line x1="{x(s["score_min"]):.1f}" x2="{x(s["score_max"]):.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        marks.extend(f'<circle cx="{x(v):.1f}" cy="{y}" r="5" fill="white" stroke="{color}" stroke-width="2"/>' for v in s["scores"])
        marks.append(f'<circle cx="{x(s["score_mean"]):.1f}" cy="{y}" r="8" fill="{color}"/>')
        marks.append(f'<text x="930" y="{y+5}">{s["score_mean"]:.3f} · {s["cost_mean"]:.4f} {escape(study["currency"])}</text>')
    svg = f'<svg viewBox="0 0 1160 330" role="img" aria-label="{len(active)}配置三遍分数与范围">' + ''.join(marks) + '</svg>'
    is_tokens = study["currency"] == "TOKENS" and study.get("exact_token_accounting") is True
    if study["currency"] == "TOKENS" and not is_tokens:
        raise ValueError("TOKENS figure needs audited exact token accounting")
    notice = ('TEST 联调数据，不代表正式效果或真实花费。' if study["test_only"] else
              '自选 token 预算的内部冻结题库研究；非现金节省、非官方比赛成绩。数字可追溯到运行账本。' if is_tokens else
              '正式考试数据；数字可追溯到运行账本。')
    cost_label = 'token 消耗' if is_tokens else '花费'
    amount_label = 'token 数' if is_tokens else '金额'
    quote_error_label = '平均绝对 token 数误差' if is_tokens else '平均绝对金额误差'
    amount_unit_note = 'token 数以 TOKENS 计。' if is_tokens else '金额以本报告币种计。'
    research_label = '研究 token 消耗' if is_tokens else '研究成本'
    if study["admission"] and study["admission"]["claim"] == "rejected":
        notice += ' 研究版未通过验证准入，考试只运行其余三条线。'
    if study.get("excluded_tasks"):
        notice += (' 已从所有运行中剔除 holdout 评分失败的题，未按零分计：'
                   + '、'.join(item["task_id"] for item in study["excluded_tasks"]) + '。')
    roi = study["breakeven"]
    if study["admission"] and study["admission"]["claim"] == "rejected":
        spent = (f'已审计{research_label} {roi["research_cost"]:.4f} {escape(study["currency"])}；'
                 if roi["research_cost"] is not None else f'{research_label}尚无审计报告；')
        roi_text = spent + '研究版未通过验证准入，未进入考试；不计算研究版回本点。'
    elif roi["research_cost"] is None:
        roi_text = (f'{research_label}尚无审计报告，回本点未知。' if is_tokens else
                    '研究自己的成本尚无审计报告，回本点未知。')
    elif roi["tasks"] is None:
        roi_text = '当前未观察到每题节省，无法计算正的回本点。'
    else:
        roi_text = (f'{research_label} {roi["research_cost"]:.4f} {escape(study["currency"])}；'
                    f'每题节省 {roi["savings_per_task"]:.4f}'
                    f'{" " + escape(study["currency"]) if is_tokens else ""}；数学回本点 {roi["tasks"]} 题。')
        if not roi["eligible"]:
            roi_text += ' 研究版未满足三遍分数不降的准入规则，不应宣称实际回本。'
    score_min = min(0.0, min(row["score"] for row in study["task_rows"]))
    score_max = max(1.0, max(row["score"] for row in study["task_rows"]))
    progress = []
    for config in active:
        index = CONFIGS.index(config)
        points = []
        for j, task_id in enumerate(study["tasks"]):
            scores = [row["score"] for row in study["task_rows"]
                      if row["config"] == config and row["task_id"] == task_id]
            points.append(f'{70 + j * 100},{210 - (mean(scores) - score_min) / (score_max - score_min) * 170:.1f}')
        progress.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index]}" stroke-width="3"/>')
    chart_width = max(1160, 120 + len(study["tasks"]) * 100)
    timeline = f'<svg viewBox="0 0 {chart_width} 240" role="img" aria-label="逐题平均分数走势">' + ''.join(progress) + '</svg>'
    quote_points = []
    for j, task_id in enumerate(study["tasks"]):
        errors = [abs(row["quote_error"]["cost"]) for row in study["task_rows"]
                  if row["task_id"] == task_id and isinstance(row["quote_error"], dict)
                  and isinstance(row["quote_error"].get("cost"), (int, float))
                  and math.isfinite(row["quote_error"]["cost"])]
        if errors:
            quote_points.append((j, mean(errors)))
    max_error = max((value for _, value in quote_points), default=0)
    quote_svg = ''
    if quote_points:
        points = ' '.join(f'{70 + j * 100},{210 - (value / max_error * 170 if max_error else 0):.1f}' for j, value in quote_points)
        quote_svg = f'<svg viewBox="0 0 {chart_width} 240" role="img" aria-label="逐题平均绝对报价误差"><polyline points="' + points + '" fill="none" stroke="#bb7a17" stroke-width="3"/></svg>'
    comparisons = '<ul>' + ''.join(f'<li>{escape(k)}：{escape(v["claim"])}；平均分差 {v["mean_score_gap"]:.4f}</li>' for k, v in study["comparisons"].items()) + '</ul>'
    body = []
    for row in study["task_rows"]:
        s = row["seconds"]
        citation = f'{row["run_id"]} / {row["task_id"]} / ledger#{row["task_end_seq"]}'
        cells = (LABELS[CONFIGS.index(row["config"])], row["rep"], row["task_id"],
                 f'{row["score"]:.4f}', f'{row["cost"]:.4f}',
                 f'{s["llm"]:.1f}/{s["run"]:.1f}/{s["queue"]:.1f}/{s["wall"]:.1f}', citation)
        body.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in cells) + '</tr>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Research Agent 配置对照报告</title>'
            '<style>body{font:15px system-ui;margin:40px auto;max-width:1200px;color:#263247}svg{width:100%}'
            'table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #dce3ef;text-align:left}'
            '.notice{background:#fff4d7;padding:12px;border-radius:6px}section{margin-top:28px}</style>'
            f'<h1>{len(active)} 配置对照 · {escape(study["phase"])}</h1><p class="notice">{notice}</p>'
            f'<section><h2>分数与{cost_label}</h2>{svg}<p>每行三个空心点对应三遍；线段为范围；实心点为均值。{amount_label}为每遍全部任务的均值。</p></section>'
            f'<section><h2>逐题分数走势</h2>{timeline}<p>按任务顺序绘制四种配置的三遍平均分数；纵轴范围 {score_min:.3f}–{score_max:.3f}。</p></section>'
            f'<section><h2>报价误差</h2>{quote_svg or "<p>没有可核对的报价误差。</p>"}<p>按任务顺序绘制{quote_error_label}；{amount_unit_note}</p></section>'
            f'<section><h2>判定</h2>{comparisons}<p>{roi_text}</p></section>'
            '<section><h2>逐题证据</h2><p>耗时依次为模型 / 运行 / 排队 / 墙钟，单位秒。</p>'
            f'<table><thead><tr><th>配置</th><th>遍数</th><th>任务</th><th>分数</th><th>{cost_label}</th><th>耗时</th><th>证据</th></tr></thead><tbody>'
            + ''.join(body) + '</tbody></table></section></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build an audited exam study report")
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--research-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--auto-study-dir", type=Path,
                        help="Include all proposal and practice costs from auto self-research")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    study = build_study_from_batch(args.batch_dir,
                                   research_reports=[run_report(path) for path in args.research_run_dir],
                                   auto_study_dir=args.auto_study_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(study), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(study, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
