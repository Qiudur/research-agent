"""Build a source-linked, self-contained HTML roadshow report from committed files.

The report changes only when a non-report commit reaches HEAD. This makes an
hourly scheduled run cheap and prevents report-only commits from feeding back
into the next report.

Optional formal input: --formal-batch-dir with --auto-study-dir audits a
finished formal batch read-only (scripts/audit_formal_outputs.py), writes the
condensed evidence to docs/evidence/formal-batch-roadshow.json and shows real
scores, four-bucket tokens, admission, exam line count, commit and config SHA.
A running batch keeps the pending wording; a failed audit writes nothing.
Later runs without these options read the committed evidence file.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
REPORT = "docs/路演汇报.html"
SOURCE_HEAD = re.compile(r"<!-- source-head: ([0-9a-f]{40}) -->")
EXCLUDE_REPORT = ":(exclude)" + REPORT
FORMAL_EVIDENCE = "docs/evidence/formal-batch-roadshow.json"
FORMAL_SCHEMA = "roadshow_formal/1"
PHASE_NAMES = {"validation": "验证", "exam": "考试"}
CONFIG_NAMES = {"naive": "朴素版", "default": "默认设置／空库",
                "default_lib": "默认设置／练过的库", "tuned_lib": "研究版／练过的库"}
COMPARISON_NAMES = {"default_vs_naive": "默认 vs 朴素", "default_lib_vs_default": "练过的库 vs 空库",
                    "tuned_lib_vs_default_lib": "研究版 vs 练过的库"}
CLAIM_NAMES = {"better": "有明确优势", "no_clear_difference": "无明确差异"}
ADMISSION_NAMES = {"admitted": "已准入", "rejected": "未准入"}


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=check
    )
    return result.stdout.strip()


def committed_text(repo: Path, path: str) -> str:
    return git(repo, "show", "HEAD:" + path, check=False)


def plain(value: str) -> str:
    value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
    return value.replace("**", "").replace(chr(96), "").strip()


def safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def source_commits(repo: Path, revision: str) -> list[tuple[str, str, str]]:
    raw = git(
        repo, "log", "--date=short", "--format=%H%x1f%ad%x1f%s",
        revision, "--", ".", EXCLUDE_REPORT,
    )
    return [tuple(row.split("\x1f", 2)) for row in raw.splitlines() if "\x1f" in row]


def changed_files(repo: Path, revision: str) -> list[str]:
    raw = git(repo, "log", "--name-only", "--format=", revision, "--", ".", EXCLUDE_REPORT)
    return sorted({row for row in raw.splitlines() if row.strip()})


def previous_head(output: Path) -> str | None:
    if not output.exists():
        return None
    match = SOURCE_HEAD.search(output.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def is_ancestor(repo: Path, older: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, "HEAD"],
        cwd=repo, capture_output=True,
    )
    return result.returncode == 0


def section(text: str, name: str) -> str:
    match = re.search(r"(?ms)^## " + re.escape(name) + r"\s*\n(.*?)(?=^## |\Z)", text)
    return match.group(1).strip() if match else ""


def first_paragraph(text: str) -> str:
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    return plain(next((block for block in blocks if not block.startswith("#")), ""))


def number_list(text: str, limit: int) -> list[str]:
    items = re.findall(r"(?m)^\d+\.\s+(.+)$", text)
    return [plain(item) for item in items[:limit]]


def excerpt(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit].rstrip("，； 。") + "…"


def table_rows(text: str) -> list[list[str]]:
    return [[plain(cell) for cell in line.strip().strip("|").split("|")]
            for line in text.splitlines() if line.startswith("|")]


def numeric(text: str) -> float | None:
    match = re.search(r"-?[\d,]+(?:\.\d+)?", text)
    return float(match.group().replace(",", "")) if match else None


def fmt(value: float | None, decimals: int = 0) -> str:
    return "待核对" if value is None else f"{value:,.{decimals}f}"


def category(path: str) -> str:
    if path.startswith("tests/") or "/tests/" in path:
        return "验证与测试"
    if path.startswith(("research_agent/", "library/", "pilot/")):
        return "研究代理与经验库"
    if path.startswith(("tasks/", "harness", "tabular_harness")):
        return "题库与评分"
    if path.startswith(("evaluation/", "dashboard/")) or path == "ledger.py":
        return "评估与看板"
    if path.startswith(("docs/",)) or path == "README.md":
        return "文档与证据"
    return "运行与配置"


def calibration_evidence(repo: Path) -> str:
    """Summarize only a frozen bank backed by both committed dev reports."""
    try:
        bank = json.loads(committed_text(repo, "tasks/bank.json"))
    except (ValueError, TypeError):
        return ""
    tasks = bank.get("tasks")
    if bank.get("frozen") is not True or not isinstance(tasks, list) or not tasks:
        return ""
    documents = [committed_text(repo, path) for path in
                 ("docs/提速朴素成本校准.md", "docs/表格朴素成本校准.md")]
    counts = []
    for document in documents:
        result = re.search(r"(?:总计|合计) *([0-9]+) *次尝试：([0-9]+) *次正常收尾、([0-9]+) *次提前错误", document)
        tokens = re.search(r"(?:实耗|消耗) *[*]{0,2} *([0-9,]+) *token", document)
        if (result is None or tokens is None
                or re.search(r"holdout *尝试(?:为)? *0", document) is None):
            return ""
        counts.append((*map(int, result.groups()),
                       int(tokens.group(1).replace(",", ""))))
    attempts, completed, incomplete, total_tokens = map(sum, zip(*counts))
    if (attempts != len(tasks) * 3 or completed + incomplete != attempts
            or f"{attempts}/{attempts} 次尝试" not in documents[1]
            or "ready=true" not in documents[1]):
        return ""
    audit = re.search(r"runs/naive-cost-audit-[A-Za-z0-9_-]+/", documents[1])
    audit_path = audit.group() + "audit.json" if audit else "私有 runs/ 审计记录"
    return (f"{bank.get('version', '题库')} 已冻结：{len(tasks)} 题、{attempts} 次 dev 尝试，"
            f"{completed} 次正常完成、{incomplete} 次提前停止，四桶合计 {total_tokens:,} token；"
            f"校准 holdout 0。完整审计：{audit_path}。不是现金价格或正式考试成绩。")


class FormalAuditError(ValueError):
    """A finished formal batch exists but its saved outputs fail the read-only audit."""


def load_formal_evidence(batch_dir: Path, auto_study_dir: Path) -> dict | None:
    """Audit a finished formal batch and condense it for the roadshow page.

    Returns None while batch_report.json is absent (the batch is still running).
    Raises FormalAuditError when scripts/audit_formal_outputs.py rejects any
    saved report, ledger, marker or sealed input; callers then render nothing.
    """
    batch_dir = Path(batch_dir).expanduser().resolve()
    auto_study_dir = Path(auto_study_dir).expanduser().resolve()
    if not (batch_dir / "batch_report.json").is_file():
        return None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.audit_formal_outputs import audit_formal_outputs
    from scripts.summarize_formal_batch import BUCKETS, summarize

    try:
        audit = audit_formal_outputs(batch_dir, auto_study_dir)
        summary = summarize(batch_dir, research_cost=float(audit["total_research_cost"]))
        project = batch_dir.parent.parent
        batch_path = batch_dir / "batch_report.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        config_sha: dict[str, str] = {}
        price_sha: set[str] = set()
        for row in batch["runs"]:
            manifest = json.loads((project / "runs" / row["run_id"] / "manifest.json")
                                  .read_text(encoding="utf-8"))
            sha = manifest["agent_config"]["sha256"]
            if config_sha.setdefault(row["config"], sha) != sha:
                raise ValueError(f"Configuration changed between runs: {row['config']}")
            price_sha.add(manifest["prices"]["sha256"])
        if len(price_sha) != 1:
            raise ValueError("Batch runs used different price tables")
        admitted = audit["admission"] == "admitted"
        phases = {}
        for phase in ("validation", "exam"):
            rows = {}
            for config, item in summary["config_summary"][phase].items():
                if phase == "exam" and config == "tuned_lib" and not admitted:
                    continue
                buckets = {key: 0 for key in BUCKETS}
                for run in summary["runs"]:
                    if run["phase"] == phase and run["config"] == config:
                        for key in BUCKETS:
                            buckets[key] += int(run["buckets"][key])
                rows[config] = {"scores": item["scores"], "score_mean": item["score_mean"],
                                "tokens_mean": item["tokens_mean"], "tokens_sum": item["tokens_sum"],
                                "buckets": buckets}
            phases[phase] = rows
        kept = ("available", "claim", "mean_score_gap", "paired_differences", "within_config_noise",
                "baseline_mean_cost", "candidate_mean_cost", "mean_cost_change_pct", "reason")
        comparisons = {key: {name: item[name] for name in kept if name in item}
                       for key, item in summary["exam"]["comparisons"].items()}
        admission = summary["validation"]["admission"]
        evidence = {
            "schema": FORMAL_SCHEMA, "label": "FORMAL", "currency": "TOKENS",
            "bank_version": audit["bank_version"], "code_commit": audit["code_commit"],
            "prices_sha256": price_sha.pop(), "config_sha256": config_sha,
            "selected_config_sha256": audit["selected_config_sha256"],
            "batch_report_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
            "audit": {key: audit[key] for key in (
                "ready", "task_budget", "batch_runs", "phase_runs", "study_runs",
                "study_task_cost", "proposal_cost", "total_research_cost", "admission",
                "over_budget_tasks")},
            "admission": {"claim": admission["claim"],
                          "score_not_lower_each_rep": admission["score_not_lower_each_rep"],
                          "cost_reduced_on_average": admission["cost_reduced_on_average"],
                          "pairs": summary["validation"]["pairs"]},
            "phases": phases, "comparisons": comparisons,
            "breakeven": (summary["breakeven"] if admitted else
                          {"computed": False, "reasons": ["研究版未通过验证准入，未参加考试"]}),
            "warnings": summary["warnings"],
        }
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise FormalAuditError(str(exc)) from exc
    if not formal_evidence_ok(evidence):
        raise FormalAuditError("Audited batch could not be condensed into roadshow evidence")
    return evidence


def formal_evidence_ok(data: object) -> bool:
    """Accept only audited, internally consistent FORMAL TOKENS evidence."""
    try:
        admitted = data["admission"]["claim"] == "admitted"
        return (data["schema"] == FORMAL_SCHEMA and data["label"] == "FORMAL"
                and data["currency"] == "TOKENS" and data["audit"]["ready"] is True
                and data["admission"]["claim"] in ADMISSION_NAMES
                and data["audit"]["admission"] == data["admission"]["claim"]
                and re.fullmatch(r"[0-9a-f]{40}", data["code_commit"]) is not None
                and bool(data["phases"]["validation"])
                and set(data["phases"]["exam"]) == set(CONFIG_NAMES if admitted else
                                                       ("naive", "default", "default_lib"))
                and data["audit"]["phase_runs"]["exam"] == (12 if admitted else 9))
    except (KeyError, TypeError):
        return False


def committed_formal(repo: Path) -> dict | None:
    try:
        data = json.loads(committed_text(repo, FORMAL_EVIDENCE))
    except ValueError:
        return None
    return data if formal_evidence_ok(data) else None


def formal_results(data: dict | None) -> str:
    """HTML for audited formal evidence; empty unless the evidence is complete and usable."""
    if data is None or not formal_evidence_ok(data):
        return ""
    try:
        admitted = data["admission"]["claim"] == "admitted"
        audit, runs, sha = data["audit"], data["audit"]["phase_runs"], data["config_sha256"]
        rows = []
        for phase in ("validation", "exam"):
            for config, item in data["phases"][phase].items():
                buckets = item["buckets"]
                scores = " / ".join(f"{float(value):.4f}" for value in item["scores"])
                rows.append(
                    f"<tr><td>{PHASE_NAMES[phase]}</td><td>{safe(CONFIG_NAMES[config])}</td>"
                    f"<td>{float(item['score_mean']):.4f}</td><td>{safe(scores)}</td>"
                    f"<td>{float(item['tokens_mean']):,.0f}</td>"
                    f"<td>{int(buckets['input']):,} / {int(buckets['cached_input']):,} / "
                    f"{int(buckets['output']):,} / {int(buckets['reasoning']):,}</td></tr>")
        lines = [
            f"独立审计通过（scripts/audit_formal_outputs.py）：{safe(data['bank_version'])}，"
            f"代码提交 {safe(data['code_commit'][:12])}，共 {int(audit['batch_runs'])} 个 run"
            f"（练习 {int(runs['practice'])} / 验证 {int(runs['validation'])} / 考试 {int(runs['exam'])}），"
            f"单题上限 {int(audit['task_budget']):,} token，超限 {int(audit['over_budget_tasks'])} 题。",
            f"配置 SHA-256：朴素版 {safe(sha['naive'][:12])}…；默认设置 {safe(sha['default'][:12])}…；"
            f"研究版 {safe(data['selected_config_sha256'][:12])}…；价目表 {safe(data['prices_sha256'][:12])}…。",
            f"验证准入：{ADMISSION_NAMES[data['admission']['claim']]}（三遍分数均不低于练过的库："
            f"{'是' if data['admission']['score_not_lower_each_rep'] else '否'}；平均 token 更低："
            f"{'是' if data['admission']['cost_reduced_on_average'] else '否'}）；"
            f"考试 {len(data['phases']['exam'])} 条线各三遍。",
        ]
        if not admitted:
            lines.append("研究版未通过准入、未参加考试：不展示研究版考试成绩，也不计算回本点。")
        for key, name in COMPARISON_NAMES.items():
            if key == "tuned_lib_vs_default_lib" and not admitted:
                continue
            comparison = data["comparisons"][key]
            if not comparison.get("available"):
                lines.append(f"考试 {name}：未比较（{safe(comparison.get('reason', ''))}）。")
                continue
            change = comparison.get("mean_cost_change_pct")
            lines.append(
                f"考试 {name}：{safe(CLAIM_NAMES.get(comparison['claim'], comparison['claim']))}；"
                f"三遍平均分差 {float(comparison['mean_score_gap']):+.4f}；"
                + (f"每遍平均 token 变化 {float(change):+.1f}%。" if change is not None
                   else "每遍平均 token 变化未计算。"))
        if admitted:
            breakeven = data["breakeven"]
            if breakeven.get("computed"):
                lines.append(
                    f"研究版数学回本点 {int(breakeven['breakeven_tasks'])} 题"
                    f"（研究总成本 {float(breakeven['research_cost_tokens']):,.0f} token，"
                    f"考试每题平均少用 {float(breakeven['savings_per_task_tokens']):,.1f} token）。")
            else:
                lines.append("研究版回本点：不计算（"
                             + safe("；".join(map(str, breakeven.get("reasons", [])))) + "）。")
        lines.append(f"自研究总成本 {float(audit['total_research_cost']):,.0f} token"
                     f"（{int(audit['study_runs'])} 次练习运行加提案调用）。")
        if data.get("warnings"):
            lines.append(f"汇总含 {len(data['warnings'])} 条数据提醒，对外引用前请先核对。")
    except (KeyError, TypeError, ValueError, AttributeError):
        return ""
    return (
        '<div class="card"><div class="label">正式批次结果 · 已通过独立审计</div>'
        '<table class="results"><thead><tr><th>阶段</th><th>配置</th><th>三遍平均归一化分数</th>'
        '<th>各遍分数</th><th>每遍平均 token</th><th>四桶合计（三遍）input / cached / output / reasoning</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
        + "".join(f'<p class="small">{line}</p>' for line in lines)
        + '<p class="small">全部成本为内部 TOKENS，不是现金价格或官方比赛成绩。来源：'
        + safe(FORMAL_EVIDENCE) + '（生成时从已保存批次输出只读审计并汇总）。</p></div>')


def evidence(repo: Path, formal: dict | None = None) -> dict[str, str]:
    readme = committed_text(repo, "README.md")
    concurrency = committed_text(repo, "docs/真实模型并发基准.md")
    study = section(committed_text(repo, "docs/真实模型自研究试验.md"), "第二轮：逐题证据提案")
    transfer = committed_text(repo, "docs/真实模型迁移复评.md")
    decisions = committed_text(repo, "docs/项目交付与决策.md")
    calibration = calibration_evidence(repo)

    test_match = (re.search(r"主测试\s*(\d+)\s*项、技能层\s*(\d+)\s*项、看板\s*(\d+)\s*项", readme)
                  or re.search(r"主回归\s*(\d+)\s*项.*?技能层\s*(\d+)\s*项.*?看板\s*(\d+)\s*项",
                               readme, re.S))
    tests = sum(map(int, test_match.groups())) if test_match else None
    task_match = re.search(r"(\d+)\s*道入库题", readme)
    task_count = task_match.group(1) if task_match else "待核对"

    parallel_rows = table_rows(concurrency)
    wall = next((row for row in parallel_rows if row and row[0] == "墙钟"), [])
    parallel = numeric(wall[1]) if len(wall) > 1 else None
    serial = numeric(wall[2]) if len(wall) > 2 else None

    def paired_results(body: str, second_label: str) -> tuple[list[str], list[str]]:
        rows = table_rows(body)
        default = next((row for row in rows if row and row[0] == "默认 v2"), [])
        candidate = next((row for row in rows if row and second_label in row[0]), [])
        return default, candidate

    study_default, study_candidate = paired_results(study, "auto-01")
    transfer_default, transfer_candidate = paired_results(transfer, "auto-01")

    def value(row: list[str], index: int) -> float | None:
        return numeric(row[index]) if len(row) > index else None

    study_score = (value(study_candidate, 1), value(study_default, 1))
    study_tokens = (value(study_candidate, 2), value(study_default, 2))
    transfer_score = (value(transfer_candidate, 1), value(transfer_default, 1))
    transfer_tokens = (value(transfer_candidate, 2), value(transfer_default, 2))

    def delta(pair: tuple[float | None, float | None], percent: bool = False) -> str:
        current, baseline = pair
        if current is None or baseline is None or (percent and baseline == 0):
            return "待核对"
        difference = (current / baseline - 1) * 100 if percent else current - baseline
        return (f"{difference:+.1f}%" if percent else f"{difference:+.4f}")

    project_intro = next(
        (plain(block) for block in readme.split("\n\n") if block.startswith("这是按《分工文档》")),
        first_paragraph(readme),
    )
    current_limits = first_paragraph(section(readme, "当前限制"))
    limit_sentences = [part.strip() for part in re.split(r"(?<=。)", current_limits) if part.strip()]
    priority = [part for part in limit_sentences
                if "bank-v1" in part or "正式货币" in part or "正式自研究" in part]
    current_limits = "".join(priority[:2] or limit_sentences[:2])
    phase_sentence = next((part.strip() for part in project_intro.split("。")
                           if "v2" in part and "验证与考试尚未启动" in part), "")
    if phase_sentence and phase_sentence not in current_limits:
        current_limits = (current_limits + phase_sentence + "。").strip()
    next_source = (section(readme, "冻结后下一步") if calibration else
                   section(decisions, "尚待外部输入与正式验收"))
    next_steps = [excerpt(item, 120) for item in number_list(next_source, 3)]
    v2_pending = bool(phase_sentence and "v2 正在" in phase_sentence)
    formal_pending = v2_pending or any(phrase in readme for phrase in (
        "正式四线比较尚未执行",
        "正式自研究、一次性验证和考试仍待运行",
        "验证与考试尚未启动",
    ))
    try:
        prices = json.loads(committed_text(repo, "config/prices.json"))
    except (ValueError, TypeError):
        prices = {}
    if not isinstance(prices, dict):
        prices = {}
    internal_tokens = (prices.get("schema") == "prices/1"
                       and prices.get("currency") == "TOKENS"
                       and prices.get("test_only") is False)
    if internal_tokens:
        accounting_notice = (
            "真实模型 dev 试验的 TEST ONLY 账本按网关返回的 token 计量；"
            "脚本角色 TEST 联调不是实际模型成本。冻结题库内部正式价格表为 "
            "TOKENS、test_only=false，同样不是现金价格或官方成绩。"
        )
    else:
        accounting_notice = "TEST 联调不能代表真实模型成本、现金价格或官方成绩。"
    results = formal_results(formal if formal is not None else committed_formal(repo))
    if results:
        pass  # committed formal results replace the pending wording
    elif v2_pending:
        accounting_notice += "v2 自研究、验证与考试数字均待对应真实账本审计后填入。"
    elif formal_pending:
        accounting_notice += "验证与考试数字待对应真实账本审计后填入。"
    return {
        "intro": project_intro,
        "formal_results": results,
        "status": ("内部 token 正式批次已完成 · 独立审计通过" if results else
                   "内部 token v2 练习研究进行中 · 验证／考试待真实账本" if v2_pending else
                   "题库已冻结 · 正式四线待运行" if calibration and formal_pending else
                   "DEV / TEST ONLY · 正式四线比较未执行" if formal_pending else
                   "阶段状态请以 README 为准"),
        "tasks": task_count,
        "tests": str(tests) if tests is not None else "待核对",
        "parallel": fmt(parallel, 3),
        "serial": fmt(serial, 3),
        "wall_ratio": fmt(serial / parallel if parallel and serial else None, 2),
        "study_score": delta(study_score),
        "study_tokens": delta(study_tokens, percent=True),
        "transfer_score": delta(transfer_score),
        "transfer_tokens": delta(transfer_tokens, percent=True),
        "limits": current_limits or "请查看 README 的当前限制。",
        "accounting_notice": accounting_notice,
        "calibration": calibration,
        "next_steps": next_steps,
    }


HTML = Template("""<!doctype html>
<!-- source-head: $head -->
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>自进化 Research Agent · 路演汇报</title>
  <style>
    :root { color-scheme: dark; --bg: #081421; --panel: #10283a; --ink: #ecf7fb; --muted: #a6c0cc; --cyan: #5ce0d4; --amber: #ffd179; --line: #2a5060; }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; scroll-snap-type: y mandatory; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif; }
    a { color: var(--cyan); text-underline-offset: .2em; }
    .slide { min-height: 100svh; scroll-snap-align: start; display: flex; flex-direction: column; justify-content: center; padding: clamp(32px, 6vw, 96px); position: relative; overflow: hidden; }
    .slide::before { content: ""; position: absolute; width: 45vw; height: 45vw; right: -16vw; top: -18vw; border-radius: 50%; background: radial-gradient(circle, #136c79 0, transparent 68%); opacity: .34; pointer-events: none; }
    .slide:nth-child(even) { background: #0b1c2a; }
    .inner { position: relative; z-index: 1; max-width: 1450px; width: 100%; margin: auto; }
    .eyebrow { font-size: clamp(14px, 1.35vw, 19px); letter-spacing: .17em; text-transform: uppercase; font-weight: 750; color: var(--cyan); }
    h1 { font-size: clamp(56px, 8vw, 124px); letter-spacing: -.055em; line-height: 1.02; margin: .2em 0; }
    h2 { font-size: clamp(40px, 5vw, 76px); letter-spacing: -.045em; line-height: 1.08; margin: .18em 0 .45em; }
    h3 { font-size: clamp(24px, 2.5vw, 36px); margin: 0 0 .55em; }
    p { line-height: 1.65; }
    .lead { font-size: clamp(20px, 2.2vw, 32px); max-width: 1080px; color: var(--muted); }
    .small { font-size: clamp(14px, 1.15vw, 18px); color: var(--muted); }
    .tag { display: inline-block; border: 1px solid #ffcd7777; background: #ffcd7719; color: var(--amber); padding: .55em .85em; border-radius: 999px; font-size: 15px; font-weight: 700; }
    .grid { display: grid; gap: 22px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .card { background: linear-gradient(145deg, #133145, #0e2637); border: 1px solid var(--line); border-radius: 22px; padding: clamp(22px, 2.5vw, 38px); box-shadow: 0 20px 50px #0002; }
    .card p { color: var(--muted); font-size: clamp(17px, 1.5vw, 22px); margin-bottom: 0; }
    .metric { display: block; font-size: clamp(44px, 5vw, 76px); font-weight: 780; letter-spacing: -.055em; color: var(--cyan); line-height: 1.1; }
    .metric.amber { color: var(--amber); }
    .label { font-size: 17px; font-weight: 700; color: var(--muted); letter-spacing: .06em; }
    .split { display: grid; grid-template-columns: 1.1fr .9fr; gap: clamp(25px, 5vw, 80px); align-items: start; }
    .timeline { list-style: none; padding: 0; margin: 0; max-height: 57vh; overflow: auto; }
    .timeline li { border-left: 2px solid var(--line); padding: 0 0 18px 22px; position: relative; }
    .timeline li::before { content: ""; position: absolute; left: -6px; top: 5px; width: 10px; height: 10px; border-radius: 50%; background: var(--cyan); }
    .timeline strong { display: block; font-size: clamp(17px, 1.5vw, 23px); line-height: 1.35; }
    .timeline span { font-size: 14px; color: var(--muted); }
    .chips { display: flex; gap: 10px; flex-wrap: wrap; }
    .chips span { border: 1px solid var(--line); padding: .55em .8em; border-radius: 999px; color: var(--muted); }
    .notice { border-left: 4px solid var(--amber); background: #ffd17915; padding: 18px 25px; border-radius: 0 14px 14px 0; font-size: clamp(17px, 1.5vw, 22px); }
    .footer { position: absolute; left: clamp(32px, 6vw, 96px); right: clamp(32px, 6vw, 96px); bottom: 24px; display: flex; justify-content: space-between; color: var(--muted); font-size: 13px; z-index: 1; }
    .nav { position: fixed; z-index: 20; right: 22px; top: 50%; transform: translateY(-50%); display: grid; gap: 12px; }
    .nav a { display: block; width: 12px; height: 12px; border: 1px solid var(--muted); border-radius: 50%; background: #ffffff20; }
    .nav a:hover, .nav a:focus { background: var(--cyan); }
    .results { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: clamp(14px, 1.3vw, 18px); }
    .results th, .results td { padding: .45em .7em; border-bottom: 1px solid var(--line); text-align: right; }
    .results th:first-child, .results td:first-child, .results th:nth-child(2), .results td:nth-child(2) { text-align: left; }
    .results th { color: var(--muted); font-weight: 700; }
    details { color: var(--muted); font-size: 15px; }
    details summary { cursor: pointer; color: var(--cyan); }
    .files { columns: 2; max-height: 28vh; overflow: auto; padding-left: 20px; overflow-wrap: anywhere; }
    @media (max-width: 900px) { html { scroll-snap-type: none; } .slide { min-height: auto; padding: 60px 26px 75px; } .grid, .split { grid-template-columns: 1fr; } .nav { display: none; } .footer { left: 26px; right: 26px; } .timeline { max-height: none; } }
    @media print { html { scroll-snap-type: none; } .slide { min-height: 100vh; page-break-after: always; -webkit-print-color-adjust: exact; print-color-adjust: exact; } .nav { display: none; } .timeline, .files { max-height: none; } }
  </style>
</head>
<body>
  <nav class="nav" aria-label="汇报页面导航"><a href="#cover" aria-label="封面"></a><a href="#system" aria-label="系统"></a><a href="#evidence" aria-label="证据"></a><a href="#changes" aria-label="更新"></a><a href="#next" aria-label="下一步"></a></nav>
  <section class="slide" id="cover"><div class="inner">
    <div class="eyebrow">Research Agent / Repository Briefing</div>
    <h1>自进化<br>Research Agent</h1>
    <p class="lead">让 Research Agent 研究自己。</p>
    <p class="small" style="max-width:1050px;font-size:18px">$intro</p>
    <p><span class="tag">$status</span></p>
    <p class="small">截至 $updated · 代码 $short_head · 按仓库已提交内容生成</p>
  </div><div class="footer"><span>自进化 Research Agent · A / B / C</span><span>01 / 05</span></div></section>
  <section class="slide" id="system"><div class="inner">
    <div class="eyebrow">01 / System</div><h2>从研究到审计，三层闭环</h2>
    <div class="grid">
      <article class="card"><div class="eyebrow">A / Agent</div><h3>规划与实验</h3><p>规划、写解、实验树、预算和能力库。记录尝试与回放证据。</p></article>
      <article class="card"><div class="eyebrow">B / Bench</div><h3>任务与隔离评分</h3><p>提速与表格二分类两类任务，沙箱隔离，并保留 dev 测量。</p></article>
      <article class="card"><div class="eyebrow">C / Control</div><h3>编排与报告</h3><p>价格账本、练习／验证／考试、自研究、报告和只读看板。</p></article>
    </div>
    <div class="grid" style="margin-top:22px"><div class="card"><span class="metric">$tasks</span><span class="label">入库题 · README</span></div><div class="card"><span class="metric">$tests</span><span class="label">已通过测试 · README</span></div><div class="card"><span class="metric">$total_commits</span><span class="label">累计源码相关提交</span></div></div>
  </div><div class="footer"><span>来源：<a href="../README.md">README</a> · <a href="项目交付与决策.md">决策记录</a></span><span>02 / 05</span></div></section>
  <section class="slide" id="evidence"><div class="inner">
    <div class="eyebrow">02 / Evidence</div><h2>真实模型 dev 证据</h2>
    <div class="grid">
      <article class="card"><div class="label">三路并发默认配置</div><span class="metric">$parallel<span style="font-size:.4em"> 秒</span></span><p>旧串行约 $serial 秒；观测墙钟比值 $wall_ratio×。两次代码与环境不同，不能视为稳定加速。</p><p class="small"><a href="真实模型并发基准.md">查看审计记录 ↗</a></p></article>
      <article class="card"><div class="label">第二轮自动自研究</div><span class="metric">$study_tokens</span><p>候选每遍 token 相对默认版变化；平均归一化分数变化 $study_score。质量未持平，候选未推广。</p><p class="small"><a href="真实模型自研究试验.md">查看自研究记录 ↗</a></p></article>
      <article class="card"><div class="label">跨题迁移复评</div><span class="metric amber">$transfer_score</span><p>归一化分数变化，但每遍 token 变化 $transfer_tokens；结果仍属于 dev，不能证明正式准入。</p><p class="small"><a href="真实模型迁移复评.md">查看迁移记录 ↗</a></p></article>
    </div>
    $calibration_notice
    $formal_results
    <p class="notice">$accounting_notice</p>
  </div><div class="footer"><span>数值从仓库已提交证据文档提取</span><span>03 / 05</span></div></section>
  <section class="slide" id="changes"><div class="inner">
    <div class="eyebrow">03 / Repository Delta</div><h2>本轮仓库更新</h2>
    <div class="split"><div>
      <div class="grid" style="grid-template-columns:repeat(2,minmax(0,1fr));margin-bottom:24px"><div class="card"><span class="metric">$period_commits</span><span class="label">新增相关提交</span></div><div class="card"><span class="metric">$file_count</span><span class="label">涉及文件</span></div></div>
      <p class="small">$period_label</p><div class="chips">$categories</div>
      <details style="margin-top:24px"><summary>展开变更文件</summary><ul class="files">$files</ul></details>
    </div><ol class="timeline">$commit_items</ol></div>
  </div><div class="footer"><span>提交记录：<a href="https://github.com/QZTnb/research-agent/commits/main/">Git 历史 ↗</a></span><span>04 / 05</span></div></section>
  <section class="slide" id="next"><div class="inner">
    <div class="eyebrow">04 / Decision</div><h2>下一步与边界</h2>
    <div class="split"><div><div class="notice">$limits</div><p class="small">上方状态摘自 <a href="../README.md">README</a>；请以原始证据和正式预检为准。</p></div><div class="card"><h3>内部研究后续路径</h3><ol>$next_items</ol><p class="small"><a href="项目交付与决策.md">查看完整决策记录 ↗</a></p></div></div>
    <p class="small">每小时检查仓库；仅当新的源码或文档提交出现时重生成。路演时可用方向键翻页，也可打印为 PDF。</p>
  </div><div class="footer"><span>生成器：scripts/generate_roadshow_report.py</span><span>05 / 05</span></div></section>
  <script>
    const slides = Array.from(document.querySelectorAll(".slide"));
    document.addEventListener("keydown", function (event) {
      if (!["ArrowRight", "ArrowDown", "PageDown", "ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) return;
      const current = Math.max(0, slides.findIndex(function (slide) { return slide.getBoundingClientRect().top > -window.innerHeight / 2; }));
      const next = Math.min(slides.length - 1, Math.max(0, current + (["ArrowRight", "ArrowDown", "PageDown"].includes(event.key) ? 1 : -1)));
      slides[next].scrollIntoView({ behavior: "smooth" }); event.preventDefault();
    });
  </script>
</body>
</html>
""")


def build(repo: Path, head: str, commits: list[tuple[str, str, str]], files: list[str], label: str,
          formal: dict | None = None) -> str:
    facts = evidence(repo, formal)
    categories = Counter(category(path) for path in files)
    category_html = "".join(f"<span>{safe(name)} · {count}</span>" for name, count in categories.most_common())
    file_html = "".join(f"<li>{safe(path)}</li>" for path in files)
    commit_html = "".join(
        f"<li><strong>{safe(subject)}</strong><span>{safe(date)} · {safe(sha[:7])}</span></li>"
        for sha, date, subject in commits[:10]
    ) or "<li><strong>本轮没有新增提交</strong></li>"
    next_html = "".join(f"<li>{safe(item)}</li>" for item in facts["next_steps"])
    calibration_html = (
        '<p class="notice">' + safe(facts["calibration"])
        + ' <a href="提速朴素成本校准.md">提速明细 ↗</a>'
        + ' · <a href="表格朴素成本校准.md">表格明细 ↗</a></p>'
        if facts["calibration"] else ""
    )
    total = len(source_commits(repo, "HEAD"))
    updated = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M UTC+08:00")
    return HTML.substitute(
        head=head, short_head=head[:7], updated=updated,
        intro=safe(facts["intro"]), status=safe(facts["status"]),
        tasks=safe(facts["tasks"]), tests=safe(facts["tests"]), total_commits=total,
        parallel=safe(facts["parallel"]), serial=safe(facts["serial"]), wall_ratio=safe(facts["wall_ratio"]),
        study_score=safe(facts["study_score"]), study_tokens=safe(facts["study_tokens"]),
        transfer_score=safe(facts["transfer_score"]), transfer_tokens=safe(facts["transfer_tokens"]),
        limits=safe(facts["limits"]), next_items=next_html,
        accounting_notice=safe(facts["accounting_notice"]),
        calibration_notice=calibration_html, formal_results=facts["formal_results"],
        period_commits=len(commits), file_count=len(files), period_label=safe(label),
        categories=category_html, files=file_html, commit_items=commit_html,
    )


def generate(repo: Path, output: Path, formal: dict | None = None) -> bool:
    head = git(repo, "rev-parse", "HEAD")
    older = previous_head(output)
    if older and is_ancestor(repo, older):
        revision = older + "..HEAD"
        label = "对比上次生成时的源码提交 " + older[:7] + " → " + head[:7]
    else:
        revision = "HEAD"
        label = "首次基线：汇总当前仓库全部已提交历史"
    commits = source_commits(repo, revision)
    if older and not commits and formal is None:
        return False
    if formal is not None:
        target = repo / FORMAL_EVIDENCE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(formal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    files = changed_files(repo, revision)
    content = build(repo, head, commits, files, label, formal)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") == content:
        return False
    output.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal-batch-dir", type=Path,
                        help="finished formal batch to audit and show (needs --auto-study-dir)")
    parser.add_argument("--auto-study-dir", type=Path,
                        help="self-research directory that selected the batch's tuned config")
    args = parser.parse_args()
    if (args.formal_batch_dir is None) != (args.auto_study_dir is None):
        parser.error("--formal-batch-dir and --auto-study-dir must be given together")
    repo = args.repo.resolve()
    output = args.output.resolve() if args.output else repo / REPORT
    formal = None
    if args.formal_batch_dir is not None:
        try:
            formal = load_formal_evidence(args.formal_batch_dir, args.auto_study_dir)
        except FormalAuditError as exc:
            print(f"formal audit failed; nothing written: {exc}", file=sys.stderr)
            raise SystemExit(1)
        if formal is None:
            print("formal batch has no batch_report.json yet; keeping the pending wording",
                  file=sys.stderr)
    print("updated" if generate(repo, output, formal) else "unchanged", output)


if __name__ == "__main__":
    main()
