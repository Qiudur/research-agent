"""Build a source-linked, self-contained HTML roadshow report from committed files.

The report changes only when a non-report commit reaches HEAD. This makes an
hourly scheduled run cheap and prevents report-only commits from feeding back
into the next report.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
REPORT = "docs/路演汇报.html"
SOURCE_HEAD = re.compile(r"<!-- source-head: ([0-9a-f]{40}) -->")
EXCLUDE_REPORT = ":(exclude)" + REPORT


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


def evidence(repo: Path) -> dict[str, str]:
    readme = committed_text(repo, "README.md")
    concurrency = committed_text(repo, "docs/真实模型并发基准.md")
    study = section(committed_text(repo, "docs/真实模型自研究试验.md"), "第二轮：逐题证据提案")
    transfer = committed_text(repo, "docs/真实模型迁移复评.md")
    decisions = committed_text(repo, "docs/项目交付与决策.md")

    test_match = re.search(r"主测试\s*(\d+)\s*项、技能层\s*(\d+)\s*项、看板\s*(\d+)\s*项", readme)
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

    current_limits = first_paragraph(section(readme, "当前限制"))
    limit_sentences = [part.strip() for part in re.split(r"(?<=。)", current_limits) if part.strip()]
    priority = [part for part in limit_sentences if "正式货币" in part or "正式自研究" in part]
    current_limits = "".join(priority[:2] or limit_sentences[:2])
    next_steps = [excerpt(item, 120) for item in number_list(
        section(decisions, "尚待外部输入与正式验收"), 3
    )]
    project_intro = first_paragraph(readme)
    formal_pending = "正式四线比较尚未执行" in readme or "正式验证" in current_limits
    return {
        "intro": project_intro,
        "status": "DEV / TEST ONLY · 正式四线比较未执行" if formal_pending else "阶段状态请以 README 为准",
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
    <div class="grid" style="margin-top:22px"><div class="card"><span class="metric">$tasks</span><span class="label">入库题 · README</span></div><div class="card"><span class="metric">$tests</span><span class="label">三组通过测试 · README</span></div><div class="card"><span class="metric">$total_commits</span><span class="label">累计源码相关提交</span></div></div>
  </div><div class="footer"><span>来源：<a href="../README.md">README</a> · <a href="项目交付与决策.md">决策记录</a></span><span>02 / 05</span></div></section>
  <section class="slide" id="evidence"><div class="inner">
    <div class="eyebrow">02 / Evidence</div><h2>真实模型 dev 证据</h2>
    <div class="grid">
      <article class="card"><div class="label">三路并发默认配置</div><span class="metric">$parallel<span style="font-size:.4em"> 秒</span></span><p>旧串行约 $serial 秒；观测墙钟比值 $wall_ratio×。两次代码与环境不同，不能视为稳定加速。</p><p class="small"><a href="真实模型并发基准.md">查看审计记录 ↗</a></p></article>
      <article class="card"><div class="label">第二轮自动自研究</div><span class="metric">$study_tokens</span><p>候选每遍 token 相对默认版变化；平均归一化分数变化 $study_score。质量未持平，候选未推广。</p><p class="small"><a href="真实模型自研究试验.md">查看自研究记录 ↗</a></p></article>
      <article class="card"><div class="label">跨题迁移复评</div><span class="metric amber">$transfer_score</span><p>归一化分数变化，但每遍 token 变化 $transfer_tokens；结果仍属于 dev，不能证明正式准入。</p><p class="small"><a href="真实模型迁移复评.md">查看迁移记录 ↗</a></p></article>
    </div>
    <p class="notice">TEST 单位对应报告的 token 用量，不代表现金成本。正式验证、考试及四线效果仍待完成。</p>
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
    <div class="split"><div><div class="notice">$limits</div><p class="small">上方状态摘自 <a href="../README.md">README 当前限制</a>；请以原始证据和正式预检为准。</p></div><div class="card"><h3>正式验收路径</h3><ol>$next_items</ol><p class="small"><a href="项目交付与决策.md">查看完整决策记录 ↗</a></p></div></div>
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


def build(repo: Path, head: str, commits: list[tuple[str, str, str]], files: list[str], label: str) -> str:
    facts = evidence(repo)
    categories = Counter(category(path) for path in files)
    category_html = "".join(f"<span>{safe(name)} · {count}</span>" for name, count in categories.most_common())
    file_html = "".join(f"<li>{safe(path)}</li>" for path in files)
    commit_html = "".join(
        f"<li><strong>{safe(subject)}</strong><span>{safe(date)} · {safe(sha[:7])}</span></li>"
        for sha, date, subject in commits[:10]
    ) or "<li><strong>本轮没有新增提交</strong></li>"
    next_html = "".join(f"<li>{safe(item)}</li>" for item in facts["next_steps"])
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
        period_commits=len(commits), file_count=len(files), period_label=safe(label),
        categories=category_html, files=file_html, commit_items=commit_html,
    )


def generate(repo: Path, output: Path) -> bool:
    head = git(repo, "rev-parse", "HEAD")
    older = previous_head(output)
    if older and is_ancestor(repo, older):
        revision = older + "..HEAD"
        label = "对比上次生成时的源码提交 " + older[:7] + " → " + head[:7]
    else:
        revision = "HEAD"
        label = "首次基线：汇总当前仓库全部已提交历史"
    commits = source_commits(repo, revision)
    if older and not commits:
        return False
    files = changed_files(repo, revision)
    content = build(repo, head, commits, files, label)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") == content:
        return False
    output.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve() if args.output else repo / REPORT
    print("updated" if generate(repo, output) else "unchanged", output)


if __name__ == "__main__":
    main()
