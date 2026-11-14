"""Analyze an A/B experiment campaign written by scripts/run_ab_experiment.py.

Reads every run of the campaign (its ab_run.json label and repetition, plus the
task ledgers, experiment trees and holdout files), builds a per-task table with
evaluation.metrics, and compares each candidate with the baseline (the first
configuration of the protocol) using evaluation.compare.compare_paired_tasks:
paired by (task, repetition), tabular quality as raw AUC lift, speedup quality
clipped to [-1, 2], cost as total tokens. Writes a JSON file and a one-page
Markdown summary. Nothing here calls a model or a scorer.

    python -m scripts.analyze_ab runs/ab-20261115-120000-ab12 \
        --json /tmp/ab.json --markdown /tmp/ab.md
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from evaluation.compare import QUALITY_CLIP, compare_paired_tasks, truncate_score
from evaluation.metrics import COMPLETION_MODES, find_run_dirs, run_metrics

_TABLE_FIELDS = (
    "task_id", "scenario", "stop_reason", "total_tokens", "main_loop_tokens", "tokens_by_role",
    "tokens_to_reference", "tokens_to_best", "tokens_after_best", "reached_reference",
    "experiments", "accepted_nodes", "model_calls", "failed_calls", "best_dev_score",
    "holdout_score", "holdout_raw", "holdout_raw_lift", "holdout_failed", "has_task_end")
_VERDICT_TEXT = {
    "cheaper_and_not_worse": "更省，且质量没有显著下降（按区间判断，成立）",
    "cheaper_quality_inconclusive": "更省，但质量差异的区间覆盖了负值，不能说质量没降",
    "cheaper_but_worse": "更省，但质量显著下降",
    "no_clear_saving": "成本没有显著差异",
    "costlier": "更贵",
    "insufficient_data": "数据不足，无法判断",
}


def collect_rows(experiment_dir: Path, *, completion: str = "experiment_score") -> tuple[list[dict], list[dict]]:
    """One row per finished task per (configuration, repetition); plus what was left out."""
    rows, excluded = [], []
    for run_dir in find_run_dirs([experiment_dir]):
        sidecar_path = run_dir / "ab_run.json"
        if not sidecar_path.is_file():
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        manifest_tasks = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("tasks", [])
        seen = set()
        for task in run_metrics(run_dir, completion=completion)["tasks"]:
            seen.add(task["task_id"])
            if task["stop_reason"] is None:
                excluded.append({"run_id": run_dir.name, "label": sidecar["label"], "rep": sidecar["rep"],
                                 "task_id": task["task_id"], "reason": "main loop did not finish"})
                continue
            rows.append({**{key: task.get(key) for key in _TABLE_FIELDS},
                         "config": sidecar["label"], "rep": sidecar["rep"], "run_id": run_dir.name})
        for task_id in manifest_tasks:
            if task_id not in seen:
                excluded.append({"run_id": run_dir.name, "label": sidecar["label"], "rep": sidecar["rep"],
                                 "task_id": task_id, "reason": "task never started"})
    rows.sort(key=lambda row: (row["config"], row["rep"], row["task_id"]))
    return rows, excluded


def _config_summary(rows: list[dict]) -> dict:
    def avg(key):
        values = [row[key] for row in rows if row[key] is not None]
        return mean(values) if values else None

    truncated = [truncate_score(row["holdout_score"]) for row in rows]
    truncated = [value for value in truncated if value is not None]
    reached = [row["tokens_to_reference"] for row in rows if row["tokens_to_reference"] is not None]
    by_task = {}
    for task_id in sorted({row["task_id"] for row in rows}):
        group = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = {
            "runs": len(group),
            "mean_total_tokens": mean(row["total_tokens"] for row in group),
            "mean_main_loop_tokens": mean(row["main_loop_tokens"] for row in group),
            "reached_reference": sum(1 for row in group if row["reached_reference"]),
        }
    return {
        "rows": len(rows), "runs": len({row["run_id"] for row in rows}),
        "mean_total_tokens": avg("total_tokens"), "mean_main_loop_tokens": avg("main_loop_tokens"),
        "mean_tokens_after_best": avg("tokens_after_best"),
        "reached_reference": len(reached),
        "mean_tokens_to_reference": mean(reached) if reached else None,
        "stop_reasons": dict(sorted(Counter(row["stop_reason"] for row in rows).items())),
        "mean_truncated_holdout_score": mean(truncated) if truncated else None,
        "holdout_failures": sum(1 for row in rows if row["holdout_failed"]),
        "by_task": by_task,
    }


def analyze(experiment_dir: Path, *, baseline: str | None = None, quality_margin=0.0,
            completion: str = "experiment_score") -> dict:
    """Per-task table, per-configuration summaries and paired comparisons."""
    experiment_dir = Path(experiment_dir)
    protocol_path = experiment_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    rows, excluded = collect_rows(experiment_dir, completion=completion)
    labels = list(protocol.get("configs", {})) or sorted({row["config"] for row in rows})
    labels = [label for label in labels if any(row["config"] == label for row in rows)]
    if not labels:
        raise ValueError(f"No A/B runs with ab_run.json found under {experiment_dir}")
    baseline = baseline or protocol.get("baseline") or labels[0]
    if baseline not in labels:
        raise ValueError(f"Baseline {baseline!r} has no finished runs; found {labels}")
    comparisons = {label: compare_paired_tasks(rows, baseline=baseline, candidate=label,
                                               quality_margin=quality_margin)
                   for label in labels if label != baseline}
    return {
        "schema": "ab_analysis/1", "experiment_dir": str(experiment_dir),
        "label": protocol.get("label"), "dry_run": protocol.get("dry_run"),
        "tasks": protocol.get("tasks"), "reps": protocol.get("reps"),
        "baseline": baseline, "configs": {label: _config_summary(
            [row for row in rows if row["config"] == label]) for label in labels},
        "table": rows, "excluded": excluded, "comparisons": comparisons,
        "completion_mode": completion, "quality_clip": list(QUALITY_CLIP),
    }


def _fmt(value, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def render_markdown(analysis: dict) -> str:
    lines = [f"# A/B 实验摘要：{analysis['baseline']} 为基线",
             "", f"- 数据：`{analysis['experiment_dir']}`；题目 {', '.join(analysis.get('tasks') or [])}；"
                 f"每个配置 {analysis.get('reps')} 遍。",
             "- 性质：dev 研究（练习题，TEST 价目，1 单位 = 1 个 token），不是正式考试。"]
    if analysis.get("dry_run"):
        lines.append("- **这是 dry-run：模型和打分都是合成的，数字没有意义。**")
    lines += ["", "## 各配置概况（每题每遍一行）", "",
              "| 配置 | 样本 | 平均总 token | 平均主循环 token | 达到参考解 | 平均到参考解 token | 截断均分 | 停止原因 | holdout 失败 |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for label, item in analysis["configs"].items():
        reasons = "，".join(f"{key} {count}" for key, count in item["stop_reasons"].items())
        lines.append(
            f"| {label} | {item['rows']} | {_fmt(item['mean_total_tokens'])} | "
            f"{_fmt(item['mean_main_loop_tokens'])} | {item['reached_reference']}/{item['rows']} | "
            f"{_fmt(item['mean_tokens_to_reference'])} | {_fmt(item['mean_truncated_holdout_score'], 3)} | "
            f"{reasons} | {item['holdout_failures']} |")
    for label, comparison in analysis["comparisons"].items():
        cost, quality = comparison["cost"], comparison["quality"]
        overall = cost["overall"]
        lines += ["", f"## {label} 对 {analysis['baseline']}（{comparison['n_pairs']} 对 题×遍）", "",
                  f"**结论：{_VERDICT_TEXT.get(comparison['conclusion'], comparison['conclusion'])}**", ""]
        if overall["mean"] is not None:
            interval = overall["ci95"]
            interval_text = "无（样本不足）" if interval is None else f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
            change = overall.get("relative_change")
            lines.append(
                f"- 成本：每题平均 token 差 {_fmt(overall['mean'])}"
                f"（{'-' if change is None else format(change, '+.1%')}），95% 区间 {interval_text}；"
                f"符号检验 省 {overall['sign_test']['candidate_lower']} / 费 "
                f"{overall['sign_test']['candidate_higher']} / 平 {overall['sign_test']['ties']}，"
                f"p={overall['sign_test']['p_two_sided']:.3f}")
        names = {"tab": "表格题（raw AUC 提升差）", "spd": "提速题（截断分差）"}
        for scenario, entry in quality["by_scenario"].items():
            interval = entry["ci95"]
            interval_text = "无（样本不足）" if interval is None else f"[{_fmt(interval[0], 4)}, {_fmt(interval[1], 4)}]"
            lines.append(
                f"- 质量 {names.get(scenario, scenario)}：平均差 {_fmt(entry['mean'], 4)}，"
                f"95% 区间 {interval_text}，判定 {entry['verdict']}（容差 {entry['margin']}），"
                f"符号检验 升 {entry['sign_test']['candidate_higher']} / 降 "
                f"{entry['sign_test']['candidate_lower']} / 平 {entry['sign_test']['ties']}，"
                f"p={entry['sign_test']['p_two_sided']:.3f}")
        truncated = comparison["truncated_mean_holdout_score"]
        lines.append(f"- 截断到 {comparison['truncated_mean_holdout_score']['clip']} 后的 holdout 均分："
                     f"{analysis['baseline']} {_fmt(truncated[analysis['baseline']]['overall']['mean'], 3)}，"
                     f"{label} {_fmt(truncated[label]['overall']['mean'], 3)}")
        lines += ["", "| 题 | 遍 | 基线 token | 候选 token | token 差 | 质量差 | 质量口径 |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for pair in comparison["pairs"]:
            lines.append(
                f"| {pair['task_id']} | {pair['rep']} | {_fmt(pair['baseline_tokens'])} | "
                f"{_fmt(pair['candidate_tokens'])} | {_fmt(pair['token_diff'])} | "
                f"{_fmt(pair['quality_diff'], 4)} | {pair['quality_basis'] or '-'} |")
        if comparison["unpaired"]:
            lines.append(f"\n未配对：{len(comparison['unpaired'])} 条（对方缺这题这遍），不计入。")
    if analysis["excluded"]:
        lines.append(f"\n未纳入：{len(analysis['excluded'])} 个（主循环没跑完或没开始），见 JSON 的 excluded。")
    lines.append("\n说明：成本是每题全部 token；质量判定按场景分别做（两种质量量纲不同）；区间是固定种子的 bootstrap，"
                 "把每个（题，遍）当独立样本，同一题的几遍其实共享难度，所以题少时区间偏乐观；"
                 "样本很少时区间也很宽，“不能判断”是正常结果。")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path, help="campaign directory of run_ab_experiment")
    parser.add_argument("--baseline", help="baseline label (default: the protocol's first config)")
    parser.add_argument("--json", type=Path, dest="json_path", help="default: <dir>/analysis.json")
    parser.add_argument("--markdown", type=Path, help="default: <dir>/analysis.md")
    parser.add_argument("--quality-margin", nargs="*", default=[], metavar="SCENARIO=VALUE",
                        help="non-inferiority margins, e.g. tab=0.005 spd=0.1 (default 0)")
    parser.add_argument("--completion", choices=COMPLETION_MODES, default="experiment_score")
    args = parser.parse_args(argv)
    margins: dict[str, float] = {}
    for item in args.quality_margin:
        scenario, separator, value = item.partition("=")
        try:
            margin = float(value)
        except ValueError:
            margin = -1.0
        if not separator or scenario not in ("tab", "spd") or not margin >= 0:
            parser.error("--quality-margin entries look like tab=0.005 or spd=0.1 (nonnegative)")
        margins[scenario] = margin
    analysis = analyze(args.experiment_dir, baseline=args.baseline,
                       quality_margin=margins or 0.0, completion=args.completion)
    json_path = args.json_path or args.experiment_dir / "analysis.json"
    markdown_path = args.markdown or args.experiment_dir / "analysis.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    for label, comparison in analysis["comparisons"].items():
        print(f"{label} vs {analysis['baseline']}: {comparison['conclusion']} "
              f"({comparison['n_pairs']} pairs)")
    print(f"Wrote {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
