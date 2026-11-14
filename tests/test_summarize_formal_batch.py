"""Synthetic saved-output fixtures only: no model, scorer, sandbox or hidden data."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from evaluation.admission import admit_configuration
from scripts.summarize_formal_batch import (
    SummaryError, _cost_argument, main, render_markdown, summarize, write_outputs)


MODEL = "qwen3.5-27b"
CONFIGS = ("naive", "default", "default_lib", "tuned_lib")
TASKS = {
    "practice": ["spd-practice-0", "tab-practice-0"],
    "validation": ["spd-validation-0", "tab-validation-0", "tab-validation-small"],
    "exam": ["spd-exam-0", "tab-exam-0", "tab-exam-small"],
}
BASE_SCORE = {"naive": 0.0, "default": 0.2, "default_lib": 0.3, "tuned_lib": 0.4}
BASE_TOKENS = {"naive": 4000, "default": 3000, "default_lib": 2000, "tuned_lib": 1500}
BANK_TASKS = [
    {"task_id": "spd-practice-0", "tier": "practice", "baseline_raw": 1.0, "ceiling_raw": 50.0},
    {"task_id": "tab-practice-0", "tier": "practice", "baseline_raw": 0.5, "ceiling_raw": 0.6},
    {"task_id": "spd-validation-0", "tier": "validation", "baseline_raw": 1.0, "ceiling_raw": 40.0},
    {"task_id": "tab-validation-0", "tier": "validation", "baseline_raw": 0.80, "ceiling_raw": 0.805},
    {"task_id": "tab-validation-small", "tier": "validation", "baseline_raw": 0.9909, "ceiling_raw": 0.9926},
    {"task_id": "spd-exam-0", "tier": "exam", "baseline_raw": 1.0, "ceiling_raw": 30.0},
    {"task_id": "tab-exam-0", "tier": "exam", "baseline_raw": 0.80, "ceiling_raw": 0.8025},
    {"task_id": "tab-exam-small", "tier": "exam", "baseline_raw": 0.70, "ceiling_raw": 0.7004},
]


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def split_buckets(total: int) -> dict:
    inputs, cached, output = total // 2, total // 10, total // 5
    return {"input": inputs, "cached_input": cached, "output": output,
            "reasoning": total - inputs - cached - output}


def add_buckets(*parts: dict) -> dict:
    return {key: sum(part[key] for part in parts)
            for key in ("input", "cached_input", "output", "reasoning")}


def snapshot(root: Path) -> dict:
    return {str(path): path.stat().st_mtime_ns for path in root.rglob("*")}


class Fixture:
    """Builds a synthetic formal project tree (plan, runs, reports, bank)."""

    def __init__(self, root: Path, *, admitted: bool = True, tweak=None, with_raw: bool = True):
        self.root = root.resolve()
        self.admitted = admitted
        self.tweak = tweak
        self.with_raw = with_raw
        self.batch_dir = self.root / "runs" / "batch-synthetic"
        self.counter = 0
        self.reports: dict[tuple, dict] = {}

    def write_bank(self) -> None:
        write_json(self.root / "tasks" / "bank.json",
                   {"schema": "bank/1", "version": "bank-v1", "frozen": True, "tasks": BANK_TASKS})

    def write_plan(self) -> Path:
        bank = self.root / "tasks" / "bank.json"
        path = self.batch_dir / "plan.json"
        write_json(path, {
            "schema": "batch_plan/1", "label": "FORMAL", "bank_version": "bank-v1",
            "tasks": TASKS, "configs": list(CONFIGS), "reps": [1, 2, 3],
            "run_validation": True, "run_exam": True, "max_workers": 8,
            "sealed_inputs": {str(bank): hashlib.sha256(bank.read_bytes()).hexdigest()},
        })
        return path

    def task_row(self, phase: str, config: str, rep: int, index: int, task_id: str) -> dict:
        score = BASE_SCORE[config] + 0.01 * rep + 0.001 * index
        if phase == "validation" and config == "tuned_lib" and not self.admitted:
            score -= 0.2
        row = {"score": score, "tokens": BASE_TOKENS[config] + 100 * rep + 10 * index,
               "failed": 0, "stop": "ceiling",
               "raw": 0.9 + 0.01 * index + 0.0001 * rep if self.with_raw else None}
        if self.tweak is not None:
            self.tweak(phase, config, rep, task_id, row)
        return row

    def write_task(self, run_id: str, task_id: str, row: dict, *, finished: bool = True) -> dict | None:
        task_dir = self.root / "runs" / run_id / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        failed = row["failed"]
        main_tokens = row["tokens"] - 10 * failed
        main = split_buckets(main_tokens)
        bad = {"input": 10, "cached_input": 0, "output": 0, "reasoning": 0}
        events = [{"kind": "task_start", "budget": 40_000},
                  {"kind": "llm_call", "role": "planner", "model": MODEL, "tokens": main,
                   "cost": float(main_tokens), "ok": True, "error": None}]
        events += [{"kind": "llm_call", "role": "executor", "model": MODEL, "tokens": bad,
                    "cost": 10.0, "ok": False, "error": "bad json"} for _ in range(failed)]
        if not finished:
            events = events[:2]
        else:
            events += [{"kind": "stop", "reason": row["stop"], "best_node": "n0001", "dev_score": 1.0},
                       {"kind": "task_end", "holdout_score": row["score"], "total_cost": float(row["tokens"])}]
        lines = [json.dumps({"schema": "ledger/1", "seq": seq, "run_id": run_id, "task_id": task_id, **event})
                 for seq, event in enumerate(events, 1)]
        (task_dir / "ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not finished:
            with (task_dir / "ledger.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"schema": "ledger/1", "seq": 9, "kind": "llm_call", "cos')  # live half line
            return None
        holdout = {"schema": "score/1", "split": "holdout", "score": row["score"], "ok": True}
        if row["raw"] is not None:
            holdout["raw"] = row["raw"]
        write_json(task_dir / "holdout_score.json", holdout)
        index = int(row.get("index", 0))
        return {"schema": "task_report/1", "run_id": run_id, "task_id": task_id,
                "holdout_score": row["score"], "total_cost": float(row["tokens"]), "currency": "TOKENS",
                "seconds": {"llm": 10.0, "run": 2.0, "queue": 0.5, "wall": 20.0 + index},
                "model_calls": 1 + failed, "tokens": add_buckets(main, *([bad] * failed))}

    def write_run(self, phase: str, config: str, rep: int, *, started: float, finished: bool = True,
                  with_report: bool = True, done_tasks: int | None = None, tasks_of: str | None = None) -> dict | None:
        self.counter += 1
        run_id = f"run-20261114-130000-{self.counter:04d}"
        ids = TASKS[tasks_of or phase]
        write_json(self.root / "runs" / run_id / "manifest.json", {
            "schema": "run/1", "run_id": run_id, "phase": phase, "config": config, "rep": rep,
            "bank_version": "bank-v1", "tasks": ids, "started_ts": iso(started),
            "ended_ts": iso(started + 300) if finished else None})
        reports = []
        for index, task_id in enumerate(ids):
            if done_tasks is not None and index > done_tasks:
                break
            row = self.task_row(tasks_of or phase, config, rep, index, task_id)
            row["index"] = index
            if done_tasks is not None and index == done_tasks:
                self.write_task(run_id, task_id, row, finished=False)
                break
            reports.append(self.write_task(run_id, task_id, row))
        if not with_report:
            return None
        report = {"schema": "run_report/1", "run_id": run_id, "phase": phase, "config": config,
                  "rep": rep, "currency": "TOKENS", "exact_token_accounting": True,
                  "total_cost": sum(task["total_cost"] for task in reports), "tasks": reports,
                  "evidence": {"run_end_seq": 2}}
        write_json(self.root / "runs" / run_id / "report.json", report)
        self.reports[(phase, config, rep)] = report
        return report

    def complete_batch(self) -> "Fixture":
        """Practice, then validation, then an exam sized by the real admission result."""
        self.write_bank()
        self.write_plan()
        rows, reports = [], {tier: [] for tier in TASKS}
        clock = [time.time() - 7200]

        def run_tier(tier: str, members: tuple) -> None:
            for rep in (1, 2, 3):
                for config in members:
                    report = self.write_run(tier, config, rep, started=clock[0])
                    clock[0] += 10
                    reports[tier].append(report)
                    rows.append({"logical_phase": tier, "manifest_phase": tier, "config": config, "rep": rep,
                                 "run_id": report["run_id"], "test_only": False,
                                 "report_path": str(self.root / "runs" / report["run_id"] / "report.json")})

        run_tier("practice", CONFIGS)
        run_tier("validation", ("default_lib", "tuned_lib"))
        self.admission = admit_configuration(reports["validation"], baseline="default_lib", candidate="tuned_lib")
        run_tier("exam", CONFIGS if self.admission["claim"] == "admitted" else CONFIGS[:3])
        write_json(self.batch_dir / "batch_report.json", {
            "schema": "batch_report/1", "label": "FORMAL", "bank_version": "bank-v1",
            "runs": rows, "reports": reports, "admission": self.admission,
            "tuned_exam_skipped": self.admission["claim"] != "admitted"})
        return self


def build(root: Path, **kwargs) -> Fixture:
    return Fixture(root / "project", **kwargs).complete_batch()


class SummarizeFormalBatchTests(unittest.TestCase):
    def test_admitted_batch_has_twelve_exam_runs_and_exact_numbers(self):
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            before = snapshot(fx.root)
            summary = summarize(fx.batch_dir, research_cost=2_094_728)
            markdown = render_markdown(summary)
            self.assertEqual(before, snapshot(fx.root))  # strictly read-only
            self.assertEqual(summary["mode"], "complete")
            self.assertEqual(summary["progress"]["completed_runs"], {"practice": 12, "validation": 6, "exam": 12})
            self.assertEqual(summary["warnings"], [])
            self.assertTrue(summary["exact_token_accounting"])
            self.assertTrue(summary["sources"]["bank_sealed_match"])
            run = next(item for item in summary["runs"] if (item["phase"], item["config"], item["rep"]) == ("exam", "default_lib", 2))
            self.assertEqual(run["tokens_total"], 6630)  # 3*(2000+200) + 10*(0+1+2)
            self.assertEqual(sum(run["buckets"].values()), 6630)
            self.assertEqual(run["buckets"]["input"], sum(split_buckets(t)["input"] for t in (2200, 2210, 2220)))
            self.assertAlmostEqual(run["mean_score"], 0.321)
            self.assertEqual(run["seconds"], {"llm": 30.0, "run": 6.0, "queue": 1.5, "wall": 63.0})
            self.assertEqual(run["run_elapsed_seconds"], 300.0)
            self.assertEqual((run["model_calls"], run["failed_calls"]), (3, 0))
            self.assertEqual(run["stop_reasons"], {"ceiling": 3})
            self.assertEqual((run["max_task_tokens"], run["max_task_id"]), (2220.0, "tab-exam-small"))
            self.assertEqual(run["over_limit_tasks"], 0)
            config = summary["config_summary"]["exam"]["default_lib"]
            self.assertEqual((config["tokens_sum"], config["tokens_mean"]), (19890, 6630))
            # Validation admission is quoted verbatim and pairs are listed.
            batch_report = json.loads((fx.batch_dir / "batch_report.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["validation"]["admission"], batch_report["admission"])
            self.assertEqual(summary["validation"]["admission"]["claim"], "admitted")
            self.assertTrue(summary["validation"]["admission_recomputed_matches"])
            self.assertEqual([pair["rep"] for pair in summary["validation"]["pairs"]], [1, 2, 3])
            self.assertAlmostEqual(summary["validation"]["pairs"][0]["score_difference"], 0.1)
            self.assertEqual(summary["validation"]["pairs"][0]["token_difference"], -1500.0)
            # Three-repeat comparisons and direction flags.
            comparisons = summary["exam"]["comparisons"]
            self.assertTrue(all(item["available"] for item in comparisons.values()))
            tuned = comparisons["tuned_lib_vs_default_lib"]
            self.assertEqual((tuned["baseline"], tuned["candidate"]), ("default_lib", "tuned_lib"))
            self.assertEqual(tuned["claim"], "better")
            self.assertEqual(tuned["cost_differences"], [-1500.0, -1500.0, -1500.0])
            self.assertEqual((tuned["baseline_mean_cost"], tuned["candidate_mean_cost"]), (6630.0, 5130.0))
            self.assertEqual((tuned["score_direction"], tuned["cost_direction"]), ("all_higher", "all_lower"))
            self.assertTrue(tuned["cost_direction_consistent"])
            naive = comparisons["default_vs_naive"]
            self.assertEqual((naive["baseline"], naive["candidate"]), ("naive", "default"))
            self.assertAlmostEqual(naive["mean_cost_change_pct"], (9630 / 12630 - 1) * 100)
            self.assertEqual(comparisons["default_lib_vs_default"]["cost_differences"], [-3000.0] * 3)
            # Break-even: ceil(2,094,728 / ((6630 - 5130) / 3)) = ceil(4189.456).
            breakeven = summary["breakeven"]
            self.assertTrue(breakeven["computed"])
            self.assertEqual(breakeven["savings_per_task_tokens"], 500.0)
            self.assertEqual(breakeven["breakeven_tasks"], 4190)
            for expected in ("内部 TOKENS", "不是现金价格，也不是官方比赛成绩", "### exam（12 个 run",
                             "default_lib vs default", "tuned_lib vs default_lib", "数学回本点 4190 题", "admission 原文"):
                self.assertIn(expected, markdown)
            self.assertNotIn("现金节省已", markdown)

    def test_rejected_batch_has_nine_exam_runs_and_skips_tuned(self):
        with TemporaryDirectory() as temp:
            fx = build(Path(temp), admitted=False)
            summary = summarize(fx.batch_dir, research_cost=2_094_728)
            self.assertEqual(summary["progress"]["completed_runs"], {"practice": 12, "validation": 6, "exam": 9})
            self.assertEqual(summary["validation"]["admission"]["claim"], "rejected")
            self.assertTrue(summary["exam"]["tuned_exam_skipped"])
            comparisons = summary["exam"]["comparisons"]
            self.assertTrue(comparisons["default_vs_naive"]["available"])
            self.assertTrue(comparisons["default_lib_vs_default"]["available"])
            self.assertFalse(comparisons["tuned_lib_vs_default_lib"]["available"])
            self.assertFalse(any(run["config"] == "tuned_lib" and run["phase"] == "exam" for run in summary["runs"]))
            self.assertFalse(summary["breakeven"]["computed"])
            self.assertIn("未通过 validation 准入", " ".join(summary["breakeven"]["reasons"]))
            markdown = render_markdown(summary)
            self.assertIn("### exam（9 个 run", markdown)
            self.assertIn("**不计算。**", markdown)

    def test_failed_model_calls_and_stop_reasons_are_counted(self):
        def tweak(phase, config, rep, task_id, row):
            if (phase, config, rep, task_id) == ("exam", "default_lib", 2, "tab-exam-0"):
                row.update(failed=2, stop="error")
            if (phase, config, rep, task_id) == ("exam", "default_lib", 2, "spd-exam-0"):
                row.update(stop="budget_cap")

        with TemporaryDirectory() as temp:
            fx = build(Path(temp), tweak=tweak)
            summary = summarize(fx.batch_dir)
            run = next(item for item in summary["runs"] if (item["phase"], item["config"], item["rep"]) == ("exam", "default_lib", 2))
            self.assertEqual(run["failed_calls"], 2)
            self.assertEqual(run["model_calls"], 5)  # 3 normal calls + 2 failed calls
            self.assertEqual(run["tokens_total"], 6630)  # failed calls stay inside the task token total
            self.assertEqual(run["stop_reasons"], {"budget_cap": 1, "ceiling": 1, "error": 1})
            other = next(item for item in summary["runs"] if (item["phase"], item["config"], item["rep"]) == ("exam", "default_lib", 1))
            self.assertEqual((other["failed_calls"], other["model_calls"]), (0, 3))
            self.assertEqual(summary["warnings"], [])
            row = next(item for item in summary["per_task"]["exam"] if (item["task_id"], item["config"]) == ("tab-exam-0", "default_lib"))
            self.assertEqual([rep["failed_calls"] for rep in row["reps"]], [0, 2, 0])
            markdown = render_markdown(summary)
            self.assertIn("预算闸门 1、达到评分上限 1、错误 1", markdown)
            self.assertIn("失败调用 2", markdown)

    def test_small_reference_gap_is_flagged_and_diagnostic_view_leaves_formal_untouched(self):
        def tweak(phase, config, rep, task_id, row):
            # Only the tiny-reference-gap task makes tuned_lib look worse in validation rep 2.
            if (phase, config, rep, task_id) == ("validation", "tuned_lib", 2, "tab-validation-small"):
                row["score"] = -5.0

        with TemporaryDirectory() as temp:
            fx = build(Path(temp), tweak=tweak)
            summary = summarize(fx.batch_dir, small_gap=0.002)
            flagged = {item["task_id"] for item in summary["reference_gaps"] if item["small_gap"]}
            self.assertEqual(flagged, {"tab-validation-small", "tab-exam-small"})
            by_task = {item["task_id"]: item for item in summary["reference_gaps"]}
            self.assertAlmostEqual(by_task["tab-validation-small"]["gap"], 0.0017)
            self.assertFalse(by_task["tab-exam-0"]["small_gap"])  # gap 0.0025, not below 0.002
            self.assertNotIn("spd-exam-0", by_task)  # speedup tasks are not flagged
            tagged = [row["task_id"] for row in summary["per_task"]["exam"] if row["small_reference_gap"]]
            self.assertEqual(sorted(set(tagged)), ["tab-exam-small"])
            # Formal admission stays "rejected": rep 2 of tuned_lib is lower.
            formal = summary["validation"]["admission"]
            self.assertEqual(formal["claim"], "rejected")
            self.assertFalse(formal["score_not_lower_each_rep"])
            diagnostic = summary["diagnostic_excluding_small_gap"]
            self.assertEqual(diagnostic["note"], "仅诊断，不改正式成绩")
            validation = diagnostic["phases"]["validation"]
            self.assertEqual(validation["excluded_in_phase"], ["tab-validation-small"])
            self.assertEqual(validation["remaining_tasks"], ["spd-validation-0", "tab-validation-0"])
            self.assertEqual(validation["admission_replay"]["claim"], "admitted")  # diagnostic replay only
            default_lib = validation["config_summary"]["default_lib"]
            self.assertAlmostEqual(default_lib["scores"][0], 0.3 + 0.01 + 0.0005)
            self.assertEqual(default_lib["tokens_by_rep"], [4010.0 + 200 * 1, 4010.0 + 200 * 2, 4010.0 + 200 * 3])
            formal_default_lib = summary["config_summary"]["validation"]["default_lib"]
            self.assertAlmostEqual(formal_default_lib["scores"][0], 0.3 + 0.01 + 0.001)  # formal figures unchanged
            self.assertEqual(formal_default_lib["tokens_by_rep"], [6330.0, 6630.0, 6930.0])
            exam = diagnostic["phases"]["exam"]
            self.assertEqual(exam["excluded_in_phase"], ["tab-exam-small"])
            self.assertEqual(exam["comparisons"]["default_vs_naive"]["claim"], "better")
            self.assertNotIn("practice", diagnostic["phases"])  # no flagged task in practice
            markdown = render_markdown(summary)
            self.assertIn("仅诊断，不改正式成绩", markdown)
            self.assertIn("小参考差 0.000400（仅标记）", markdown)
            self.assertIn("### validation：剔除 tab-validation-small", markdown)

    def test_incomplete_batch_errors_without_partial_and_writes_nothing(self):
        with TemporaryDirectory() as temp:
            fx = Fixture(Path(temp) / "project")
            fx.write_bank()
            fx.write_plan()
            with self.assertRaisesRegex(SummaryError, "batch_report.json"):
                summarize(fx.batch_dir)
            out = Path(temp) / "out"
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = main(["--batch-dir", str(fx.batch_dir), "--output-dir", str(out)])
            self.assertEqual(code, 1)
            self.assertIn("批次尚未完成", stderr.getvalue())
            self.assertFalse(out.exists())

    def test_breakeven_is_not_computed_when_exam_condition_fails(self):
        cases = {
            "score_drop": (lambda phase, config, rep, task_id, row: row.update(score=row["score"] - 0.5)
                           if (phase, config, rep) == ("exam", "tuned_lib", 2) else None, "分数下降"),
            "cost_not_lower": (lambda phase, config, rep, task_id, row: row.update(tokens=row["tokens"] + 5000)
                               if (phase, config) == ("exam", "tuned_lib") else None, "平均 token 不低于"),
        }
        for name, (tweak, reason) in cases.items():
            with self.subTest(name), TemporaryDirectory() as temp:
                fx = build(Path(temp), tweak=tweak)
                summary = summarize(fx.batch_dir, research_cost=2_094_728)
                self.assertEqual(summary["validation"]["admission"]["claim"], "admitted")
                self.assertTrue(summary["exam"]["comparisons"]["tuned_lib_vs_default_lib"]["available"])
                self.assertFalse(summary["breakeven"]["computed"])
                self.assertEqual(summary["breakeven"]["display"], "不计算")
                self.assertIn(reason, " ".join(summary["breakeven"]["reasons"]))
                markdown = render_markdown(summary)
                self.assertIn("**不计算。**", markdown)
                self.assertNotIn("数学回本点", markdown)
        with TemporaryDirectory() as temp:  # all conditions hold but no research cost was supplied
            summary = summarize(build(Path(temp)).batch_dir)
            self.assertFalse(summary["breakeven"]["computed"])
            self.assertIn("--research-cost", " ".join(summary["breakeven"]["reasons"]))

    def test_partial_mode_lists_only_finished_runs_after_plan(self):
        with TemporaryDirectory() as temp:
            fx = Fixture(Path(temp) / "project")
            fx.write_bank()
            plan = fx.write_plan()
            now = time.time()
            os.utime(plan, (now - 3600, now - 3600))
            done = fx.write_run("practice", "naive", 1, started=now - 1800)
            fx.write_run("practice", "default", 1, started=now - 1700, finished=False,
                         with_report=False, done_tasks=1)  # task 0 done, task 1 mid-write
            (fx.root / "runs" / "run-20261114-130000-9999").mkdir()  # reserved, no manifest yet
            old = fx.write_run("practice", "naive", 2, started=now - 7200)  # before the plan
            research = fx.write_run("self_research", "default_lib", 1, started=now - 1600, tasks_of="practice")
            before = snapshot(fx.root)
            with self.assertRaisesRegex(SummaryError, "batch_report.json"):
                summarize(fx.batch_dir)
            summary = summarize(fx.batch_dir, partial=True)
            markdown = render_markdown(summary)
            self.assertEqual(before, snapshot(fx.root))
            self.assertEqual(summary["mode"], "partial")
            self.assertFalse(summary["complete"])
            self.assertEqual([run["run_id"] for run in summary["runs"]], [done["run_id"]])
            self.assertNotIn(old["run_id"], markdown)
            self.assertNotIn(research["run_id"], markdown)
            self.assertEqual(summary["progress"]["completed_runs"], {"practice": 1, "validation": 0, "exam": 0})
            running = summary["progress"]["running"]
            self.assertEqual(len(running), 1)
            self.assertEqual((running[0]["config"], running[0]["tasks_done"], running[0]["tasks_total"]),
                             ("default", 1, 2))
            self.assertEqual(running[0]["current_task"], "tab-practice-0")
            self.assertEqual(running[0]["tokens_so_far"], 3100 + 3110)  # finished task + live task
            self.assertEqual(summary["progress"]["reserved_without_manifest"], ["run-20261114-130000-9999"])
            self.assertEqual(summary["progress"]["expected_runs"]["practice"], 12)
            self.assertFalse(summary["breakeven"]["computed"])
            self.assertIn("尚未完成", " ".join(summary["breakeven"]["reasons"]))
            self.assertIn("进行中", markdown)
            self.assertIn("排队等并发槽位", markdown)
            self.assertIsNone(summary["validation"]["admission"])

    def test_partial_mode_warns_instead_of_failing_on_missing_task_files(self):
        with TemporaryDirectory() as temp:
            fx = Fixture(Path(temp) / "project")
            fx.write_bank()
            plan = fx.write_plan()
            now = time.time()
            os.utime(plan, (now - 3600, now - 3600))
            report = fx.write_run("practice", "naive", 1, started=now - 1800)
            task_dir = fx.root / "runs" / report["run_id"] / "tasks" / "spd-practice-0"
            (task_dir / "ledger.jsonl").unlink()
            (fx.root / "runs" / report["run_id"] / "tasks" / "tab-practice-0" / "holdout_score.json").unlink()
            summary = summarize(fx.batch_dir, partial=True)
            run = summary["runs"][0]
            self.assertIsNone(run["failed_calls"])  # one task ledger is missing, so it is unknown
            self.assertEqual(run["stop_reasons"], {"ceiling": 1})
            text = " ".join(summary["warnings"])
            self.assertIn("缺少 ledger.jsonl", text)
            self.assertIn("缺少 holdout_score.json", text)
            markdown = render_markdown(summary)  # unknown counts render as a dash instead of crashing
            self.assertIn("缺少 ledger.jsonl", markdown.split("## 7. 数据提醒")[1])

    def test_partial_mode_without_any_finished_run_is_not_a_failure(self):
        with TemporaryDirectory() as temp:
            fx = Fixture(Path(temp) / "project")
            fx.write_bank()
            plan = fx.write_plan()
            now = time.time()
            os.utime(plan, (now - 3600, now - 3600))
            fx.write_run("practice", "naive", 1, started=now - 1800, finished=False, with_report=False, done_tasks=0)
            out = Path(temp) / "out"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--batch-dir", str(fx.batch_dir), "--output-dir", str(out), "--partial"])
            self.assertEqual(code, 0)
            self.assertIn("暂无可汇总 run", stdout.getvalue())
            summary = json.loads((out / "formal-batch-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["runs"], [])
            self.assertEqual(len(summary["progress"]["running"]), 1)
            self.assertIn("暂无可汇总 run", (out / "formal-batch-summary.md").read_text(encoding="utf-8"))

    def test_partial_flag_on_completed_batch_uses_full_summary(self):
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            summary = summarize(fx.batch_dir, partial=True)
            self.assertEqual(summary["mode"], "complete")
            self.assertEqual(summary["progress"]["completed_runs"]["exam"], 12)
            self.assertTrue(any("--partial 不起作用" in note for note in summary["notes"]))

    def test_raw_metric_is_quoted_when_saved_and_marked_missing_otherwise(self):
        def tweak(phase, config, rep, task_id, row):
            if (phase, config, rep, task_id) == ("exam", "default", 3, "tab-exam-0"):
                row["raw"] = None

        with TemporaryDirectory() as temp:
            fx = build(Path(temp), tweak=tweak)
            summary = summarize(fx.batch_dir)
            row = next(item for item in summary["per_task"]["exam"] if (item["task_id"], item["config"]) == ("tab-exam-0", "default"))
            self.assertEqual([rep["raw_saved"] for rep in row["reps"]], [True, True, False])
            self.assertAlmostEqual(row["reps"][0]["raw"], 0.9 + 0.01 + 0.0001)
            markdown = render_markdown(summary)
            self.assertIn("0.9101 / 0.9102 / 未保存", markdown)
        with TemporaryDirectory() as temp:
            fx = build(Path(temp), with_raw=False)
            markdown = render_markdown(summarize(fx.batch_dir))
            self.assertIn("未保存 / 未保存 / 未保存", markdown)

    def test_cli_writes_outputs_outside_project_and_refuses_inside(self):
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            before = snapshot(fx.root)
            out = Path(temp) / "out"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--batch-dir", str(fx.batch_dir), "--output-dir", str(out),
                             "--name", "formal-batch-v2-summary", "--research-cost", "2,094,728"])
            self.assertEqual(code, 0)
            self.assertEqual(before, snapshot(fx.root))
            written = json.loads((out / "formal-batch-v2-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], "formal_batch_summary/1")
            self.assertEqual(written["breakeven"]["breakeven_tasks"], 4190)
            self.assertTrue((out / "formal-batch-v2-summary.md").is_file())
            self.assertIn("formal-batch-v2-summary.json", stdout.getvalue())
            inside = fx.root / "docs"
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = main(["--batch-dir", str(fx.batch_dir), "--output-dir", str(inside)])
            self.assertEqual(code, 1)
            self.assertIn("拒绝写入", stderr.getvalue())
            self.assertFalse(inside.exists())
            summary = summarize(fx.batch_dir)
            with self.assertRaises(SummaryError):
                write_outputs(summary, fx.batch_dir, "x")

    def test_missing_evidence_and_tampered_saved_report_fail_closed(self):
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            report = fx.reports[("exam", "default", 1)]
            ledger = fx.root / "runs" / report["run_id"] / "tasks" / "spd-exam-0" / "ledger.jsonl"
            ledger.unlink()
            with self.assertRaisesRegex(SummaryError, "缺少任务账本"):
                summarize(fx.batch_dir)
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            report = fx.reports[("exam", "default", 1)]
            path = fx.root / "runs" / report["run_id"] / "report.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["tasks"][0]["holdout_score"] += 0.5
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(SummaryError, "report.json 缺失或与 batch_report.json"):
                summarize(fx.batch_dir)

    def test_validation_score_or_task_ledger_tampering_fails_closed(self):
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            path = fx.batch_dir / "batch_report.json"
            batch = json.loads(path.read_text(encoding="utf-8"))
            tuned = next(report for report in batch["reports"]["validation"]
                         if report["config"] == "tuned_lib" and report["rep"] == 1)
            tuned["tasks"][0]["holdout_score"] -= 1.0
            write_json(path, batch)
            with self.assertRaisesRegex(SummaryError, "内嵌报告不一致"):
                summarize(fx.batch_dir, research_cost=2_094_728)
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            path = fx.batch_dir / "batch_report.json"
            batch = json.loads(path.read_text(encoding="utf-8"))
            batch["admission"]["comparison"]["candidate_scores"][0] += 1.0
            write_json(path, batch)
            with self.assertRaisesRegex(SummaryError, "admission 与按已保存 validation 报告重算"):
                summarize(fx.batch_dir, research_cost=2_094_728)
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            report = fx.reports[("exam", "default", 1)]
            ledger = fx.root / "runs" / report["run_id"] / "tasks" / "spd-exam-0" / "ledger.jsonl"
            rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
            rows[1]["tokens"]["input"] += 1
            ledger.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(SummaryError, "账本调用成本或四桶不一致"):
                summarize(fx.batch_dir)

    def test_partial_with_validation_reports_does_not_claim_admission_or_payback(self):
        with TemporaryDirectory() as temp:
            fx = build(Path(temp))
            plan = fx.batch_dir / "plan.json"
            old = time.time() - 10_000
            os.utime(plan, (old, old))
            (fx.batch_dir / "batch_report.json").unlink()
            summary = summarize(fx.batch_dir, partial=True, research_cost=2_094_728)
            markdown = render_markdown(summary)
            self.assertEqual(summary["progress"]["completed_runs"]["validation"], 6)
            self.assertIsNone(summary["validation"]["admission"])
            self.assertFalse(summary["breakeven"]["computed"])
            self.assertNotIn("admission_replay",
                             summary["diagnostic_excluding_small_gap"]["phases"]["validation"])
            self.assertNotIn("判定：**admitted**", markdown)
            self.assertNotIn("数学回本点 4190 题", markdown)

    def test_cost_argument_accepts_thousands_separators(self):
        self.assertEqual(_cost_argument("2,094,728"), 2094728.0)
        self.assertEqual(_cost_argument("2094728"), 2094728.0)
        self.assertTrue(math.isfinite(_cost_argument("0")))
        with self.assertRaises(Exception):
            _cost_argument("-1")


if __name__ == "__main__":
    unittest.main()
