from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.admission import admit_configuration
from evaluation.figures import CONFIGS, build_study, build_study_from_batch, render_html


def sample_reports():
    result = []
    for config_index, config in enumerate(CONFIGS):
        for rep in (1, 2, 3):
            score = .4 + config_index * .15 + rep * .001
            cost = (4.0, 3.0, 2.5, 2.0)[config_index]
            result.append({
                "schema": "run_report/1", "run_id": f"run-{config}-{rep}",
                "phase": "dev", "currency": "TEST", "config": config, "rep": rep,
                "total_cost": cost, "tasks": [{
                    "task_id": "spd-example", "holdout_score": score,
                    "total_cost": cost, "seconds": {
                        "llm": 1.0, "run": 2.0, "queue": 3.0, "wall": 6.0},
                    "evidence": {"task_end_seq": 7},
                }],
            })
    return result


class StudyFiguresTests(unittest.TestCase):
    def test_test_only_report_and_break_even(self):
        study = build_study(sample_reports(), research_reports=[{
            "schema": "run_report/1", "run_id": "self-1", "phase": "dev",
            "currency": "TEST", "total_cost": 1.1,
        }])
        self.assertTrue(study["test_only"])
        self.assertEqual(study["breakeven"]["tasks"], 3)
        self.assertEqual(len(study["task_rows"]), 12)
        html = render_html(study)
        self.assertIn("TEST 联调数据", html)
        self.assertIn("ledger#7", html)
        self.assertIn("<svg", html)

    def test_reject_incomplete_and_cross_phase_data(self):
        reports = sample_reports()
        with self.assertRaises(ValueError):
            build_study(reports[:-1])
        changed = deepcopy(reports)
        changed[0]["phase"] = "exam"
        with self.assertRaises(ValueError):
            build_study(changed)
        changed = deepcopy(reports)
        changed[0]["tasks"][0]["task_id"] = "spd-other"
        with self.assertRaises(ValueError):
            build_study(changed)

    def test_official_exam_rejects_test_prices(self):
        reports = sample_reports()
        for report in reports:
            report["phase"] = "exam"
        with self.assertRaises(ValueError):
            build_study(reports)

    def test_token_study_report_avoids_cash_and_official_claims(self):
        reports = sample_reports()
        for report in reports:
            report["phase"] = "exam"
            report["currency"] = "TOKENS"
        admission = {"claim": "admitted", "baseline": "default_lib", "candidate": "tuned_lib"}
        study = build_study(reports, admission=admission)
        html = render_html(study)
        self.assertIn("自选 token 预算的内部冻结题库研究", html)
        self.assertIn("非现金节省、非官方比赛成绩", html)
        self.assertIn("平均绝对 token 数误差", html)
        self.assertNotIn("正式考试数据", html)
        self.assertNotIn("金额为每遍", html)
        for report in reports:
            report["currency"] = "CNY"
        cash_html = render_html(build_study(reports, admission=admission))
        self.assertIn("正式考试数据", cash_html)
        self.assertIn("研究自己的成本尚无审计报告", cash_html)
        self.assertNotIn("内部冻结题库研究", cash_html)

    def test_rejected_candidate_has_honest_three_line_report(self):
        reports = [report for report in sample_reports() if report["config"] != "tuned_lib"]
        admission = {"claim": "rejected", "baseline": "default_lib",
                     "candidate": "tuned_lib"}
        study = build_study(reports, research_total_cost=4.0, admission=admission)
        self.assertEqual(len(study["configurations"]), 3)
        self.assertIsNone(study["breakeven"]["savings_per_task"])
        self.assertIsNone(study["breakeven"]["tasks"])
        self.assertFalse(study["breakeven"]["eligible"])
        html = render_html(study)
        self.assertIn("未进入考试", html)
        self.assertIn("已审计研究成本 4.0000 TEST", html)
        self.assertNotIn("研究版／练过的库</text>", html)

    def test_batch_membership_and_sealed_source_are_checked(self):
        with TemporaryDirectory() as temp:
            project = Path(temp)
            batch_dir = project / "batches/first"
            batch_dir.mkdir(parents=True)
            source = project / "config.json"
            source.write_text("stable", encoding="utf-8")
            reports = sample_reports()
            validation = []
            for report in reports:
                if report["config"] in {"default_lib", "tuned_lib"}:
                    copy = deepcopy(report)
                    copy["run_id"] = "validation-" + report["run_id"]
                    validation.append(copy)
            admission = admit_configuration(validation, baseline="default_lib",
                                            candidate="tuned_lib")
            self.assertEqual(admission["claim"], "admitted")
            validation_runs = [{"logical_phase": "validation", "run_id": report["run_id"],
                                "config": report["config"], "rep": report["rep"],
                                "report_path": str(project / "runs" / report["run_id"] / "report.json")}
                               for report in validation]
            runs = [{"logical_phase": "exam", "run_id": report["run_id"],
                     "config": report["config"], "rep": report["rep"],
                     "report_path": str(project / "runs" / report["run_id"] / "report.json")}
                    for report in reports]
            plan = {"schema": "batch_plan/1", "label": "TEST ONLY", "run_exam": True,
                    "sealed_inputs": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
            batch = {"schema": "batch_report/1", "label": "TEST ONLY",
                     "runs": validation_runs + runs,
                     "tuned_exam_skipped": False, "admission": admission}
            (batch_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (batch_dir / "batch_report.json").write_text(json.dumps(batch), encoding="utf-8")
            by_id = {report["run_id"]: report for report in reports + validation}
            with patch("evaluation.figures.run_report", side_effect=lambda path: by_id[path.name]):
                self.assertTrue(build_study_from_batch(batch_dir)["test_only"])
                batch["admission"]["comparison"]["candidate_mean_cost"] = 999
                (batch_dir / "batch_report.json").write_text(json.dumps(batch), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "admission differs"):
                    build_study_from_batch(batch_dir)
                batch["admission"] = admission
                batch["admission"]["comparison"]["candidate_mean_cost"] = 2.0
                (batch_dir / "batch_report.json").write_text(json.dumps(batch), encoding="utf-8")
                source.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source file changed"):
                    build_study_from_batch(batch_dir)


if __name__ == "__main__":
    unittest.main()
