"""Regression checks for the hourly roadshow report generator."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.generate_roadshow_report import SUMMARY_PATH, generate
from scripts.summarize_formal_batch import summarize
from tests.test_summarize_formal_batch import build


class RoadshowReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.output = self.repo / "docs" / "路演汇报.html"
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Report Test")
        self.run_git("config", "user.email", "report@example.test")
        (self.repo / "README.md").write_text("# Demo\n\n初始状态。\n", encoding="utf-8")
        self.run_git("add", "README.md")
        self.run_git("commit", "-qm", "Initial source")

    def run_git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True, text=True).stdout.strip()

    def test_report_only_commit_does_not_regenerate(self):
        self.assertTrue(generate(self.repo, self.output))
        initial = self.output.read_text(encoding="utf-8")
        self.run_git("add", "docs/路演汇报.html")
        self.run_git("commit", "-qm", "Update hourly roadshow report")
        self.assertFalse(generate(self.repo, self.output))
        self.assertEqual(initial, self.output.read_text(encoding="utf-8"))

        (self.repo / "README.md").write_text("# Demo\n\n更新后的状态。\n", encoding="utf-8")
        self.run_git("add", "README.md")
        self.run_git("commit", "-qm", "Describe updated source")
        self.assertTrue(generate(self.repo, self.output))
        refreshed = self.output.read_text(encoding="utf-8")
        self.assertIn("更新后的状态", refreshed)
        self.assertIn("Describe updated source", refreshed)
        self.assertIn("<span class=\"metric\">1</span><span class=\"label\">新增相关提交", refreshed)

    def test_commit_subject_is_html_escaped(self):
        (self.repo / "README.md").write_text("# Demo\n\n安全内容。\n", encoding="utf-8")
        self.run_git("add", "README.md")
        self.run_git("commit", "-qm", "Add <script> & evidence")
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        self.assertIn("Add &lt;script&gt; &amp; evidence", report)
        self.assertNotIn("<strong>Add <script>", report)

    def test_cover_keeps_description_after_tagline(self):
        (self.repo / "README.md").write_text(
            "# 自进化 Research Agent 系统\n\n让 Research Agent 研究自己。\n\n"
            "这是按《分工文档》实现的可审计研究系统。\n", encoding="utf-8"
        )
        self.run_git("add", "README.md")
        self.run_git("commit", "-qm", "Name the project")
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        self.assertIn(
            '<p class="small" style="max-width:1050px;font-size:18px">'
            '这是按《分工文档》实现的可审计研究系统。</p>', report
        )

    def write_calibration_sources(self, second=None):
        (self.repo / "README.md").write_text(
            "# Demo\n\n## 当前限制\n\nbank-v1 已冻结。正式四线比较尚未执行。\n\n"
            "## 冻结后下一步\n\n1. 在 practice 题上运行自动自研究。\n",
            encoding="utf-8",
        )
        (self.repo / "tasks").mkdir()
        (self.repo / "tasks" / "bank.json").write_text(
            json.dumps({"version": "bank-v1", "frozen": True,
                        "tasks": [{"task_id": str(i)} for i in range(24)]}),
            encoding="utf-8",
        )
        (self.repo / "docs").mkdir(exist_ok=True)
        (self.repo / "docs" / "提速朴素成本校准.md").write_text(
            "总计 36 次尝试：34 次正常收尾、2 次提前错误；"
            "实耗 531,929 token；holdout 尝试为 0。\n", encoding="utf-8",
        )
        if second is not None:
            (self.repo / "docs" / "表格朴素成本校准.md").write_text(
                second, encoding="utf-8",
            )
        self.run_git("add", "README.md", "tasks", "docs")
        self.run_git("commit", "-qm", "Commit calibration evidence")

    def test_frozen_calibration_appears_with_committed_evidence(self):
        self.write_calibration_sources(
            "合计 36 次尝试：35 次正常收尾、1 次提前错误；"
            "消耗 **1,041,097 token**；holdout 尝试 0；"
            "72/72 次尝试；ready=true；"
            "runs/naive-cost-audit-20261114-1041-agent/ 留有 audit.json。\n"
        )
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        for expected in (
            "题库已冻结 · 正式四线待运行", "bank-v1 已冻结", "24 题",
            "72 次 dev 尝试", "69 次正常完成", "3 次提前停止",
            "1,573,026 token", "校准 holdout 0",
            'href="提速朴素成本校准.md"', 'href="表格朴素成本校准.md"',
            "在 practice 题上运行自动自研究",
        ):
            self.assertIn(expected, report)

    def test_frozen_calibration_requires_complete_consistent_evidence(self):
        self.write_calibration_sources()
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        self.assertNotIn("题库已冻结 · 正式四线待运行", report)
        (self.repo / "docs" / "表格朴素成本校准.md").write_text(
            "合计 35 次尝试：35 次正常收尾、1 次提前错误；"
            "消耗 **1,041,097 token**；holdout 尝试 0；"
            "72/72 次尝试；ready=true。\n", encoding="utf-8",
        )
        self.run_git("add", "docs/表格朴素成本校准.md")
        self.run_git("commit", "-qm", "Add inconsistent calibration evidence")
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        self.assertNotIn("题库已冻结 · 正式四线待运行", report)
        self.assertNotIn("1,573,026 token", report)

    def test_internal_token_v2_status_survives_regeneration(self):
        self.write_calibration_sources(
            "合计 36 次尝试：35 次正常收尾、1 次提前错误；"
            "消耗 **1,041,097 token**；holdout 尝试 0；"
            "72/72 次尝试；ready=true。\n"
        )
        (self.repo / "README.md").write_text(
            "# Demo\n\n这是按《分工文档》实现的可审计研究系统。"
            "首次内部练习批次没有有效候选；修复后的 v2 正在独立工作区运行，"
            "验证与考试尚未启动。TOKENS 不是现金价格或官方成绩。\n\n"
            "## 验证\n\n主回归 203 项通过；技能层 6 项、看板 7 项也已通过。\n\n"
            "## 当前限制\n\nbank-v1 已冻结并完成 dev 校准。\n\n"
            "## 冻结后下一步\n\n1. 等待 v2 十题 practice 自研究。\n",
            encoding="utf-8",
        )
        (self.repo / "config").mkdir()
        (self.repo / "config/prices.json").write_text(
            json.dumps({"schema": "prices/1", "currency": "TOKENS",
                        "test_only": False}), encoding="utf-8",
        )
        self.run_git("add", "README.md", "config/prices.json")
        self.run_git("commit", "-qm", "Record internal token v2 study")
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        for expected in (
            "内部 token v2 练习研究进行中 · 验证／考试待真实账本",
            "首次内部练习批次没有有效候选；修复后的 v2 正在独立工作区运行，验证与考试尚未启动",
            "216", "TOKENS、test_only=false",
            "脚本角色 TEST 联调不是实际模型成本",
            "v2 自研究、验证与考试数字均待对应真实账本审计后填入",
        ):
            self.assertIn(expected, report)
        self.assertNotIn("TEST 单位对应报告的 token 用量", report)
        self.assertNotIn("阶段状态请以 README 为准", report)

    def test_no_internal_price_table_does_not_claim_token_tariff(self):
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        self.assertIn("TEST 联调不能代表真实模型成本、现金价格或官方成绩", report)
        self.assertNotIn("TOKENS、test_only=false", report)

    def test_completed_validation_is_not_mislabeled_pending(self):
        (self.repo / "README.md").write_text(
            "# Demo\n\n## 当前限制\n\n正式验证已完成，考试账本待审。\n",
            encoding="utf-8",
        )
        self.run_git("add", "README.md")
        self.run_git("commit", "-qm", "Record validation status")
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        self.assertIn("阶段状态请以 README 为准", report)
        self.assertNotIn("正式四线比较未执行", report)
        self.assertNotIn("验证与考试数字待对应真实账本审计后填入", report)

    V2_README = (
        "# Demo\n\n这是按《分工文档》实现的可审计研究系统。"
        "首次内部练习批次没有有效候选；修复后的 v2 正在独立工作区运行，"
        "验证与考试尚未启动。TOKENS 不是现金价格或官方成绩。\n\n"
        "## 当前限制\n\nbank-v1 已冻结并完成 dev 校准。\n"
    )

    def commit_v2_sources(self, summary_text: str | None):
        (self.repo / "README.md").write_text(self.V2_README, encoding="utf-8")
        (self.repo / "config").mkdir(exist_ok=True)
        (self.repo / "config/prices.json").write_text(
            json.dumps({"schema": "prices/1", "currency": "TOKENS", "test_only": False}), encoding="utf-8")
        paths = ["README.md", "config/prices.json"]
        if summary_text is not None:
            target = self.repo / SUMMARY_PATH
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(summary_text, encoding="utf-8")
            paths.append(SUMMARY_PATH)
        self.run_git("add", *paths)
        self.run_git("commit", "-qm", "Record v2 sources")

    def minimal_summary(self, **overrides):
        unavailable = {"available": False, "reason": "未比较"}
        summary = {
            "schema": "formal_batch_summary/1", "mode": "complete", "label": "FORMAL", "currency": "TOKENS",
            "validation": {"admission": {"claim": "rejected", "score_not_lower_each_rep": False,
                                         "cost_reduced_on_average": True}},
            "config_summary": {
                "validation": {"default_lib": {"scores": [0.1, 0.2, 0.3], "score_mean": 0.2, "tokens_mean": 1000.0}},
                "exam": {"default": {"scores": [0.4, 0.5, 0.6], "score_mean": 0.5, "tokens_mean": 2000.0}}},
            "exam": {"comparisons": {"default_vs_naive": unavailable, "default_lib_vs_default": unavailable,
                                     "tuned_lib_vs_default_lib": unavailable}},
            "breakeven": {"computed": False, "reasons": ["研究版未通过准入"]},
            "warnings": [],
        }
        summary.update(overrides)
        return json.dumps(summary, ensure_ascii=False)

    def test_committed_formal_summary_shows_validation_and_exam_results(self):
        # Real summarizer output on a synthetic batch, so both scripts stay schema-compatible.
        with tempfile.TemporaryDirectory() as temp:
            fixture = build(Path(temp))
            summary = summarize(fixture.batch_dir, research_cost=2_094_728)
        self.commit_v2_sources(json.dumps(summary, ensure_ascii=False, indent=2))
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        for expected in (
            "内部 token 正式批次已完成 · 验证／考试结果已汇总", "正式批次结果 · validation 与 exam",
            "validation 准入：已准入", "研究版／练过的库", "0.4210", "考试 默认 vs 朴素：有明确优势",
            "每遍平均 token 变化 -23.8%", "研究版数学回本点 4190 题", "不是现金价格或官方比赛成绩",
            SUMMARY_PATH,
        ):
            self.assertIn(expected, report)
        self.assertNotIn("均待对应真实账本审计后填入", report)
        self.assertNotIn("内部 token v2 练习研究进行中", report)

    def test_formal_summary_absent_partial_or_malformed_keeps_pending_wording(self):
        pending = "v2 自研究、验证与考试数字均待对应真实账本审计后填入"
        for name, text in (("absent", None), ("partial", self.minimal_summary(mode="partial")),
                           ("malformed", "{not json"), ("other_currency", self.minimal_summary(currency="TEST")),
                           ("test_only", self.minimal_summary(label="TEST ONLY")),
                           ("no_admission", self.minimal_summary(validation={"admission": None}))):
            with self.subTest(name):
                self.setUp()
                self.commit_v2_sources(text)
                self.assertTrue(generate(self.repo, self.output))
                report = self.output.read_text(encoding="utf-8")
                self.assertIn(pending, report)
                self.assertIn("内部 token v2 练习研究进行中", report)
                self.assertNotIn("正式批次结果 · validation 与 exam", report)

    def test_formal_summary_text_is_html_escaped_and_pending_reasons_are_shown(self):
        self.commit_v2_sources(self.minimal_summary(
            breakeven={"computed": False, "reasons": ["<b>未准入</b>"]}, warnings=["w1", "w2"]))
        self.assertTrue(generate(self.repo, self.output))
        report = self.output.read_text(encoding="utf-8")
        self.assertIn("validation 准入：未准入", report)
        self.assertIn("研究版回本点：不计算（&lt;b&gt;未准入&lt;/b&gt;）", report)
        self.assertNotIn("<b>未准入</b>", report)
        self.assertIn("汇总含 2 条数据提醒", report)
        self.assertIn("考试 默认 vs 朴素：未比较", report)


if __name__ == "__main__":
    unittest.main()
