"""Regression checks for the hourly roadshow report generator."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.generate_roadshow_report import generate


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


if __name__ == "__main__":
    unittest.main()
