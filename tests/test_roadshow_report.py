"""Regression checks for the hourly roadshow report generator."""

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


if __name__ == "__main__":
    unittest.main()
