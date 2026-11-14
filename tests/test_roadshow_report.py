"""Regression checks for the hourly roadshow report generator."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.generate_roadshow_report import (FORMAL_EVIDENCE, FormalAuditError, generate,
                                              load_formal_evidence)
import tests.test_audit_formal_outputs as audit_fixtures


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
    PENDING = "v2 自研究、验证与考试数字均待对应真实账本审计后填入"

    def commit_v2_sources(self, evidence_text: str | None = None):
        (self.repo / "README.md").write_text(self.V2_README, encoding="utf-8")
        (self.repo / "config").mkdir(exist_ok=True)
        (self.repo / "config/prices.json").write_text(
            json.dumps({"schema": "prices/1", "currency": "TOKENS", "test_only": False}), encoding="utf-8")
        paths = ["README.md", "config/prices.json"]
        if evidence_text is not None:
            target = self.repo / FORMAL_EVIDENCE
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(evidence_text, encoding="utf-8")
            paths.append(FORMAL_EVIDENCE)
        self.run_git("add", *paths)
        self.run_git("commit", "-qm", "Record v2 sources")

    def formal_fixture(self, admitted: bool = True):
        """A synthetic formal project that passes scripts/audit_formal_outputs.py."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        case = audit_fixtures.FormalOutputAuditTests("test_admitted_and_rejected_batches_are_read_only")
        return case.fixture(Path(temp.name), admitted=admitted)

    def report(self) -> str:
        return self.output.read_text(encoding="utf-8")

    def test_audited_admitted_batch_requires_source_reaudit_for_regeneration(self):
        batch, study, _ = self.formal_fixture(admitted=True)
        formal = load_formal_evidence(batch, study)
        self.commit_v2_sources()
        self.assertTrue(generate(self.repo, self.output, formal=formal))
        report = self.report()
        for expected in (
            "内部 token 正式批次已完成 · 独立审计通过", "正式批次结果 · 已通过独立审计",
            "验证准入：已准入", "考试 4 条线各三遍", "<td>考试</td><td>研究版／练过的库</td>",
            "代码提交 aaaaaaaaaaaa", formal["selected_config_sha256"][:12],
            formal["config_sha256"]["naive"][:12], "考试 研究版 vs 练过的库：有明确优势",
            "研究版数学回本点", "超限 0 题", "不是现金价格或官方比赛成绩", FORMAL_EVIDENCE,
        ):
            self.assertIn(expected, report)
        self.assertNotIn(self.PENDING, report)
        self.assertEqual(json.loads((self.repo / FORMAL_EVIDENCE).read_text(encoding="utf-8")), formal)
        start = report.index("正式批次结果 · 已通过独立审计")
        block = report[start:report.index("</div>", start)]
        # Even with no new commit, a run without source paths must remove old
        # formal numbers from the previously generated HTML.
        self.assertTrue(generate(self.repo, self.output))
        self.assertNotIn("正式批次结果 · 已通过独立审计", self.report())
        self.assertTrue(generate(self.repo, self.output, formal=load_formal_evidence(batch, study)))
        self.assertIn(block, self.report())
        self.run_git("add", FORMAL_EVIDENCE, "docs/路演汇报.html")
        self.run_git("commit", "-qm", "Commit audited formal evidence")
        self.assertTrue(generate(self.repo, self.output))
        self.assertNotIn("正式批次结果 · 已通过独立审计", self.report())
        self.assertIn(self.PENDING, self.report())
        self.assertTrue(generate(self.repo, self.output, formal=load_formal_evidence(batch, study)))
        self.assertIn(block, self.report())

    def test_rejected_batch_hides_tuned_exam_scores_and_breakeven(self):
        batch, study, _ = self.formal_fixture(admitted=False)
        formal = load_formal_evidence(batch, study)
        self.assertNotIn("tuned_lib", formal["phases"]["exam"])
        self.assertFalse(formal["breakeven"]["computed"])
        self.commit_v2_sources()
        self.assertTrue(generate(self.repo, self.output, formal=formal))
        report = self.report()
        for expected in ("验证准入：未准入", "考试 3 条线各三遍", "研究版未通过准入、未参加考试",
                         "<td>验证</td><td>研究版／练过的库</td>"):
            self.assertIn(expected, report)
        for absent in ("<td>考试</td><td>研究版／练过的库</td>", "研究版数学回本点",
                       "考试 研究版 vs 练过的库", self.PENDING):
            self.assertNotIn(absent, report)

    def test_tampered_or_missing_formal_outputs_fail_closed(self):
        batch, study, info = self.formal_fixture()
        run_id = json.loads((batch / "batch_report.json").read_text(encoding="utf-8"))["runs"][0]["run_id"]
        saved = info["root"] / "runs" / run_id / "report.json"
        data = json.loads(saved.read_text(encoding="utf-8"))
        data["tasks"][0]["tokens"]["input"] += 1
        saved.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(FormalAuditError):
            load_formal_evidence(batch, study)
        self.commit_v2_sources()
        script = Path(__file__).resolve().parents[1] / "scripts" / "generate_roadshow_report.py"
        result = subprocess.run(
            [sys.executable, str(script), "--repo", str(self.repo), "--output", str(self.output),
             "--formal-batch-dir", str(batch), "--auto-study-dir", str(study)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("formal audit failed", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.repo / FORMAL_EVIDENCE).exists())

        batch, study, info = self.formal_fixture()
        next((info["root"] / "runs").glob("run-*/tasks/*/holdout_score.json")).unlink()
        with self.assertRaises(FormalAuditError):
            load_formal_evidence(batch, study)

        batch, study, _ = self.formal_fixture()
        (batch / "batch_report.json").unlink()
        self.assertIsNone(load_formal_evidence(batch, study))
        self.assertTrue(generate(self.repo, self.output, formal=None))
        self.assertIn(self.PENDING, self.report())
        self.assertNotIn("正式批次结果", self.report())

    def test_committed_evidence_never_substitutes_for_source_audit(self):
        batch, study, _ = self.formal_fixture()
        good = load_formal_evidence(batch, study)
        forged_score = json.loads(json.dumps(good))
        forged_score["phases"]["exam"]["default"]["scores"][0] = 999.0
        forged_score["phases"]["exam"]["default"]["score_mean"] = 999.0
        forged_sha = json.loads(json.dumps(good))
        forged_sha["selected_config_sha256"] = "0" * 64
        exam_without_tuned = {key: value for key, value in good["phases"]["exam"].items()
                              if key != "tuned_lib"}
        variants = {
            "malformed": "{not json",
            "unaudited": dict(good, audit=dict(good["audit"], ready=False)),
            "other_currency": dict(good, currency="TEST"),
            "test_only": dict(good, label="TEST ONLY"),
            "admission_mismatch": dict(good, admission=dict(good["admission"], claim="rejected")),
            "missing_tuned_line": dict(good, phases=dict(good["phases"], exam=exam_without_tuned)),
            "short_commit": dict(good, code_commit="aaaaaaa"),
            "forged_score_999": forged_score,
            "forged_selected_sha": forged_sha,
            "otherwise_valid_snapshot": good,
        }
        for name, value in variants.items():
            with self.subTest(name):
                self.setUp()
                self.commit_v2_sources(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
                self.assertTrue(generate(self.repo, self.output))
                self.assertIn(self.PENDING, self.report())
                self.assertNotIn("正式批次结果", self.report())

    def test_formal_text_is_html_escaped(self):
        batch, study, _ = self.formal_fixture()
        evidence = load_formal_evidence(batch, study)
        evidence["breakeven"] = {"computed": False, "reasons": ["<b>未达标</b>"]}
        evidence["comparisons"]["default_vs_naive"] = {"available": False, "reason": "<i>缺</i>"}
        evidence["warnings"] = ["w1", "w2"]
        self.commit_v2_sources(json.dumps(evidence, ensure_ascii=False))
        self.assertTrue(generate(self.repo, self.output, formal=evidence))
        report = self.report()
        self.assertIn("研究版回本点：不计算（&lt;b&gt;未达标&lt;/b&gt;）", report)
        self.assertIn("考试 默认 vs 朴素：未比较（&lt;i&gt;缺&lt;/i&gt;）", report)
        self.assertNotIn("<b>未达标</b>", report)
        self.assertIn("汇总含 2 条数据提醒", report)


if __name__ == "__main__":
    unittest.main()
