from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import analyze_ab, run_ab_experiment as ab
from tests.ledger_fixtures import TaskWriter, write_manifest

PROJECT = Path(__file__).resolve().parents[1]
SECRET = "test-secret-never-record"
TASKS = ab.DEFAULT_TASKS


def make_source(root: Path) -> Path:
    """A throwaway project root holding the real configs and the three default practice tasks."""
    shutil.copytree(PROJECT / "config", root / "config")
    (root / "tasks").mkdir()
    shutil.copyfile(PROJECT / "tasks/bank.json", root / "tasks/bank.json")
    for task_id in TASKS + ("spd-prefix-sum",):
        shutil.copytree(PROJECT / "tasks" / task_id, root / "tasks" / task_id)
    tuned = json.loads((root / "config/agent_token_pilot_v2.json").read_text())
    tuned["plateau_nodes"] = 4
    (root / "config/agent_ab_tuned.json").write_text(json.dumps(tuned, indent=2))
    return root


def run_cli(module, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            return module.main(argv), out.getvalue(), err.getvalue()
        except SystemExit as exc:
            return exc.code, out.getvalue(), err.getvalue()


def campaign_args(source: Path, out: Path, *extra):
    return ["--project-root", str(source), "--out", str(out),
            "--configs", "baseline=config/agent_token_pilot_v2.json",
            "--configs", "tuned=config/agent_ab_tuned.json", *extra]


class DriverValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = make_source(Path(self.temp.name).resolve() / "source")
        self.out = self.source / "runs" / "ab-test"

    def fails(self, argv, message):
        with patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}):
            code, _, error = run_cli(ab, argv)
        self.assertNotEqual(code, 0)
        self.assertIn(message, str(code) + error)
        self.assertNotIn(SECRET, str(code) + error)
        self.assertFalse(self.out.exists())

    def test_only_practice_tasks_are_accepted(self):
        self.fails(campaign_args(self.source, self.out, "--dry-run", "--tasks", "spd-prefix-sum"),
                   "practice tasks only")
        self.fails(campaign_args(self.source, self.out, "--dry-run", "--tasks", "spd-nope"),
                   "Unknown task")

    def test_config_specs_must_be_distinct_labelled_files_inside_the_project(self):
        base = ["--project-root", str(self.source), "--out", str(self.out), "--dry-run"]
        self.fails(base + ["--configs", "config/agent.json"], "label=path")
        self.fails(base + ["--configs", "Bad Label=config/agent.json"], "label=path")
        self.fails(base + ["--configs", "a=config/agent.json", "--configs", "a=config/naive.json"],
                   "Duplicate configuration label")
        self.fails(base + ["--configs", "a=" + str(PROJECT / "config/agent.json")],
                   "inside the project")
        self.fails(base + ["--configs", "a=config/nope.json"], "existing file")
        shutil.copyfile(self.source / "config/agent.json", self.source / "config/same.json")
        self.fails(base + ["--configs", "a=config/agent.json", "--configs", "b=config/same.json"],
                   "same settings and prompts")
        broken = json.loads((self.source / "config/agent.json").read_text())
        del broken["prompts"]["summarizer"]
        (self.source / "config/broken.json").write_text(json.dumps(broken))
        self.fails(base + ["--configs", "a=config/broken.json"], "exactly four role prompts")

    def test_counts_and_output_location_are_bounded(self):
        for flag, value in (("--reps", "4"), ("--reps", "0"), ("--workers", "5"), ("--workers", "0")):
            with self.subTest(flag=flag, value=value):
                code, _, _ = run_cli(ab, campaign_args(self.source, self.out, "--dry-run", flag, value))
                self.assertEqual(code, 2)
        self.fails(campaign_args(self.source, self.source / "elsewhere", "--dry-run"),
                   "--out must be a new directory under")
        self.out.mkdir(parents=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}):
            code, _, error = run_cli(ab, campaign_args(self.source, self.out, "--dry-run"))
        self.assertIn("--out must be a new directory", str(code) + error)

    def test_real_mode_needs_the_key_in_the_environment_and_a_safe_gateway(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            code, _, error = run_cli(ab, campaign_args(self.source, self.out))
        self.assertIn("OPENAI_API_KEY must be injected", str(code) + error)
        self.assertFalse(self.out.exists())
        for url in ("http://gateway.example/v1", "https://user:pw@gateway.example/v1",
                    "https://gateway.example/v1?key=" + SECRET):
            with patch.dict(os.environ, {"OPENAI_BASE_URL": url}):
                self.fails(campaign_args(self.source, self.out), "must be an HTTPS URL")
        code, _, _ = run_cli(ab, campaign_args(self.source, self.out, "--base-url", "https://x/v1"))
        self.assertEqual(code, 2)  # the gateway is not a command-line option


class DryRunCampaignTests(unittest.TestCase):
    def test_dry_run_campaign_runs_every_pair_and_analyzes_cleanly(self):
        with TemporaryDirectory() as temp, patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}):
            source = make_source(Path(temp).resolve() / "source")
            out = source / "runs" / "ab-dry"
            code, stdout, _ = run_cli(ab, campaign_args(source, out, "--dry-run", "--reps", "2",
                                                        "--workers", "2"))
            self.assertEqual(code, 0, stdout)
            self.assertIn("Runs completed: 4/4", stdout)
            self.assertIn("python -m scripts.analyze_ab", stdout)
            protocol = json.loads((out / "protocol.json").read_text())
            self.assertEqual((protocol["dry_run"], protocol["reps"], protocol["baseline"]),
                             (True, 2, "baseline"))
            self.assertEqual(protocol["tasks"], list(TASKS))
            self.assertEqual(set(protocol["configs"]), {"baseline", "tuned"})
            self.assertEqual(protocol["gateway"], "none (dry run)")
            self.assertNotIn("base_url", protocol)
            self.assertEqual(protocol["budget_tokens_per_task"], 40_000)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(sorted((r["label"], r["rep"]) for r in summary["runs"]),
                             [("baseline", 1), ("baseline", 2), ("tuned", 1), ("tuned", 2)])
            self.assertTrue(all(r["status"] == "completed" for r in summary["runs"]))
            runs = sorted((out / "project/runs").glob("run-*"))
            self.assertEqual(len(runs), 4)
            for run in runs:
                manifest = json.loads((run / "manifest.json").read_text())
                sidecar = json.loads((run / "ab_run.json").read_text())
                self.assertEqual((manifest["phase"], manifest["tasks"]), ("dev", list(TASKS)))
                self.assertEqual(manifest["rep"], sidecar["rep"])
                self.assertTrue((run / "report.json").is_file())
                for task_id in TASKS:
                    kinds = [json.loads(line)["kind"] for line in
                             (run / "tasks" / task_id / "ledger.jsonl").read_text().splitlines()]
                    self.assertEqual(kinds.count("task_end"), 1)
                    self.assertTrue((run / "tasks" / task_id / "holdout_score.json").is_file())
            # The key must never reach any file of the campaign.
            for path in out.rglob("*"):
                if path.is_file():
                    self.assertNotIn(SECRET.encode(), path.read_bytes(), path)
            # The official bank and task packages stay untouched in the source project.
            self.assertEqual((source / "tasks/bank.json").read_bytes(),
                             (PROJECT / "tasks/bank.json").read_bytes())

            analysis = analyze_ab.analyze(out)
            self.assertEqual(analysis["baseline"], "baseline")
            comparison = analysis["comparisons"]["tuned"]
            self.assertEqual(comparison["n_pairs"], 6)  # 3 tasks x 2 reps
            self.assertEqual(len(analysis["table"]), 12)
            self.assertEqual(analysis["excluded"], [])
            self.assertEqual({row["scenario"] for row in analysis["table"]}, {"tab", "spd"})
            self.assertTrue(all(row["total_tokens"] > 0 for row in analysis["table"]))
            json.dumps(analysis)
            markdown = analyze_ab.render_markdown(analysis)
            self.assertIn("dry-run", markdown)
            self.assertIn("tuned 对 baseline", markdown)
            json_path, markdown_path = Path(temp) / "a.json", Path(temp) / "a.md"
            code, stdout, _ = run_cli(analyze_ab, [str(out), "--json", str(json_path),
                                                   "--markdown", str(markdown_path)])
            self.assertEqual(code, 0)
            self.assertIn("tuned vs baseline", stdout)
            self.assertEqual(json.loads(json_path.read_text())["schema"], "ab_analysis/1")
            self.assertTrue(markdown_path.read_text().startswith("# A/B 实验摘要"))

    def test_configs_may_follow_one_flag_or_repeat_it(self):
        base = "baseline=config/agent_token_pilot_v2.json"
        tuned = "tuned=config/agent_ab_tuned.json"
        forms = {"space": ["--configs", base, tuned], "repeat": ["--configs", base, "--configs", tuned]}
        with TemporaryDirectory() as temp:
            source = make_source(Path(temp).resolve() / "source")
            for name, configs in forms.items():
                out = source / "runs" / f"ab-{name}"
                code, stdout, _ = run_cli(ab, ["--project-root", str(source), "--out", str(out),
                                               "--dry-run", "--tasks", "spd-distance-matrix",
                                               "--reps", "1", "--workers", "1", *configs])
                self.assertEqual(code, 0, stdout)
                protocol = json.loads((out / "protocol.json").read_text())
                self.assertEqual(list(protocol["configs"]), ["baseline", "tuned"])

    def test_real_mode_wiring_uses_environment_key_and_gateway_only_in_memory(self):
        url = "https://gateway.example/v1"
        captured = []
        responses = ab.DryRunResponses("any-task")

        def fake_openai(**kwargs):
            captured.append(kwargs)
            return SimpleNamespace(responses=responses)

        with TemporaryDirectory() as temp, patch.dict(
                os.environ, {"OPENAI_API_KEY": SECRET, "OPENAI_BASE_URL": url}):
            source = make_source(Path(temp).resolve() / "source")
            out = source / "runs" / "ab-real"
            argv = ["--project-root", str(source), "--out", str(out),
                    "--tasks", "spd-distance-matrix", "--reps", "1", "--workers", "1",
                    "--configs", "baseline=config/agent_token_pilot_v2.json"]
            with (patch.object(ab, "harness", ab.DryRunHarness()),
                  patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=fake_openai)})):
                code, stdout, stderr = run_cli(ab, argv)
            self.assertEqual(code, 0, stdout)
            self.assertEqual(len(captured), 1)
            self.assertEqual((captured[0]["api_key"], captured[0]["base_url"], captured[0]["max_retries"]),
                             (SECRET, url, 0))
            protocol = json.loads((out / "protocol.json").read_text())
            self.assertEqual((protocol["gateway"], protocol["dry_run"]), ("OPENAI_BASE_URL", False))
            for secret in (SECRET, "gateway.example"):
                self.assertNotIn(secret, stdout + stderr)
                for path in out.rglob("*"):
                    if path.is_file():
                        self.assertNotIn(secret.encode(), path.read_bytes(), path)

    def test_a_failing_task_keeps_the_rest_of_the_campaign_and_its_evidence(self):
        class Exploding(ab.DryRunHarness):
            def score(self, task_id, node_dir, split, seed=None):
                if split == "holdout" and task_id == "tab-german-credit":
                    raise RuntimeError(f"scorer exploded with {SECRET}")
                return super().score(task_id, node_dir, split, seed)

        with TemporaryDirectory() as temp, patch.dict(os.environ, {"OPENAI_API_KEY": SECRET}):
            source = make_source(Path(temp).resolve() / "source")
            out = source / "runs" / "ab-fail"
            with patch.object(ab, "DryRunHarness", Exploding):
                code, stdout, _ = run_cli(ab, campaign_args(source, out, "--dry-run", "--reps", "1",
                                                            "--workers", "1"))
            self.assertEqual(code, 1)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(all(r["status"] == "partial" for r in summary["runs"]))
            failure = summary["runs"][0]["tasks"]["tab-german-credit"]
            self.assertEqual(failure["status"], "failed")
            self.assertNotIn(SECRET, json.dumps(summary))
            self.assertEqual(summary["runs"][0]["tasks"]["tab-online-shoppers"]["status"], "completed")
            analysis = analyze_ab.analyze(out)
            # The failed task finished its main loop and recorded the failed holdout:
            # its cost counts but it has no holdout score.
            self.assertTrue(failure["holdout_failed"])
            failed_rows = [row for row in analysis["table"] if row["task_id"] == "tab-german-credit"]
            self.assertEqual(len(failed_rows), 2)
            self.assertTrue(all(row["holdout_failed"] is True and row["holdout_score"] is None
                                for row in failed_rows))


def synthetic_campaign(root: Path, *, lean_scale: float = 0.85, drop_lean_rep3_tab_b: bool = False):
    """Hand-built evidence: two configs, three tasks, three reps; lean spends less."""
    out = root / "ab-synthetic"
    out.mkdir()
    (out / "protocol.json").write_text(json.dumps({
        "schema": "ab_protocol/1", "label": "DEV A/B", "dry_run": False, "reps": 3,
        "tasks": ["tab-a", "tab-b", "spd-c"], "baseline": "base",
        "configs": {"base": {}, "lean": {}}}))
    index = 0
    for label, scale in (("base", 1.0), ("lean", lean_scale)):
        for rep in (1, 2, 3):
            index += 1
            run = out / "project/runs" / f"run-20261115-0000{index:02d}-aaaa"
            write_manifest(run, phase="dev", rep=rep, tasks=["tab-a", "tab-b", "spd-c"])
            (run / "ab_run.json").write_text(json.dumps({"schema": "ab_run/1", "label": label, "rep": rep}))
            for task_id, unit in (("tab-a", 3000), ("tab-b", 2400), ("spd-c", 3600)):
                if drop_lean_rep3_tab_b and label == "lean" and rep == 3 and task_id == "tab-b":
                    TaskWriter(run, task_id).plan(100).finish()  # main loop never finished
                    continue
                writer = TaskWriter(run, task_id)
                size = unit + rep * 70
                writer.plan(int(size * scale)).node("n0001", "accepted", 0.5, int(size * 3 * scale))
                writer.node("n0002", "rejected", 0.4, int(size * 2 * scale)).stop("planner", "n0001")
                writer.finish(holdout={"score": 1.0 + rep * 0.01, "raw": 0.8 + rep * 0.001},
                              task_end={"holdout_raw_lift": 0.02 + rep * 0.001})
    return out


class AnalysisTests(unittest.TestCase):
    def test_saving_with_unchanged_quality_is_reported_as_supported(self):
        with TemporaryDirectory() as temp:
            analysis = analyze_ab.analyze(synthetic_campaign(Path(temp)))
            comparison = analysis["comparisons"]["lean"]
            self.assertEqual(analysis["baseline"], "base")
            self.assertEqual(comparison["n_pairs"], 9)
            self.assertEqual(comparison["conclusion"], "cheaper_and_not_worse")
            self.assertAlmostEqual(comparison["cost"]["overall"]["relative_change"], -0.15, places=2)
            self.assertEqual(analysis["configs"]["base"]["rows"], 9)
            self.assertEqual(analysis["configs"]["base"]["stop_reasons"], {"planner": 9})
            self.assertIsNone(analysis["configs"]["base"]["mean_tokens_to_reference"])
            markdown = analyze_ab.render_markdown(analysis)
            self.assertIn("更省，且质量没有显著下降", markdown)
            self.assertIn("lean 对 base（9 对 题×遍）", markdown)
            self.assertIn("tab-b", markdown)
            self.assertNotIn("dry-run", markdown)

    def test_unfinished_tasks_are_excluded_and_never_paired(self):
        with TemporaryDirectory() as temp:
            analysis = analyze_ab.analyze(synthetic_campaign(Path(temp), drop_lean_rep3_tab_b=True))
        self.assertEqual(analysis["excluded"], [{
            "run_id": analysis["excluded"][0]["run_id"], "label": "lean", "rep": 3,
            "task_id": "tab-b", "reason": "main loop did not finish"}])
        comparison = analysis["comparisons"]["lean"]
        self.assertEqual(comparison["n_pairs"], 8)
        self.assertEqual(comparison["unpaired"], [{"config": "base", "task_id": "tab-b", "rep": 3}])

    def test_margins_baseline_choice_and_errors(self):
        with TemporaryDirectory() as temp:
            out = synthetic_campaign(Path(temp), lean_scale=1.2)
            self.assertEqual(analyze_ab.analyze(out)["comparisons"]["lean"]["conclusion"], "costlier")
            flipped = analyze_ab.analyze(out, baseline="lean")
            self.assertEqual(flipped["comparisons"]["base"]["conclusion"], "cheaper_and_not_worse")
            with self.assertRaisesRegex(ValueError, "no finished runs"):
                analyze_ab.analyze(out, baseline="ghost")
            empty = Path(temp) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "No A/B runs"):
                analyze_ab.analyze(empty)
            code, _, error = run_cli(analyze_ab, [str(out), "--quality-margin", "tab=x=1"])
            self.assertEqual(code, 2)
            code, stdout, _ = run_cli(analyze_ab, [str(out), "--baseline", "base",
                                                   "--quality-margin", "tab=0.005", "spd=0.1"])
            self.assertEqual(code, 0)
            self.assertIn("lean vs base: costlier", stdout)
            self.assertTrue((out / "analysis.json").is_file() and (out / "analysis.md").is_file())


if __name__ == "__main__":
    unittest.main()
