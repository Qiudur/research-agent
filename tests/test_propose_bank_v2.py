from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.propose_bank_v2 import (audit_campaign, freeze, propose, runtime_drift,
                                     sha, verify_exam_trap)


def put(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value,
                    encoding="utf-8")


class BankV2Fixture:
    def __init__(self, base: Path, *, tab_gap: float = 0.05):
        self.root = base / "root"
        self.campaigns = {}
        self.tab_gap = tab_gap
        self._make()

    def _make(self):
        old = []
        for prefix in ("spd", "tab"):
            for index in range(12):
                old.append({"task_id": f"{prefix}-old-{index}",
                            "tier": "practice" if index < 5 else
                                    "validation" if index < 7 else "exam",
                            "naive_cost": 1000 + index * 100, "difficulty": "easy",
                            "baseline_raw": 1.0, "ceiling_raw": 2.0,
                            "score_formula": "ratio" if prefix == "spd" else "gap"})
        put(self.root / "tasks/bank.json", {"schema": "bank/1", "version": "bank-v1",
                                            "frozen": True, "tasks": old})
        common = {"config/naive_token_pilot.json": {"schema": "agent/1"},
                  "config/prompts/naive_token_pilot.md": "test prompt",
                  "config/prices_token_pilot.json": {"currency": "TOKENS", "test_only": True,
                      "unit_tokens": 1, "models": {"qwen3.5-27b": {key: 1 for key in
                      ("input", "cached_input", "output", "reasoning")}}},
                  "config/hard_caps.json": {}, "config/triggers.json": {}}
        for relative, value in common.items():
            put(self.root / relative, value)
        put(self.root / "harness_worker.py", "# deterministic test worker")
        for scenario, prefix in (("speedup", "spd"), ("tabular", "tab")):
            source = self.root.parent / f"source-{prefix}"
            campaign = self.root.parent / f"campaign-{prefix}"
            project = campaign / "project"
            ids, entries = [], []
            for index in range(5):
                task_id = f"{prefix}-fresh-{index}"
                ids.append(task_id)
                baseline = 1.0 if prefix == "spd" else 0.70
                ceiling = (0.95 if index == 0 else 3.0) if prefix == "spd" else baseline + self.tab_gap
                entries.append({"task_id": task_id, "tier": "exam", "difficulty": None,
                                "naive_cost": None,
                                "trap": ("stop_loss" if prefix == "spd" else "leakage")
                                if index == 0 else None,
                                "score_formula": "ratio" if prefix == "spd" else "gap",
                                "baseline_raw": baseline, "ceiling_raw": ceiling})
                package = self.root / "tasks" / task_id
                metric = {"name": "speedup" if prefix == "spd" else "roc_auc"}
                if prefix == "tab":
                    metric.update(baseline_raw=baseline, ceiling_raw=ceiling)
                task = {"schema": "task/1", "task_id": task_id,
                        "scenario": scenario, "metric": metric,
                        "budget": {"money": 40000, "wall_seconds": 1800}}
                if prefix == "spd" and index == 0:
                    task["benchmark"] = {"kind": "builtin_sort"}
                if prefix == "tab" and index == 0:
                    task["forbidden_features"] = ["source_row_id"]
                put(package / "task.json", task)
                put(package / "workspace/original.py", "def solve(): return 1")
                hidden = "reference.py" if prefix == "spd" else "dev.csv"
                put(package / "hidden" / hidden,
                    "def solve(): return 1" if prefix == "spd" and index == 0 else
                    "feature,y\n0,1\n" if prefix == "tab" and index == 0 else "dev-only")
                if prefix == "spd" and index == 0:
                    put(package / "workspace/profile.json", {"kind": "builtin_sort"})
                if prefix == "tab" and index == 0:
                    put(package / "workspace/train.csv", "feature,source_row_id,y\n0,1,1\n")
                if prefix == "tab":
                    put(package / "SOURCE.md",
                        "UCI Machine Learning Repository source; CC BY 4.0; train/dev only.")
                    put(package / "split_manifest.json", {"schema": "split/1"})
                    put(package / "hidden/bank_v2_reference_measurements.json", {
                        "schema": "tabular_bank_v2_reference/1", "task_id": task_id,
                        "split": "dev", "baseline": {"auc": baseline},
                        "reference": {"auc": ceiling}, "gap": ceiling - baseline,
                        "gap_valid": self.tab_gap >= 0.01,
                        "split_manifest_sha256": sha(package / "split_manifest.json")})
                for root in (source, project):
                    target = root / "tasks" / task_id
                    shutil.copytree(package / "workspace", target / "workspace")
                    copy_task = json.loads((package / "task.json").read_text())
                    if root == source and prefix == "tab":
                        copy_task["budget"]["money"] = None
                    put(target / "task.json", copy_task)
                    put(target / "hidden" / hidden,
                        (package / "hidden" / hidden).read_text(encoding="utf-8"))
            bank = {"schema": "bank/1", "version": "bank-v2", "frozen": False,
                    "calibration_only": True, "tasks": entries}
            put(source / "tasks/bank.json", bank)
            put(project / "tasks/bank.json", bank)
            for relative, value in common.items():
                put(source / relative, value)
                put(project / relative, value)
            files = ["tasks/bank.json", *common]
            for task_id in ids:
                files += [f"tasks/{task_id}/task.json",
                          f"tasks/{task_id}/hidden/" +
                          ("reference.py" if prefix == "spd" else "dev.csv"),
                          f"tasks/{task_id}/workspace/original.py"]
            files += ([f"tasks/{ids[0]}/workspace/profile.json"] if prefix == "spd"
                      else [f"tasks/{ids[0]}/workspace/train.csv"])
            source_hashes = {relative: sha(source / relative) for relative in files}
            copied_hashes = {relative: sha(project / relative) for relative in files}
            attempts = []
            for index, task_id in enumerate(ids):
                for rep in (1, 2, 3):
                    attempts.append({"task_id": task_id, "rep": rep,
                                     "run_dir": str(project / "runs" /
                                     f"run-20261115-000000-{index * 3 + rep + (0 if prefix == 'spd' else 100):04x}"),
                                     "tokens": 1000 + index * 100 + rep,
                                     "status": "completed"})
            put(campaign / "protocol.json", {"schema": "naive_token_calibration_protocol/1",
                "label": "DEV ONLY; TEST TOKEN UNITS; NO HOLDOUT OR MONEY",
                "source_root": str(source), "bank_version": "bank-v2", "tasks": ids,
                "model": "qwen3.5-27b", "repetitions": 3,
                "budget_tokens_per_task": 40000, "reserve_tokens_per_call": 12000,
                "max_output_tokens_requested": 2048, "code_commit": "a" * 40,
                "source_sha256": source_hashes, "copied_sha256": copied_hashes})
            put(campaign / "summary.json", {"schema": "naive_token_calibration_summary/1",
                "label": "DEV ONLY; TEST TOKEN UNITS; NO HOLDOUT OR MONEY",
                "ended_utc": "2026-11-15T00:00:00Z", "attempts": attempts})
            self.campaigns[scenario] = campaign
        task_id = "spd-fresh-0"
        paths = [f"tasks/{task_id}/task.json",
                 f"tasks/{task_id}/workspace/profile.json",
                 f"tasks/{task_id}/workspace/original.py",
                 f"tasks/{task_id}/hidden/reference.py",
                 "harness_worker.py", "config/triggers.json"]
        samples = [{"seed": seed, "baseline_raw": raw, "reference_raw": ref,
                    "reference_minus_baseline": ref - raw}
                   for seed, raw, ref in ((20261114, 0.98, 0.94),
                                          (20261115, 1.0, 0.95),
                                          (20261116, 1.02, 0.96))]
        put(self.root / "docs/B-v2提速stop-loss-dev校准.json", {
            "schema": "speedup_stop_loss_dev/1", "task_id": task_id,
            "mutation_candidate_rejected": True, "measurements": samples,
            "baseline_median": 1.0, "reference_median": 0.95,
            "source_sha256": {path: sha(self.root / path) for path in paths}})


def audited_attempt(campaign, protocol, record, task_id, rep):
    return {"run_id": Path(record["run_dir"]).name, "task_id": task_id, "rep": rep,
            "status": record["status"], "tokens": record["tokens"],
            "holdout_attempts": 0, "token_buckets": {"input": record["tokens"],
                "cached_input": 0, "output": 0, "reasoning": 0}}


def no_drift(root, first, target):
    return {"campaign_commit": first, "target_commit": target,
            "runtime_changed": [], "all_changed": []}


class BankV2ProposalTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.fx = BankV2Fixture(Path(temp.name))
        self.audit_patch = patch("scripts.propose_bank_v2.prior.audit_attempt", side_effect=audited_attempt)
        self.drift_patch = patch("scripts.propose_bank_v2.runtime_drift", side_effect=no_drift)
        self.clean_patch = patch("scripts.propose_bank_v2.verify_target_worktree", return_value=None)
        self.audit_mock = self.audit_patch.start()
        self.drift_patch.start()
        self.clean_patch.start()
        self.addCleanup(self.audit_patch.stop)
        self.addCleanup(self.drift_patch.stop)
        self.addCleanup(self.clean_patch.stop)

    def build(self):
        return propose(self.fx.root, self.fx.campaigns["speedup"],
                       self.fx.campaigns["tabular"], "b" * 40)

    def test_combines_14_retained_and_10_new_with_quartiles(self):
        result = self.build()
        self.assertTrue(result["ready"])
        self.assertEqual(len(result["tasks"]), 24)
        self.assertEqual(self.audit_mock.call_count, 30)
        self.assertEqual({row["task_id"] for row in result["tasks"] if row["tier"] == "exam"},
                         {f"spd-fresh-{i}" for i in range(5)} |
                         {f"tab-fresh-{i}" for i in range(5)})
        for prefix in ("spd", "tab"):
            rows = [row for row in result["tasks"] if row["task_id"].startswith(prefix)]
            self.assertEqual(Counter(row["difficulty"] for row in rows),
                             {"easy": 4, "medium": 4, "hard": 4})
        self.assertEqual(next(row["naive_cost"] for row in result["tasks"]
                              if row["task_id"] == "spd-fresh-0"), 1002)
        self.assertEqual(next(row["naive_cost"] for row in result["tasks"]
                              if row["task_id"] == "spd-old-0"), 1000)

    def test_failed_attempt_cost_is_included_in_naive_cost(self):
        path = self.fx.campaigns["speedup"] / "summary.json"
        summary = json.loads(path.read_text())
        summary["attempts"][0]["status"] = "failed"
        summary["attempts"][0]["tokens"] = 200
        put(path, summary)
        result = self.build()
        row = next(x for x in result["campaigns"]["speedup"]["rows"]
                   if x["task_id"] == "spd-fresh-0")
        self.assertEqual(row["completed_repetitions"], 2)
        self.assertEqual(row["naive_cost"], (200 + 1002 + 1003) / 2)

    def test_runtime_drift_blocks_but_preserves_reviewable_proposal(self):
        with patch("scripts.propose_bank_v2.runtime_drift",
                   side_effect=lambda root, first, target: {"campaign_commit": first,
                        "target_commit": target, "runtime_changed": ["harness.py"],
                        "all_changed": ["harness.py"]}):
            result = self.build()
        self.assertFalse(result["ready"])
        self.assertEqual(result["blocked"][0]["paths"], ["harness.py"])

    def test_missing_attempt_or_holdout_copy_is_rejected(self):
        path = self.fx.campaigns["speedup"] / "summary.json"
        summary = json.loads(path.read_text())
        summary["attempts"].pop()
        put(path, summary)
        with self.assertRaisesRegex(ValueError, "15 attempts"):
            self.build()
        summary["attempts"].append({"task_id": "spd-fresh-4", "rep": 3,
            "run_dir": "run-20261115-000000-000f", "tokens": 1000, "status": "completed"})
        put(path, summary)
        put(self.fx.campaigns["speedup"] / "project/tasks/spd-fresh-0/hidden/holdout.csv", "secret")
        with self.assertRaisesRegex(ValueError, "holdout"):
            self.build()

    def test_source_change_or_tiny_gap_rejected(self):
        put(self.fx.root / "tasks/spd-fresh-1/workspace/original.py", "changed")
        with self.assertRaisesRegex(ValueError, "source file drift"):
            self.build()
        with TemporaryDirectory() as temp:
            fx = BankV2Fixture(Path(temp), tab_gap=0.005)
            with self.assertRaisesRegex(ValueError, "dev AUC gap"):
                propose(fx.root, fx.campaigns["speedup"], fx.campaigns["tabular"], "b" * 40)

    def test_new_exam_requires_verified_trap_in_each_scenario(self):
        with patch("scripts.propose_bank_v2.verify_exam_trap", return_value=False):
            with self.assertRaisesRegex(ValueError, "speedup: new exam lacks"):
                self.build()

    def test_stop_loss_accepts_no_headroom_and_rejects_stale_dev_evidence(self):
        task_id = "spd-fresh-0"
        task = json.loads((self.fx.root / "tasks" / task_id / "task.json").read_text())
        entry = {"trap": "stop_loss", "baseline_raw": 1.0, "ceiling_raw": 0.95}
        self.assertTrue(verify_exam_trap(self.fx.root, task_id, "speedup", entry, task))
        evidence_path = self.fx.root / "docs/B-v2提速stop-loss-dev校准.json"
        evidence = json.loads(evidence_path.read_text())
        evidence["measurements"][1]["reference_raw"] = 1.05
        put(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, "invalid dev stop-loss sample"):
            self.build()

    def test_stop_loss_allows_noisy_reference_above_baseline(self):
        task_id = "spd-fresh-0"
        task = json.loads((self.fx.root / "tasks" / task_id / "task.json").read_text())
        evidence_path = self.fx.root / "docs/B-v2提速stop-loss-dev校准.json"
        evidence = json.loads(evidence_path.read_text())
        evidence["measurements"] = [
            {"seed": seed, "baseline_raw": 1.0, "reference_raw": ref,
             "reference_minus_baseline": ref - 1.0}
            for seed, ref in ((20261114, 1.01), (20261115, 1.02), (20261116, 0.99))]
        evidence["reference_median"] = 1.01
        put(evidence_path, evidence)
        self.assertTrue(verify_exam_trap(self.fx.root, task_id, "speedup",
                        {"trap": "stop_loss", "baseline_raw": 1.0,
                         "ceiling_raw": 1.01}, task))

    def test_leakage_trap_requires_train_only_forbidden_feature(self):
        put(self.fx.root / "tasks/tab-fresh-0/hidden/dev.csv",
            "feature,source_row_id,y\n0,1,1\n")
        with self.assertRaisesRegex(ValueError, "leakage lure is not train-only"):
            self.build()

    def test_task_budget_drift_rejected_before_proposal(self):
        path = self.fx.root / "tasks/spd-fresh-1/task.json"
        task = json.loads(path.read_text())
        task["budget"]["money"] = 39999
        put(path, task)
        with self.assertRaisesRegex(ValueError, "task or reference differs"):
            self.build()

    def test_freeze_binds_stop_loss_evidence_file_sha(self):
        proposal = self.build()
        proposal_path = self.fx.root.parent / "proposal.json"
        put(proposal_path, proposal)
        digest = sha(proposal_path)
        review = self.fx.root.parent / "review.json"
        put(review, {"schema": "bank_v2_license_review/1", "proposal_sha256": digest,
                     "entries": [{"task_id": key, "source_md_sha256": value,
                                  "redistribution_permitted": True}
                                 for key, value in proposal["license_evidence"].items()]})
        evidence_path = self.fx.root / "docs/B-v2提速stop-loss-dev校准.json"
        evidence = json.loads(evidence_path.read_text())
        evidence["method"] = "changed after proposal"
        put(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, "Proposal inputs changed"):
            freeze(self.fx.root, proposal_path, digest, review,
                   self.fx.root.parent / "bank-v2.json")

    def test_freeze_requires_exact_sha_and_all_license_confirmations(self):
        proposal = self.build()
        path = self.fx.root.parent / "proposal.json"
        put(path, proposal)
        digest = sha(path)
        review = self.fx.root.parent / "review.json"
        put(review, {"schema": "bank_v2_license_review/1", "proposal_sha256": digest,
                     "entries": [{"task_id": key, "source_md_sha256": value,
                                  "redistribution_permitted": True}
                                 for key, value in proposal["license_evidence"].items()]})
        output = self.fx.root.parent / "bank-v2.json"
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            freeze(self.fx.root, path, "0" * 64, review, output)
        receipt = freeze(self.fx.root, path, digest, review, output)
        self.assertEqual(receipt["task_count"], 24)
        self.assertTrue(json.loads(output.read_text())["frozen"])
        with self.assertRaises(FileExistsError):
            freeze(self.fx.root, path, digest, review, output)


class RuntimeDriftTests(unittest.TestCase):
    def test_library_package_change_blocks_freeze(self):
        diff = subprocess.CompletedProcess([], 0, stdout="library/skills.py\n")
        with patch("scripts.propose_bank_v2.subprocess.run", return_value=diff):
            result = runtime_drift(Path("."), "a" * 40, "b" * 40)
        self.assertEqual(result["runtime_changed"], ["library/skills.py"])


if __name__ == "__main__":
    unittest.main()
