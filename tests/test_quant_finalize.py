"""Sealed quant holdout finalization and one-shot failure behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ledger import Ledger, Usage
from quant_research.backtest import equal_weight_signal, run_backtest
from quant_research.demo_data import write_demo_dataset
from quant_research.data import load_dataset
from quant_research.finalize import finalize_holdout
from quant_research.harness import QuantHarness
from research_agent.tree import ExperimentTree


BASELINE = "def compute_signal(history):\n    return {symbol: 0.0 for symbol in history}\n"


def put(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuantFinalizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        workspace = self.root / "tasks/quant-demo/workspace"
        workspace.mkdir(parents=True)
        (workspace / "original.py").write_text(BASELINE, encoding="utf-8")
        put(workspace / "profile.json", {
            "schema": "profile/1", "features": [], "summary": {"frequency": "daily"}})
        self.dataset = write_demo_dataset(self.root / "private")
        panel = load_dataset(self.dataset)
        self.harness = QuantHarness(
            workspace.parent, self.dataset,
            train_end=panel.dates[50], dev_start=panel.dates[52],
            dev_end=panel.dates[56], holdout_start=panel.dates[58],
            budget_money=1000.0, wall_seconds=60, top_k=1,
            fee_bps=2.0, slippage_bps=3.0, run_seconds=10, memory_mb=512)
        self.run = self.root / "runs/run-test"
        self.selected = self.run / "tasks/quant-demo/nodes/n0001/solution.py"
        self.selected.parent.mkdir(parents=True)
        self.selected.write_text(BASELINE, encoding="utf-8")
        put(self.selected.parent / "score.json", {
            "schema": "score/1", "split": "dev", "ok": True,
            "correct": True, "score": 0.125})
        baseline = self.run / "tasks/quant-demo/nodes/n0000/score.json"
        put(baseline, {"schema": "score/1", "split": "dev", "ok": True,
                       "correct": True, "score": 0.0})
        (baseline.parent / "solution.py").write_text(BASELINE, encoding="utf-8")
        tree = ExperimentTree.open(self.run, "quant-demo")
        tree.create_root(_read(baseline), "lib-0")
        selected_node = tree.enqueue(parent="n0000", hypothesis="improved signal",
                                     rationale="fixture", origin="plan", cites=[],
                                     estimate={"gain": 0.125, "cost": 0.0, "seconds": 0.0},
                                     priority=0.125, library_version="lib-0")
        self.assertEqual(selected_node["node_id"], "n0001")
        tree.start_attempt("n0001")
        tree.succeed("n0001", _read(self.selected.parent / "score.json"))
        tree.judge("n0001", _read(self.selected.parent / "score.json"))
        rejected = tree.enqueue(parent="n0000", hypothesis="negative signal",
                                rationale="fixture", origin="plan", cites=[],
                                estimate={"gain": -0.25, "cost": 0.0, "seconds": 0.0},
                                priority=0.0, library_version="lib-0")
        self.assertEqual(rejected["node_id"], "n0002")
        self.rejected_code = self.run / "tasks/quant-demo/nodes/n0002/solution.py"
        self.rejected_code.write_text(BASELINE + "# rejected\n", encoding="utf-8")
        self.rejected_score = self.rejected_code.with_name("score.json")
        put(self.rejected_score, {"schema": "score/1", "split": "dev", "ok": True,
                                  "correct": True, "score": -0.25})
        tree.start_attempt("n0002")
        tree.succeed("n0002", _read(self.rejected_score))
        tree.judge("n0002", _read(self.rejected_score))
        failed = tree.enqueue(parent="n0000", hypothesis="failed signal",
                              rationale="fixture", origin="plan", cites=[],
                              estimate={"gain": 0.0, "cost": 0.0, "seconds": 0.0},
                              priority=0.0, library_version="lib-0")
        self.assertEqual(failed["node_id"], "n0003")
        tree.start_attempt("n0003")
        tree.fail("n0003", "fixture", "no candidate generated")
        config = self.root / "config/quant_agent.json"
        prices = self.root / "config/prices.json"
        put(config, {"schema": "agent/1"})
        put(prices, {"schema": "prices/1", "currency": "TEST", "unit_tokens": 1,
                     "models": {"test-scripted": {"input": 1, "cached_input": 1,
                                                 "output": 1, "reasoning": 1}}})
        self.manifest = {
            "schema": "quant_run/1", "run_id": self.run.name, "phase": "dev",
            "tasks": ["quant-demo"], "code_commit": "a" * 40,
            "model": "test-scripted",
            "mode": "scripted",
            "label": "DEVELOPMENT ONLY / SYNTHETIC DATA",
            "library": {"root": str((self.run / "library").relative_to(self.root)),
                        "start_version": "lib-0", "frozen": False},
            "hard_caps": {"budget_ratio": 0.9},
            "split": {"train_end": self.harness.splits.train[-1].isoformat(),
                      "dev_start": self.harness.splits.dev[0].isoformat(),
                      "dev_end": self.harness.splits.dev[-1].isoformat(),
                      "holdout_start": self.harness.splits.holdout[0].isoformat(),
                      "label_horizon": self.harness.splits.label_horizon},
            "execution": {"top_k": 1, "fee_bps": 2.0, "slippage_bps": 3.0},
            "limits": {"budget_tokens": 1000, "wall_seconds": 60,
                       "run_seconds": 10, "memory_mb": 512},
            "dataset": {"path": str(self.dataset.relative_to(self.root)), "sha256": sha(self.dataset)},
            "agent_config": {"path": str(config.relative_to(self.root)), "sha256": sha(config)},
            "prices": {"path": str(prices.relative_to(self.root)), "sha256": sha(prices)}}
        put(self.run / "manifest.json", self.manifest)
        task_ledger = Ledger.open(self.run, "quant-demo")
        task_ledger.emit("task_start", budget=1000.0, wall_seconds=60)
        task_ledger.emit("task_end", best_node="n0001", total_cost=0.0)
        run_ledger = Ledger.open(self.run)
        run_ledger.emit("run_start")
        run_ledger.emit("run_end", status="dev_complete", tasks_completed=1, total_cost=0.0)
        node_artifacts = {}
        for node in tree.all_nodes():
            node_dir = self.run / "tasks/quant-demo/nodes" / node["node_id"]
            node_artifacts[node["node_id"]] = {}
            for kind, filename in (("solution", "solution.py"), ("score", "score.json")):
                path = node_dir / filename
                node_artifacts[node["node_id"]][kind] = (
                    {"path": str(path.relative_to(self.root)), "sha256": sha(path)}
                    if path.exists() else None)
        self.selection = {
            "schema": "quant_dev_selection/1", "task_id": "quant-demo",
            "run_id": self.run.name,
            "candidate": {"path": str(self.selected.relative_to(self.root)),
                          "sha256": sha(self.selected)},
            "dev_evidence": {"path": str((self.selected.parent / "score.json").relative_to(self.root)),
                             "sha256": sha(self.selected.parent / "score.json")},
            "baseline_evidence": {"path": str(baseline.relative_to(self.root)),
                                  "sha256": sha(baseline)},
            "dev_score": 0.125, "dev_baseline_score": 0.0,
            "dataset_manifest_sha256": sha(self.dataset),
            "agent_config_sha256": sha(config), "prices_sha256": sha(prices),
            "manifest_sha256": sha(self.run / "manifest.json"),
            "task_ledger_sha256": sha(task_ledger.path),
            "run_ledger_sha256": sha(run_ledger.path),
            "tree_sha256": sha(tree.path),
            "node_artifacts": node_artifacts,
            "code_commit": "a" * 40, "selected_at": "2026-11-16T00:00:00Z",
            "holdout_scored": False}
        self.selection_path = self.run / "selection.json"
        put(self.selection_path, self.selection)
        self.final = self.run / "final"

    def _fake_candidate_result(self):
        return run_backtest(
            self.harness.panel, equal_weight_signal,
            first_result_date=self.harness.splits.holdout[0],
            last_result_date=self.harness.splits.holdout[-1], top_k=1,
            fee_bps=self.harness.fee_bps, slippage_bps=self.harness.slippage_bps)

    def test_success_is_scored_once_with_baseline_costs_and_provenance(self):
        with patch("quant_research.finalize._score_candidate",
                   return_value=self._fake_candidate_result()) as scorer:
            result = finalize_holdout(self.harness, self.selection_path, self.final)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["label"], "DEVELOPMENT ONLY / SYNTHETIC DATA")
        self.assertEqual(result["mode"], "scripted")
        self.assertEqual(result["selected_node"], "n0001")
        self.assertEqual(result["selected_origin"], "research_candidate")
        self.assertEqual(result["holdout"]["observations"], len(self.harness.splits.holdout))
        self.assertIn("equal_weight", result["holdout"])
        self.assertEqual(result["assumptions"]["fee_bps"], 2.0)
        self.assertEqual(result["provenance"]["dataset_csv_sha256"], self.harness.panel.csv_sha256)
        self.assertEqual(result["provenance"]["tree_sha256"], self.selection["tree_sha256"])
        self.assertEqual(result["dev_model_usage"]["model_calls"], 0)
        self.assertEqual(result["dev_model_usage"]["total_token_buckets"]["input"], 0)
        self.assertEqual(result["dev_model_usage"]["reported_resource_cost"], 0.0)
        self.assertIsNone(result["dev_model_usage"]["cash_cost"])
        self.assertEqual(len(result["warnings"]), 4)
        self.assertEqual(sha(scorer.call_args.args[1]), sha(self.selected))
        self.assertEqual(_read(self.final / "report.json"), result)
        self.assertEqual(_read(self.final / "attempt.json")["status"], "attempted")
        with self.assertRaises(FileExistsError), patch(
                "quant_research.finalize._score_candidate") as second:
            finalize_holdout(self.harness, self.selection_path, self.final)
        second.assert_not_called()

    def test_tamper_and_missing_dev_evidence_block_before_attempt(self):
        cases = (
            ("candidate", lambda: self.selected.write_text(BASELINE + "# changed\n", encoding="utf-8")),
            ("path escape", lambda: self.selection["candidate"].update(path="../secret/solution.py")),
            ("dev score", lambda: put(self.selected.parent / "score.json", {
                "schema": "score/1", "split": "dev", "ok": True,
                "correct": True, "score": 0.5})),
            ("dataset", lambda: self.dataset.write_text("{}", encoding="utf-8")),
        )
        for label, change in cases:
            with self.subTest(label=label), TemporaryDirectory() as temp:
                # Restore fixture after each independent mutation.
                original_candidate = self.selected.read_bytes()
                original_score = (self.selected.parent / "score.json").read_bytes()
                original_dataset = self.dataset.read_bytes()
                original_selection = json.loads(json.dumps(self.selection))
                change()
                put(self.selection_path, self.selection)
                with self.assertRaises((ValueError, FileNotFoundError)), patch(
                        "quant_research.finalize._score_candidate") as scorer:
                    finalize_holdout(self.harness, self.selection_path, self.final)
                scorer.assert_not_called()
                self.assertFalse((self.final / "attempt.json").exists())
                self.selected.write_bytes(original_candidate)
                (self.selected.parent / "score.json").write_bytes(original_score)
                self.dataset.write_bytes(original_dataset)
                self.selection = original_selection
                put(self.selection_path, self.selection)

    def test_failed_scoring_records_no_private_error_and_cannot_retry(self):
        with patch("quant_research.finalize._score_candidate",
                   side_effect=RuntimeError("PRIVATE-MARKER future prices")):
            result = finalize_holdout(self.harness, self.selection_path, self.final)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["holdout"])
        self.assertEqual(result["failure"]["kind"], "scoring_error")
        self.assertNotIn("PRIVATE-MARKER", (self.final / "report.json").read_text())
        with self.assertRaises(FileExistsError):
            finalize_holdout(self.harness, self.selection_path, self.final)

    def test_manifest_and_ledgers_are_frozen_before_attempt(self):
        for label, path in (
            ("manifest", self.run / "manifest.json"),
            ("task ledger", self.run / "tasks/quant-demo/ledger.jsonl"),
            ("run ledger", self.run / "ledger.jsonl"),
        ):
            with self.subTest(label=label):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    with self.assertRaisesRegex(ValueError, "differs from frozen selection"), patch(
                            "quant_research.finalize._score_candidate") as scorer:
                        finalize_holdout(self.harness, self.selection_path, self.final)
                    scorer.assert_not_called()
                    self.assertFalse((self.final / "attempt.json").exists())
                finally:
                    path.write_bytes(original)

    def test_tree_and_rejected_node_files_are_frozen_before_attempt(self):
        for label, path in (
            ("tree", self.run / "tasks/quant-demo/tree.jsonl"),
            ("rejected solution", self.rejected_code),
            ("rejected score", self.rejected_score),
        ):
            with self.subTest(label=label):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    with self.assertRaises(ValueError), patch(
                            "quant_research.finalize._score_candidate") as scorer:
                        finalize_holdout(self.harness, self.selection_path, self.final)
                    scorer.assert_not_called()
                    self.assertFalse((self.final / "attempt.json").exists())
                finally:
                    path.write_bytes(original)

    def test_node_inventory_requires_every_tree_node_and_true_nulls(self):
        failed_solution = self.run / "tasks/quant-demo/nodes/n0003/solution.py"
        for label in ("missing node", "wrong path", "unlisted failed solution"):
            with self.subTest(label=label):
                original = json.loads(json.dumps(self.selection))
                try:
                    if label == "missing node":
                        self.selection["node_artifacts"].pop("n0002")
                    elif label == "wrong path":
                        self.selection["node_artifacts"]["n0002"]["solution"]["path"] = (
                            self.selection["candidate"]["path"])
                    else:
                        failed_solution.write_text(BASELINE, encoding="utf-8")
                    put(self.selection_path, self.selection)
                    with self.assertRaises(ValueError), patch(
                            "quant_research.finalize._score_candidate") as scorer:
                        finalize_holdout(self.harness, self.selection_path, self.final)
                    scorer.assert_not_called()
                    self.assertFalse((self.final / "attempt.json").exists())
                finally:
                    failed_solution.unlink(missing_ok=True)
                    self.selection = original
                    put(self.selection_path, self.selection)

    def test_unknown_usage_is_partial_not_a_reported_total(self):
        task = Ledger.open(self.run, "quant-demo")
        run = Ledger.open(self.run)
        task.path.unlink()
        run.path.unlink()
        task.emit("task_start", budget=1000.0, wall_seconds=60)
        task.llm_call(node_id="n0001", role="planner", purpose="plan",
                      effort="low", model="test-scripted", usage=Usage(input=10),
                      seconds=1.0, prompt_hash="b" * 64)
        task.llm_call(node_id="n0001", role="planner", purpose="quote",
                      effort="low", model="test-scripted", usage=Usage(),
                      seconds=1.0, prompt_hash="c" * 64, usage_status="unknown")
        task.emit("task_end", best_node="n0001", total_cost=10.0)
        run.emit("run_start")
        run.emit("run_end", status="dev_complete", tasks_completed=1, total_cost=10.0)
        self.selection["task_ledger_sha256"] = sha(task.path)
        self.selection["run_ledger_sha256"] = sha(run.path)
        put(self.selection_path, self.selection)
        with patch("quant_research.finalize._score_candidate",
                   return_value=self._fake_candidate_result()):
            result = finalize_holdout(self.harness, self.selection_path, self.final)
        usage = result["dev_model_usage"]
        self.assertEqual(usage["reported_token_buckets"]["input"], 10)
        self.assertIsNone(usage["total_token_buckets"])
        self.assertEqual(usage["model_calls_without_reported_usage"], 1)
        self.assertFalse(usage["usage_complete"])
        self.assertEqual(usage["reported_resource_cost"], 10.0)
        self.assertIsNone(usage["resource_cost"])
        self.assertIn("partial", result["warnings"][-1])

    def test_no_selection_and_abandoned_attempt_cannot_score(self):
        self.selection_path.unlink()
        with self.assertRaises(FileNotFoundError):
            finalize_holdout(self.harness, self.selection_path, self.final)
        put(self.selection_path, self.selection)
        put(self.final / "attempt.json", {"schema": "quant_holdout_attempt/1"})
        with self.assertRaises(FileExistsError), patch(
                "quant_research.finalize._score_candidate") as scorer:
            finalize_holdout(self.harness, self.selection_path, self.final)
        scorer.assert_not_called()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "requires macOS sandbox-exec")
class QuantFinalizeSandboxTests(QuantFinalizeTests):
    def test_real_worker_sees_only_past_prefixes(self):
        result = finalize_holdout(self.harness, self.selection_path, self.final)
        self.assertEqual(result["status"], "scored", result.get("failure"))
        self.assertAlmostEqual(result["holdout"]["net_excess_return"],
                               result["holdout"]["candidate"]["total_return"]
                               - result["holdout"]["equal_weight"]["total_return"])


if __name__ == "__main__":
    unittest.main()
