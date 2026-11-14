"""Source, split, and sandbox checks for independently sourced UCI tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

import harness
import tabular_harness
from scripts.prepare_public_tabular import SOURCE_CITATIONS, SOURCES, source_license_block


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("tab-adult-income", "tab-mushroom-edibility", "tab-spambase",
         "tab-wine-quality", "tab-online-shoppers",
         "tab-breast-cancer-diagnostic", "tab-heart-disease")


class PublicTabularPackageTests(unittest.TestCase):
    def test_bank_sources_preserve_uci_license_and_attribution(self):
        bank = json.loads((ROOT / "tasks/bank.json").read_text(encoding="utf-8"))
        bank_ids = {entry["task_id"] for entry in bank["tasks"]}
        self.assertEqual(set(SOURCE_CITATIONS), {
            "tab-adult-income", "tab-mushroom-edibility", "tab-online-shoppers",
            "tab-breast-cancer-diagnostic", "tab-heart-disease",
        })
        self.assertTrue(set(SOURCE_CITATIONS) <= bank_ids)
        for task_id in SOURCE_CITATIONS:
            with self.subTest(task_id=task_id):
                source = (ROOT / "tasks" / task_id / "SOURCE.md").read_text(encoding="utf-8")
                self.assertIn(source_license_block(task_id, SOURCES[task_id][2]), source)
                self.assertIn("https://creativecommons.org/licenses/by/4.0/", source)
                self.assertIn("- Changes:", source)

    def test_sources_and_visible_split_integrity(self):
        bank = json.loads((ROOT / "tasks/bank.json").read_text(encoding="utf-8"))
        bank_ids = {item["task_id"] for item in bank["tasks"]}
        for task_id in TASKS:
            with self.subTest(task_id=task_id):
                task = harness.load_task(task_id)
                self.assertEqual(task["scenario"], "tabular")
                self.assertEqual(task["deliverable"]["function"], "fit_predict")
                expected_budget = 40_000 if bank["frozen"] and task_id in bank_ids else None
                self.assertEqual(task["budget"]["money"], expected_budget)
                root = task["workspace"].parent
                self.assertTrue((root / "SOURCE.md").is_file())
                manifest = json.loads((root / "split_manifest.json").read_text())
                self.assertEqual(sum(manifest["split_counts"].values()), manifest["total_rows"])
                self.assertEqual(manifest["feature_hash_overlap"], 0)
                self.assertTrue((root / "hidden/holdout.csv").is_file())
                fingerprints = []
                for split in ("train", "dev"):
                    path = root / ("workspace" if split == "train" else "hidden") / f"{split}.csv"
                    with path.open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), manifest["split_counts"][split])
                    self.assertTrue(all(row["y"] in {"0", "1"} for row in rows))
                    self.assertEqual(set(rows[0]) - {"y"}, set(manifest["features"]))
                    fingerprints.append({hashlib.sha256(json.dumps(
                        [row[name] for name in manifest["features"]], separators=(",", ":")
                    ).encode()).hexdigest() for row in rows})
                self.assertFalse(fingerprints[0] & fingerprints[1])

    def test_dev_measurements_are_real_and_task_metric_is_consistent(self):
        for task_id in TASKS:
            with self.subTest(task_id=task_id):
                root = ROOT / "tasks" / task_id
                measurement = json.loads((root / "hidden/reference_measurements.json").read_text())
                task = json.loads((root / "task.json").read_text())
                self.assertEqual(measurement["split"], "dev")
                self.assertEqual(measurement["task_id"], task_id)
                self.assertEqual(measurement["train_rows"],
                                 json.loads((root / "split_manifest.json").read_text())["split_counts"]["train"])
                v2_path = root / "hidden/bank_v2_reference_measurements.json"
                if v2_path.is_file():
                    v2 = json.loads(v2_path.read_text())
                    self.assertEqual(v2["split"], "dev")
                    self.assertGreaterEqual(v2["gap"], 0.01)
                    self.assertEqual(task["metric"]["baseline_raw"], v2["baseline"]["auc"])
                    self.assertEqual(task["metric"]["ceiling_raw"], v2["reference"]["auc"])
                elif measurement["gap_valid"]:
                    self.assertEqual(task["metric"]["baseline_raw"], measurement["baseline"]["auc"])
                    self.assertEqual(task["metric"]["ceiling_raw"], measurement["reference"]["auc"])
                else:
                    self.assertIsNone(task["metric"]["baseline_raw"])
                    self.assertIsNone(task["metric"]["ceiling_raw"])


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "real tabular scoring requires macOS sandbox-exec")
class PublicTabularScoringTests(unittest.TestCase):
    def test_all_original_solutions_score_dev_in_sandbox(self):
        for task_id in TASKS:
            with self.subTest(task_id=task_id), TemporaryDirectory() as temp:
                node = Path(temp)
                shutil.copyfile(ROOT / "tasks" / task_id / "workspace/original.py",
                                node / "solution.py")
                result = tabular_harness.raw_score(task_id, node, "dev", seed=20261114)
                self.assertTrue(result["ok"], result)
                self.assertTrue(result["correct"])
                self.assertGreaterEqual(result["raw"], 0.5)
                previous = os.environ.pop("RA_EVAL", None)
                try:
                    with self.assertRaises(PermissionError):
                        tabular_harness.raw_score(task_id, node, "holdout", seed=20261114)
                finally:
                    if previous is not None:
                        os.environ["RA_EVAL"] = previous


if __name__ == "__main__":
    unittest.main()
