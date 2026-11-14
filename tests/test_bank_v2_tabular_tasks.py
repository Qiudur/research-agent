"""Metadata-only checks for five fresh bank-v2 tabular candidates.

No hidden/holdout.csv contents are read.
"""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("tab-spambase", "tab-qsar-biodegradation", "tab-wine-quality",
         "tab-ionosphere", "tab-monks-1")


class BankV2TabularCandidatesTests(unittest.TestCase):
    def test_dev_gap_provenance_license_and_split_metadata(self):
        bank_v1 = json.loads((ROOT / "tasks/bank.json").read_text(encoding="utf-8"))
        prior_ids = {row["task_id"] for row in bank_v1["tasks"]}
        for task_id in TASKS:
            with self.subTest(task_id=task_id):
                self.assertNotIn(task_id, prior_ids)
                root = ROOT / "tasks" / task_id
                task = json.loads((root / "task.json").read_text(encoding="utf-8"))
                measurement = json.loads((root / "hidden/bank_v2_reference_measurements.json").read_text(encoding="utf-8"))
                manifest_path = root / "split_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                source = (root / "SOURCE.md").read_text(encoding="utf-8")
                self.assertEqual(measurement["schema"], "tabular_bank_v2_reference/1")
                self.assertEqual(measurement["split"], "dev")
                self.assertEqual(measurement["task_id"], task_id)
                self.assertGreaterEqual(measurement["gap"], 0.01)
                self.assertEqual(task["metric"]["baseline_raw"], measurement["baseline"]["auc"])
                self.assertEqual(task["metric"]["ceiling_raw"], measurement["reference"]["auc"])
                self.assertEqual(measurement["split_manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
                self.assertEqual(manifest["feature_hash_overlap"], 0)
                source_rows = manifest.get("total_rows", manifest.get("source_rows"))
                self.assertEqual(sum(manifest["split_counts"].values()), source_rows)
                self.assertIn("Creative Commons Attribution 4.0 International", source)
                self.assertIn("holdout", source.lower())
                self.assertIn("dev", source.lower())
                self.assertTrue((root / "hidden/holdout.csv").is_file())
                if task_id == "tab-monks-1":
                    self.assertEqual(task["forbidden_features"], ["source_row_id"])
                    self.assertEqual(manifest["train_only_forbidden_features"], ["source_row_id"])
                    with (root / "workspace/train.csv").open(newline="", encoding="utf-8") as stream:
                        self.assertIn("source_row_id", next(csv.reader(stream)))
                    with (root / "hidden/dev.csv").open(newline="", encoding="utf-8") as stream:
                        self.assertNotIn("source_row_id", next(csv.reader(stream)))


if __name__ == "__main__":
    unittest.main()
