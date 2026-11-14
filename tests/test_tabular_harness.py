"""Bank Marketing task separation, scoring contract, and sandbox checks."""

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
from unittest.mock import patch

import harness
import tabular_harness


ROOT = Path(__file__).resolve().parents[1]
TASK = "tab-bank-marketing"


class TabularTaskTests(unittest.TestCase):
    def test_project_override_is_dynamic_after_import(self):
        with TemporaryDirectory() as temp:
            project = Path(temp).resolve()
            (project / "tasks").mkdir()
            (project / "config").mkdir()
            (project / "tasks/bank.json").write_text("{}")
            (project / "config/triggers.json").write_text("{}")
            python = project / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("placeholder")
            with patch.dict(os.environ, {"RA_PROJECT_ROOT": str(project)}):
                self.assertEqual(tabular_harness._project_root(), project)
                self.assertEqual(tabular_harness._python(), python)

    def test_additional_uci_packages_have_disjoint_splits(self):
        for task_id in ("tab-magic-gamma", "tab-htru2", "tab-eeg-eye-state",
                        "tab-default-credit", "tab-qsar-biodegradation",
                        "tab-phishing-websites", "tab-german-credit",
                        "tab-australian-credit"):
            with self.subTest(task_id=task_id):
                task = harness.load_task(task_id)
                self.assertEqual(task["scenario"], "tabular")
                root = task["workspace"].parent
                manifest = json.loads((root / "split_manifest.json").read_text())
                self.assertEqual(sum(manifest["split_counts"].values()), manifest["source_rows"])
                seen = []
                for split in ("train", "dev", "holdout"):
                    path = root / ("workspace" if split == "train" else "hidden") / f"{split}.csv"
                    with path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), manifest["split_counts"][split])
                    self.assertEqual(sum(row["y"] == "1" for row in rows),
                                     manifest["positive_counts"][split])
                    seen.append({hashlib.sha256(json.dumps([row[k] for k in manifest["features"]],
                                                        separators=(",", ":")).encode()).hexdigest()
                                 for row in rows})
                self.assertFalse(seen[0] & seen[1])
                self.assertFalse(seen[0] & seen[2])
                self.assertFalse(seen[1] & seen[2])

    def test_task_package_and_no_feature_overlap(self):
        task = harness.load_task(TASK)
        self.assertEqual(task["scenario"], "tabular")
        root = task["workspace"].parent
        manifest = json.loads((root / "split_manifest.json").read_text())
        self.assertEqual(sum(manifest["split_counts"].values()), 41188)
        self.assertEqual(manifest["excluded"], ["duration"])
        self.assertGreater(manifest["train_only_duration_auc"], 0.8)
        seen = []
        for split in ("train", "dev", "holdout"):
            path = root / ("workspace" if split == "train" else "hidden") / f"{split}.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), manifest["split_counts"][split])
            self.assertEqual(sum(row["y"] == "1" for row in rows), manifest["positive_counts"][split])
            if split == "train":
                self.assertIn("duration", rows[0])
            else:
                self.assertNotIn("duration", rows[0])
            seen.append({hashlib.sha256(json.dumps([row[k] for k in manifest["features"]],
                                                    separators=(",", ":")).encode()).hexdigest()
                         for row in rows})
        self.assertFalse(seen[0] & seen[1])
        self.assertFalse(seen[0] & seen[2])
        self.assertFalse(seen[1] & seen[2])

    def test_auc_ties(self):
        self.assertAlmostEqual(tabular_harness.roc_auc([0, 0, 1, 1], [.1, .4, .35, .8]), .75)
        self.assertAlmostEqual(tabular_harness.roc_auc([0, 1], [.5, .5]), .5)

    def test_score_uses_three_seeds_and_measured_spread(self):
        task = harness.load_task(TASK)
        baseline = task["metric"]["baseline_raw"]
        gap = task["metric"]["ceiling_raw"] - baseline
        values = [baseline, baseline + gap, baseline + 2 * gap]
        measurements = [
            {"ok": True, "raw": value, "correct": True, "cheat_flags": [],
             "error": None, "seconds": 0.1, "seed": 41 + i}
            for i, value in enumerate(values)
        ]
        with patch.object(tabular_harness, "raw_score", side_effect=measurements) as mocked:
            result = tabular_harness.score(TASK, ROOT, "dev", seed=41)
        self.assertEqual([call.args[-1] for call in mocked.call_args_list], [41, 42, 43])
        self.assertAlmostEqual(result["raw"], values[1])
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertAlmostEqual(result["spread"], (2 / 3) ** 0.5)
        self.assertAlmostEqual(result["seconds"]["run"], 0.3)

    def test_holdout_gate_even_for_raw_measurement(self):
        with TemporaryDirectory() as temp, patch.dict(os.environ, {"RA_EVAL": "0"}):
            (Path(temp) / "solution.py").write_text("def fit_predict(*args): return []\n")
            with self.assertRaises(PermissionError):
                tabular_harness.raw_score(TASK, Path(temp), "holdout", seed=7)


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "real scorer requires macOS sandbox-exec")
class TabularSandboxTests(unittest.TestCase):
    def _score(self, code: str):
        with TemporaryDirectory() as temp:
            node = Path(temp)
            (node / "solution.py").write_text(code, encoding="utf-8")
            return tabular_harness.score(TASK, node, "dev", seed=13)

    def test_constant_candidate_receives_no_labels(self):
        result = self._score("""def fit_predict(train_csv, test_rows):
    assert train_csv.endswith('workspace/train.csv')
    assert test_rows and 'y' not in test_rows[0] and 'duration' not in test_rows[0]
    return [0.5] * len(test_rows)
""")
        self.assertEqual(result["schema"], "score/1")
        self.assertTrue(result["ok"])
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["raw"], .5)
        self.assertIsInstance(result["score"], float)
        self.assertAlmostEqual(result["spread"], 0.0)

    def test_bank_duration_leakage_lure_fails_at_inference(self):
        result = self._score("""import csv
def fit_predict(train_csv, test_rows):
    with open(train_csv, newline='') as source:
        assert 'duration' in next(csv.DictReader(source))
    return [float(row['duration']) for row in test_rows]
""")
        self.assertFalse(result["ok"])
        self.assertFalse(result["correct"])
        self.assertIn("duration", result["error"]["message"])

    def test_invalid_probability_is_rejected(self):
        result = self._score("def fit_predict(train_csv, test_rows): return [-0.1] * len(test_rows)\n")
        self.assertTrue(result["ok"])
        self.assertFalse(result["correct"])
        self.assertEqual(result["score"], 0.0)
        self.assertIn("invalid_probability", result["cheat_flags"])

    def test_hidden_read_is_flagged(self):
        hidden = ROOT / "tasks" / TASK / "hidden" / "holdout.csv"
        result = self._score(f"from pathlib import Path\ndef fit_predict(train_csv, test_rows):\n    Path({str(hidden)!r}).read_text()\n    return [0.5] * len(test_rows)\n")
        self.assertFalse(result["ok"])
        self.assertIn("reads_hidden", result["cheat_flags"])

    def test_home_private_read_is_blocked(self):
        with TemporaryDirectory(dir=Path.home()) as private:
            secret = Path(private) / "secret.txt"
            secret.write_text("probe", encoding="utf-8")
            result = self._score(f"from pathlib import Path\ndef fit_predict(train_csv, test_rows):\n    Path({str(secret)!r}).read_text()\n    return [0.5] * len(test_rows)\n")
        self.assertFalse(result["ok"])
        self.assertIn("permitted", result["error"]["message"].lower())


if __name__ == "__main__":
    unittest.main()
