from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quant_research.demo import run_demo


class QuantDemoTests(unittest.TestCase):
    def test_key_free_demo_records_only_synthetic_development(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            report_path = run_demo(output)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "quant_demo_report/1")
            self.assertIn("SYNTHETIC", report["label"])
            self.assertEqual(report["model_calls"], 0)
            self.assertFalse(report["holdout_scored"])
            self.assertIsNone(report["selected_candidate"])
            self.assertEqual(len(report["development"]["equal_weight"]["return_dates"]), 44)
            self.assertEqual(len(report["development"]["trailing_momentum_5"]["return_dates"]), 44)
            self.assertEqual(len(report["dataset_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                run_demo(output)


if __name__ == "__main__":
    unittest.main()
