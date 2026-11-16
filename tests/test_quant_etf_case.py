from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from quant_research.etf_case import run_case
from tests.test_quant_data import fixture


class ETFCaseTests(unittest.TestCase):
    def test_download_failure_retains_failed_state_without_study_or_report(self):
        with TemporaryDirectory() as temporary, patch(
                "quant_research.etf_data.fetch_etf_dataset", side_effect=OSError("offline")), patch(
                "quant_research.etf_study.run_study") as study, patch(
                "quant_research.etf_report.generate_report") as report:
            output = Path(temporary) / "case"
            with self.assertRaisesRegex(OSError, "offline"):
                run_case(output_dir=output)
            state = json.loads((output / "pipeline.json").read_text())
            self.assertEqual((state["status"], state["phase"]), ("failed", "data"))
            self.assertEqual(state["model_calls"], 0)
            study.assert_not_called()
            report.assert_not_called()
            self.assertFalse((output / "report").exists())

    def test_verified_local_input_reuses_data_without_download(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            output = root / "case"
            expected = output / "report/index.html"
            with patch("quant_research.etf_data.fetch_etf_dataset") as download, patch(
                    "quant_research.etf_study.run_study", return_value=output / "study/study.json") as study, patch(
                    "quant_research.etf_report.generate_report", return_value=expected):
                self.assertEqual(run_case(output_dir=output, dataset_manifest=manifest), expected)
                download.assert_not_called()
                self.assertEqual(study.call_args.args[0], manifest.resolve())
                self.assertEqual(json.loads((output / "pipeline.json").read_text())["status"], "complete")
            previous = (output / "pipeline.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_case(output_dir=output, dataset_manifest=manifest)
            self.assertEqual((output / "pipeline.json").read_bytes(), previous)

    def test_changed_local_prices_fail_before_creating_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            (root / "prices.csv").write_text("changed\n")
            output = root / "case"
            with patch("quant_research.etf_data.fetch_etf_dataset") as download, self.assertRaisesRegex(
                    ValueError, "SHA-256 mismatch"):
                run_case(output_dir=output, dataset_manifest=manifest)
            download.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
