"""Portable report evidence, failure paths and untrusted text rendering."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quant_research.backtest import equal_weight_signal, run_backtest
from quant_research.data import PricePanel
from quant_research.etf_report import generate_report


def fixture() -> dict:
    days = tuple(date(2020, 1, value) for value in range(1, 7))
    panel = PricePanel(days, ("A", "B"),
                       {"A": (100, 100, 90, 95, 93, 96),
                        "B": (100, 100, 101, 102, 103, 104)}, "0" * 64)
    kwargs = dict(first_result_date=days[2], last_result_date=days[-1],
                  fee_bps=2, slippage_bps=3)
    ew = run_backtest(panel, equal_weight_signal, top_k=2, **kwargs).to_dict()
    bad = run_backtest(panel, lambda _: {"A": 1, "B": 0}, top_k=1, **kwargs).to_dict()
    window = {"id": "dev", "start": days[2].isoformat(), "end": days[-1].isoformat(), "observations": 4}
    candidate = {"id": "momentum-21", "kind": "momentum", "lookback": 21}
    excess = bad["metrics"]["total_return"] - ew["metrics"]["total_return"]

    def evidence(result, amount):
        return {"window_id": "dev", "result": deepcopy(result), "excess_return": amount}

    return {
        "schema": "etf_case_study/1", "status": "completed",
        "protocol": {"id": "fixture", "question": "Do stated costs matter?", "symbols": ["A", "B"],
                     "data_start": "2020-01-01", "fee_bps": 2, "slippage_bps": 3, "top_k": 1,
                     "fixed_momentum_lookback": 21, "selection_notes": "Maximize worst excess, then mean, then ID.",
                     "execution_notes": "Signal t close; trade t+1 close; return through t+2 close.",
                     "universe_notes": "Fixed surviving ETFs; no point-in-time universe claim."},
        "windows": {"warmup_end": "2020-01-02", "development": [window], "holdout": {**window, "id": "holdout"}},
        "development": {"selection_rule": "max_min_dev_excess_then_mean_then_id",
                        "baselines": [{"id": "equal_weight", "candidate": None, "windows": [evidence(ew, 0)]},
                                      {"id": "fixed_momentum", "candidate": candidate, "windows": [evidence(bad, excess)]}],
                        "candidates": [{**candidate, "windows": [evidence(bad, excess)],
                                        "worst_excess_return": excess, "mean_excess_return": excess}]},
        "selected_candidate": candidate,
        "holdout": {"status": "completed", "series": [
            {"id": "equal_weight", "candidate": None, "result": deepcopy(ew), "excess_return": 0},
            {"id": "fixed_momentum", "candidate": candidate, "result": deepcopy(bad), "excess_return": excess},
            {"id": "selected_grid", "candidate": candidate, "result": deepcopy(bad), "excess_return": excess}]},
        "provenance": {"source": "https://example.com/prices", "license": "Fixture only",
                       "price_basis": "Fixture close", "universe_policy": "fixed_demo",
                       "dataset_manifest": {"path": "fixture.json", "sha256": "a" * 64},
                       "dataset_csv": {"path": "fixture.csv", "sha256": "b" * 64},
                       "protocol": {"path": "protocol.json", "sha256": "c" * 64}, "source_files": []},
        "qwen": {"status": "not_run"},
        "model_usage": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "limitations": ["A deliberately tiny test fixture, not market evidence."],
    }


class EtfReportTests(unittest.TestCase):
    def render(self, study, root):
        source = root / "input.json"
        raw = json.dumps(study, indent=2, ensure_ascii=False).encode() + b"\n"
        source.write_bytes(raw)
        path = generate_report(source, root / "report")
        return path, raw

    def test_evidence_is_copied_exactly_and_all_daily_points_are_plotted(self):
        with TemporaryDirectory() as temp:
            path, raw = self.render(fixture(), Path(temp))
            html = path.read_text()
            self.assertEqual((path.parent / "study.json").read_bytes(), raw)
            self.assertIn(hashlib.sha256(raw).hexdigest(), html)
            self.assertIn('href="study.json"', html)
            self.assertIn("Qwen NOT RUN", html)
            self.assertIn("not Agent performance", html)
            self.assertIn("below equal weight in every observed dev window", html)
            self.assertIn('data-observations="5"', html)
            self.assertEqual(html.count('data-series="selected_grid"'), 4)
            self.assertIn("2020-01-06", html)
            self.assertIn("No model fitting", html)
            self.assertTrue((path.parent / "holdout-net-equity.svg").is_file())
            self.assertTrue((path.parent / "holdout-drawdown.svg").is_file())
            markdown = (path.parent / "report.md").read_text()
            self.assertIn("[Exact evidence: study.json](study.json)", markdown)
            self.assertIn("holdout-net-equity.svg", markdown)
            self.assertNotIn("<script", html)

    def test_failed_or_missing_holdout_has_no_holdout_curve_or_fake_results(self):
        for kind in ("failed", "missing", "incomplete"):
            study = fixture()
            if kind == "failed":
                study["status"] = "failed"
                study["holdout"]["status"] = "failed"
                study["error"] = {"message": "upstream timeout <diagnostic>"}
            elif kind == "missing":
                del study["holdout"]
            else:
                study["holdout"]["series"].pop()
            with self.subTest(kind=kind), TemporaryDirectory() as temp:
                path, _ = self.render(study, Path(temp))
                html = path.read_text()
                self.assertIn("NO COMPLETE HOLDOUT RESULT", html)
                self.assertNotIn("holdout-net-equity", html)
                self.assertFalse((path.parent / "holdout-net-equity.svg").exists())

    def test_unknown_model_usage_is_unknown_while_recorded_zero_stays_zero(self):
        study = fixture()
        study["model_usage"] = {"model_calls": 0, "total_tokens": None}
        with TemporaryDirectory() as temp:
            path, _ = self.render(study, Path(temp))
            html = path.read_text()
            self.assertIn("<td>model_calls</td><td>0</td>", html)
            self.assertIn("<td>total_tokens</td><td>Unknown</td>", html)
            self.assertIn("<td>input_tokens</td><td>Unknown</td>", html)

    def test_untrusted_fields_are_escaped_and_unsafe_urls_are_not_links(self):
        study = fixture()
        attack = '<script>alert("x")</script>'
        study["protocol"]["question"] = attack
        study["provenance"]["source"] = "javascript:alert(1)"
        study["provenance"]["license"] = '<img src=x onerror="alert(1)">'
        study["limitations"] = [attack, "[malicious](javascript:alert(1))"]
        with TemporaryDirectory() as temp:
            path, _ = self.render(study, Path(temp))
            html = path.read_text()
            self.assertNotIn(attack, html)
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn('href="javascript:', html)
            self.assertNotIn('<img src=x', html)
            self.assertIn(r"\[malicious\]", (path.parent / "report.md").read_text())

    def test_optional_source_snapshot_and_missing_date_disclosures_are_visible(self):
        study = fixture()
        study["provenance"].update(
            actual_start_date="2020-01-01", actual_end_date="2020-01-06", date_count=6,
            source_snapshot={"url": "https://example.com/snapshot", "version": 1,
                             "retrieved_at_utc": "2026-11-16T00:00:00Z", "metadata_sha256": "d" * 64,
                             "source_files": [{"source_file": "etfs/A.csv", "sha256": "e" * 64}]},
            missing_date_policy={"method": "intersection; no forward fill", "discarded_date_count": 2,
                                 "calendar_validation": "No independent exchange calendar"})
        with TemporaryDirectory() as temp:
            path, _ = self.render(study, Path(temp))
            html = path.read_text()
            self.assertIn('href="https://example.com/snapshot"', html)
            self.assertIn("intersection; no forward fill", html)
            self.assertIn("No independent exchange calendar", html)
            self.assertIn("discarded date count", html)
            self.assertIn("d" * 64, html)
            self.assertIn("e" * 64, html)

    def test_inconsistent_curves_calendars_and_excess_summaries_are_rejected(self):
        variants = []
        study = fixture()
        study["holdout"]["series"][2]["result"]["equity"].pop()
        variants.append(study)
        study = fixture()
        study["holdout"]["series"][2]["result"]["equity"][-1] += .5
        variants.append(study)
        study = fixture()
        study["holdout"]["series"][2]["result"]["return_dates"][-1] = "2020-01-07"
        variants.append(study)
        study = fixture()
        study["development"]["candidates"][0]["worst_excess_return"] = 5
        variants.append(study)
        study = fixture()
        study["holdout"]["series"][2]["excess_return"] = 1
        variants.append(study)
        study = fixture()
        study["holdout"]["series"][2]["result"]["equity"][-1] = float("nan")
        variants.append(study)
        for index, study in enumerate(variants):
            with self.subTest(index=index), TemporaryDirectory() as temp:
                root = Path(temp)
                with self.assertRaises(ValueError):
                    self.render(study, root)
                self.assertFalse((root / "report").exists())

    def test_existing_report_is_not_overwritten(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = self.render(fixture(), root)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                generate_report(root / "input.json", path.parent)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
