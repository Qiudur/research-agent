"""Causal development selection, retained failures and single-use holdout evidence."""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from quant_research import etf_study
from quant_research.etf_study import run_study


class ETFStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(40)]
        self.csv = self.root / "prices.csv"
        self.manifest_path = self.root / "dataset.json"
        self.protocol_path = self.root / "protocol.json"
        self.manifest = {
            "schema": "quant_dataset/1", "csv": "prices.csv", "csv_sha256": "",
            "source": "Deterministic unit-test prices; not real market data",
            "license": "CC0 synthetic fixture", "price_basis": "synthetic closing levels",
            "universe_policy": "fixed_demo",
            "requested_start_date": self.dates[0].isoformat(),
            "requested_end_date": self.dates[-1].isoformat(),
            "actual_start_date": self.dates[0].isoformat(),
            "actual_end_date": self.dates[-1].isoformat(),
        }
        self.protocol = {
            "schema": "etf_study_protocol/1", "symbols": ["A", "B"],
            "data_start": self.dates[0].isoformat(), "data_end": self.dates[-1].isoformat(),
            "warmup_end": self.dates[10].isoformat(),
            "development_windows": [
                self._window("dev-1", 12, 16), self._window("dev-2", 19, 23),
                self._window("dev-3", 26, 30),
            ],
            "holdout": self._window("holdout", 33, 38),
            "label_horizon": 2, "top_k": 1, "fee_bps": 2, "slippage_bps": 3,
            "fixed_momentum_lookback": 3,
            "candidates": [
                {"id": "momentum-2", "kind": "momentum", "lookback": 2},
                {"id": "momentum-3", "kind": "momentum", "lookback": 3},
                {"id": "reversal-2", "kind": "reversal", "lookback": 2},
                {"id": "low-vol-3", "kind": "low_volatility", "lookback": 3},
            ],
            "selection_rule": "max_min_dev_excess_then_mean_then_id",
        }
        self._write_data()
        self._write_protocol()

    def _window(self, name: str, first: int, last: int) -> dict:
        return {"id": name, "start": self.dates[first].isoformat(), "end": self.dates[last].isoformat()}

    def _write_protocol(self) -> None:
        self.protocol_path.write_text(json.dumps(self.protocol) + "\n")

    def _write_data(self, *, different_holdout: bool = False) -> None:
        rows = ["date,symbol,close"]
        for index, day in enumerate(self.dates):
            for symbol, rate in (("A", 1.005), ("B", 0.995)):
                price = 100 * rate ** index
                if different_holdout and index >= 33:
                    price *= (0.96 if symbol == "A" else 1.03) ** (index - 32)
                rows.append(f"{day},{symbol},{price:.12f}")
        self.csv.write_text("\n".join(rows) + "\n")
        self.manifest["csv_sha256"] = hashlib.sha256(self.csv.read_bytes()).hexdigest()
        self.manifest_path.write_text(json.dumps(self.manifest) + "\n")

    def _run(self, name: str = "study") -> dict:
        result = run_study(self.manifest_path, self.protocol_path, self.root / name)
        self.assertEqual(result, self.root / name / "study.json")
        return json.loads(result.read_text())

    def test_complete_study_keeps_all_candidates_and_daily_evidence(self) -> None:
        study = self._run()
        self.assertEqual(study["status"], "completed")
        self.assertEqual(study["selected_candidate"]["id"], "momentum-2")
        candidates = study["development"]["candidates"]
        self.assertEqual(len(candidates), 4)
        reversal = next(candidate for candidate in candidates if candidate["id"] == "reversal-2")
        self.assertLess(reversal["worst_excess_return"], 0)
        self.assertLess(reversal["mean_excess_return"], 0)
        self.assertTrue(all(window["excess_return"] < 0 for window in reversal["windows"]))
        self.assertEqual(study["model_usage"], {
            "model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        self.assertEqual(study["qwen"]["status"], "not_run")
        self.assertTrue(study["assumptions"]["warmup_not_model_fit"])
        series = study["holdout"]["series"]
        self.assertEqual([item["id"] for item in series], ["equal_weight", "fixed_momentum", "selected_grid"])
        self.assertEqual(series[0]["excess_return"], 0)
        for result in [item["result"] for item in series] + [
                window["result"] for candidate in candidates for window in candidate["windows"]]:
            count = len(result["return_dates"])
            self.assertGreaterEqual(count, 2)
            for field in ("weights", "gross_returns", "net_returns", "turnovers"):
                self.assertEqual(len(result[field]), count)
            self.assertEqual(len(result["equity"]), count + 1)
            self.assertEqual(result["symbols"], ["A", "B"])
        frozen_raw = (self.root / "study/dev_selection.json").read_bytes()
        frozen = json.loads(frozen_raw)
        marker = json.loads((self.root / "study/holdout_attempt.json").read_text())
        self.assertEqual(marker["status"], "completed")
        self.assertEqual(marker["selection_sha256"], hashlib.sha256(frozen_raw).hexdigest())
        self.assertLessEqual(frozen["frozen_at"], marker["started_at"])
        self.assertEqual(frozen["selected_candidate"], study["selected_candidate"])
        self.assertEqual({Path(item["path"]).name for item in study["provenance"]["source_files"]},
                         {"etf_study.py", "backtest.py", "data.py", "signals.py", "io.py"})
        for field in ("requested_start_date", "requested_end_date", "actual_start_date", "actual_end_date"):
            self.assertEqual(study["provenance"][field], self.manifest[field])
        self.assertEqual(study["provenance"]["path_semantics"],
                         "logical labels; actual locations verified in memory")
        self.assertEqual(study["provenance"]["dataset_manifest"]["path"], "dataset/manifest.json")
        self.assertEqual(study["provenance"]["dataset_csv"]["path"], "dataset/prices.csv")
        self.assertEqual(study["provenance"]["protocol"]["path"], "config/protocol.json")
        for descriptor in [study["provenance"][name] for name in
                           ("dataset_manifest", "dataset_csv", "protocol")] + study["provenance"]["source_files"]:
            self.assertFalse(Path(descriptor["path"]).is_absolute())
        for artifact in ("study.json", "dev_selection.json", "holdout_attempt.json"):
            raw = (self.root / "study" / artifact).read_text()
            self.assertNotIn(str(self.root), raw)
            self.assertNotIn(str(Path(etf_study.__file__).resolve().parent), raw)
        self.assertFalse((self.root / "study/prices.csv").exists())

    def test_changed_future_prices_do_not_change_development_or_selection(self) -> None:
        first = self._run("original")
        self._write_data(different_holdout=True)
        second = self._run("changed")
        self.assertEqual(first["development"], second["development"])
        self.assertEqual(first["selected_candidate"], second["selected_candidate"])
        self.assertNotEqual(first["holdout"]["series"], second["holdout"]["series"])
        self.assertNotEqual(first["provenance"]["dataset_csv"], second["provenance"]["dataset_csv"])

    def test_output_directory_is_never_reused(self) -> None:
        self._run()
        original = (self.root / "study/study.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self._run()
        self.assertEqual(original, (self.root / "study/study.json").read_bytes())
        (self.root / "empty").mkdir()
        with self.assertRaises(FileExistsError):
            self._run("empty")

    def test_selection_prioritizes_worst_window_then_mean_then_id(self) -> None:
        candidates = self.protocol["candidates"]
        for candidate, candidate_id in zip(candidates, ("best-mean-z", "best-mean-a", "best-worst", "best-average")):
            candidate["id"] = candidate_id
        self._write_protocol()
        excesses = {
            "best-average": [0.5, 0.5, -0.01],
            "best-worst": [0.02, 0.02, 0.02],
            "best-mean-z": [0.02, 0.03, 0.03],
            "best-mean-a": [0.02, 0.03, 0.03],
        }
        original = etf_study._run_window
        references = {}

        def controlled_development_returns(panel, candidate, window, protocol):
            result = original(panel, candidate, window, protocol)
            if window["id"] != "holdout":
                if candidate is None:
                    references[window["id"]] = result["metrics"]["total_return"]
                elif candidate["id"] in excesses:
                    index = int(window["id"][-1]) - 1
                    result["metrics"]["total_return"] = references[window["id"]] + excesses[candidate["id"]][index]
            return result

        with patch("quant_research.etf_study._run_window", side_effect=controlled_development_returns):
            study = self._run()
        self.assertEqual(study["selected_candidate"]["id"], "best-mean-a")

    def test_a_negative_only_grid_still_retains_and_selects_its_candidate(self) -> None:
        self.protocol["candidates"] = [self.protocol["candidates"][2]]
        self._write_protocol()
        study = self._run()
        self.assertEqual(study["selected_candidate"]["id"], "reversal-2")
        self.assertLess(study["development"]["candidates"][0]["worst_excess_return"], 0)

    def test_costs_reduce_returns_without_changing_holdings(self) -> None:
        paid = self._run("paid")
        self.protocol.update(fee_bps=0, slippage_bps=0)
        self._write_protocol()
        free = self._run("free")
        for paid_item, free_item in zip(paid["holdout"]["series"], free["holdout"]["series"]):
            self.assertEqual(paid_item["result"]["weights"], free_item["result"]["weights"])
            self.assertLess(paid_item["result"]["metrics"]["total_return"],
                            free_item["result"]["metrics"]["total_return"])

    def test_holdout_failure_spends_attempt_and_preserves_development(self) -> None:
        original = etf_study._run_window

        def fail_on_holdout(panel, candidate, window, protocol):
            if window["id"] == "holdout":
                self.assertTrue((self.root / "study/dev_selection.json").exists())
                self.assertTrue((self.root / "study/holdout_attempt.json").exists())
                raise RuntimeError("simulated holdout failure")
            return original(panel, candidate, window, protocol)

        with patch("quant_research.etf_study._run_window", side_effect=fail_on_holdout):
            with self.assertRaisesRegex(RuntimeError, "simulated holdout"):
                self._run()
        study = json.loads((self.root / "study/study.json").read_text())
        marker = json.loads((self.root / "study/holdout_attempt.json").read_text())
        self.assertEqual(study["status"], "failed")
        self.assertEqual(study["holdout"]["status"], "failed")
        self.assertEqual(study["holdout"]["series"], [])
        self.assertEqual(len(study["development"]["candidates"]), 4)
        self.assertEqual(marker["status"], "failed")
        self.assertEqual(marker["error"]["type"], "RuntimeError")
        with self.assertRaises(FileExistsError):
            self._run()

    def test_sealed_protocol_change_during_holdout_fails_attempt(self) -> None:
        original = etf_study._run_window

        def mutate_protocol(panel, candidate, window, protocol):
            if window["id"] == "holdout":
                self.protocol_path.write_text(self.protocol_path.read_text() + " ")
            return original(panel, candidate, window, protocol)

        with patch("quant_research.etf_study._run_window", side_effect=mutate_protocol):
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                self._run()
        marker = json.loads((self.root / "study/holdout_attempt.json").read_text())
        self.assertEqual(marker["status"], "failed")
        frozen = json.loads((self.root / "study/dev_selection.json").read_text())
        self.assertNotEqual(frozen["provenance"]["protocol"]["sha256"],
                            hashlib.sha256(self.protocol_path.read_bytes()).hexdigest())

    def test_changed_csv_during_development_prevents_holdout(self) -> None:
        original = etf_study._run_window

        def mutate_csv(panel, candidate, window, protocol):
            self.csv.write_text(self.csv.read_text() + "\n")
            return original(panel, candidate, window, protocol)

        with patch("quant_research.etf_study._run_window", side_effect=mutate_csv):
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                self._run()
        self.assertFalse((self.root / "study/holdout_attempt.json").exists())
        study = json.loads((self.root / "study/study.json").read_text())
        self.assertEqual(study["holdout"]["status"], "not_run")
        self.assertEqual(study["status"], "failed")

    def test_bad_dataset_digest_is_rejected(self) -> None:
        self.csv.write_text(self.csv.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self._run()

    def test_missing_preregistered_boundaries_cannot_silently_shorten_windows(self) -> None:
        original = self.csv.read_text().splitlines()
        cases = {
            "truncated_holdout": [line for line in original[1:] if line[:10] <= self.dates[34].isoformat()],
            "missing_dev_start": [line for line in original[1:] if line[:10] != self.dates[12].isoformat()],
        }
        for name, rows in cases.items():
            with self.subTest(case=name):
                self.csv.write_text("\n".join([original[0]] + rows) + "\n")
                self.manifest["csv_sha256"] = hashlib.sha256(self.csv.read_bytes()).hexdigest()
                self.manifest_path.write_text(json.dumps(self.manifest) + "\n")
                with self.assertRaisesRegex(ValueError, "preregistered start and end must both be observed"):
                    self._run(name)
                self.assertFalse((self.root / name / "holdout_attempt.json").exists())

    def test_invalid_protocol_is_rejected(self) -> None:
        original = json.dumps(self.protocol)
        changes = [
            lambda p: p.update(symbols=["B", "A"]),
            lambda p: p.update(selection_rule="best_holdout"),
            lambda p: p.update(top_k=3),
            lambda p: p.update(top_k=True),
            lambda p: p.update(label_horizon=1),
            lambda p: p.update(fee_bps=-1),
            lambda p: p.update(slippage_bps=float("nan")),
            lambda p: p.update(candidates=[]),
            lambda p: p["candidates"][0].update(lookback=20),
            lambda p: p["candidates"][1].update(id=p["candidates"][0]["id"]),
            lambda p: p["candidates"][0].update(kind="future_return"),
            lambda p: p.update(development_windows=p["development_windows"][:2]),
            lambda p: p["development_windows"][1].update(start=self.dates[18].isoformat()),
            lambda p: p["holdout"].update(start=self.dates[29].isoformat()),
            lambda p: p["holdout"].update(end=self.dates[33].isoformat()),
            lambda p: p["holdout"].update(end="2021-01-01"),
            lambda p: p.update(warmup_end=self.dates[12].isoformat()),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                self.protocol = json.loads(original)
                change(self.protocol)
                self._write_protocol()
                with self.assertRaises(ValueError):
                    self._run(f"invalid-{index}")


if __name__ == "__main__":
    unittest.main()
