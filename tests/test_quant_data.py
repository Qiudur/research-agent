from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quant_research.data import PricePanel, load_dataset


GOOD_CSV = ("date,symbol,close\n"
            "2024-01-02,B,20\n"
            "2024-01-02,A,10\n"
            "2024-01-03,A,11\n"
            "2024-01-03,B,21\n"
            "2024-01-04,A,12\n"
            "2024-01-04,B,22\n")


def fixture(root: Path, csv_text: str = GOOD_CSV, **changes: object) -> Path:
    csv_path = root / "prices.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    manifest = {
        "schema": "quant_dataset/1", "csv": "prices.csv",
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source": "deterministic test fixture", "license": "synthetic",
        "price_basis": "synthetic unadjusted prices",
        "universe_policy": "fixed_demo",
    }
    manifest.update(changes)
    path = root / "dataset.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class QuantDataTests(unittest.TestCase):
    def test_loads_hashed_complete_panel_in_deterministic_order(self):
        with TemporaryDirectory() as temp:
            manifest = fixture(Path(temp))
            panel = load_dataset(manifest)
            self.assertEqual(panel.dates, (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)))
            self.assertEqual(panel.symbols, ("A", "B"))
            self.assertEqual(panel.closes, {"A": (10.0, 11.0, 12.0),
                                            "B": (20.0, 21.0, 22.0)})
            self.assertEqual(panel.csv_sha256, hashlib.sha256(
                (Path(temp) / "prices.csv").read_bytes()).hexdigest())
            with self.assertRaises(TypeError):
                panel.closes["A"] = (1.0,)

    def test_rejects_changed_data_and_incomplete_provenance(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = fixture(root)
            (root / "prices.csv").write_text(GOOD_CSV + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_dataset(manifest)
            for change, message in (({"schema": "other"}, "quant_dataset/1"),
                                    ({"source": "  "}, "source"),
                                    ({"license": ""}, "license"),
                                    ({"price_basis": None}, "price_basis"),
                                    ({"universe_policy": "current_survivors"}, "fixed_demo"),
                                    ({"csv_sha256": "placeholder"}, "SHA-256")):
                with self.subTest(change=change):
                    bad = fixture(root, **change)
                    with self.assertRaisesRegex(ValueError, message):
                        load_dataset(bad)

    def test_rejects_csv_paths_escaping_manifest_directory(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inner = root / "inner"
            inner.mkdir()
            outside = root / "outside.csv"
            outside.write_text(GOOD_CSV, encoding="utf-8")
            for name in ("../outside.csv", str(outside)):
                with self.subTest(name=name):
                    manifest = fixture(inner, csv=name)
                    with self.assertRaisesRegex(ValueError, "relative path|within its manifest"):
                        load_dataset(manifest)
            (inner / "link.csv").symlink_to(outside)
            manifest = fixture(inner, csv="link.csv")
            with self.assertRaisesRegex(ValueError, "within its manifest"):
                load_dataset(manifest)

    def test_rejects_bad_or_incomplete_rows(self):
        cases = (
            ("date,symbol,close\n", "no price rows"),
            (GOOD_CSV + "2024-01-04,A,12\n", "duplicates"),
            (GOOD_CSV.replace("2024-01-04,B,22\n", ""), "same symbols"),
            (GOOD_CSV.replace("2024-01-03,B,21\n", "2024-01-01,B,21\n"), "in increasing order"),
            (GOOD_CSV.replace("2024-01-03,B,21", "2024-01-03,B,0"), "invalid symbol or close"),
            (GOOD_CSV.replace("2024-01-03,B,21", "2024-01-03,B,nan"), "invalid symbol or close"),
            (GOOD_CSV.replace("2024-01-03,B,21", "2024-01-03,B,nope"), "invalid date or close"),
            (GOOD_CSV.replace("2024-01-03,B,21", "2024-13-03,B,21"), "invalid date or close"),
            (GOOD_CSV.replace("2024-01-03,B,21", "2024-01-03, ,21"), "invalid symbol or close"),
            (GOOD_CSV.replace("date,symbol,close", "date,symbol,price"), "columns"),
            (GOOD_CSV.replace("2024-01-03,B,21", "2024-01-03,B,21,extra"), "three values"),
        )
        for csv_text, message in cases:
            with self.subTest(message=message, csv_text=csv_text), TemporaryDirectory() as temp:
                manifest = fixture(Path(temp), csv_text)
                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(manifest)

    def test_direct_panel_constructor_checks_complete_immutable_shape(self):
        days = (date(2024, 1, 1), date(2024, 1, 2))
        with self.assertRaisesRegex(ValueError, "complete"):
            PricePanel(days, ("A",), {"A": (1.0,)}, "test")
        with self.assertRaisesRegex(ValueError, "increasing"):
            PricePanel(tuple(reversed(days)), ("A",), {"A": (1.0, 2.0)}, "test")
