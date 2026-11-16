from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.parse import unquote

from quant_research.data import load_dataset
from quant_research.etf_data import (
    DATASET, MAX_DOWNLOAD_BYTES, METADATA_URL, SYMBOLS,
    _download, _source_prices, fetch_etf_dataset,
)


def source_csv(rows: list[tuple[str, str]]) -> bytes:
    # Deliberately different Close and Adj Close to detect wrong-column selection.
    lines = ["Date,Open,High,Low,Close,Adj Close,Volume"]
    lines.extend(f"{day},100,102,98,100,{price},1000" for day, price in rows)
    return ("\n".join(lines) + "\n").encode()


class ETFDataTests(unittest.TestCase):
    def fake_download(self, url):
        if url == METADATA_URL:
            return json.dumps({"ref": DATASET, "currentVersionNumber": 2,
                               "licenseName": "CC0: Public Domain",
                               "lastUpdated": "2020-04-02T10:55:57.307Z"}).encode()
        self.assertIn("datasetVersionNumber=2", url)
        symbol = unquote(url).split("/etfs/")[1].split(".csv")[0]
        rows = [("2007-12-31", "40"), ("2008-01-02", "50"),
                ("2008-01-03", "51"), ("2020-01-02", "60")]
        if symbol == "GLD":
            rows = [row for row in rows if row[0] != "2008-01-03"]
        return source_csv(rows)

    def test_normalizes_adjusted_close_and_records_intersection_and_hashes(self):
        with TemporaryDirectory() as temporary, patch(
                "quant_research.etf_data._download", side_effect=self.fake_download) as download:
            path = fetch_etf_dataset(Path(temporary) / "market")
            panel = load_dataset(path)
            manifest = json.loads(path.read_text())
            self.assertEqual(download.call_count, 9)
            self.assertEqual(panel.symbols, tuple(sorted(SYMBOLS)))
            self.assertEqual([day.isoformat() for day in panel.dates], ["2008-01-02"])
            self.assertEqual(panel.closes["SPY"], (50,))
            self.assertEqual(manifest["data_kind"], "market")
            self.assertEqual(manifest["requested_start_date"], "2008-01-01")
            self.assertEqual(manifest["requested_end_date"], "2019-12-31")
            policy = manifest["missing_date_policy"]
            self.assertEqual(policy["discarded_dates"], ["2008-01-03"])
            self.assertEqual(policy["missing_dates_by_symbol"]["GLD"], ["2008-01-03"])
            self.assertEqual(policy["discarded_observations_by_symbol"]["SPY"], 1)
            self.assertEqual(policy["discarded_observations_by_symbol"]["GLD"], 0)
            snapshot = manifest["source_snapshot"]
            self.assertEqual(snapshot["metadata_sha256"], hashlib.sha256(
                (path.parent / snapshot["metadata_path"]).read_bytes()).hexdigest())
            for source in snapshot["source_files"]:
                self.assertEqual(source["sha256"], hashlib.sha256(
                    (path.parent / source["path"]).read_bytes()).hexdigest())
            self.assertIn("not independently verified", manifest["license"])
            with self.assertRaises(FileExistsError):
                fetch_etf_dataset(path.parent)
            self.assertEqual(download.call_count, 9)

    def test_rejects_invalid_prices_duplicate_dates_and_wrong_schema(self):
        for price in ("nan", "inf", "0", "-1", "null", ""):
            with self.subTest(price=price), self.assertRaises(ValueError):
                _source_prices(source_csv([("2008-01-02", price)]), "SPY")
        for rows in (
                [("2008-01-02", "50"), ("2008-01-02", "51")],
                [("2008-01-03", "50"), ("2008-01-02", "51")],
                [("2008-02-30", "50")],
                [("2020-01-02", "50")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                _source_prices(source_csv(rows), "SPY")
        with self.assertRaises(ValueError):
            _source_prices(b"Date,Close\n2008-01-02,50\n", "SPY")

    def test_network_failure_never_produces_manifest_or_synthetic_fallback(self):
        with TemporaryDirectory() as temporary, patch(
                "quant_research.etf_data._download", side_effect=OSError("offline")):
            output = Path(temporary) / "market"
            with self.assertRaisesRegex(OSError, "offline"):
                fetch_etf_dataset(output)
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "prices.csv").exists())

    def test_changed_source_version_stops_before_downloading_prices(self):
        changed = json.dumps({"ref": DATASET, "currentVersionNumber": 3}).encode()
        with TemporaryDirectory() as temporary, patch(
                "quant_research.etf_data._download", return_value=changed) as download:
            output = Path(temporary) / "market"
            with self.assertRaisesRegex(ValueError, "version changed"):
                fetch_etf_dataset(output)
            self.assertEqual(download.call_count, 1)
            self.assertFalse((output / "manifest.json").exists())

    def test_download_size_limit_and_permanent_http_error(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"x" * (MAX_DOWNLOAD_BYTES + 1)
        with patch("quant_research.etf_data.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "5 MB"):
                _download("https://example.com/prices.csv")
            response.__enter__.return_value.read.assert_called_once_with(MAX_DOWNLOAD_BYTES + 1)
        error = HTTPError("https://example.com/prices.csv", 404, "missing", {}, None)
        with patch("quant_research.etf_data.urlopen", side_effect=error) as request:
            with self.assertRaises(HTTPError):
                _download("https://example.com/prices.csv")
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
