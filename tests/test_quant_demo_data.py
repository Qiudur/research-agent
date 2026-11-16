from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from quant_research.data import load_dataset
from quant_research.demo_data import write_demo_dataset


class DemoDataTests(unittest.TestCase):
    def test_generates_hashed_synthetic_fixture(self):
        with TemporaryDirectory() as temporary:
            path = write_demo_dataset(Path(temporary) / "synthetic")
            panel = load_dataset(path)
            self.assertEqual(len(panel.dates), 120)
            self.assertEqual(panel.symbols, ("SYN-A", "SYN-B", "SYN-C", "SYN-D"))
            self.assertEqual(len(panel.csv_sha256), 64)
            with self.assertRaises(FileExistsError):
                write_demo_dataset(path.parent)


if __name__ == "__main__":
    unittest.main()
