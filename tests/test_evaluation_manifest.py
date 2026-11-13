from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from evaluation.manifest import create_manifest


PROJECT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def setup_root(self, root: Path) -> None:
        config = root / "config"
        config.mkdir()
        for name in ("agent.json", "hard_caps.json"):
            shutil.copyfile(PROJECT / "config" / name, config / name)
        shutil.copyfile(PROJECT / "tests/fixtures/prices_test_only.json", config / "prices_test_only.json")

    def create(self, root: Path, *, phase: str = "dev") -> dict:
        run_dir = root / "runs/run-20261114-120000-abcd"
        return create_manifest(
            project_root=root, run_dir=run_dir, phase=phase, config="default", rep=1,
            bank_version="bank-v1", tasks=["spd-distance-matrix"],
            library_root=run_dir / "library", library_start_version="lib-0",
            library_frozen=phase == "exam", model="test-scripted",
            agent_config_path=root / "config/agent.json",
            prices_path=root / "config/prices_test_only.json", code_commit="a" * 40,
        )

    def test_create_sealed_dev_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_root(root)
            manifest = self.create(root)
            saved = json.loads((root / "runs/run-20261114-120000-abcd/manifest.json").read_text())
            self.assertEqual(saved, manifest)
            self.assertEqual(saved["library"]["start_version"], "lib-0")
            self.assertEqual(saved["prices"]["sha256"], hashlib.sha256((root / "config/prices_test_only.json").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                self.create(root)

    def test_test_prices_cannot_be_exam_prices(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_root(root)
            with self.assertRaisesRegex(ValueError, "TEST prices"):
                self.create(root, phase="exam")

    def test_unset_prices_fail_before_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_root(root)
            (root / "config/prices_test_only.json").write_text(json.dumps({"schema": "prices/1", "currency": None, "unit_tokens": 1000000, "models": {}}))
            with self.assertRaisesRegex(ValueError, "currency is unset"):
                self.create(root)
            self.assertFalse((root / "runs/run-20261114-120000-abcd/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
