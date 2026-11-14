"""License-review tests use only Markdown and JSON fixture files, never CSV."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from scripts.review_bank_v2_licenses import SPECS, audit, digest, main, review


PROJECT = Path(__file__).resolve().parents[1]
ALLOWED_FILES = ("SOURCE.md", "split_manifest.json", "task.json",
                 "hidden/bank_v2_reference_measurements.json")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class LicenseReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for task_id in SPECS:
            for relative in ALLOWED_FILES:
                target = self.root / "tasks" / task_id / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / "tasks" / task_id / relative, target)
        self.proposal_path = self.root / "proposal.json"

    def package(self, task_id: str) -> Path:
        return self.root / "tasks" / task_id

    def proposal(self) -> dict:
        return {"schema": "bank_v2_proposal/1", "ready": True, "blocked": [],
                "campaigns": {"tabular": {"task_ids": list(SPECS)}},
                "license_evidence": {item["task_id"]: item["source_md_sha256"]
                                     for item in audit(self.root)}}

    def signed(self, proposal: dict | None = None) -> dict:
        write_json(self.proposal_path, proposal or self.proposal())
        return review(self.root, self.proposal_path, digest(self.proposal_path))

    def test_audits_all_five_and_binds_ready_proposal(self):
        audited = audit(self.root)
        self.assertEqual([row["task_id"] for row in audited], list(SPECS))
        result = self.signed()
        self.assertEqual(result["schema"], "bank_v2_license_review/1")
        self.assertEqual(result["proposal_sha256"], digest(self.proposal_path))
        self.assertEqual([item["task_id"] for item in result["entries"]], list(SPECS))
        self.assertTrue(all(item["redistribution_permitted"] is True
                            for item in result["entries"]))

    def test_doi_license_and_archive_mismatches_fail(self):
        for task_id, old, new, message in (
            ("tab-spambase", "10.24432/C53G6X", "10.24432/BAD", "SOURCE.md"),
            ("tab-qsar-biodegradation", "CC BY 4.0", "private", "SOURCE.md"),
            ("tab-wine-quality", SPECS["tab-wine-quality"]["archive_sha256"],
             "0" * 64, "SOURCE.md"),
        ):
            path = self.package(task_id) / "SOURCE.md"
            original = path.read_text(encoding="utf-8")
            try:
                path.write_text(original.replace(old, new), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    audit(self.root)
            finally:
                path.write_text(original, encoding="utf-8")

    def test_split_seed_and_provenance_mismatches_fail(self):
        path = self.package("tab-ionosphere") / "split_manifest.json"
        original = path.read_text(encoding="utf-8")
        try:
            manifest = json.loads(original)
            manifest["seed"] = 20261114
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "split seed"):
                audit(self.root)
            manifest["seed"] = 20261115
            manifest["archive_sha256"] = "0" * 64
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "archive provenance"):
                audit(self.root)
        finally:
            path.write_text(original, encoding="utf-8")

    def test_dev_gap_and_split_hash_mismatches_fail(self):
        path = self.package("tab-monks-1") / "hidden/bank_v2_reference_measurements.json"
        original = path.read_text(encoding="utf-8")
        try:
            measurement = json.loads(original)
            measurement["reference"]["auc"] = measurement["baseline"]["auc"] + .005
            measurement["gap"] = .005
            write_json(path, measurement)
            with self.assertRaisesRegex(ValueError, "dev gap"):
                audit(self.root)
            measurement = json.loads(original)
            measurement["split_manifest_sha256"] = "0" * 64
            write_json(path, measurement)
            with self.assertRaisesRegex(ValueError, "measurement identity"):
                audit(self.root)
        finally:
            path.write_text(original, encoding="utf-8")

    def test_rejects_stale_source_sha_missing_task_and_unready_proposal(self):
        proposal = self.proposal()
        proposal["license_evidence"]["tab-spambase"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SOURCE.md SHA"):
            self.signed(proposal)
        proposal = self.proposal()
        proposal["campaigns"]["tabular"]["task_ids"].pop()
        with self.assertRaisesRegex(ValueError, "five audited"):
            self.signed(proposal)
        proposal = self.proposal()
        proposal["ready"] = False
        proposal["blocked"] = ["calibration incomplete"]
        with self.assertRaisesRegex(ValueError, "not ready"):
            self.signed(proposal)
        proposal = self.proposal()
        write_json(self.proposal_path, proposal)
        with self.assertRaisesRegex(ValueError, "confirmation differs"):
            review(self.root, self.proposal_path, "0" * 64)

    def test_review_cli_refuses_to_overwrite_existing_output(self):
        write_json(self.proposal_path, self.proposal())
        output = self.root / "review.json"
        args = ["review", "--project-root", str(self.root),
                "--proposal", str(self.proposal_path),
                "--confirm-proposal-sha", digest(self.proposal_path),
                "--output", str(output)]
        self.assertEqual(main(args), 0)
        prior_hash = digest(output)
        with self.assertRaises(SystemExit) as error:
            main(args)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(digest(output), prior_hash)


if __name__ == "__main__":
    unittest.main()
