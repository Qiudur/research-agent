from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.freeze_internal_bank import freeze_bank, prepare_changes, sha


def make_fixture(root: Path):
    (root / "config").mkdir()
    (root / "tasks").mkdir()
    prices = {"schema": "prices/1", "currency": "TOKENS", "unit_tokens": 1,
              "test_only": False, "models": {"qwen3.5-27b": {
                  key: 1 for key in ("input", "cached_input", "output", "reasoning")}}}
    (root / "config/prices.json").write_text(json.dumps(prices), encoding="utf-8")
    rows = []
    updates = []
    for prefix, scenario in (("spd", "speedup"), ("tab", "tabular")):
        for index in range(12):
            task_id = f"{prefix}-task-{index:02d}"
            level = "easy" if index < 4 else "medium" if index < 8 else "hard"
            rows.append({"task_id": task_id, "tier": "practice", "difficulty": None,
                         "naive_cost": None, "sentinel": "keep"})
            updates.append({"task_id": task_id, "difficulty": level,
                            "naive_cost": float(1000 + index)})
            directory = root / "tasks" / task_id
            directory.mkdir()
            task = {"schema": "task/1", "task_id": task_id, "scenario": scenario,
                    "budget": {"money": None, "wall_seconds": 1800}, "sentinel": "keep"}
            (directory / "task.json").write_text(json.dumps(task), encoding="utf-8")
    bank = {"schema": "bank/1", "version": "bank-v1", "frozen": False,
            "tasks": rows, "sentinel": "keep"}
    bank_path = root / "tasks/bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    digest = sha(bank_path)
    proposal = {"schema": "naive_cost_bank_proposal/1",
                "label": "DEV TOKEN RESOURCE PROPOSAL; NOT MONEY OR APPLIED",
                "bank_version": "bank-v1", "source_bank_sha256": digest,
                "model": "qwen3.5-27b", "currency": "TOKENS",
                "budget_tokens_per_task": 40000, "updates": updates}
    audit = {"schema": "naive_cost_aggregation_audit/1", "ready": True,
             "blocked": [], "bank_version": "bank-v1",
             "source_bank_sha256": digest, "bank_proposal": proposal}
    proposal_path = root / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    return proposal_path, proposal, audit


class FreezeBankTests(unittest.TestCase):
    def test_check_then_apply_changes_only_measured_fields(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_path, _, audit = make_fixture(root)
            before = sha(root / "tasks/bank.json")
            kwargs = {"auditor": lambda *_: audit}
            checked = freeze_bank(root, proposal_path, Path("speed"), Path("tab"),
                                  check_only=True, **kwargs)
            self.assertFalse(checked["applied"])
            self.assertEqual(sha(root / "tasks/bank.json"), before)
            applied = freeze_bank(root, proposal_path, Path("speed"), Path("tab"), **kwargs)
            self.assertTrue(applied["applied"])
            self.assertEqual(applied["frozen_bank_sha256"], sha(root / "tasks/bank.json"))
            bank = json.loads((root / "tasks/bank.json").read_text())
            self.assertTrue(bank["frozen"])
            self.assertEqual(bank["sentinel"], "keep")
            self.assertEqual(len(bank["tasks"]), 24)
            for row in bank["tasks"]:
                self.assertEqual(row["sentinel"], "keep")
                self.assertGreater(row["naive_cost"], 0)
                self.assertIn(row["difficulty"], ("easy", "medium", "hard"))
                task = json.loads((root / "tasks" / row["task_id"] / "task.json").read_text())
                self.assertEqual(task["budget"], {"money": 40000, "wall_seconds": 1800})
                self.assertEqual(task["sentinel"], "keep")
            with self.assertRaisesRegex(ValueError, "frozen state"):
                freeze_bank(root, proposal_path, Path("speed"), Path("tab"), **kwargs)

    def test_rejects_unready_or_changed_source_without_writing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_path, _, audit = make_fixture(root)
            not_ready = deepcopy(audit)
            not_ready["ready"] = False
            with self.assertRaisesRegex(ValueError, "ready campaign audit"):
                freeze_bank(root, proposal_path, Path("speed"), Path("tab"),
                            auditor=lambda *_: not_ready)
            bank_path = root / "tasks/bank.json"
            bank_path.write_bytes(bank_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                freeze_bank(root, proposal_path, Path("speed"), Path("tab"),
                            auditor=lambda *_: audit)
            task = json.loads((root / "tasks/spd-task-00/task.json").read_text())
            self.assertIsNone(task["budget"]["money"])

    def test_rejects_missing_and_fabricated_costs_without_writing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, proposal, audit = make_fixture(root)
            for value in (None, -1, float("nan"), True):
                altered = deepcopy(proposal)
                altered["updates"][0]["naive_cost"] = value
                checked_audit = deepcopy(audit)
                checked_audit["bank_proposal"] = altered
                with self.assertRaisesRegex(ValueError, "invalid identity, cost"):
                    prepare_changes(root, altered, checked_audit)
            bank = json.loads((root / "tasks/bank.json").read_text())
            self.assertFalse(bank["frozen"])


if __name__ == "__main__":
    unittest.main()
