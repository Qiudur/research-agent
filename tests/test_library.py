from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from library import Library, LibraryConflict, LibraryError, LibraryFrozen, selected_settings


SOURCE = {"run_id": "run-20261113-221500-a3f9", "task_id": "tab-demo", "node_ids": ["n0001"]}


class LibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs" / "r1" / "library"
        self.lib = Library.open(self.root)

    def draft(self, action: str = "try target encoding") -> dict:
        return {
            "scope": "domain", "scenario": "tabular",
            "triggers": ["high_cardinality_categorical"],
            "conditions": "many categories", "action": action, "negative": False,
        }

    def test_initial_state_write_snapshot_and_replay(self) -> None:
        self.assertEqual("lib-0", self.lib.version)
        [entry_id] = self.lib.write_experiences([self.draft()], source=SOURCE)
        self.assertEqual("exp-0001", entry_id)
        self.assertEqual("lib-1", self.lib.version)
        self.assertTrue((self.root / "snapshots" / "lib-0.json").exists())
        self.assertTrue((self.root / "snapshots" / "lib-1.json").exists())
        self.assertEqual("lib-0", Library.open(self.root, version="lib-0").version)
        self.assertEqual([], Library.open(self.root, version="lib-0").query("tabular", ["high_cardinality_categorical"]))
        historical = Library.open(self.root, version="lib-1")
        self.assertEqual(entry_id, historical.get(entry_id)["id"])
        with self.assertRaises(LibraryFrozen):
            historical.narrow(entry_id, "larger samples only", reason="test", evidence=[])

    def test_query_brief_and_trigger_validation(self) -> None:
        [entry_id] = self.lib.write_experiences([self.draft()], source=SOURCE)
        selected = self.lib.query("tabular", ["high_cardinality_categorical"])
        self.assertEqual([entry_id], [entry["id"] for entry in selected])
        self.assertIn(entry_id, self.lib.brief(selected))
        self.assertEqual([], self.lib.query("speedup", ["nested_python_loop"]))
        with self.assertRaises(ValueError):
            self.lib.query("tabular", ["unrecognized"])
        with self.assertRaises(LibraryError):
            self.lib.write_experiences([{**self.draft(), "triggers": ["nested_python_loop"]}], source=SOURCE)

    def test_record_use_credibility_demote_retire(self) -> None:
        [entry_id] = self.lib.write_experiences([self.draft()], source=SOURCE)
        self.lib.record_use(entry_id, run_id="r", task_id="t", node_id="n1", accepted=False, gain=-0.1, cost=0.2)
        self.assertEqual(0.35, self.lib.get(entry_id)["credibility"])
        self.lib.record_use(entry_id, run_id="r", task_id="t", node_id="n2", accepted=False, gain=0, cost=0.4)
        self.assertEqual("demoted", self.lib.get(entry_id)["status"])
        self.assertEqual([], self.lib.query("tabular", ["high_cardinality_categorical"]))
        for index in range(3, 6):
            self.lib.record_use(entry_id, run_id="r", task_id="t", node_id=f"n{index}", accepted=False, gain=0, cost=0.1)
        entry = self.lib.get(entry_id)
        self.assertEqual("retired", entry["status"])
        self.assertEqual(5, entry["effect"]["uses"])
        self.assertAlmostEqual(0.18, entry["effect"]["mean_cost"])
        with self.assertRaises(LibraryError):
            self.lib.record_use(entry_id, run_id="r", task_id="t", node_id="n6", accepted=False, gain=0, cost=0)

    def test_narrow_and_changelog(self) -> None:
        [entry_id] = self.lib.write_experiences([self.draft()], source=SOURCE)
        self.lib.narrow(entry_id, "sample size > 10000", reason="failed on small data", evidence=[SOURCE])
        self.assertEqual("lib-2", self.lib.version)
        self.assertEqual("sample size > 10000", self.lib.get(entry_id)["conditions"])
        changes = self.lib.changelog(since="lib-1")
        self.assertEqual(1, len(changes))
        self.assertEqual("narrow", changes[0]["op"])
        self.assertEqual("many categories", changes[0]["before"]["conditions"])
        self.assertEqual("sample size > 10000", changes[0]["after"]["conditions"])
        self.assertEqual("many categories", Library.open(self.root, version="lib-1").get(entry_id)["conditions"])

    def test_duplicate_merge_and_stale_writer(self) -> None:
        [entry_id] = self.lib.write_experiences([self.draft()], source=SOURCE)
        stale = Library.open(self.root)
        another_source = {"run_id": "r2", "task_id": "t2", "node_ids": ["n0003"]}
        self.assertEqual([entry_id], self.lib.write_experiences([self.draft()], source=another_source))
        self.assertEqual("merge", self.lib.changelog()[-1]["op"])
        self.assertEqual(2, len(self.lib.get(entry_id)["sources"]))
        with self.assertRaises(LibraryConflict):
            stale.narrow(entry_id, "different", reason="test", evidence=[])

    def test_frozen_library(self) -> None:
        [entry_id] = self.lib.write_experiences([self.draft()], source=SOURCE)
        frozen = Library.open(self.root, frozen=True)
        with self.assertRaises(LibraryFrozen):
            frozen.record_use(entry_id, run_id="r", task_id="t", node_id="n", accepted=True, gain=1, cost=0)
        with self.assertRaises(LibraryFrozen):
            frozen.write_experiences([self.draft("other")], source=SOURCE)
        self.assertEqual("lib-1", frozen.version)

    def test_skill_brief_only_summary_and_safe_loading(self) -> None:
        self.lib.write_experiences([self.draft()], source=SOURCE)
        body = self.root / "skills" / "skl-0001.md"
        body.parent.mkdir(parents=True)
        body.write_text("Detailed steps with expensive context", encoding="utf-8")
        skill = {
            **self.lib.get("exp-0001"), "id": "skl-0001", "kind": "skill",
            "summary": "Target encode high-cardinality columns",
            "body_path": "skills/skl-0001.md",
        }
        self.lib.put(skill, op="promote", reason="repeated wins", evidence=[SOURCE], actor="online")
        text = self.lib.brief(self.lib.query("tabular", ["high_cardinality_categorical"], kinds=("skill",)))
        self.assertIn("Target encode high-cardinality columns", text)
        self.assertNotIn("Detailed steps", text)
        self.assertEqual("Detailed steps with expensive context", self.lib.load_skill("skl-0001"))
        bad = self.lib.get("skl-0001")
        bad["body_path"] = "../../outside.md"
        self.lib.put(bad, op="update", reason="test path validation", evidence=[], actor="human")
        with self.assertRaises(LibraryError):
            self.lib.load_skill("skl-0001")

    def test_cap_retires_lowest_credibility(self) -> None:
        agent_config = json.loads((Path(__file__).resolve().parents[1] / "config" / "agent.json").read_text(encoding="utf-8"))
        agent_config["library_rules"]["max_experiences"] = 1
        config = Path(self.temp.name) / "config.json"
        config.write_text(json.dumps(agent_config), encoding="utf-8")
        with patch.dict(os.environ, {"RA_AGENT_CONFIG": str(config)}):
            limited = Library.open(self.root)
            limited.write_experiences([self.draft("one"), self.draft("two")], source=SOURCE)
            active = [item for item in limited.entries if item["status"] == "active"]
            self.assertEqual(1, len(active))
            self.assertEqual(2, len(limited.entries))

    def test_parallel_runs_use_independent_library_settings(self) -> None:
        project = Path(__file__).resolve().parents[1]
        source = json.loads((project / "config/agent.json").read_text(encoding="utf-8"))
        triggers = project / "config/triggers.json"
        configs = []
        for index, limit in enumerate((1, 2)):
            config = json.loads(json.dumps(source))
            config["library_rules"]["max_experiences"] = limit
            path = Path(self.temp.name) / f"agent-{index}.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            configs.append(path)
        barrier = threading.Barrier(2)

        def inspect(path):
            with selected_settings(path, triggers):
                barrier.wait(timeout=5)
                return Library.open(self.root).rules["max_experiences"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(inspect, path) for path in configs]
            self.assertEqual([future.result() for future in futures], [1, 2])

    def test_state_tampering_is_detected(self) -> None:
        self.lib.write_experiences([self.draft()], source=SOURCE)
        path = self.root / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["entries"][0]["action"] = "tampered"
        path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(LibraryError):
            Library.open(self.root)


if __name__ == "__main__":
    unittest.main()
