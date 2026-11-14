"""The planner's ability brief: credibility floor, per-entry clip, total cap."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from library import Library, LibraryError, selected_settings
from tests.test_runner import CAPS, CONFIG, FakeHarness, FakeLedger, FakeRoles, idea, score
from research_agent.runner import TaskRunner

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = {"run_id": "run-20261113-221500-a3f9", "task_id": "tab-demo", "node_ids": ["n0001"]}
AGENT_CONFIG = json.loads((PROJECT / "config/agent.json").read_text(encoding="utf-8"))


def draft(action, credibility=0.5, **extra):
    return {"scope": "domain", "scenario": "tabular", "triggers": ["missing_heavy"],
            "conditions": "many gaps", "action": action, "negative": False,
            "credibility": credibility, **extra}


class BriefTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs/r1/library"

    def library(self, drafts, **settings):
        """A library holding the drafts; settings become agent/1 library_brief."""
        Library.open(self.root).write_experiences(drafts, source=SOURCE)
        if not settings:
            return Library.open(self.root)
        config = {**AGENT_CONFIG, "library_brief": settings}
        path = Path(self.temp.name) / "agent.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with selected_settings(path, PROJECT / "config/triggers.json"):
            return Library.open(self.root)

    def query(self, library):
        return library.query("tabular", ["missing_heavy"], k=20)

    def test_without_any_setting_the_brief_is_the_old_text(self):
        library = self.library([draft("alpha", 0.9), draft("beta" * 100, 0.1)])
        entries = self.query(library)
        expected = "\n".join(
            f"[{e['id']}] 可试：{e['action']}；适用条件：many gaps" for e in entries)
        self.assertEqual(library.brief(entries), expected)
        self.assertEqual([e["id"] for e in library.brief_entries(entries)], [e["id"] for e in entries])

    def test_min_credibility_drops_weaker_entries_and_keeps_the_boundary(self):
        library = self.library([draft("strong", 0.9), draft("edge", 0.5), draft("weak", 0.35)])
        entries = self.query(library)
        texts = {name: name in library.brief(entries, min_credibility=0.5) for name in ("strong", "edge", "weak")}
        self.assertEqual(texts, {"strong": True, "edge": True, "weak": False})
        self.assertEqual(library.brief(entries, min_credibility=0.95), "")
        self.assertEqual(library.brief_entries(entries, min_credibility=0.95), [])

    def test_a_float_that_drifted_below_the_threshold_still_counts(self):
        library = self.library([draft("drifted", 0.5 + 0.1 + 0.1 - 1e-15)])
        self.assertIn("drifted", library.brief(self.query(library), min_credibility=0.7))

    def test_max_chars_clips_each_line_but_keeps_the_id_visible(self):
        library = self.library([draft("short", 0.9), draft("long " * 40, 0.8)])
        entries = self.query(library)
        lines = library.brief(entries, max_chars=40).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= 40 for line in lines))
        self.assertTrue(lines[0].startswith("[exp-0001] 可试：short"))
        self.assertTrue(lines[1].startswith("[exp-0002]") and lines[1].endswith("…"))

    def test_total_length_is_capped_and_lower_ranked_entries_are_dropped(self):
        library = self.library([draft(f"entry {i} " * 10, 0.9 - i * 0.01) for i in range(12)])
        entries = self.query(library)
        text = library.brief(entries, max_chars=40)  # total defaults to 8 full-length lines
        self.assertLessEqual(len(text), 320)
        self.assertEqual(len(text.splitlines()), 7)
        self.assertEqual([line[:10] for line in text.splitlines()], [e["id"].join("[]") for e in entries[:7]])
        tight = library.brief(entries, max_chars=40, max_total_chars=100)
        self.assertLessEqual(len(tight), 100)
        self.assertEqual(len(tight.splitlines()), 2)
        only_total = library.brief(entries, max_total_chars=100)  # no per-line clip: one long line may fill it
        self.assertTrue(0 < len(only_total) <= 100)

    def test_agent_config_block_applies_and_explicit_arguments_override_it(self):
        library = self.library([draft("strong " * 20, 0.9), draft("weak", 0.4)],
                               min_credibility=0.6, max_chars=50)
        entries = self.query(library)
        text = library.brief(entries)
        self.assertEqual([line[:10] for line in text.splitlines()], ["[exp-0001]"])
        self.assertLessEqual(len(text), 50)
        self.assertEqual([e["id"] for e in library.brief_entries(entries)], ["exp-0001"])
        both = library.brief(entries, min_credibility=0.0, max_chars=200)
        self.assertEqual(len(both.splitlines()), 2)
        self.assertEqual(len(library.brief(entries, min_credibility=0.0).splitlines()), 2)

    def test_skill_summary_is_clipped_like_any_other_line(self):
        library = self.library([draft("base", 0.9)])
        body = self.root / "skills/skl-0001.md"
        body.parent.mkdir(parents=True)
        body.write_text("full text", encoding="utf-8")
        library.put({**library.get("exp-0001"), "id": "skl-0001", "kind": "skill",
                     "summary": "summary " * 20, "body_path": "skills/skl-0001.md"},
                    op="promote", reason="test", evidence=[SOURCE], actor="human")
        line = library.brief(library.query("tabular", ["missing_heavy"], kinds=("skill",)), max_chars=40)
        self.assertEqual(len(line), 40)
        self.assertTrue(line.startswith("[skl-0001] 可试：summary"))

    def test_invalid_blocks_are_rejected_when_the_library_opens(self):
        Library.open(self.root)
        bad = [[], "all", {"min_credibility": 1.5}, {"min_credibility": -0.1}, {"min_credibility": "0.5"},
               {"min_credibility": True}, {"max_chars": 5}, {"max_chars": True}, {"max_chars": "40"},
               {"max_chars": 40.0}, {"max_total_chars": 0}, {"colour": "red"}]
        for block in bad:
            config = {**AGENT_CONFIG, "library_brief": block}
            path = Path(self.temp.name) / "agent-bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.subTest(block=block), selected_settings(path, PROJECT / "config/triggers.json"):
                with self.assertRaises(LibraryError):
                    Library.open(self.root)

    def test_invalid_explicit_arguments_are_rejected(self):
        library = self.library([draft("alpha")])
        entries = self.query(library)
        for kwargs in ({"max_chars": 3}, {"min_credibility": 2}, {"max_total_chars": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(LibraryError):
                library.brief(entries, **kwargs)


class PlannerPayloadTests(unittest.TestCase):
    def test_the_planner_only_sees_credible_clipped_entries(self):
        class Recording(FakeRoles):
            def plan(self, payload, *, node_id=None):
                self.briefs = getattr(self, "briefs", []) + [(payload["ability_brief"], payload["available_entry_ids"])]
                return super().plan(payload, node_id=node_id)

        with TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "agent.json"
            config_path.write_text(json.dumps({**AGENT_CONFIG, "library_brief": {
                "min_credibility": 0.45, "max_chars": 60}}), encoding="utf-8")
            with selected_settings(config_path, PROJECT / "config/triggers.json"):
                library = Library.open(root / "runs/run-test/library")
                library.write_experiences(
                    [draft("strong " * 30, 0.8), draft("lost once", 0.35)], source=SOURCE)
                ledger = FakeLedger()
                roles = Recording(ledger, [idea("good", 0.3)], draft=False)
                TaskRunner(run_dir=root / "runs/run-test", task_id="tab-sample",
                           harness=FakeHarness(root), ledger=ledger, roles=roles, library=library,
                           baseline_score=score(0.1), config=CONFIG, hard_caps=CAPS).run()
            brief, available = roles.briefs[0]
            self.assertEqual([line[:10] for line in brief.splitlines()], ["[exp-0001]"])
            self.assertLessEqual(len(brief), 60)
            self.assertEqual(available, ["exp-0001"])  # citable ids follow what the brief shows


if __name__ == "__main__":
    unittest.main()
