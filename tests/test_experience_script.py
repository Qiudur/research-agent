"""Script-written experiences: the evidence rules, and the full runner path."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from library import Library
from research_agent.experience_script import (MAX_NEGATIVE, MAX_POSITIVE, build_drafts,
                                              drafts_from_tree, read_attempt_details, writer_mode)
from research_agent.io import read_jsonl
from research_agent.model import ModelClient
from research_agent.roles import RoleAgents
from research_agent.runner import TaskRunner
from tests.test_reasoning_controls import ROOT, ScriptedGateway, TokenLedger
from tests.test_runner import CAPS, CONFIG, FakeHarness, FakeLedger, FakeRoles, idea, score

FEATURES = ["missing_heavy", "small_sample"]
SCRIPT = {**CONFIG, "experience_writer": "script"}
SOURCE = {"run_id": "run-test", "task_id": "tab-sample"}


def node(node_id, parent, status, *, dev=None, spread=0.01, flags=(), failure=None,
         hypothesis=None, cost=100.0, attempts=1, origin="plan", correct=True):
    scored = None if dev is None else {
        "dev": dev, "raw": dev, "spread": spread, "correct": correct, "cheat_flags": list(flags)}
    return {"schema": "node/1", "node_id": node_id, "parent": parent, "origin": origin,
            "status": status, "score": scored, "failure": failure, "cost": cost,
            "attempts": attempts, "cites": [], "hypothesis": hypothesis or f"idea {node_id}"}


ROOT_NODE = node("n0000", None, "accepted", dev=0.40, origin="root", hypothesis="Baseline")


def drafts(*nodes, features=FEATURES, **kwargs):
    return build_drafts([ROOT_NODE, *nodes], scenario="tabular", features=features, **kwargs)


def failed(node_id, summary, kind="exception", **kwargs):
    return node(node_id, "n0000", "failed", failure={"kind": kind, "summary": summary}, **kwargs)


class WriterModeTests(unittest.TestCase):
    def test_default_is_the_model_summarizer(self):
        self.assertEqual(writer_mode({}), "model")
        self.assertEqual(writer_mode({"experience_writer": "model"}), "model")
        self.assertEqual(writer_mode({"experience_writer": "script"}), "script")

    def test_anything_else_is_an_error(self):
        for value in ("Script", "scripted", "", None, True, ["script"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                writer_mode({"experience_writer": value})


class EvidenceRuleTests(unittest.TestCase):
    def test_accepted_gain_beyond_both_spreads_is_a_positive_experience(self):
        [draft] = drafts(node("n0001", "n0000", "accepted", dev=0.50, hypothesis="对类别列做目标编码"))
        self.assertEqual(draft["node_ids"], ["n0001"])
        self.assertIs(draft["negative"], False)
        self.assertEqual((draft["scope"], draft["scenario"]), ("domain", "tabular"))
        self.assertEqual(draft["triggers"], FEATURES)
        self.assertIn("对类别列做目标编码", draft["action"])
        self.assertIn("0.4000 → 0.5000", draft["action"])
        self.assertTrue(draft["conditions"].strip())

    def test_gain_must_clear_the_parents_spread_too(self):
        wobbly_parent = node("n0001", "n0000", "accepted", dev=0.90, spread=0.20)
        child = node("n0002", "n0001", "accepted", dev=0.95, spread=0.01)  # +0.05 against a 0.20 parent spread
        self.assertEqual([d["node_ids"] for d in drafts(wobbly_parent, child)], [["n0001"]])

    def test_rejection_that_only_missed_the_noise_line_is_not_a_negative_experience(self):
        inside = [
            node("n0001", "n0000", "rejected", dev=0.40, hypothesis="use collections.Counter"),
            node("n0002", "n0000", "rejected", dev=0.405, hypothesis="faster by 0.005"),
            node("n0003", "n0000", "rejected", dev=0.395, hypothesis="slower by 0.005"),
            node("n0004", "n0000", "rejected", dev=0.31, spread=0.10, hypothesis="noisy score"),
        ]
        self.assertEqual(drafts(*inside), [])

    def test_rejection_clearly_below_the_parent_is_a_negative_experience(self):
        [draft] = drafts(node("n0001", "n0000", "rejected", dev=0.30, hypothesis="丢弃全部类别特征"))
        self.assertIs(draft["negative"], True)
        self.assertEqual(draft["node_ids"], ["n0001"])
        self.assertIn("丢弃全部类别特征", draft["action"])
        self.assertIn("下降 0.1000", draft["action"])

    def test_the_drop_is_measured_from_the_parent_not_from_the_best_node(self):
        best = node("n0001", "n0000", "accepted", dev=0.80)
        branch = node("n0002", "n0000", "rejected", dev=0.395)  # far below the best, inside the noise of n0000
        self.assertEqual([d["node_ids"] for d in drafts(best, branch)], [["n0001"]])

    def test_cheat_flags_on_a_rejected_node_are_a_negative_experience_with_the_flags(self):
        [draft] = drafts(node("n0001", "n0000", "rejected", dev=0.20,
                              flags=["invalid_probability", "behavior_change"]))
        self.assertIs(draft["negative"], True)
        self.assertIn("invalid_probability, behavior_change", draft["action"])
        self.assertNotIn("下降", draft["action"])  # one lesson, not two

    def test_failure_causes_that_name_a_concrete_category(self):
        cases = {
            "Scoring timed out": ("timeout", "超时"),
            "Candidate exceeded wall time": ("exception", "超时"),
            "Scoring exceeded memory": ("exception", "内存超限"),
            "Candidate exceeded memory limit": ("exception", "内存超限"),
            "Correctness check failed": ("exception", "不一致"),
        }
        for index, (summary, (kind, label)) in enumerate(cases.items(), 1):
            with self.subTest(summary=summary):
                [draft] = drafts(failed(f"n{index:04d}", summary, kind=kind, attempts=3))
                self.assertIs(draft["negative"], True)
                self.assertIn(label, draft["action"])
                self.assertIn("尝试 3 次", draft["action"])

    def test_failures_that_say_nothing_about_the_idea_are_skipped(self):
        skipped = [
            failed("n0001", "bad candidate"),
            failed("n0002", "Traceback ... KeyError: 'x'"),
            failed("n0003", "Executor output failed: invalid JSON"),
            failed("n0004", "Dev scoring failed: harness crashed"),
            failed("n0005", "Model call blocked before dispatch by budget reserve", kind="budget"),
            failed("n0006", "Correctness check failed", kind="budget"),
            node("n0007", "n0000", "queued"),
            node("n0008", "n0000", "running"),
            node("n0009", "n0000", "succeeded", dev=0.9),
        ]
        self.assertEqual(drafts(*skipped), [])

    def test_saved_score_gives_a_failed_node_its_flags_or_error_kind(self):
        wrong, broke = "Correctness check failed", "scoring blew up"
        with TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run-x"
            nodes = [failed("n0001", wrong, attempts=3), failed("n0002", wrong, attempts=3),
                     failed("n0003", broke, attempts=3), failed("n0004", broke, attempts=3),
                     failed("n0005", broke, attempts=3), failed("n0006", wrong, attempts=3)]
            saved = {
                "n0001": {"ok": True, "correct": False, "cheat_flags": ["behavior_change"], "error": None},
                "n0002": {"ok": True, "correct": False, "cheat_flags": [], "error": None},
                "n0003": {"ok": False, "correct": False, "cheat_flags": [], "error": {"kind": "oom"}},
                "n0004": {"ok": False, "correct": False, "cheat_flags": [], "error": {"kind": "exception"}},
            }
            for node_id, data in saved.items():
                path = run_dir / "tasks/tab-sample/nodes" / node_id / "score.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"schema": "score/1", **data}), encoding="utf-8")
            for node_id in ("n0005", "n0006"):  # unreadable score file
                path = run_dir / "tasks/tab-sample/nodes" / node_id / "score.json"
                path.parent.mkdir(parents=True)
                path.write_text("{not json", encoding="utf-8")
            tree = SimpleNamespace(run_dir=run_dir, task_id="tab-sample", all_nodes=lambda: [ROOT_NODE, *nodes])
            details = read_attempt_details(tree, nodes)
            self.assertEqual(sorted(details), ["n0001", "n0002", "n0003", "n0004"])
            everything = build_drafts(nodes + [ROOT_NODE], scenario="tabular", features=FEATURES,
                                      attempt_details=details, max_negative=10)
            actions = {d["node_ids"][0]: d["action"] for d in everything}
            self.assertEqual(sorted(actions), ["n0001", "n0002", "n0003", "n0006"])
            self.assertEqual(drafts_from_tree(tree, scenario="tabular", features=FEATURES),
                             build_drafts([ROOT_NODE, *nodes], scenario="tabular", features=FEATURES,
                                          attempt_details=details))
            self.assertIn("behavior_change", actions["n0001"])
            self.assertIn("不一致", actions["n0002"])
            self.assertIn("内存超限", actions["n0003"])
            self.assertIn("不一致", actions["n0006"])  # no saved score, but the summary names the cause

    def test_scoring_breakage_is_not_mistaken_for_wrong_output(self):
        # score/1 says correct=false whenever scoring itself broke; only ok=true means "ran, but wrong".
        [draft] = drafts(failed("n0001", "scoring blew up"),
                         attempt_details={"n0001": {"ok": True, "correct": False, "cheat_flags": []}})
        self.assertIn("不一致", draft["action"])
        self.assertEqual(drafts(failed("n0001", "scoring blew up"),
                                attempt_details={"n0001": {"ok": False, "correct": False, "cheat_flags": [],
                                                           "error_kind": "exception"}}), [])

    def test_per_task_caps_keep_the_strongest_evidence(self):
        wins = [node(f"n{i:04d}", "n0000", "accepted", dev=0.40 + 0.01 * i) for i in range(1, 6)]
        losses = [node(f"n{i:04d}", "n0000", "rejected", dev=0.40 - 0.02 * (i - 5)) for i in range(6, 11)]
        result = drafts(*wins, *losses)
        positives = [d["node_ids"][0] for d in result if not d["negative"]]
        negatives = [d["node_ids"][0] for d in result if d["negative"]]
        self.assertEqual(positives, ["n0005", "n0004", "n0003"])
        self.assertEqual(negatives, ["n0010", "n0009", "n0008"])
        self.assertEqual((MAX_POSITIVE, MAX_NEGATIVE), (3, 3))
        self.assertEqual(len(drafts(*wins, *losses, max_positive=1, max_negative=0)), 1)
        self.assertEqual([d["negative"] for d in result], [False] * 3 + [True] * 3)

    def test_without_profile_features_nothing_is_written(self):
        good = node("n0001", "n0000", "accepted", dev=0.9)
        self.assertEqual(drafts(good, features=[]), [])
        self.assertEqual(drafts(good, features=["small_sample", "small_sample"])[0]["triggers"], ["small_sample"])

    def test_nodes_without_a_usable_parent_or_score_are_ignored(self):
        orphan = node("n0001", "n9999", "accepted", dev=0.9)
        no_score = node("n0002", "n0000", "accepted", dev=None)
        blank = node("n0003", "n0000", "accepted", dev=0.9, hypothesis="   ")
        self.assertEqual(drafts(orphan, no_score, blank), [])


class LibraryValidationTests(unittest.TestCase):
    def test_every_draft_kind_passes_the_library_checks(self):
        nodes = [
            node("n0001", "n0000", "accepted", dev=0.50),
            node("n0002", "n0000", "rejected", dev=0.30),
            node("n0003", "n0000", "rejected", dev=0.20, flags=["invalid_probability"]),
            failed("n0004", "Scoring timed out", kind="timeout", attempts=3),
        ]
        result = drafts(*nodes)
        self.assertEqual(len(result), 4)
        with TemporaryDirectory() as temp:
            root = Path(temp) / "runs/r1/library"
            library = Library.open(root)
            source = {**SOURCE, "node_ids": ["n0001"]}
            ids = library.write_experiences(
                [{key: value for key, value in draft.items() if key != "node_ids"} for draft in result],
                source=source)
            self.assertEqual(ids, ["exp-0001", "exp-0002", "exp-0003", "exp-0004"])
            reopened = Library.open(root)  # re-validates every stored entry
            self.assertEqual([e["negative"] for e in reopened.entries], [False, True, True, True])
            brief = reopened.brief(reopened.query("tabular", ["small_sample"]))
            self.assertEqual(brief.count("避免："), 3)
            self.assertEqual(brief.count("可试："), 1)

    def test_speedup_drafts_use_speedup_triggers(self):
        result = build_drafts([ROOT_NODE, node("n0001", "n0000", "accepted", dev=0.9)],
                              scenario="speedup", features=["nested_python_loop"])
        with TemporaryDirectory() as temp:
            library = Library.open(Path(temp) / "runs/r1/library")
            library.write_experiences([{k: v for k, v in result[0].items() if k != "node_ids"}],
                                      source={**SOURCE, "node_ids": ["n0001"]})
            self.assertEqual(library.get("exp-0001")["scenario"], "speedup")


class NoSummarizer(FakeRoles):
    def summarize(self, payload, *, node_id=None):
        raise AssertionError("script mode must not call the summarizer")


class NoiseHarness(FakeHarness):
    """Scores by hypothesis text so each test chooses what the tree will contain."""

    SCORES = {"worse": score(0.02), "cheater": score(0.0, correct=False, flags=["behavior_change"]),
              "flat": score(0.10), "up": score(0.105), "down": score(0.095)}

    def score(self, task_id, node_dir, split, seed=None):
        code = (node_dir / "solution.py").read_text(encoding="utf-8").strip()
        return dict(self.SCORES.get(code, score(0.4)))


class RunnerScriptModeTests(unittest.TestCase):
    def run_task(self, root, ideas, *, harness=None, roles_class=NoSummarizer, config=SCRIPT, library=None,
                 **roles_kwargs):
        ledger = FakeLedger()
        library = library or Library.open(root / "runs/run-test/library")
        runner = TaskRunner(
            run_dir=root / "runs/run-test", task_id="tab-sample", harness=harness or NoiseHarness(root),
            ledger=ledger, roles=roles_class(ledger, ideas, **roles_kwargs), library=library,
            baseline_score=score(0.1), config=config, hard_caps=CAPS)
        return runner.run(), ledger, library

    def test_gain_is_written_and_noise_is_not_without_any_summarizer_tokens(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            outcome, ledger, library = self.run_task(root, [idea("good", 0.3), idea("weak", 0.1)],
                                                     harness=FakeHarness(root))
            self.assertEqual((outcome.reason, outcome.best_node, outcome.experiments), ("planner", "n0001", 2))
            [entry] = library.entries
            self.assertEqual((entry["negative"], entry["scenario"], entry["triggers"]),
                             (False, "tabular", ["missing_heavy"]))
            self.assertEqual(entry["sources"], [{**SOURCE, "run_id": "run-test", "node_ids": ["n0001"]}])
            self.assertIn("good", entry["action"])
            self.assertNotIn("weak", json.dumps(library.entries, ensure_ascii=False))
            self.assertAlmostEqual(ledger.money, 0.04)  # two plans and two executions, nothing for a summary
            written = [e for e in ledger.events if e["kind"] == "library" and e["op"] == "write"]
            self.assertEqual([e["entry_ids"] for e in written], [[entry["id"]]])
            tree = {n["node_id"]: n for n in read_jsonl(
                root / "runs/run-test/tasks/tab-sample/tree.jsonl", "node/1")}
            self.assertEqual(tree["n0002"]["status"], "rejected")
            self.assertFalse((root / "runs/run-test/tasks/tab-sample/experience_rejections.jsonl").exists())
            self.assertEqual(Library.open(root / "runs/run-test/library").entries, library.entries)

    def test_noise_level_rejections_never_become_negative_experiences(self):
        counter = "use collections.Counter instead of a dict loop"  # the lesson the old summarizer got wrong
        ideas = [idea(counter, 0.3), idea("flat", 0.2), idea("up", 0.2), idea("down", 0.1)]
        harness_scores = dict(NoiseHarness.SCORES)
        harness_scores[counter] = score(0.10, spread=0.02)

        class Harness(NoiseHarness):
            SCORES = harness_scores

        with TemporaryDirectory() as temp:
            root = Path(temp)
            outcome, ledger, library = self.run_task(root, ideas, harness=Harness(root))
            self.assertEqual(outcome.experiments, 4)
            self.assertEqual(outcome.best_node, "n0000")
            self.assertEqual(library.entries, ())
            self.assertEqual(library.version, "lib-0")
            self.assertFalse(any(e["kind"] == "library" and e["op"] == "write" for e in ledger.events))

    def test_clear_loss_and_flagged_failure_become_negative_experiences_that_validate(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ideas = [idea("worse", 0.3), idea("cheater", 0.2), idea("good", 0.1)]
            outcome, ledger, library = self.run_task(root, ideas)
            by_node = {entry["sources"][0]["node_ids"][0]: entry for entry in library.entries}
            self.assertEqual(sorted(by_node), ["n0001", "n0002", "n0003"])
            worse, cheater = by_node["n0001"], by_node["n0002"]
            self.assertTrue(worse["negative"] and "worse" in worse["action"] and "下降" in worse["action"])
            self.assertTrue(cheater["negative"] and "behavior_change" in cheater["action"])
            self.assertFalse(by_node["n0003"]["negative"])
            reopened = Library.open(root / "runs/run-test/library")
            self.assertEqual(reopened.entries, library.entries)
            brief = reopened.brief(reopened.query("tabular", ["missing_heavy"]))
            self.assertEqual(brief.count("避免："), 2)
            self.assertFalse((root / "runs/run-test/tasks/tab-sample/experience_rejections.jsonl").exists())

    def test_generic_failure_is_skipped_but_the_rest_of_the_task_is_still_written(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            outcome, _, library = self.run_task(
                root, [idea("broken")], harness=FakeHarness(root, failures=True), fail_first=True)
            self.assertEqual(outcome.best_node, "n0002")  # the diagnoser's alternative won
            self.assertEqual([e["negative"] for e in library.entries], [False])
            self.assertEqual(library.entries[0]["sources"][0]["node_ids"], ["n0002"])

    def test_frozen_library_only_receives_the_drafts_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            lib_root = root / "runs/run-test/library"
            Library.open(lib_root).write_experiences(
                [{"scope": "domain", "scenario": "tabular", "triggers": ["missing_heavy"], "conditions": "",
                  "action": "Older lesson", "negative": False}],
                source={"run_id": "older", "task_id": "tab-old", "node_ids": ["n0001"]})
            outcome, ledger, frozen = self.run_task(
                root, [idea("good", 0.3), idea("worse", 0.2)], library=Library.open(lib_root, frozen=True))
            self.assertEqual(Library.open(lib_root).version, "lib-1")
            self.assertEqual(frozen.version, "lib-1")
            [row] = read_jsonl(root / "runs/run-test/tasks/tab-sample/experience_drafts.jsonl",
                               "experience_drafts/1")
            self.assertEqual([(d["negative"], d["node_ids"]) for d in row["drafts"]],
                             [(False, ["n0001"]), (True, ["n0002"])])
            self.assertEqual(row["narrowings"], [])
            self.assertTrue(any(e["kind"] == "library" and e["op"] == "freeze" for e in ledger.events))
            self.assertFalse(any(e["kind"] == "library" and e["op"] == "write" for e in ledger.events))

    def test_drafts_cite_only_real_nodes_of_this_task(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, library = self.run_task(root, [idea("good", 0.3), idea("worse", 0.2)])
            runner_nodes = {n["node_id"] for n in read_jsonl(
                root / "runs/run-test/tasks/tab-sample/tree.jsonl", "node/1")}
            for entry in library.entries:
                for source in entry["sources"]:
                    self.assertTrue(set(source["node_ids"]) <= runner_nodes - {"n0000"})
                    self.assertEqual(source["task_id"], "tab-sample")
                self.assertTrue(set(entry["triggers"]) <= {"missing_heavy"})

    def test_model_writer_keeps_calling_the_summarizer(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = FakeLedger()
            library = Library.open(root / "runs/run-test/library")
            roles = FakeRoles(ledger, [idea("good", 0.3)])
            TaskRunner(run_dir=root / "runs/run-test", task_id="tab-sample", harness=FakeHarness(root),
                       ledger=ledger, roles=roles, library=library, baseline_score=score(0.1),
                       config={**CONFIG, "experience_writer": "model"}, hard_caps=CAPS).run()
            self.assertEqual([e["action"] for e in library.entries], ["Check missing values"])

    def test_unknown_writer_fails_closed_without_losing_the_stop_event(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError) as caught:
                self.run_task(root, [idea("good", 0.3)], config={**CONFIG, "experience_writer": "bogus"})
            self.assertIsInstance(caught.exception.__cause__, ValueError)
            self.assertIn("experience_writer", str(caught.exception.__cause__))


class ScriptModeIgnoresTheReserveGateTests(unittest.TestCase):
    def test_experiences_are_written_after_spending_30000_tokens_with_no_summary_call(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            gateway, ledger = ScriptedGateway(), TokenLedger()
            model = ModelClient("m", ledger, client=gateway, clock=lambda: 1.0, max_output_tokens=2048,
                                budget_limit=40_000, reserve_cost=12_000)
            library = Library.open(root / "runs/run-test/library")
            TaskRunner(run_dir=root / "runs/run-test", task_id="tab-sample",
                       harness=FakeHarness(root, money=40_000), ledger=ledger,
                       roles=RoleAgents(model, SCRIPT, project_root=ROOT), library=library,
                       baseline_score=score(0.1), config=SCRIPT, hard_caps=CAPS).run()
            names = [request["text"]["format"]["name"] for request in gateway.requests]
            self.assertNotIn("summarizer_summarize", names)
            self.assertEqual(ledger.money, 30_000)
            self.assertEqual([e["sources"][0]["node_ids"] for e in library.entries], [["n0001"]])

    def test_experiences_are_still_written_when_the_reserve_gate_stops_the_loop(self):
        class Expensive(ScriptedGateway):
            EXECUTOR_TOKENS = 14_000  # 3,000 + 2 x 14,000 = 31,000; the next plan cannot reserve 12,000

        with TemporaryDirectory() as temp:
            root = Path(temp)
            gateway, ledger = Expensive(), TokenLedger()
            model = ModelClient("m", ledger, client=gateway, clock=lambda: 1.0, max_output_tokens=2048,
                                budget_limit=40_000, reserve_cost=12_000)
            library = Library.open(root / "runs/run-test/library")
            outcome = TaskRunner(run_dir=root / "runs/run-test", task_id="tab-sample",
                                 harness=FakeHarness(root, money=40_000), ledger=ledger,
                                 roles=RoleAgents(model, SCRIPT, project_root=ROOT), library=library,
                                 baseline_score=score(0.1), config=SCRIPT, hard_caps=CAPS).run()
            names = [request["text"]["format"]["name"] for request in gateway.requests]
            self.assertEqual(outcome.reason, "budget_cap")
            self.assertNotIn("summarizer_summarize", names)
            self.assertEqual(ledger.money, 31_000)
            self.assertEqual([e["sources"][0]["node_ids"] for e in library.entries], [["n0001"]])
            stops = [e for e in ledger.events if e["kind"] == "stop"]
            self.assertEqual([e["reason"] for e in stops], ["budget_cap"])


if __name__ == "__main__":
    unittest.main()
