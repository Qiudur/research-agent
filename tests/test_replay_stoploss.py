from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.replay_stoploss import (apply_rule, consecutive_no_gain_fire, first_n_no_gain_fire,
                                     main, replay, replay_task)
from tests.ledger_fixtures import TaskWriter, write_manifest


def quiet(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(argv)


def tab_early_miss_then_gain(run_dir: Path) -> None:
    # Two rejected nodes, then two accepted ones: the later gain is what a stop would forgo.
    writer = TaskWriter(run_dir, "tab-one")
    writer.plan(1000).node("n0001", "rejected", 0.1, 2000).node("n0002", "rejected", 0.1, 2000)
    writer.plan(800).node("n0003", "accepted", 0.5, 2200).node("n0004", "accepted", 0.8, 2000)
    writer.summarize(600).stop("planner", "n0004")
    writer.finish()


def tab_gain_first_then_dry(run_dir: Path) -> None:
    writer = TaskWriter(run_dir, "tab-two")
    writer.plan(1000).node("n0001", "accepted", 0.4, 2000)
    for number in (2, 3, 4):
        writer.node(f"n000{number}", "rejected", 0.3, 2000)
    writer.stop("planner", "n0001").finish()


def spd_dry_then_reference(run_dir: Path) -> None:
    writer = TaskWriter(run_dir, "spd-one")
    writer.plan(500)
    for number in (1, 2, 3, 4):
        writer.node(f"n000{number}", "rejected", 0.2, 1000)
    writer.node("n0005", "accepted", 1.1, 1500).summarize(300).stop("ceiling", "n0005")
    writer.finish()


def build_study(root: Path) -> Path:
    base = root / "campaign/project/runs"
    run_a, run_b = base / "run-20261115-000001-aaaa", base / "run-20261115-000002-bbbb"
    write_manifest(run_a, phase="practice", tasks=["tab-one", "tab-two"])
    tab_early_miss_then_gain(run_a)
    tab_gain_first_then_dry(run_a)
    write_manifest(run_b, phase="self_research", tasks=["spd-one"])
    spd_dry_then_reference(run_b)
    # Held-out phases must never contribute to rule tuning.
    for name, phase in (("run-20261115-000003-cccc", "exam"), ("run-20261115-000004-dddd", "validation")):
        held_out = base / name
        write_manifest(held_out, phase=phase, tasks=["tab-one"])
        tab_early_miss_then_gain(held_out)
    return root / "campaign"


class RuleTests(unittest.TestCase):
    def test_first_n_needs_every_one_of_the_first_n_unaccepted(self):
        self.assertEqual(first_n_no_gain_fire([False, False, True], 2), 2)
        self.assertIsNone(first_n_no_gain_fire([False, True, False], 2))
        self.assertIsNone(first_n_no_gain_fire([False], 2))  # not enough completed nodes yet
        self.assertEqual(first_n_no_gain_fire([False, False, False], 1), 1)
        with self.assertRaises(ValueError):
            first_n_no_gain_fire([], 0)

    def test_consecutive_streak_resets_on_acceptance(self):
        self.assertEqual(consecutive_no_gain_fire([False, False, True, False, False, False], 3), 6)
        self.assertEqual(consecutive_no_gain_fire([True, False, False], 2), 3)
        self.assertIsNone(consecutive_no_gain_fire([False, True, False, True], 2))
        with self.assertRaises(ValueError):
            consecutive_no_gain_fire([], 0)


class ReplayTests(unittest.TestCase):
    def test_single_task_outcomes(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "run-20261115-000001-aaaa"
            tab_early_miss_then_gain(run)
            task = replay_task(run / "tasks/tab-one")
            self.assertEqual(task["main_loop_tokens"], 10000)  # summarizer excluded
            self.assertEqual([row["main_tokens"] for row in task["completions"]],
                             [3000, 5000, 8000, 10000])
            early = apply_rule(task, "first_n_no_gain", 2)
            self.assertEqual((early["fired_at_node"], early["tokens_saved"]), (2, 5000))
            self.assertTrue(early["later_improved"] and early["stopped_early"])
            self.assertAlmostEqual(early["forgone_dev_gain"], 0.7)
            self.assertFalse(early["later_reached_reference"])
            self.assertEqual(apply_rule(task, "first_n_no_gain", 3)["fired_at_node"], None)
            self.assertEqual(apply_rule(task, "first_n_no_gain", 1)["tokens_saved"], 7000)

    def test_firing_with_no_spend_left_is_not_counted_as_stopped_early(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "run-20261115-000001-aaaa"
            tab_gain_first_then_dry(run)
            task = replay_task(run / "tasks/tab-two")
            last = apply_rule(task, "consecutive_no_gain", 3)
            self.assertEqual(last["fired_at_node"], 4)
            self.assertFalse(last["stopped_early"])
            self.assertEqual(last["tokens_saved"], 0)
            mid = apply_rule(task, "consecutive_no_gain", 2)
            self.assertEqual((mid["fired_at_node"], mid["tokens_saved"]), (3, 2000))
            self.assertFalse(mid["later_improved"])  # nothing was accepted after the stop

    def test_unfinished_tasks_cannot_be_replayed(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "run-20261115-000001-aaaa"
            TaskWriter(run, "tab-one").plan(100).finish()  # no stop event
            self.assertIsNone(replay_task(run / "tasks/tab-one"))
            self.assertIsNone(replay_task(run / "tasks/tab-missing"))

    def test_grid_aggregates_by_scenario_and_ignores_held_out_phases(self):
        with TemporaryDirectory() as temp:
            report = replay([build_study(Path(temp))])
        self.assertEqual(report["schema"], "stoploss_replay/1")
        self.assertEqual({run["phase"] for run in report["runs"]["used"]},
                         {"practice", "self_research"})
        self.assertEqual({row["reason"] for row in report["runs"]["skipped"]},
                         {"phase=exam", "phase=validation"})
        tab, spd = report["scenarios"]["tab"], report["scenarios"]["spd"]
        self.assertEqual((tab["tasks"], tab["main_loop_tokens"]), (2, 19000))
        self.assertEqual(tab["tasks_with_an_accepted_node"], 2)
        first_two = tab["rules"]["first_n_no_gain"]["2"]
        self.assertEqual((first_two["tokens_saved"], first_two["stopped_early"],
                          first_two["later_improved"]), (5000, 1, 1))
        self.assertAlmostEqual(first_two["saved_fraction"], 5000 / 19000)
        self.assertEqual(tab["rules"]["first_n_no_gain"]["3"]["tokens_saved"], 0)
        streak_two = tab["rules"]["consecutive_no_gain"]["2"]
        self.assertEqual((streak_two["tokens_saved"], streak_two["stopped_early"],
                          streak_two["later_improved"]), (7000, 2, 1))
        self.assertEqual(tab["rules"]["consecutive_no_gain"]["3"]["stopped_early"], 0)
        self.assertEqual(sorted(tab["rules"]["first_n_no_gain"]), ["1", "2", "3", "4"])
        self.assertEqual(sorted(tab["rules"]["consecutive_no_gain"]), [str(k) for k in range(2, 9)])
        # Speedup: four dry nodes, then the node that reaches the reference.
        streak_four = spd["rules"]["consecutive_no_gain"]["4"]
        self.assertEqual((spd["tasks"], spd["main_loop_tokens"]), (1, 6000))
        self.assertEqual((streak_four["tokens_saved"], streak_four["stopped_early"],
                          streak_four["later_improved"],
                          streak_four["later_reached_reference"]), (1500, 1, 1, 1))
        self.assertAlmostEqual(streak_four["forgone_dev_gain"], 1.0)  # baseline 0.1 -> 1.1
        self.assertEqual(streak_four["improved_share_of_stopped"], 1.0)

    def test_phase_overrides_and_forbidden_phases(self):
        with TemporaryDirectory() as temp:
            root = build_study(Path(temp))
            only_practice = replay([root], phases=["practice"])
            self.assertEqual(only_practice["scenarios"]["spd"]["tasks"], 0)
            with self.assertRaisesRegex(ValueError, "may not be used"):
                replay([root], phases=["exam"])
            with self.assertRaisesRegex(ValueError, "may not be used"):
                replay([root], phases=["validation"])

    def test_completion_modes_change_savings_when_diagnoser_calls_trail_a_node(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "run-20261115-000001-aaaa"
            write_manifest(run, phase="practice", tasks=["tab-one"])
            writer = TaskWriter(run, "tab-one")
            writer.plan(1000).node("n0001", "rejected", 0.1, 2000, diag=400)
            writer.node("n0002", "rejected", 0.1, 2000).node("n0003", "accepted", 0.9, 2000)
            writer.stop("planner", "n0003").finish()
            narrow = replay([Path(temp)], completion="experiment_score")
            wide = replay([Path(temp)], completion="node_events")
        key = lambda report: report["scenarios"]["tab"]["rules"]["first_n_no_gain"]["1"]["tokens_saved"]
        self.assertEqual(key(narrow), 7400 - 3000)  # trailing diagnosis counted as saved
        self.assertEqual(key(wide), 7400 - 3400)

    def test_details_rows_are_optional(self):
        with TemporaryDirectory() as temp:
            root = build_study(Path(temp))
            self.assertNotIn("details", replay([root]))
            detailed = replay([root], details=True, first_n=[2], consecutive=[4])
        rows = [row for row in detailed["details"] if row["rule"] == "first_n_no_gain"]
        self.assertEqual(len(rows), 3)  # three tasks, one parameter
        self.assertTrue(all("run_id" in row and "task_id" in row for row in rows))

    def test_cli_writes_json_and_reports_missing_data(self):
        with TemporaryDirectory() as temp:
            root = build_study(Path(temp))
            output = Path(temp) / "out/replay.json"
            self.assertEqual(quiet([str(root), "--output", str(output),
                                    "--first-n", "2", "--consecutive", "4"]), 0)
            data = json.loads(output.read_text())
            self.assertEqual(data["grid"], {"first_n_no_gain": [2], "consecutive_no_gain": [4]})
            self.assertEqual(data["scenarios"]["tab"]["rules"]["first_n_no_gain"]["2"]["tokens_saved"],
                             5000)
            empty = Path(temp) / "empty"
            empty.mkdir()
            self.assertEqual(quiet([str(empty), "--output", str(Path(temp) / "none.json")]), 2)
            self.assertFalse((Path(temp) / "none.json").exists())
            with self.assertRaises(SystemExit):
                quiet([str(root), "--output", str(output), "--phases", "exam"])


if __name__ == "__main__":
    unittest.main()
