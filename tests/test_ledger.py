from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ledger import Ledger, Usage
from evaluation.quote import make_quote_callback, record_quote


def fixture(root: Path, rates: dict | None = None) -> Path:
    config = root / "config"
    config.mkdir()
    prices = {
        "schema": "prices/1", "currency": "TEST", "unit_tokens": 1000,
        "models": {"test-scripted": rates or {
            "input": 2, "cached_input": 1, "output": 4, "reasoning": 6,
        }},
    }
    price_path = config / "prices_test_only.json"
    price_path.write_text(json.dumps(prices), encoding="utf-8")
    run_dir = root / "runs" / "run-test"
    run_dir.mkdir(parents=True)
    manifest = {
        "schema": "run/1", "run_id": run_dir.name, "model": "test-scripted",
        "tasks": ["spd-demo"],
        "prices": {"path": "config/prices_test_only.json",
                   "sha256": hashlib.sha256(price_path.read_bytes()).hexdigest()},
        "started_ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


class LedgerTests(unittest.TestCase):
    def test_four_exclusive_buckets_and_elapsed(self):
        with TemporaryDirectory() as temp:
            run = fixture(Path(temp))
            ledger = Ledger.open(run, "spd-demo")
            ledger.emit("task_start", budget=1.0, wall_seconds=60)
            cost = ledger.llm_call(
                node_id="n0001", role="executor", purpose="code", effort="low",
                model="test-scripted", usage=Usage(100, 20, 30, 5), seconds=1.25,
                prompt_hash="a" * 64,
            )
            self.assertAlmostEqual(cost, 0.37)
            ledger.emit("experiment", node_id="n0001", attempt=1, seconds_run=0.5, exit_code=0, timed_out=False, oom=False)
            ledger.emit("score", node_id="n0001", split="dev", score=1.2, raw=3.0, spread=0.1, correct=True, cheat_flags=[], seconds_queue=0.3)
            reopened = Ledger.open(run, "spd-demo")
            self.assertAlmostEqual(reopened.spent(), 0.37)
            elapsed = reopened.elapsed()
            self.assertEqual((elapsed["llm"], elapsed["run"], elapsed["queue"]), (1.25, 0.5, 0.3))
            self.assertGreaterEqual(elapsed["wall"], 0)
            lines = [json.loads(line) for line in reopened.path.read_text().splitlines()]
            self.assertEqual([line["seq"] for line in lines], [1, 2, 3, 4])
            self.assertEqual(lines[1]["tokens"], {"input": 100, "cached_input": 20, "output": 30, "reasoning": 5})

    def test_missing_price_fails_without_booking_zero(self):
        with TemporaryDirectory() as temp:
            run = fixture(Path(temp), {"input": 1, "cached_input": 1, "output": None, "reasoning": 1})
            ledger = Ledger.open(run, "spd-demo")
            with self.assertRaisesRegex(ValueError, "Missing output price"):
                ledger.llm_call(node_id=None, role="planner", purpose="plan", effort="high", model="test-scripted", usage=Usage(2), seconds=1, prompt_hash="a" * 64)
            self.assertEqual(ledger.spent(), 0)
            self.assertFalse(ledger.path.exists())

    def test_sealed_price_and_identity(self):
        with TemporaryDirectory() as temp:
            run = fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "Invalid task_id"):
                Ledger.open(run, "../escape")
            prices = Path(temp) / "config/prices_test_only.json"
            prices.write_text(prices.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after manifest"):
                Ledger.open(run, "spd-demo")

    def test_append_from_two_instances_keeps_local_sequence(self):
        with TemporaryDirectory() as temp:
            run = fixture(Path(temp))
            first = Ledger.open(run, "spd-demo")
            second = Ledger.open(run, "spd-demo")
            first.emit("task_start", budget=1.0, wall_seconds=60)
            second.emit("stop", reason="planner", best_node="n0000", dev_score=0.0)
            run_level = Ledger.open(run, None)
            run_level.emit("run_start")
            self.assertEqual([e["seq"] for e in first._events()], [1, 2])
            self.assertEqual([e["seq"] for e in run_level._events()], [1])

    def test_quote_precedes_experiment_and_corrects_from_real_history(self):
        with TemporaryDirectory() as temp:
            run = fixture(Path(temp))
            ledger = Ledger.open(run, "spd-demo")
            ledger.emit("task_start", budget=1, wall_seconds=60)
            callback = make_quote_callback(ledger, history=[
                {"predicted": {"cost": 2, "seconds": 10, "score": 0.5},
                 "actual": {"cost": 3, "seconds": 20, "score": 0.6}},
            ])
            callback({"predicted": {"cost": 4, "seconds": 5, "score": 0.7}})
            quote = ledger._events()[1]
            self.assertEqual(quote["predicted"]["cost"], 6.0)
            self.assertEqual(quote["predicted"]["seconds"], 10.0)
            self.assertAlmostEqual(quote["predicted"]["score"], 0.8)
            self.assertEqual(quote["basis"]["history_n"], 1)
            ledger.emit("experiment", attempt=1, seconds_run=0.5, exit_code=0, timed_out=False, oom=False)
            with self.assertRaisesRegex(ValueError, "before the first experiment"):
                record_quote(ledger, {"predicted": {"cost": 1, "seconds": 1, "score": 1}})


if __name__ == "__main__":
    unittest.main()
