from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from scripts.aggregate_naive_costs import MANIFEST_KEYS, aggregate, main, sha


def put(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((json.dumps(value) if isinstance(value, dict) else value) + "\n",
                    encoding="utf-8")


class AggregateNaiveCostsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.ids = {"speedup": [f"spd-{i:02d}" for i in range(12)],
                    "tabular": [f"tab-{i:02d}" for i in range(12)]}
        bank = {"schema": "bank/1", "version": "bank-test", "frozen": False,
                "tasks": [{"task_id": task_id} for group in self.ids.values()
                          for task_id in group]}
        put(self.root / "tasks/bank.json", bank)
        for scenario, ids in self.ids.items():
            for task_id in ids:
                task = self.root / "tasks" / task_id
                put(task / "task.json", {"task_id": task_id, "scenario": scenario,
                                         "budget": {"money": None, "wall_seconds": 1800}})
                put(task / "workspace/original.py", "def solution(): return 1")
                hidden = "reference.py" if scenario == "speedup" else "dev.csv"
                put(task / "hidden" / hidden, "sample")
        for name in ("naive_token_pilot.json", "hard_caps.json", "triggers.json"):
            put(self.root / "config" / name, {"test": True})
        put(self.root / "config/prompts/naive_token_pilot.md", "one prompt")
        put(self.root / "config/prices_token_pilot.json", {
            "currency": "TOKENS", "test_only": True, "unit_tokens": 1,
            "models": {"qwen3.5-27b": {key: 1 for key in
                                    ("input", "cached_input", "output", "reasoning")}}})
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.git("add", ".")
        self.git("commit", "-qm", "source")
        self.commit_a = self.git("rev-parse", "HEAD").strip()
        self.campaigns = {name: self.make_campaign(name)
                          for name in ("speedup", "tabular")}

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.root), *args],
                                       text=True)

    def make_campaign(self, scenario: str) -> Path:
        campaign = Path(self.temp.name) / f"campaign-{scenario}"
        project = campaign / "project"
        shutil.copytree(self.root / "config", project / "config")
        source_bank = json.loads((self.root / "tasks/bank.json").read_text())
        source_bank["tasks"] = [row for row in source_bank["tasks"]
                                if row["task_id"] in self.ids[scenario]]
        put(project / "tasks/bank.json", source_bank)
        source_hashes = {"tasks/bank.json": sha(self.root / "tasks/bank.json")}
        copied_hashes = {"tasks/bank.json": sha(project / "tasks/bank.json")}
        for path in (self.root / "config").rglob("*"):
            if path.is_file():
                key = str(path.relative_to(self.root))
                source_hashes[key] = sha(path)
                copied_hashes[key] = sha(project / key)
        for task_id in self.ids[scenario]:
            source = self.root / "tasks" / task_id
            copy = project / "tasks" / task_id
            shutil.copytree(source, copy)
            task = json.loads((copy / "task.json").read_text())
            task["budget"]["money"] = 40000
            put(copy / "task.json", task)
            for path in source.rglob("*"):
                if path.is_file():
                    key = str(path.relative_to(self.root))
                    source_hashes[key] = sha(path)
                    copied_hashes[key] = sha(project / key)
        protocol = {"schema": "naive_token_calibration_protocol/1",
                    "label": "DEV ONLY; TEST TOKEN UNITS; NO HOLDOUT OR MONEY",
                    "bank_version": "bank-test", "tasks": self.ids[scenario],
                    "model": "qwen3.5-27b", "repetitions": 3,
                    "budget_tokens_per_task": 40000,
                    "reserve_tokens_per_call": 12000,
                    "max_output_tokens_requested": 2048,
                    "code_commit": self.commit_a,
                    "source_sha256": source_hashes,
                    "copied_sha256": copied_hashes}
        put(campaign / "protocol.json", protocol)
        attempts = []
        offset = 0 if scenario == "speedup" else 36
        for index, task_id in enumerate(self.ids[scenario]):
            for rep in (1, 2, 3):
                run_id = f"run-20261114-000000-{offset + index * 3 + rep:04x}"
                run = project / "runs" / run_id
                manifest = {"schema": "run/1", "run_id": run_id,
                            "phase": "dev", "config": "naive",
                            "tasks": [task_id], "rep": rep,
                            "model": "qwen3.5-27b", "code_commit": self.commit_a,
                            "bank_version": "bank-test",
                            "agent_config": {"sha256": copied_hashes["config/naive_token_pilot.json"]},
                            "prices": {"sha256": copied_hashes["config/prices_token_pilot.json"]},
                            "library": {"start_version": "lib-0", "frozen": False}}
                put(run / "manifest.json", manifest)
                task_root = run / "tasks" / task_id
                tokens = (index + 1) * 10
                events = [
                    {"schema": "ledger/1", "seq": 1, "run_id": run_id,
                     "task_id": task_id, "kind": "task_start", "budget": 40000},
                    {"schema": "ledger/1", "seq": 2, "run_id": run_id,
                     "task_id": task_id, "kind": "llm_call", "model": "qwen3.5-27b",
                     "tokens": {"input": tokens, "cached_input": 0,
                                "output": 0, "reasoning": 0}, "cost": float(tokens)},
                    {"schema": "ledger/1", "seq": 3, "run_id": run_id,
                     "task_id": task_id, "kind": "stop", "reason": "ceiling",
                     "best_node": "n0000", "dev_score": 1.0},
                ]
                put(task_root / "ledger.jsonl", "\n".join(json.dumps(row) for row in events))
                node_dir = task_root / "nodes/n0000"
                put(node_dir / "solution.py", "def solution(): return 1")
                put(node_dir / "score.json", {"schema": "score/1", "split": "dev",
                                              "score": 1.0})
                snapshot = run / "library/snapshots/lib-0.json"
                put(snapshot, {"schema": "library_snapshot/1"})
                tree = {"schema": "node/1", "run_id": run_id, "task_id": task_id,
                        "node_id": "n0000", "status": "accepted",
                        "score": {"dev": 1.0}, "library_version": "lib-0"}
                put(task_root / "tree.jsonl", json.dumps(tree))
                put(task_root / "replay_context.json", {
                    "schema": "a_replay_context/1", "run_id": run_id,
                    "task_id": task_id, "model": "qwen3.5-27b",
                    "max_output_tokens": 2048,
                    "manifest_fields": {key: manifest.get(key) for key in MANIFEST_KEYS},
                    "tree_sha256": sha(task_root / "tree.jsonl"),
                    "nodes": {"n0000": {
                        "library_version": "lib-0",
                        "library_snapshot_sha256": sha(snapshot),
                        "solution_sha256": sha(node_dir / "solution.py"),
                        "score_sha256": sha(node_dir / "score.json")}}})
                attempts.append({"task_id": task_id, "rep": rep,
                                 "run_dir": "/remote-machine/" + run_id,
                                 "status": "completed", "stop_reason": "ceiling",
                                 "total_tokens": tokens,
                                 "token_buckets": {"input": tokens, "cached_input": 0,
                                                   "output": 0, "reasoning": 0},
                                 "model_calls": 1, "budget_respected": True,
                                 "dev_score": 1.0, "best_node": "n0000"})
        put(campaign / "summary.json", {
            "schema": "naive_token_calibration_summary/1",
            "label": protocol["label"], "ended_utc": "2026-11-14T00:00:00Z",
            "attempts": attempts})
        return campaign

    def change_attempt(self, scenario: str, task_id: str, rep: int,
                       *, status: str, tokens: int, reason: str) -> None:
        campaign = self.campaigns[scenario]
        path = campaign / "summary.json"
        summary = json.loads(path.read_text())
        record = next(row for row in summary["attempts"]
                      if row["task_id"] == task_id and row["rep"] == rep)
        record.update(status=status, total_tokens=tokens, stop_reason=reason,
                      token_buckets={"input": tokens, "cached_input": 0,
                                     "output": 0, "reasoning": 0})
        put(path, summary)
        run = campaign / "project/runs" / Path(record["run_dir"]).name
        ledger = run / "tasks" / task_id / "ledger.jsonl"
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        events[1]["tokens"]["input"] = tokens
        events[1]["cost"] = float(tokens)
        events[2]["reason"] = reason
        put(ledger, "\n".join(json.dumps(row) for row in events))

    def test_ready_proposal_and_failure_adjusted_cost(self):
        self.change_attempt("speedup", "spd-00", 3,
                            status="incomplete", tokens=1, reason="error")
        result = aggregate(self.root, self.campaigns["speedup"],
                           self.campaigns["tabular"])
        self.assertTrue(result["ready"])
        self.assertEqual(len(result["tasks"]), 24)
        self.assertEqual(result["tasks"][0]["naive_cost"], 10.5)
        self.assertEqual(len(result["bank_proposal"]["updates"]), 24)
        for scenario in self.ids:
            self.assertEqual([sum(row["scenario"] == scenario and
                                  row["difficulty"] == difficulty
                                  for row in result["tasks"])
                              for difficulty in ("easy", "medium", "hard")],
                             [4, 4, 4])
        self.assertIsNone(json.loads((self.root / "tasks/bank.json").read_text())
                          ["tasks"][0].get("naive_cost"))

    def test_fewer_than_two_completions_blocks_proposal(self):
        for rep in (2, 3):
            self.change_attempt("tabular", "tab-00", rep,
                                status="incomplete", tokens=1, reason="error")
        result = aggregate(self.root, self.campaigns["speedup"],
                           self.campaigns["tabular"])
        self.assertFalse(result["ready"])
        self.assertIsNone(result["bank_proposal"])
        self.assertEqual(result["blocked"][0]["task_id"], "tab-00")

    def test_holdout_or_ledger_tamper_is_rejected(self):
        campaign = self.campaigns["speedup"]
        record = json.loads((campaign / "summary.json").read_text())["attempts"][0]
        run = campaign / "project/runs" / Path(record["run_dir"]).name
        holdout = run / "tasks" / record["task_id"] / "holdout_attempt.json"
        put(holdout, {"bad": True})
        with self.assertRaisesRegex(ValueError, "Holdout opened"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        holdout.unlink()
        ledger = run / "tasks" / record["task_id"] / "ledger.jsonl"
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        events[1]["cost"] += 1
        put(ledger, "\n".join(json.dumps(row) for row in events))
        with self.assertRaisesRegex(ValueError, "Four-bucket/price mismatch"):
            aggregate(self.root, campaign, self.campaigns["tabular"])

    def test_source_hash_and_unfinished_campaign_are_rejected(self):
        campaign = self.campaigns["tabular"]
        summary_path = campaign / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["ended_utc"] = None
        put(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            aggregate(self.root, self.campaigns["speedup"], campaign)
        summary["ended_utc"] = "2026-11-14T00:00:00Z"
        put(summary_path, summary)
        put(campaign / "project/tasks/tab-00/hidden/dev.csv", "changed")
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            aggregate(self.root, self.campaigns["speedup"], campaign)

    def test_rehashed_copy_changes_and_extra_workspace_files_are_rejected(self):
        campaign = self.campaigns["tabular"]
        for relative, replacement, expected in (
            ("tasks/tab-00/workspace/original.py", "modified code", "source/copy content"),
            ("tasks/tab-00/hidden/dev.csv", "modified dev rows", "source/copy content"),
            ("tasks/tab-00/task.json", {"task_id": "tab-00", "scenario": "tabular",
                                         "budget": {"money": 40000, "wall_seconds": 999}},
             "copied task differs"),
            ("tasks/bank.json", {"schema": "bank/1", "version": "bank-test",
                                 "frozen": False,
                                 "tasks": [{"task_id": task_id, "tier": "changed"}
                                           for task_id in self.ids["tabular"]]},
             "copied bank differs"),
            ("tasks/tab-00/workspace/extra.py", "unexpected file", "workspace file set"),
        ):
            with self.subTest(relative=relative):
                path = campaign / "project" / relative
                before = path.read_bytes() if path.exists() else None
                protocol_path = campaign / "protocol.json"
                protocol = json.loads(protocol_path.read_text())
                previous_hash = protocol["copied_sha256"].get(relative)
                put(path, replacement)
                protocol["copied_sha256"][relative] = sha(path)
                put(protocol_path, protocol)
                with self.assertRaisesRegex(ValueError, expected):
                    aggregate(self.root, self.campaigns["speedup"], campaign)
                if before is None:
                    path.unlink()
                    del protocol["copied_sha256"][relative]
                else:
                    path.write_bytes(before)
                    protocol["copied_sha256"][relative] = previous_hash
                put(protocol_path, protocol)

    def test_python_reference_cache_only_is_ignored(self):
        campaign = self.campaigns["speedup"]
        hidden = campaign / "project/tasks/spd-00/hidden"
        cache = hidden / "__pycache__/reference.cpython-311.pyc"
        put(cache, "compiled reference cache")
        workspace_key = "tasks/spd-00/workspace/__pycache__/original.cpython-311.pyc"
        source_cache = self.root / workspace_key
        copied_cache = campaign / "project" / workspace_key
        put(source_cache, "source bytecode")
        put(copied_cache, "copied bytecode")
        protocol_path = campaign / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["source_sha256"][workspace_key] = sha(source_cache)
        protocol["copied_sha256"][workspace_key] = sha(copied_cache)
        put(protocol_path, protocol)
        put(copied_cache, "recompiled bytecode")
        self.assertTrue(aggregate(self.root, campaign,
                                  self.campaigns["tabular"])["ready"])
        extra = copied_cache.parent / "unrelated.txt"
        put(extra, "unexpected")
        with self.assertRaisesRegex(ValueError, "Unexpected workspace cache data"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        extra.unlink()
        extra = hidden / "__pycache__/unrelated.txt"
        put(extra, "unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected hidden data"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        extra.unlink()
        extra = hidden / "__pycache__/other.cpython-311.pyc"
        put(extra, "unrelated bytecode")
        with self.assertRaisesRegex(ValueError, "unexpected hidden data"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        extra.unlink()
        extra = hidden / "unrelated.csv"
        put(extra, "unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected hidden data"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        extra.unlink()
        reference = hidden / "reference.py"
        put(reference, "changed reference")
        protocol["copied_sha256"]["tasks/spd-00/hidden/reference.py"] = sha(reference)
        put(protocol_path, protocol)
        with self.assertRaisesRegex(ValueError, "source/copy content differs"):
            aggregate(self.root, campaign, self.campaigns["tabular"])

    def test_dev_result_and_replay_seal_tampering_are_rejected(self):
        campaign = self.campaigns["speedup"]
        summary_path = campaign / "summary.json"
        summary = json.loads(summary_path.read_text())
        record = summary["attempts"][0]
        run = campaign / "project/runs" / Path(record["run_dir"]).name
        task_dir = run / "tasks" / record["task_id"]
        record["dev_score"] = 0.5
        put(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "Summary/ledger dev result mismatch"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        record["dev_score"] = 1.0
        put(summary_path, summary)
        record["best_node"] = "n9999"
        put(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "Summary/ledger dev result mismatch"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        record["best_node"] = "n0000"
        put(summary_path, summary)
        tree_path = task_dir / "tree.jsonl"
        tree = json.loads(tree_path.read_text())
        tree["score"]["dev"] = 0.5
        put(tree_path, json.dumps(tree))
        with self.assertRaisesRegex(ValueError, "Ledger/experiment tree dev result mismatch"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        tree["score"]["dev"] = 1.0
        put(tree_path, json.dumps(tree))
        score_path = task_dir / "nodes/n0000/score.json"
        put(score_path, {"schema": "score/1", "split": "dev", "score": 0.5})
        with self.assertRaisesRegex(ValueError, "Replay seal artifact changed"):
            aggregate(self.root, campaign, self.campaigns["tabular"])
        put(score_path, {"schema": "score/1", "split": "dev", "score": 1.0})
        context_path = task_dir / "replay_context.json"
        context = json.loads(context_path.read_text())
        context["tree_sha256"] = "0" * 64
        put(context_path, context)
        with self.assertRaisesRegex(ValueError, "Replay seal/tree/manifest mismatch"):
            aggregate(self.root, campaign, self.campaigns["tabular"])

    def test_cli_writes_separate_audit_and_proposal(self):
        bank_before = (self.root / "tasks/bank.json").read_bytes()
        output = self.root / "runs/audit-v1"
        code = main(["--project-root", str(self.root),
                     "--speedup-campaign", str(self.campaigns["speedup"]),
                     "--tabular-campaign", str(self.campaigns["tabular"]),
                     "--output-dir", str(output)])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads((output / "audit.json").read_text())["ready"])
        self.assertEqual(len(json.loads((output / "bank_proposal.json").read_text())
                             ["updates"]), 24)
        self.assertEqual((self.root / "tasks/bank.json").read_bytes(), bank_before)

    def test_docs_only_commit_exception_and_code_change_rejection(self):
        put(self.root / "docs/note.md", "documentation")
        self.git("add", "docs/note.md")
        self.git("commit", "-qm", "docs")
        docs_commit = self.git("rev-parse", "HEAD").strip()
        self.set_tabular_commit(docs_commit)
        result = aggregate(self.root, self.campaigns["speedup"],
                           self.campaigns["tabular"])
        self.assertTrue(result["commit_check"]["docs_only_exception"])
        self.assertEqual(result["commit_check"]["changed_docs"], ["docs/note.md"])
        put(self.root / "scripts/new.py", "value = 1")
        self.git("add", "scripts/new.py")
        self.git("commit", "-qm", "code")
        self.set_tabular_commit(self.git("rev-parse", "HEAD").strip())
        with self.assertRaisesRegex(ValueError, "outside docs"):
            aggregate(self.root, self.campaigns["speedup"],
                      self.campaigns["tabular"])

    def test_same_nonexistent_commit_is_rejected(self):
        missing = "f" * 40
        self.set_campaign_commit("speedup", missing)
        self.set_campaign_commit("tabular", missing)
        with self.assertRaisesRegex(ValueError, "Missing local Git commit"):
            aggregate(self.root, self.campaigns["speedup"],
                      self.campaigns["tabular"])

    def set_tabular_commit(self, commit: str) -> None:
        self.set_campaign_commit("tabular", commit)

    def set_campaign_commit(self, scenario: str, commit: str) -> None:
        campaign = self.campaigns[scenario]
        protocol = json.loads((campaign / "protocol.json").read_text())
        protocol["code_commit"] = commit
        put(campaign / "protocol.json", protocol)
        for path in (campaign / "project/runs").glob("run-*/manifest.json"):
            manifest = json.loads(path.read_text())
            manifest["code_commit"] = commit
            put(path, manifest)
            task_id = manifest["tasks"][0]
            context_path = path.parent / "tasks" / task_id / "replay_context.json"
            context = json.loads(context_path.read_text())
            context["manifest_fields"]["code_commit"] = commit
            put(context_path, context)


if __name__ == "__main__":
    unittest.main()
