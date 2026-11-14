"""Synthetic saved-output fixtures; no model, sandbox or scoring calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_formal_outputs import audit_formal_outputs


COMMIT = "a" * 40
MODEL = "qwen3.5-27b"
ROLES = ("planner", "executor", "diagnoser", "summarizer")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalOutputAuditTests(unittest.TestCase):
    def fixture(self, root: Path, *, admitted: bool = True) -> tuple[Path, Path, dict]:
        root = root.resolve()
        tasks = {tier: [f"spd-{tier}-{number}" for number in range(count)]
                 for tier, count in (("practice", 10), ("validation", 4), ("exam", 10))}
        bank_path = root / "tasks/bank.json"
        write_json(bank_path, {"schema": "bank/1", "version": "bank-v1", "frozen": True,
                               "tasks": [{"task_id": task_id, "tier": tier}
                                         for tier, ids in tasks.items() for task_id in ids]})
        price_path = root / "config/prices.json"
        write_json(price_path, {"schema": "prices/1", "currency": "TOKENS",
                                "unit_tokens": 1, "test_only": False,
                                "models": {MODEL: {key: 1 for key in
                                                   ("input", "cached_input", "output", "reasoning")}}})
        price_sha = digest(price_path)

        def candidate(name: str) -> dict:
            prompts = {}
            paths = {}
            for role in ROLES:
                path = root / "config" / f"{name}-{role}.md"
                path.write_text(f"{name} {role}\n", encoding="utf-8")
                prompts[role] = {"path": str(path), "sha256": digest(path)}
                paths[role] = str(path.relative_to(root))
            path = root / "config" / f"{name}.json"
            write_json(path, {"schema": "agent/1", "prompts": paths})
            return {"config": {"path": str(path), "sha256": digest(path)},
                    "prompts": prompts}

        default = candidate("default")
        tuned = candidate("tuned")
        naive = candidate("naive")
        configs = {"default": default, "default_lib": default,
                   "tuned_lib": tuned, "naive": naive}
        next_run = 0

        def make_run(phase: str, config: str, rep: int, ids: list[str],
                     *, unit: int = 10, score: float = 0.0) -> tuple[dict, Path]:
            nonlocal next_run
            run_id = f"run-20261114-120000-{next_run:04d}"
            next_run += 1
            run = root / "runs" / run_id
            agent = configs[config]["config"]
            write_json(run / "manifest.json", {
                "schema": "run/1", "run_id": run_id, "phase": phase,
                "config": config, "rep": rep, "bank_version": "bank-v1",
                "tasks": ids, "code_commit": COMMIT, "model": MODEL,
                "agent_config": {"path": str(Path(agent["path"]).relative_to(root)),
                                 "sha256": agent["sha256"]},
                "prices": {"path": "config/prices.json", "sha256": price_sha},
            })
            task_reports = []
            for task_id in ids:
                task_dir = run / "tasks" / task_id
                tokens = {"input": unit, "cached_input": 0, "output": 0,
                          "reasoning": 0}
                rows = [
                    {"schema": "ledger/1", "seq": 1, "run_id": run_id,
                     "task_id": task_id, "kind": "task_start", "budget": 40_000},
                    {"schema": "ledger/1", "seq": 2, "run_id": run_id,
                     "task_id": task_id, "kind": "llm_call", "role": "planner",
                     "model": MODEL, "tokens": tokens, "cost": float(unit)},
                    {"schema": "ledger/1", "seq": 3, "run_id": run_id,
                     "task_id": task_id, "kind": "stop", "reason": "ceiling"},
                    {"schema": "ledger/1", "seq": 4, "run_id": run_id,
                     "task_id": task_id, "kind": "task_end", "total_cost": float(unit),
                     "holdout_score": score},
                ]
                write_jsonl(task_dir / "ledger.jsonl", rows)
                write_json(task_dir / "holdout_attempt.json",
                           {"schema": "holdout_attempt/1"})
                task_reports.append({
                    "schema": "task_report/1", "run_id": run_id,
                    "task_id": task_id, "currency": "TOKENS",
                    "holdout_score": score, "total_cost": float(unit),
                    "tokens": tokens, "model_calls": 1,
                    "cost_by_role": {role: float(unit) if role == "planner" else 0.0
                                     for role in ROLES},
                    "evidence": {"task_end_seq": 4, "stop_seq": 3,
                                 "model_call_seqs": [2],
                                 "ledger_path": str(task_dir / "ledger.jsonl")},
                })
            total = float(unit * len(ids))
            write_jsonl(run / "ledger.jsonl", [
                {"schema": "ledger/1", "seq": 1, "run_id": run_id,
                 "task_id": None, "kind": "run_start"},
                {"schema": "ledger/1", "seq": 2, "run_id": run_id,
                 "task_id": None, "kind": "run_end", "total_cost": total},
            ])
            report = {"schema": "run_report/1", "run_id": run_id,
                      "phase": phase, "config": config, "rep": rep,
                      "currency": "TOKENS", "exact_token_accounting": True,
                      "total_cost": total, "tasks": task_reports,
                      "evidence": {"run_end_seq": 2}}
            write_json(run / "report.json", report)
            return report, run

        study_dir = root / "runs/auto-study"
        evaluations = {}
        flat_reports = []
        task_cost = 0.0
        for name, descriptor in (("default", default), ("auto-01", tuned)):
            reports = []
            attempts = []
            for rep in (1, 2, 3):
                report, run = make_run("self_research",
                                       "default_lib" if name == "default" else "tuned_lib",
                                       rep, tasks["practice"])
                reports.append(report)
                flat_reports.append(report)
                task_cost += report["total_cost"]
                attempts.append({"candidate": name, "rep": rep,
                                 "run_id": run.name, "report_path": str(run / "report.json"),
                                 "config_sha256": descriptor["config"]["sha256"],
                                 "prompt_sha256": {role: descriptor["prompts"][role]["sha256"]
                                                   for role in ROLES},
                                 "total_cost": report["total_cost"]})
            evaluations[name] = {"candidates": {name: descriptor},
                                 "attempts": attempts, "reports": reports}
        proposal = study_dir / "proposal_ledger.jsonl"
        proposal_data = {"settings": {"plateau_nodes": 4},
                         "prompt_append": {role: "" for role in ROLES},
                         "rationale": "Use measured practice evidence."}
        proposal_hash = hashlib.sha256(json.dumps({
            key: proposal_data[key] for key in ("settings", "prompt_append")},
            ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        write_json(study_dir / "proposal_responses/response-0000000000000001.json", {
            "schema": "model_response_audit/1", "role": "planner", "purpose": "plan",
            "model": MODEL, "prompt_hash": "c" * 64,
            "output_text": json.dumps(proposal_data),
            "tokens": {"input": 7, "cached_input": 0, "output": 0, "reasoning": 0},
            "cost": 7.0,
        })
        write_jsonl(proposal, [{"schema": "proposal_ledger/1", "seq": 1,
                                "round": 1, "tokens": {"input": 7, "cached_input": 0,
                                                       "output": 0, "reasoning": 0},
                                "cost": 7.0, "price_sha256": price_sha,
                                "source_prompt_sha256": "b" * 64,
                                "model": MODEL, "role": "planner", "purpose": "plan",
                                "prompt_hash": "c" * 64}])
        write_json(study_dir / "auto_plan.json", {
            "schema": "auto_self_research_plan/1", "label": "FORMAL",
            "rounds": 1, "max_output_tokens": 2048,
            "practice_tasks": tasks["practice"], "price_sha256": price_sha,
            "source_prompt_sha256": "b" * 64, "default_files": default,
        })
        write_json(study_dir / "auto_self_research_report.json", {
            "schema": "auto_self_research/1", "label": "FORMAL",
            "currency": "TOKENS", "practice_tasks": tasks["practice"],
            "proposal_ledger_path": str(proposal),
            "proposal_ledger_sha256": digest(proposal),
            "proposal_attempts": [{"round": 1, "ledger_seq": 1,
                                   "status": "evaluated", "candidate": "auto-01",
                                   "cost": 7.0, "proposal_sha256": proposal_hash}],
            "evaluations": evaluations, "reports": flat_reports,
            "selection": {"candidate": "auto-01",
                          "config_path": tuned["config"]["path"],
                          "config_sha256": tuned["config"]["sha256"]},
            "total_task_cost": task_cost, "total_proposal_cost": 7.0,
            "total_research_cost": task_cost + 7.0,
        })

        batch_dir = root / "runs/batch"
        write_json(batch_dir / "plan.json", {
            "schema": "batch_plan/1", "label": "FORMAL", "bank_version": "bank-v1",
            "tasks": tasks, "configs": ["naive", "default", "default_lib", "tuned_lib"],
            "reps": [1, 2, 3], "run_validation": True, "run_exam": True,
            "sealed_inputs": {str(path): digest(path) for path in
                              (bank_path, price_path, Path(default["config"]["path"]),
                               Path(tuned["config"]["path"]), Path(naive["config"]["path"]))},
        })
        rows = []
        reports = {tier: [] for tier in tasks}
        phase_configs = {
            "practice": ("naive", "default", "default_lib", "tuned_lib"),
            "validation": ("default_lib", "tuned_lib"),
            "exam": ("naive", "default", "default_lib", "tuned_lib") if admitted
                    else ("naive", "default", "default_lib"),
        }
        for tier, members in phase_configs.items():
            for rep in (1, 2, 3):
                for config in members:
                    unit = 10 if config == "tuned_lib" else 20
                    score = 1.0 if config == "tuned_lib" and admitted else 0.0
                    if config == "tuned_lib" and not admitted:
                        score = -1.0
                    report, run = make_run(tier, config, rep, tasks[tier],
                                           unit=unit, score=score)
                    reports[tier].append(report)
                    rows.append({"logical_phase": tier, "manifest_phase": tier,
                                 "config": config, "rep": rep,
                                 "run_id": run.name, "report_path": str(run / "report.json"),
                                 "test_only": False})
        write_json(batch_dir / "exam_attempt.json",
                   {"schema": "exam_attempt/1", "label": "FORMAL"})
        for tier in ("validation", "exam"):
            write_json(root / "runs" / f"{tier}-bank-v1.attempt.json", {
                "schema": f"{tier}_attempt/1", "bank_version": "bank-v1",
                "code_commit": COMMIT, "batch_dir": str(batch_dir),
            })
        write_json(batch_dir / "batch_report.json", {
            "schema": "batch_report/1", "label": "FORMAL",
            "bank_version": "bank-v1", "runs": rows, "reports": reports,
            "admission": {"schema": "admission/1",
                          "claim": "admitted" if admitted else "rejected",
                          "baseline": "default_lib", "candidate": "tuned_lib",
                          "score_not_lower_each_rep": admitted,
                          "cost_reduced_on_average": True},
            "tuned_exam_skipped": not admitted,
        })
        return batch_dir, study_dir, {"tasks": tasks, "study": study_dir,
                                      "batch": batch_dir, "root": root}

    def test_admitted_and_rejected_batches_are_read_only(self):
        for admitted, expected_exam in ((True, 12), (False, 9)):
            with self.subTest(admitted=admitted), TemporaryDirectory() as temp:
                batch, study, info = self.fixture(Path(temp), admitted=admitted)
                timestamps = {path: path.stat().st_mtime_ns for path in
                              info["root"].rglob("*") if path.is_file()}
                audit = audit_formal_outputs(batch, study)
                self.assertTrue(audit["ready"])
                self.assertEqual(audit["phase_runs"], {"practice": 12,
                                                       "validation": 6,
                                                       "exam": expected_exam})
                self.assertEqual(audit["study_runs"], 6)
                self.assertEqual(audit["proposal_cost"], 7.0)
                self.assertEqual(timestamps, {path: path.stat().st_mtime_ns
                                              for path in timestamps})

    def test_task_over_budget_and_report_tampering_fail(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            row = info["root"] / "runs" / json.loads(
                (batch / "batch_report.json").read_text())["runs"][0]["run_id"]
            task = info["tasks"]["practice"][0]
            ledger = row / "tasks" / task / "ledger.jsonl"
            events = [json.loads(line) for line in ledger.read_text().splitlines()]
            events[1]["tokens"]["input"] = 40_001
            events[1]["cost"] = 40_001.0
            write_jsonl(ledger, events)
            with self.assertRaisesRegex(ValueError, "exceeds 40,000"):
                audit_formal_outputs(batch, study)

        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            row = info["root"] / "runs" / json.loads(
                (batch / "batch_report.json").read_text())["runs"][0]["run_id"]
            report = _read(row / "report.json")
            report["tasks"][0]["tokens"]["input"] += 1
            write_json(row / "report.json", report)
            with self.assertRaisesRegex(ValueError, "buckets or identity"):
                audit_formal_outputs(batch, study)

    def test_proposal_selection_marker_and_count_tampering_fail(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            ledger = study / "proposal_ledger.jsonl"
            ledger.write_text(ledger.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Proposal ledger SHA"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            chosen = info["root"] / "config/tuned.json"
            chosen.write_text(chosen.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed input changed"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            (info["root"] / "runs/exam-bank-v1.attempt.json").unlink()
            with self.assertRaises(FileNotFoundError):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            report = _read(batch / "batch_report.json")
            report["runs"].pop()
            write_json(batch / "batch_report.json", report)
            with self.assertRaisesRegex(ValueError, "stage/run count"):
                audit_formal_outputs(batch, study)

    def test_selected_sha_admission_and_private_input_tampering_fail(self):
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp))
            report = _read(study / "auto_self_research_report.json")
            report["selection"]["config_sha256"] = "0" * 64
            write_json(study / "auto_self_research_report.json", report)
            with self.assertRaisesRegex(ValueError, "Selected configuration SHA differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp))
            report = _read(batch / "batch_report.json")
            report["admission"]["score_not_lower_each_rep"] = False
            write_json(batch / "batch_report.json", report)
            with self.assertRaisesRegex(ValueError, "admission differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            private = info["root"] / "tasks/spd-practice-0/hidden/answer.txt"
            private.parent.mkdir(parents=True)
            private.write_text("secret", encoding="utf-8")
            plan = _read(batch / "plan.json")
            plan["sealed_inputs"][str(private)] = digest(private)
            write_json(batch / "plan.json", plan)
            with self.assertRaisesRegex(ValueError, "Refusing private sealed input"):
                audit_formal_outputs(batch, study)

    def test_priced_proposal_fingerprint_and_model_tampering_fail(self):
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp))
            report = _read(study / "auto_self_research_report.json")
            report["proposal_attempts"][0]["proposal_sha256"] = "0" * 64
            write_json(study / "auto_self_research_report.json", report)
            with self.assertRaisesRegex(ValueError, "Proposal fingerprint SHA differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp))
            response = study / "proposal_responses/response-0000000000000001.json"
            saved = _read(response)
            saved["model"] = "other-model"
            write_json(response, saved)
            with self.assertRaisesRegex(ValueError, "Proposal response provenance differs"):
                audit_formal_outputs(batch, study)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
