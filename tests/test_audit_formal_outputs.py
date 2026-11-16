"""Synthetic saved-output fixtures; no model, sandbox or scoring calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from evaluation.compare import compare_configurations
from scripts.audit_formal_outputs import audit_carry_forward, audit_formal_outputs


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
    def fixture(self, root: Path, *, admitted: bool = True,
                second_candidate: bool = False,
                over_unit: int | None = None,
                new_policy: bool = False) -> tuple[Path, Path, dict]:
        """over_unit sets the per-task token cost of exam naive rep 1 (budget is 40,000)."""
        root = root.resolve()
        tasks = {tier: [f"spd-{tier}-{number}" for number in range(count)]
                 for tier, count in (("practice", 10), ("validation", 4), ("exam", 10))}
        if new_policy:
            tasks["practice"] = ([f"spd-practice-{number}" for number in range(5)]
                                 + [f"tab-practice-{number}" for number in range(5)])
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
        tuned2 = candidate("tuned2") if second_candidate else None
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
                raw = 0.5 if task_id.startswith("tab-") else None
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
                     "holdout_score": score, **({"holdout_raw": raw} if raw is not None else {})},
                ]
                write_jsonl(task_dir / "ledger.jsonl", rows)
                write_json(task_dir / "holdout_attempt.json",
                           {"schema": "holdout_attempt/1"})
                write_json(task_dir / "holdout_score.json",
                           {"schema": "score/1", "split": "holdout", "score": score,
                            **({"raw": raw} if raw is not None else {})})
                task_reports.append({
                    "schema": "task_report/1", "run_id": run_id,
                    "task_id": task_id, "currency": "TOKENS",
                    "holdout_score": score, "total_cost": float(unit),
                    **({"holdout_raw": raw} if raw is not None else {}),
                    "holdout_failed": False,
                    "budget": 40_000.0, "budget_violation": unit > 40_000,
                    "budget_excess": max(0.0, unit - 40_000.0),
                    "tokens": tokens, "model_calls": 1,
                    "cost_by_role": {role: float(unit) if role == "planner" else 0.0
                                     for role in ROLES},
                    "evidence": {"task_end_seq": 4, "stop_seq": 3,
                                 "model_call_seqs": [2],
                                 "ledger_path": str(task_dir / "ledger.jsonl"),
                                 "holdout_path": str(task_dir / "holdout_score.json")},
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
        candidates = [("default", default), ("auto-01", tuned)]
        if tuned2 is not None:
            candidates.append(("auto-02", tuned2))
        for name, descriptor in candidates:
            if name == "auto-02":
                configs["tuned_lib"] = descriptor
            reports = []
            attempts = []
            for rep in (1, 2, 3):
                candidate_score = -1.0 if name == "auto-02" else 0.0
                report, run = make_run("self_research",
                                       "default_lib" if name == "default" else "tuned_lib",
                                       rep, tasks["practice"], score=candidate_score)
                reports.append(report)
                flat_reports.append(report)
                task_cost += report["total_cost"]
                attempts.append({"candidate": name, "rep": rep,
                                 "run_id": run.name, "report_path": str(run / "report.json"),
                                 "config_sha256": descriptor["config"]["sha256"],
                                 "prompt_sha256": {role: descriptor["prompts"][role]["sha256"]
                                                   for role in ROLES},
                                 "total_cost": report["total_cost"],
                                 "mean_score": candidate_score})
            evaluations[name] = {"candidates": {name: descriptor},
                                 "attempts": attempts, "reports": reports}
        configs["tuned_lib"] = tuned
        proposal = study_dir / "proposal_ledger.jsonl"
        proposal_rows = []
        proposal_attempts = []
        for seq, name in enumerate([name for name, _ in candidates if name != "default"], 1):
            proposal_data = {"settings": {"plateau_nodes": 3 + seq},
                             "prompt_append": {role: "" for role in ROLES},
                             "rationale": "Use measured practice evidence."}
            proposal_hash = hashlib.sha256(json.dumps({
                key: proposal_data[key] for key in ("settings", "prompt_append")},
                ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            prompt_hash = ("c" if seq == 1 else "d") * 64
            tokens = {"input": 7, "cached_input": 0, "output": 0, "reasoning": 0}
            write_json(study_dir / f"proposal_responses/response-{seq:016d}.json", {
                "schema": "model_response_audit/1", "role": "planner", "purpose": "plan",
                "model": MODEL, "prompt_hash": prompt_hash,
                "output_text": json.dumps(proposal_data),
                "tokens": tokens, "cost": 7.0,
            })
            proposal_rows.append({"schema": "proposal_ledger/1", "seq": seq,
                                  "round": seq, "tokens": tokens,
                                  "cost": 7.0, "price_sha256": price_sha,
                                  "source_prompt_sha256": "b" * 64,
                                  "model": MODEL, "role": "planner", "purpose": "plan",
                                  "prompt_hash": prompt_hash})
            proposal_attempts.append({"round": seq, "ledger_seq": seq,
                                      "status": "evaluated", "candidate": name,
                                      "cost": 7.0, "proposal_sha256": proposal_hash})
        write_jsonl(proposal, proposal_rows)
        write_json(study_dir / "auto_plan.json", {
            "schema": "auto_self_research_plan/1", "label": "FORMAL",
            **({"selection_policy": "paired-scenario-quality-v1"} if new_policy else {}),
            "rounds": len(proposal_rows), "max_output_tokens": 2048,
            "practice_tasks": tasks["practice"], "price_sha256": price_sha,
            "source_prompt_sha256": "b" * 64, "default_files": default,
        })
        ranking = [
            {"candidate": name,
             "mean_score": -1.0 if name == "auto-02" else 0.0,
             "mean_task_cost": 100.0,
             "proposal_cost": cost, "research_cost": 300.0 + cost,
             "score_per_currency": 0.0,
             "run_ids": [row["run_id"] for row in evaluations[name]["attempts"]]}
            for name, cost in [("default", 0.0),
                               *[(name, 7.0) for name, _ in candidates if name != "default"]]
        ]
        if new_policy:
            for row in ranking:
                row.update({
                    "paired_task_deltas": [
                        {"task_id": task_id,
                         "scenario": "tabular" if task_id.startswith("tab-") else "speedup",
                         "paired_deltas": [0.0, 0.0, 0.0], "mean_delta": 0.0}
                        for task_id in tasks["practice"]],
                    "scenario_mean_delta": {"tabular": 0.0, "speedup": 0.0},
                    "worst_scenario_mean_delta": 0.0,
                    "balanced_mean_delta": 0.0,
                })
        write_json(study_dir / "auto_self_research_report.json", {
            "schema": "auto_self_research/1", "label": "FORMAL",
            **({"selection_policy": "paired-scenario-quality-v1",
                "status": "complete", "cost_complete": True} if new_policy else {}),
            "currency": "TOKENS", "practice_tasks": tasks["practice"],
            "proposal_ledger_path": str(proposal),
            "proposal_ledger_sha256": digest(proposal),
            "proposal_attempts": proposal_attempts,
            "evaluations": evaluations, "reports": flat_reports,
            "ranking": ranking, "overall_best_on_practice": "default",
            "selection": {**ranking[1], "beats_default_on_practice": False,
                          "candidate": "auto-01",
                          "config_path": tuned["config"]["path"],
                          "config_sha256": tuned["config"]["sha256"]},
            "total_task_cost": task_cost,
            "total_proposal_cost": 7.0 * len(proposal_rows),
            "total_research_cost": task_cost + 7.0 * len(proposal_rows),
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
                    if over_unit is not None and (tier, config, rep) == ("exam", "naive", 1):
                        unit = over_unit
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
        comparison = compare_configurations(reports["validation"],
                                            baseline="default_lib", candidate="tuned_lib")
        write_json(batch_dir / "batch_report.json", {
            "schema": "batch_report/1", "label": "FORMAL",
            "bank_version": "bank-v1", "runs": rows, "reports": reports,
            "admission": {"schema": "admission/1",
                          "claim": "admitted" if admitted else "rejected",
                          "baseline": "default_lib", "candidate": "tuned_lib",
                          "score_not_lower_each_rep": admitted,
                          "cost_reduced_on_average": True,
                          "comparison": comparison},
            "tuned_exam_skipped": not admitted,
        })
        return batch_dir, study_dir, {"tasks": tasks, "study": study_dir,
                                      "batch": batch_dir, "root": root}

    def test_new_paired_policy_is_independently_audited(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp), new_policy=True)
            self.assertTrue(audit_formal_outputs(batch, study)["ready"])
            saved = _read(study / "auto_self_research_report.json")
            saved["ranking"][1]["paired_task_deltas"][5]["mean_delta"] = 1.0
            write_json(study / "auto_self_research_report.json", saved)
            with self.assertRaisesRegex(ValueError, "ranking differs"):
                audit_formal_outputs(batch, study)

    def test_unknown_usage_is_rejected_by_independent_audit(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp), new_policy=True)
            ledger = study / "proposal_ledger.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[0]["usage_status"] = "unknown"
            write_jsonl(ledger, rows)
            saved = _read(study / "auto_self_research_report.json")
            saved["proposal_ledger_sha256"] = digest(ledger)
            write_json(study / "auto_self_research_report.json", saved)
            with self.assertRaisesRegex(ValueError, "Proposal usage is unknown"):
                audit_formal_outputs(batch, study)

    def carry_fixture(self, base: Path) -> tuple[Path, Path, dict]:
        """Clone saved synthetic outputs into an independent bank-v2 root."""
        source = (base / "source-v1").resolve()
        target = (base / "target-v2").resolve()
        _, source_study, _ = self.fixture(source)
        for item in _read(source / "tasks/bank.json")["tasks"]:
            if item["tier"] in ("practice", "validation"):
                write_json(source / "tasks" / item["task_id"] / "task.json",
                           {"schema": "task/1", "task_id": item["task_id"]})
        shutil.copytree(source, target)
        old_ids = [f"spd-exam-{number}" for number in range(10)]
        for path in target.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                content = path.read_text(encoding="utf-8")
                content = content.replace(str(source), str(target))
                content = content.replace("bank-v1", "bank-v2")
                content = content.replace(COMMIT, "b" * 40)
                for old_id in old_ids:
                    content = content.replace(old_id, old_id + "-new")
                path.write_text(content, encoding="utf-8")
        for path in sorted(target.rglob("spd-exam-*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_dir() and path.name in old_ids:
                path.rename(path.with_name(path.name + "-new"))
        for tier in ("validation", "exam"):
            marker = target / "runs" / f"{tier}-bank-v1.attempt.json"
            marker.rename(target / "runs" / f"{tier}-bank-v2.attempt.json")
        batch = target / "runs/batch"
        plan = _read(batch / "plan.json")
        plan["sealed_inputs"][str(target / "tasks/bank.json")] = digest(target / "tasks/bank.json")

        def candidate(name: str) -> dict:
            config = target / "config" / f"{name}.json"
            prompts = {role: target / "config" / f"{name}-{role}.md" for role in ROLES}
            return {"config": {"path": str(config.relative_to(target)),
                               "sha256": digest(config)},
                    "prompts": {role: {"path": str(path.relative_to(target)),
                                       "sha256": digest(path)}
                                for role, path in prompts.items()}}

        copied = candidate("tuned")
        default = candidate("default")
        for item in (*copied["prompts"].values(), *default["prompts"].values()):
            plan["sealed_inputs"][str(target / item["path"])] = item["sha256"]
        source_files = {"bank": source / "tasks/bank.json",
                        "prices": source / "config/prices.json",
                        "auto_plan": source_study / "auto_plan.json",
                        "auto_report": source_study / "auto_self_research_report.json",
                        "proposal_ledger": source_study / "proposal_ledger.jsonl"}
        manifest = target / "config/bank_v2_carry_forward.json"
        write_json(manifest, {
            "schema": "bank_v2_carry_forward/1", "source_root": str(source),
            "source_study_dir": str(source_study.relative_to(source)),
            "source_bank_version": "bank-v1", "source_code_commit": COMMIT,
            "source_files": {name: {"path": str(path.relative_to(source)),
                                    "sha256": digest(path)}
                             for name, path in source_files.items()},
            "carried_candidate": copied, "carried_default": default,
        })
        plan["sealed_inputs"][str(manifest)] = digest(manifest)
        write_json(batch / "plan.json", plan)
        return batch, source_study, {"source": source, "target": target,
                                     "manifest": manifest, "candidate": copied}

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

    def test_bank_v2_carry_forward_audits_source_and_new_exam(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.carry_fixture(Path(temp))
            audit = audit_formal_outputs(batch, study)
            self.assertTrue(audit["ready"])
            self.assertTrue(audit["carry_forward"])
            self.assertFalse(audit["research_cost_applicable"])
            self.assertEqual(audit["source_code_commit"], COMMIT)
            self.assertEqual(audit["bank_version"], "bank-v2")
            self.assertEqual(audit["phase_runs"]["exam"], 12)
            from scripts.generate_roadshow_report import formal_results, load_formal_evidence
            evidence = load_formal_evidence(batch, study)
            self.assertFalse(evidence["breakeven"]["computed"])
            self.assertIn("沿用 bank-v1 自研究候选", formal_results(evidence))
            self.assertIn("不计入 bank-v2 回本点", formal_results(evidence))

    def test_bank_v2_carry_forward_rejects_unsealed_and_tampered_sources(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.carry_fixture(Path(temp))
            plan = _read(batch / "plan.json")
            del plan["sealed_inputs"][str(info["manifest"])]
            write_json(batch / "plan.json", plan)
            with self.assertRaisesRegex(ValueError, "manifest is not sealed"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.carry_fixture(Path(temp))
            (study / "proposal_ledger.jsonl").write_text(
                (study / "proposal_ledger.jsonl").read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "File SHA mismatch"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.carry_fixture(Path(temp))
            prompt = info["target"] / info["candidate"]["prompts"]["planner"]["path"]
            prompt.write_text(prompt.read_text() + "tamper", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed input changed"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            _, _, info = self.carry_fixture(Path(temp))
            target = info["target"]
            spec = target / "tasks/spd-practice-0/task.json"
            spec.write_text(spec.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "public task files changed"):
                audit_carry_forward(target, _read(target / "tasks/bank.json"),
                                    digest(target / "config/prices.json"))

    def test_bank_v2_carry_forward_rejects_exam_reuse_and_wrong_source_commit(self):
        with TemporaryDirectory() as temp:
            _, _, info = self.carry_fixture(Path(temp))
            target = info["target"]
            bank = _read(target / "tasks/bank.json")
            for item in bank["tasks"]:
                if item["tier"] == "exam":
                    item["task_id"] = item["task_id"].removesuffix("-new")
            with self.assertRaisesRegex(ValueError, "reuses bank-v1 exam IDs"):
                audit_carry_forward(target, bank, digest(target / "config/prices.json"))
        with TemporaryDirectory() as temp:
            _, _, info = self.carry_fixture(Path(temp))
            target = info["target"]
            manifest = _read(info["manifest"])
            manifest["source_code_commit"] = "c" * 40
            write_json(info["manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "Run manifest differs"):
                audit_carry_forward(target, _read(target / "tasks/bank.json"),
                                    digest(target / "config/prices.json"))

    def test_bank_v2_manifest_generator_seals_rechecked_source(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.carry_fixture(Path(temp))
            info["manifest"].unlink()
            from scripts.seal_bank_v2_carry_forward import seal_bank_v2_carry_forward
            produced = seal_bank_v2_carry_forward(
                info["target"], info["source"], study.relative_to(info["source"]),
                COMMIT, Path("config/tuned.json"), Path("config/default.json"))
            self.assertEqual(produced, info["manifest"])
            plan = _read(batch / "plan.json")
            plan["sealed_inputs"][str(produced)] = digest(produced)
            write_json(batch / "plan.json", plan)
            self.assertTrue(audit_formal_outputs(batch, study)["ready"])
    def test_over_budget_tasks_are_reported_not_assumed_zero(self):
        with TemporaryDirectory() as temp:
            clean, study, _ = self.fixture(Path(temp) / "clean")
            self.assertEqual(audit_formal_outputs(clean, study)["over_budget_tasks"], 0)
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp), over_unit=40_001)
            audit = audit_formal_outputs(batch, study)
            self.assertTrue(audit["ready"])
            self.assertEqual(audit["over_budget_tasks"], 10)
            self.assertEqual({item["task_id"] for item in audit["over_budget"]},
                             set(info["tasks"]["exam"]))
            self.assertEqual({item["excess"] for item in audit["over_budget"]}, {1.0})
            self.assertEqual(len({item["run_id"] for item in audit["over_budget"]}), 1)

    def test_task_over_budget_and_report_tampering_fail(self):
        with TemporaryDirectory() as temp:
            # The ledger shows an overspend that the saved report does not admit to.
            batch, study, info = self.fixture(Path(temp))
            row = info["root"] / "runs" / json.loads(
                (batch / "batch_report.json").read_text())["runs"][0]["run_id"]
            task = info["tasks"]["practice"][0]
            ledger = row / "tasks" / task / "ledger.jsonl"
            events = [json.loads(line) for line in ledger.read_text().splitlines()]
            events[1]["tokens"]["input"] = 40_001
            events[1]["cost"] = 40_001.0
            write_jsonl(ledger, events)
            with self.assertRaisesRegex(ValueError, "budget-violation record differs"):
                audit_formal_outputs(batch, study)

        with TemporaryDirectory() as temp:
            # A saved report may not claim a violation the ledger does not show.
            batch, study, info = self.fixture(Path(temp))
            row = info["root"] / "runs" / json.loads(
                (batch / "batch_report.json").read_text())["runs"][0]["run_id"]
            report = _read(row / "report.json")
            report["tasks"][0]["budget_violation"] = True
            report["tasks"][0]["budget_excess"] = 5.0
            write_json(row / "report.json", report)
            with self.assertRaisesRegex(ValueError, "budget-violation record differs"):
                audit_formal_outputs(batch, study)

        with TemporaryDirectory() as temp:
            # Partial batches and unscored holdouts fail closed with their own reason.
            batch, study, info = self.fixture(Path(temp))
            saved = _read(batch / "batch_report.json")
            saved["status"] = "partial"
            write_json(batch / "batch_report.json", saved)
            with self.assertRaisesRegex(ValueError, "partial"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            row = info["root"] / "runs" / json.loads(
                (batch / "batch_report.json").read_text())["runs"][0]["run_id"]
            report = _read(row / "report.json")
            report["tasks"][0]["holdout_failed"] = True
            write_json(row / "report.json", report)
            with self.assertRaisesRegex(ValueError, "holdout failed"):
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
            with self.assertRaisesRegex(ValueError, "Selected configuration SHA or ranking differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp))
            report = _read(batch / "batch_report.json")
            report["admission"]["score_not_lower_each_rep"] = False
            write_json(batch / "batch_report.json", report)
            with self.assertRaisesRegex(ValueError, "admission differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp))
            report = _read(batch / "batch_report.json")
            report["admission"]["comparison"]["candidate_mean_cost"] += 1
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

    def test_missing_or_tampered_public_holdout_score_fails_closed(self):
        for mutation in ("missing", "wrong_split", "wrong_score"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temp:
                batch, study, info = self.fixture(Path(temp))
                row = _read(batch / "batch_report.json")["runs"][0]
                score_path = (info["root"] / "runs" / row["run_id"] / "tasks"
                              / info["tasks"]["practice"][0] / "holdout_score.json")
                if mutation == "missing":
                    score_path.unlink()
                    with self.assertRaises(FileNotFoundError):
                        audit_formal_outputs(batch, study)
                else:
                    score = _read(score_path)
                    if mutation == "wrong_split":
                        score["split"] = "dev"
                    else:
                        score["score"] = 9.0
                    write_json(score_path, score)
                    with self.assertRaisesRegex(ValueError, "holdout score artifact differs"):
                        audit_formal_outputs(batch, study)

    def test_default_lib_config_swap_and_admission_evidence_tampering_fail(self):
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp))
            rows = _read(batch / "batch_report.json")["runs"]
            default_run = next(row["run_id"] for row in rows
                               if row["config"] == "default_lib")
            tuned_run = next(row["run_id"] for row in rows
                             if row["config"] == "tuned_lib")
            default_manifest = info["root"] / "runs" / default_run / "manifest.json"
            tuned_manifest = info["root"] / "runs" / tuned_run / "manifest.json"
            altered = _read(default_manifest)
            altered["agent_config"] = _read(tuned_manifest)["agent_config"]
            write_json(default_manifest, altered)
            with self.assertRaisesRegex(ValueError, "Default batch config differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, info = self.fixture(Path(temp), second_candidate=True)
            extra = info["root"] / "config/tuned2.json"
            plan = _read(batch / "plan.json")
            plan["sealed_inputs"][str(extra)] = digest(extra)
            write_json(batch / "plan.json", plan)
            row = next(item for item in _read(batch / "batch_report.json")["runs"]
                       if item["config"] == "naive")
            manifest_path = info["root"] / "runs" / row["run_id"] / "manifest.json"
            altered = _read(manifest_path)
            altered["agent_config"] = {"path": "config/tuned2.json", "sha256": digest(extra)}
            write_json(manifest_path, altered)
            with self.assertRaisesRegex(ValueError, "Naive batch config differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp))
            report = _read(batch / "batch_report.json")
            report["admission"]["comparison"]["evidence_run_ids"]["default_lib"][0] = "wrong-run"
            write_json(batch / "batch_report.json", report)
            with self.assertRaisesRegex(ValueError, "admission differs"):
                audit_formal_outputs(batch, study)

    def test_ranking_overall_winner_and_nonbest_selection_fail(self):
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp), second_candidate=True)
            self.assertEqual(audit_formal_outputs(batch, study)["study_runs"], 9)
            report = _read(study / "auto_self_research_report.json")
            report["ranking"][1]["mean_score"] = 99.0
            write_json(study / "auto_self_research_report.json", report)
            with self.assertRaisesRegex(ValueError, "ranking differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp), second_candidate=True)
            report = _read(study / "auto_self_research_report.json")
            report["overall_best_on_practice"] = "auto-01"
            write_json(study / "auto_self_research_report.json", report)
            with self.assertRaisesRegex(ValueError, "Overall practice winner differs"):
                audit_formal_outputs(batch, study)
        with TemporaryDirectory() as temp:
            batch, study, _ = self.fixture(Path(temp), second_candidate=True)
            report = _read(study / "auto_self_research_report.json")
            weaker = report["evaluations"]["auto-02"]["candidates"]["auto-02"]["config"]
            report["selection"] = {**report["ranking"][2],
                                   "config_path": weaker["path"],
                                   "config_sha256": weaker["sha256"],
                                   "beats_default_on_practice": False}
            write_json(study / "auto_self_research_report.json", report)
            with self.assertRaisesRegex(ValueError, "not the best generated"):
                audit_formal_outputs(batch, study)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
