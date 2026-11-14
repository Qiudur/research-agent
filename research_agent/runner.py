"""A-owned research loop wired to the published B/C interfaces.

The harness owns execution, scoring, and hidden data; the ledger owns pricing.
This module never imports or opens a task's hidden directory or bank.json.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

from .artifacts import append_run_log, node_directory, save_score, save_solution
from .io import append_jsonl, atomic_json, field, fields, read_json
from .scheduler import Scheduler
from .tree import ExperimentTree


@dataclass(frozen=True)
class TaskOutcome:
    run_id: str
    task_id: str
    reason: str
    best_node: str
    dev_score: float
    spent: float
    experiments: int
    library_version: str
    predicted: dict[str, float] | None


class TaskRunner:
    """Run one task within a C-created run, with injectable B/C implementations."""

    def __init__(
        self,
        *,
        run_dir: Path,
        task_id: str,
        harness: Any,
        ledger: Any,
        roles: Any,
        library: Any,
        baseline_score: Any,
        config: Mapping[str, Any],
        hard_caps: Mapping[str, Any],
        clock: Any = monotonic,
        on_initial_plan: Any | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.task_id = task_id
        self.harness = harness
        self.ledger = ledger
        self.roles = roles
        self.library = library
        self.config = dict(config)
        self.hard_caps = dict(hard_caps)
        self.clock = clock
        self.on_initial_plan = on_initial_plan
        self.task = harness.load_task(task_id)
        if field(self.task, "task_id") != task_id:
            raise ValueError("Harness returned a different task_id")
        budget = field(self.task, "budget")
        money = field(budget, "money")
        wall_seconds = field(budget, "wall_seconds")
        if money is None:
            raise ValueError("Task budget.money is unset; do not run with invented pricing")
        self.scheduler = Scheduler(self.config, self.hard_caps, money, wall_seconds)
        self.baseline_score = fields(baseline_score)
        if self.baseline_score.get("schema") != "score/1" or self.baseline_score.get("split") != "dev":
            raise ValueError("The injected baseline must be a dev score/1 result")
        if not self.baseline_score.get("ok") or self.baseline_score.get("score") is None:
            raise ValueError("The baseline score must be successful and numeric")
        self.tree = ExperimentTree.open(self.run_dir, task_id)
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path, "run/1")
            if manifest.get("run_id") != self.run_id or task_id not in manifest.get("tasks", []):
                raise ValueError("Manifest does not include this task/run")
            if manifest.get("phase") == "exam" and not getattr(library, "frozen", False):
                raise ValueError("Exam phase must open the ability library frozen")
        self.started = self.clock()
        self.context = self._task_context()
        self.predicted: dict[str, float] | None = None
        self.expected_progress: list[dict[str, float]] = []
        self.allocations: dict[str, float] = {}
        self.seen_hypotheses: set[str] = set()
        self.proposals: list[dict[str, Any]] = []
        self.proposal_counter = 0
        self.completed = 0
        self.non_improving = 0
        self.reviewed_behind = False
        self.last_review_note = ""

    def _task_context(self) -> dict[str, Any]:
        workspace = Path(field(self.task, "workspace")).resolve()
        if workspace.name != "workspace" or "hidden" in workspace.parts:
            raise ValueError("Harness task workspace must be its visible workspace directory")
        profile_path = workspace / "profile.json"
        if not profile_path.resolve().is_relative_to(workspace):
            raise ValueError("Task profile points outside the visible workspace")
        profile = read_json(profile_path, "profile/1") if profile_path.exists() else {}
        features = list(field(self.task, "features", profile.get("features", [])) or [])
        scenario = field(self.task, "scenario")
        if scenario not in {"tabular", "speedup"}:
            raise ValueError("Unknown task scenario")
        visible_sample = ""
        sample_name = "original.py" if scenario == "speedup" else "train.csv"
        sample_path = workspace / sample_name
        if not sample_path.resolve().is_relative_to(workspace):
            raise ValueError("Task sample points outside the visible workspace")
        if sample_path.is_file():
            with sample_path.open("r", encoding="utf-8", errors="replace") as handle:
                visible_sample = handle.read(12000 if scenario == "speedup" else 3500)
        return {
            "task_id": self.task_id,
            "scenario": scenario,
            "title": field(self.task, "title"),
            "description": field(self.task, "description"),
            "metric": fields(field(self.task, "metric")),
            "deliverable": fields(field(self.task, "deliverable")),
            "features": features,
            "profile_summary": profile.get("summary", {}),
            "visible_sample_file": sample_name,
            "visible_sample": visible_sample,
        }

    def _spent(self) -> float:
        value = float(self.ledger.spent())
        if not math.isfinite(value) or value < 0:
            raise ValueError("Ledger reported invalid task spend")
        return value

    def _elapsed(self) -> float:
        return max(0.0, float(self.clock() - self.started))

    def _ledger_elapsed(self) -> dict[str, float]:
        return dict(self.ledger.elapsed())

    def _emit(self, kind: str, *, node_id: str | None = None, **values: Any) -> None:
        self.ledger.emit(kind, node_id=node_id, **values)

    def _role(self, method: str, payload: dict[str, Any], *, node_id: str | None = None, **options: Any) -> dict[str, Any]:
        cost_before = self._spent()
        elapsed_before = self._ledger_elapsed()
        try:
            response = getattr(self.roles, method)(payload, node_id=node_id, **options)
        finally:
            if node_id is not None and node_id in self.tree.nodes:
                node = self.tree.get(node_id)
                delta_cost = max(0.0, self._spent() - cost_before)
                delta_llm = max(0.0, float(self._ledger_elapsed().get("llm", 0)) - float(elapsed_before.get("llm", 0)))
                seconds = dict(node["seconds"])
                seconds["llm"] += delta_llm
                role_name = {"plan": "planner", "execute": "executor", "diagnose": "diagnoser", "review": "planner", "summarize": "summarizer"}[method]
                calls = list(node["calls"])
                effort = self.config["effort"][role_name]
                match = next((item for item in calls if item["role"] == role_name and item["effort"] == effort), None)
                if match:
                    match["n"] += 1
                else:
                    calls.append({"role": role_name, "effort": effort, "n": 1})
                self.tree.patch(node_id, cost=node["cost"] + delta_cost, seconds=seconds, calls=calls)
        if not isinstance(response, dict):
            raise ValueError(f"{method} role must return a JSON object")
        return response

    def _planner_payload(self) -> dict[str, Any]:
        entries = self.library.query(self.context["scenario"], self.context["features"], k=self.config["query_k"])
        self._emit("library", op="query", entry_ids=[entry["id"] for entry in entries], version_after=self.library.version)
        history = [
            {
                "node_id": n["node_id"], "hypothesis": n["hypothesis"], "status": n["status"],
                "score": n["score"]["dev"] if n["score"] else None,
                "cost": n["cost"], "failure": n["failure"],
            }
            for n in self.tree.all_nodes() if n["status"] in {"accepted", "rejected", "failed"}
        ]
        recent = history if self.config.get("history_mode") == "full" else history[-8:]
        return {
            "task": self.context,
            "ability_brief": self.library.brief(entries),
            "available_entry_ids": [entry["id"] for entry in entries],
            "best": {"node_id": self.tree.best_node()["node_id"], "score": self.tree.best_node()["score"]["dev"]},
            "recent_experiments": recent,
            "budget": {"total": self.scheduler.budget, "spent": self._spent(), "remaining_to_cap": self.scheduler.budget_remaining(self._spent())},
            "allocations": self.allocations,
            "last_review": self.last_review_note,
        }

    def _prior_predictions(self) -> list[dict[str, float]]:
        observations = []
        for node in self.tree.all_nodes():
            if node["origin"] == "root" or node["status"] not in {"accepted", "rejected", "failed"}:
                continue
            parent = self.tree.get(node["parent"])
            actual = 0.0 if node["score"] is None else node["score"]["dev"] - parent["score"]["dev"]
            observations.append({"gain": node["estimate"]["gain"], "actual_gain": actual})
        return observations[-20:]

    def _queue_plan(self, result: dict[str, Any]) -> int:
        predicted = result.get("predicted")
        if self.predicted is None and isinstance(predicted, dict) and all(isinstance(predicted.get(key), (int, float)) for key in ("cost", "seconds", "score")):
            self.predicted = {key: float(predicted[key]) for key in ("cost", "seconds", "score")}
            opening = {
                "schema": "opening_plan/1", "run_id": self.run_id, "task_id": self.task_id,
                "predicted": self.predicted,
                "expected_progress": result.get("expected_progress", []),
                "budget_allocation": result.get("budget_allocation", []),
            }
            atomic_json(self.run_dir / "tasks" / self.task_id / "opening_plan.json", opening)
            if self.on_initial_plan is not None:
                self.on_initial_plan(opening)
        progress = result.get("expected_progress", [])
        if isinstance(progress, list):
            self.expected_progress = [p for p in progress if isinstance(p, dict) and isinstance(p.get("spent_ratio"), (int, float)) and isinstance(p.get("score"), (int, float))]
        allocations = result.get("budget_allocation", [])
        if isinstance(allocations, list):
            proposed = {str(a["direction"]): float(a["amount"]) for a in allocations if isinstance(a, dict) and a.get("direction") and isinstance(a.get("amount"), (int, float)) and a["amount"] >= 0}
            if not self.allocations and sum(proposed.values()) <= self.scheduler.budget:
                self.allocations = proposed
        allowed = {entry["id"]: entry for entry in self.library.query(self.context["scenario"], self.context["features"], k=self.config["query_k"])}
        queued = 0
        for hypothesis in result.get("hypotheses", []):
            if not isinstance(hypothesis, dict):
                continue
            title = str(hypothesis.get("hypothesis", "")).strip()
            if not title or title in self.seen_hypotheses:
                continue
            estimate = hypothesis.get("estimate", {})
            try:
                cites = [entry_id for entry_id in hypothesis.get("cites", []) if entry_id in allowed]
                priority = self.scheduler.priority_for(
                    estimate,
                    cited_effects=[allowed[entry_id]["effect"] for entry_id in cites],
                    past_predictions=self._prior_predictions(),
                )
                self.proposal_counter += 1
                self.proposals.append({
                    "node_id": f"n{self.proposal_counter:04d}",
                    "hypothesis": title,
                    "rationale": str(hypothesis.get("rationale", "")),
                    "origin": "explore" if self.scheduler.exploration_turn(self.completed + queued) else "plan",
                    "cites": cites,
                    "estimate": estimate,
                    "priority": priority,
                    "status": "queued",
                })
            except (ValueError, TypeError, KeyError):
                continue
            self.seen_hypotheses.add(title)
            queued += 1
        return queued

    def _expected_score(self) -> float | None:
        if not self.expected_progress:
            return None
        ratio = self._spent() / self.scheduler.budget
        points = [(0.0, float(self.baseline_score["score"]))]
        points += sorted((float(p["spent_ratio"]), float(p["score"])) for p in self.expected_progress)
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
            if ratio <= right_x:
                if right_x <= left_x:
                    return right_y
                return left_y + (right_y - left_y) * (ratio - left_x) / (right_x - left_x)
        return points[-1][1]

    def _run_attempt(self, node: dict[str, Any], attempt: int, previous_error: str | None) -> tuple[dict[str, Any] | None, str | None]:
        node_id = node["node_id"]
        node_dir = node_directory(self.run_dir, self.task_id, node_id)
        parent_dir = node_directory(self.run_dir, self.task_id, node["parent"])
        parent_file = parent_dir / "solution.py"
        previous_file = node_dir / "solution.py"
        payload = {
            "task": self.context,
            "hypothesis": node["hypothesis"],
            "rationale": node["rationale"],
            "parent_code": parent_file.read_text(encoding="utf-8")[:30000] if parent_file.exists() else "",
            "previous_code": previous_file.read_text(encoding="utf-8")[:30000] if previous_file.exists() else "",
            "previous_error": previous_error or "",
            "attempt": attempt,
            "max_attempts": self.config["max_attempts"],
        }
        if self.config.get("history_mode") == "full":
            payload["experiment_history"] = [
                {"node_id": item["node_id"], "hypothesis": item["hypothesis"],
                 "status": item["status"], "score": item["score"],
                 "failure": item["failure"], "cost": item["cost"]}
                for item in self.tree.all_nodes()
            ]
        try:
            response = self._role("execute", payload, node_id=node_id, fix=attempt > 1)
            save_solution(node_dir, response.get("code", ""), parent_dir)
        except Exception as exc:
            return None, f"Executor output failed: {exc}"
        try:
            run_result = self.harness.run_script(self.task_id, node_dir, ["solution.py"])
            run_data = fields(run_result)
        except Exception as exc:
            run_data = {"exit_code": -1, "timed_out": False, "oom": False, "seconds": 0.0, "stdout_tail": "", "stderr_tail": str(exc)}
        self._emit(
            "experiment", node_id=node_id, attempt=attempt,
            seconds_run=float(run_data.get("seconds", 0) or 0),
            exit_code=run_data.get("exit_code"), timed_out=bool(run_data.get("timed_out")),
            oom=bool(run_data.get("oom")),
        )
        append_run_log(node_dir, attempt, run_data)
        current = self.tree.get(node_id)
        seconds = dict(current["seconds"])
        seconds["run"] += float(run_data.get("seconds", 0) or 0)
        self.tree.patch(node_id, seconds=seconds)
        if run_data.get("exit_code") != 0 or run_data.get("timed_out") or run_data.get("oom"):
            return None, str(run_data.get("stderr_tail") or "Script failed")[-2000:]
        try:
            score_result = self.harness.score(self.task_id, node_dir, "dev")
            score_data = fields(score_result)
            if score_data.get("schema") != "score/1" or score_data.get("split") != "dev":
                raise ValueError("Harness returned an unsupported score schema or split")
            save_score(node_dir, score_data)
        except Exception as exc:
            return None, f"Dev scoring failed: {exc}"
        timing = score_data.get("seconds") or {}
        self._emit(
            "score", node_id=node_id, split="dev", score=score_data.get("score"),
            raw=score_data.get("raw"), spread=score_data.get("spread"),
            correct=score_data.get("correct"), cheat_flags=score_data.get("cheat_flags", []),
            seconds_queue=float(timing.get("queue", 0) or 0),
        )
        current = self.tree.get(node_id)
        seconds = dict(current["seconds"])
        seconds["run"] += float(timing.get("run", 0) or 0)
        seconds["queue"] += float(timing.get("queue", 0) or 0)
        self.tree.patch(node_id, seconds=seconds)
        if not score_data.get("ok") or score_data.get("score") is None:
            error = score_data.get("error") or {}
            return None, str(error.get("message", "Scoring failed"))[-2000:]
        if not score_data.get("correct", True):
            return None, "Correctness check failed"
        return score_data, None

    def _record_citations(self, node: dict[str, Any]) -> None:
        parent_score = self.tree.get(node["parent"])["score"]["dev"]
        gain = 0.0 if node["score"] is None else node["score"]["dev"] - parent_score
        for entry_id in node["cites"]:
            if not getattr(self.library, "frozen", False):
                entry = self.library.get(entry_id)
                if entry["status"] != "retired":
                    self.library.record_use(
                        entry_id, run_id=self.run_id, task_id=self.task_id,
                        node_id=node["node_id"], accepted=node["status"] == "accepted",
                        gain=gain, cost=node["cost"],
                    )
            self._emit("library", node_id=node["node_id"], op="use", entry_ids=[entry_id], version_after=self.library.version)

    def _diagnose(self, node: dict[str, Any], trigger: str) -> None:
        gate = self.scheduler.stop_reason(spent=self._spent(), elapsed_wall=self._elapsed(), best_score=self.tree.best_node()["score"]["dev"])
        if gate in {"budget_cap", "time"}:
            return
        response = self._role(
            "diagnose",
            {"task": self.context, "trigger": trigger, "node": {
                "hypothesis": node["hypothesis"], "failure": node["failure"],
                "score": node["score"], "attempts": node["attempts"],
            }, "best_score": self.tree.best_node()["score"]["dev"]},
            node_id=node["node_id"],
        )
        title = str(response.get("hypothesis", "")).strip()
        if not title or title in self.seen_hypotheses:
            return
        estimate = node["estimate"]
        self.proposal_counter += 1
        self.proposals.append({
            "node_id": f"n{self.proposal_counter:04d}",
            "hypothesis": title,
            "rationale": str(response.get("rationale") or response.get("advice") or trigger),
            "origin": "diagnose", "cites": [], "estimate": estimate,
            "priority": self.scheduler.priority_for(estimate), "status": "queued",
        })
        self.seen_hypotheses.add(title)

    def _execute_node(self, node: dict[str, Any]) -> dict[str, Any]:
        node_id = node["node_id"]
        error: str | None = None
        result: dict[str, Any] | None = None
        for attempt in range(1, int(self.config["max_attempts"]) + 1):
            if self.scheduler.stop_reason(spent=self._spent(), elapsed_wall=self._elapsed(), best_score=self.tree.best_node()["score"]["dev"]):
                break
            self.tree.start_attempt(node_id, max_attempts=int(self.config["max_attempts"]))
            result, error = self._run_attempt(node, attempt, error)
            if result is not None:
                self.tree.succeed(node_id, result)
                judged = self.tree.judge(node_id, result)
                if result.get("cheat_flags") or result["score"] > 1.2 or result["score"] < self.tree.get(node["parent"])["score"]["dev"] - 0.5:
                    self._diagnose(judged, "abnormal_score")
                self._record_citations(judged)
                return judged
        current = self.tree.get(node_id)
        if current["status"] == "running":
            failure = "timeout" if error and "timed out" in error.lower() else "exception"
            current = self.tree.fail(node_id, failure, (error or "Budget/time gate interrupted attempts")[:300])
            self._record_citations(current)
            if current["attempts"] >= int(self.config["max_attempts"]):
                self._diagnose(current, "failed_after_retries")
        return current

    def _review(self, trigger: str) -> str:
        result = self._role("review", {
            "trigger": trigger,
            "task": self.context,
            "best_score": self.tree.best_node()["score"]["dev"],
            "baseline_score": self.baseline_score["score"],
            "budget": {"total": self.scheduler.budget, "spent": self._spent()},
            "recent_nodes": [
                {"id": n["node_id"], "hypothesis": n["hypothesis"], "status": n["status"], "score": n["score"]["dev"] if n["score"] else None}
                for n in self.tree.all_nodes() if n["origin"] != "root"
            ][-8:],
            "allocations": self.allocations,
        })
        decision = result.get("decision")
        if decision not in {"continue", "switch", "reallocate", "stop"}:
            raise ValueError("Planner review returned an unknown decision")
        moves = result.get("budget_moves", [])
        valid_moves = []
        note = str(result.get("note", ""))
        if decision == "reallocate":
            candidate = dict(self.allocations)
            valid = isinstance(moves, list)
            for move in moves if valid else []:
                if not isinstance(move, dict):
                    valid = False
                    break
                source, target, amount = move.get("from"), move.get("to"), move.get("amount")
                if (source not in candidate or not target or not isinstance(amount, (int, float))
                        or not math.isfinite(amount) or amount < 0 or candidate[source] < amount):
                    valid = False
                    break
                candidate[source] -= float(amount)
                candidate[target] = candidate.get(target, 0.0) + float(amount)
                valid_moves.append(move)
            if valid:
                self.allocations = candidate
            else:
                decision = "continue"
                valid_moves = []
                note = f"Rejected invalid reallocation; {note}"
        self.last_review_note = note
        self._emit("review", trigger=trigger, decision=decision, budget_moves=valid_moves, note=self.last_review_note)
        self.non_improving = 0
        if trigger == "behind_schedule":
            self.reviewed_behind = True
        return decision

    def _summarize(self) -> None:
        compact = []
        for node in self.tree.all_nodes():
            if node["origin"] == "root":
                continue
            compact.append({
                "id": node["node_id"], "hypothesis": node["hypothesis"],
                "status": node["status"], "score": node["score"]["dev"] if node["score"] else None,
                "cost": node["cost"], "failure": node["failure"], "cites": node["cites"],
            })
        cited_ids = {entry_id for node in compact for entry_id in node["cites"]}
        cited_entries = []
        for entry_id in sorted(cited_ids):
            entry = self.library.get(entry_id)
            cited_entries.append({
                "id": entry_id, "conditions": entry["conditions"], "action": entry["action"],
                "triggers": entry["triggers"], "status": entry["status"], "effect": entry["effect"],
            })
        result = self._role("summarize", {
            "scenario": self.context["scenario"], "features": self.context["features"],
            "task_id": self.task_id, "experiments": compact[-40:],
            "cited_entries": cited_entries,
            "best": {"node_id": self.tree.best_node()["node_id"], "score": self.tree.best_node()["score"]["dev"]},
        })
        drafts = result.get("drafts", [])
        narrowings = result.get("narrowings", [])
        if not isinstance(drafts, list) or not isinstance(narrowings, list):
            raise ValueError("Summarizer must return arrays of drafts and narrowings")
        if not drafts and not narrowings:
            return
        if getattr(self.library, "frozen", False):
            append_jsonl(
                self.run_dir / "tasks" / self.task_id / "experience_drafts.jsonl",
                {"schema": "experience_drafts/1", "run_id": self.run_id, "task_id": self.task_id,
                 "drafts": drafts, "narrowings": narrowings},
            )
            self._emit("library", op="freeze", entry_ids=[], version_after=self.library.version)
            return
        seen_nodes = {node["id"] for node in compact}
        written_ids = []
        for draft in drafts:
            if not isinstance(draft, dict):
                continue
            evidence_nodes = draft.get("node_ids", [])
            if not isinstance(evidence_nodes, list) or not evidence_nodes or any(node_id not in seen_nodes for node_id in evidence_nodes):
                continue
            source = {"run_id": self.run_id, "task_id": self.task_id, "node_ids": sorted(set(evidence_nodes))}
            entry_draft = {key: value for key, value in draft.items() if key != "node_ids"}
            written_ids.extend(self.library.write_experiences([entry_draft], source=source))
        if written_ids:
            self._emit("library", op="write", entry_ids=written_ids, version_after=self.library.version)
        failed_citations: dict[str, list[str]] = {}
        for node in compact:
            if node["status"] in {"failed", "rejected"}:
                for entry_id in node["cites"]:
                    failed_citations.setdefault(entry_id, []).append(node["id"])
        for proposal in narrowings:
            if not isinstance(proposal, dict):
                continue
            entry_id = proposal.get("entry_id")
            if entry_id not in failed_citations:
                continue
            entry = self.library.get(entry_id)
            old = entry["conditions"].strip()
            new = str(proposal.get("conditions", "")).strip()
            reason = str(proposal.get("reason", "")).strip()
            if entry["status"] == "retired" or not new or new == old or (old and old not in new) or not reason:
                continue
            evidence = [{"run_id": self.run_id, "task_id": self.task_id, "node_ids": sorted(set(failed_citations[entry_id]))}]
            self.library.narrow(entry_id, new, reason=reason, evidence=evidence)
            self._emit("library", op="narrow", entry_ids=[entry_id], version_after=self.library.version)

    def run(self) -> TaskOutcome:
        """Run until a hard cap, planner stop, or exhausted hypotheses."""
        if self.tree.nodes:
            raise RuntimeError("Existing tree found; use replay for prior nodes or a fresh run ID")
        self._emit("task_start", budget=self.scheduler.budget, wall_seconds=self.scheduler.wall_seconds)
        self.tree.create_root(self.baseline_score, self.library.version)
        save_score(node_directory(self.run_dir, self.task_id, "n0000"), self.baseline_score)
        if self.context["scenario"] == "speedup":
            original = Path(field(self.task, "workspace")) / "original.py"
            if original.is_file():
                save_solution(node_directory(self.run_dir, self.task_id, "n0000"), original.read_text(encoding="utf-8"))
        reason = "planner"
        try:
            while True:
                best = self.tree.best_node()
                hard_reason = self.scheduler.stop_reason(spent=self._spent(), elapsed_wall=self._elapsed(), best_score=best["score"]["dev"])
                if hard_reason:
                    reason = hard_reason
                    break
                pending = list(self.proposals)
                if not pending:
                    result = self._role("plan", self._planner_payload())
                    if self._queue_plan(result) == 0:
                        reason = "planner"
                        break
                    pending = list(self.proposals)
                affordable = [n for n in pending if self.scheduler.can_expand(
                    spent=self._spent(), elapsed_wall=self._elapsed(),
                    estimated_cost=n["estimate"]["cost"], estimated_seconds=n["estimate"]["seconds"])]
                if not affordable:
                    reason = "budget_cap" if self.scheduler.budget_remaining(self._spent()) < min(n["estimate"]["cost"] for n in pending) else "time"
                    break
                accepted = [n for n in self.tree.all_nodes() if n["status"] == "accepted"]
                chosen_parent = self.scheduler.choose_parent(accepted, expanded_count=self.completed)
                candidates = [{**candidate, "parent": chosen_parent} for candidate in affordable]
                selected = self.scheduler.choose_next(candidates, expanded_count=self.completed, best_node_id=best["node_id"])
                if selected is None:
                    reason = "planner"
                    break
                self.proposals = [candidate for candidate in self.proposals if candidate["node_id"] != selected["node_id"]]
                origin = selected["origin"]
                if origin == "plan" and self.scheduler.exploration_turn(self.completed):
                    origin = "explore"
                node = self.tree.enqueue(
                    parent=chosen_parent, hypothesis=selected["hypothesis"],
                    rationale=selected["rationale"], origin=origin,
                    cites=selected["cites"], estimate=selected["estimate"],
                    priority=selected["priority"], library_version=self.library.version,
                )
                before_best = best["score"]["dev"]
                finished = self._execute_node(node)
                if finished["status"] == "queued":
                    reason = self.scheduler.stop_reason(spent=self._spent(), elapsed_wall=self._elapsed(), best_score=self.tree.best_node()["score"]["dev"]) or "error"
                    break
                self.completed += 1
                if self.tree.best_node()["score"]["dev"] > before_best:
                    self.non_improving = 0
                else:
                    self.non_improving += 1
                trigger = self.scheduler.review_trigger(
                    non_improving_nodes=self.non_improving,
                    spent=self._spent(), best_score=self.tree.best_node()["score"]["dev"],
                    baseline_score=float(self.baseline_score["score"]),
                    expected_score_at_spend=None if self.reviewed_behind else self._expected_score(),
                )
                if trigger and not self.scheduler.stop_reason(spent=self._spent(), elapsed_wall=self._elapsed(), best_score=self.tree.best_node()["score"]["dev"]):
                    decision = self._review(trigger)
                    if decision == "stop":
                        reason = "planner"
                        break
                    if decision in {"switch", "reallocate"}:
                        self.proposals.clear()
                        if decision == "switch":
                            self.allocations.clear()
                        self.reviewed_behind = False
            if self._elapsed() < self.scheduler.wall_seconds and self._spent() < self.scheduler.budget:
                self._summarize()
        except Exception as exc:
            reason = "error"
            self._emit("stop", reason=reason, best_node=self.tree.best_node()["node_id"], dev_score=self.tree.best_node()["score"]["dev"])
            raise RuntimeError(f"Task {self.task_id} failed after {self.completed} experiments") from exc
        self._emit("stop", reason=reason, best_node=self.tree.best_node()["node_id"], dev_score=self.tree.best_node()["score"]["dev"])
        best = self.tree.best_node()
        return TaskOutcome(self.run_id, self.task_id, reason, best["node_id"], best["score"]["dev"], self._spent(), self.completed, self.library.version, self.predicted)
