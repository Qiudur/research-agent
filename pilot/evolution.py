"""Two-run local smoke of experience writing, retrieval and outcome updates.

The roles are scripted. This verifies A's cross-run library wiring with a real
correctness/timing scorer; it is not autonomous model learning or a gain study.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import shutil

from library import Library
from research_agent.io import atomic_json, read_json, read_jsonl
from research_agent.runner import TaskRunner

from .harness import PilotHarness, TASK_ID, WORKSPACE
from .ledger import PilotLedger
from .roles import ScriptedRoles


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "pilot" / "candidate.py"


class DiscoveryRoles(ScriptedRoles):
    def summarize(self, payload: dict, *, node_id=None) -> dict:
        accepted = [node["id"] for node in payload["experiments"] if node["status"] == "accepted"]
        if not accepted:
            return {"drafts": [], "narrowings": []}
        return {"drafts": [{
            "scope": "domain", "scenario": "speedup",
            "triggers": ["nested_python_loop"],
            "conditions": "2D pairwise distances with list-of-lists output and NumPy available",
            "action": "Try NumPy broadcasting with in-place distance arithmetic",
            "negative": False, "node_ids": accepted,
        }], "narrowings": []}


class ReuseRoles(ScriptedRoles):
    def __init__(self, candidate_path: Path):
        super().__init__(candidate_path)
        self.seen_entry_ids: list[str] = []
        self.seen_brief = ""

    def plan(self, payload: dict, *, node_id=None) -> dict:
        self.plans += 1
        self.seen_entry_ids = list(payload["available_entry_ids"])
        self.seen_brief = payload["ability_brief"]
        hypotheses = []
        if self.plans == 1 and self.seen_entry_ids:
            hypotheses = [{
                "hypothesis": "Apply the retrieved NumPy distance strategy",
                "rationale": "The saved experience matches the nested-loop profile",
                "estimate": {"gain": 0.2, "cost": 0.01, "seconds": 5.0},
                "cites": [self.seen_entry_ids[0]],
            }]
        return {
            "hypotheses": hypotheses,
            "predicted": {"cost": 0.01, "seconds": 5.0, "score": 0.8},
            "budget_allocation": [], "expected_progress": [],
        }


def _round(run_dir: Path, library: Library, roles: ScriptedRoles, *, size: int, repetitions: int):
    baseline_dir = run_dir / "tasks" / TASK_ID / "nodes" / "n0000"
    baseline_dir.mkdir(parents=True)
    shutil.copyfile(WORKSPACE / "original.py", baseline_dir / "solution.py")
    harness = PilotHarness(size=size, repetitions=repetitions, measurements_path=run_dir / "measurements.jsonl")
    baseline = harness.score(TASK_ID, baseline_dir, "dev")
    if not baseline["ok"] or not baseline["correct"]:
        raise RuntimeError("Baseline correctness failed")
    outcome = TaskRunner(
        run_dir=run_dir, task_id=TASK_ID, harness=harness, ledger=PilotLedger(run_dir, TASK_ID),
        roles=roles, library=library, baseline_score=baseline,
        config=read_json(ROOT / "config/agent.json", "agent/1"),
        hard_caps=read_json(ROOT / "config/hard_caps.json", "hard_caps/1"),
    ).run()
    score = read_json(run_dir / "tasks" / TASK_ID / "nodes" / "n0001" / "score.json", "score/1")
    if outcome.best_node != "n0001" or not score["correct"]:
        raise RuntimeError("The real-scored candidate was not accepted")
    return outcome, score


def run_evolution(*, size: int = 1500, repetitions: int = 5, root_dir: Path | None = None) -> dict:
    if root_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        root_dir = ROOT / "runs" / f"evolution-{stamp}-{secrets.token_hex(2)}"
    root_dir = Path(root_dir).resolve()
    if root_dir.exists():
        raise FileExistsError(root_dir)
    library_root = root_dir / "library"
    first, first_score = _round(
        root_dir / "round-1", Library.open(library_root), DiscoveryRoles(CANDIDATE),
        size=size, repetitions=repetitions,
    )
    after_first = Library.open(library_root)
    entries = after_first.query("speedup", ["nested_python_loop"])
    if len(entries) != 1 or after_first.version != "lib-1":
        raise RuntimeError("First round did not write one retrievable experience")
    entry_id = entries[0]["id"]
    reuse = ReuseRoles(CANDIDATE)
    second, second_score = _round(
        root_dir / "round-2", after_first, reuse, size=size, repetitions=repetitions,
    )
    after_second = Library.open(library_root)
    entry = after_second.get(entry_id)
    second_nodes = read_jsonl(root_dir / "round-2" / "tasks" / TASK_ID / "tree.jsonl", "node/1")
    cited = next(node["cites"] for node in reversed(second_nodes) if node["node_id"] == "n0001")
    if (entry_id not in reuse.seen_entry_ids or entry_id not in reuse.seen_brief
            or cited != [entry_id]
            or entry["effect"]["uses"] != 1 or entry["effect"]["wins"] != 1
            or after_second.version != "lib-2"):
        raise RuntimeError("Second round did not retrieve, cite and update the experience")
    report = {
        "schema": "pilot_evolution_smoke/1", "run_id": root_dir.name,
        "input_size": size, "repetitions": repetitions,
        "round_1": {"best_node": first.best_node, "correct": first_score["correct"],
                    "raw_speedup": first_score["raw"], "library_version_after": first.library_version},
        "round_2": {"best_node": second.best_node, "correct": second_score["correct"],
                    "raw_speedup": second_score["raw"], "cited_entry": entry_id,
                    "library_version_after": second.library_version},
        "experience": {"id": entry_id, "uses": entry["effect"]["uses"],
                       "wins": entry["effect"]["wins"],
                       "credibility": entry["credibility"]},
        "model_calls": 0, "model_cost": None,
        "note": "Scripted roles reused the same known candidate. This verifies cross-run library wiring with real scoring, not autonomous learning or a causal performance gain.",
    }
    atomic_json(root_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1500)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_evolution(size=args.size, repetitions=args.repetitions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
