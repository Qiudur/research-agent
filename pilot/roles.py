"""Fixed role responses to exercise A against a genuine scorer without API billing."""

from __future__ import annotations

from pathlib import Path


class ScriptedRoles:
    def __init__(self, candidate_path: Path):
        self.candidate_path = candidate_path
        self.plans = 0
        self.executions = 0

    def plan(self, payload: dict, *, node_id=None) -> dict:
        self.plans += 1
        hypotheses = [] if self.plans > 1 else [{
            "hypothesis": "Vectorize pairwise distances with NumPy",
            "rationale": "The profile identifies nested Python loops over all point pairs",
            "estimate": {"gain": 0.2, "cost": 0.01, "seconds": 5.0},
            "cites": [],
        }]
        return {
            "hypotheses": hypotheses,
            "predicted": {"cost": 0.01, "seconds": 5.0, "score": 0.8},
            "budget_allocation": [], "expected_progress": [],
        }

    def execute(self, payload: dict, *, fix=False, node_id=None) -> dict:
        self.executions += 1
        return {"code": self.candidate_path.read_text(encoding="utf-8"), "notes": "AI-generated pilot candidate"}

    def diagnose(self, payload: dict, *, node_id=None) -> dict:
        return {"hypothesis": "", "rationale": "Reference ceiling may be conservative", "advice": "Retest on fresh seeds"}

    def review(self, payload: dict, *, node_id=None) -> dict:
        return {"decision": "continue", "budget_moves": [], "note": "Pilot has one candidate"}

    def summarize(self, payload: dict, *, node_id=None) -> dict:
        return {"drafts": [], "narrowings": []}
