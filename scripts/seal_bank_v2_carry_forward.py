"""Seal a bank-v1 self-research winner for a fresh bank-v2 formal batch.

This reads saved evidence only. Run before committing the bank-v2 working tree;
the formal preflight will reject an absent, changed or unaudited manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_formal_outputs import (
    _agent_files, _audit_study, _inside, _json, _sha, audit_carry_forward,
)


def seal_bank_v2_carry_forward(
    root: Path, source_root: Path, source_study_dir: Path, source_code_commit: str,
    carried_candidate: Path, carried_default: Path,
) -> Path:
    root = Path(root).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    source_study = _inside(source_root, source_study_dir)
    target = root / "config/bank_v2_carry_forward.json"
    if target.exists():
        raise FileExistsError(f"Carry-forward manifest already exists: {target}")
    source_bank = _json(source_root / "tasks/bank.json")
    price_sha = _sha(source_root / "config/prices.json")
    practice = [item["task_id"] for item in source_bank["tasks"]
                if item.get("tier") == "practice"]
    _audit_study(source_root, source_study, practice, source_bank["version"],
                 source_code_commit, price_sha)

    def relative_files(config: Path) -> dict:
        files = _agent_files(root, _inside(root, config))
        return {"config": {**files["config"],
                           "path": str(Path(files["config"]["path"]).relative_to(root))},
                "prompts": {role: {**item,
                                   "path": str(Path(item["path"]).relative_to(root))}
                            for role, item in files["prompts"].items()}}

    sources = {"bank": source_root / "tasks/bank.json",
               "prices": source_root / "config/prices.json",
               "auto_plan": source_study / "auto_plan.json",
               "auto_report": source_study / "auto_self_research_report.json",
               "proposal_ledger": source_study / "proposal_ledger.jsonl"}
    manifest = {
        "schema": "bank_v2_carry_forward/1",
        "source_root": str(source_root),
        "source_study_dir": str(source_study.relative_to(source_root)),
        "source_bank_version": source_bank["version"],
        "source_code_commit": source_code_commit,
        "source_files": {name: {"path": str(path.relative_to(source_root)),
                                "sha256": _sha(path)}
                         for name, path in sources.items()},
        "carried_candidate": relative_files(carried_candidate),
        "carried_default": relative_files(carried_default),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    try:
        audit_carry_forward(
            root, _json(root / "tasks/bank.json"), _sha(root / "config/prices.json"),
            candidate_path=_inside(root, carried_candidate),
            default_path=_inside(root, carried_default), study_dir=source_study)
    except Exception:
        target.unlink()
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-study-dir", type=Path, required=True)
    parser.add_argument("--source-code-commit", required=True)
    parser.add_argument("--carried-candidate", type=Path, required=True)
    parser.add_argument("--carried-default", type=Path, required=True)
    args = parser.parse_args()
    print(seal_bank_v2_carry_forward(
        args.project_root, args.source_root, args.source_study_dir,
        args.source_code_commit, args.carried_candidate, args.carried_default))


if __name__ == "__main__":
    main()
