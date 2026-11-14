"""Review bank-v2 UCI redistribution evidence without opening any CSV data.

An audit checks pinned source, license, split and dev-reference metadata. A
review additionally binds the audit to one ready proposal by its exact SHA-256.
Only SOURCE.md, split_manifest.json, task.json and the dev measurement JSON are
read from task packages; no train, dev or holdout CSV is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SHA = re.compile(r"[0-9a-f]{64}\Z")
EVAL_SEED = 20261114  # Scorer seed, distinct from the new tasks' split seed.
SPLITS = ("train", "dev", "holdout")
SPECS: dict[str, dict[str, Any]] = {
    "tab-spambase": {
        "uci_id": 94, "slug": "spambase", "doi": "10.24432/C53G6X",
        "archive_sha256": "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636",
        "raw_sha256": {"spambase.data": "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12"},
        "seed": 20261114, "counts": (2761, 921, 919), "positives": (1089, 363, 361),
        "authors": ("Hopkins", "Reeber", "Forman", "Suermondt"), "year": "1999",
        "changes": ("predictors renamed", "train/dev/holdout"),
    },
    "tab-qsar-biodegradation": {
        "uci_id": 254, "slug": "qsar+biodegradation", "doi": "10.24432/C5H60M",
        "archive_sha256": "b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac",
        "raw_sha256": "c66ac1a23aba11fb166fd5054fea14ac97b8e4f7e4940ae679dc863d1cc4b3c9",
        "seed": 20261114, "counts": (634, 211, 210), "positives": (214, 71, 71),
        "authors": ("Mansouri", "Ringsted", "Ballabio", "Todeschini", "Consonni"),
        "year": "2013", "changes": ("RB", "NRB", "train/dev/holdout"),
    },
    "tab-wine-quality": {
        "uci_id": 186, "slug": "wine+quality", "doi": "10.24432/C56S3T",
        "archive_sha256": "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4",
        "raw_sha256": {
            "winequality-red.csv": "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
            "winequality-white.csv": "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
        },
        "seed": 20261114, "counts": (3905, 1294, 1298), "positives": (773, 249, 255),
        "authors": ("Cortez", "Cerdeira", "Almeida", "Matos", "Reis"), "year": "2009",
        "changes": ("red and white", "quality mapped to >=7", "train/dev/holdout"),
    },
    "tab-ionosphere": {
        "uci_id": 52, "slug": "ionosphere", "doi": "10.24432/C5W01B",
        "archive_sha256": "4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860",
        "raw_sha256": "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd",
        "seed": 20261115, "counts": (211, 70, 70), "positives": (135, 45, 45),
        "authors": ("Sigillito", "Wing", "Hutton", "Baker"), "year": "1989",
        "changes": ("labels mapped to 0/1", "train/dev/holdout"),
    },
    "tab-monks-1": {
        "uci_id": 70, "slug": "monk+s+problems", "doi": "10.24432/C5R30R",
        "archive_sha256": "7501acef87251b1d427726d7c381a2488c166fbbf70ac4096e09f05951c30a29",
        "raw_sha256": "e501ada3b21d876ab86f3e1614747bd2edc443a685f229a25e480aa1dea629f6",
        "seed": 20261115, "counts": (260, 86, 86), "positives": (130, 43, 43),
        "authors": ("Wnek",), "year": "1993",
        "changes": ("monks-1.test", "source row ID", "train/dev/holdout"),
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def auc(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def audit_task(root: Path, task_id: str) -> dict[str, str]:
    require(task_id in SPECS, f"Unexpected tabular task: {task_id}")
    spec = SPECS[task_id]
    package = root / "tasks" / task_id
    source_path = package / "SOURCE.md"
    manifest_path = package / "split_manifest.json"
    measure_path = package / "hidden/bank_v2_reference_measurements.json"
    source = source_path.read_text(encoding="utf-8")
    manifest = read_json(manifest_path)
    measure = read_json(measure_path)
    task = read_json(package / "task.json")

    page = f"https://archive.ics.uci.edu/dataset/{spec['uci_id']}/{spec['slug']}"
    archive = f"https://archive.ics.uci.edu/static/public/{spec['uci_id']}/{spec['slug']}.zip"
    for evidence in (page, archive, spec["archive_sha256"],
                     f"https://doi.org/{spec['doi']}", "CC BY 4.0", "Attribution:",
                     spec["year"], *spec["authors"], *spec["changes"]):
        require(evidence in source, f"{task_id}: SOURCE.md missing {evidence}")
    require("holdout" in source.lower() and ("unopened" in source.lower()
            or "not opened" in source.lower() or "never read" in source.lower()),
            f"{task_id}: SOURCE.md omits holdout limitation")
    require(manifest.get("source", manifest.get("source_url")) == archive
            and manifest.get("archive_sha256") == spec["archive_sha256"],
            f"{task_id}: manifest archive provenance differs")
    raw = manifest.get("raw_file_sha256", manifest.get("data_sha256"))
    require(raw == spec["raw_sha256"], f"{task_id}: manifest raw hash differs")
    require(type(manifest.get("seed")) is int and manifest["seed"] == spec["seed"],
            f"{task_id}: split seed differs")
    counts = manifest.get("split_counts")
    positives = manifest.get("positive_counts")
    require(isinstance(counts, dict) and isinstance(positives, dict)
            and tuple(counts.get(key) for key in SPLITS) == spec["counts"]
            and tuple(positives.get(key) for key in SPLITS) == spec["positives"]
            and all(type(counts[key]) is int and type(positives[key]) is int
                    and 0 < positives[key] < counts[key] for key in SPLITS)
            and manifest.get("total_rows", manifest.get("source_rows")) == sum(spec["counts"]),
            f"{task_id}: split counts differ")
    require(manifest.get("feature_hash_overlap") == 0
            and isinstance(manifest.get("features"), list)
            and len(manifest["features"]) > 0
            and len(set(manifest["features"])) == len(manifest["features"]),
            f"{task_id}: feature split metadata invalid")
    grouped = (manifest.get("strategy") == "group_stratified"
               if task_id == "tab-qsar-biodegradation" else
               "grouped by exact feature vector" in manifest.get("split_method", ""))
    require(grouped, f"{task_id}: grouped split strategy missing")
    if task_id == "tab-monks-1":
        require(manifest.get("train_only_forbidden_features") == ["source_row_id"]
                and task.get("forbidden_features") == ["source_row_id"],
                f"{task_id}: source ID leakage protection differs")

    baseline, reference = measure.get("baseline"), measure.get("reference")
    require(measure.get("schema") == "tabular_bank_v2_reference/1"
            and measure.get("task_id") == task_id and measure.get("split") == "dev"
            and measure.get("seed") == EVAL_SEED
            and measure.get("split_manifest_sha256") == digest(manifest_path)
            and measure.get("scope") == "train/dev only; hidden holdout unopened and unscored"
            and isinstance(baseline, dict) and isinstance(reference, dict),
            f"{task_id}: dev measurement identity differs")
    require(auc(baseline.get("auc")) and auc(reference.get("auc"))
            and baseline.get("model") == "tabular_v2_logistic.py"
            and reference.get("model") == "tabular_v2_lightgbm.py"
            and measure.get("min_gap") == 0.01 and measure.get("gap_valid") is True
            and type(measure.get("gap")) in (int, float)
            and math.isclose(measure["gap"], reference["auc"] - baseline["auc"],
                             abs_tol=1e-12, rel_tol=0)
            and measure["gap"] >= 0.01,
            f"{task_id}: dev gap invalid")
    metric = task.get("metric")
    require(task.get("task_id") == task_id and task.get("scenario") == "tabular"
            and isinstance(metric, dict)
            and metric.get("baseline_raw") == baseline["auc"]
            and metric.get("ceiling_raw") == reference["auc"],
            f"{task_id}: task metric differs from dev measurement")
    return {"task_id": task_id, "source_md_sha256": digest(source_path),
            "split_manifest_sha256": digest(manifest_path),
            "dev_measurement_sha256": digest(measure_path)}


def audit(root: Path) -> list[dict[str, str]]:
    return [audit_task(root, task_id) for task_id in SPECS]


def review(root: Path, proposal_path: Path, confirmed_sha256: str) -> dict[str, Any]:
    require(SHA.fullmatch(confirmed_sha256) is not None
            and digest(proposal_path) == confirmed_sha256,
            "Proposal SHA-256 confirmation differs")
    proposal = read_json(proposal_path)
    require(proposal.get("schema") == "bank_v2_proposal/1"
            and proposal.get("ready") is True and proposal.get("blocked") == [],
            "Proposal is not ready")
    campaigns = proposal.get("campaigns")
    require(isinstance(campaigns, dict) and isinstance(campaigns.get("tabular"), dict)
            and set(campaigns["tabular"].get("task_ids", [])) == set(SPECS)
            and len(campaigns["tabular"].get("task_ids", [])) == len(SPECS),
            "Proposal does not contain exactly five audited tabular tasks")
    evidence = proposal.get("license_evidence")
    require(isinstance(evidence, dict) and set(evidence) == set(SPECS),
            "Proposal license evidence incomplete")
    audited = audit(root)
    for row in audited:
        require(evidence[row["task_id"]] == row["source_md_sha256"],
                f"{row['task_id']}: proposal SOURCE.md SHA differs")
    return {"schema": "bank_v2_license_review/1", "proposal_sha256": confirmed_sha256,
            "entries": [{"task_id": row["task_id"],
                         "source_md_sha256": row["source_md_sha256"],
                         "redistribution_permitted": True} for row in audited]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("audit", help="Audit five task packages; do not sign")
    check.add_argument("--project-root", type=Path, required=True)
    sign = commands.add_parser("review", help="Bind a ready proposal to reviewed sources")
    sign.add_argument("--project-root", type=Path, required=True)
    sign.add_argument("--proposal", type=Path, required=True)
    sign.add_argument("--confirm-proposal-sha", required=True)
    sign.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.project_root.expanduser().resolve()
        if args.command == "audit":
            rows = audit(root)
            print(json.dumps({"audited": [row["task_id"] for row in rows],
                              "source_md_sha256": {row["task_id"]: row["source_md_sha256"]
                                                   for row in rows}}, sort_keys=True))
        else:
            proposal_path = args.proposal.expanduser().resolve()
            result = review(root, proposal_path, args.confirm_proposal_sha)
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            print(json.dumps({"review_path": str(output),
                              "review_sha256": digest(output),
                              "proposal_sha256": result["proposal_sha256"]}))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
