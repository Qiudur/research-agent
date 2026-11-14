"""Prepare two new UCI tabular tasks using only pinned source archives.

Preparation processes raw source labels to make grouped, class-stratified
splits. Subsequent calibration must use train and dev only; this script does
not score or inspect the generated holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261115  # fixed before inspecting the replacement dev split
SOURCES = {
    "tab-ionosphere": {
        "url": "https://archive.ics.uci.edu/static/public/52/ionosphere.zip",
        "archive_sha256": "4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860",
        "member": "ionosphere.data",
        "data_sha256": "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd",
        "rows": 351, "title": "Ionosphere",
        "description": "Classify radar returns as good or bad from 34 signal features",
        "citation": "Sigillito, V., Wing, S., Hutton, L., & Baker, K. (1989). Ionosphere [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W01B.",
        "page": "https://archive.ics.uci.edu/dataset/52/ionosphere",
    },
    "tab-monks-1": {
        "url": "https://archive.ics.uci.edu/static/public/70/monk+s+problems.zip",
        "archive_sha256": "7501acef87251b1d427726d7c381a2488c166fbbf70ac4096e09f05951c30a29",
        "member": "monks-1.test",
        "data_sha256": "e501ada3b21d876ab86f3e1614747bd2edc443a685f229a25e480aa1dea629f6",
        "rows": 432, "title": "MONK's Problems, task 1",
        "description": "Predict the binary MONK task-1 class from six categorical attributes",
        "citation": "Wnek, J. (1993). MONK's Problems [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5R30R.",
        "page": "https://archive.ics.uci.edu/dataset/70/monk+s+problems",
    },
}


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _records(task_id: str, source: str) -> tuple[list[dict[str, str]], list[str]]:
    rows = ([line.split() for line in source.splitlines() if line.strip()]
            if task_id == "tab-monks-1" else list(csv.reader(io.StringIO(source))))
    records = []
    if task_id == "tab-ionosphere":
        features = [f"radar_feature_{i:02d}" for i in range(1, 35)]
        for row in rows:
            if len(row) != 35 or row[-1] not in {"g", "b"}:
                raise ValueError("Invalid Ionosphere row")
            records.append({**dict(zip(features, row[:-1])), "y": "1" if row[-1] == "g" else "0"})
    elif task_id == "tab-monks-1":
        features = [f"attribute_{i}" for i in range(1, 7)]
        for row in rows:
            if len(row) != 8 or row[0] not in {"0", "1"}:
                raise ValueError("Invalid MONK task-1 row")
            records.append({**dict(zip(features, row[1:7])),
                            "source_row_id": row[7], "y": row[0]})
    else:
        raise KeyError(task_id)
    if len(records) != SOURCES[task_id]["rows"]:
        raise ValueError("Unexpected source row count")
    return records, features


def _split(records: list[dict[str, str]], features: list[str]) -> tuple[dict[str, list[dict[str, str]]], int]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        groups[tuple(row[name] for name in features)].append(row)
    by_class: dict[int, list[tuple[tuple[str, ...], list[dict[str, str]]]]] = {0: [], 1: []}
    for key, group in groups.items():
        positive = sum(row["y"] == "1" for row in group)
        by_class[int(positive * 2 >= len(group))].append((key, group))
    splits: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "holdout": []}
    for grouped in by_class.values():
        grouped.sort(key=lambda item: _sha((str(SEED) + "|" + json.dumps(item[0])).encode()))
        total = sum(len(group) for _, group in grouped)
        placed = 0
        for _, group in grouped:
            name = "train" if placed < total * .6 else "dev" if placed < total * .8 else "holdout"
            splits[name].extend(group)
            placed += len(group)
    for index, rows in enumerate(splits.values()):
        random.Random(SEED + index).shuffle(rows)
    hashed = [{_sha(json.dumps([row[name] for name in features], separators=(",", ":")).encode())
               for row in rows} for rows in splits.values()]
    overlap = sum(len(hashed[i] & hashed[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    return splits, overlap


def build(task_id: str, archive: Path | None = None, *, supersede: bool = False) -> dict:
    spec = SOURCES[task_id]
    task_root = ROOT / "tasks" / task_id
    existing_task = task_root / "task.json"
    if existing_task.is_file():
        previous = json.loads(existing_task.read_text(encoding="utf-8"))
        if not supersede and (previous.get("metric", {}).get("baseline_raw") is not None
                              or (task_root / "hidden/bank_v2_reference_measurements.json").is_file()):
            raise ValueError("Refusing to overwrite a calibrated task package")
    blob = archive.read_bytes() if archive else urlopen(spec["url"], timeout=30).read()
    if _sha(blob) != spec["archive_sha256"]:
        raise ValueError("UCI archive SHA-256 changed")
    with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
        data = zipped.read(spec["member"])
    if _sha(data) != spec["data_sha256"]:
        raise ValueError("UCI data SHA-256 changed")
    records, features = _records(task_id, data.decode("utf-8-sig"))
    splits, overlap = _split(records, features)
    if overlap != 0 or any(set(row["y"] for row in rows) != {"0", "1"}
                           for rows in splits.values()):
        raise ValueError("Split overlap or missing class")
    if supersede:
        (task_root / "hidden/bank_v2_reference_measurements.json").unlink(missing_ok=True)
    for name, rows in splits.items():
        path = task_root / ("workspace" if name == "train" else "hidden") / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            columns = [*features, "source_row_id", "y"] if name == "train" and task_id == "tab-monks-1" else [*features, "y"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({column: row[column] for column in columns} for row in rows)
    shutil.copyfile(ROOT / "scripts/tabular_v2_logistic.py", task_root / "workspace/original.py")
    task = {
        "schema": "task/1", "task_id": task_id, "scenario": "tabular",
        "title": spec["title"],
        "description": spec["description"] + ". Fit only on workspace/train.csv.",
        "metric": {"name": "roc_auc", "higher_is_better": True,
                   "baseline_raw": None, "ceiling_raw": None},
        "deliverable": {"file": "solution.py", "function": "fit_predict",
                        "signature": "fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]",
                        "returns": "One finite probability in [0,1] per test row, preserving row order"},
        "budget": {"money": None, "wall_seconds": 1800},
        "limits": {"run_seconds": 300, "memory_mb": 4096},
    }
    if task_id == "tab-monks-1":
        task["forbidden_features"] = ["source_row_id"]
        task["description"] += (" The training CSV alone includes source_row_id,"
                                 " an ID-like leakage lure absent from dev/holdout inference;"
                                 " do not fit on it.")
    (task_root / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    triggers = ["small_sample", "many_numeric"] if task_id == "tab-ionosphere" else ["small_sample", "id_like_column"]
    profile = {"schema": "profile/1", "features": triggers,
               "summary": {"dataset": "UCI " + spec["title"],
                           "rows_train": len(splits["train"]), "predictors": len(features),
                           "positive_rate_train": Counter(row["y"] for row in splits["train"])["1"] / len(splits["train"]),
                           "target": spec["description"],
                           "input": "Public train.csv has y; hidden test rows are feature dictionaries"}}
    (task_root / "workspace/profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    manifest = {"source": spec["url"], "archive_sha256": spec["archive_sha256"],
                "data_sha256": spec["data_sha256"], "seed": SEED, "features": features,
                "total_rows": len(records),
                "split_counts": {name: len(rows) for name, rows in splits.items()},
                "positive_counts": {name: Counter(row["y"] for row in rows)["1"]
                                    for name, rows in splits.items()},
                "feature_hash_overlap": overlap,
                "split_method": "class-stratified 60/20/20 grouped by exact feature vector"}
    if task_id == "tab-monks-1":
        manifest["train_only_forbidden_features"] = ["source_row_id"]
    (task_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    source_doc = (f"# UCI {spec['title']} source\n\n"
                  f"- Dataset page: {spec['page']}\n"
                  f"- Archive: {spec['url']}\n"
                  f"- Archive SHA-256: {spec['archive_sha256']}\n"
                  f"- Source data SHA-256: {spec['data_sha256']}\n"
                  "- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.\n"
                  f"- Attribution: {spec['citation']}\n"
                  "- Changes: deterministic grouped 60/20/20 train/dev/holdout split; original column names normalized, labels mapped to 0/1, missing markers normalized where present.\n\n"
                  f"Preparation: scripts/prepare_bank_v2_tabular.py, seed {SEED}; split counts and duplicate check in split_manifest.json. "
                  "Calibration uses only train and dev; generated holdout must stay unopened.\n")
    if task_id == "tab-monks-1":
        source_doc += ("\nThe original `monks-1.test` is treated only as the pinned source table; this project makes new train/dev/holdout splits. "
                       "Its final source row ID is visible in training only as an explicit ID-like leakage trap. "
                       "The scorer omits it from dev/holdout inference features, and task.json forbids using it; the baseline and reference drop it.\n")
    (task_root / "SOURCE.md").write_text(source_doc, encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", choices=sorted(SOURCES))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--supersede", action="store_true",
                        help="replace an explicitly invalidated pre-freeze split")
    args = parser.parse_args()
    result = build(args.task_id, args.archive, supersede=args.supersede)
    print(json.dumps({"task_id": args.task_id, "split_counts": result["split_counts"],
                      "feature_hash_overlap": result["feature_hash_overlap"]}))
