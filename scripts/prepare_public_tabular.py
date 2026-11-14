"""Rebuild pinned UCI binary-classification task packages.

Only this preparation step handles source labels for splitting. Calibration
reads train/dev only; the holdout files are created and then left unopened.
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
from urllib.request import urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
SEED = 20261114
SOURCES = {
    "tab-adult-income": (
        "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
        "Adult", 48842, ["high_cardinality_categorical"],
        "Predict whether annual income exceeds $50K from census attributes"),
    "tab-mushroom-edibility": (
        "https://archive.ics.uci.edu/static/public/73/mushroom.zip",
        "face32f32647e0d939f6233f36dd30dd5d619ae9f3f9b8e10bea4ac7e1f60b1a",
        "Mushroom", 8124, ["missing_heavy"],
        "Predict poisonous versus edible from a limited set of visible traits"),
    "tab-spambase": (
        "https://archive.ics.uci.edu/static/public/94/spambase.zip",
        "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636",
        "Spambase", 4601, ["many_numeric"],
        "Predict spam versus non-spam from message statistics"),
    "tab-wine-quality": (
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4",
        "Wine Quality", 6497, ["class_imbalance", "many_numeric"],
        "Predict whether a wine has quality score at least seven"),
    "tab-online-shoppers": (
        "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        "2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14",
        "Online Shoppers Purchasing Intention", 12330,
        ["class_imbalance", "many_numeric"],
        "Predict session purchase intent from browsing behavior"),
    "tab-breast-cancer-diagnostic": (
        "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        "bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
        "Breast Cancer Wisconsin Diagnostic", 569,
        ["small_sample", "many_numeric"],
        "Classify malignant versus benign tissue measurements"),
    "tab-heart-disease": (
        "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        "b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc",
        "Heart Disease", 303, ["small_sample"],
        "Classify presence of heart disease from Cleveland clinical attributes"),
}
ADULT_FEATURES = ["age", "workclass", "fnlwgt", "education", "education_num",
                  "marital_status", "occupation", "relationship", "race", "sex",
                  "capital_gain", "capital_loss", "hours_per_week", "native_country"]
MUSHROOM_FEATURES = [
    "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
    "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring",
    "stalk_color_above_ring", "stalk_color_below_ring", "veil_type",
    "veil_color", "ring_number", "ring_type", "spore_print_color",
    "population", "habitat",
]
# Odor and other near-direct identification clues are deliberately omitted.
MUSHROOM_SELECTED = ["cap_shape", "cap_color", "stalk_root", "habitat"]
HEART_FEATURES = ["age", "sex", "chest_pain", "resting_blood_pressure", "cholesterol",
                  "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
                  "st_depression", "st_slope", "major_vessels", "thalassemia"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(task_id: str, archive_dir: Path | None) -> tuple[zipfile.ZipFile, dict]:
    url, expected, *_ = SOURCES[task_id]
    name = {"tab-adult-income": "adult", "tab-mushroom-edibility": "mushroom",
            "tab-spambase": "spambase", "tab-wine-quality": "wine",
            "tab-online-shoppers": "online", "tab-breast-cancer-diagnostic": "breast",
            "tab-heart-disease": "heart"}[task_id]
    blob = ((archive_dir / f"{name}.zip").read_bytes() if archive_dir
            else urlopen(url, timeout=30).read())
    if sha(blob) != expected:
        raise ValueError(f"UCI source archive changed: {task_id}")
    return zipfile.ZipFile(io.BytesIO(blob)), {"url": url, "archive_sha256": expected,
                                            "raw_file_sha256": {}}


def _read(zipped: zipfile.ZipFile, filename: str, provenance: dict) -> str:
    blob = zipped.read(filename)
    provenance["raw_file_sha256"][filename] = sha(blob)
    return blob.decode("utf-8-sig")


def _parse(task_id: str, zipped: zipfile.ZipFile, provenance: dict) -> tuple[list[dict], list[str], list[str]]:
    records: list[dict] = []
    excluded: list[str] = []
    if task_id == "tab-adult-income":
        features = ADULT_FEATURES
        for filename in ("adult.data", "adult.test"):
            source = _read(zipped, filename, provenance)
            for row in csv.reader(io.StringIO(source), skipinitialspace=True):
                if not row or row[0].startswith("|"):
                    continue
                if len(row) != 15:
                    raise ValueError("Adult row must have 15 fields")
                target = row[-1].strip().rstrip(".")
                if target not in {"<=50K", ">50K"}:
                    raise ValueError("Unknown Adult class")
                record = {name: ("" if value.strip() == "?" else value.strip())
                          for name, value in zip(features, row[:-1])}
                record["y"] = "1" if target == ">50K" else "0"
                records.append(record)
    elif task_id == "tab-mushroom-edibility":
        features = MUSHROOM_SELECTED
        excluded = [name for name in MUSHROOM_FEATURES if name not in features]
        source = _read(zipped, "agaricus-lepiota.data", provenance)
        for row in csv.reader(io.StringIO(source)):
            if len(row) != 23 or row[0] not in {"p", "e"}:
                raise ValueError("Invalid Mushroom row")
            full = dict(zip(MUSHROOM_FEATURES, row[1:]))
            record = {name: ("" if full[name] == "?" else full[name]) for name in features}
            record["y"] = "1" if row[0] == "p" else "0"
            records.append(record)
    elif task_id == "tab-spambase":
        features = [f"feature_{i:02d}" for i in range(1, 58)]
        source = _read(zipped, "spambase.data", provenance)
        for row in csv.reader(io.StringIO(source)):
            if len(row) != 58 or row[-1] not in {"0", "1"}:
                raise ValueError("Invalid Spambase row")
            records.append({**dict(zip(features, row[:-1])), "y": row[-1]})
    elif task_id == "tab-wine-quality":
        features = []
        for color in ("red", "white"):
            filename = f"winequality-{color}.csv"
            source = _read(zipped, filename, provenance)
            reader = csv.DictReader(io.StringIO(source), delimiter=";")
            names = [name for name in reader.fieldnames or [] if name != "quality"]
            if not features:
                features = [*names, "wine_type"]
            elif features != [*names, "wine_type"]:
                raise ValueError("Wine feature mismatch")
            for row in reader:
                quality = int(row["quality"])
                records.append({**{name: row[name] for name in names},
                                "wine_type": color, "y": "1" if quality >= 7 else "0"})
    elif task_id == "tab-online-shoppers":
        source = _read(zipped, "online_shoppers_intention.csv", provenance)
        reader = csv.DictReader(io.StringIO(source))
        features = [name for name in reader.fieldnames or [] if name != "Revenue"]
        for row in reader:
            if row["Revenue"] not in {"TRUE", "FALSE"}:
                raise ValueError("Unknown Online Shoppers class")
            record = {name: ("1" if row[name] == "TRUE" else "0"
                             if row[name] == "FALSE" else row[name]) for name in features}
            record["y"] = "1" if row["Revenue"] == "TRUE" else "0"
            records.append(record)
    elif task_id == "tab-breast-cancer-diagnostic":
        features = [f"measurement_{index:02d}" for index in range(1, 31)]
        excluded = ["sample_id"]
        source = _read(zipped, "wdbc.data", provenance)
        for row in csv.reader(io.StringIO(source)):
            if len(row) != 32 or row[1] not in {"M", "B"}:
                raise ValueError("Invalid WDBC row")
            records.append({**dict(zip(features, row[2:])),
                            "y": "1" if row[1] == "M" else "0"})
    elif task_id == "tab-heart-disease":
        features = HEART_FEATURES
        source = _read(zipped, "processed.cleveland.data", provenance)
        for row in csv.reader(io.StringIO(source)):
            if len(row) != 14 or row[-1] not in {"0", "1", "2", "3", "4"}:
                raise ValueError("Invalid Cleveland row")
            records.append({**{name: ("" if value == "?" else value)
                               for name, value in zip(features, row[:-1])},
                            "y": "0" if row[-1] == "0" else "1"})
    else:
        raise KeyError(task_id)
    return records, features, excluded


def _split(records: list[dict], features: list[str]) -> tuple[dict[str, list[dict]], int]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in records:
        groups[tuple(row[name] for name in features)].append(row)
    by_class: dict[int, list[tuple[tuple[str, ...], list[dict]]]] = {0: [], 1: []}
    for key, group in groups.items():
        positive = sum(row["y"] == "1" for row in group)
        by_class[int(positive * 2 >= len(group))].append((key, group))
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "holdout": []}
    for grouped in by_class.values():
        grouped.sort(key=lambda item: sha((str(SEED) + "|" + json.dumps(item[0])).encode()))
        total = sum(len(group) for _, group in grouped)
        placed = 0
        for _, group in grouped:
            name = "train" if placed < total * .6 else "dev" if placed < total * .8 else "holdout"
            splits[name].extend(group)
            placed += len(group)
    for index, rows in enumerate(splits.values()):
        random.Random(SEED + index).shuffle(rows)
    hashed = [{sha(json.dumps([row[name] for name in features], separators=(",", ":")).encode())
               for row in rows} for rows in splits.values()]
    overlap = sum(len(hashed[i] & hashed[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    if overlap:
        raise AssertionError("Feature vectors overlap across splits")
    return splits, overlap


def build(task_id: str, archive_dir: Path | None = None) -> dict:
    if task_id not in SOURCES:
        raise KeyError(task_id)
    zipped, provenance = _source(task_id, archive_dir)
    with zipped:
        records, features, excluded = _parse(task_id, zipped, provenance)
    url, _, title, expected, triggers, description = SOURCES[task_id]
    if len(records) != expected:
        raise ValueError(f"Unexpected row count for {task_id}: {len(records)}")
    splits, overlap = _split(records, features)
    task_root = TASKS / task_id
    if task_root.exists():
        raise FileExistsError(f"Refusing to overwrite task package: {task_root}")
    for part in ("workspace", "hidden"):
        (task_root / part).mkdir(parents=True)
    for name, rows in splits.items():
        path = task_root / ("workspace" if name == "train" else "hidden") / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*features, "y"])
            writer.writeheader()
            writer.writerows(rows)
    original = (TASKS / "tab-bank-marketing/workspace/original.py").read_text(encoding="utf-8")
    (task_root / "workspace/original.py").write_text(original, encoding="utf-8")
    task = {
        "schema": "task/1", "task_id": task_id, "scenario": "tabular",
        "title": title, "description": description + ". Fit only on workspace/train.csv.",
        "metric": {"name": "roc_auc", "higher_is_better": True,
                   "baseline_raw": None, "ceiling_raw": None},
        "deliverable": {"file": "solution.py", "function": "fit_predict",
                        "signature": "fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]",
                        "returns": "One finite probability in [0, 1] per test row, preserving row order"},
        "budget": {"money": None, "wall_seconds": 1800},
        "limits": {"run_seconds": 300, "memory_mb": 4096},
    }
    (task_root / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    profile = {"schema": "profile/1", "features": triggers,
               "summary": {"dataset": f"UCI {title}", "rows_train": len(splits["train"]),
                           "predictors": len(features),
                           "positive_rate_train": Counter(row["y"] for row in splits["train"])["1"] / len(splits["train"]),
                           "target": description, "excluded_features": excluded,
                           "input": "Public train.csv has y; hidden test rows are feature dictionaries"}}
    (task_root / "workspace/profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    manifest = {"source": url, **provenance, "seed": SEED, "features": features,
                "excluded": excluded, "total_rows": len(records),
                "split_counts": {name: len(rows) for name, rows in splits.items()},
                "positive_counts": {name: Counter(row["y"] for row in rows)["1"]
                                    for name, rows in splits.items()},
                "feature_hash_overlap": overlap,
                "split_method": "class-stratified 60/20/20 grouped by exact feature vector"}
    (task_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    source_doc = (f"# UCI {title} source\n\n- Dataset page: {url.replace('/static/public/', '/dataset/').replace('.zip', '')}\n"
                  f"- Archive: {url}\n- SHA-256: {provenance['archive_sha256']}\n"
                  f"- Raw file SHA-256 hashes: see split_manifest.json.\n\n"
                  f"Preparation: scripts/prepare_public_tabular.py, seed {SEED}. "
                  "Exact feature-vector duplicates stay in one split; train/dev/holdout are "
                  "class-stratified 60/20/20. Hidden holdout is never read during calibration.\n\n"
                  "Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular "
                  "medium_quality, 180-second cap, measured on dev only. "
                  "See hidden/reference_measurements.json after calibration.\n")
    if task_id == "tab-mushroom-edibility":
        source_doc += ("\nThis task deliberately uses only four visible traits, omitting odor and "
                       "other near-direct identification clues. An initial eight-feature "
                       "pilot achieved a perfect LightGBM dev AUC, so the final four-feature "
                       "definition was narrowed before AutoGluon reference calibration. "
                       "Because dev feedback shaped the task, use this package only for "
                       "practice, not validation or exam.\n")
    if task_id == "tab-adult-income":
        source_doc += ("\nThe public adult.data and adult.test files are combined before "
                       "the new grouped split. Question-mark fields become missing values.\n")
    if task_id == "tab-spambase":
        source_doc += "\nThe source is unheaded; its 57 predictors are named feature_01 through feature_57 in source order.\n"
    if task_id == "tab-wine-quality":
        source_doc += "\nRed and white wines are combined; y=1 means source quality >= 7.\n"
    if task_id == "tab-online-shoppers":
        source_doc += "\nBoolean source fields are normalized to 1/0 strings before splitting.\n"
    if task_id == "tab-breast-cancer-diagnostic":
        source_doc += "\nThe sample ID is excluded; all 30 tissue measurements are retained. M is the positive class.\n"
    if task_id == "tab-heart-disease":
        source_doc += ("\nOnly the 303-row processed Cleveland subset is used. "
                       "The source disease severity (0-4) is binarized as any disease versus none; ? becomes missing.\n")
    (task_root / "SOURCE.md").write_text(source_doc, encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=tuple(SOURCES), action="append", required=True)
    parser.add_argument("--archive-dir", type=Path)
    args = parser.parse_args()
    for task_id in args.task_id:
        print(json.dumps(build(task_id, args.archive_dir), ensure_ascii=False))
