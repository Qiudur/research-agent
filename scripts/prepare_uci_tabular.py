"""Build independent CC BY 4.0 UCI binary-classification task packages.

Download the pinned UCI archives into --archive-dir as ra-uci-<id>.zip first.
The script validates archive checksums and never exposes hidden labels in the
candidate workspace.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261114
ORIGINAL = ROOT / "tasks/tab-bank-marketing/workspace/original.py"

SOURCES = {
    "tab-magic-gamma": {
        "uci_id": 159, "slug": "magic+gamma+telescope",
        "title": "Classify gamma-ray telescope events",
        "archive_sha256": "252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d",
        "data_file": "magic04.data", "format": "csv", "positive": "g",
        "features": ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
                     "fM3Long", "fM3Trans", "fAlpha", "fDist"],
        "features_tags": ["many_numeric"],
        "target_note": "g (gamma event) is positive; h (hadron event) is negative",
    },
    "tab-htru2": {
        "uci_id": 372, "slug": "htru2", "title": "Identify pulsar candidates",
        "archive_sha256": "ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379",
        "data_file": "HTRU_2.csv", "format": "csv", "positive": "1",
        "features": ["profile_mean", "profile_std", "profile_kurtosis", "profile_skewness",
                     "dm_snr_mean", "dm_snr_std", "dm_snr_kurtosis", "dm_snr_skewness"],
        "features_tags": ["class_imbalance", "many_numeric"],
        "target_note": "1 means pulsar candidate; 0 means non-pulsar",
    },
    "tab-eeg-eye-state": {
        "uci_id": 264, "slug": "eeg+eye+state", "title": "Predict eye state from EEG signals",
        "archive_sha256": "49e7cbd14d303d63e772c7fcffa6b063bc52b97a7fa01b1135a9907b7715f415",
        "data_file": "EEG Eye State.arff", "format": "arff", "positive": "1",
        "features_tags": ["many_numeric", "possible_leakage"],
        "target_note": "Source eyeDetection=1 is positive; rows retain source temporal order",
        "split": "chronological",
    },
    "tab-default-credit": {
        "uci_id": 350, "slug": "default+of+credit+card+clients",
        "title": "Predict next-month credit-card default",
        "archive_sha256": "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602",
        "data_file": "default of credit card clients.xls", "format": "xls", "positive": "1",
        "features_tags": ["class_imbalance", "many_numeric", "id_like_column"],
        "target_note": "1 means default next month; source ID column is removed",
    },
    "tab-qsar-biodegradation": {
        "uci_id": 254, "slug": "qsar+biodegradation",
        "title": "Predict ready biodegradability from molecular descriptors",
        "archive_sha256": "b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac",
        "data_file": "biodeg.csv", "format": "semicolon", "positive": "RB",
        "features": [f"descriptor_{i:02d}" for i in range(1, 42)],
        "features_tags": ["small_sample", "many_numeric"],
        "target_note": "RB (readily biodegradable) is positive; NRB is negative",
    },
    "tab-phishing-websites": {
        "uci_id": 327, "slug": "phishing+websites",
        "title": "Predict the UCI phishing-website source label",
        "archive_sha256": "514f582cb44cba17251baf1d59f64cddff16efa2b644d091ce8d2a1e8d492940",
        "data_file": "Training Dataset.arff", "format": "arff", "positive": "1",
        "features_tags": ["many_numeric"],
        "target_note": "Source Result=1 is positive and Result=-1 is negative",
    },
    "tab-german-credit": {
        "uci_id": 144, "slug": "statlog+german+credit+data",
        "title": "Predict bad German credit risk",
        "archive_sha256": "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b",
        "data_file": "german.data", "format": "space", "positive": "2", "negative": "1",
        "features": ["checking_status", "duration_months", "credit_history", "purpose",
                     "credit_amount", "savings_status", "employment_status", "installment_rate",
                     "personal_status", "other_debtors", "residence_since", "property",
                     "age", "other_installment_plans", "housing", "existing_credits",
                     "job", "people_liable", "telephone", "foreign_worker"],
        "features_tags": ["small_sample"],
        "target_note": "Source class 2 (bad credit risk) is positive; class 1 (good) is negative",
    },
    "tab-australian-credit": {
        "uci_id": 143, "slug": "statlog+australian+credit+approval",
        "title": "Predict Australian credit approval",
        "archive_sha256": "f0bd9cd9b1d133cb20456b8953e1f8139fd30ed2b3f66da540db75136f1e9e0b",
        "data_file": "australian.dat", "format": "space", "positive": "1", "negative": "0",
        "features": [f"attribute_{i:02d}" for i in range(1, 15)],
        "features_tags": ["small_sample", "many_numeric"],
        "target_note": "Source class 1 is positive; class 0 is negative",
    },
}


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _parse_arff(blob: bytes) -> tuple[list[str], list[list[str]]]:
    lines = blob.decode("utf-8-sig", errors="replace").splitlines()
    names = []
    data_start = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith("@attribute"):
            names.append(line.split(maxsplit=2)[1].strip("'\""))
        if line.lower() == "@data":
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("ARFF has no @data")
    records = [row for row in csv.reader(lines[data_start:]) if row and not row[0].lstrip().startswith("%")]
    return names[:-1], records


def _parse(config: dict, blob: bytes) -> tuple[list[str], list[dict[str, str]]]:
    fmt = config["format"]
    if fmt == "xls":
        import pandas as pd
        frame = pd.read_excel(io.BytesIO(blob), header=1, engine="xlrd")
        features = [name for name in frame.columns if name not in {"ID", "default payment next month"}]
        records = []
        for values in frame[[*features, "default payment next month"]].itertuples(index=False, name=None):
            records.append([str(value) for value in values])
    elif fmt == "arff":
        features, records = _parse_arff(blob)
    elif fmt == "space":
        features = config["features"]
        records = [line.split() for line in blob.decode("utf-8-sig").splitlines() if line.strip()]
    else:
        features = config["features"]
        delimiter = ";" if fmt == "semicolon" else ","
        records = list(csv.reader(blob.decode("utf-8-sig").splitlines(), delimiter=delimiter))
    if not features or not records:
        raise ValueError("No features or records")
    positive = config["positive"]
    output = []
    for values in records:
        if len(values) != len(features) + 1:
            raise ValueError(f"Expected {len(features) + 1} fields, got {len(values)}")
        label = values[-1].strip()
        if label != positive and label not in {config.get("negative"), "h", "0", "NRB", "-1"}:
            raise ValueError(f"Unknown label: {label}")
        row = {str(name).replace(" ", "_"): value.strip() for name, value in zip(features, values[:-1])}
        row["y"] = "1" if label == positive else "0"
        output.append(row)
    return [str(name).replace(" ", "_") for name in features], output


def _split(rows: list[dict[str, str]], features: list[str], strategy: str) -> dict[str, list[dict]]:
    if strategy == "chronological":
        n = len(rows)
        splits = {"train": rows[:round(n * 0.6)],
                  "dev": rows[round(n * 0.6):round(n * 0.8)],
                  "holdout": rows[round(n * 0.8):]}
    else:
        groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[name] for name in features)].append(row)
        by_class = {0: [], 1: []}
        for key, records in groups.items():
            positive = sum(row["y"] == "1" for row in records)
            by_class[int(positive * 2 >= len(records))].append((key, records))
        splits = {"train": [], "dev": [], "holdout": []}
        for grouped in by_class.values():
            grouped.sort(key=lambda item: sha((str(SEED) + "|" + json.dumps(item[0])).encode()))
            total = sum(len(records) for _, records in grouped)
            placed = 0
            for _, records in grouped:
                split = "train" if placed < total * 0.6 else "dev" if placed < total * 0.8 else "holdout"
                splits[split].extend(records)
                placed += len(records)
        for i, records in enumerate(splits.values()):
            random.Random(SEED + i).shuffle(records)

    feature_hashes = []
    for records in splits.values():
        feature_hashes.append({sha(json.dumps([row[name] for name in features], separators=(",", ":")).encode())
                               for row in records})
        if set(row["y"] for row in records) != {"0", "1"}:
            raise ValueError("A split has only one class")
    if any(feature_hashes[i] & feature_hashes[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Identical feature vectors cross splits")
    return splits


def build(task_id: str, archive_dir: Path) -> dict:
    config = SOURCES[task_id]
    archive = archive_dir / f"ra-uci-{config['uci_id']}.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    blob = archive.read_bytes()
    if sha(blob) != config["archive_sha256"]:
        raise ValueError(f"Archive hash mismatch for {task_id}")
    with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
        data = zipped.read(config["data_file"])
    features, rows = _parse(config, data)
    strategy = config.get("split", "group_stratified")
    splits = _split(rows, features, strategy)
    task_root = ROOT / "tasks" / task_id
    for split, records in splits.items():
        target = task_root / ("workspace" if split == "train" else "hidden") / f"{split}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*features, "y"])
            writer.writeheader()
            writer.writerows(records)
    (task_root / "workspace/original.py").write_text(ORIGINAL.read_text(encoding="utf-8"), encoding="utf-8")
    profile = {"schema": "profile/1", "features": config["features_tags"],
               "summary": {"dataset": config["title"], "predictors": len(features),
                           "rows_train": len(splits["train"]), "target": config["target_note"],
                           "split_strategy": strategy}}
    (task_root / "workspace/profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    task = {"schema": "task/1", "task_id": task_id, "scenario": "tabular",
            "title": config["title"],
            "description": f"Train on the public UCI training split and predict y=1 probabilities for hidden {strategy} test rows.",
            "metric": {"name": "roc_auc", "higher_is_better": True,
                       "baseline_raw": None, "ceiling_raw": None},
            "deliverable": {"file": "solution.py", "function": "fit_predict",
                            "signature": "fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]",
                            "returns": "One finite probability in [0,1] per test row, in the same order"},
            "budget": {"money": None, "wall_seconds": 1800},
            "limits": {"run_seconds": 300, "memory_mb": 4096}}
    (task_root / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    manifest = {"source_url": f"https://archive.ics.uci.edu/static/public/{config['uci_id']}/{config['slug']}.zip",
                "archive_sha256": config["archive_sha256"], "data_file": config["data_file"],
                "data_sha256": sha(data), "seed": SEED, "strategy": strategy,
                "features": features, "source_rows": len(rows),
                "split_counts": {key: len(value) for key, value in splits.items()},
                "positive_counts": {key: Counter(row["y"] for row in value)["1"]
                                    for key, value in splits.items()},
                "feature_hash_overlap": 0}
    (task_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source = f"""# {config['title']} — data provenance

UCI Machine Learning Repository dataset {config['uci_id']}:
https://archive.ics.uci.edu/dataset/{config['uci_id']}/{config['slug']}

Download: {manifest['source_url']}
Pinned ZIP SHA-256: {config['archive_sha256']}
Pinned source data SHA-256: {manifest['data_sha256']}

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: {config['target_note']}.

Split strategy: {strategy}, seed {SEED}; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
"""
    (task_root / "SOURCE.md").write_text(source, encoding="utf-8")
    return {"task_id": task_id, "source_rows": len(rows), "split_counts": manifest["split_counts"],
            "positive_counts": manifest["positive_counts"], "strategy": strategy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--task-id", choices=SOURCES)
    args = parser.parse_args()
    selected = [args.task_id] if args.task_id else list(SOURCES)
    for task_id in selected:
        print(json.dumps(build(task_id, args.archive_dir)))
