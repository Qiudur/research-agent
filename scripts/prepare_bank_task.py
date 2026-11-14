"""Rebuild the Bank Marketing task from the pinned UCI archive."""

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
TASK = ROOT / "tasks" / "tab-bank-marketing"
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ARCHIVE_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
CSV_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
SEED = 20261114


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def duration_train_auc(rows: list[dict[str, str]]) -> float:
    ranked = sorted((int(row["duration"]), int(row["y"])) for row in rows)
    positives = sum(label for _, label in ranked)
    negatives = len(ranked) - positives
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += sum(label for _, label in ranked[i:j]) * (i + j + 1) / 2
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def source_csv(archive: Path | None) -> bytes:
    blob = archive.read_bytes() if archive else urlopen(URL, timeout=30).read()
    if sha(blob) != ARCHIVE_SHA256:
        raise ValueError("UCI archive SHA-256 changed")
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        inner_blob = outer.read("bank-additional.zip")
    with zipfile.ZipFile(io.BytesIO(inner_blob)) as inner:
        data = inner.read("bank-additional/bank-additional-full.csv")
    if sha(data) != CSV_SHA256:
        raise ValueError("UCI CSV SHA-256 changed")
    return data


def build(archive: Path | None = None) -> dict:
    data = source_csv(archive)
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter=";")
    features = [name for name in reader.fieldnames or [] if name not in {"duration", "y"}]
    if len(features) != 19:
        raise ValueError("Expected 19 features after removing duration")
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in reader:
        if row["y"] not in {"yes", "no"}:
            raise ValueError("Unknown target")
        record = {name: row[name] for name in features}
        # Keep the post-call field in visible training rows as an explicit
        # leakage lure. The dev/holdout inference rows never expose it.
        record["duration"] = row["duration"]
        record["y"] = "1" if row["y"] == "yes" else "0"
        groups[tuple(record[name] for name in features)].append(record)

    # Identical feature vectors stay together; allocate each class stratum
    # by a salted SHA order, so features cannot occur in multiple splits.
    by_class: dict[int, list[tuple[tuple[str, ...], list[dict]]]] = {0: [], 1: []}
    for key, records in groups.items():
        positive = sum(row["y"] == "1" for row in records)
        by_class[int(positive * 2 >= len(records))].append((key, records))
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "holdout": []}
    for grouped in by_class.values():
        grouped.sort(key=lambda item: sha((str(SEED) + "|" + json.dumps(item[0])).encode()))
        total = sum(len(records) for _, records in grouped)
        placed = 0
        for _, records in grouped:
            split = "train" if placed < total * 0.6 else "dev" if placed < total * 0.8 else "holdout"
            splits[split].extend(records)
            placed += len(records)
    for i, split in enumerate(splits.values()):
        random.Random(SEED + i).shuffle(split)

    hashes = [
        {sha(json.dumps([row[name] for name in features], separators=(",", ":")).encode())
         for row in splits[split]}
        for split in ("train", "dev", "holdout")
    ]
    if any(hashes[i] & hashes[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Feature overlap between splits")
    if sum(map(len, splits.values())) != 41188:
        raise AssertionError("Unexpected row count")

    for split, records in splits.items():
        path = TASK / ("workspace" if split == "train" else "hidden") / f"{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            columns = [*features, "duration", "y"] if split == "train" else [*features, "y"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({name: row[name] for name in columns} for row in records)
    report = {
        "source": URL, "archive_sha256": ARCHIVE_SHA256, "csv_sha256": CSV_SHA256,
        "seed": SEED, "features": features, "excluded": ["duration"],
        "train_only_forbidden_features": ["duration"],
        "train_only_duration_auc": duration_train_auc(splits["train"]),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "positive_counts": {k: Counter(row["y"] for row in v)["1"] for k, v in splits.items()},
        "feature_hash_overlap": 0,
    }
    (TASK / "split_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    print(json.dumps(build(parser.parse_args().archive), indent=2))
