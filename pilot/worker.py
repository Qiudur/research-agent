"""Subprocess worker for real correctness checks and alternating timing."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import random
from statistics import median
import sys
from time import perf_counter
import traceback


def load_function(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "distance_matrix")
    if not callable(function):
        raise TypeError("distance_matrix is not callable")
    return function


def verify(original, candidate, seed: int) -> tuple[bool, str | None]:
    rng = random.Random(seed + 10001)
    cases = [
        [], [(0.0, 0.0)], [(1.0, 2.0), (1.0, 2.0), (-3.0, 4.0)],
        [(rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(31)],
    ]
    for points in cases:
        expected = original(points)
        actual = candidate(points)
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False, "Output must be a list with one row per point"
        for row_actual, row_expected in zip(actual, expected):
            if not isinstance(row_actual, list) or len(row_actual) != len(row_expected):
                return False, "Output rows must be lists with one distance per point"
            for got, wanted in zip(row_actual, row_expected):
                if not isinstance(got, (int, float)) or not math.isclose(float(got), wanted, rel_tol=1e-9, abs_tol=1e-12):
                    return False, "Distance values differ from original"
    return True, None


def timed(function, points) -> float:
    gc.collect()
    start = perf_counter()
    output = function(points)
    seconds = perf_counter() - start
    if not isinstance(output, list) or len(output) != len(points):
        raise ValueError("Benchmark output has the wrong outer shape")
    if output and (not isinstance(output[0], list) or len(output[0]) != len(points)):
        raise ValueError("Benchmark output has the wrong inner shape")
    del output
    return seconds


def benchmark(original, candidate, reference, seed: int, size: int, repetitions: int) -> dict:
    rng = random.Random(seed)
    points = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(size)]
    warmup = points[:5]
    for function in (original, candidate, reference):
        function(warmup)
    timings = {"original": [], "candidate": [], "reference": []}
    functions = {"original": original, "candidate": candidate, "reference": reference}
    for index in range(repetitions):
        order = ("original", "candidate", "reference") if index % 2 == 0 else ("reference", "candidate", "original")
        for label in order:
            timings[label].append(timed(functions[label], points))
    medians = {label: median(values) for label, values in timings.items()}
    speedup = medians["original"] / medians["candidate"]
    normalized = medians["reference"] / medians["candidate"]
    ordered = sorted(timings["candidate"])
    iqr = ordered[3 * (len(ordered) - 1) // 4] - ordered[(len(ordered) - 1) // 4]
    spread = iqr / medians["candidate"] * normalized
    return {"timings": timings, "medians": medians, "speedup": speedup, "normalized_score": normalized, "spread": spread}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", type=int, default=2500)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    try:
        original = load_function(args.original, "pilot_original")
        candidate = load_function(args.candidate, "pilot_candidate")
        reference = load_function(args.reference, "pilot_reference")
        correct, error = verify(original, candidate, args.seed)
        if not correct:
            print(json.dumps({"ok": True, "correct": False, "error": error}))
            return 0
        metrics = benchmark(original, candidate, reference, args.seed, args.size, args.repetitions)
        print(json.dumps({"ok": True, "correct": True, "error": None, **metrics}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "correct": False, "error": {"kind": "exception", "message": str(exc), "trace_tail": traceback.format_exc()[-2000:]}}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
