"""Untrusted candidate entry point. The parent invokes this inside macOS sandbox-exec.

This module never opens hidden task data. Its JSON response is validated by harness.py.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import runpy
import resource
import statistics
import sys
import threading
import time
import traceback


def _load(path: Path, function: str):
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load solution.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    result = getattr(module, function)
    if not callable(result):
        raise TypeError(f"{function} is not callable")
    return result


def _flags(node: Path, workspace: Path, task_root: Path):
    flags: set[str] = set()
    own = task_root.resolve()
    node = node.resolve()
    workspace = workspace.resolve()
    tasks = own.parent

    def audit(event, args):
        if event.startswith("socket.") or event in {"urllib.Request", "http.client.connect"}:
            flags.add("network")
            raise PermissionError("Candidate network access denied")
        if event == "subprocess.Popen":
            raise PermissionError("Candidate subprocess access denied")
        if event in {"open", "os.listdir", "os.scandir", "os.stat", "os.remove", "os.rename", "os.mkdir"}:
            if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            try:
                path = Path(args[0]).resolve()
            except (OSError, ValueError):
                return
            writing = event in {"os.remove", "os.rename", "os.mkdir"}
            if event == "open":
                mode = args[1] if len(args) > 1 else "r"
                writing = isinstance(mode, str) and any(c in mode for c in "wax+")
                if isinstance(mode, int):
                    writing = bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if path.is_relative_to(tasks) and not (path.is_relative_to(own) and path.is_relative_to(workspace)):
                # The candidate may inspect its own task.json, but never any hidden tree.
                if not path.is_relative_to(workspace):
                    flags.add("reads_hidden")
                    raise PermissionError("Task-private path denied")
            if writing and not path.is_relative_to(node):
                flags.add("writes_outside_node")
                raise PermissionError("Write outside node denied")

    return flags, audit


SPEED_KINDS = {"distance_matrix", "rolling_mean", "max_drawdown", "edit_distance",
               "membership_filter", "prefix_sum", "dedupe_order", "group_counts",
               "string_build", "top_frequency", "weighted_groups", "already_vectorized"}


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _equal(kind: str, actual, expected) -> bool:
    if kind == "distance_matrix":
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(isinstance(row, list) and len(row) == len(expected_row)
                   and all(_number(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                           for a, b in zip(row, expected_row))
                   for row, expected_row in zip(actual, expected))
    if kind == "rolling_mean":
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_number(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                        for a, b in zip(actual, expected)))
    if kind == "prefix_sum":
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_number(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                        for a, b in zip(actual, expected)))
    if kind == "max_drawdown":
        return _number(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    if kind == "already_vectorized":
        return _number(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    if kind == "edit_distance":
        return type(actual) is int and actual == expected
    if kind in {"membership_filter", "dedupe_order", "top_frequency"}:
        if type(actual) is not list or actual != expected:
            return False
        if kind == "top_frequency":
            return all(type(pair) is tuple and len(pair) == 2
                       and all(type(value) is int for value in pair) for pair in actual)
        return all(type(value) is int for value in actual)
    if kind == "string_build":
        return type(actual) is str and actual == expected
    if kind == "group_counts":
        return (type(actual) is dict and actual.keys() == expected.keys()
                and all(type(value) is int and value == expected[key]
                        for key, value in actual.items()))
    if kind == "weighted_groups":
        return (type(actual) is dict and actual.keys() == expected.keys()
                and all(_number(value) and math.isclose(value, expected[key], rel_tol=1e-9, abs_tol=1e-12)
                        for key, value in actual.items()))
    raise ValueError(f"Unknown speedup kind: {kind}")


def _points(seed: int, n: int):
    rng = random.Random(seed)
    return [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(n)]


def _case(kind: str, seed: int, size: int) -> tuple:
    if kind not in SPEED_KINDS or size < 0:
        raise ValueError("Invalid speedup workload")
    rng = random.Random(seed)
    if kind == "distance_matrix":
        return (_points(seed, size),)
    if kind == "rolling_mean":
        return ([rng.uniform(-10.0, 10.0) for _ in range(size)], max(1, size // 15))
    if kind == "max_drawdown":
        return ([max(1.0, 100.0 + 20.0 * math.sin(i / 47.0)
                     + rng.uniform(-8.0, 8.0)) for i in range(size)],)
    alphabet = "ACGTxyz"
    if kind == "edit_distance":
        return ("".join(rng.choices(alphabet, k=size)),
                "".join(rng.choices(alphabet, k=size)))
    if kind == "membership_filter":
        return ([rng.randrange(max(2, size)) for _ in range(size)],
                [rng.randrange(max(2, size)) for _ in range(max(1, size // 3))])
    if kind == "prefix_sum":
        return ([rng.uniform(0.1, 10.0) for _ in range(size)],)
    if kind == "dedupe_order":
        return ([rng.randrange(max(2, size // 2)) for _ in range(size)],)
    if kind in {"group_counts", "weighted_groups"}:
        categories = [f"g{i}" for i in range(max(2, size // 100))]
        keys = [rng.choice(categories) for _ in range(size)]
        if kind == "group_counts":
            return (keys,)
        return (keys, [rng.uniform(0.1, 10.0) for _ in range(size)])
    if kind == "string_build":
        return (["".join(rng.choices(alphabet, k=6)) for _ in range(size)],)
    if kind == "top_frequency":
        return ([rng.randrange(max(2, size // 20)) for _ in range(size)], min(10, size))
    if kind == "already_vectorized":
        return ([rng.uniform(-1.0, 1.0) for _ in range(size)],)
    raise ValueError(f"Unknown speedup kind: {kind}")


def _edge_cases(kind: str, seed: int) -> list[tuple]:
    if kind == "distance_matrix":
        return [([],), ([(0.0, 0.0)],),
                ([(1.0, 2.0), (1.0, 2.0), (-3.0, 4.0)],),
                _case(kind, seed + 10001, 13), _case(kind, seed + 20002, 39)]
    if kind == "rolling_mean":
        return [([], 3), ([5.0], 1), ([1.0, -2.0, 3.0], 2),
                ([1.0, 2.0, 3.0], 9), _case(kind, seed + 10001, 13)]
    if kind == "max_drawdown":
        return [([],), ([100.0],), ([1.0, 2.0, 3.0],),
                ([3.0, 2.0, 1.0],), ([4.0, 4.0, 2.0, 5.0, 1.0],),
                _case(kind, seed + 10001, 13)]
    if kind == "edit_distance":
        return [("", ""), ("", "abc"), ("abc", ""),
                ("same", "same"), ("kitten", "sitting"),
                _case(kind, seed + 10001, 13)]
    if kind == "membership_filter":
        return [([], []), ([1, 1, 2, 3], [1, 3]),
                ([4, 5], []), _case(kind, seed + 10001, 13)]
    if kind == "prefix_sum":
        return [([],), ([2.0],), ([1.0, 2.0, 3.0],),
                _case(kind, seed + 10001, 13)]
    if kind == "dedupe_order":
        return [([],), ([1],), ([2, 1, 2, 3, 1],),
                _case(kind, seed + 10001, 13)]
    if kind == "group_counts":
        return [([],), (["a"],), (["b", "a", "b", "c", "a"],),
                _case(kind, seed + 10001, 13)]
    if kind == "string_build":
        return [([],), (["a"],), (["a", "", "bc", "a"],),
                _case(kind, seed + 10001, 13)]
    if kind == "top_frequency":
        return [([], 0), ([3], 1), ([2, 1, 2, 1, 3], 2),
                _case(kind, seed + 10001, 13)]
    if kind == "weighted_groups":
        return [([], []), (["a"], [2.0]),
                (["a", "b", "a"], [1.0, 2.0, 3.0]),
                _case(kind, seed + 10001, 13)]
    if kind == "already_vectorized":
        return [([],), ([2.0],), ([1.0, -1.0, 3.0],),
                _case(kind, seed + 10001, 13)]
    raise ValueError(f"Unknown speedup kind: {kind}")


def _start_memory_guard(memory_mb: int):
    if memory_mb <= 0:
        raise ValueError("memory limit must be positive")
    stopped = threading.Event()

    def watch():
        while not stopped.wait(0.02):
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_bytes = rss if sys.platform == "darwin" else rss * 1024
            if rss_bytes > memory_mb * 1024 * 1024:
                os._exit(137)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return stopped


def _memory_exceeded(memory_mb: int) -> bool:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    return rss_bytes > memory_mb * 1024 * 1024


def _speed(kind: str, candidate, original, seed: int, size: int, repeats: int):
    if kind not in SPEED_KINDS or size <= 0 or repeats <= 0:
        raise ValueError("Invalid speedup benchmark")
    with contextlib.redirect_stdout(io.StringIO()):
        for case in _edge_cases(kind, seed):
            if not _equal(kind, candidate(*case), original(*case)):
                return {"correct": False, "reason": "behavior_change"}
        for case in (_case(kind, seed + 31003, 17), _case(kind, seed + 41004, 47)):
            if not _equal(kind, candidate(*case), original(*case)):
                return {"correct": False, "reason": "output_cache"}
        warmup = _case(kind, seed, 5)
        candidate(*warmup)
        original(*warmup)
        times = {"original": [], "candidate": []}
        for rep in range(repeats):
            # A fresh input on every repetition prevents a memoized answer table.
            case = _case(kind, seed + 100000 + rep, size)
            order = ("original", "candidate") if rep % 2 == 0 else ("candidate", "original")
            results = {}
            for label in order:
                func = original if label == "original" else candidate
                t0 = time.perf_counter()
                result = func(*case)
                elapsed = time.perf_counter() - t0
                results[label] = result
                times[label].append(elapsed)
            # Compare after both timers stop, including the full matrix. A
            # sampled matrix could let a sparse, incorrect output score well.
            if not _equal(kind, results["candidate"], results["original"]):
                return {"correct": False, "reason": "output_cache"}
    return {"correct": True, "timings": times,
            "medians": {k: statistics.median(v) for k, v in times.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("script", "speed"), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--function", default="distance_matrix")
    parser.add_argument("--task-kind", default="distance_matrix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=1800)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--script-args", default="[]")
    parser.add_argument("--memory-mb", type=int, required=True)
    args = parser.parse_args()
    flags, hook = _flags(args.candidate.parent, args.workspace, args.workspace.parent)
    sys.addaudithook(hook)
    memory_guard = _start_memory_guard(args.memory_mb)
    try:
        if args.mode == "script":
            out, err = io.StringIO(), io.StringIO()
            script_args = json.loads(args.script_args)
            if not isinstance(script_args, list) or not all(isinstance(item, str) for item in script_args):
                raise ValueError("script arguments must be a JSON string list")
            sys.argv = [str(args.candidate), *script_args]
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                runpy.run_path(str(args.candidate), run_name="__main__")
            if _memory_exceeded(args.memory_mb):
                payload = {"ok": False, "error": {"kind": "oom",
                            "message": "Candidate exceeded memory limit", "trace_tail": ""}}
            else:
                payload = {"ok": True, "stdout": out.getvalue()[-4096:], "stderr": err.getvalue()[-4096:]}
        else:
            candidate = _load(args.candidate, args.function)
            original = _load(args.workspace / "original.py", args.function)
            payload = {"ok": True, **_speed(args.task_kind, candidate, original,
                                             args.seed, args.size, args.repeats)}
            if _memory_exceeded(args.memory_mb):
                payload = {"ok": False, "error": {"kind": "oom",
                            "message": "Candidate exceeded memory limit", "trace_tail": ""}}
    except Exception as exc:
        payload = {"ok": False, "error": {"kind": "exception", "message": str(exc),
                    "trace_tail": traceback.format_exc()[-2000:]}}
    finally:
        memory_guard.set()
    payload["cheat_flags"] = sorted(flags)
    print(json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    main()
