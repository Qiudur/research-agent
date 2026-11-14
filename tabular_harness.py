"""Bank Marketing tabular scorer; candidate sees training data and test features only."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import resource
import secrets
import shutil
import statistics
import subprocess
import sys
import time

import harness


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKER = PACKAGE_ROOT / "tabular_worker.py"


def _project_root() -> Path:
    return harness._project_root()


def _node(node_dir: Path) -> Path:
    node = Path(node_dir).expanduser().resolve()
    if not node.is_dir() or node.is_relative_to(harness._tasks_root()):
        raise ValueError("Node directory must exist outside task packages")
    if node == _project_root() or not (node / "solution.py").is_file():
        raise ValueError("Node must contain solution.py")
    return node


def _python() -> Path:
    configured = os.environ.get("RA_TABULAR_PYTHON")
    candidate = Path(configured) if configured else _project_root() / ".venv/bin/python"
    # Preserve the venv symlink path: resolving it would bypass pyvenv.cfg.
    if configured and not candidate.expanduser().is_file():
        raise FileNotFoundError(f"RA_TABULAR_PYTHON does not exist: {candidate}")
    return candidate.expanduser().absolute() if candidate.exists() else Path(sys.executable).absolute()


def _python_environment(python: Path) -> Path | None:
    """Return the selected interpreter's environment, never the whole HOME."""
    root = python.parent.parent.resolve()
    if python.parent.name != "bin" or root == Path.home().resolve():
        return None
    if (root / "pyvenv.cfg").is_file():
        return root
    if python == Path(sys.executable).absolute() and root == Path(sys.prefix).resolve():
        return root  # active Conda-style environment
    return None


def _profile(workspace: Path, node: Path, python: Path) -> str:
    quote = lambda path: json.dumps(str(path.resolve()))
    home = Path.home().resolve()
    project = _project_root()
    lines = ["(version 1)", "(allow default)", "(deny network*)",
             f"(deny file-read* (subpath {quote(home)}))",
             f"(deny file-read* (subpath {quote(project)}))",
             f"(deny file-read* (subpath {quote(harness._tasks_root())}))"]
    venv = _python_environment(python)
    if venv is not None and (venv.is_relative_to(harness._tasks_root()) or
                             (venv.is_relative_to(project) and venv != project / ".venv")):
        raise ValueError("Python environment cannot overlap private task or project data")
    # realpath() walks each ancestor; allowing metadata on the exact directory
    # does not permit reading its children outside the narrow grants below.
    paths = (workspace, node, WORKER, python, *([venv] if venv else []))
    ancestors = {ancestor for path in paths
                 for ancestor in path.parents if ancestor == home or ancestor.is_relative_to(home)}
    lines.extend(f"(allow file-read* (literal {quote(path)}))"
                 for path in sorted(ancestors, key=lambda value: len(value.parts)))
    if venv is not None:
        lines.append(f"(allow file-read* (subpath {quote(venv)}))")
    if python.is_relative_to(home) and venv is None:
        lines.append(f"(allow file-read* (literal {quote(python)}))")
    lines.extend([f"(allow file-read* (subpath {quote(workspace)}))",
                  f"(allow file-read* (subpath {quote(node)}))",
                  f"(allow file-read* (literal {quote(WORKER)}))",
                  "(deny file-write*)",
                  f"(allow file-write* (subpath {quote(node)}))",
                  "(allow file-write* (literal \"/dev/null\"))"])
    return "\n".join(lines)


def _read_split(task_id: str, split: str, seed: int) -> tuple[list[dict[str, str]], list[int]]:
    task = harness.load_task(task_id)
    path = task["workspace"].parent / "hidden" / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("y") not in {"0", "1"} for row in rows):
        raise ValueError("Missing or invalid hidden labels")
    random.Random(seed).shuffle(rows)
    labels = [int(row.pop("y")) for row in rows]
    forbidden = set(task.get("forbidden_features", []))
    if any(forbidden.intersection(row) for row in rows):
        raise ValueError("Forbidden feature present in hidden split")
    return rows, labels


def roc_auc(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("AUC inputs differ in size or are empty")
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUC requires both classes")
    ranked = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += sum(label for _, label in ranked[i:j]) * (i + j + 1) / 2
        i = j
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def _child_limit(seconds: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    return apply


def raw_score(task_id: str, node_dir: Path, split: str, seed: int | None = None) -> dict:
    """Return a raw AUC measurement, also used to calibrate the task baseline."""
    task = harness.load_task(task_id)
    if task["scenario"] != "tabular" or task["metric"]["name"] != "roc_auc":
        raise ValueError("Task must be tabular binary roc_auc")
    if split not in {"dev", "holdout"}:
        raise ValueError("Split must be dev or holdout")
    if split == "holdout" and os.environ.get("RA_EVAL") != "1":
        raise PermissionError("RA_EVAL=1 is required for holdout")
    node = _node(node_dir)
    used_seed = int(seed if seed is not None else secrets.randbits(63))
    rows, labels = _read_split(task_id, split, used_seed)
    sandbox = shutil.which("sandbox-exec")
    if sys.platform != "darwin" or not sandbox:
        raise RuntimeError("macOS sandbox-exec is required; refusing unisolated candidate execution")
    python = _python()
    limits = task["limits"]
    seconds, memory_mb = int(limits["run_seconds"]), int(limits["memory_mb"])
    if seconds <= 0 or memory_mb <= 0:
        raise ValueError("Limits must be positive")
    command = [sandbox, "-p", _profile(task["workspace"], node, python), str(python),
               str(WORKER), "--candidate", str(node / "solution.py"),
               "--workspace", str(task["workspace"]), "--function", task["deliverable"]["function"],
               "--memory-mb", str(memory_mb)]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "TMPDIR": str(node), "HOME": str(node), "XDG_CACHE_HOME": str(node),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join((str(node), str(task["workspace"]))) }
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=node, env=env,
                              input=json.dumps({"test_rows": rows}), capture_output=True, text=True,
                              timeout=seconds + 2, preexec_fn=_child_limit(seconds))
    except subprocess.TimeoutExpired:
        return {"ok": False, "raw": None, "correct": False, "cheat_flags": [],
                "error": {"kind": "timeout", "message": "Scoring timed out", "trace_tail": ""},
                "seconds": time.perf_counter() - started, "seed": used_seed}
    elapsed = time.perf_counter() - started
    try:
        payload = json.loads(proc.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {"ok": False, "error": {"kind": "exception",
                   "message": "Sandbox worker returned no JSON", "trace_tail": proc.stderr[-2000:]}}
    flags = payload.get("cheat_flags", [])
    if proc.returncode != 0 or not payload.get("ok"):
        kind = ("oom" if proc.returncode in {-9, -11, 137} else
                "timeout" if proc.returncode in {-24, -25} else "exception")
        error = payload.get("error") or {"kind": kind, "message": "Scoring worker failed",
                                         "trace_tail": proc.stderr[-2000:]}
        return {"ok": False, "raw": None, "correct": False, "cheat_flags": flags,
                "error": error, "seconds": elapsed, "seed": used_seed}
    if flags or not payload.get("valid"):
        reason = payload.get("reason", "cheat_flag")
        return {"ok": True, "raw": 0.0, "correct": False,
                "cheat_flags": flags or [reason], "error": None,
                "seconds": elapsed, "seed": used_seed}
    predictions = payload["predictions"]
    if len(predictions) != len(labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in predictions):
        return {"ok": True, "raw": 0.0, "correct": False,
                "cheat_flags": ["invalid_probability"], "error": None,
                "seconds": elapsed, "seed": used_seed}
    return {"ok": True, "raw": roc_auc(labels, predictions), "correct": True,
            "cheat_flags": [], "error": None, "seconds": elapsed, "seed": used_seed,
            "labels": labels, "predictions": predictions}


def score(task_id: str, node_dir: Path, split: str, seed: int | None = None) -> dict:
    task = harness.load_task(task_id)
    baseline = task["metric"].get("baseline_raw")
    ceiling = task["metric"].get("ceiling_raw")
    if not isinstance(baseline, (float, int)) or not isinstance(ceiling, (float, int)) or ceiling <= baseline:
        raise ValueError("Tabular task needs measured baseline_raw < ceiling_raw")
    used_seed = int(seed if seed is not None else secrets.randbits(63))
    measurements = []
    for offset in range(3):
        measured = raw_score(task_id, node_dir, split, used_seed + offset)
        measurements.append(measured)
        if not measured["ok"] or not measured["correct"]:
            break
    failed = measurements[-1] if not measurements[-1]["ok"] or not measurements[-1]["correct"] else None
    result = {"schema": "score/1", "ok": failed is None or failed["ok"], "split": split,
              "score": None, "raw": None, "spread": 0.0,
              "correct": failed is None, "cheat_flags": failed["cheat_flags"] if failed else [],
              "error": failed["error"] if failed else None,
              "seconds": {"run": sum(item["seconds"] for item in measurements), "queue": 0.0},
              "seed": used_seed}
    if failed:
        if failed["ok"]:
            result["raw"] = 0.0
            result["score"] = 0.0
        elif harness._bank_v2_candidate_failure(Path(node_dir).resolve(), failed.get("error")):
            result["ok"] = True
            result["raw"] = 0.0
            result["score"] = 0.0
            result["error"], result["cheat_flags"] = harness._safe_candidate_failure(
                failed["error"], failed["cheat_flags"])
    else:
        scores = [(item["raw"] - baseline) / (ceiling - baseline) for item in measurements]
        result["raw"] = statistics.median(item["raw"] for item in measurements)
        result["score"] = statistics.median(scores)
        result["spread"] = statistics.pstdev(scores)
    return result
