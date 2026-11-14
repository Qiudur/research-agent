"""Task package loader and isolated scoring interface (task/1, score/1)."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import secrets
import shutil
import statistics
import subprocess
import sys
import time

from harness_worker import _case as _speed_case, _edge_cases as _speed_edge_cases
from harness_worker import _equal as _speed_equal


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "harness_worker.py"
SCORE_SCHEMA = "score/1"


@dataclass(frozen=True)
class Limits:
    run_seconds: int
    memory_mb: int


def _tasks_root() -> Path:
    # A temporary, explicitly named fixture root may override only task packages.
    # This is for end-to-end TEST ONLY budgets; production task.json stays immutable.
    override = os.environ.get("RA_TASKS_ROOT")
    return (Path(override).expanduser().resolve() if override
            else _project_root() / "tasks")


def _project_root() -> Path:
    override = os.environ.get("RA_PROJECT_ROOT")
    if not override:
        return ROOT
    supplied = Path(override).expanduser()
    if not supplied.is_absolute():
        raise ValueError("RA_PROJECT_ROOT must be an absolute path")
    project = supplied.resolve()
    if not project.is_dir() or not (project / "tasks/bank.json").is_file() or not (
            project / "config/triggers.json").is_file():
        raise ValueError("RA_PROJECT_ROOT needs tasks/bank.json and config/triggers.json")
    return project


def _read_json(path: Path, schema: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != schema:
        raise ValueError(f"Expected {schema} in {path}")
    return data


def _task_root(task_id: str) -> Path:
    if not task_id or not all(c.isalnum() or c == "-" for c in task_id):
        raise ValueError("Invalid task ID")
    root = _tasks_root()
    path = (root / task_id).resolve()
    if path.parent != root or not (path / "task.json").is_file():
        raise KeyError(task_id)
    return path


def load_task(task_id: str) -> dict:
    path = _task_root(task_id)
    task = _read_json(path / "task.json", "task/1")
    if task.get("task_id") != task_id:
        raise ValueError("Task ID mismatch")
    if task.get("scenario") not in {"speedup", "tabular"}:
        raise ValueError("Unsupported scenario")
    profile = _read_json(path / "workspace" / "profile.json", "profile/1")
    triggers = _read_json(_project_root() / "config" / "triggers.json", "triggers/1")
    if not set(profile.get("features", [])).issubset(triggers[task["scenario"]]):
        raise ValueError("Profile uses unknown trigger features")
    return {**task, "workspace": path / "workspace", "features": profile["features"]}


def list_tasks(bank_version: str, tier: str | None = None, scenario: str | None = None) -> list[str]:
    bank = _read_json(_tasks_root() / "bank.json", "bank/1")
    if bank.get("version") != bank_version:
        raise ValueError(f"Unknown bank version: {bank_version}")
    if tier is not None and tier not in {"practice", "validation", "exam"}:
        raise ValueError("Unknown tier")
    if scenario is not None and scenario not in {"tabular", "speedup"}:
        raise ValueError("Unknown scenario")
    result = []
    for entry in bank["tasks"]:
        if tier is not None and entry["tier"] != tier:
            continue
        task_id = entry["task_id"]
        if scenario is not None and load_task(task_id)["scenario"] != scenario:
            continue
        result.append(task_id)
    return result


def _node(task_id: str, node_dir: Path) -> Path:
    node = Path(node_dir).expanduser().resolve()
    if not node.is_dir() or node.is_relative_to(_tasks_root()):
        raise ValueError("Node directory must exist outside task packages")
    if node in {ROOT, _project_root()} or not (node / "solution.py").is_file():
        raise ValueError("Node must contain solution.py")
    return node


def _sandbox_profile(task_root: Path, node: Path) -> str:
    # Rule precedence is verified in tests on macOS: these narrow allows take
    # precedence over the project-root read denial.
    quote = lambda path: json.dumps(str(path.resolve()))
    project = _project_root()
    home = Path.home().resolve()
    venv = Path(sys.prefix).resolve()
    python = Path(sys.executable).expanduser().absolute()  # Preserve the venv symlink path.
    lines = ["(version 1)", "(allow default)", "(deny network*)",
             f"(deny file-read* (subpath {quote(home)}))",
             f"(deny file-read* (subpath {quote(project)}))",
             f"(deny file-read* (subpath {quote(_tasks_root())}))"]
    if (venv.is_dir() and sys.prefix != sys.base_prefix
            and python.is_relative_to(venv)
            and not venv.is_relative_to(_tasks_root())
            and (not venv.is_relative_to(project) or venv == project / ".venv")):
        # pyvenv.cfg and site-packages are needed at interpreter startup.
        # The task hidden tree remains denied by the narrower allow rules.
        ancestors = {ancestor for ancestor in venv.parents
                     if ancestor == home or ancestor.is_relative_to(home)}
        lines.extend(f"(allow file-read* (literal {quote(path)}))"
                     for path in sorted(ancestors, key=lambda path: len(path.parts)))
        lines.append(f"(allow file-read* (subpath {quote(venv)}))")
    lines.extend([f"(allow file-read* (subpath {quote(task_root / 'workspace')}))",
                  f"(allow file-read* (subpath {quote(node)}))",
                  f"(allow file-read* (literal {quote(WORKER)}))",
                  "(deny file-write*)",
                  f"(allow file-write* (subpath {quote(node)}))",
                  "(allow file-write* (literal \"/dev/null\"))"])
    return "\n".join(lines)


def _child_limit(limits: Limits):
    def apply():
        # macOS reports RLIMIT_AS but rejects finite values (EINVAL); monitor
        # resident memory from the parent instead of silently skipping the cap.
        seconds = max(1, int(limits.run_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    return apply


def _execute(task_id: str, node: Path, mode: str, *, seed: int = 0,
             size: int = 1800, repeats: int = 7, limits: Limits | None = None,
             script_args: list[str] | None = None,
             candidate_path: Path | None = None) -> tuple[dict, dict]:
    task = load_task(task_id)
    task_root = _task_root(task_id)
    configured = task["limits"]
    limits = limits or Limits(int(configured["run_seconds"]), int(configured["memory_mb"]))
    if limits.run_seconds <= 0 or limits.memory_mb <= 0:
        raise ValueError("Limits must be positive")
    sandbox = shutil.which("sandbox-exec")
    if sys.platform != "darwin" or not sandbox:
        raise RuntimeError("macOS sandbox-exec is required; refusing unisolated candidate execution")
    candidate_path = (candidate_path or (node / "solution.py")).resolve()
    if candidate_path.parent != node:
        raise ValueError("Candidate must be directly inside node")
    command = [sandbox, "-p", _sandbox_profile(task_root, node), sys.executable,
               str(WORKER), "--mode", mode, "--candidate", str(candidate_path),
               "--workspace", str(task["workspace"]), "--function", task["deliverable"]["function"],
               "--task-kind", task.get("benchmark", {}).get("kind", "distance_matrix"),
               "--seed", str(seed), "--size", str(size), "--repeats", str(repeats),
               "--script-args", json.dumps(script_args or []),
               "--memory-mb", str(limits.memory_mb)]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(node), "HOME": str(node),
           "PYTHONPATH": os.pathsep.join((str(node), str(task["workspace"]))) }
    start = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=node, env=env, capture_output=True, text=True,
                              timeout=limits.run_seconds + 2, preexec_fn=_child_limit(limits))
        elapsed = time.perf_counter() - start
        lines = proc.stdout.splitlines()
        try:
            payload = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            payload = {}
        meta = {"exit_code": proc.returncode,
                "timed_out": proc.returncode in {-24, -25},
                "oom": proc.returncode in {-9, -11, 137}, "seconds": elapsed,
                "peak_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576,
                "stdout_tail": proc.stdout[-4096:], "stderr_tail": proc.stderr[-4096:]}
        if not payload:
            payload = {"ok": False, "cheat_flags": [], "error": {"kind": "exception",
                       "message": "Sandbox worker returned no JSON", "trace_tail": proc.stderr[-2000:]}}
        if isinstance(payload.get("error"), dict) and payload["error"].get("kind") == "oom":
            meta["oom"] = True
        return payload, meta
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "cheat_flags": [], "error": {"kind": "timeout",
                "message": "Candidate exceeded wall time", "trace_tail": ""}}, {
                "exit_code": -1, "timed_out": True, "oom": False,
                "seconds": time.perf_counter() - start, "peak_mb": 0,
                "stdout_tail": "", "stderr_tail": str(exc)[-4096:]}


def run_script(task_id: str, node_dir: Path, argv: list[str], limits: Limits | None = None) -> dict:
    node = _node(task_id, node_dir)
    if not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("argv must contain a script name")
    script = (node / argv[0]).resolve()
    if script.parent != node or not script.is_file() or script.suffix != ".py":
        raise ValueError("Script must be a Python file directly inside the node directory")
    payload, meta = _execute(task_id, script.parent, "script", limits=limits,
                             script_args=[str(item) for item in argv[1:]],
                             candidate_path=script)
    meta["stdout_tail"] = payload.get("stdout", meta["stdout_tail"])[-4096:]
    meta["stderr_tail"] = payload.get("stderr", meta["stderr_tail"])[-4096:]
    if payload.get("cheat_flags") or not payload.get("ok"):
        meta["exit_code"] = meta["exit_code"] or 1
        if payload.get("cheat_flags"):
            meta["stderr_tail"] += " Cheat flags: " + ", ".join(payload["cheat_flags"])
        elif payload.get("error"):
            meta["stderr_tail"] += " " + str(payload["error"].get("message", ""))
    return meta


def _reference_timings(task_id: str, seed: int, size: int, repeats: int) -> list[float]:
    path = _task_root(task_id) / "hidden" / "reference.py"
    spec = importlib.util.spec_from_file_location("trusted_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError("Missing trusted reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, load_task(task_id)["deliverable"]["function"])
    kind = load_task(task_id).get("benchmark", {}).get("kind", "distance_matrix")
    original_path = _task_root(task_id) / "workspace" / "original.py"
    original_spec = importlib.util.spec_from_file_location("trusted_original", original_path)
    if original_spec is None or original_spec.loader is None:
        raise ValueError("Missing original implementation")
    original_module = importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original_module)
    original = getattr(original_module, load_task(task_id)["deliverable"]["function"])
    for case in _speed_edge_cases(kind, seed):
        if not _speed_equal(kind, function(*case), original(*case)):
            raise ValueError("Reference implementation disagrees with original")
    timings = []
    for rep in range(repeats):
        case = _speed_case(kind, seed + 100000 + rep, size)
        start = time.perf_counter()
        value = function(*case)
        timings.append(time.perf_counter() - start)
        if kind == "distance_matrix":
            if not isinstance(value, list) or len(value) != size or any(
                    not isinstance(row, list) or len(row) != size for row in value):
                raise ValueError("Reference output shape changed")
        elif kind == "rolling_mean" and (not isinstance(value, list) or len(value) != size):
            raise ValueError("Reference output shape changed")
        elif kind == "edit_distance" and type(value) is not int:
            raise ValueError("Reference output type changed")
        elif kind == "max_drawdown" and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Reference output type changed")
    return timings


def _bank_v2_candidate_failure(node: Path, error: dict) -> bool:
    """Score only known sandbox failures as zero for bank-v2 candidates."""
    bank = _read_json(_tasks_root() / "bank.json", "bank/1")
    return (bank.get("version") == "bank-v2" and node.name != "n0000"
            and isinstance(error, dict) and error.get("kind") in {"timeout", "oom", "exception"})


def _score_error(split: str, seed: int, elapsed: float, queue: float,
                 error: dict, node: Path, flags: list[str] | None = None) -> dict:
    zero = _bank_v2_candidate_failure(node, error)
    return {"schema": SCORE_SCHEMA, "ok": zero, "split": split, "score": 0.0 if zero else None,
            "raw": 0.0 if zero else None, "spread": 0.0, "correct": False, "cheat_flags": flags or [],
            "error": error, "seconds": {"run": elapsed, "queue": queue}, "seed": seed}


def score(task_id: str, node_dir: Path, split: str, seed: int | None = None) -> dict:
    if split not in {"dev", "holdout"}:
        raise ValueError("Split must be dev or holdout")
    if split == "holdout" and os.environ.get("RA_EVAL") != "1":
        raise PermissionError("RA_EVAL=1 is required for holdout")
    node = _node(task_id, node_dir)
    task = load_task(task_id)
    used_seed = int(seed if seed is not None else secrets.randbits(63))
    if task["scenario"] == "tabular":
        from tabular_harness import score as score_tabular
        return score_tabular(task_id, node, split, seed=used_seed)
    benchmark = task.get("benchmark", {})
    repeats = 7 if split == "dev" else 15
    size = int(benchmark.get("dev_size" if split == "dev" else "holdout_size", 1800))
    lock_path = _project_root() / "runs" / ".timing.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    queued = time.perf_counter()
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = time.perf_counter() - queued
        started = time.perf_counter()
        try:
            payload, meta = _execute(task_id, node, "speed", seed=used_seed,
                                     size=size, repeats=repeats)
            elapsed = time.perf_counter() - started
            flags = payload.get("cheat_flags", [])
            if meta["timed_out"]:
                return _score_error(split, used_seed, elapsed, queue,
                                    {"kind": "timeout", "message": "Scoring timed out", "trace_tail": ""}, node, flags)
            if meta["oom"]:
                return _score_error(split, used_seed, elapsed, queue,
                                    {"kind": "oom", "message": "Scoring exceeded memory", "trace_tail": meta["stderr_tail"]}, node, flags)
            if not payload.get("ok") or meta["exit_code"] != 0:
                error = payload.get("error") or {"kind": "exception", "message": "Scoring worker failed",
                                                 "trace_tail": meta["stderr_tail"][-2000:]}
                return _score_error(split, used_seed, elapsed, queue, error, node, flags)
            if flags or not payload.get("correct"):
                if not flags:
                    flags = [payload.get("reason", "behavior_change")]
                return {"schema": SCORE_SCHEMA, "ok": True, "split": split, "score": 0.0,
                        "raw": 0.0, "spread": 0.0, "correct": False, "cheat_flags": flags,
                        "error": None, "seconds": {"run": elapsed, "queue": queue}, "seed": used_seed}
            ref = _reference_timings(task_id, used_seed, size, repeats)
            elapsed = time.perf_counter() - started
            medians = payload["medians"]
            raw = medians["original"] / medians["candidate"]
            normalized = statistics.median(ref) / medians["candidate"]
            q1, q3 = statistics.quantiles(payload["timings"]["candidate"], n=4)[0::2]
            spread = (q3 - q1) / medians["candidate"] * normalized
            return {"schema": SCORE_SCHEMA, "ok": True, "split": split,
                    "score": normalized, "raw": raw, "spread": spread,
                    "correct": True, "cheat_flags": [], "error": None,
                    "seconds": {"run": elapsed, "queue": queue}, "seed": used_seed}
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def profile_code(task_id: str, node_dir: Path) -> dict:
    if load_task(task_id)["scenario"] != "speedup":
        raise ValueError("Code profile is for speedup tasks")
    node = _node(task_id, node_dir)
    size = int(load_task(task_id).get("benchmark", {}).get("profile_size", 300))
    payload, meta = _execute(task_id, node, "speed", seed=91731, size=size, repeats=1)
    if not payload.get("ok") or not payload.get("correct") or payload.get("cheat_flags"):
        return {"schema": "code_profile/1", "ok": False,
                "error": payload.get("error") or payload.get("reason") or payload.get("cheat_flags")}
    return {"schema": "code_profile/1", "ok": True, "function": load_task(task_id)["deliverable"]["function"],
            "features": load_task(task_id)["features"], "timings": payload["medians"],
            "seconds": meta["seconds"]}
