"""Untrusted tabular candidate entry point, invoked only inside sandbox-exec."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import traceback


def _audit(node: Path, workspace: Path):
    flags: set[str] = set()
    task_root = workspace.parent.resolve()
    tasks_root = task_root.parent
    node = node.resolve()
    workspace = workspace.resolve()

    def hook(event, args):
        if event.startswith("socket.") or event in {"urllib.Request", "http.client.connect"}:
            flags.add("network")
            raise PermissionError("Candidate network access denied")
        if event == "subprocess.Popen":
            flags.add("subprocess")
            raise PermissionError("Candidate subprocess access denied")
        if event not in {"open", "os.listdir", "os.scandir", "os.stat", "os.remove", "os.rename", "os.mkdir"}:
            return
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            path = Path(args[0]).resolve()
        except (OSError, ValueError):
            return
        writing = event in {"os.remove", "os.rename", "os.mkdir"}
        if event == "open":
            mode = args[1] if len(args) > 1 else "r"
            writing = (any(c in mode for c in "wax+") if isinstance(mode, str)
                       else bool(mode & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if path.is_relative_to(tasks_root) and not path.is_relative_to(workspace):
            flags.add("reads_hidden")
            raise PermissionError("Task-private path denied")
        if writing and not path.is_relative_to(node):
            flags.add("writes_outside_node")
            raise PermissionError("Write outside node denied")

    return flags, hook


def _memory_guard(memory_mb: int) -> threading.Event:
    stopped = threading.Event()

    def watch():
        while not stopped.wait(0.02):
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_bytes = rss if sys.platform == "darwin" else rss * 1024
            if rss_bytes > memory_mb * 1024 * 1024:
                os._exit(137)

    threading.Thread(target=watch, daemon=True).start()
    return stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--memory-mb", type=int, required=True)
    args = parser.parse_args()
    if args.memory_mb <= 0:
        raise ValueError("memory limit must be positive")
    # The parent sends features only. Hidden targets never enter this process.
    request = json.loads(sys.stdin.read())
    rows = request["test_rows"]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Invalid feature rows")
    flags, hook = _audit(args.candidate.parent, args.workspace)
    sys.addaudithook(hook)
    stopped = _memory_guard(args.memory_mb)
    try:
        spec = importlib.util.spec_from_file_location("candidate_solution", args.candidate)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load solution.py")
        module = importlib.util.module_from_spec(spec)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            spec.loader.exec_module(module)
            function = getattr(module, args.function)
            if not callable(function):
                raise TypeError(f"{args.function} is not callable")
            values = function(str(args.workspace / "train.csv"), rows)
        if not isinstance(values, list) or len(values) != len(rows):
            payload = {"ok": True, "valid": False, "reason": "output_shape"}
        elif any(isinstance(p, bool) or not isinstance(p, (float, int))
                 or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            payload = {"ok": True, "valid": False, "reason": "invalid_probability"}
        else:
            payload = {"ok": True, "valid": True, "predictions": [float(p) for p in values]}
    except Exception as exc:
        payload = {"ok": False, "error": {"kind": "exception", "message": str(exc),
                   "trace_tail": traceback.format_exc()[-2000:]}}
    finally:
        stopped.set()
    payload["cheat_flags"] = sorted(flags)
    print(json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    main()
