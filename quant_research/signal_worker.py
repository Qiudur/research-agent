"""Pure-signal worker. Always launch through QuantHarness's macOS sandbox.

One JSON history prefix is sent at a time. The candidate never receives the
dataset path, later prices, a model key, or the protocol request identifier.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
from pathlib import Path
import sys


MAX_SOURCE_BYTES = 30_000
MAX_MESSAGE_BYTES = 5_000_000
SAFE_BUILTINS = {
    name: getattr(__import__("builtins"), name)
    for name in ("abs", "all", "any", "bool", "dict", "enumerate", "float",
                 "int", "len", "list", "max", "min", "range", "reversed",
                 "round", "set", "sorted", "sum", "tuple", "zip")
}


class _Discard:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def validate_source(source: str) -> ast.Module:
    """Allow a single, side-effect-free-at-import signal function definition."""
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Candidate source is too large")
    try:
        tree = ast.parse(source, filename="<candidate>")
    except SyntaxError as exc:
        raise ValueError("Candidate is not valid Python") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Candidate must define only compute_signal")
    function = tree.body[0]
    args = function.args
    if (function.name != "compute_signal" or function.decorator_list or
            args.posonlyargs or len(args.args) != 1 or args.args[0].arg != "history" or
            args.vararg or args.kwarg or args.kwonlyargs or args.defaults or
            args.kw_defaults or args.args[0].annotation or function.returns):
        raise ValueError("Candidate must define compute_signal(history) only")
    banned = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
              ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom,
              ast.With, ast.AsyncWith)
    for node in ast.walk(tree):
        if isinstance(node, banned):
            raise ValueError("Candidate contains a disallowed construct")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Candidate cannot access Python internals")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Candidate cannot access Python internals")
    return tree


def _deny_io(event, args):
    if event.startswith("socket.") or event in {"subprocess.Popen", "os.system"}:
        raise PermissionError("Candidate network/process access denied")
    if event in {"open", "os.listdir", "os.scandir", "os.stat", "os.remove",
                 "os.rename", "os.mkdir"}:
        raise PermissionError("Candidate file access denied")


def _load_candidate(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = validate_source(source)
    namespace = {"__builtins__": SAFE_BUILTINS}
    sys.addaudithook(_deny_io)
    with contextlib.redirect_stdout(_Discard()):
        exec(compile(tree, "<candidate>", "exec"), namespace)
    function = namespace.get("compute_signal")
    if not callable(function):
        raise ValueError("Candidate has no compute_signal function")
    return function


def _response(payload: dict) -> None:
    print(json.dumps(payload, allow_nan=False, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        signal = _load_candidate(args.candidate)
    except Exception:
        _response({"ready": False, "error": "candidate_load_failed"})
        return 1
    _response({"ready": True})
    for line in sys.stdin.buffer:
        if len(line) > MAX_MESSAGE_BYTES:
            _response({"error": "request_too_large"})
            return 1
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("history"), dict) \
                    or not isinstance(request.get("id"), str):
                raise ValueError("Malformed worker request")
            history = {symbol: tuple(prices) for symbol, prices in request["history"].items()}
            with contextlib.redirect_stdout(_Discard()):
                scores = signal(history)
            _response({"id": request["id"], "scores": scores})
        except Exception:
            _response({"id": request.get("id") if isinstance(request, dict) else None,
                       "error": "candidate_execution_failed"})
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
