"""Dev-only quant task adapter for the existing research experiment loop.

The candidate runs only inside macOS sandbox-exec. The parent owns the price
panel; the child receives one past-only JSON history prefix at a time and no
market-data path. This module never scores the sealed holdout interval.
"""

from __future__ import annotations

from datetime import date
import json
import math
import os
from pathlib import Path
import resource
import secrets
import select
import shutil
import subprocess
import sys
import time
from typing import Mapping

from .backtest import equal_weight_signal, run_backtest
from .data import load_dataset
from .signal_worker import validate_source
from .splits import make_splits


WORKER = Path(__file__).with_name("signal_worker.py")
PROJECT = Path(__file__).resolve().parents[1]
MAX_PROTOCOL_BYTES = 5_000_000
QUANT_FEATURES = frozenset({"time_series", "cross_sectional_rank", "lookback_window",
                            "high_turnover", "missing_prices"})


def _quote(path: Path) -> str:
    return json.dumps(str(path.resolve()))


def _child_cpu_limit(seconds: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    return apply


def _child_env(node: Path) -> dict[str, str]:
    """Explicit allowlist: never inherit API keys, credentials or PYTHONPATH."""
    return {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(node), "TMPDIR": str(node),
    }


class _Protocol:
    def __init__(self, process: subprocess.Popen, *, deadline: float, memory_mb: int):
        self.process = process
        self.deadline = deadline
        self.memory_mb = memory_mb
        self.buffer = bytearray()

    def _check_memory(self) -> None:
        # RLIMIT_AS is rejected by macOS for Python. Observe resident memory
        # from the parent instead; the wall/CPU limits remain independent.
        try:
            result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(self.process.pid)],
                                    capture_output=True, text=True, timeout=1, check=False)
            if result.returncode == 0 and result.stdout.strip() \
                    and int(result.stdout.strip()) > self.memory_mb * 1024:
                raise MemoryError("Candidate exceeded memory limit")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Cannot verify candidate memory use") from exc

    def read(self) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("Candidate protocol stdout is missing")
        while b"\n" not in self.buffer:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Candidate exceeded wall time")
            ready, _, _ = select.select([self.process.stdout], [], [], min(0.25, remaining))
            self._check_memory()
            if not ready:
                continue
            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk:
                raise RuntimeError("Candidate worker ended without a response")
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_PROTOCOL_BYTES:
                raise RuntimeError("Candidate protocol output is too large")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        try:
            response = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Candidate protocol response is not JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Candidate protocol response must be an object")
        return response

    def send(self, history: Mapping[str, tuple[float, ...]]) -> Mapping[str, float]:
        if self.process.stdin is None:
            raise RuntimeError("Candidate protocol stdin is missing")
        request_id = secrets.token_hex(12)
        request = json.dumps({"id": request_id, "history": dict(history)},
                             allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(request) > MAX_PROTOCOL_BYTES:
            raise ValueError("Candidate history prefix is too large")
        try:
            self.process.stdin.write(request)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Candidate worker closed before the next signal") from exc
        response = self.read()
        if response.get("id") != request_id or response.get("error"):
            raise RuntimeError("Candidate signal execution failed")
        if not isinstance(response.get("scores"), dict):
            raise ValueError("Candidate signal must return a score mapping")
        return response["scores"]


class QuantHarness:
    """Injectable task/1 interface; only the development window is scoreable."""

    def __init__(
        self, task_root: Path, dataset_manifest: Path, *,
        train_end: date, dev_start: date, dev_end: date, holdout_start: date,
        budget_money: float, wall_seconds: int, top_k: int = 1,
        fee_bps: float = 0.0, slippage_bps: float = 0.0,
        run_seconds: int = 20, memory_mb: int = 512, label_horizon: int = 1,
    ) -> None:
        self.task_root = Path(task_root).expanduser().resolve(strict=True)
        self.workspace = (self.task_root / "workspace").resolve(strict=True)
        if self.workspace.parent != self.task_root:
            raise ValueError("Quant workspace must be directly inside task_root")
        original = (self.workspace / "original.py").resolve(strict=True)
        profile_path = (self.workspace / "profile.json").resolve(strict=True)
        if original.parent != self.workspace or profile_path.parent != self.workspace:
            raise ValueError("Quant workspace files must remain in workspace")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or profile.get("schema") != "profile/1":
            raise ValueError("Quant profile must be profile/1")
        features = profile.get("features", [])
        if not isinstance(features, list) \
                or any(not isinstance(feature, str) or feature not in QUANT_FEATURES
                       for feature in features) or len(features) != len(set(features)):
            raise ValueError("Quant profile contains invalid trigger features")
        self.features = tuple(features)
        self.profile_summary = profile.get("summary", {})
        self.dataset_manifest = Path(dataset_manifest).expanduser().resolve(strict=True)
        if self.dataset_manifest.is_relative_to(self.workspace):
            raise ValueError("Dataset manifest cannot be in visible workspace")
        self.panel = load_dataset(self.dataset_manifest)
        self.splits = make_splits(
            self.panel.dates, train_end=train_end, dev_start=dev_start,
            dev_end=dev_end, holdout_start=holdout_start,
            label_horizon=label_horizon,
        )
        if self.panel.dates.index(self.splits.dev[0]) < 2:
            raise ValueError("Development window lacks two prior price dates")
        if type(top_k) is not int or not 1 <= top_k <= len(self.panel.symbols):
            raise ValueError("top_k must be within the fixed universe")
        if type(wall_seconds) is not int or wall_seconds < 1 \
                or type(run_seconds) is not int or run_seconds < 1 \
                or type(memory_mb) is not int or memory_mb < 1:
            raise ValueError("Quant time and memory limits must be positive integers")
        if isinstance(budget_money, bool) or not isinstance(budget_money, (int, float)) \
                or not math.isfinite(budget_money) or budget_money <= 0:
            raise ValueError("Quant model budget must be positive and finite")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in (fee_bps, slippage_bps)):
            raise ValueError("Quant trading costs must be finite and nonnegative")
        self.budget_money = float(budget_money)
        self.wall_seconds = wall_seconds
        self.top_k = top_k
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.run_seconds = run_seconds
        self.memory_mb = memory_mb

    def load_task(self, task_id: str) -> dict:
        if task_id != "quant-demo":
            raise KeyError(task_id)
        return {
            "schema": "task/1", "task_id": task_id, "scenario": "quant",
            "title": "Research a daily cross-asset signal",
            "description": "Rank a fixed demo universe using past closes only; test on development dates.",
            "metric": {"name": "net_excess_return", "higher_is_better": True},
            "deliverable": {"file": "solution.py", "function": "compute_signal",
                            "signature": "compute_signal(history) -> dict[str, float]",
                            "returns": "One finite score per symbol"},
            "budget": {"money": self.budget_money, "wall_seconds": self.wall_seconds},
            "limits": {"run_seconds": self.run_seconds, "memory_mb": self.memory_mb},
            "workspace": self.workspace, "features": list(self.features),
        }

    def _node_file(self, node_dir: Path) -> Path:
        node = Path(node_dir).expanduser().resolve(strict=True)
        if not node.is_dir() or node.is_relative_to(self.task_root):
            raise ValueError("Candidate node must be outside task package")
        candidate = (node / "solution.py").resolve(strict=True)
        if candidate.parent != node or not candidate.is_file():
            raise ValueError("Candidate must be a direct solution.py file")
        return candidate

    def run_script(self, task_id: str, node_dir: Path, argv: list[str]) -> dict:
        self.load_task(task_id)
        if argv != ["solution.py"]:
            raise ValueError("Quant task accepts only solution.py")
        started = time.perf_counter()
        try:
            source = self._node_file(node_dir).read_text(encoding="utf-8")
            validate_source(source)
            error = ""
        except (OSError, UnicodeError, ValueError) as exc:
            error = str(exc)[:300]
        return {"exit_code": 1 if error else 0, "timed_out": False, "oom": False,
                "seconds": time.perf_counter() - started, "stdout_tail": "",
                "stderr_tail": error}

    def _sandbox_profile(self, candidate: Path) -> str:
        home = Path.home().resolve()
        venv = Path(sys.prefix).resolve()
        python = Path(sys.executable).expanduser().absolute()
        lines = ["(version 1)", "(allow default)", "(deny network*)",
                 f"(deny file-read* (subpath {_quote(home)}))",
                 f"(deny file-read* (subpath {_quote(PROJECT)}))",
                 f"(deny file-read* (subpath {_quote(self.task_root)}))",
                 f"(deny file-read* (subpath {_quote(self.dataset_manifest.parent)}))",
                 "(deny file-write*)",
                 '(allow file-write* (literal "/dev/null"))']
        if sys.prefix != sys.base_prefix and python.is_relative_to(venv):
            # Python's startup needs the venv even when it lives under HOME.
            lines.append(f"(allow file-read* (subpath {_quote(venv)}))")
            lines.extend(f"(allow file-read* (literal {_quote(parent)}))"
                         for parent in sorted(venv.parents, key=lambda item: len(item.parts)))
        lines.extend([f"(allow file-read* (literal {_quote(candidate)}))",
                      f"(allow file-read* (literal {_quote(WORKER)}))"])
        return "\n".join(lines)

    def _run_candidate(self, candidate: Path):
        sandbox = shutil.which("sandbox-exec")
        if sys.platform != "darwin" or sandbox is None:
            raise RuntimeError("macOS sandbox-exec is required; refusing unisolated quant scoring")
        node = candidate.parent
        command = [sandbox, "-p", self._sandbox_profile(candidate), sys.executable,
                   "-I", "-S", str(WORKER), "--candidate", str(candidate)]
        process = subprocess.Popen(
            command, cwd=node, env=_child_env(node), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            preexec_fn=_child_cpu_limit(self.run_seconds),
        )
        protocol = _Protocol(process, deadline=time.monotonic() + self.run_seconds + 2,
                             memory_mb=self.memory_mb)
        try:
            ready = protocol.read()
            if ready != {"ready": True}:
                raise RuntimeError("Candidate worker could not start")
            result = run_backtest(
                self.panel, protocol.send,
                first_result_date=self.splits.dev[0],
                last_result_date=self.splits.dev[-1], top_k=self.top_k,
                fee_bps=self.fee_bps, slippage_bps=self.slippage_bps,
            )
            assert process.stdin is not None
            process.stdin.close()
            process.wait(timeout=max(1.0, protocol.deadline - time.monotonic()))
            if process.returncode != 0:
                raise RuntimeError("Candidate worker failed after scoring")
            return result
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def score(self, task_id: str, node_dir: Path, split: str,
              seed: int | None = None) -> dict:
        self.load_task(task_id)
        if split != "dev":
            raise PermissionError("QuantHarness does not score sealed holdout dates")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("Quant seed must be an integer")
        if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
            raise RuntimeError("macOS sandbox-exec is required; refusing unisolated quant scoring")
        started = time.perf_counter()
        candidate = self._node_file(node_dir)
        check = self.run_script(task_id, node_dir, ["solution.py"])
        if check["exit_code"] != 0:
            raise ValueError(check["stderr_tail"])
        baseline = run_backtest(
            self.panel, equal_weight_signal,
            first_result_date=self.splits.dev[0],
            last_result_date=self.splits.dev[-1], top_k=len(self.panel.symbols),
            fee_bps=self.fee_bps, slippage_bps=self.slippage_bps,
        )
        try:
            result = self._run_candidate(candidate)
        except (ValueError, RuntimeError, TimeoutError, MemoryError, OSError) as exc:
            kind = ("timeout" if isinstance(exc, TimeoutError) else
                    "oom" if isinstance(exc, MemoryError) else "exception")
            return {
                "schema": "score/1", "ok": False, "split": "dev", "score": None,
                "raw": None, "spread": 0.0, "correct": False, "cheat_flags": [],
                "error": {"kind": kind, "message": str(exc)[:300], "trace_tail": ""},
                "seconds": {"run": time.perf_counter() - started, "queue": 0.0},
                "seed": 0 if seed is None else seed,
            }
        excess = result.metrics["total_return"] - baseline.metrics["total_return"]
        return {
            "schema": "score/1", "ok": True, "split": "dev", "score": excess,
            "raw": result.metrics["total_return"], "spread": 0.0,
            "correct": True, "cheat_flags": [], "error": None,
            "seconds": {"run": time.perf_counter() - started, "queue": 0.0},
            "seed": 0 if seed is None else seed,
            "quant_metrics": {"candidate": dict(result.metrics),
                              "equal_weight": dict(baseline.metrics),
                              "dataset_sha256": self.panel.csv_sha256,
                              "dev_start": self.splits.dev[0].isoformat(),
                              "dev_end": self.splits.dev[-1].isoformat()},
        }


__all__ = ["QuantHarness"]
