"""Independent, one-shot quant holdout finalizer; never used by A's dev loop."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from ledger import Ledger, TOKEN_BUCKETS
from research_agent.tree import ExperimentTree

from .backtest import equal_weight_signal, run_backtest
from .data import load_dataset
from .harness import QuantHarness, WORKER, _Protocol, _child_cpu_limit, _child_env
from .signal_worker import validate_source

SHA = re.compile(r"[0-9a-f]{64}")
NODE = re.compile(r"n[0-9]{4}")
COMMIT = re.compile(r"[0-9a-f]{7,40}")
SELECTION_FIELDS = {"schema", "task_id", "run_id", "candidate", "dev_score",
                    "dev_baseline_score", "dataset_manifest_sha256",
                    "agent_config_sha256", "prices_sha256", "manifest_sha256",
                    "task_ledger_sha256", "run_ledger_sha256", "tree_sha256",
                    "node_artifacts", "code_commit",
                    "selected_at"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Selected dev score must be finite")
    return float(value)


def _check_file(root: Path, descriptor: Any, digest: Any, label: str) -> Path:
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ValueError(f"Missing {label} descriptor")
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes project")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} path escapes project")
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise ValueError(f"Missing {label} SHA-256")
    if descriptor.get("sha256") != digest or _sha(path) != digest:
        raise ValueError(f"{label} differs from sealed SHA-256")
    return path


def _check_dev(path: Path, expected: Any) -> None:
    score = _read(path)
    if (score.get("schema") != "score/1" or score.get("split") != "dev"
            or score.get("ok") is not True or score.get("correct") is not True
            or _number(score.get("score")) != _number(expected)):
        raise ValueError("Dev score does not match frozen selection")


def _check_tree_artifacts(root: Path, run_dir: Path, task_id: str, selection: dict,
                          selected_node: str) -> None:
    """Seal every final tree node, including rejected and failed experiments."""
    tree_path = run_dir / "tasks" / task_id / "tree.jsonl"
    tree_digest = selection["tree_sha256"]
    if (tree_path.resolve(strict=True) != tree_path or not isinstance(tree_digest, str)
            or not SHA.fullmatch(tree_digest) or _sha(tree_path) != tree_digest):
        raise ValueError("Dev tree differs from frozen selection")
    tree = ExperimentTree.open(run_dir, task_id)
    artifacts = selection["node_artifacts"]
    if (not isinstance(artifacts, dict) or "n0000" not in tree.nodes
            or set(artifacts) != set(tree.nodes)
            or any(not isinstance(node_id, str) or not NODE.fullmatch(node_id)
                   for node_id in artifacts)):
        raise ValueError("Node artifacts do not match final dev tree nodes")
    if tree.best_node()["node_id"] != selected_node:
        raise ValueError("Selected node is not the best accepted dev node")
    for node_id, entry in artifacts.items():
        node_dir = run_dir / "tasks" / task_id / "nodes" / node_id
        if (not isinstance(entry, dict) or set(entry) != {"solution", "score"}
                or tree.nodes[node_id].get("code_dir") != f"nodes/{node_id}"
                or node_dir.resolve(strict=True) != node_dir or not node_dir.is_dir()):
            raise ValueError(f"Invalid node artifacts for {node_id}")
        for kind, filename in (("solution", "solution.py"), ("score", "score.json")):
            path = node_dir / filename
            descriptor = entry[kind]
            if descriptor is None:
                if path.exists() or path.is_symlink():
                    raise ValueError(f"Unsealed {kind} file for {node_id}")
                continue
            if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
                raise ValueError(f"Invalid {kind} descriptor for {node_id}")
            checked = _check_file(root, descriptor, descriptor["sha256"],
                                  f"{node_id} {kind}")
            if checked != path:
                raise ValueError(f"{kind} path does not match {node_id}")
    if (artifacts[selected_node]["solution"] != selection["candidate"]
            or artifacts[selected_node]["score"] != selection["dev_evidence"]
            or artifacts["n0000"]["score"] != selection["baseline_evidence"]):
        raise ValueError("Selected and baseline evidence differ from sealed tree artifacts")


def _dev_usage(run_dir: Path, task_id: str, node_id: str) -> dict:
    """Require a completed dev run and expose only audited resource usage."""
    ledger = Ledger.open(run_dir, task_id)
    events = ledger._events()
    run_events = Ledger.open(run_dir)._events()
    starts = [event for event in events if event["kind"] == "task_start"]
    run_starts = [event for event in run_events if event["kind"] == "run_start"]
    ends = [event for event in events if event["kind"] == "task_end"]
    run_ends = [event for event in run_events if event["kind"] == "run_end"]
    if (len(starts) != 1 or len(run_starts) != 1 or len(ends) != 1
            or len(run_ends) != 1 or ends[0].get("best_node") != node_id
            or run_ends[0].get("status") != "dev_complete"
            or run_ends[0].get("tasks_completed") != 1):
        raise ValueError("Dev run has not completed the selected node")
    calls = [event for event in events if event["kind"] == "llm_call"]
    cost = math.fsum(_number(event.get("cost")) for event in calls)
    if (cost < 0 or not math.isclose(_number(ends[0].get("total_cost")), cost, abs_tol=1e-9)
            or not math.isclose(_number(run_ends[0].get("total_cost")), cost, abs_tol=1e-9)):
        raise ValueError("Dev cost differs from sealed ledger")
    tokens = {bucket: 0 for bucket in TOKEN_BUCKETS}
    reported_cost = 0.0
    unknown_calls = 0
    for event in calls:
        usage = event.get("tokens")
        if not isinstance(usage, dict) or event.get("usage_status") not in {"reported", "unknown"}:
            raise ValueError("Dev model usage is invalid")
        reported = event["usage_status"] == "reported"
        if reported:
            reported_cost += _number(event["cost"])
        else:
            unknown_calls += 1
        for bucket in TOKEN_BUCKETS:
            amount = usage.get(bucket)
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise ValueError("Dev token usage is invalid")
            if reported:
                tokens[bucket] += amount
    complete = unknown_calls == 0
    return {"model_calls": len(calls), "reported_token_buckets": tokens,
            "total_token_buckets": tokens if complete else None,
            "model_calls_without_reported_usage": unknown_calls,
            "usage_complete": complete,
            "reported_resource_cost": reported_cost,
            "resource_cost": cost if complete else None,
            "currency": ledger.prices["currency"], "cash_cost": None}


def _preflight(harness: QuantHarness, selection_path: Path, output_dir: Path,
               task_id: str) -> tuple[dict, Path, Path, dict, dict, dict]:
    if type(harness) is not QuantHarness or task_id != "quant-demo":
        raise ValueError("Independent quant finalizer requires QuantHarness and quant-demo")
    selection_path = selection_path.expanduser().resolve(strict=True)
    run_dir = selection_path.parent
    if (selection_path.name != "selection.json"
            or output_dir.expanduser().resolve() != run_dir / "final"):
        raise ValueError("Expected run_dir/selection.json and run_dir/final")
    if (output_dir / "attempt.json").exists() or (output_dir / "report.json").exists():
        raise FileExistsError("Sealed quant holdout was already attempted")
    selection = _read(selection_path)
    if not SELECTION_FIELDS.issubset(selection) or selection.get("schema") != "quant_dev_selection/1":
        raise ValueError("Missing complete quant_dev_selection/1")
    if selection.get("holdout_scored") is not False:
        raise ValueError("Selection must explicitly record unopened holdout")
    if selection["run_id"] != run_dir.name or selection["task_id"] != task_id:
        raise ValueError("Selection belongs to another run or task")
    if not isinstance(selection["selected_at"], str) or not selection["selected_at"].strip():
        raise ValueError("Selection timestamp is missing")
    record = selection["candidate"]
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Selection needs candidate path and SHA-256")
    if not isinstance(record["path"], str) or not isinstance(record["sha256"], str):
        raise ValueError("Selection candidate has invalid path or hash")
    root = run_dir.parent.parent.resolve()
    relative = Path(record["path"])
    parts = relative.parts
    if (relative.is_absolute() or len(parts) != 7
            or parts[:5] != ("runs", run_dir.name, "tasks", task_id, "nodes")
            or not NODE.fullmatch(parts[5]) or parts[6] != "solution.py"
            or not SHA.fullmatch(record["sha256"])):
        raise ValueError("Candidate must be a frozen run node solution.py")
    candidate = _check_file(root, record, record["sha256"], "selected candidate")
    if candidate != run_dir / "tasks" / task_id / "nodes" / parts[5] / "solution.py":
        raise ValueError("Selected candidate changed or escaped its node")
    validate_source(candidate.read_text(encoding="utf-8"))
    dev = selection.get("dev_evidence")
    base = selection.get("baseline_evidence")
    if not isinstance(dev, dict) or not isinstance(base, dict):
        raise ValueError("Selection needs dev and baseline score evidence")
    dev_path = _check_file(root, dev, dev.get("sha256"), "dev score")
    base_path = _check_file(root, base, base.get("sha256"), "baseline score")
    if (dev_path != candidate.parent / "score.json"
            or base_path != run_dir / "tasks" / task_id / "nodes" / "n0000" / "score.json"):
        raise ValueError("Selection score paths do not match selected nodes")
    _check_dev(dev_path, selection["dev_score"])
    _check_dev(base_path, selection["dev_baseline_score"])
    manifest_path = (run_dir / "manifest.json").resolve(strict=True)
    manifest_sha = selection["manifest_sha256"]
    if (manifest_path.parent != run_dir or not isinstance(manifest_sha, str)
            or not SHA.fullmatch(manifest_sha) or _sha(manifest_path) != manifest_sha):
        raise ValueError("Dev manifest differs from frozen selection")
    manifest = _read(manifest_path)
    if (manifest.get("schema") != "quant_run/1" or manifest.get("phase") != "dev"
            or manifest.get("run_id") != run_dir.name
            or task_id not in manifest.get("tasks", [])):
        raise ValueError("Finalization requires a sealed quant dev manifest")
    if (manifest.get("mode") not in {"scripted", "qwen"}
            or not isinstance(manifest.get("label"), str)
            or not manifest["label"].strip()):
        raise ValueError("Dev manifest lacks an explicit run mode and data label")
    commit = selection["code_commit"]
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit) or manifest.get("code_commit") != commit:
        raise ValueError("Code commit differs from dev selection")
    if (manifest.get("split") != {
            "train_end": harness.splits.train[-1].isoformat(),
            "dev_start": harness.splits.dev[0].isoformat(),
            "dev_end": harness.splits.dev[-1].isoformat(),
            "holdout_start": harness.splits.holdout[0].isoformat(),
            "label_horizon": harness.splits.label_horizon}):
        raise ValueError("Harness split differs from sealed dev run")
    if (manifest.get("execution") != {"top_k": harness.top_k,
                                      "fee_bps": harness.fee_bps,
                                      "slippage_bps": harness.slippage_bps}):
        raise ValueError("Harness execution assumptions differ from dev run")
    if (manifest.get("limits") != {"budget_tokens": harness.budget_money,
                                   "wall_seconds": harness.wall_seconds,
                                   "run_seconds": harness.run_seconds,
                                   "memory_mb": harness.memory_mb}):
        raise ValueError("Harness limits differ from sealed dev run")
    dataset = _check_file(root, manifest.get("dataset"),
                          selection["dataset_manifest_sha256"], "dataset manifest")
    if dataset != harness.dataset_manifest or load_dataset(dataset).csv_sha256 != harness.panel.csv_sha256:
        raise ValueError("Harness dataset differs from dev selection")
    _check_file(root, manifest.get("agent_config"), selection["agent_config_sha256"], "agent config")
    _check_file(root, manifest.get("prices"), selection["prices_sha256"], "prices")
    if not harness.splits.holdout:
        raise ValueError("Sealed holdout window is empty")
    for label, filename, digest_key in (
        ("task ledger", run_dir / "tasks" / task_id / "ledger.jsonl",
         "task_ledger_sha256"),
        ("run ledger", run_dir / "ledger.jsonl", "run_ledger_sha256"),
    ):
        path = filename.resolve(strict=True)
        digest = selection[digest_key]
        if (path != filename or not path.is_file() or not isinstance(digest, str)
                or not SHA.fullmatch(digest) or _sha(path) != digest):
            raise ValueError(f"{label} differs from frozen selection")
    _check_tree_artifacts(root, run_dir, task_id, selection, parts[5])
    usage = _dev_usage(run_dir, task_id, parts[5])
    return selection, run_dir, candidate, _read(dataset), usage, manifest


def _score_candidate(harness: QuantHarness, candidate: Path):
    """Use the dev scorer's sandbox profile, limits, worker and prefix protocol."""
    sandbox = shutil.which("sandbox-exec")
    if sys.platform != "darwin" or sandbox is None:
        raise RuntimeError("macOS sandbox-exec is required for quant holdout")
    node = candidate.parent
    process = subprocess.Popen(
        [sandbox, "-p", harness._sandbox_profile(candidate), sys.executable,
         "-I", "-S", str(WORKER), "--candidate", str(candidate)],
        cwd=node, env=_child_env(node), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        preexec_fn=_child_cpu_limit(harness.run_seconds))
    protocol = _Protocol(process, deadline=time.monotonic() + harness.run_seconds + 2,
                         memory_mb=harness.memory_mb)
    try:
        if protocol.read() != {"ready": True}:
            raise RuntimeError("Candidate worker could not start")
        result = run_backtest(
            harness.panel, protocol.send,
            first_result_date=harness.splits.holdout[0],
            last_result_date=harness.splits.holdout[-1], top_k=harness.top_k,
            fee_bps=harness.fee_bps, slippage_bps=harness.slippage_bps)
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


def _write_report(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_holdout(harness: QuantHarness, selection_path: Path, output_dir: Path, *,
                     task_id: str = "quant-demo") -> dict:
    """Score a frozen dev selection once; failures consume the attempt."""
    selection_path, output_dir = Path(selection_path), Path(output_dir)
    selected, run_dir, candidate, source, usage, manifest = _preflight(
        harness, selection_path, output_dir, task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = {"schema": "quant_holdout_attempt/1", "run_id": run_dir.name,
              "task_id": task_id, "started_at": _now(),
              "selection_sha256": _sha(run_dir / "selection.json"),
              "candidate_sha256": selected["candidate"]["sha256"], "status": "attempted"}
    fd = os.open(output_dir / "attempt.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    started = time.perf_counter()
    report = {
        "schema": "quant_holdout_report/1", "run_id": run_dir.name,
        "task_id": task_id, "status": "failed", "attempt": marker,
        "label": manifest["label"], "mode": manifest["mode"],
        "selected_node": candidate.parent.name,
        "selected_origin": ("original_signal" if candidate.parent.name == "n0000"
                            else "research_candidate"),
        "selected_candidate": selected["candidate"],
        "dev_selection": {"score": selected["dev_score"],
                          "baseline_score": selected["dev_baseline_score"]},
        "dev_model_usage": usage,
        "holdout": None, "failure": None,
        "provenance": {
            "manifest_sha256": selected["manifest_sha256"],
            "task_ledger_sha256": selected["task_ledger_sha256"],
            "run_ledger_sha256": selected["run_ledger_sha256"],
            "tree_sha256": selected["tree_sha256"],
            "dataset_manifest_sha256": selected["dataset_manifest_sha256"],
            "dataset_csv_sha256": harness.panel.csv_sha256,
            "source": source["source"], "license": source["license"],
            "price_basis": source["price_basis"], "universe_policy": source["universe_policy"],
            "agent_config_sha256": selected["agent_config_sha256"],
            "prices_sha256": selected["prices_sha256"], "code_commit": selected["code_commit"]},
        "assumptions": {
            "execution": "signal after t close, trade at t+1 close, realize through t+2 close",
            "fee_bps": harness.fee_bps, "slippage_bps": harness.slippage_bps,
            "top_k": harness.top_k,
            "holdout_start": harness.splits.holdout[0].isoformat(),
            "holdout_end": harness.splits.holdout[-1].isoformat()},
        "warnings": [
            "A single sealed holdout is not repeated or tuned against.",
            "Fixed-universe close-only results omit delistings, live fills and market impact.",
            "Historical or synthetic performance is not a live-return guarantee.",
            ("Some dev model calls have unknown usage; reported token buckets and resource cost are partial, and totals are unavailable."
             if not usage["usage_complete"] else
             "Cash cost is unavailable because dev pricing uses test-only token units.")]}
    try:
        private_dir = output_dir / "candidate"
        private_dir.mkdir(exist_ok=False)
        private = private_dir / "solution.py"
        shutil.copyfile(candidate, private)
        if _sha(private) != marker["candidate_sha256"]:
            raise ValueError("Private candidate copy failed integrity check")
        candidate_result = _score_candidate(harness, private)
        if (_sha(private) != marker["candidate_sha256"]
                or _sha(candidate) != marker["candidate_sha256"]):
            raise ValueError("Candidate changed during sealed scoring")
        baseline = run_backtest(
            harness.panel, equal_weight_signal,
            first_result_date=harness.splits.holdout[0],
            last_result_date=harness.splits.holdout[-1],
            top_k=len(harness.panel.symbols), fee_bps=harness.fee_bps,
            slippage_bps=harness.slippage_bps)
        report["holdout"] = {
            "net_excess_return": candidate_result.metrics["total_return"] - baseline.metrics["total_return"],
            "candidate": dict(candidate_result.metrics), "equal_weight": dict(baseline.metrics),
            "observations": len(candidate_result.return_dates)}
        report["status"] = "scored"
    except Exception as exc:
        # Worker exception text may carry private data, so persist only generic diagnostics.
        report["failure"] = {
            "kind": ("timeout" if isinstance(exc, TimeoutError) else
                     "oom" if isinstance(exc, MemoryError) else
                     "integrity" if isinstance(exc, ValueError) else "scoring_error"),
            "type": type(exc).__name__,
            "message": "Sealed holdout attempt failed; it cannot be retried"}
    report["wall_seconds"] = max(0.0, time.perf_counter() - started)
    report["completed_at"] = _now()
    _write_report(output_dir / "report.json", report)
    return report


__all__ = ["finalize_holdout"]
