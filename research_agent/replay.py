"""Re-score a saved experiment with its exact library snapshot and seed."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from library import Library, selected_settings

from .artifacts import node_directory
from .io import atomic_json, fields, read_json, utc_now
from .tree import ExperimentTree


CONTEXT_SCHEMA = "a_replay_context/1"  # A-only sidecar; published schemas stay unchanged.
CONTEXT_NAME = "replay_context.json"
MANIFEST_KEYS = (
    "run_id", "phase", "config", "rep", "bank_version", "tasks", "library",
    "agent_config", "prices", "hard_caps", "code_commit", "model",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def selected_library_settings(agent_config: Path, triggers_config: Path):
    """Pin library rules and trigger vocabulary to the audited run files."""
    with selected_settings(agent_config, triggers_config):
        yield


def _manifest_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    fields = {key: manifest.get(key) for key in MANIFEST_KEYS}
    if manifest.get("schema") == "run/2":
        fields.update(schema="run/2", models=manifest.get("models"), routing=manifest.get("routing"))
    return fields


def _file_record(path: Path) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Replay input is not a file: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def write_replay_context(
    *,
    run_dir: Path,
    task_id: str,
    library_root: Path,
    source_files: Mapping[str, Path],
    model: str,
    max_output_tokens: int,
) -> Path:
    """Seal exact inputs and node artifacts after a completed CLI task run.

    The sidecar is internal to A. It does not add fields to run/1, node/1,
    score/1 or any B/C schema. A SHA256 detects accidental/tampered changes
    after sealing, though it is not a cryptographic signature.
    """
    run_dir = Path(run_dir).resolve()
    library_root = Path(library_root).resolve()
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("schema") not in {"run/1", "run/2"}:
        raise ValueError("Unsupported run manifest schema")
    if manifest.get("run_id") != run_dir.name or task_id not in manifest.get("tasks", []):
        raise ValueError("Manifest does not describe this task run")
    tree = ExperimentTree.open(run_dir, task_id)
    if not tree.nodes:
        raise ValueError("Cannot seal an empty experiment tree")
    context_path = tree.task_dir / CONTEXT_NAME
    if context_path.exists():
        raise FileExistsError(f"Refusing to replace sealed replay context: {context_path}")
    required = {"baseline", "agent_config", "prices", "hard_caps", "triggers"}
    if required - source_files.keys():
        raise ValueError(f"Missing replay sources: {sorted(required - source_files.keys())}")
    if manifest["schema"] == "run/2" and "routing" not in source_files:
        raise ValueError("Routed replay requires its sealed policy source")
    node_records: dict[str, dict[str, Any]] = {}
    for node in tree.all_nodes():
        node_id = node["node_id"]
        node_dir = node_directory(run_dir, task_id, node_id)
        solution = node_dir / "solution.py"
        score = node_dir / "score.json"
        version = node["library_version"]
        snapshot = library_root / "snapshots" / f"{version}.json"
        node_records[node_id] = {
            "library_version": version,
            "library_snapshot_sha256": sha256_file(snapshot),
            "solution_sha256": sha256_file(solution) if solution.is_file() else None,
            "score_sha256": sha256_file(score) if score.is_file() else None,
        }
    record = {
        "schema": CONTEXT_SCHEMA,
        "created_ts": utc_now(),
        "run_id": run_dir.name,
        "task_id": task_id,
        "manifest_fields": _manifest_fields(manifest),
        "library_root": str(library_root),
        "model": model,
        "max_output_tokens": max_output_tokens,
        "sources": {label: _file_record(path) for label, path in sorted(source_files.items())},
        "tree_sha256": sha256_file(tree.path),
        "nodes": node_records,
    }
    atomic_json(context_path, record)
    return context_path


def _verify_replay_context(
    *, run_dir: Path, task_id: str, node: Mapping[str, Any], library_root: Path,
) -> dict[str, Any]:
    context = read_json(run_dir / "tasks" / task_id / CONTEXT_NAME, CONTEXT_SCHEMA)
    if context.get("run_id") != run_dir.name or context.get("task_id") != task_id:
        raise ValueError("Replay context belongs to a different run or task")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("schema") not in {"run/1", "run/2"}:
        raise ValueError("Unsupported run manifest schema")
    if context.get("manifest_fields") != _manifest_fields(manifest):
        raise ValueError("Run manifest changed since replay context was sealed")
    if context.get("library_root") != str(Path(library_root).resolve()):
        raise ValueError("Replay library root differs from the recorded run")
    if context.get("model") != manifest.get("model"):
        raise ValueError("Replay model differs from the run manifest")
    tree_path = run_dir / "tasks" / task_id / "tree.jsonl"
    if context.get("tree_sha256") != sha256_file(tree_path):
        raise ValueError("Experiment tree changed since replay context was sealed")
    for label, source in context.get("sources", {}).items():
        if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
            raise ValueError(f"Incomplete source provenance: {label}")
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Replay source changed: {label}")
    source_labels = set(context.get("sources", {}))
    if {"baseline", "agent_config", "prices", "hard_caps", "triggers"} - source_labels:
        raise ValueError("Replay context lacks mandatory source provenance")
    if manifest["schema"] == "run/2" and "routing" not in source_labels:
        raise ValueError("Routed replay context lacks route provenance")
    records = context.get("nodes", {})
    saved = records.get(node["node_id"])
    if not isinstance(saved, dict) or saved.get("library_version") != node["library_version"]:
        raise ValueError("Node's historical library version changed")
    snapshot = Path(library_root) / "snapshots" / f"{node['library_version']}.json"
    if saved.get("library_snapshot_sha256") != sha256_file(snapshot):
        raise ValueError("Historical library snapshot changed")
    node_dir = node_directory(run_dir, task_id, node["node_id"])
    for filename, key in (("solution.py", "solution_sha256"), ("score.json", "score_sha256")):
        path = node_dir / filename
        actual = sha256_file(path) if path.is_file() else None
        if saved.get(key) != actual:
            raise ValueError(f"Saved {filename} changed after the original run")
    return context


def replay_node(
    *, run_dir: Path, task_id: str, node_id: str, harness: Any,
    library_root: Path, require_provenance: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    library_root = Path(library_root).resolve()
    manifest_exists = (run_dir / "manifest.json").is_file()
    context_path = run_dir / "tasks" / task_id / CONTEXT_NAME
    if (require_provenance or manifest_exists) and not context_path.is_file():
        raise ValueError("Sealed replay context is required for a manifest-backed run")
    tree = ExperimentTree.open(run_dir, task_id)
    node = tree.get(node_id)
    if node["status"] in {"queued", "running"}:
        raise ValueError("Only completed experiment nodes can be replayed")
    context = None
    if require_provenance or manifest_exists:
        context = _verify_replay_context(run_dir=run_dir, task_id=task_id, node=node, library_root=library_root)
    integrity_verified = context is not None
    if context is None:
        library = Library.open(library_root, version=node["library_version"], frozen=True)
    else:
        sources = context["sources"]
        with selected_library_settings(sources["agent_config"]["path"], sources["triggers"]["path"]):
            library = Library.open(library_root, version=node["library_version"], frozen=True)
    cited_entries = []
    for entry_id in node["cites"]:
        try:
            cited_entries.append(library.get(entry_id))
        except KeyError as exc:
            raise ValueError(f"Node cites {entry_id}, absent from its historical library snapshot") from exc
    cited_hash = hashlib.sha256(
        json.dumps(cited_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    node_dir = node_directory(run_dir, task_id, node_id)
    solution = node_dir / "solution.py"
    if not solution.is_file():
        raise FileNotFoundError(f"No saved solution for {task_id}/{node_id}")
    code_hash = sha256_file(solution)
    previous_path = node_dir / "score.json"
    if previous_path.is_file():
        previous = read_json(previous_path, "score/1")
        if previous.get("split") != "dev":
            raise ValueError("Only dev scores may be replayed from experiment nodes")
        if previous.get("seed") is None:
            raise ValueError("Saved dev score lacks a replayable seed")
        current = fields(harness.score(task_id, node_dir, "dev", seed=previous.get("seed")))
        if current.get("schema") != "score/1" or current.get("split") != "dev":
            raise ValueError("Harness returned a non-dev or unknown score")
        tolerance = max(float(previous.get("spread") or 0), float(current.get("spread") or 0))
        comparable = previous.get("score") is not None and current.get("score") is not None
        score_matches = comparable and abs(float(current["score"]) - float(previous["score"])) <= tolerance
        ok = bool(
            current.get("ok") == previous.get("ok")
            and current.get("correct") == previous.get("correct")
            and current.get("cheat_flags") == previous.get("cheat_flags")
            and score_matches
        )
        observation: dict[str, Any] = {"type": "score", "previous": previous, "current": current, "tolerance": tolerance}
    else:
        current = fields(harness.run_script(task_id, node_dir, ["solution.py"]))
        ok = node["status"] == "failed" and (
            current.get("exit_code") != 0 or current.get("timed_out") or current.get("oom")
        )
        observation = {"type": "run", "current": current, "expected_failure": node.get("failure")}
    if sha256_file(solution) != code_hash:
        raise RuntimeError("Harness changed the saved solution during replay")
    if integrity_verified:
        _verify_replay_context(run_dir=run_dir, task_id=task_id, node=node, library_root=library_root)
    report = {
        "schema": "replay/1", "ts": utc_now(), "run_id": run_dir.name,
        "task_id": task_id, "node_id": node_id, "library_version": library.version,
        "library_snapshot_sha256": sha256_file(library_root / "snapshots" / f"{library.version}.json"),
        "cited_entries_sha256": cited_hash,
        "code_sha256": code_hash, "integrity_verified": integrity_verified,
        "replay_context_sha256": sha256_file(context_path) if integrity_verified else None,
        "ok": ok, "observation": observation,
    }
    report_dir = run_dir / "replays"
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_dir / f"{task_id}-{node_id}-{report['ts'].replace(':', '-')}.json", report)
    return report
