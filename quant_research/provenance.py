"""Seal and recheck every trusted input used by a quant development run.

Hashes detect changes after capture; they are not signatures or an authority to
choose new input paths. Prompt paths are therefore checked against the sealed
agent configuration, and task/config paths retain their published identities.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


_ROLES = ("planner", "executor", "diagnoser", "summarizer")
REQUIRED_INPUTS = frozenset({
    "agent_config", "prices", "hard_caps", "triggers",
    *(f"prompt_{role}" for role in _ROLES),
    "task_original", "task_profile",
})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _root(project_root: Path) -> Path:
    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Project root must be an existing directory") from exc
    if not root.is_dir():
        raise ValueError("Project root must be an existing directory")
    return root


def _relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} path must be a nonempty relative string")
    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} path is invalid") from exc
    if path.is_absolute() or ".." in path.parts or str(path) == "." or "\x00" in value:
        raise ValueError(f"{label} path must remain project-relative")
    return path


def _resolve(root: Path, relative: Path, label: str) -> Path:
    try:
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{label} must be a project-local file")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} must be an existing project-local file") from exc
    return path


def _capture_path(root: Path, value: Path, label: str) -> Path:
    try:
        path = Path(value).expanduser()
        if ".." in path.parts:
            raise ValueError("Parent traversal is disallowed")
        if path.is_absolute():
            relative = path.relative_to(root)
        else:
            relative = path
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a project-local path") from exc
    return _resolve(root, relative, label)


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read {label}") from exc


def _config(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Sealed agent configuration is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != "agent/1":
        raise ValueError("Expected a sealed agent/1 configuration")
    prompts = value.get("prompts")
    if not isinstance(prompts, dict) or set(prompts) != set(_ROLES):
        raise ValueError("Agent configuration must identify exactly four role prompts")
    return value


def _task_workspace(root: Path, original: Path, profile: Path) -> Path:
    parts = original.relative_to(root).parts
    if (len(parts) != 6 or parts[0] != "runs" or not parts[1]
            or parts[2:] != ("task_package", "quant-demo", "workspace", "original.py")
            or profile != original.parent / "profile.json"):
        raise ValueError("Task inputs must share the quant run's task workspace")
    return original.parent


def capture_inputs(
    project_root: Path, config: dict, *, agent_config: Path, prices: Path,
    hard_caps: Path, triggers: Path, task_workspace: Path,
) -> dict[str, dict[str, str]]:
    """Capture source identities before execution, including every role prompt."""
    root = _root(project_root)
    agent_path = _capture_path(root, agent_config, "agent_config")
    agent_raw = _read(agent_path, "agent_config")
    sealed_config = _config(agent_raw)
    if not isinstance(config, dict) or config != sealed_config:
        raise ValueError("Loaded agent configuration differs from the source file")
    try:
        workspace = Path(task_workspace).expanduser()
    except (TypeError, ValueError) as exc:
        raise ValueError("Task workspace is invalid") from exc
    paths = {
        "agent_config": agent_path,
        "prices": _capture_path(root, prices, "prices"),
        "hard_caps": _capture_path(root, hard_caps, "hard_caps"),
        "triggers": _capture_path(root, triggers, "triggers"),
        "task_original": _capture_path(root, workspace / "original.py", "task_original"),
        "task_profile": _capture_path(root, workspace / "profile.json", "task_profile"),
    }
    for role in _ROLES:
        relative = _relative(sealed_config["prompts"][role], f"prompt_{role}")
        paths[f"prompt_{role}"] = _resolve(root, relative, f"prompt_{role}")
    records = {
        label: {"path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(
                    agent_raw if label == "agent_config" else _read(path, label)).hexdigest()}
        for label, path in paths.items()
    }
    verify_inputs(root, records)
    return records


def verify_inputs(project_root: Path, inputs: dict) -> None:
    """Reject changed bytes, substituted identities, and missing/extra inputs."""
    root = _root(project_root)
    if not isinstance(inputs, dict) or set(inputs) != REQUIRED_INPUTS:
        raise ValueError("Quant provenance must contain exactly the required inputs")
    paths: dict[str, Path] = {}
    contents: dict[str, bytes] = {}
    for label in sorted(REQUIRED_INPUTS):
        descriptor = inputs[label]
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError(f"Invalid {label} input descriptor")
        relative = _relative(descriptor["path"], label)
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"Invalid {label} SHA-256")
        path = _resolve(root, relative, label)
        raw = _read(path, label)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{label} differs from its sealed SHA-256")
        paths[label], contents[label] = path, raw
    config = _config(contents["agent_config"])
    for label, relative in (("hard_caps", "config/hard_caps.json"),
                            ("triggers", "config/triggers.json")):
        expected = _resolve(root, Path(relative), label)
        if paths[label] != expected or _relative(inputs[label]["path"], label) != Path(relative):
            raise ValueError(f"{label} input path was substituted")
    for role in _ROLES:
        label = f"prompt_{role}"
        expected = _resolve(root, _relative(config["prompts"][role], label), label)
        if paths[label] != expected:
            raise ValueError(f"{label} path differs from the sealed agent configuration")
    _task_workspace(root, paths["task_original"], paths["task_profile"])


__all__ = ["REQUIRED_INPUTS", "capture_inputs", "verify_inputs"]
