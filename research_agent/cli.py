"""Command-line entry points for A's task runner and deterministic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from library import Library

from .io import read_json
from .model import ModelClient
from .replay import replay_node, selected_library_settings, sha256_file, write_replay_context
from .roles import RoleAgents
from .runner import TaskRunner


def _project_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    resolved_root = root.resolve()
    resolved = (path if path.is_absolute() else resolved_root / path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Manifest path escapes project root: {value}")
    return resolved


def _manifest_file(project_root: Path, manifest: dict, key: str, schema: str) -> tuple[Path, dict]:
    descriptor = manifest.get(key)
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ValueError(f"Manifest {key}.path is missing")
    expected = descriptor.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Manifest {key}.sha256 is missing or a placeholder; C must seal it before a run")
    path = _project_path(project_root, descriptor["path"])
    data = read_json(path, schema)
    if sha256_file(path) != expected.lower():
        raise ValueError(f"Manifest {key}.sha256 does not match {path}")
    return path, data


def _price_ready(prices: dict, model: str) -> None:
    rates = prices.get("models", {}).get(model)
    if not isinstance(rates, dict) or any(rates.get(key) is None for key in
                                          ("input", "cached_input", "output", "reasoning")):
        raise ValueError(f"Price table has no complete rates for model {model!r}")
    if not isinstance(prices.get("unit_tokens"), int) or prices["unit_tokens"] <= 0:
        raise ValueError("Price table unit_tokens must be positive")


def _model_name(manifest: dict, requested: str | None) -> str:
    name = manifest.get("model")
    if not isinstance(name, str) or not name.strip() or name == "开赛给的模型名":
        raise ValueError("Set a real model in manifest.model before running")
    if requested is not None and requested != name:
        raise ValueError("--model may only repeat manifest.model; update the C-owned manifest for a different model")
    return name


def _harness():
    try:
        import harness
    except ImportError as exc:
        raise RuntimeError("B-owned harness module is missing; install it before running a task") from exc
    return harness


def _ledger(run_dir: Path, task_id: str):
    try:
        from ledger import Ledger
    except ImportError as exc:
        raise RuntimeError("C-owned ledger module is missing; install it before running a task") from exc
    return Ledger.open(run_dir, task_id)


def _manifest(run_dir: Path) -> dict:
    data = read_json(run_dir / "manifest.json", "run/1")
    if data.get("run_id") != run_dir.name:
        raise ValueError("Manifest run_id does not match --run-dir")
    return data


def _library_root(project_root: Path, manifest: dict) -> Path:
    value = manifest.get("library", {}).get("root")
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest must identify library.root")
    return _project_path(project_root, value)


def _run(args: argparse.Namespace) -> dict:
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    manifest = _manifest(run_dir)
    if args.task_id not in manifest.get("tasks", []):
        raise ValueError("Task ID is not listed in the run manifest")
    config_path, config = _manifest_file(project_root, manifest, "agent_config", "agent/1")
    if args.config is not None and _project_path(project_root, str(args.config)) != config_path:
        raise ValueError("--config may only repeat manifest.agent_config.path; update the C-owned manifest and hash for a tuned config")
    model_name = _model_name(manifest, args.model)
    prices_path, prices = _manifest_file(project_root, manifest, "prices", "prices/1")
    _price_ready(prices, model_name)
    hard_caps_path = _project_path(project_root, "config/hard_caps.json")
    hard_caps = read_json(hard_caps_path, "hard_caps/1")
    if manifest.get("hard_caps", {}).get("budget_ratio") != hard_caps.get("budget_ratio"):
        raise ValueError("Manifest hard cap does not match config/hard_caps.json")
    baseline_path = args.baseline_json.resolve(strict=True)
    baseline = read_json(baseline_path, "score/1")
    source_files = {
        "baseline": baseline_path,
        "agent_config": config_path,
        "prices": prices_path,
        "hard_caps": hard_caps_path,
        "triggers": _project_path(project_root, "config/triggers.json"),
    }
    for role, value in config["prompts"].items():
        source_files[f"prompt_{role}"] = _project_path(project_root, value)
    source_hashes = {label: sha256_file(path) for label, path in source_files.items()}
    ledger = _ledger(run_dir, args.task_id)
    library_root = _library_root(project_root, manifest)
    with selected_library_settings(config_path, source_files["triggers"]):
        library = Library.open(
            library_root,
            frozen=bool(manifest["library"].get("frozen", False)),
        )
        start_version = manifest["library"].get("start_version")
        if not isinstance(start_version, str) or not re.fullmatch(r"lib-(0|[1-9][0-9]*)", start_version):
            raise ValueError("Manifest library.start_version is invalid")
        if int(library.version[4:]) < int(start_version[4:]):
            raise ValueError("Library is older than manifest library.start_version")
        if (manifest["library"].get("frozen") or args.task_id == manifest["tasks"][0]) and library.version != start_version:
            raise ValueError("Library start version differs from the manifest at the first or frozen task")
        roles = RoleAgents(
            ModelClient(model_name, ledger, max_output_tokens=args.max_output_tokens),
            config, project_root=project_root,
        )
        result = TaskRunner(
            run_dir=run_dir, task_id=args.task_id, harness=_harness(), ledger=ledger,
            roles=roles, library=library, baseline_score=baseline,
            config=config, hard_caps=hard_caps,
        ).run()
    for label, path in source_files.items():
        if sha256_file(path) != source_hashes[label]:
            raise RuntimeError(f"Run source changed while the task was executing: {label}")
    if _manifest(run_dir).get("model") != model_name:
        raise RuntimeError("Manifest model changed while the task was executing")
    opening_plan = run_dir / "tasks" / args.task_id / "opening_plan.json"
    sealed_files = dict(source_files)
    if opening_plan.is_file():
        sealed_files["opening_plan"] = opening_plan
    write_replay_context(
        run_dir=run_dir, task_id=args.task_id, library_root=library_root,
        source_files=sealed_files, model=model_name,
        max_output_tokens=args.max_output_tokens,
    )
    return vars(result)


def _replay(args: argparse.Namespace) -> dict:
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    manifest = _manifest(run_dir)
    if args.task_id not in manifest.get("tasks", []):
        raise ValueError("Task ID is not listed in the run manifest")
    _manifest_file(project_root, manifest, "agent_config", "agent/1")
    _manifest_file(project_root, manifest, "prices", "prices/1")
    return replay_node(
        run_dir=run_dir, task_id=args.task_id, node_id=args.node_id,
        harness=_harness(), library_root=_library_root(project_root, manifest),
        require_provenance=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-agent", description="Run or replay the A workstream")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="Run one task in a C-created run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--task-id", required=True)
    run.add_argument("--baseline-json", type=Path, required=True, help="B/C-provided dev score/1 for n0000")
    run.add_argument("--config", type=Path, help="Must match manifest.agent_config.path and SHA256")
    run.add_argument("--model", help="Must match manifest.model")
    run.add_argument("--max-output-tokens", type=int, default=30000)
    replay = subcommands.add_parser("replay", help="Re-score a saved node with its library version and seed")
    replay.add_argument("--run-dir", type=Path, required=True)
    replay.add_argument("--task-id", required=True)
    replay.add_argument("--node-id", required=True)
    args = parser.parse_args(argv)
    try:
        result = _run(args) if args.command == "run" else _replay(args)
    except Exception as exc:
        print(f"research-agent: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
