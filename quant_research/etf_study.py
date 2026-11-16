"""Preregistered, deterministic ETF comparisons without model or network calls.

The grid is ranked on three development windows. Its complete selection and
input hashes are written before the three named holdout series are evaluated.
An output directory is single use, including when a holdout attempt fails.
This is local evidence management, not protection against rerunning the study
in a different directory or against someone who already inspected the data.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, pstdev
from typing import Mapping

from research_agent import io as agent_io
from research_agent.io import atomic_json, utc_now

from .backtest import equal_weight_signal, run_backtest
from .data import PricePanel, load_dataset
from .signals import trailing_momentum


_SELECTION_RULE = "max_min_dev_excess_then_mean_then_id"
_KINDS = {"momentum", "reversal", "low_volatility"}
_LIMITATIONS = [
    "A fixed universe of surviving ETFs does not establish point-in-time universe membership or remove survivorship bias.",
    "Close-price data and the declared price basis may omit distributions or contain vendor adjustments; inspect the dataset manifest.",
    "Daily-close execution with fixed fees and slippage is a research assumption, not evidence of achievable fills or capacity.",
    "The grid was selected on development data; a single historical holdout is descriptive evidence, not statistical proof or investment advice.",
    "The exclusive holdout marker applies only to this output directory; copies and previously viewed data cannot be made unseen.",
    "No model or Agent comparison was run. A later comparison needs a newly frozen protocol and an unopened final interval.",
]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_document(path: Path, schema: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise ValueError(f"Expected {schema}: {path}")
    return raw, document


def _descriptor(path: Path, logical_path: str, raw: bytes | None = None) -> dict:
    return {"path": logical_path, "sha256": _sha(path.read_bytes() if raw is None else raw)}


def _verify_inputs(provenance: dict, actual_paths: list[Path]) -> None:
    descriptors = [provenance[name] for name in
                   ("dataset_manifest", "dataset_csv", "protocol")] + provenance["source_files"]
    for descriptor, actual_path in zip(descriptors, actual_paths, strict=True):
        try:
            digest = _sha(actual_path.read_bytes())
        except OSError as exc:
            raise ValueError(f"Sealed input is unavailable: {descriptor['path']}") from exc
        if digest != descriptor["sha256"]:
            raise ValueError(f"Sealed input SHA-256 changed: {descriptor['path']}")


def _day(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must be an ISO calendar date")
    return parsed


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _candidate(value: object, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"id", "kind", "lookback"}:
        raise ValueError(f"{label} needs exactly id, kind and lookback")
    candidate_id = value["id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip() or candidate_id.strip() != candidate_id:
        raise ValueError(f"{label} needs a nonempty candidate id")
    if not isinstance(value["kind"], str) or value["kind"] not in _KINDS:
        raise ValueError(f"{label} has unsupported kind")
    _positive_integer(value["lookback"], f"{label}.lookback")
    return dict(value)


def _window(panel: PricePanel, value: object, window_id: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{window_id} must be a window object")
    start = _day(value.get("start"), f"{window_id}.start")
    end = _day(value.get("end"), f"{window_id}.end")
    if start not in panel.dates or end not in panel.dates:
        raise ValueError(f"{window_id} preregistered start and end must both be observed dates; no silent window truncation")
    indices = [index for index, day in enumerate(panel.dates) if start <= day <= end]
    if len(indices) < 2:
        raise ValueError(f"{window_id} needs at least two observations in inclusive date bounds")
    return {"id": window_id, "start": panel.dates[indices[0]].isoformat(),
            "end": panel.dates[indices[-1]].isoformat(), "observations": len(indices)}


def _validate_protocol(panel: PricePanel, protocol: dict) -> tuple[dict, list[dict]]:
    symbols = protocol.get("symbols")
    if not isinstance(symbols, list) or symbols != list(panel.symbols):
        raise ValueError("Protocol symbols must exactly match the sorted dataset universe")
    if protocol.get("selection_rule") != _SELECTION_RULE:
        raise ValueError("Unsupported development selection_rule")
    data_start = _day(protocol.get("data_start"), "data_start")
    data_end = _day(protocol.get("data_end"), "data_end")
    warmup_end = _day(protocol.get("warmup_end"), "warmup_end")
    if not data_start <= panel.dates[0] <= warmup_end < data_end or panel.dates[-1] > data_end:
        raise ValueError("Dataset and warmup must lie within protocol data bounds")
    top_k = _positive_integer(protocol.get("top_k"), "top_k")
    if top_k > len(panel.symbols):
        raise ValueError("top_k exceeds the symbol universe")
    horizon = _positive_integer(protocol.get("label_horizon"), "label_horizon")
    if horizon != 2:
        raise ValueError("label_horizon must be 2 for this backtest's t+2 timing")
    for field in ("fee_bps", "slippage_bps"):
        value = protocol.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be a finite nonnegative number")
    fixed_lookback = _positive_integer(protocol.get("fixed_momentum_lookback"), "fixed_momentum_lookback")
    values = protocol.get("candidates")
    if not isinstance(values, list) or not values:
        raise ValueError("Protocol needs a nonempty predeclared candidate grid")
    candidates = [_candidate(value, f"candidates[{index}]") for index, value in enumerate(values)]
    if len({value["id"] for value in candidates}) != len(candidates):
        raise ValueError("Candidate ids must be unique")
    development = protocol.get("development_windows")
    if not isinstance(development, list) or len(development) != 3:
        raise ValueError("Protocol needs exactly three development windows")
    ids = [value.get("id") if isinstance(value, dict) else None for value in development]
    if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != 3 or "holdout" in ids:
        raise ValueError("Development window ids must be unique nonempty strings")
    windows = [_window(panel, value, value["id"]) for value in development]
    holdout = _window(panel, protocol.get("holdout"), "holdout")
    for value in development + [protocol["holdout"]]:
        if not data_start <= _day(value["start"], "window.start") <= _day(value["end"], "window.end") <= data_end:
            raise ValueError("Every evaluation window must lie within protocol data bounds")
    combined = windows + [holdout]
    for previous, current in zip(combined, combined[1:]):
        previous_end = panel.dates.index(date.fromisoformat(previous["end"]))
        current_start = panel.dates.index(date.fromisoformat(current["start"]))
        if current_start - previous_end - 1 < horizon:
            raise ValueError("Development and holdout windows need chronological label_horizon gaps")
    first_start = panel.dates.index(date.fromisoformat(windows[0]["start"]))
    max_lookback = max([fixed_lookback] + [value["lookback"] for value in candidates])
    # At result index i the signal sees i-1 observations, through index i-2.
    if first_start < max_lookback + 2 or warmup_end >= date.fromisoformat(windows[0]["start"]):
        raise ValueError("Insufficient warmup for every predeclared lookback and t+2 timing")
    warmup_dates = [day for day in panel.dates if day <= warmup_end]
    if len(warmup_dates) < max_lookback + 1:
        raise ValueError("Insufficient observations before warmup_end")
    return {"warmup_end": warmup_dates[-1].isoformat(), "development": windows, "holdout": holdout}, candidates


def _signal(candidate: dict):
    lookback = candidate["lookback"]
    kind = candidate["kind"]

    def compute(history: Mapping[str, tuple[float, ...]]) -> dict[str, float]:
        if any(len(closes) <= lookback for closes in history.values()):
            raise ValueError("Candidate has insufficient trailing history")
        if kind == "momentum":
            return trailing_momentum(history, lookback=lookback)
        if kind == "reversal":
            return {symbol: -score for symbol, score in trailing_momentum(history, lookback=lookback).items()}
        return {symbol: -pstdev([right / left - 1.0 for left, right in
                                zip(closes[-lookback - 1:-1], closes[-lookback:])])
                for symbol, closes in history.items()}

    return compute


def _run_window(panel: PricePanel, candidate: dict | None, window: dict, protocol: dict) -> dict:
    return run_backtest(
        panel, equal_weight_signal if candidate is None else _signal(candidate),
        first_result_date=date.fromisoformat(window["start"]),
        last_result_date=date.fromisoformat(window["end"]),
        top_k=len(panel.symbols) if candidate is None else protocol["top_k"],
        fee_bps=protocol["fee_bps"], slippage_bps=protocol["slippage_bps"],
    ).to_dict()


def _exclusive_json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_study(dataset_manifest: Path, protocol_path: Path, output_dir: Path) -> Path:
    """Write study.json in a new directory; invalid or reused inputs raise.

    Each window starts from cash. Excess return means the difference in total
    net return against the daily equal-weight baseline in that same window.
    A failure is recorded and re-raised; failed holdout attempts cannot resume.
    The source CSV stays in place and is never copied into the output folder.
    """
    dataset_manifest = Path(dataset_manifest).expanduser().resolve(strict=True)
    protocol_path = Path(protocol_path).expanduser().resolve(strict=True)
    manifest_raw, manifest = _read_document(dataset_manifest, "quant_dataset/1")
    protocol_raw, protocol = _read_document(protocol_path, "etf_study_protocol/1")
    panel = load_dataset(dataset_manifest)
    windows, candidates = _validate_protocol(panel, protocol)
    csv_path = (dataset_manifest.parent / manifest["csv"]).resolve(strict=True)
    module_root = Path(__file__).resolve().parent
    source_files = [(module_root / name, f"quant_research/{name}") for name in
                    ("etf_study.py", "backtest.py", "data.py", "signals.py")]
    source_files.append((Path(agent_io.__file__).resolve(strict=True), "research_agent/io.py"))
    # Real filesystem locations are used only for verification. Public artifacts
    # retain stable labels, so their original bytes can be shared unchanged.
    actual_paths = [dataset_manifest, csv_path, protocol_path] + [path for path, _ in source_files]
    manifest_limitations = manifest.get("limitations", [])
    if not isinstance(manifest_limitations, list) or any(not isinstance(value, str) for value in manifest_limitations):
        raise ValueError("Optional dataset limitations must be a list of strings")
    provenance = {
        "path_semantics": "logical labels; actual locations verified in memory",
        "dataset_manifest": _descriptor(dataset_manifest, "dataset/manifest.json", manifest_raw),
        "dataset_csv": {"path": f"dataset/{csv_path.name}", "sha256": panel.csv_sha256},
        "protocol": _descriptor(protocol_path, f"config/{protocol_path.name}", protocol_raw),
        "source_files": [_descriptor(path, logical_path) for path, logical_path in source_files],
        **{name: manifest[name] for name in ("source", "license", "price_basis", "universe_policy")},
        **{name: manifest[name] for name in (
            "data_kind", "source_snapshot", "missing_date_policy", "requested_universe",
            "requested_start_date", "requested_end_date", "actual_start_date", "actual_end_date", "date_count", "row_count",
        ) if name in manifest},
    }
    _verify_inputs(provenance, actual_paths)
    output_dir = Path(output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    study_path = output_dir / "study.json"
    fixed = {"id": "fixed_momentum", "kind": "momentum", "lookback": protocol["fixed_momentum_lookback"]}
    study = {
        "schema": "etf_case_study/1", "status": "running", "protocol": protocol,
        "windows": windows, "provenance": provenance,
        "development": {"baselines": [], "candidates": [], "selection_rule": _SELECTION_RULE},
        "selected_candidate": None, "holdout": {"status": "not_run", "series": []},
        "model_usage": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "qwen": {"status": "not_run", "reason": "No model calls were made in this deterministic study."},
        "assumptions": {
            "warmup_not_model_fit": True,
            "warmup_usage": "Historical context for fixed trailing features only; no fitted parameters or labels.",
            "signal_history": "Immutable observed price prefix through result session t-2 only.",
            "selection_target": "Difference in total net return against daily equal weight within each development window.",
            "window_initialization": "Each evaluation window begins in cash and pays initial entry costs.",
            "label_horizon_gaps": "At least two excluded sessions between development windows and before holdout; warmup is feature context.",
        },
        "limitations": list(dict.fromkeys(_LIMITATIONS + manifest_limitations)),
        "timing": {"started_at": utc_now()},
    }
    marker_path = output_dir / "holdout_attempt.json"
    attempt = None
    try:
        reference_returns = {}
        for baseline_id, candidate in (("equal_weight", None), ("fixed_momentum", fixed)):
            evidence = []
            for window in windows["development"]:
                result = _run_window(panel, candidate, window, protocol)
                total = result["metrics"]["total_return"]
                if candidate is None:
                    reference_returns[window["id"]] = total
                evidence.append({"window_id": window["id"], "result": result,
                                 "excess_return": total - reference_returns[window["id"]]})
            study["development"]["baselines"].append({"id": baseline_id, "candidate": candidate, "windows": evidence})
        for candidate in candidates:
            evidence = []
            for window in windows["development"]:
                result = _run_window(panel, candidate, window, protocol)
                evidence.append({"window_id": window["id"], "result": result,
                                 "excess_return": result["metrics"]["total_return"] - reference_returns[window["id"]]})
            excesses = [value["excess_return"] for value in evidence]
            study["development"]["candidates"].append({
                **candidate, "windows": evidence, "worst_excess_return": min(excesses),
                "mean_excess_return": fmean(excesses),
            })
        ranked = sorted(study["development"]["candidates"], key=lambda candidate: (
            -candidate["worst_excess_return"], -candidate["mean_excess_return"], candidate["id"]))
        selected = {name: ranked[0][name] for name in ("id", "kind", "lookback")}
        study["selected_candidate"] = selected
        _verify_inputs(provenance, actual_paths)
        selection = {
            "schema": "etf_dev_selection/1", "frozen_at": utc_now(),
            "protocol": protocol, "provenance": provenance, "windows": windows,
            "candidates": candidates, "development": study["development"],
            "selected_candidate": selected, "selection_rule": _SELECTION_RULE,
        }
        selection_path = output_dir / "dev_selection.json"
        _exclusive_json(selection_path, selection)
        selection_hash = _sha(selection_path.read_bytes())
        study["dev_selection"] = {"path": "dev_selection.json", "sha256": selection_hash}
        study["timing"]["selection_frozen_at"] = selection["frozen_at"]
        attempt = {
            "schema": "etf_holdout_attempt/1", "status": "started", "started_at": utc_now(),
            "selection_sha256": selection_hash, "selected_candidate": selected,
            "input_hashes": {name: provenance[name]["sha256"] for name in
                             ("dataset_manifest", "dataset_csv", "protocol")},
            "series_ids": ["equal_weight", "fixed_momentum", "selected_grid"],
        }
        _exclusive_json(marker_path, attempt)
        study["holdout"] = {"status": "running", "series": []}
        atomic_json(study_path, study)
        _verify_inputs(provenance, actual_paths)
        for series_id, candidate in (("equal_weight", None), ("fixed_momentum", fixed), ("selected_grid", selected)):
            _verify_inputs(provenance, actual_paths)
            if _sha(selection_path.read_bytes()) != selection_hash:
                raise ValueError("Frozen development selection SHA-256 changed")
            result = _run_window(panel, candidate, windows["holdout"], protocol)
            _verify_inputs(provenance, actual_paths)
            if candidate is None:
                reference_holdout = result["metrics"]["total_return"]
            study["holdout"]["series"].append({
                "id": series_id, "candidate": candidate, "result": result,
                "excess_return": result["metrics"]["total_return"] - reference_holdout,
            })
        _verify_inputs(provenance, actual_paths)
        if _sha(selection_path.read_bytes()) != selection_hash:
            raise ValueError("Frozen development selection SHA-256 changed")
        study["status"] = study["holdout"]["status"] = "completed"
        study["timing"]["completed_at"] = utc_now()
        attempt.update(status="completed", completed_at=study["timing"]["completed_at"])
        atomic_json(study_path, study)
        atomic_json(marker_path, attempt)
        return study_path
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        study["status"] = "failed"
        study["error"] = error
        study["timing"]["failed_at"] = utc_now()
        if attempt is not None:
            study["holdout"]["status"] = "failed"
            attempt.update(status="failed", error=error, failed_at=study["timing"]["failed_at"])
            atomic_json(marker_path, attempt)
        atomic_json(study_path, study)
        raise
