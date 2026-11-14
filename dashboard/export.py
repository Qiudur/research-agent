"""Export a self-contained, read-only dashboard from v1 run artifacts.

The input run is never modified. Missing or unverified prices produce unknown
costs, not a misleading zero. No task hidden data or solution code is loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_accounting import is_exact_token_accounting, require_exact_token_accounting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("index.html")


def _read_json(path: Path, schema: str) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"Unexpected schema in {path}: expected {schema}")
    return value


def _read_jsonl(path: Path, schema: str) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{index}") from exc
        if not isinstance(value, dict) or value.get("schema") != schema:
            raise ValueError(f"Unexpected schema at {path}:{index}: expected {schema}")
        records.append(value)
    return records


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _safe_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    candidate = (path if path.is_absolute() else project_root / path).resolve()
    if not candidate.is_relative_to(project_root.resolve()):
        raise ValueError("Manifest path escapes project root")
    return candidate


def _price_info(manifest: dict | None, project_root: Path) -> dict:
    unknown = {"verified": False, "currency": None, "test_only": False,
               "internal_token_study": False, "exact_token_accounting": False,
               "accounting_note": None,
               "message": "价格或 manifest 缺失，真实成本未知"}
    if manifest is None:
        return unknown
    descriptor = manifest.get("prices")
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        return unknown
    path = _safe_project_path(project_root, descriptor["path"])
    expected = descriptor.get("sha256")
    if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
        return unknown
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected.lower():
        return unknown
    prices = _read_json(path, "prices/1")
    model = manifest.get("model")
    require_exact_token_accounting(prices, model)
    models = (prices or {}).get("models")
    rates = models.get(model) if isinstance(models, dict) else None
    if not isinstance(rates, dict) or any(_number(rates.get(key)) is None for key in
                                          ("input", "cached_input", "output", "reasoning")):
        return unknown
    test_only = bool(prices.get("test_only")) or "fixture" in path.parts
    is_tokens = prices.get("currency") == "TOKENS"
    accounting_note = ("自选 token 预算的内部研究；非现金节省、非官方比赛成绩。"
                       if is_tokens else None)
    message = "测试价目，仅供联调" if test_only else "价目文件哈希已核验"
    if is_tokens:
        message = ("测试 token 计量，仅供联调；" if test_only else "token 计量表哈希已核验；") + accounting_note
    return {"verified": True, "currency": prices.get("currency"),
            "test_only": test_only, "internal_token_study": is_tokens and not test_only,
            "exact_token_accounting": is_exact_token_accounting(prices, model),
            "accounting_note": accounting_note, "message": message}


def _compact_node(node: dict) -> dict:
    score = node.get("score") or {}
    return {
        "id": node.get("node_id"), "parent": node.get("parent"),
        "hypothesis": node.get("hypothesis"), "origin": node.get("origin"),
        "status": node.get("status"), "score": score.get("dev"),
        "raw": score.get("raw"), "spread": score.get("spread"),
        "calls": node.get("calls") or [], "cites": node.get("cites") or [],
        "cost": node.get("cost"), "attempts": node.get("attempts"),
        "failure": node.get("failure"), "library_version": node.get("library_version"),
    }


def _task_summary(task_id: str, task_dir: Path, price_info: dict) -> dict:
    events = _read_jsonl(task_dir / "ledger.jsonl", "ledger/1")
    snapshots = _read_jsonl(task_dir / "tree.jsonl", "node/1")
    latest: dict[str, dict] = {}
    for snapshot in snapshots:
        latest[str(snapshot["node_id"])] = snapshot
    nodes = [_compact_node(latest[key]) for key in sorted(latest)]
    starts = [e for e in events if e.get("kind") == "task_start"]
    ends = [e for e in events if e.get("kind") == "task_end"]
    quotes = [e for e in events if e.get("kind") == "quote"]
    llm_calls = [e for e in events if e.get("kind") == "llm_call"]
    budget = _number(starts[-1].get("budget")) if starts else None
    prices_ok = bool(price_info["verified"])
    costs = [_number(e.get("cost")) for e in llm_calls]
    end_total = _number(ends[-1].get("total_cost")) if ends else None
    cost_known = prices_ok and ((bool(llm_calls) and all(value is not None for value in costs)) or end_total is not None)
    total_cost = None
    if cost_known:
        total_cost = end_total if end_total is not None else sum(costs)
    tokens = {name: sum(int((e.get("tokens") or {}).get(name, 0) or 0) for e in llm_calls)
              for name in ("input", "cached_input", "output", "reasoning")}
    cache_base = tokens["input"] + tokens["cached_input"]
    seconds = {
        "llm": sum(float(e.get("seconds", 0) or 0) for e in llm_calls),
        "run": sum(float(e.get("seconds_run", 0) or 0) for e in events if e.get("kind") == "experiment"),
        "queue": sum(float(e.get("seconds_queue", 0) or 0) for e in events if e.get("kind") == "score"),
        "wall": None,
    }
    if ends and isinstance(ends[-1].get("seconds"), dict):
        seconds["wall"] = _number(ends[-1]["seconds"].get("wall"))
    accepted = [node for node in nodes if node["status"] == "accepted" and _number(node["score"]) is not None]
    best = max(accepted, key=lambda node: node["score"]) if accepted else None
    series = []
    accumulated = 0.0
    root = next((n for n in nodes if n["id"] == "n0000" and n["score"] is not None), None)
    if root is not None:
        series.append({"node_id": root["id"], "score": root["score"], "spent": 0.0 if cost_known else None})
    for event in events:
        if event.get("kind") == "llm_call" and cost_known:
            accumulated += float(event["cost"])
        elif event.get("kind") == "score" and event.get("split") == "dev" and _number(event.get("score")) is not None:
            series.append({"node_id": event.get("node_id"), "score": event["score"],
                           "spent": accumulated if cost_known else None})
    timeline = [{"seq": e.get("seq"), "ts": e.get("ts"), "kind": e.get("kind"),
                 "node_id": e.get("node_id"), "op": e.get("op"),
                 "score": e.get("score"), "cost": e.get("cost"),
                 "reason": e.get("reason")} for e in events[-80:]]
    return {
        "id": task_id, "nodes": nodes, "series": series, "timeline": timeline,
        "budget": budget if prices_ok else None, "cost": total_cost, "cost_known": cost_known,
        "best": {"node_id": best["id"], "score": best["score"], "raw": best["raw"]} if best else None,
        "holdout": ends[-1].get("holdout_score") if ends else None,
        "quote": quotes[-1].get("predicted") if quotes else None,
        "quote_basis": quotes[-1].get("basis") if quotes else None,
        "quote_error": ends[-1].get("quote_error") if ends else None,
        "seconds": seconds, "llm_calls": len(llm_calls), "tokens": tokens,
        "cache_rate": tokens["cached_input"] / cache_base if cache_base else None,
        "event_count": len(events),
    }


def load_run(run_dir: Path, *, project_root: Path = PROJECT_ROOT, library_root: Path | None = None) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest = _read_json(run_dir / "manifest.json", "run/1")
    if manifest is not None and manifest.get("run_id") != run_dir.name:
        raise ValueError("Manifest run_id does not match directory")
    price_info = _price_info(manifest, project_root)
    tasks_dir = run_dir / "tasks"
    task_ids = list(manifest.get("tasks") or []) if manifest else []
    if not task_ids and tasks_dir.is_dir():
        task_ids = sorted(path.name for path in tasks_dir.iterdir() if path.is_dir())
    tasks = [_task_summary(task_id, tasks_dir / task_id, price_info) for task_id in task_ids]
    if library_root is None:
        library_root = run_dir / "library"
        if manifest and isinstance(manifest.get("library"), dict) and isinstance(manifest["library"].get("root"), str):
            library_root = _safe_project_path(project_root, manifest["library"]["root"])
    library_root = Path(library_root).expanduser().resolve()
    state = _read_json(library_root / "state.json", "library/1")
    changes = _read_jsonl(library_root / "changelog.jsonl", "libchange/1")
    entries = state.get("entries", []) if state else []
    run_events = _read_jsonl(run_dir / "ledger.jsonl", "ledger/1")
    return {
        "run": {"id": run_dir.name, "phase": manifest.get("phase") if manifest else None,
                "config": manifest.get("config") if manifest else None,
                "rep": manifest.get("rep") if manifest else None,
                "model": manifest.get("model") if manifest else None,
                "started_ts": manifest.get("started_ts") if manifest else None,
                "ended_ts": manifest.get("ended_ts") if manifest else None,
                "manifest_present": manifest is not None, "price": price_info,
                "event_count": len(run_events)},
        "tasks": tasks,
        "library": {"version": state.get("version") if state else None,
                    "entries": [{"id": e.get("id"), "kind": e.get("kind"),
                                 "status": e.get("status"), "credibility": e.get("credibility"),
                                 "action": e.get("action"), "effect": e.get("effect")} for e in entries],
                    "changes": [{"ts": c.get("ts"), "version": c.get("version"),
                                 "op": c.get("op"), "entry_id": c.get("entry_id"),
                                 "reason": c.get("reason")} for c in changes[-80:]]},
    }


def render_html(data: dict) -> str:
    price = data["run"].get("price") or {}
    if price.get("currency") == "TOKENS" and price.get("exact_token_accounting") is not True:
        raise ValueError("TOKENS dashboard needs audited exact token accounting")
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return TEMPLATE.read_text(encoding="utf-8").replace("__DASHBOARD_DATA__", serialized).replace("__GENERATED_AT__", generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a read-only run dashboard")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--library-root", type=Path)
    args = parser.parse_args()
    data = load_run(args.run_dir, project_root=args.project_root, library_root=args.library_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(data), encoding="utf-8")
    print(f"Exported {args.output} ({len(data['tasks'])} tasks)")


if __name__ == "__main__":
    main()
