"""Replay stop-loss rules offline on finished runs and report what they would save.

Input is one or more directories that contain run directories (a run is a
directory with a run/1 manifest.json); nested campaign layouts are searched.
Only runs whose manifest phase is practice or self_research are used by
default, so the rules are never tuned on validation or exam evidence.

For every finished task the main loop (all model calls except the summarizer)
is replayed in ledger order. A node completes at its last experiment/score
event and its final status is the last tree.jsonl snapshot (see
evaluation.metrics.node_completions). Two rule families are evaluated for each
scenario over a parameter grid:

* first_n_no_gain=N: stop right after the N-th completed node when none of
  the first N nodes was accepted;
* consecutive_no_gain=K: stop right after K completed nodes in a row without
  an accepted one.

A task counts as stopped early when the rule would have fired with main-loop
spend still to come. The dev score is the proxy for quality: a stopped task
"would have improved" when a node accepted after the stopping point raised the
best dev score. Nothing here calls a model or a scorer.

    python -m scripts.replay_stoploss runs/token-practice10-default-... --output /tmp/stoploss_replay.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

from evaluation.metrics import (COMPLETION_MODES, REFERENCE_SCORE, event_tokens, find_run_dirs,
                                load_events, load_nodes, node_completions, scenario_of)

DEFAULT_PHASES = ("practice", "self_research")
ALLOWED_PHASES = ("practice", "self_research", "dev", "demo")  # never validation or exam
FIRST_N_GRID = (1, 2, 3, 4)
CONSECUTIVE_GRID = (2, 3, 4, 5, 6, 7, 8)
_EPS = 1e-12


def first_n_no_gain_fire(accepted: Sequence[bool], n: int) -> int | None:
    """1-based completion index at which "first n nodes all rejected" fires."""
    if n < 1:
        raise ValueError("first_n_no_gain must be at least 1")
    if len(accepted) >= n and not any(accepted[:n]):
        return n
    return None


def consecutive_no_gain_fire(accepted: Sequence[bool], k: int) -> int | None:
    """1-based completion index at which k unaccepted nodes in a row first occur."""
    if k < 1:
        raise ValueError("consecutive_no_gain must be at least 1")
    streak = 0
    for index, flag in enumerate(accepted, 1):
        streak = 0 if flag else streak + 1
        if streak >= k:
            return index
    return None


RULES: dict[str, Callable[[Sequence[bool], int], int | None]] = {
    "first_n_no_gain": first_n_no_gain_fire,
    "consecutive_no_gain": consecutive_no_gain_fire,
}


def replay_task(task_dir: Path, *, completion: str = "experiment_score") -> dict | None:
    """Replay inputs of one finished task, or None when it cannot be replayed."""
    task_dir = Path(task_dir)
    if not (task_dir / "ledger.jsonl").is_file() or not (task_dir / "tree.jsonl").is_file():
        return None
    events, nodes = load_events(task_dir), load_nodes(task_dir)
    if not any(event.get("kind") == "stop" for event in events) or "n0000" not in nodes:
        return None
    task_id = next((e["task_id"] for e in events if e.get("task_id")), task_dir.name)
    completions = node_completions(events, nodes, completion=completion)
    root = nodes["n0000"].get("score") or {}
    baseline = root.get("dev")
    if baseline is None:
        return None
    return {
        "task_id": task_id, "scenario": scenario_of(task_id),
        "main_loop_tokens": sum(event_tokens(e) for e in events
                                if e.get("kind") == "llm_call" and e.get("role") != "summarizer"),
        "baseline_dev": float(baseline),
        "final_best_dev": completions[-1]["best_dev_so_far"] if completions else float(baseline),
        "completions": completions,
    }


def apply_rule(task: dict, rule: str, parameter: int) -> dict:
    """What one rule would have done to one replayed task."""
    flags = [row["accepted"] for row in task["completions"]]
    fired_at = RULES[rule](flags, parameter)
    result = {"fired_at_node": fired_at, "stopped_early": False, "tokens_saved": 0,
              "later_improved": False, "later_reached_reference": False, "forgone_dev_gain": 0.0}
    if fired_at is None:
        return result
    point = task["completions"][fired_at - 1]
    saved = max(0, task["main_loop_tokens"] - point["main_tokens"])
    if saved == 0:
        return result
    best_then, best_final = point["best_dev_so_far"], task["final_best_dev"]
    result.update(
        stopped_early=True, tokens_saved=saved,
        later_improved=best_final > best_then + _EPS,
        later_reached_reference=best_final >= REFERENCE_SCORE - _EPS and best_then < REFERENCE_SCORE - _EPS,
        forgone_dev_gain=max(0.0, best_final - best_then))
    return result


def _cell(tasks: list[dict], rule: str, parameter: int, details: list | None) -> dict:
    outcomes = [apply_rule(task, rule, parameter) for task in tasks]
    total = sum(task["main_loop_tokens"] for task in tasks)
    saved = sum(o["tokens_saved"] for o in outcomes)
    stopped = [o for o in outcomes if o["stopped_early"]]
    improved = sum(1 for o in stopped if o["later_improved"])
    if details is not None:
        details.extend({"task_id": t["task_id"], "scenario": t["scenario"], "run_id": t["run_id"],
                        "rule": rule, "parameter": parameter, **o}
                       for t, o in zip(tasks, outcomes))
    return {
        "tasks": len(tasks), "main_loop_tokens": total, "tokens_saved": saved,
        "saved_fraction": saved / total if total else None,
        "stopped_early": len(stopped), "later_improved": improved,
        "later_reached_reference": sum(1 for o in stopped if o["later_reached_reference"]),
        "forgone_dev_gain": sum(o["forgone_dev_gain"] for o in stopped),
        "improved_share_of_stopped": improved / len(stopped) if stopped else None,
    }


def replay(roots: Sequence[Path], *, phases: Sequence[str] = DEFAULT_PHASES,
           first_n: Sequence[int] = FIRST_N_GRID, consecutive: Sequence[int] = CONSECUTIVE_GRID,
           completion: str = "experiment_score", details: bool = False) -> dict:
    """Replay both rule families over the grid for every usable task."""
    if completion not in COMPLETION_MODES:
        raise ValueError(f"completion must be one of {COMPLETION_MODES}")
    unknown = set(phases) - set(ALLOWED_PHASES)
    if unknown:
        raise ValueError(f"Phases {sorted(unknown)} may not be used for rule replay "
                         f"(allowed: {', '.join(ALLOWED_PHASES)})")
    used_runs, skipped, tasks = [], [], []
    for run_dir in find_run_dirs(roots):
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("phase") not in phases:
            skipped.append({"run_id": run_dir.name, "reason": f"phase={manifest.get('phase')}"})
            continue
        replayed = 0
        for task_id in manifest.get("tasks", []):
            task = replay_task(run_dir / "tasks" / task_id, completion=completion)
            if task is None:
                skipped.append({"run_id": run_dir.name, "task_id": task_id,
                                "reason": "no finished ledger and tree"})
                continue
            task["run_id"] = run_dir.name
            tasks.append(task)
            replayed += 1
        if replayed:
            used_runs.append({"run_id": run_dir.name, "phase": manifest.get("phase"),
                              "tasks": replayed})
    detail_rows: list | None = [] if details else None
    scenarios = {}
    for scenario in ("tab", "spd"):
        group = [task for task in tasks if task["scenario"] == scenario]
        scenarios[scenario] = {
            "tasks": len(group),
            "main_loop_tokens": sum(t["main_loop_tokens"] for t in group),
            "tasks_with_an_accepted_node": sum(1 for t in group if any(
                row["accepted"] for row in t["completions"])),
            "rules": {
                "first_n_no_gain": {str(n): _cell(group, "first_n_no_gain", n, detail_rows)
                                    for n in first_n},
                "consecutive_no_gain": {str(k): _cell(group, "consecutive_no_gain", k, detail_rows)
                                        for k in consecutive},
            },
        }
    report = {
        "schema": "stoploss_replay/1",
        "inputs": [str(Path(root)) for root in roots], "phases": list(phases),
        "completion": completion,
        "grid": {"first_n_no_gain": list(first_n), "consecutive_no_gain": list(consecutive)},
        "runs": {"used": used_runs, "skipped": skipped},
        "scenarios": scenarios,
        "notes": [
            "tab maps to stop_loss.tab.first_n_no_gain, spd to stop_loss.spd.consecutive_no_gain; "
            "both families are evaluated for both scenarios.",
            "tokens_saved counts main-loop tokens (everything but the summarizer) after the "
            "stopping point; the dev score is only a proxy for later improvement.",
        ],
    }
    if detail_rows is not None:
        report["details"] = detail_rows
    return report


def _print_summary(report: dict) -> None:
    for scenario, block in report["scenarios"].items():
        print(f"[{scenario}] tasks={block['tasks']} main-loop tokens={block['main_loop_tokens']}")
        for family, cells in block["rules"].items():
            for parameter, cell in cells.items():
                fraction = cell["saved_fraction"]
                share = "n/a" if fraction is None else f"{fraction:.1%}"
                print(f"  {family}={parameter}: saved {cell['tokens_saved']} ({share}), "
                      f"stopped early {cell['stopped_early']}/{cell['tasks']}, "
                      f"would have improved {cell['later_improved']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dirs", nargs="+", type=Path,
                        help="directories containing run directories (searched recursively)")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--phases", nargs="+", choices=ALLOWED_PHASES, default=list(DEFAULT_PHASES),
                        help="manifest phases to use (validation and exam are never allowed)")
    parser.add_argument("--first-n", nargs="+", type=int, default=list(FIRST_N_GRID))
    parser.add_argument("--consecutive", nargs="+", type=int, default=list(CONSECUTIVE_GRID))
    parser.add_argument("--completion", choices=COMPLETION_MODES, default="experiment_score",
                        help="node completion point: last experiment/score event (default) or "
                             "last event of any kind (also counts trailing diagnoser calls)")
    parser.add_argument("--details", action="store_true", help="include one row per task and rule")
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.first_n + args.consecutive):
        parser.error("grid values must be at least 1")
    missing = [path for path in args.runs_dirs if not path.exists()]
    if missing:
        parser.error(f"no such directory: {missing[0]}")
    report = replay(args.runs_dirs, phases=args.phases, first_n=args.first_n,
                    consecutive=args.consecutive, completion=args.completion,
                    details=args.details)
    if not report["runs"]["used"]:
        reasons = Counter(item["reason"] for item in report["runs"]["skipped"])
        detail = ", ".join(f"{reason} x{count}" for reason, count in sorted(reasons.items()))
        print("No usable runs found (check --phases and that the runs finished)"
              + (f"; skipped: {detail}" if detail else "") + ".", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
