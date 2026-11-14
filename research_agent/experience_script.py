"""Task-end experience drafts built straight from the experiment tree.

``experience_writer: "script"`` replaces the summarizer model call. Nothing here
spends tokens, so the reserve gate never applies. The runner feeds these drafts
through the same checks as the summarizer's: every draft must cite real nodes of
this task, triggers must be features of this task's profile, and a frozen library
only receives the drafts file.

What counts as evidence. Every comparison is with the node's parent, and the
noise line is the larger of the two dev-score spreads, the same line the tree
uses to accept a node:

* positive: accepted, and better than its parent by more than the noise line.
* negative: rejected and worse than its parent by more than the noise line;
  rejected with scorer ``cheat_flags``; or failed with a concrete cause
  (``cheat_flags``, timeout, memory, wrong output).
* nothing: a node rejected only because its gain did not clear the noise line.
  Measurement noise must not turn into "avoid this" advice.

A budget-blocked or otherwise unexplained failure says nothing about the idea
and is skipped, and so is a task whose profile has no features (an experience
tied to no trigger would be offered to every task of the scenario).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from .artifacts import node_directory

WRITER_MODES = ("model", "script")
DEFAULT_WRITER = "model"
# Per task, so one lucky or unlucky run cannot flood the library.
MAX_POSITIVE = 3
MAX_NEGATIVE = 3
_HYPOTHESIS_CHARS = 200
_MIN_DELTA = 1e-9

_CAUSE_LABELS = {
    "timeout": "运行或评分超时",
    "oom": "内存超限",
    "incorrect_output": "输出与参考实现不一致（正确性检查失败）",
}


def writer_mode(config: Mapping[str, Any]) -> str:
    """The configured experience writer; a missing key keeps the model summarizer."""
    mode = config.get("experience_writer", DEFAULT_WRITER)
    if not isinstance(mode, str) or mode not in WRITER_MODES:
        raise ValueError(f"experience_writer must be one of {list(WRITER_MODES)}")
    return mode


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _dev_and_spread(node: Mapping[str, Any]) -> tuple[float, float] | None:
    score = node.get("score")
    if not isinstance(score, Mapping):
        return None
    dev, spread = _number(score.get("dev")), _number(score.get("spread", 0.0))
    if dev is None or spread is None or spread < 0:
        return None
    return dev, spread


def _flags(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _short(text: Any, limit: int = _HYPOTHESIS_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def read_attempt_details(tree: Any, nodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """What the scorer said about the last attempt of each failed node.

    A failed node keeps no score in the tree, but the runner saves every scored
    attempt to ``nodes/<id>/score.json``. Missing or unreadable files are skipped.
    """
    details: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if node.get("status") != "failed":
            continue
        try:
            path = node_directory(tree.run_dir, tree.task_id, node["node_id"]) / "score.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            continue
        if not isinstance(saved, dict):
            continue
        error = saved.get("error")
        details[node["node_id"]] = {
            "cheat_flags": _flags(saved.get("cheat_flags")),
            "error_kind": error.get("kind") if isinstance(error, dict) else None,
            "ok": saved.get("ok"),
            "correct": saved.get("correct"),
        }
    return details


def _cause_from_text(text: str) -> str | None:
    low = text.lower()
    if "timed out" in low or "exceeded wall time" in low or "timeout" in low:
        return "timeout"
    if "exceeded memory" in low or "memory limit" in low or "memoryerror" in low or "out of memory" in low:
        return "oom"
    if "correctness check failed" in low:
        return "incorrect_output"
    return None


def _failure_cause(node: Mapping[str, Any], detail: Mapping[str, Any]) -> tuple[str, list[str]] | None:
    """A concrete cause for a failed node, or None when it says nothing about the idea."""
    failure = node.get("failure")
    if not isinstance(failure, Mapping) or failure.get("kind") == "budget":
        return None
    flags = _flags(detail.get("cheat_flags"))
    if flags:
        return "cheat_flags", flags
    if detail.get("error_kind") in {"timeout", "oom"}:
        return str(detail["error_kind"]), []
    # score/1 also says correct=false when scoring itself broke (ok=false); only a
    # scored-but-wrong result is a wrong-output failure.
    if detail.get("ok") is True and detail.get("correct") is False:
        return "incorrect_output", []
    cause = _cause_from_text(str(failure.get("summary") or ""))
    if cause is None and failure.get("kind") == "timeout":
        cause = "timeout"
    return (cause, []) if cause else None


def _draft(scenario: str, triggers: list[str], node_id: str, *, negative: bool, action: str) -> dict[str, Any]:
    return {
        "scope": "domain", "scenario": scenario, "triggers": list(triggers),
        "conditions": f"题目特征含 {'、'.join(triggers)}；目前只有本题一次实测",
        "action": action, "negative": negative, "node_ids": [node_id],
    }


def build_drafts(
    nodes: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    features: Sequence[str],
    attempt_details: Mapping[str, Mapping[str, Any]] | None = None,
    max_positive: int = MAX_POSITIVE,
    max_negative: int = MAX_NEGATIVE,
) -> list[dict[str, Any]]:
    """Experience drafts (the summarizer's draft shape) for one finished task.

    ``nodes`` are the tree's node/1 snapshots. Positive drafts come first, each
    group strongest evidence first, capped per task by max_positive/max_negative.
    """
    triggers = list(dict.fromkeys(features))
    if not triggers:
        return []
    details = attempt_details or {}
    by_id = {node["node_id"]: node for node in nodes}
    positives: list[tuple[float, str, dict[str, Any]]] = []
    negatives: list[tuple[int, float, str, dict[str, Any]]] = []
    for node in nodes:
        node_id = node["node_id"]
        parent = by_id.get(node.get("parent"))
        if node.get("origin") == "root" or parent is None:
            continue
        status = node.get("status")
        title = _short(node.get("hypothesis", ""))
        if not title:
            continue
        cost = _number(node.get("cost")) or 0.0
        own, before = _dev_and_spread(node), _dev_and_spread(parent)
        if status == "accepted" and own and before:
            gain, noise = own[0] - before[0], max(own[1], before[1])
            score = node["score"]
            if gain > noise and gain > _MIN_DELTA and score.get("correct") and not _flags(score.get("cheat_flags")):
                positives.append((gain, node_id, _draft(
                    scenario, triggers, node_id, negative=False,
                    action=(f"{title}（本题实测：dev {before[0]:.4f} → {own[0]:.4f}，提升 {gain:.4f}，"
                            f"高于波动 {noise:.4f}；花费 {cost:g}）"))))
        elif status == "rejected" and own and before:
            flags = _flags(node["score"].get("cheat_flags"))
            if flags:
                negatives.append((1, 0.0, node_id, _draft(
                    scenario, triggers, node_id, negative=True,
                    action=f"{title}（评分器标记 cheat_flags：{', '.join(flags)}，结果不可信；花费 {cost:g}）")))
                continue
            drop, noise = before[0] - own[0], max(own[1], before[1])
            if drop > noise and drop > _MIN_DELTA:  # a rejection inside the noise is not evidence
                negatives.append((0, drop, node_id, _draft(
                    scenario, triggers, node_id, negative=True,
                    action=(f"{title}（本题实测：dev {before[0]:.4f} → {own[0]:.4f}，下降 {drop:.4f}，"
                            f"超过波动 {noise:.4f}；花费 {cost:g}）"))))
        elif status == "failed":
            cause = _failure_cause(node, details.get(node_id, {}))
            if cause is None:
                continue
            kind, flags = cause
            reason = (f"评分器标记 cheat_flags：{', '.join(flags)}" if kind == "cheat_flags"
                      else _CAUSE_LABELS[kind])
            negatives.append((1, 0.0, node_id, _draft(
                scenario, triggers, node_id, negative=True,
                action=f"{title}（失败：{reason}；尝试 {node.get('attempts', 0)} 次仍未通过；花费 {cost:g}）")))
    positives.sort(key=lambda item: (-item[0], item[1]))
    negatives.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return ([item[2] for item in positives[:max(0, max_positive)]]
            + [item[3] for item in negatives[:max(0, max_negative)]])


def drafts_from_tree(tree: Any, *, scenario: str, features: Sequence[str]) -> list[dict[str, Any]]:
    """Convenience entry point for the runner: read the tree, build the drafts."""
    nodes = tree.all_nodes()
    return build_drafts(nodes, scenario=scenario, features=features,
                        attempt_details=read_attempt_details(tree, nodes))


__all__ = ["DEFAULT_WRITER", "MAX_NEGATIVE", "MAX_POSITIVE", "WRITER_MODES", "build_drafts",
           "drafts_from_tree", "read_attempt_details", "writer_mode"]
