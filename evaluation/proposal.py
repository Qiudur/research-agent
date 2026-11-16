"""Append-only, separately priced ledger for self-research proposals."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re

from ledger import TOKEN_BUCKETS, Usage, price_usage


_HASH = re.compile(r"^[0-9a-f]{64}$")


class ProposalLedger:
    """ModelClient-compatible sink; no artificial B task or score is created."""

    def __init__(self, path: Path, *, prices: dict, price_sha256: str,
                 source_prompt_sha256: str, model: str):
        if not _HASH.fullmatch(price_sha256) or not _HASH.fullmatch(source_prompt_sha256):
            raise ValueError("Proposal source hashes must be SHA256")
        self.path = Path(path).resolve()
        self.prices = prices
        self.price_sha256 = price_sha256
        self.source_prompt_sha256 = source_prompt_sha256
        self.model = model
        self.round = 0

    def llm_call(
        self, *, node_id: str | None, role: str, purpose: str, effort: str,
        model: str, usage: Usage, seconds: float, prompt_hash: str,
        ok: bool = True, error: str | None = None,
        usage_status: str = "reported",
    ) -> float:
        if node_id is not None or role != "planner" or purpose != "plan" or effort != "high":
            raise ValueError("Proposal calls must be planner/plan/high without a node")
        if model != self.model or self.round < 1:
            raise ValueError("Proposal model or round differs from the sealed study")
        if not isinstance(prompt_hash, str) or not _HASH.fullmatch(prompt_hash.lower()):
            raise ValueError("Proposal prompt hash must be SHA256")
        if usage_status not in {"reported", "unknown"}:
            raise ValueError("Proposal usage_status must be reported or unknown")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Proposal duration must be nonnegative and finite")
        tokens = {key: getattr(usage, key) for key in TOKEN_BUCKETS}
        cost = price_usage(self.prices, model, usage)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                prior = [json.loads(line) for line in handle if line.strip()]
                if any(row.get("schema") != "proposal_ledger/1" or row.get("seq") != i
                       for i, row in enumerate(prior, 1)):
                    raise ValueError("Corrupt proposal ledger sequence")
                event = {
                    "schema": "proposal_ledger/1", "seq": len(prior) + 1,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                    "round": self.round, "model": model, "role": role,
                    "purpose": purpose, "effort": effort, "tokens": tokens,
                    "cost": cost, "seconds": float(seconds),
                    "price_sha256": self.price_sha256,
                    "source_prompt_sha256": self.source_prompt_sha256,
                    "prompt_hash": prompt_hash.lower(),
                    "response_status": "completed" if ok else "failed",
                    "ok": bool(ok), "error": error,
                    "usage_status": usage_status,
                }
                handle.seek(0, 2)
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + chr(10))
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return cost

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for seq, row in enumerate(rows, 1):
            if (row.get("schema") != "proposal_ledger/1" or row.get("seq") != seq
                    or row.get("price_sha256") != self.price_sha256
                    or row.get("source_prompt_sha256") != self.source_prompt_sha256
                    or row.get("model") != self.model
                    or row.get("response_status") not in {"completed", "failed"}):
                raise ValueError("Corrupt proposal ledger provenance")
            if row.get("usage_status", "reported") not in {"reported", "unknown"}:
                raise ValueError("Corrupt proposal usage status")
            tokens = row.get("tokens")
            if not isinstance(tokens, dict) or set(tokens) != set(TOKEN_BUCKETS):
                raise ValueError("Corrupt proposal token buckets")
            expected = price_usage(self.prices, self.model, Usage(**tokens))
            if not math.isclose(row.get("cost"), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Proposal cost does not reconcile with prices")
        return rows

    def spent(self) -> float:
        return sum(row["cost"] for row in self.events())


__all__ = ["ProposalLedger"]
