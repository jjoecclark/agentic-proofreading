"""The spend ledger (PLAN.md Section 6.2): one JSON line per Claude API request.

The proxy is the only writer. ``outer/loop.py`` and ``outer/evaluate.py`` read it; it is the single
source of truth for spending. Each line:

    {"ledger_version": "1", "ts": <unix s>, "owner": ..., "group": ..., "model": ..., "request_id": ...,
     "status": <HTTP status or "refused_budget" / "refused_model">, "stream": bool,
     "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_creation_5m",
     "cache_creation_1h", "cache_read_input_tokens", "usd"}
"""

from __future__ import annotations

import fcntl
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

LEDGER_VERSION = "1"

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_creation_5m",
                "cache_creation_1h", "cache_read_input_tokens")


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Anthropic ``usage`` → flat token counts. Cache writes are split by TTL when the API reports
    ``cache_creation``; otherwise all writes are counted as 5-minute writes."""
    u = usage or {}
    out = {k: int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                             "cache_read_input_tokens")}
    cc = u.get("cache_creation") or {}
    w5 = int(cc.get("ephemeral_5m_input_tokens") or 0)
    w1 = int(cc.get("ephemeral_1h_input_tokens") or 0)
    if w5 + w1 == 0:
        w5 = out["cache_creation_input_tokens"]
    out["cache_creation_5m"], out["cache_creation_1h"] = w5, w1
    return out


def cost_usd(model: str, tokens: dict[str, int], prices: dict[str, dict[str, float]]) -> float:
    """Dollars for one request from the price table in config.yaml (USD per million tokens)."""
    p = prices.get(model)
    if p is None:
        raise KeyError(f"no price for model {model!r}")
    inp = p["input"]
    return (tokens.get("input_tokens", 0) * inp
            + tokens.get("output_tokens", 0) * p["output"]
            + tokens.get("cache_creation_5m", 0) * p.get("cache_write_5m", 1.25 * inp)
            + tokens.get("cache_creation_1h", 0) * p.get("cache_write_1h", 2.0 * inp)
            + tokens.get("cache_read_input_tokens", 0) * p.get("cache_read", 0.1 * inp)) / 1e6


class Ledger:
    """Append-only ledger with running totals per owner and per group (thread-safe)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.by_owner: dict[str, float] = defaultdict(float)
        self.by_group: dict[str, float] = defaultdict(float)
        self.total = 0.0
        for e in read_entries(self.path):
            self._add(e)

    def _add(self, e: dict[str, Any]) -> None:
        usd = float(e.get("usd") or 0.0)
        self.by_owner[e.get("owner", "")] += usd
        if e.get("group"):
            self.by_group[e["group"]] += usd
        self.total += usd

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        e = {"ledger_version": LEDGER_VERSION, "ts": time.time(), **entry}
        line = json.dumps(e, sort_keys=True) + "\n"
        with self._lock:
            with open(self.path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(line)
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
            self._add(e)
        return e

    def spent(self, owner: str) -> float:
        with self._lock:
            return self.by_owner.get(owner, 0.0)

    def spent_group(self, group: str | None) -> float:
        if not group:
            return 0.0
        with self._lock:
            return self.by_group.get(group, 0.0)


def read_entries(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line from a crash; ignore
    return out


def summarize(path: Path, owner_prefix: str | None = None) -> dict[str, Any]:
    """Totals over ledger lines whose owner starts with ``owner_prefix`` (all lines if None)."""
    tot = {k: 0 for k in USAGE_FIELDS}
    usd, n, refused = 0.0, 0, 0
    for e in read_entries(path):
        if owner_prefix is not None and not str(e.get("owner", "")).startswith(owner_prefix):
            continue
        n += 1
        usd += float(e.get("usd") or 0.0)
        if str(e.get("status", "")).startswith("refused"):
            refused += 1
        for k in USAGE_FIELDS:
            tot[k] += int(e.get(k) or 0)
    return {"requests": n, "refused": refused, "usd": usd, **tot}
