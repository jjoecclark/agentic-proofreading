"""Signed tokens for the two services (PLAN.md Section 6).

A token is ``<base64url(JSON payload)>.<base64url(HMAC-SHA256)>``. The outer harness signs; the
`world` server and the `llm` proxy verify with the same secret, which never enters a container.
Payloads are readable by whoever holds the token, so they carry nothing secret: a world token holds
the task's time T, a run token holds its owner, dollar cap, and model allowlist.

    kind "world": {task_id, t_utc, exp}
    kind "llm":   {owner, cap_usd, models, group, group_cap_usd, log_dir, exp}
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

TOKEN_VERSION = "1"


class TokenError(Exception):
    pass


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign(payload: dict[str, Any], secret: bytes, ttl_s: float | None = 7 * 24 * 3600) -> str:
    body = dict(payload, v=TOKEN_VERSION)
    if ttl_s is not None:
        body["exp"] = time.time() + ttl_s
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    mac = hmac.new(secret, raw, hashlib.sha256).digest()
    return f"{_b64e(raw)}.{_b64e(mac)}"


def verify(token: str, secret: bytes, kind: str) -> dict[str, Any]:
    try:
        raw_b64, mac_b64 = token.strip().split(".")
        raw = _b64d(raw_b64)
        mac = _b64d(mac_b64)
    except Exception as e:  # noqa: BLE001
        raise TokenError("malformed token") from e
    if not hmac.compare_digest(mac, hmac.new(secret, raw, hashlib.sha256).digest()):
        raise TokenError("bad token signature")
    payload = json.loads(raw)
    if payload.get("v") != TOKEN_VERSION:
        raise TokenError("unsupported token version")
    if payload.get("kind") != kind:
        raise TokenError(f"token is not a {kind} token")
    if payload.get("exp") is not None and payload["exp"] < time.time():
        raise TokenError("token expired")
    return payload


def world_token(secret: bytes, task_id: str, t_utc: str, ttl_s: float | None = 7 * 24 * 3600) -> str:
    return sign({"kind": "world", "task_id": task_id, "t_utc": t_utc}, secret, ttl_s)


def run_token(secret: bytes, owner: str, cap_usd: float, models: list[str], log_dir: str,
              group: str | None = None, group_cap_usd: float | None = None,
              ttl_s: float | None = 7 * 24 * 3600) -> str:
    return sign({"kind": "llm", "owner": owner, "cap_usd": float(cap_usd), "models": list(models),
                 "log_dir": log_dir, "group": group,
                 "group_cap_usd": None if group_cap_usd is None else float(group_cap_usd)},
                secret, ttl_s)
