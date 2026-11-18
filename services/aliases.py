"""Opaque per-task ids (world API version 2).

FlyWire numbers new roots and L2 nodes from counters, so raw ids carry their creation time: the two
segments a split just produced have nearly consecutive ids, for example. A harness could tell a
no-edit task from an edit task by that alone. So every task sees its own ids: root and L2 ids are
passed through a keyed 64-bit permutation (a 6-round Feistel network) whose key depends on the token
secret, the task, and the kind of id. The same id gets the same alias within a task and a different
one in every other task; 0 (background) stays 0. Supervoxel ids are shown as they are: they come
from the original segmentation and carry no edit time.

Used by the world service (translates ids in requests and replies) and by the outer harness (writes
the aliased segment ids into the task state a harness or the meta-agent sees).
"""

from __future__ import annotations

import copy
import hashlib
import hmac
from typing import Any

import numpy as np

ALIAS_VERSION = "1"
_ROUNDS = 6
_M32 = np.uint64(0xFFFFFFFF)
_C1, _C2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


class Aliases:
    """A keyed permutation of uint64 ids for one (task, kind of id)."""

    def __init__(self, secret: bytes, task_id: str, kind: str):
        self.keys = [np.uint64(int.from_bytes(hmac.new(secret, f"alias|{ALIAS_VERSION}|{kind}|{task_id}|{i}".encode(),
                                                       hashlib.sha256).digest()[:8], "big")) for i in range(_ROUNDS)]

    def _f(self, i: int, r: np.ndarray) -> np.ndarray:
        x = (r ^ self.keys[i]) * _C1  # uint64 arithmetic wraps
        x ^= x >> np.uint64(31)
        x *= _C2
        x ^= x >> np.uint64(29)
        return x & _M32

    def alias(self, ids: Any) -> np.ndarray:
        x = np.asarray(ids, dtype=np.uint64)
        left, right = x >> np.uint64(32), x & _M32
        for i in range(_ROUNDS):
            left, right = right, left ^ self._f(i, right)
        out = (left << np.uint64(32)) | right
        return np.where(x == 0, np.uint64(0), out)

    def real(self, ids: Any) -> np.ndarray:
        x = np.asarray(ids, dtype=np.uint64)
        left, right = x >> np.uint64(32), x & _M32
        for i in reversed(range(_ROUNDS)):
            left, right = right ^ self._f(i, left), left
        out = (left << np.uint64(32)) | right
        return np.where(x == 0, np.uint64(0), out)


def roots_of(secret: bytes, task_id: str) -> Aliases:
    return Aliases(secret, task_id, "root")


def l2_of(secret: bytes, task_id: str) -> Aliases:
    return Aliases(secret, task_id, "l2")


def alias_state(state: dict[str, Any], secret: bytes) -> dict[str, Any]:
    """The task state as a harness sees it: segment ids replaced by the task's root aliases."""
    out = copy.deepcopy(state)
    layer = out["neuroglancer"]["layers"][1]
    al = roots_of(secret, state["task_id"])
    layer["segments"] = [str(int(a)) for a in al.alias([int(s) for s in layer["segments"]])]
    return out
