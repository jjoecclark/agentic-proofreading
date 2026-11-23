"""Scoring (PLAN.md Section 5, score_version 3). With build/, the only reader of truth/.

Hit radius R = scoring.hit_radius_um. A point "lies on root X at time t" when the supervoxel at that
point (base resolution of the supervoxel layer) maps to root X at t; background lies on no root.

- Merge hit: action "merge", exactly 2 points, one on each of the two pre-edit roots at T, and both
  points within R of the edit center.
- Split hit: action "split", >= 1 source and >= 1 sink, every point on the pre-edit root at T, all
  sources on one post-edit root and all sinks on the other at T + 2 ms, and the midpoint of the closest
  source-sink pair within R of the edit center.
- No-edit hit (negative tasks, kind `no_split` or `no_merge`; build/build_negatives.py): action
  "none". Any merge or split is a miss (`edit_not_needed`).

The **score** of a set of tasks is the mean of the hit rates of the kinds present in it (merge, split,
no_merge, no_split), so each kind weighs the same however many tasks it has: a harness that always
answers "none" scores 50% on a set with all four kinds. Hits are still counted per task.
- Anything else is a miss, with its reason recorded.

Task kinds: `merge`, `split` (edit tasks) and `no_merge`, `no_split` (negatives). Summaries report
the score, each kind, edit tasks, no-edit tasks, and all tasks.

Per task: hit, miss reason, distance from the predicted location to the edit center (merge: midpoint
of the two points; split: midpoint of the closest source-sink pair), cost, tokens, wall time.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SCORE_VERSION = "4"  # 2: negative tasks; 3: score = mean of per-kind hit rates; 4: true_location_nm_xyz (monitor)
DECISION_VERSION = "1"

# miss reasons
NO_DECISION = "no_decision"
CRASH = "crash"
TIMEOUT = "timeout"
BUDGET = "budget_exhausted"
INVALID_JSON = "invalid_json"
INVALID_SCHEMA = "invalid_schema"
NONE_ACTION = "none"
WRONG_ACTION = "wrong_action"
WRONG_POINT_COUNT = "wrong_point_count"
OFF_SEGMENTS = "points_not_on_segments"
NOT_SEPARATED = "sides_not_separated"
TOO_FAR = "too_far"
EDIT_NOT_NEEDED = "edit_not_needed"

EDIT_KINDS = ("merge", "split")
NEGATIVE_KINDS = ("no_merge", "no_split")


def true_location(truth: dict[str, Any]) -> list[float] | None:
    """Where to look at a task: the edit center, or a negative's first neuron's recorded point. Stored in
    score.json for the outer harness's monitor (outer/dashboard_export.py); never shown to the meta-agent
    (outer/feedback.py and score_train pass only whitelisted fields)."""
    if truth.get("edit_center_nm_xyz") is not None:
        return [float(v) for v in truth["edit_center_nm_xyz"]]
    neurons = truth.get("neurons") or []
    return [float(v) for v in neurons[0]["position_nm_xyz"]] if neurons else None


def kind_of(truth_or_score: dict[str, Any]) -> str:
    return truth_or_score.get("kind") or truth_or_score.get("operation_type", "?")


class RootLookup(Protocol):
    def supervoxels_at(self, points_nm_xyz: np.ndarray) -> np.ndarray: ...
    def roots(self, supervoxel_ids: np.ndarray, t: datetime) -> np.ndarray: ...


# ----------------------------------------------------------------------------- truth access


class Truth:
    """truth/ reader: task_id -> the truth record of its edit (truth/<op>.json) or, for a negative
    task, of its no-edit case (truth/negatives/<op>.json)."""

    def __init__(self, truth_dir: Path):
        self.dir = Path(truth_dir)
        tm = json.loads((self.dir / "task_map.json").read_text())
        self.path_of = {v["task_id"]: self.dir / f"{op}.json" for op, v in tm["operations"].items()}
        self.path_of.update({v["task_id"]: self.dir / "negatives" / f"{op}.json"
                             for op, v in tm.get("negatives", {}).items()})

    def __call__(self, task_id: str) -> dict[str, Any]:
        return json.loads(self.path_of[task_id].read_text())


# ----------------------------------------------------------------------------- decisions


def _points(v: Any) -> np.ndarray | None:
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[1] != 3 or not np.all(np.isfinite(a)):
        return None
    return a


def validate_decision(d: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return (normalized decision, None) or (None, miss reason)."""
    if not isinstance(d, dict):
        return None, INVALID_SCHEMA
    if str(d.get("decision_version")) != DECISION_VERSION:
        return None, INVALID_SCHEMA
    a = d.get("action")
    if a == "none":
        return {"action": "none"}, None
    if a == "merge":
        p = _points(d.get("points_nm_xyz"))
        if p is None:
            return None, INVALID_SCHEMA
        return {"action": "merge", "points": p}, None
    if a == "split":
        s, k = _points(d.get("sources_nm_xyz")), _points(d.get("sinks_nm_xyz"))
        if s is None or k is None:
            return None, INVALID_SCHEMA
        return {"action": "split", "sources": s, "sinks": k}, None
    return None, INVALID_SCHEMA


def closest_pair_midpoint(sources: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(sources[:, None, :] - sinks[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return (sources[i] + sinks[j]) / 2.0


def score_decision(dec: dict[str, Any], truth: dict[str, Any], lookup: RootLookup,
                   radius_nm: float) -> dict[str, Any]:
    """Hit / miss reason / distance for a validated decision."""
    if kind_of(truth) in NEGATIVE_KINDS:
        hit = dec["action"] == "none"
        return {"hit": hit, "miss_reason": None if hit else EDIT_NOT_NEEDED, "distance_nm": None}
    typ = truth["operation_type"]
    center = np.asarray(truth["edit_center_nm_xyz"], dtype=float)
    t = datetime.fromisoformat(truth["t_utc"])
    t_post = datetime.fromisoformat(truth["t_post_utc"])
    pre = {int(r) for r in truth["pre_edit_roots"]}
    post = {int(r) for r in truth["post_edit_roots"]}
    out: dict[str, Any] = {"hit": False, "miss_reason": None, "distance_nm": None}
    if dec["action"] == "none":
        out["miss_reason"] = NONE_ACTION
        return out

    if dec["action"] == "merge":
        p = dec["points"]
        if len(p):
            out["distance_nm"] = float(np.linalg.norm(p.mean(axis=0) - center)) if len(p) == 2 else None
        if typ != "merge":
            out["miss_reason"] = WRONG_ACTION
            return out
        if len(p) != 2:
            out["miss_reason"] = WRONG_POINT_COUNT
            return out
        roots = [int(r) for r in lookup.roots(lookup.supervoxels_at(p), t)]
        out["point_roots_at_t"] = [str(r) for r in roots]
        if set(roots) != pre or len(set(roots)) != 2:
            out["miss_reason"] = OFF_SEGMENTS
            return out
        if np.any(np.linalg.norm(p - center, axis=1) > radius_nm):
            out["miss_reason"] = TOO_FAR
            return out
        out["hit"] = True
        return out

    s, k = dec["sources"], dec["sinks"]
    if len(s) and len(k):
        out["distance_nm"] = float(np.linalg.norm(closest_pair_midpoint(s, k) - center))
    if typ != "split":
        out["miss_reason"] = WRONG_ACTION
        return out
    if not len(s) or not len(k):
        out["miss_reason"] = WRONG_POINT_COUNT
        return out
    pts = np.concatenate([s, k])
    svs = lookup.supervoxels_at(pts)
    r_t = {int(r) for r in lookup.roots(svs, t)}
    if r_t != pre:
        out["miss_reason"] = OFF_SEGMENTS
        return out
    r_post = [int(r) for r in lookup.roots(svs, t_post)]
    src_post, snk_post = set(r_post[:len(s)]), set(r_post[len(s):])
    if not (len(src_post) == 1 and len(snk_post) == 1 and src_post != snk_post and (src_post | snk_post) <= post):
        out["miss_reason"] = NOT_SEPARATED
        return out
    if out["distance_nm"] > radius_nm:
        out["miss_reason"] = TOO_FAR
        return out
    out["hit"] = True
    return out


# ----------------------------------------------------------------------------- one run directory


def read_json(p: Path) -> Any:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def score_run(run_dir: Path, task_id: str, truth: dict[str, Any], lookup: RootLookup,
              radius_nm: float) -> dict[str, Any]:
    """Score runs/<harness>/<split>/<task_id>/ and write score.json there."""
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run_meta.json") or {}
    usage = read_json(run_dir / "usage.json") or {}
    res: dict[str, Any] = {"hit": False, "miss_reason": None, "distance_nm": None}
    dpath = run_dir / "decision.json"
    action = None
    if dpath.exists():
        try:
            raw = json.loads(dpath.read_text())
        except json.JSONDecodeError:
            raw = None
            res["miss_reason"] = INVALID_JSON
        if raw is not None:
            dec, why = validate_decision(raw)
            action = raw.get("action") if isinstance(raw, dict) else None
            if dec is None:
                res["miss_reason"] = why
            else:
                res = score_decision(dec, truth, lookup, radius_nm)
    else:
        if meta.get("timed_out"):
            res["miss_reason"] = TIMEOUT
        elif usage.get("refused_budget"):
            res["miss_reason"] = BUDGET
        elif meta.get("error") or meta.get("exit_code") not in (0, None):
            res["miss_reason"] = CRASH
        else:
            res["miss_reason"] = NO_DECISION
    score = {
        "score_version": SCORE_VERSION,
        "task_id": task_id,
        "kind": kind_of(truth),
        "operation_type": truth["operation_type"],
        "true_location_nm_xyz": true_location(truth),
        "action": action,
        **res,
        "usd": float(usage.get("usd") or 0.0),
        "tokens": {k: int(usage.get(k) or 0) for k in ("input_tokens", "output_tokens",
                                                        "cache_creation_input_tokens", "cache_read_input_tokens")},
        "requests": int(usage.get("requests") or 0),
        "wall_s": meta.get("wall_s"),
        "radius_nm": radius_nm,
        "scored_at": time.time(),
    }
    (run_dir / "score.json").write_text(json.dumps(score, indent=1) + "\n")
    return score


# ----------------------------------------------------------------------------- summaries


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


KINDS = (*EDIT_KINDS, *NEGATIVE_KINDS)


def kind_score(scores: list[dict[str, Any]]) -> float | None:
    """Mean of the per-kind hit rates over the kinds present (None for no tasks)."""
    rates = []
    for k in KINDS:
        ss = [s for s in scores if kind_of(s) == k]
        if ss:
            rates.append(sum(bool(s["hit"]) for s in ss) / len(ss))
    return sum(rates) / len(rates) if rates else None


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    def block(ss: list[dict[str, Any]]) -> dict[str, Any]:
        n, k = len(ss), sum(bool(s["hit"]) for s in ss)
        lo, hi = wilson(k, n)
        d = [s["distance_nm"] for s in ss if s.get("distance_nm") is not None]
        return {"n": n, "hits": k, "hit_rate": (k / n) if n else None, "ci95": [lo, hi],
                "median_distance_nm": statistics.median(d) if d else None,
                "n_with_distance": len(d)}

    out = {"score_version": SCORE_VERSION, "score": kind_score(scores), "all": block(scores),
           "edit": block([s for s in scores if kind_of(s) in EDIT_KINDS]),
           "no_edit": block([s for s in scores if kind_of(s) in NEGATIVE_KINDS]),
           **{k: block([s for s in scores if kind_of(s) == k]) for k in KINDS}}
    usd = [float(s.get("usd") or 0.0) for s in scores]
    out["usd_total"] = sum(usd)
    out["usd_per_task"] = (sum(usd) / len(usd)) if usd else None
    walls = [s["wall_s"] for s in scores if s.get("wall_s") is not None]
    out["median_wall_s"] = statistics.median(walls) if walls else None
    reasons: dict[str, int] = {}
    for s in scores:
        if not s["hit"]:
            reasons[s["miss_reason"] or "unknown"] = reasons.get(s["miss_reason"] or "unknown", 0) + 1
    out["miss_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return out


def paired_gain(new: dict[str, dict[str, Any]], best: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Two harnesses on the same tasks (dicts task_id -> score): hits, and the score (mean per-kind hit
    rate) with its gain in percentage points."""
    common = sorted(set(new) & set(best))
    a = sum(bool(new[t]["hit"]) for t in common)
    b = sum(bool(best[t]["hit"]) for t in common)
    sa, sb = kind_score([new[t] for t in common]), kind_score([best[t] for t in common])
    return {"n": len(common), "new_hits": a, "best_hits": b, "gain": a - b,
            "new_only": sum(bool(new[t]["hit"]) and not best[t]["hit"] for t in common),
            "best_only": sum(bool(best[t]["hit"]) and not new[t]["hit"] for t in common),
            "new_score": sa, "best_score": sb,
            "score_gain_pp": None if sa is None or sb is None else 100.0 * (sa - sb)}


class CachedLookup:
    """Memoizes point -> supervoxel and (supervoxel, t) -> root lookups of a backend."""

    def __init__(self, backend: Any):
        self.b = backend
        self._sv: dict[tuple[float, float, float], int] = {}
        self._root: dict[tuple[int, str], int] = {}

    def supervoxels_at(self, points_nm_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_nm_xyz, dtype=float).reshape(-1, 3)
        keys = [tuple(map(float, p)) for p in pts]
        todo = [k for k in dict.fromkeys(keys) if k not in self._sv]
        if todo:
            got = self.b.supervoxels_at(np.array(todo))
            self._sv.update({k: int(v) for k, v in zip(todo, got, strict=True)})
        return np.array([self._sv[k] for k in keys], dtype=np.uint64)

    def roots(self, supervoxel_ids: np.ndarray, t: datetime) -> np.ndarray:
        tk = t.isoformat()
        ids = [int(x) for x in supervoxel_ids]
        todo = [x for x in dict.fromkeys(ids) if (x, tk) not in self._root]
        if todo:
            got = self.b.roots(np.array(todo, dtype=np.uint64), t)
            self._root.update({(x, tk): int(r) for x, r in zip(todo, got, strict=True)})
        return np.array([self._root[(x, tk)] for x in ids], dtype=np.uint64)
