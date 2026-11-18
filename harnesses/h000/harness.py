"""FlyWire proofreading harness v1. See NOTES.md."""
import json, time, traceback, sys
import numpy as np
from scipy.spatial import cKDTree

import geom


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def merge_score(c):
    s = 0.0
    for r, wt in ((1000, 1.0), (2000, 1.0), (3000, 0.5)):
        s -= wt * (abs(min(c[f"ka{r}"], 4) - 1) + abs(min(c[f"kb{r}"], 4) - 1))
        s += 0.7 * wt * c[f"opp{r}"]
    s += 0.3 * np.log1p(c["n"])
    if "gap" in c:
        s -= 4 + c["gap"] / 300
    return s


def merge_candidates(world, ra, rb, limits):
    A = geom.MeshInfo(world.mesh(ra))
    B = geom.MeshInfo(world.mesh(rb))
    cands = geom.contact_candidates(A.v, B.v, B.tree)
    cc = np.array([c["center"] for c in cands]) if cands else np.zeros((0, 3))
    for g in geom.gap_candidates(A, B, max_gap=2500.0):
        if len(cc) == 0 or np.min(np.linalg.norm(cc - g["center"], axis=1)) > 1000:
            cands.append(g)
    if not cands:
        d, j = B.tree.query(A.v)
        k = int(np.argmin(d))
        cands = [{"center": (A.v[k] + B.v[j[k]]) / 2, "n": 1, "gap": float(d[k]), "pa": A.v[k], "pb": B.v[j[k]],
                  "cluster_n": 1, "cluster_extent": 0.0}]
    t0 = time.time()
    for c in cands:
        for r in (1000, 2000, 3000):
            ka, oa, _ = A.exits(c["center"], r)
            kb, ob, _ = B.exits(c["center"], r)
            c[f"ka{r}"], c[f"kb{r}"] = ka, kb
            c[f"opp{r}"] = -float(np.dot(oa, ob) / (np.linalg.norm(oa) * np.linalg.norm(ob) + 1e-9))
        c["score"] = merge_score(c)
        if time.time() - t0 > 600:
            break
    cands = [c for c in cands if "score" in c]
    cands.sort(key=lambda c: -c["score"])
    return cands


def solve_merge(state, world, limits):
    ra, rb = [int(s) for s in state["neuroglancer"]["layers"][1]["segments"]]
    cands = merge_candidates(world, ra, rb, limits)
    for c in cands[:10]:
        log("cand", np.round(c["center"]).tolist(),
            {k: round(float(v), 3) for k, v in c.items() if k not in ("center", "pa", "pb")})
    bc = cands[0]
    best = bc["center"]
    sp = None
    if bc.get("gap", 0) > 300 and "pa" in bc:
        pa, pb = geom.snap_one(world, ra, bc["pa"]), geom.snap_one(world, rb, bc["pb"])
        if pa is not None and pb is not None:
            sp = (pa, pb)
    if sp is None:
        sp = geom.snap_pair(world, ra, rb, best)
    if sp is None:
        sp = (geom.snap_one(world, ra, bc.get("pa", best)), geom.snap_one(world, rb, bc.get("pb", best)))
        sp = tuple(best if p is None else p for p in sp)
    return {"decision_version": "1", "action": "merge",
            "points_nm_xyz": [list(map(float, sp[0])), list(map(float, sp[1]))],
            "notes": f"{len(cands)} candidates; best score {cands[0]['score']:.2f}"}


def solve_split(state, world, limits, llm=None):
    import split
    return split.solve(state, world, limits, llm=llm)


def fallback_merge(state, world):
    """Cheap fallback: closest pair of L2 centroids, snapped to each segment."""
    ra, rb = [int(s) for s in state["neuroglancer"]["layers"][1]["segments"]]
    pa = world.l2_positions(world.l2_graph(ra).l2_ids)[0]
    pb = world.l2_positions(world.l2_graph(rb).l2_ids)[0]
    d, j = cKDTree(pb).query(pa)
    k = int(np.argmin(d))
    a = geom.snap_one(world, ra, pa[k]); b = geom.snap_one(world, rb, pb[j[k]])
    a = pa[k] if a is None else a; b = pb[j[k]] if b is None else b
    return {"decision_version": "1", "action": "merge",
            "points_nm_xyz": [list(map(float, a)), list(map(float, b))], "notes": "fallback"}


def run(state, world, llm, workdir, limits):
    segs = state["neuroglancer"]["layers"][1]["segments"]
    try:
        if len(segs) == 2:
            return solve_merge(state, world, limits)
        return solve_split(state, world, limits, llm=llm)
    except Exception:
        log(traceback.format_exc())
    try:
        if len(segs) == 2:
            return fallback_merge(state, world)
        return solve_split(state, world, limits, llm=None)
    except Exception:
        log(traceback.format_exc())
    return {"decision_version": "1", "action": "none", "notes": "error"}
