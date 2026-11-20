"""FlyWire proofreading harness (round 3). See NOTES.md."""
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


# Below this best-candidate score, answer "none" (no_merge). Chosen on h000 train logs:
# hits vs threshold are flat (70-71/98) for -1.5..-0.5; -1.0 sits in the middle.
MERGE_THRESHOLD = -1.0
# LLM verifier v2 (merge_verify2.py), round 3: up to VERIFY_K spatially distinct candidates
# (>= 3 um apart) whose score could still pass are each rendered and judged; z = score +
# VERIFY_W*(p_same-0.5); merge at argmax z iff z >= VERIFY_THRESHOLD. Tuned on train (plateau
# W 3-4, threshold -1.4..-0.7), see NOTES.md.
VERIFY_W = 3.5
VERIFY_THRESHOLD = -1.2
VERIFY_SAMPLES = 2
VERIFY_K = 3
VERIFY_SEP = 3000.0
# Split localisation hit 1/50 on train in h000; answering "none" wins every no_split task.
SPLIT_ENABLED = False


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
    if cands:
        cands[0]["_A"], cands[0]["_B"] = (A.v, A.f), (B.v, B.f)
    return cands


def verify_cands(llm, world, ra, rb, cands, workdir, limits):
    """Judge eligible candidates with the LLM. Returns list of (index, z, p) for judged ones."""
    if llm is None:
        return []
    import merge_verify2 as mv2
    from concurrent.futures import ThreadPoolExecutor
    idx = []
    for i, c in enumerate(cands):
        if c["score"] + VERIFY_W * 0.5 < VERIFY_THRESHOLD:
            break  # sorted by score: no later candidate can pass either
        if all(np.linalg.norm(np.asarray(c["center"]) - np.asarray(cands[j]["center"])) > VERIFY_SEP for j in idx):
            idx.append(i)
        if len(idx) >= VERIFY_K:
            break
    if not idx:
        return []
    A, B = cands[0]["_A"], cands[0]["_B"]
    pngs = {}
    for i in idx:
        if limits.remaining_s() < 300:
            break
        try:
            pngs[i] = mv2.renders(world, ra, rb, A, B, cands[i]["center"], workdir=workdir, tag=f"_c{i}")
        except Exception:
            log(traceback.format_exc())
    jobs = [(i, s) for i in pngs for s in range(VERIFY_SAMPLES)]
    def one(job):
        i, s = job
        for attempt in range(2):
            try:
                return i, mv2.ask(llm, pngs[i], log=log)
            except Exception as ex:
                log("verify failed", i, repr(ex)[:300])
                if "402" in repr(ex) or "budget" in repr(ex):
                    return i, None
        return i, None
    ps = {}
    if jobs:
        with ThreadPoolExecutor(min(6, len(jobs))) as ex:
            for i, p in ex.map(one, jobs):
                if p is not None:
                    ps.setdefault(i, []).append(p)
    out = []
    for i, v in ps.items():
        p = float(np.mean(v))
        out.append((i, cands[i]["score"] + VERIFY_W * (p - 0.5), p))
    return out


def solve_merge(state, world, limits, llm=None, workdir=None):
    ra, rb = [int(s) for s in state["neuroglancer"]["layers"][1]["segments"]]
    cands = merge_candidates(world, ra, rb, limits)
    for c in cands[:10]:
        log("cand", np.round(c["center"]).tolist(),
            {k: round(float(v), 3) for k, v in c.items() if k not in ("center", "pa", "pb") and not k.startswith("_")})
    judged = verify_cands(llm, world, ra, rb, cands, workdir, limits)
    log("verify", [(i, round(z, 3), round(p, 3)) for i, z, p in judged])
    if judged:
        bi, comb, p = max(judged, key=lambda x: x[1])
        th = VERIFY_THRESHOLD
    else:  # LLM unavailable / nothing eligible: score-only rule of h001
        bi, comb, p, th = 0, cands[0]["score"], None, MERGE_THRESHOLD
    bc = cands[bi]
    log("merge decision", {"cand": bi, "score": bc["score"], "p_same": p, "combined": comb, "threshold": th})
    if comb < th:
        return {"decision_version": "1", "action": "none",
                "notes": f"cand {bi} score {bc['score']:.2f} p_same {p} combined {comb:.2f} < {th}"}
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
            "notes": f"{len(cands)} candidates; chose {bi} score {bc['score']:.2f} p_same {p}"}


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
            return solve_merge(state, world, limits, llm=llm, workdir=workdir)
        if not SPLIT_ENABLED:
            return {"decision_version": "1", "action": "none", "notes": "split disabled"}
        return solve_split(state, world, limits, llm=llm)
    except Exception:
        log(traceback.format_exc())
    # On error, "none" is the better bet: merge-task fallback was rarely right.
    return {"decision_version": "1", "action": "none", "notes": "error"}
