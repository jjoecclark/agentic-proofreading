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


# Below this best-candidate score, answer "none" (no_merge). Chosen on h000 train logs:
# hits vs threshold are flat (70-71/98) for -1.5..-0.5; -1.0 sits in the middle.
MERGE_THRESHOLD = -1.0
# Decision (round 2). Fit on train merge/no_merge logs (see NOTES.md):
#   combined = score + P_W*(p_same-0.5) + BLANK_BONUS*[blank EM sections within 3 sections]
#   merge iff combined >= COMB_TH
VERIFY_LO = -2.2      # LLM is asked only when it can flip the decision:
VERIFY_HI = 1.6       #   VERIFY_LO <= score + blank bonus <= VERIFY_HI
P_W = 4.0
BLANK_BONUS = 3.0
COMB_TH = -0.3
VERIFY_SAMPLES = 3
ALT_K = 1             # verify only the top candidate (runners-up added as many FPs as fixes)
ALT_MARGIN = 1.5


def blank_near(f, k=3):
    return bool(f and f.get("blank") and any(abs(z) <= k for z in f["blank"]))


def combined(score, p, facts):
    """p None (not asked / LLM failed) counts as 0.5."""
    return score + P_W * ((0.5 if p is None else p) - 0.5) + (BLANK_BONUS if blank_near(facts) else 0.0)


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
        cands[0]["_va"], cands[0]["_vb"] = A.v, B.v
    return cands


def verify_p(llm, world, ra, rb, cand, va, vb, workdir, n=VERIFY_SAMPLES, need_llm=lambda f: True, tag=""):
    """(mean P(same cell) from the LLM over n samples at this candidate or None, facts dict or None)."""
    import merge_verify
    from concurrent.futures import ThreadPoolExecutor
    c = cand["center"]
    try:
        facts, content = merge_verify.prepare(world, ra, rb, va, vb, c, workdir=workdir, tag=tag)
    except Exception:
        log(traceback.format_exc())
        return None, None
    log("verify facts", tag, facts)
    if llm is None or not need_llm(facts):
        return None, facts
    content = list(content)
    content[-1] = dict(content[-1], cache_control={"type": "ephemeral"})
    def one(i):
        try:
            return merge_verify.ask_content(llm, content, log=log)
        except Exception:
            log(traceback.format_exc())
            return None
    ps = [one(0)]  # first call writes the prompt cache, the rest read it
    with ThreadPoolExecutor(max(1, n - 1)) as ex:
        ps += list(ex.map(one, range(1, n)))
    ps = [p for p in ps if p is not None]
    log("p_same samples", tag, ps)
    return (float(np.mean(ps)) if ps else None), facts


def solve_merge(state, world, limits, llm=None, workdir=None):
    ra, rb = [int(s) for s in state["neuroglancer"]["layers"][1]["segments"]]
    cands = merge_candidates(world, ra, rb, limits)
    for c in cands[:10]:
        log("cand", np.round(c["center"]).tolist(),
            {k: round(float(v), 3) for k, v in c.items() if k not in ("center", "pa", "pb") and not k.startswith("_")})
    va, vb = cands[0]["_va"], cands[0]["_vb"]
    # Verify the top candidate, plus up to ALT_K-1 runners-up within ALT_MARGIN of it.
    top = cands[0]["score"]
    chk = [c for i, c in enumerate(cands[:ALT_K]) if i == 0 or c["score"] >= max(top - ALT_MARGIN, VERIFY_LO)]
    from concurrent.futures import ThreadPoolExecutor
    def ver(ic):
        i, c = ic
        p, f = verify_p(llm, world, ra, rb, c, va, vb, workdir if i == 0 else None,
                        need_llm=lambda f, s=c["score"]: VERIFY_LO <= s + (BLANK_BONUS if blank_near(f) else 0.0) <= VERIFY_HI,
                        tag=str(i))
        c["_p"], c["_facts"] = p, f
        c["_comb"] = combined(c["score"], p, f)
        return c
    with ThreadPoolExecutor(len(chk)) as ex:
        chk = list(ex.map(ver, enumerate(chk)))
    for i, c in enumerate(chk):
        log("merge cand", i, {"score": c["score"], "p_same": c["_p"], "blank_near": blank_near(c["_facts"]), "comb": c["_comb"]})
    bc = max(chk, key=lambda c: c["_comb"])
    do_merge = bc["_comb"] >= COMB_TH
    log("merge decision", {"score": bc["score"], "p_same": bc["_p"], "comb": bc["_comb"], "rank": chk.index(bc), "merge": do_merge})
    if not do_merge:
        return {"decision_version": "1", "action": "none",
                "notes": f"best comb {bc['_comb']:.2f} < {COMB_TH} (score {bc['score']:.2f} p_same {bc['_p']})"}
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
            return solve_merge(state, world, limits, llm=llm, workdir=workdir)
        if not SPLIT_ENABLED:
            return {"decision_version": "1", "action": "none", "notes": "split disabled"}
        return solve_split(state, world, limits, llm=llm)
    except Exception:
        log(traceback.format_exc())
    # On error, "none" is the better bet: merge-task fallback was rarely right.
    return {"decision_version": "1", "action": "none", "notes": "error"}
