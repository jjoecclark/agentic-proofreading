"""Split-task heuristics (v1): find X-shaped crossings in the L2 graph."""
import sys, time, itertools
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
import geom


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class L2:
    def __init__(self, world, root):
        g = world.l2_graph(root)
        self.ids = g.l2_ids
        self.pos, self.approx = world.l2_positions(g.l2_ids)
        ix = {int(x): i for i, x in enumerate(self.ids)}
        e = np.array([(ix[int(u)], ix[int(v)]) for u, v in g.edges], dtype=np.int64).reshape(-1, 2)
        e = e[e[:, 0] != e[:, 1]]
        self.e = e
        n = len(self.ids)
        self.nbr = [[] for _ in range(n)]
        for u, v in e:
            self.nbr[u].append(v); self.nbr[v].append(u)
        self.nbr = [sorted(set(x)) for x in self.nbr]
        self.deg = np.array([len(x) for x in self.nbr])

    def branch_dir(self, c, first, dist=6000.0, maxhops=4):
        """Direction from node c along the branch starting at neighbor `first` (BFS-free walk:
        follow the neighbor farthest from c)."""
        P = self.pos
        prev, cur = c, first
        pts = [P[cur]]
        for _ in range(maxhops - 1):
            if np.linalg.norm(P[cur] - P[c]) >= dist:
                break
            nxt = [x for x in self.nbr[cur] if x != prev and x != c]
            if not nxt:
                break
            cur, prev = max(nxt, key=lambda x: np.linalg.norm(P[x] - P[c])), cur
            pts.append(P[cur])
        d = np.mean(pts, 0) - P[c]
        return d / (np.linalg.norm(d) + 1e-9)


def candidates(l2):
    out = []
    for c in np.where(l2.deg >= 3)[0]:
        if l2.approx[c]:
            continue
        nb = l2.nbr[c]
        D = np.array([l2.branch_dir(c, b) for b in nb])
        best = None
        if len(nb) >= 4:
            # choose two disjoint anti-parallel pairs
            for (i, j), (k, l) in _pairings(len(nb)):
                s = -(D[i] @ D[j]) - (D[k] @ D[l])
                if best is None or s > best[0]:
                    best = (s, [nb[i], nb[j]], [nb[k], nb[l]])
            s, g1, g2 = best
            # remaining neighbors go with the line they align with best
            rest = [b for b in nb if b not in g1 + g2]
            out.append({"node": int(c), "score": float(s), "kind": "X", "g1": g1, "g2": g2, "rest": rest})
    out.sort(key=lambda x: -x["score"])
    return out


def _pairings(n):
    for a in itertools.combinations(range(n), 2):
        for b in itertools.combinations(range(n), 2):
            if a < b and not set(a) & set(b):
                yield a, b


def place_points(world, root, l2, mi, cand, step=1500.0):
    """One point per branch in cand['g1'] / cand['g2'], ~step from the junction, each inside the
    angular sector of its own branch (competing against all branches of the junction)."""
    node = cand["node"]
    c = l2.pos[node]
    comp = mi.local_component(c, 4000.0)
    V = mi.v[comp] if len(comp) > 20 else mi.v[mi.tree.query_ball_point(c, 4000.0)]
    allb = l2.nbr[node]
    D = {b: l2.branch_dir(node, b) for b in allb}
    off = V - c
    r = np.linalg.norm(off, axis=1) + 1e-9
    U = off / r[:, None]
    M = np.array([U @ D[b] for b in allb]).reshape(len(allb), -1) if len(V) else np.zeros((len(allb), 0))
    def pt(b):
        if len(V) == 0:
            p = c + step * D[b]
        else:
            k = allb.index(b)
            mine = (M[k] >= M.max(0) - 1e-9) & (r > 400)
            sc = M[k] - np.abs(r - step) / step
            sc[~mine] -= 10
            p = V[np.argmax(sc)]
        q = geom.snap_one(world, root, p)
        return (q if q is not None else p).tolist()
    src = [pt(b) for b in cand["g1"]]
    snk = [pt(b) for b in cand["g2"]]
    return src, snk


def branch_groups(l2, node):
    """Split the neighbours of a junction into (sources, sinks) single-branch groups."""
    nb = l2.nbr[node]
    if len(nb) < 2:
        return None
    D = np.array([l2.branch_dir(node, b) for b in nb])
    if len(nb) == 2:
        return [nb[0]], [nb[1]]
    if len(nb) >= 4:
        best = max(_pairings(len(nb)), key=lambda pr: -(D[pr[0][0]] @ D[pr[0][1]]) - (D[pr[1][0]] @ D[pr[1][1]]))
        (i, j), (k, l) = best
        return [nb[i]], [nb[k]]
    # degree 3
    pairs = [(0, 1), (0, 2), (1, 2)]
    cos = {p: D[p[0]] @ D[p[1]] for p in pairs}
    par = max(pairs, key=lambda p: cos[p])
    if cos[par] > 0.7:                       # two strands fusing: separate them
        return [nb[par[0]]], [nb[par[1]]]
    thr = min(pairs, key=lambda p: cos[p])   # through line; odd branch is foreign
    odd = ({0, 1, 2} - set(thr)).pop()
    return [nb[odd]], [nb[thr[0]]]


def junction_marks(l2, center, half, kmax=24):
    P = l2.pos
    sel = np.where((l2.deg >= 3) & ~l2.approx & np.all(np.abs(P - center) < half * 0.95, axis=1))[0]
    if len(sel) == 0:
        return []
    # greedy thinning: keep marks at least 2.5 um apart, nearest-to-centre first
    sel = sel[np.argsort(np.linalg.norm(P[sel] - center, axis=1))]
    keep = []
    for i in sel:
        if all(np.linalg.norm(P[i] - P[j]) > 2500 for j in keep):
            keep.append(i)
        if len(keep) >= kmax:
            break
    return keep


def nearest_junction(l2, p):
    P = l2.pos
    ok = np.where((l2.deg >= 3) & ~l2.approx)[0]
    if len(ok) == 0:
        ok = np.arange(len(P))
    return int(ok[np.argmin(np.linalg.norm(P[ok] - p, axis=1))])


def make_decision(world, root, l2, mi, node, note):
    g = branch_groups(l2, node)
    if g is None:
        return None
    cand = {"node": node, "g1": g[0], "g2": g[1]}
    src, snk = place_points(world, root, l2, mi, cand)
    return {"decision_version": "1", "action": "split", "sources_nm_xyz": src[:1], "sinks_nm_xyz": snk[:1],
            "notes": note}


def solve(state, world, limits, llm=None):
    root = int(state["neuroglancer"]["layers"][1]["segments"][0])
    l2 = L2(world, root)
    mi = geom.MeshInfo(world.mesh(root))
    cands = candidates(l2)
    node = cands[0]["node"] if cands else int(np.argmax(l2.deg))
    note = "X heuristic"
    mode = "pick"
    if llm is not None and mode == "pick" and len(cands) > 1:
        try:
            import llm_split
            # top candidates, spatially thinned (>= 8 um apart)
            top = []
            for c in cands:
                if all(np.linalg.norm(l2.pos[c["node"]] - l2.pos[t["node"]]) > 8000 for t in top):
                    top.append(c)
                if len(top) >= 12:
                    break
            rk, reason = llm_split.pick_candidate(llm, mi.v, [l2.pos[t["node"]] for t in top], log=log)
            if rk:
                node = top[rk[0]]["node"]
                note = f"llm pick {rk[0]}: {reason[:200]}"
        except Exception as e:  # noqa: BLE001
            log("llm failed", repr(e))
    elif llm is not None and mode == "locate":
        try:
            import llm_split
            pts, reason = llm_split.ask_location(llm, mi.v, log=log)
            half = 25000.0
            marks = junction_marks(l2, pts[0], half)
            p, ch = llm_split.ask_zoom(llm, mi.v, pts[0], half, [l2.pos[i] for i in marks], reason, log=log)
            node = marks[ch] if ch is not None else nearest_junction(l2, p)
            note = f"llm: {reason[:200]}"
        except Exception as e:  # noqa: BLE001
            log("llm failed", repr(e))
    d = make_decision(world, root, l2, mi, node, note)
    return d or {"decision_version": "1", "action": "none"}
