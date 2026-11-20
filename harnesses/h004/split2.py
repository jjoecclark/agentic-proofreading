"""Split tasks v2 (round 2): balanced-cut candidates on the L2 graph + LLM gate.

Idea (measured on train by trilaterating true cut locations): a false merge joins two cells of
comparable size, so removing a 2 um ball around the true cut leaves two large pieces. For every
(thinned) L2 node we remove the ball and take the size of the 2nd-largest remaining component;
the top 3 spatially diverse maxima contain the true cut in ~1/3 of split tasks. But a correctly
merged segment (no_split) looks the same, so each candidate is shown to Opus (red / blue = the two
sides) and we split only where Opus is confident the two sides are different cells.
"""
import sys, time, base64, io, json, re
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geom
import merge_verify as mv

MODEL = "claude-opus-5-5"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class L2:
    def __init__(self, world, root):
        g = world.l2_graph(root)
        self.pos, self.approx = world.l2_positions(g.l2_ids)
        self.pos = np.asarray(self.pos, float)
        ix = {int(x): i for i, x in enumerate(g.l2_ids)}
        e = np.array([(ix[int(u)], ix[int(v)]) for u, v in g.edges], dtype=np.int64).reshape(-1, 2)
        self.e = e[e[:, 0] != e[:, 1]]
        self.n = len(g.l2_ids)

    def labels(self, c, R):
        """Components after removing nodes within R of c. Returns (lab, a, b, size_b); lab=-1 inside."""
        d = np.linalg.norm(self.pos - c, axis=1)
        keep = d > R
        e = self.e
        m = keep[e[:, 0]] & keep[e[:, 1]]
        g = sp.coo_matrix((np.ones(m.sum()), (e[m, 0], e[m, 1])), shape=(self.n, self.n))
        _, lab = cg.connected_components(g, directed=False)
        lab = np.where(keep, lab, -1)
        s = np.bincount(lab[keep]) if keep.any() else np.zeros(1, int)
        o = np.argsort(-s)
        if len(o) < 2:
            return lab, (o[0] if len(o) else -5), -5, 0
        return lab, int(o[0]), int(o[1]), int(s[o[1]])


def balanced_candidates(l2, R=2000.0, thin=1000.0, sep=3000.0, k=3, t_budget=240.0):
    cand = np.where(~l2.approx)[0]
    tr = cKDTree(l2.pos[cand])
    keep = np.ones(len(cand), bool)
    sel = []
    for i in np.random.default_rng(0).permutation(len(cand)):
        if keep[i]:
            sel.append(cand[i]); keep[tr.query_ball_point(l2.pos[cand[i]], thin)] = False
    t0 = time.time()
    sc = []
    for i in sel:
        sc.append(l2.labels(l2.pos[i], R)[3])
        if time.time() - t0 > t_budget:
            break
    sel = np.array(sel[:len(sc)]); sc = np.array(sc, float)
    out = []
    for i in np.argsort(-sc):
        p = l2.pos[sel[i]]
        if all(np.linalg.norm(p - q["center"]) > sep for q in out):
            out.append({"center": p, "second": int(sc[i]), "frac": float(sc[i] / l2.n)})
        if len(out) >= k:
            break
    return out


def mesh_sides(M, l2, lab, a, b, c, box=6000.0, r0s=(700, 1000, 1400, 1900), want=1100.0):
    """Split local mesh vertices into side a / side b around c. Returns dict with points pa, pb
    (mesh vertices ~1 um from c on each side) and labelled local vertices, or None."""
    loc = np.where(np.all(np.abs(M.v - c) < box, axis=1))[0]
    if len(loc) < 10:
        return None
    lv = M.v[loc]; d = np.linalg.norm(lv - c, axis=1)
    _, j = cKDTree(l2.pos).query(lv); L = lab[j]
    sub = M.adj[loc][:, loc]
    for r0 in r0s:
        ki = np.where(d > r0)[0]
        k, cl = cg.connected_components(sub[ki][:, ki], directed=False)
        far = d[ki] > 2000
        va = np.bincount(cl[far & (L[ki] == a)], minlength=k)
        vb = np.bincount(cl[far & (L[ki] == b)], minlength=k)
        ca = (va > vb) & (va > 0); cb = (vb > va) & (vb > 0)
        if not (ca.any() and cb.any()):
            continue
        ina = ki[ca[cl]]; inb = ki[cb[cl]]
        def pick(ii):
            return lv[ii[np.argmin(np.abs(d[ii] - max(want, r0 + 100)))]]
        pa, pb = pick(ina), pick(inb)
        # label the rest (incl. the ball) by nearest labelled vertex, for the pictures
        labd = np.full(len(lv), -1)
        labd[ina] = 0; labd[inb] = 1
        lt = cKDTree(lv[labd >= 0])
        rest = np.where(labd < 0)[0]
        if len(rest):
            dd, jj = lt.query(lv[rest])
            ok = dd < 1500
            labd[rest[ok]] = labd[labd >= 0][jj[ok]]
        return {"pa": pa, "pb": pb, "r0": r0, "va": lv[labd == 0], "vb": lv[labd == 1]}
    return None


def render_overview(l2, lab, a, b, c, dpi=55):
    P = l2.pos / 1000.0
    col = np.full(l2.n, "lightgray", dtype=object)
    col[lab == a] = "red"; col[lab == b] = "blue"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, ((i, j), name) in zip(axes, mv.VIEWS):
        for u, v in l2.e:
            ax.plot(P[[u, v], i], P[[u, v], j], c="k", lw=0.25, alpha=0.5)
        ax.scatter(P[:, i], P[:, j], c=list(col), s=5, linewidths=0)
        ax.plot(c[i] / 1000, c[j] / 1000, "x", color="lime", ms=18, mew=3)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_title(name + " (um)")
    plt.tight_layout()
    return mv._png(fig, dpi)


def render_em_sides(world, root, va, vb, c, half_xy=2000.0, dz_secs=(-6, -2, 0, 2, 6), dpi=60):
    zh = max(abs(d) for d in dz_secs) * 40 + 40
    box = [[c[0] - half_xy, c[1] - half_xy, c[2] - zh], [c[0] + half_xy, c[1] + half_xy, c[2] + zh]]
    em = world.em(box, (8, 8, 40))
    seg = world.segmentation(box, (16, 16, 40))
    E = em.data_zyx; S = seg.data_zyx == np.uint64(root)
    pts = np.concatenate([va, vb]); labs = np.r_[np.zeros(len(va)), np.ones(len(vb))]
    tr = cKDTree(pts)
    zc_e = em.index_of(c)[0]; zc_s = seg.index_of(c)[0]
    res = np.asarray(seg.resolution_nm_xyz, float); org = np.asarray(seg.origin_nm_xyz, float)
    fig, axes = plt.subplots(1, len(dz_secs), figsize=(5 * len(dz_secs), 5.3))
    for ax, dz in zip(axes, dz_secs):
        ze = int(np.clip(zc_e + dz, 0, E.shape[0] - 1)); zs = int(np.clip(zc_s + dz, 0, S.shape[0] - 1))
        img = E[ze].astype(float)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(0, img.shape[1], img.shape[0], 0))
        m = S[zs]
        if m.any():
            yy, xx = np.nonzero(m)
            q = np.c_[(xx + 0.5) * res[0] + org[0], (yy + 0.5) * res[1] + org[1],
                      np.full(len(xx), (zs + 0.5) * res[2] + org[2])]
            _, jj = tr.query(q)
            L = np.full(m.shape, -1.0); L[yy, xx] = labs[jj]
            for k, colr in ((0, "red"), (1, "blue")):
                mk = (L == k)
                if mk.any():
                    mk2 = np.kron(mk, np.ones((2, 2)))[:img.shape[0], :img.shape[1]]
                    ax.contour(mk2.astype(float), levels=[0.5], colors=colr, linewidths=1.2)
        ax.plot(img.shape[1] / 2, img.shape[0] / 2, "+", color="yellow", ms=10, mew=1)
        ax.set_title(f"section {dz:+d} ({dz * 40:+d} nm)"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return mv._png(fig, dpi)


PROMPT = """You are an expert FlyWire connectome proofreader (adult Drosophila brain, FAFB serial-section EM,
voxels 4x4x40 nm, z = section axis).

You see ONE segment of the automatic segmentation. We consider cutting it at the marked place; red and blue
are the two sides that the cut would separate.
Image 1: the whole segment (L2 skeleton nodes; red/blue = the two sides, grey = within 2 um of the cut,
green cross = the cut place), three orthogonal projections, axis units um.
Image 2: mesh point clouds of the two sides in a {cube} um cube around the cut place (three projections).
Image 3: EM sections through the cut place ({fov} um wide panels), the two sides outlined red / blue,
panels at different z.

Question: is this segment a FALSE MERGE of two different cells (or a cell plus a glial / foreign fragment)
joined exactly here, so that a proofreader should cut here? Or is it ONE cell, and red and blue are just two
parts of the same neuron (e.g. the cut would sever a normal neurite or a normal branch point)?

Evidence for a FALSE MERGE here: red and blue each look like a separate neuron / large arbor and meet only
here; they meet at an odd angle (X or T crossing, a neurite touching another one side-on), both continue
past the contact, a membrane separates red and blue profiles in EM, very different calibre or texture,
two parallel strands fused, a thin bridge between two backbones.
Evidence for ONE cell: one smooth continuous neurite passes through the cut place with consistent calibre,
a normal branch point (a thinner branch leaving a parent), no membrane between red and blue in EM.

Most segments shown are a single correct cell, and most proposed cut places are wrong; only answer a high
probability when the evidence is clear. Think briefly, then end with JSON on the last line:
{{"p_false_merge": <0..1>, "reason": "<one sentence>"}}"""


def ask(llm, world, root, l2, lab, a, b, c, sides, log=log, max_tokens=1500, workdir=None, tag=""):
    c = np.asarray(c, float)
    p0 = render_overview(l2, lab, a, b, c)
    p1 = mv.render_meshes(sides["va"], sides["vb"], c)
    p2 = render_em_sides(world, root, sides["va"], sides["vb"], c)
    if workdir is not None:
        for nm, p in (("ov", p0), ("mesh", p1), ("em", p2)):
            open(f"{workdir}/sv{tag}_{nm}.png", "wb").write(p)
    msg = llm.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [mv._img(p0), mv._img(p1), mv._img(p2),
                                               {"type": "text", "text": PROMPT.format(cube=12, fov=4)}]}])
    txt = "".join(bl.text for bl in msg.content if getattr(bl, "type", "") == "text")
    log("split_verify", tag, txt[-400:])
    m = re.findall(r"\{[^{}]*\"p_false_merge\"[^{}]*\}", txt, re.S)
    if m:
        try:
            return float(json.loads(m[-1])["p_false_merge"])
        except Exception:
            pass
    m = re.findall(r"p_false_merge\"?\s*[:=]\s*([0-9.]+)", txt)
    if m:
        return float(m[-1])
    raise ValueError("no p_false_merge in reply")


def solve(state, world, limits, llm=None, workdir=None, p_th=0.6, k=3, samples=1):
    from concurrent.futures import ThreadPoolExecutor
    root = int(state["neuroglancer"]["layers"][1]["segments"][0])
    none = {"decision_version": "1", "action": "none"}
    l2 = L2(world, root)
    cands = balanced_candidates(l2, k=k)
    log("split cands", [(np.round(q["center"]).tolist(), q["second"], round(q["frac"], 2)) for q in cands])
    if not cands or llm is None:
        return dict(none, notes="no candidates")
    M = geom.MeshInfo(world.mesh(root))
    jobs = []
    for ci, q in enumerate(cands):
        lab, a, b, s2 = l2.labels(q["center"], 2000.0)
        if b < 0:
            continue
        sides = mesh_sides(M, l2, lab, a, b, q["center"])
        if sides is None:
            continue
        q.update(lab=lab, a=a, b=b, sides=sides, mid=(sides["pa"] + sides["pb"]) / 2)
        for s in range(samples):
            jobs.append((ci, s))
    def one(job):
        ci, s = job
        q = cands[ci]
        try:
            return ci, ask(llm, world, root, l2, q["lab"], q["a"], q["b"], q["mid"], q["sides"],
                           workdir=workdir if s == 0 else None, tag=f"{ci}")
        except Exception as ex:
            log("split ask failed", repr(ex)[:300])
            return ci, None
    ps = {}
    with ThreadPoolExecutor(max(1, min(6, len(jobs)))) as ex:
        for ci, p in ex.map(one, jobs):
            if p is not None:
                ps.setdefault(ci, []).append(p)
    if not ps:
        return dict(none, notes="no verdicts")
    pm = {ci: float(np.mean(v)) for ci, v in ps.items()}
    log("split p_false_merge", pm)
    best = max(pm, key=lambda ci: pm[ci])
    if pm[best] < p_th:
        return dict(none, notes=f"max p_false_merge {pm[best]:.2f} < {p_th}")
    q = cands[best]
    pa = geom.snap_one(world, root, q["sides"]["pa"]); pb = geom.snap_one(world, root, q["sides"]["pb"])
    if pa is None or pb is None:
        return dict(none, notes="snap failed")
    return {"decision_version": "1", "action": "split", "sources_nm_xyz": [list(map(float, pa))],
            "sinks_nm_xyz": [list(map(float, pb))],
            "notes": f"cand {best} p_false_merge {pm[best]:.2f} second {q['second']}"}
