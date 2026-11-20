"""Geometry helpers for the FlyWire proofreading harness."""
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
import scipy.sparse as sp
import scipy.sparse.csgraph as cg


def weld(mesh):
    """Merge duplicate vertices (chunk boundaries). Returns (verts float64, faces)."""
    v = mesh.vertices_nm_xyz.astype(np.float64)
    f = mesh.faces.astype(np.int64)
    if len(v) == 0:
        return v, f
    key = np.round(v).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    return v[first], inv[f]


class MeshInfo:
    def __init__(self, mesh):
        self.v, self.f = weld(mesh)
        self.tree = cKDTree(self.v)
        n = len(self.v)
        e = np.concatenate([self.f[:, [0, 1]], self.f[:, [1, 2]], self.f[:, [2, 0]]])
        self.adj = sp.coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        self.adj = (self.adj + self.adj.T).tocsr()

    def local_component(self, c, r):
        """Indices of vertices within r of c that are mesh-connected (inside the ball) to the vertex nearest c."""
        idx = np.array(self.tree.query_ball_point(c, r), dtype=np.int64)
        if len(idx) == 0:
            return idx
        sub = self.adj[idx][:, idx]
        _, lab = cg.connected_components(sub, directed=False)
        d = np.linalg.norm(self.v[idx] - c, axis=1)
        return idx[lab == lab[np.argmin(d)]]

    def exits(self, c, r, shell=250.0, link=200.0):
        """Number of separate places the local component crosses the sphere of radius r."""
        comp = self.local_component(c, r)
        if len(comp) == 0:
            return 0, np.zeros(3), 0
        p = self.v[comp]
        d = np.linalg.norm(p - c, axis=1)
        sh = p[d > r - shell]
        mean_off = (p - c).mean(0)
        if len(sh) == 0:
            return 0, mean_off, len(comp)
        t = cKDTree(sh)
        pr = t.query_pairs(link, output_type='ndarray')
        n = len(sh)
        g = sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)) if len(pr) else sp.coo_matrix((n, n))
        k, lab = cg.connected_components(g, directed=False)
        # ignore tiny shell clusters
        cnt = np.bincount(lab)
        k = int((cnt >= 3).sum()) or 1
        return k, mean_off, len(comp)


def _cluster(points, link):
    n = len(points)
    if n == 1:
        return np.zeros(1, int)
    t = cKDTree(points)
    pairs = t.query_pairs(link, output_type='ndarray')
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else sp.coo_matrix((n, n))
    return cg.connected_components(g, directed=False)[1]


def contact_candidates(va, vb, tb, thresh=150.0, link=500.0, cell=2000.0):
    """Candidate merge locations where A comes within `thresh` of B. Big contact
    clusters are subdivided into ~cell-sized pieces."""
    d, _ = tb.query(va, distance_upper_bound=thresh)
    close = va[np.isfinite(d)]
    out = []
    if len(close) == 0:
        return out
    lab = _cluster(close, link)
    for l in np.unique(lab):
        pts = close[lab == l]
        ext = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 0.0
        groups = [pts]
        if ext > 1.5 * cell:
            key = np.floor(pts / cell).astype(np.int64)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            groups = [pts[inv.ravel() == j] for j in range(inv.max() + 1)]
        for g in groups:
            c = g.mean(0)
            c = g[np.argmin(np.linalg.norm(g - c, axis=1))]
            out.append({"center": c, "n": len(g), "cluster_n": len(pts), "cluster_extent": ext})
    return out


def gap_candidates(ia, ib, max_gap=2500.0, k=10):
    """Near-approach candidates (no contact), using vertices of A near B."""
    d, j = ib.tree.query(ia.v, distance_upper_bound=max_gap)
    ok = np.isfinite(d)
    if not ok.any():
        return []
    pa = ia.v[ok]; pb = ib.v[j[ok]]; dd = d[ok]
    lab = _cluster(pa, 1000.0)
    out = []
    for l in np.unique(lab):
        m = lab == l
        q = np.argmin(dd[m])
        out.append({"center": (pa[m][q] + pb[m][q]) / 2, "n": int(m.sum()), "gap": float(dd[m][q]),
                    "pa": pa[m][q], "pb": pb[m][q],
                    "cluster_n": int(m.sum()), "cluster_extent": 0.0})
    out.sort(key=lambda c: c["gap"])
    return out[:k]


def snap_pair(world, ra, rb, c):
    """Fetch segmentation around c; return (pa, pb) voxel centers on ra / rb near c and near each other."""
    c = np.asarray(c, float)
    for half in [(1200, 1200, 400), (2400, 2400, 800), (4000, 4000, 1600)]:
        h = np.asarray(half, float)
        seg = world.segmentation([(c - h).tolist(), (c + h).tolist()], (16, 16, 40))
        d = seg.data_zyx
        ma = d == np.uint64(ra)
        mb = d == np.uint64(rb)
        if not (ma.any() and mb.any()):
            continue
        res = np.asarray(seg.resolution_nm_xyz, float)[::-1]
        ci = np.array(seg.index_of(c))
        st = ndimage.generate_binary_structure(3, 1)
        adjA = ma & ndimage.binary_dilation(mb, st)
        if adjA.any():
            idx = np.argwhere(adjA)
            ia = idx[np.argmin(np.linalg.norm((idx - ci) * res, axis=1))]
            idb = np.argwhere(mb)
            ib = idb[np.argmin(np.linalg.norm((idb - ia) * res, axis=1))]
        else:
            # pair minimizing |a-b| + 0.5*(|a-c|+|b-c|)
            dtb, indb = ndimage.distance_transform_edt(~mb, sampling=res, return_indices=True)
            idxa = np.argwhere(ma)
            cost = dtb[tuple(idxa.T)] + 0.5 * np.linalg.norm((idxa - ci) * res, axis=1)
            ia = idxa[np.argmin(cost)]
            ib = indb[:, ia[0], ia[1], ia[2]]
        return seg.center_nm_of(ia), seg.center_nm_of(ib)
    return None


def snap_one(world, root, p):
    """Voxel center on `root` nearest to p (nm), or None."""
    p = np.asarray(p, float)
    for half in [(400, 400, 120), (1000, 1000, 300), (2500, 2500, 800)]:
        h = np.asarray(half, float)
        seg = world.segmentation([(p - h).tolist(), (p + h).tolist()], (16, 16, 40))
        m = seg.data_zyx == np.uint64(root)
        if not m.any():
            continue
        res = np.asarray(seg.resolution_nm_xyz, float)[::-1]
        idx = np.argwhere(m)
        ci = np.array(seg.index_of(p))
        return seg.center_nm_of(idx[np.argmin(np.linalg.norm((idx - ci) * res, axis=1))])
    return None
