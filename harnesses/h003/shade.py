import numpy as np, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

VIEWS = [((0, 1, 2), "x right, y down"), ((0, 2, 1), "x right, z down"), ((2, 1, 0), "z right, y down")]

def crop(v, f, c, half):
    inside = np.all(np.abs(v - c) < half, axis=1)
    keep = inside[f].all(1)
    return f[keep]

def render_shaded(meshes, c, half=6000.0, dpi=70, size=6, max_faces=120000, title="", mark=None):
    """meshes: list of (v, f, rgb). Painter's-algorithm shaded render in 3 orthogonal views."""
    fig, axes = plt.subplots(1, 3, figsize=(3 * size, size))
    light = np.array([0.3, -0.5, -0.8]); light /= np.linalg.norm(light)
    data = []
    rng = np.random.default_rng(0)
    for v, f, col in meshes:
        ff = crop(v, f, c, half)
        if len(ff) > max_faces:
            ff = ff[rng.choice(len(ff), max_faces, replace=False)]
        tri = v[ff]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        data.append((tri, n, np.array(col, float)))
    for a, ((i, j, k), name) in zip(axes, VIEWS):
        polys, cols, depth = [], [], []
        for tri, n, col in data:
            if len(tri) == 0:
                continue
            # view direction is +k (looking along k); shade by |n_k| mostly
            sh = 0.35 + 0.65 * np.abs(n[:, k])
            polys.append(tri[:, :, [i, j]] / 1000.0)
            cols.append(np.clip(col[None, :] * sh[:, None], 0, 1))
            depth.append(tri[:, :, k].mean(1))
        if polys:
            P = np.concatenate(polys); C = np.concatenate(cols); D = np.concatenate(depth)
            o = np.argsort(-D)  # far first (larger k = farther; we look from -k)
            a.add_collection(PolyCollection(P[o], facecolors=C[o], edgecolors="none", antialiased=False))
        mk = c if mark is None else np.asarray(mark, float)
        a.plot(mk[i] / 1000, mk[j] / 1000, "+", color="lime", ms=16, mew=2.5)
        a.set_xlim((c[i] - half) / 1000, (c[i] + half) / 1000)
        a.set_ylim((c[j] - half) / 1000, (c[j] + half) / 1000)
        a.invert_yaxis(); a.set_aspect("equal"); a.set_title(f"{title}{name} (um)")
        a.set_facecolor("white")
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi); plt.close(fig)
    return buf.getvalue()
