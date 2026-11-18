"""Render mesh point-cloud projections to PNG for LLM inspection."""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEWS = [((0, 1), "top view: x right, y down"), ((0, 2), "x right, z down"), ((2, 1), "z right, y down")]


def render(v, center=None, half=None, marks=None, size_in=7, dpi=80, title=None, max_pts=300000):
    """v: (N,3) nm. marks: list of (label, xyz, color). Returns PNG bytes (3 panels)."""
    p = v
    if center is not None and half is not None:
        sel = np.all(np.abs(v - np.asarray(center)) < half, axis=1)
        p = v[sel]
    if len(p) > max_pts:
        p = p[np.random.default_rng(0).choice(len(p), max_pts, replace=False)]
    fig, axes = plt.subplots(1, 3, figsize=(3 * size_in, size_in))
    for a, ((i, j), name) in zip(axes, VIEWS):
        k = 3 - i - j
        o = np.argsort(p[:, k]) if len(p) else []
        if len(p):
            a.scatter(p[o, i] / 1000, p[o, j] / 1000, s=0.3, c=p[o, k], cmap="viridis", linewidths=0)
        for lab, xyz, col in (marks or []):
            a.plot(xyz[i] / 1000, xyz[j] / 1000, "o", mfc="none", mec=col, ms=14, mew=2)
            a.text(xyz[i] / 1000, xyz[j] / 1000, f" {lab}", color=col, fontsize=13, weight="bold")
        if center is not None and half is not None:
            a.set_xlim((center[i] - half) / 1000, (center[i] + half) / 1000)
            a.set_ylim((center[j] - half) / 1000, (center[j] + half) / 1000)
        a.invert_yaxis()
        a.set_aspect("equal")
        a.set_title(name + " (µm)")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
