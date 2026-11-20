"""LLM check of a merge candidate: are segments A and B one cell joined at this location?

Renders (1) mesh point clouds of A (red) and B (blue) around the candidate in three projections, and
(2) EM sections through the candidate with A/B mask outlines; asks Opus for P(same cell)."""
import base64, io, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

MODEL = "claude-opus-5-5"

VIEWS = [((0, 1), "x right, y down"), ((0, 2), "x right, z down"), ((2, 1), "z right, y down")]

PROMPT = """You are an expert FlyWire connectome proofreader (adult Drosophila brain, FAFB serial-section EM,
voxels 4x4x40 nm, z = section axis).

Two segments from the automatic segmentation are shown: A (red) and B (blue). The white cross / image
centre marks the place where they come closest / touch. Image 1: point clouds of both meshes in a
{cube} um cube around that place, in three orthogonal projections (axis units um). Image 2: EM sections
through that place (each panel {fov} um wide), A outlined red, B outlined blue; panels are at different z
(section offsets given in the titles).

Question: are A and B parts of the SAME neuron, wrongly split by the automatic segmentation, so that a
proofreader should merge them here? Or are they two DIFFERENT cells that merely touch / run alongside
(e.g. a proofreader has just cut a false merge here)?

Evidence for SAME cell: one segment ends abruptly exactly where the other begins, the two continue
each other's direction with similar calibre (a smooth continuation of one neurite), the boundary
between them in EM is not a real membrane (or coincides with a misaligned / damaged / dark section),
an orphan fragment that fills a gap in an arbor.
Evidence for DIFFERENT cells: both segments continue past the contact (they cross or run side by side),
a clear membrane separates the red and blue profiles in EM over several sections, they meet at an
odd angle (X or T crossing), very different calibre or cytoplasm texture, the joined shape would form a
loop or put a thin twig between two backbones, or the contact is a flat cut face along a long side-by-side
stretch.

About half of such cases in this dataset are SAME and half DIFFERENT. Think briefly, then end with JSON
on the last line: {{"p_same": <0..1>, "reason": "<one sentence>"}}"""


def _png(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def render_meshes(va, vb, c, half=6000.0, dpi=70):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sa = va[np.all(np.abs(va - c) < half, axis=1)]
    sb = vb[np.all(np.abs(vb - c) < half, axis=1)]
    rng = np.random.default_rng(0)
    if len(sa) > 60000: sa = sa[rng.choice(len(sa), 60000, replace=False)]
    if len(sb) > 60000: sb = sb[rng.choice(len(sb), 60000, replace=False)]
    for a, ((i, j), name) in zip(axes, VIEWS):
        a.scatter(sa[:, i] / 1000, sa[:, j] / 1000, s=0.4, c="red", linewidths=0, alpha=0.5)
        a.scatter(sb[:, i] / 1000, sb[:, j] / 1000, s=0.4, c="blue", linewidths=0, alpha=0.5)
        a.plot(c[i] / 1000, c[j] / 1000, "x", color="k", ms=14, mew=2)
        a.set_xlim((c[i] - half) / 1000, (c[i] + half) / 1000)
        a.set_ylim((c[j] - half) / 1000, (c[j] + half) / 1000)
        a.invert_yaxis(); a.set_aspect("equal"); a.set_title(name + " (um)")
    plt.tight_layout()
    return _png(fig, dpi)


def render_em(world, ra, rb, c, half_xy=2000.0, dz_secs=(-6, -2, 0, 2, 6), dpi=60):
    zh = max(abs(d) for d in dz_secs) * 40 + 40
    box = [[c[0] - half_xy, c[1] - half_xy, c[2] - zh], [c[0] + half_xy, c[1] + half_xy, c[2] + zh]]
    em = world.em(box, (8, 8, 40))
    seg = world.segmentation(box, (16, 16, 40))
    E = em.data_zyx; S = seg.data_zyx
    zc_e = em.index_of(c)[0]; zc_s = seg.index_of(c)[0]
    n = len(dz_secs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.3))
    for a, dz in zip(axes, dz_secs):
        ze = int(np.clip(zc_e + dz, 0, E.shape[0] - 1)); zs = int(np.clip(zc_s + dz, 0, S.shape[0] - 1))
        img = E[ze].astype(float)
        a.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(0, img.shape[1], img.shape[0], 0))
        sl = S[zs]
        sl = np.kron(sl, np.ones((2, 2), dtype=sl.dtype))[:img.shape[0], :img.shape[1]]
        for rid, col in ((ra, "red"), (rb, "blue")):
            m = sl == np.uint64(rid)
            if m.any():
                a.contour(m.astype(float), levels=[0.5], colors=col, linewidths=1.2)
        a.plot(img.shape[1] / 2, img.shape[0] / 2, "+", color="yellow", ms=10, mew=1)
        a.set_title(f"section {dz:+d} ({dz * 40:+d} nm)"); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    return _png(fig, dpi)


def _img(png):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}}


def ask(llm, world, ra, rb, va, vb, c, log=print, max_tokens=1500, workdir=None, tag=""):
    c = np.asarray(c, float)
    p1 = render_meshes(va, vb, c)
    p2 = render_em(world, ra, rb, c)
    if workdir is not None:
        open(f"{workdir}/mv{tag}_mesh.png", "wb").write(p1)
        open(f"{workdir}/mv{tag}_em.png", "wb").write(p2)
    msg = llm.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [_img(p1), _img(p2),
                                               {"type": "text", "text": PROMPT.format(cube=12, fov=4)}]}])
    txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    log("merge_verify:", txt[-600:])
    m = re.findall(r"\{[^{}]*\"p_same\"[^{}]*\}", txt, re.S)
    if m:
        try:
            js = json.loads(m[-1])
            return float(js["p_same"]), js.get("reason", "")
        except Exception:
            pass
    m = re.findall(r"p_same\"?\s*[:=]\s*([0-9.]+)", txt)
    if m:
        return float(m[-1]), ""
    raise ValueError("no p_same in reply")
