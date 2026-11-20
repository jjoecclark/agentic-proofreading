"""LLM check of a merge candidate (v2): are segments A and B one cell joined at this location?

Evidence shown to Opus:
  1. mesh point clouds of A (red) / B (blue) in a 12 um cube around the candidate, 3 projections;
  2. native-resolution EM (8 nm px, 2.4 um FOV) on 7 sections around the candidate with thin A/B
     outlines, plus one raw (no overlay) panel of the centre section;
  3. measured facts: blank (missing) EM sections near the candidate, z-ranges of A and B near it,
     amount of A-B contact.
Returns P(same cell)."""
import base64, io, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import ndimage

MODEL = "claude-opus-5-5"

VIEWS = [((0, 1), "x right, y down"), ((0, 2), "x right, z down"), ((2, 1), "z right, y down")]

PROMPT = """You are an expert FlyWire connectome proofreader (adult Drosophila brain, FAFB serial-section EM,
voxels 4x4x40 nm, z = section axis).

Two segments are shown: A (red) and B (blue). The image centre / black cross marks the place where they
come closest or touch. Image 1: point clouds of both meshes in a {cube} um cube around that place, three
orthogonal projections (axis units um). Image 2: EM sections at full 8 nm resolution, each panel {fov} um
wide, A outlined red, B outlined blue, section offset in the panel title; the last panel is the centre
section without outlines so you can inspect membranes.

Measured facts near this place:
{facts}

Question: are A and B parts of the SAME neuron that the automatic segmentation broke, so a proofreader
should merge them here? Or are they DIFFERENT cells?

How these cases arise (about half are each):
- SAME: the segmentation broke one neurite, typically at missing/blank or misaligned EM sections, at a
  very thin neck, or at a dark/damaged patch. Typical look: one segment ends abruptly exactly where the
  other begins, they continue each other's direction with similar calibre, and in EM the red and blue
  profiles are the same cell (no membrane between them; the red profile in one section sits where the
  blue profile is in the next, with the same cytoplasm texture and organelles).
- DIFFERENT: the automatic segmentation had wrongly joined two cells here and a proofreader has just
  cut them apart, so they always touch and their shared boundary can look weak. Typical look: both
  segments continue past the contact (crossing, T-junction, or running side by side), a continuous dark
  membrane separates red from blue in EM, different calibre or cytoplasm (e.g. glia: darker, sheet-like,
  wrapping), or the join would form an implausible shape (twig bridging two backbones, loop).
Look carefully at the EM: is there a real membrane between red and blue in the sections where both
appear?

Think briefly, then end with JSON on the last line: {{"p_same": <0..1>, "reason": "<one sentence>"}}"""


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


def fetch_em(world, box):
    """EM at 8 nm (fallbacks: 16 nm upsampled, 4 nm downsampled). Returns (zyx uint8 at 8 nm, origin)."""
    last = None
    for res, f in (((8, 8, 40), 1), ((16, 16, 40), 2), ((4, 4, 40), 0.5)):
        try:
            v = world.em(box, res)
            d = v.data_zyx
            if f == 2:
                d = np.repeat(np.repeat(d, 2, axis=1), 2, axis=2)
            elif f == 0.5:
                d = d[:, ::2, ::2]
            return d, np.asarray(v.origin_nm_xyz, float)
        except Exception as e:  # server bounds quirks on some levels
            last = e
    raise last


def blank_sections(world, c, n=10, half_xy=1000.0):
    """Offsets (in sections, -n..n) of EM sections that are entirely blank near c (missing data)."""
    c = np.asarray(c, float)
    box = [[c[0] - half_xy, c[1] - half_xy, c[2] - n * 40 - 20], [c[0] + half_xy, c[1] + half_xy, c[2] + n * 40 + 20]]
    d, org = fetch_em(world, box)
    zc = int(np.floor((c[2] - org[2]) / 40))
    flat = d.reshape(d.shape[0], -1)
    if (flat.max(1) == 0).all():
        return None  # everything blank: unknown (missing chunk), not informative
    return [int(z - zc) for z in np.nonzero(flat.max(1) == 0)[0]]


def context(world, ra, rb, c, half=(1200, 1200, 400)):
    """Facts + arrays for rendering: seg (16 nm) and EM (8 nm) around c."""
    c = np.asarray(c, float); h = np.asarray(half, float)
    box = [(c - h).tolist(), (c + h).tolist()]
    seg = world.segmentation(box, (16, 16, 40))
    S = seg.data_zyx
    E, eorg = fetch_em(world, box)
    zc = seg.index_of(c)[0]
    A = S == np.uint64(ra); B = S == np.uint64(rb)
    za = np.nonzero(A.any((1, 2)))[0] - zc
    zb = np.nonzero(B.any((1, 2)))[0] - zc
    zab = (A[:-1] & B[1:]) | (B[:-1] & A[1:])
    xy = 0
    for ax in (1, 2):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(None, -1); s2[ax] = slice(1, None)
        xy += int(((A[tuple(s1)] & B[tuple(s2)]) | (B[tuple(s1)] & A[tuple(s2)])).sum())
    try:
        bl = blank_sections(world, c)
    except Exception:
        bl = None
    facts = {"blank": bl, "za": (int(za.min()), int(za.max())) if len(za) else None,
             "zb": (int(zb.min()), int(zb.max())) if len(zb) else None,
             "shared_secs": int((A.any((1, 2)) & B.any((1, 2))).sum()),
             "z_contacts": int(zab.sum()), "xy_contacts": xy}
    return facts, S, E, zc


def facts_text(f):
    lines = []
    bl = f["blank"]
    if bl is None:
        lines.append("- blank (missing) EM sections: unknown")
    else:
        near = [z for z in bl if abs(z) <= 3]
        lines.append(f"- blank (missing) EM sections within +-10 sections of the centre: "
                     f"{bl if bl else 'none'}" + (" -> a segmentation break at missing data is likely here"
                                                  if near else ""))
    lines.append(f"- within +-10 sections (+-400 nm) of the centre, A occupies sections {f['za']} and B {f['zb']} "
                 f"(relative to the centre); both appear in {f['shared_secs']} of the same sections")
    lines.append(f"- A-B touching voxel faces (16 nm voxels): {f['xy_contacts']} in-plane, {f['z_contacts']} across sections")
    return "\n".join(lines)


def _outline(m):
    return m & ~ndimage.binary_erosion(m)


def render_em(S, E, zc, ra, rb, fov_px=300, dz_secs=(-5, -3, -1, 0, 1, 3, 5)):
    """Grid of native 8 nm EM crops with 1-px outlines; last panel raw centre section."""
    ny, nx = E.shape[1], E.shape[2]
    cy, cx = ny // 2, nx // 2
    hy = min(fov_px // 2, cy); hx = min(fov_px // 2, cx)
    panels = []
    for dz in list(dz_secs) + [None]:
        z = zc + (0 if dz is None else dz)
        z = int(np.clip(z, 0, min(E.shape[0], S.shape[0]) - 1))
        img = E[z, cy - hy:cy + hy, cx - hx:cx + hx]
        rgb = np.stack([img] * 3, -1).astype(np.uint8)
        title = "centre, raw" if dz is None else f"section {dz:+d}"
        if img.max() == 0:
            title += " BLANK"
        if dz is not None:
            sl = np.repeat(np.repeat(S[z], 2, axis=0), 2, axis=1)[:ny, :nx][cy - hy:cy + hy, cx - hx:cx + hx]
            for rid, col in ((ra, (255, 0, 0)), (rb, (40, 80, 255))):
                o = _outline(sl == np.uint64(rid))
                rgb[o] = col
        rgb[hy, hx - 4:hx + 5] = (255, 255, 0); rgb[hy - 4:hy + 5, hx] = (255, 255, 0)
        panels.append((title, rgb))
    h, w = 2 * hy, 2 * hx
    cols = 4; rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (w + 6), rows * (h + 22)), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for i, (title, rgb) in enumerate(panels):
        x0 = (i % cols) * (w + 6); y0 = (i // cols) * (h + 22)
        canvas.paste(Image.fromarray(rgb), (x0, y0 + 20))
        dr.text((x0 + 4, y0 + 4), title, fill=(0, 0, 0))
    buf = io.BytesIO(); canvas.save(buf, format="PNG")
    return buf.getvalue()


def _img(png):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}}


def prepare(world, ra, rb, va, vb, c, workdir=None, tag=""):
    c = np.asarray(c, float)
    p1 = render_meshes(va, vb, c)
    f, S, E, zc = context(world, ra, rb, c)
    p2 = render_em(S, E, zc, ra, rb)
    if workdir is not None:
        open(f"{workdir}/mv{tag}_mesh.png", "wb").write(p1)
        open(f"{workdir}/mv{tag}_em.png", "wb").write(p2)
    return f, [_img(p1), _img(p2), {"type": "text", "text": PROMPT.format(cube=12, fov=2.4, facts=facts_text(f))}]


def ask_content(llm, content, log=print, max_tokens=1200):
    msg = llm.messages.create(model=MODEL, max_tokens=max_tokens,
                              messages=[{"role": "user", "content": content}])
    txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    log("merge_verify:", txt[-500:].replace("\n", " "))
    m = re.findall(r"\{[^{}]*\"p_same\"[^{}]*\}", txt, re.S)
    if m:
        try:
            return float(json.loads(m[-1])["p_same"])
        except Exception:
            pass
    m = re.findall(r"p_same\"?\s*[:=]\s*([0-9.]+)", txt)
    if m:
        return float(m[-1])
    raise ValueError("no p_same in reply")
