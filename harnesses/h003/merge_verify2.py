"""LLM check v2 (round 3): shaded 3D surface renders at two scales (40 um context, 12 um close-up)
plus EM sections with A/B outlines; Opus returns P(same cell). See NOTES.md."""
import json, re
import numpy as np
import shade
from merge_verify import render_em, _img

MODEL = "claude-opus-5-5"

PROMPT = """You are an expert FlyWire connectome proofreader (adult Drosophila brain, FAFB serial-section EM;
z is the section axis, 40 nm sections).

Two segments of the automatic segmentation are shown: A (red) and B (blue). The green cross marks the
place where they come closest / touch, which is the candidate join point.
Image 1: shaded 3D surface renders of A and B in a {big} um cube around the candidate (three orthogonal
projections, axes in um) - the CONTEXT view: overall shape, calibre and branching style of each segment.
Image 2: the same in a {small} um cube - the CLOSE-UP of the junction.
Image 3: EM sections through the candidate (each panel {fov} um wide), A outlined red, B outlined blue;
panel titles give the section offset.

Question: are A and B parts of the SAME neuron that the automatic segmentation wrongly split, so a
proofreader should merge them here? Or are they DIFFERENT cells that merely touch or run close (for
example, a proofreader has just cut a false merge here and the two pieces should stay apart)?

Look for:
- SAME: a segment ends abruptly (flat or jagged cut face) right at the candidate and the other picks up
  the same path with similar calibre and direction; the combined shape is a plausible single neurite /
  arbor (no loop, no impossible branching); similar morphology style on both sides (e.g. both smooth
  axon with boutons, or both fine dendrite); the EM shows no clear membrane between the red and blue
  profiles, or there is a misaligned / blurry / missing section right there.
- DIFFERENT: both segments continue past the contact (crossing, X or T contact, side by side);
  the two have clearly different morphology (calibre, varicosities, branching pattern, darkness);
  a clean continuous membrane separates red and blue in EM over several sections; joining them would
  attach a thin twig sideways onto another backbone or create a loop; one piece looks like glia.

About half of such cases are SAME and half DIFFERENT. Reason step by step but briefly (what A does at the
cross, what B does, the EM, the overall morphology), then end with JSON on the last line:
{{"p_same": <0..1>, "reason": "<one sentence>"}}"""

RED, BLUE = (1, .15, .15), (.15, .35, 1)


def renders(world, ra, rb, A, B, c, workdir=None, tag=""):
    """A, B: (vertices, faces) of the welded meshes. Returns 3 PNGs."""
    c = np.asarray(c, float)
    ms = [(A[0], A[1], RED), (B[0], B[1], BLUE)]
    p1 = shade.render_shaded(ms, c, half=20000.0, dpi=60)
    p2 = shade.render_shaded(ms, c, half=6000.0, dpi=60)
    p3 = render_em(world, ra, rb, c)
    if workdir is not None:
        for k, p in enumerate((p1, p2, p3)):
            open(f"{workdir}/mv2{tag}_{k}.png", "wb").write(p)
    return [p1, p2, p3]


def ask(llm, pngs, log=print, max_tokens=2000):
    msg = llm.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [_img(p) for p in pngs] +
                   [{"type": "text", "text": PROMPT.format(big=40, small=12, fov=4)}]}])
    txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    log("merge_verify2:", txt[-500:])
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
