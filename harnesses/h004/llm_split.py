"""LLM-vision localisation of a false merge in a single segment."""
import base64, json, re, sys
import numpy as np
import anthropic
import render

MODEL = "claude-opus-5-5"

PROMPT = """You are an expert FlyWire connectome proofreader. The images show ONE segment from the automatic
segmentation of the adult Drosophila brain (FAFB EM volume), drawn as a point cloud of its mesh in three
orthogonal projections (axis units are micrometres; colour encodes depth along the hidden axis).

This segment contains exactly one false merge that a human proofreader is about to split: parts of two
different cells (or a cell and a glial/orphan fragment) have been joined. Typical signs: two parallel strands
that fuse into one; a branch that crosses another making an X; a piece whose calibre, texture or direction is
inconsistent with the rest of the arbor; a second soma or a second primary neurite; a straight, unbranched
passing fibre attached to an arbor; a loop (real neurons are trees).

Find the single location where the cut should be made (the point where the wrong piece attaches).
Read coordinates carefully from the axes of at least two views. Reply with JSON only:
{"reason": "<one or two sentences>", "x_um": <float>, "y_um": <float>, "z_um": <float>,
 "alternatives": [[x_um, y_um, z_um], ...up to 3 more]}"""


def _img(png):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}}


def ask_location(llm, verts, log=print, max_tokens=1500):
    png = render.render(verts, dpi=80)
    msg = llm.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [_img(png), {"type": "text", "text": PROMPT}]}])
    txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    log("llm:", txt[:800])
    m = re.search(r"\{.*\}", txt, re.S)
    js = json.loads(m.group(0))
    pts = [[js["x_um"], js["y_um"], js["z_um"]]] + list(js.get("alternatives", []))[:3]
    return [np.asarray(p, float) * 1000 for p in pts], js.get("reason", "")


ZOOM_PROMPT = """Zoomed views ({w:.0f} µm box) of the same segment around the region you flagged
("{reason}"). Numbered circles mark branch points of the segment's skeleton.
Which numbered branch point is where the wrongly merged piece attaches (where the cut belongs)?
If none is right, give your best coordinates instead. Reply with JSON only:
{{"reason": "<short>", "choice": <number or null>, "x_um": <float>, "y_um": <float>, "z_um": <float>}}"""


def ask_zoom(llm, verts, center, half, marks, reason, log=print, max_tokens=800):
    png = render.render(verts, center=center, half=half,
                        marks=[(str(i), p, "red") for i, p in enumerate(marks)], dpi=80)
    msg = llm.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [_img(png), {"type": "text", "text": ZOOM_PROMPT.format(
            w=2 * half / 1000, reason=reason.replace('"', "'"))}]}])
    txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    log("llm zoom:", txt[:600])
    js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    ch = js.get("choice")
    if isinstance(ch, int) and 0 <= ch < len(marks):
        return np.asarray(marks[ch], float), ch
    return np.array([js["x_um"], js["y_um"], js["z_um"]], float) * 1000, None


PICK_PROMPT = """You are an expert FlyWire connectome proofreader. The images show ONE segment from the automatic
segmentation of the adult Drosophila brain (FAFB EM), drawn as a point cloud of its mesh in three orthogonal
projections (µm; colour = depth). The segment contains exactly one false merge (parts of two different cells,
or a cell and a glial/orphan fragment, joined). Typical signs: two parallel strands fusing into one; a fibre
crossing another making an X; a piece whose calibre or direction is inconsistent with the arbor; a second soma
or primary neurite; a straight passing fibre attached to an arbor; a loop.

Numbered circles mark candidate junctions found by a geometric detector. Which one is most likely the
place where the wrong piece attaches? Reply with JSON only:
{"reason": "<short>", "ranking": [<best number>, <second>, <third>]}"""


def pick_candidate(llm, verts, marks, log=print, max_tokens=600):
    png = render.render(verts, marks=[(str(i), p, "red") for i, p in enumerate(marks)], dpi=80)
    msg = llm.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [_img(png), {"type": "text", "text": PICK_PROMPT}]}])
    txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    log("llm pick:", txt[:600])
    js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    rk = [int(i) for i in js.get("ranking", []) if isinstance(i, (int, float)) and 0 <= int(i) < len(marks)]
    return rk, js.get("reason", "")
