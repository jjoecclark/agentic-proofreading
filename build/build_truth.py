"""Build truth/<operation_id>.json for every operation (PLAN.md Section 3.3).

Run once with the CAVE token (flywire_fafb_public). Read-only toward FlyWire. For each operation:

- operation type, timestamp, T (= timestamp - 1 ms) and T_post (= T + 2 ms)
- pre-edit roots (edited supervoxels mapped at T) and post-edit roots (at T_post), both checked
  against the CSV (and the post-edit roots against the operation record)
- source and sink coordinates in nm (recorded coordinates x the segmentation's base resolution) and
  the edit center = mean of all source and sink points (the flywire-edit-sampling definition)
- a coordinate check: the root at T of the supervoxel under each source/sink point

truth/ is gitignored and never visible to harnesses or the meta-agent.

    .venv/bin/python -m build.build_truth --limit 5
    .venv/bin/python -m build.build_truth
"""

from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from build.make_splits import read_operations
from outer.config import load_config

TRUTH_VERSION = "1"


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace(" ", "T"))


def iso(t: datetime) -> str:
    return t.isoformat(timespec="microseconds")


def decode_coords(coords: list[list[Any]]) -> tuple[np.ndarray, bool]:
    """Recorded coordinates in segmentation voxels. Some 2019 records (e.g. operations 1703, 4557) store
    float64 coordinates as their raw 64-bit patterns read as integers (values ~4.7e18). Those are decoded
    back to floats; the decoded points land in the recorded supervoxels' chunks."""
    raw = [[int(v) if isinstance(v, int) else v for v in p] for p in coords]
    if any(isinstance(v, int) and abs(v) > 10**9 for p in raw for v in p):
        out = [[struct.unpack("<d", struct.pack("<q", int(v)))[0] if isinstance(v, int) and abs(v) > 10**9
                else float(v) for v in p] for p in raw]
        return np.asarray(out, dtype=float).reshape(-1, 3), True
    return np.asarray(raw, dtype=float).reshape(-1, 3), False


def build_one(row: dict[str, Any], d: dict[str, Any], cg: Any, backend: Any, seg_res: np.ndarray,
              seg_source: str) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    op, typ = row["operation_id"], row["operation_type"]
    ts = parse_ts(d["timestamp"])
    if ts != parse_ts(row["timestamp_utc"]):
        problems.append(f"timestamp {ts} != CSV {row['timestamp_utc']}")
    t = ts - timedelta(milliseconds=1)
    t_post = t + timedelta(milliseconds=2)
    if typ == "merge":
        if "added_edges" not in d:
            problems.append("merge without added_edges")
        svs = np.unique(np.asarray(d.get("added_edges", []), dtype=np.uint64).ravel())
    else:
        if "removed_edges" not in d:
            problems.append("split without removed_edges")
        svs = np.unique(np.asarray(list(d.get("source_ids", [])) + list(d.get("sink_ids", [])), dtype=np.uint64))
    pre = sorted({str(int(r)) for r in cg.get_roots(svs, timestamp=t)})
    post = sorted({str(int(r)) for r in cg.get_roots(svs, timestamp=t_post)})
    if set(pre) != set(row["before_root_ids"]):
        problems.append(f"pre-edit roots {pre} != CSV {row['before_root_ids']}")
    if set(post) != set(row["after_root_ids"]):
        problems.append(f"post-edit roots {post} != CSV {row['after_root_ids']}")
    if set(post) != {str(int(r)) for r in d.get("roots", [])}:
        problems.append(f"post-edit roots {post} != operation record {d.get('roots')}")
    expect = (2, 1) if typ == "merge" else (1, 2)
    if (len(pre), len(post)) != expect:
        problems.append(f"{typ}: {len(pre)} -> {len(post)} roots, expected {expect[0]} -> {expect[1]}")

    src_vox, dec1 = decode_coords(d.get("source_coords", []))
    snk_vox, dec2 = decode_coords(d.get("sink_coords", []))
    sources, sinks = src_vox * seg_res, snk_vox * seg_res
    if not len(sources) or not len(sinks):
        problems.append("missing source or sink coordinates")
    center = np.concatenate([sources, sinks]).mean(axis=0)

    # coordinate check: which root (at T and T_post) lies under each recorded point
    pts = np.concatenate([sources, sinks])
    sv_at = backend.supervoxels_at(pts)
    r_t = backend.roots(sv_at, t)
    r_post = backend.roots(sv_at, t_post)
    n_src = len(sources)
    check = {
        "points": int(len(pts)),
        "points_on_pre_edit_roots_at_t": int(sum(str(int(x)) in pre for x in r_t)),
        "source_roots_at_t": sorted({str(int(x)) for x in r_t[:n_src]}),
        "sink_roots_at_t": sorted({str(int(x)) for x in r_t[n_src:]}),
        "source_roots_at_t_post": sorted({str(int(x)) for x in r_post[:n_src]}),
        "sink_roots_at_t_post": sorted({str(int(x)) for x in r_post[n_src:]}),
    }
    truth = {
        "truth_version": TRUTH_VERSION,
        "operation_id": op,
        "operation_type": typ,
        "timestamp_utc": iso(ts),
        "t_utc": iso(t),
        "t_post_utc": iso(t_post),
        "pre_edit_roots": pre,
        "post_edit_roots": post,
        "sources_nm_xyz": sources.tolist(),
        "sinks_nm_xyz": sinks.tolist(),
        "edit_center_nm_xyz": center.tolist(),
        "coords_decoded_from_float64_bits": bool(dec1 or dec2),
        "edited_supervoxels": [str(int(x)) for x in svs],
        "added_edges": [[str(int(a)), str(int(b))] for a, b in d.get("added_edges", [])],
        "source_supervoxels": [str(int(x)) for x in d.get("source_ids", [])],
        "sink_supervoxels": [str(int(x)) for x in d.get("sink_ids", [])],
        "segmentation_source": seg_source,
        "coordinate_check": check,
        "problems": problems,
    }
    return truth, problems


def main(argv: list[str] | None = None) -> None:
    from caveclient import CAVEclient

    from services.world.server import FlyWireBackend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N operations (CSV order)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    rows = read_operations(cfg.path("data.operations_csv"))[: args.limit]
    out_dir = cfg.root / "truth"
    out_dir.mkdir(exist_ok=True)
    todo = [r for r in rows if args.overwrite or not (out_dir / f"{r['operation_id']}.json").exists()]
    client = CAVEclient(cfg["flywire"]["datastack"])
    cg = client.chunkedgraph
    seg_res = np.asarray(cg.segmentation_info["scales"][0]["resolution"], dtype=float)
    backend = FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
    bounds = np.asarray(backend.static_info()["bounds_nm_xyz"])
    seg_source = client.info.segmentation_source()
    print(f"{len(rows)} operations, {len(todo)} to build; segmentation resolution {seg_res.tolist()} nm")
    n_bad = 0
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        details = cg.get_operation_details([int(r["operation_id"]) for r in batch])
        details = {str(k): v for k, v in details.items()}
        for r in batch:
            d = details.get(r["operation_id"])
            if d is None:
                print(f"  {r['operation_id']}: no operation details")
                n_bad += 1
                continue
            truth, problems = build_one(r, d, cg, backend, seg_res, seg_source)
            c = np.asarray(truth["edit_center_nm_xyz"])
            if np.any(c < bounds[0]) or np.any(c > bounds[1]):
                problems.append(f"edit center {c.tolist()} outside dataset bounds")
            (out_dir / f"{r['operation_id']}.json").write_text(json.dumps(truth, indent=1) + "\n")
            n_bad += bool(problems)
            ck = truth["coordinate_check"]
            print(f"  {r['operation_id']} {r['operation_type']}: pre {len(truth['pre_edit_roots'])} post "
                  f"{len(truth['post_edit_roots'])} center {np.round(c).astype(int).tolist()} "
                  f"points on pre-edit roots {ck['points_on_pre_edit_roots_at_t']}/{ck['points']}"
                  + (f"  PROBLEMS: {problems}" if problems else ""), flush=True)
    print(f"done; {n_bad} operations with problems")


if __name__ == "__main__":
    main()
