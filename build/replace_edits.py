"""Replace the edit operations that involve glia / non-neurons with the next clean ones of the same type
(user request, 2026-11-20: "glia are excluded from all examples"). Read-only toward FlyWire.

An edit "involves a non-neuron" when any of its edited supervoxels (merge: added edges; split: sources,
sinks and removed edges) lies, at the public materialization `negatives.check_materialization` (783), on
a root that `neuron_information_v2` tags with `negatives.exclude_tag_regex` (glia, not a neuron,
trachea). Such a merge joins two pieces of a non-neuron; such a split cuts one off a neuron.

This script replays the source sampler (build/replace_splits.py: flywire-edit-sampling's draw, seed 42,
same eligibility filter, now for merges as well as splits) and:

1. checks that the first 100 eligible merges and the first 100 eligible splits are exactly those of the
   original data/operations.csv (so the merge filter is the source's);
2. finds the operations of the current CSV (data.operations_csv) that involve a non-neuron;
3. continues the draw, per type, past every operation already used, to the next operations that do not
   involve a non-neuron, whose truth record builds cleanly, and (splits only, as before) whose edit
   survived to the check time;
4. writes data/operations_v3.csv (each replaced row swapped for its replacement, in draw order) and
   data/splits.json version 3 (each replacement takes the replaced operation's slot, so the split sizes
   are unchanged), and truth/<operation_id>.json for the replacements.

Older CSVs are kept for provenance. config.yaml must then point data.operations_csv at the new CSV.

    .venv/bin/python -m build.replace_edits
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from build.build_truth import build_one
from build.replace_splits import FIELDS, SAMPLER, Graph, eligible_split, operation_type, survived, unique_ids
from outer.config import load_config, sha256_file

SPLITS_VERSION = "3"  # 2: non-surviving splits replaced; 3: edits involving non-neurons (glia) replaced


def eligible_merge(g: Graph, op: int, d: dict[str, Any], threshold: int) -> dict[str, Any] | None:
    """The source sampler's filter for a merge (two roots before, the operation's one root after, each
    root before with >= threshold L2 chunks); returns its CSV row, or None."""
    if operation_type(d) != "merge" or not d.get("roots"):
        return None
    cg = g.cg()
    ts = datetime.fromisoformat(d["timestamp"])
    svs = sorted({int(sv) for e in d["added_edges"] for sv in e})
    before = sorted(set(map(int, cg.get_roots(svs, timestamp=ts - timedelta(milliseconds=1)))))
    after = sorted(set(map(int, cg.get_roots(svs, timestamp=ts + timedelta(milliseconds=1)))))
    if (len(before), len(after)) != (2, 1) or 0 in before + after or after != sorted(map(int, d["roots"])):
        return None
    before_counts = [g.l2_count(r) for r in before]
    if min(before_counts) < threshold:
        return None
    after_counts = [g.l2_count(r) for r in after]
    return {"operation_id": str(op), "operation_type": "merge", "timestamp_utc": d["timestamp"],
            "filter_side": "before", "min_l2_count": min(before_counts),
            "before_root_ids": json.dumps([str(r) for r in before]), "before_l2_counts": json.dumps(before_counts),
            "before_total_l2": sum(before_counts),
            "after_root_ids": json.dumps([str(r) for r in after]), "after_l2_counts": json.dumps(after_counts),
            "after_total_l2": sum(after_counts), "delta_l2": sum(after_counts) - sum(before_counts)}


def edit_supervoxels(d: dict[str, Any]) -> list[int]:
    svs = {int(s) for e in d.get("added_edges", []) + d.get("removed_edges", []) for s in e}
    svs |= {int(s) for s in d.get("source_ids", []) + d.get("sink_ids", [])}
    svs.discard(0)
    return sorted(svs)


def non_neuron_roots(cg: Any, d: dict[str, Any], t_check: datetime, excluded: set[int]) -> list[str]:
    svs = edit_supervoxels(d)
    roots = {int(r) for r in cg.get_roots(np.asarray(svs, dtype=np.uint64), timestamp=t_check)} if svs else set()
    return sorted(str(r) for r in roots & excluded)


def main(argv: list[str] | None = None) -> None:
    from caveclient import CAVEclient

    from services.world.server import FlyWireBackend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-candidates", type=int, default=300_000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    cfg = load_config()
    neg = cfg["negatives"]
    with open(cfg.root / "data" / "operations.csv", newline="") as f:
        orig = {r["operation_id"]: r["operation_type"] for r in csv.DictReader(f)}
    cur_csv = cfg.path("data.operations_csv")
    with open(cur_csv, newline="") as f:
        cur_rows = list(csv.DictReader(f))
    current = {r["operation_id"]: r for r in cur_rows}
    client = CAVEclient(cfg["flywire"]["datastack"])
    cg = client.chunkedgraph
    version = int(neg["check_materialization"])
    t_check = client.materialize.get_timestamp(version)
    info = client.materialize.query_table("neuron_information_v2", materialization_version=version)
    excluded = {int(r) for r in info[info.tag.str.contains(str(neg["exclude_tag_regex"]), flags=re.I, na=False)].pt_root_id}
    seg_res = np.asarray(cg.segmentation_info["scales"][0]["resolution"], dtype=float)
    backend = FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
    seg_source = client.info.segmentation_source()
    g = Graph(cfg["flywire"]["datastack"])
    thr = SAMPLER["l2_count_threshold"]

    # 2. the current operations that involve a non-neuron
    cur_ids = sorted(int(o) for o in current)
    details_cur = {}
    for i in range(0, len(cur_ids), 32):  # the server refuses long id lists
        details_cur.update({int(k): v for k, v in cg.get_operation_details(cur_ids[i:i + 32]).items()})
    drop = {op: non_neuron_roots(cg, details_cur[int(op)], t_check, excluded) for op in current}
    drop = {op: r for op, r in drop.items() if r}
    need = {t: sum(current[op]["operation_type"] == t for op in drop) for t in ("merge", "split")}
    print(f"involve a non-neuron: {len(drop)} {need}: {sorted(drop, key=int)}", flush=True)

    # 1 + 3. replay the draw
    eligible: dict[str, list[int]] = {"merge": [], "split": []}
    found: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {"merge": [], "split": []}
    rejected: dict[str, str] = {}
    checked = False
    considered = 0
    ids = unique_ids(SAMPLER["seed"], SAMPLER["min_operation_id"], SAMPLER["max_operation_id"])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while any(len(found[t]) < need[t] for t in need) or not checked:
            batch = [next(ids) for _ in range(32)]
            considered += len(batch)
            if considered > args.max_candidates:
                raise SystemExit(f"candidate limit reached; found {({t: len(v) for t, v in found.items()})}")
            details = {int(k): v for k, v in cg.get_operation_details(batch).items()}
            cands = [(op, details[op], operation_type(details[op])) for op in batch if op in details]
            cands = [c for c in cands if c[2] in ("merge", "split")]
            rows = list(pool.map(lambda c: (eligible_merge if c[2] == "merge" else eligible_split)(g, c[0], c[1], thr),
                                 cands))
            for (op, d, typ), row in zip(cands, rows, strict=True):  # draw order
                if row is None:
                    continue
                eligible[typ].append(op)
                if len(eligible[typ]) == 100:
                    want = {o for o, t in orig.items() if t == typ}
                    if {str(o) for o in eligible[typ]} != want:
                        raise SystemExit(f"replay does not reproduce the source's {typ}s")
                    print(f"replay reproduces the 100 original {typ}s ({considered} ids drawn)", flush=True)
                if str(op) in current or len(found[typ]) >= need[typ]:
                    continue
                if typ == "split" and not survived(cg, d, t_check):
                    rejected[str(op)] = "split did not survive"
                    continue
                bad = non_neuron_roots(cg, d, t_check, excluded)
                if bad:
                    rejected[str(op)] = f"non-neuron roots {bad}"
                    continue
                r = {**row, "before_root_ids": json.loads(row["before_root_ids"]),
                     "after_root_ids": json.loads(row["after_root_ids"])}
                truth, problems = build_one(r, d, cg, backend, seg_res, seg_source)
                if problems:
                    rejected[str(op)] = f"truth problems {problems}"
                    continue
                found[typ].append((op, truth, row))
                print(f"  replacement {typ} {op}", flush=True)
            checked = all(len(eligible[t]) >= 100 for t in eligible)

    truth_dir = cfg.root / "truth"
    mapping: dict[str, str] = {}
    for typ in ("merge", "split"):
        dropped = sorted((op for op in drop if current[op]["operation_type"] == typ),
                         key=lambda o: eligible[typ].index(int(o)) if int(o) in eligible[typ] else 10**9)
        for old, (op, truth, _row) in zip(dropped, found[typ], strict=True):
            (truth_dir / f"{op}.json").write_text(json.dumps(truth, indent=1) + "\n")
            mapping[old] = str(op)
    by_id = {str(op): row for t in found for op, _, row in found[t]}
    new_rows = [by_id[mapping[r["operation_id"]]] if r["operation_id"] in mapping else r for r in cur_rows]
    out_csv = cfg.root / "data" / "operations_v3.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(new_rows)

    sp_path = cfg.root / "data" / "splits.json"
    old = json.loads(sp_path.read_text())
    types = {r["operation_id"]: r["operation_type"] for r in new_rows}
    doc = {"splits_version": SPLITS_VERSION, "seed": old["seed"],
           "operations_csv": "data/operations_v3.csv", "operations_csv_sha256": sha256_file(out_csv),
           **{k: sorted((mapping.get(o, o) for o in old[k]), key=int) for k in ("train", "val", "test")},
           "replaced": mapping,
           "replaced_non_neuron_roots": drop,
           "replacement_rule": (f"edits whose edited supervoxels lie, at public materialization {version}, on a root "
                                f"tagged /{neg['exclude_tag_regex']}/ in neuron_information_v2 were replaced, per type "
                                "and in the source sampler's draw order, by the next eligible operations that are not "
                                "(splits: that also survived); each replacement takes the replaced one's slot"),
           "sampler": {**SAMPLER, "ids_drawn": considered, "rejected": rejected},
           "previous": {k: old[k] for k in old if k not in ("train", "val", "test")}}
    doc["counts"] = {k: {t: sum(types[o] == t for o in doc[k]) for t in ("merge", "split")} for k in ("train", "val", "test")}
    sp_path.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"replaced": mapping, "counts": doc["counts"], "operations_csv_v3_sha256": doc["operations_csv_sha256"]},
                     indent=1))


if __name__ == "__main__":
    main()
