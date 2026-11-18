"""Replace the split operations whose edit did not survive with the next surviving ones (user request,
2026-11-18). Read-only toward FlyWire.

The source data (flywire-edit-sampling, scripts/sample_operations.py at the pinned commit) draws
operation IDs uniformly from [1, 3,000,000] with random.Random(42), without repeats, and keeps the
first 100 merges and 100 splits in draw order whose separated pair has at least 100 L2 chunks per
segment (a split: one root before, the two operation roots after). This script replays that draw
for splits only and:

1. checks that the first 100 eligible splits are exactly the 100 splits of data/operations.csv;
2. continues the draw until 100 eligible splits have **survived** to the check time (public
   materialization negatives.check_materialization: the operation's source and sink supervoxels are
   still on different roots), and whose truth record builds without problems (build/build_truth.py);
3. writes data/operations_v2.csv (the original rows, each non-surviving split replaced by the next
   surviving one) and data/splits.json version 2: every replacement takes the slot (train / val /
   test) of the split it replaces, in draw order, so the split sizes and the merges are unchanged;
4. writes truth/<operation_id>.json for the replacements.

The original data/operations.csv is kept unchanged for provenance. config.yaml must then point
data.operations_csv at operations_v2.csv (with its sha256 prefix).

    .venv/bin/python -m build.replace_splits
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from build.build_truth import build_one
from outer.config import load_config, sha256_file

SPLITS_VERSION = "2"  # 2: non-surviving splits replaced
FIELDS = ["operation_id", "operation_type", "timestamp_utc", "filter_side", "min_l2_count",
          "before_root_ids", "before_l2_counts", "before_total_l2",
          "after_root_ids", "after_l2_counts", "after_total_l2", "delta_l2"]
SAMPLER = {"seed": 42, "min_operation_id": 1, "max_operation_id": 3_000_000, "l2_count_threshold": 100,
           "source": "flywire-edit-sampling scripts/sample_operations.py @ 05f9f38"}


def unique_ids(seed: int, lo: int, hi: int):
    """The source sampler's draw order (same algorithm)."""
    rng, seen = random.Random(seed), set()
    while len(seen) < hi - lo + 1:
        op = rng.randrange(lo, hi + 1)
        if op not in seen:
            seen.add(op)
            yield op


def operation_type(d: dict[str, Any]) -> str | None:
    added, removed = d.get("added_edges", []), d.get("removed_edges", [])
    return "merge" if added and not removed else "split" if removed and not added else None


class Graph:
    """Thread-local chunkedgraph clients and memoized L2 counts."""

    def __init__(self, datastack: str):
        self.datastack, self.local, self._counts, self._lock = datastack, threading.local(), {}, threading.Lock()

    def cg(self) -> Any:
        if not hasattr(self.local, "cg"):
            from caveclient import CAVEclient
            self.local.cg = CAVEclient(self.datastack).chunkedgraph
        return self.local.cg

    def l2_count(self, root: int) -> int:
        with self._lock:
            if root in self._counts:
                return self._counts[root]
        n = len(set(map(int, self.cg().get_leaves(root, stop_layer=2))))
        if n == 0:
            raise ValueError(f"no L2 nodes for root {root}")
        with self._lock:
            self._counts[root] = n
        return n


def eligible_split(g: Graph, op: int, d: dict[str, Any], threshold: int) -> dict[str, Any] | None:
    """The source sampler's filter for a split; returns its CSV row, or None."""
    if operation_type(d) != "split" or not d.get("roots"):
        return None
    cg = g.cg()
    ts = datetime.fromisoformat(d["timestamp"])
    svs = sorted({int(sv) for e in d["removed_edges"] for sv in e})
    before = sorted(set(map(int, cg.get_roots(svs, timestamp=ts - timedelta(milliseconds=1)))))
    after = sorted(set(map(int, cg.get_roots(svs, timestamp=ts + timedelta(milliseconds=1)))))
    if (len(before), len(after)) != (1, 2) or 0 in before + after or after != sorted(map(int, d["roots"])):
        return None
    after_counts = [g.l2_count(r) for r in after]
    if min(after_counts) < threshold:
        return None
    before_counts = [g.l2_count(r) for r in before]
    return {"operation_id": str(op), "operation_type": "split", "timestamp_utc": d["timestamp"],
            "filter_side": "after", "min_l2_count": min(after_counts),
            "before_root_ids": json.dumps([str(r) for r in before]), "before_l2_counts": json.dumps(before_counts),
            "before_total_l2": sum(before_counts),
            "after_root_ids": json.dumps([str(r) for r in after]), "after_l2_counts": json.dumps(after_counts),
            "after_total_l2": sum(after_counts), "delta_l2": sum(after_counts) - sum(before_counts)}


def survived(cg: Any, d: dict[str, Any], t_check: datetime) -> bool:
    s = set(map(int, cg.get_roots(np.asarray(d["source_ids"], dtype=np.uint64), timestamp=t_check)))
    k = set(map(int, cg.get_roots(np.asarray(d["sink_ids"], dtype=np.uint64), timestamp=t_check)))
    return not (s & k)


def main(argv: list[str] | None = None) -> None:
    from caveclient import CAVEclient

    from services.world.server import FlyWireBackend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-candidates", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    cfg = load_config()
    orig_csv = cfg.root / "data" / "operations.csv"
    with open(orig_csv, newline="") as f:
        orig_rows = list(csv.DictReader(f))
    orig_splits = {r["operation_id"] for r in orig_rows if r["operation_type"] == "split"}
    client = CAVEclient(cfg["flywire"]["datastack"])
    cg = client.chunkedgraph
    t_check = client.materialize.get_timestamp(int(cfg["negatives"]["check_materialization"]))
    seg_res = np.asarray(cg.segmentation_info["scales"][0]["resolution"], dtype=float)
    backend = FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
    seg_source = client.info.segmentation_source()
    g = Graph(cfg["flywire"]["datastack"])
    thr = SAMPLER["l2_count_threshold"]

    eligible: list[tuple[int, dict[str, Any], dict[str, Any], bool]] = []  # (op, details, row, survived)
    considered = 0
    ids = unique_ids(SAMPLER["seed"], SAMPLER["min_operation_id"], SAMPLER["max_operation_id"])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while sum(1 for e in eligible if e[3]) < 100:
            batch = [next(ids) for _ in range(32)]
            considered += len(batch)
            if considered > args.max_candidates:
                raise SystemExit(f"candidate limit reached; {len(eligible)} eligible splits so far")
            details = {int(k): v for k, v in cg.get_operation_details(batch).items()}
            cands = [(op, details[op]) for op in batch if op in details and operation_type(details[op]) == "split"]
            rows = list(pool.map(lambda x: eligible_split(g, x[0], x[1], thr), cands))
            for (op, d), row in zip(cands, rows, strict=True):  # draw order
                if row is None:
                    continue
                ok = survived(cg, d, t_check)
                eligible.append((op, d, row, ok))
                if len(eligible) == 100:  # the original sample ends here
                    got = {str(e[0]) for e in eligible}
                    if got != orig_splits:
                        raise SystemExit(f"replay does not reproduce the source sample: {len(got ^ orig_splits)} "
                                         f"operations differ")
                    print(f"replay reproduces the 100 original splits ({considered} ids drawn)", flush=True)
                if sum(1 for e in eligible if e[3]) == 100:
                    break
            print(f"  drawn {considered}: eligible splits {len(eligible)}, surviving "
                  f"{sum(1 for e in eligible if e[3])}", flush=True)

    dropped = [str(e[0]) for e in eligible[:100] if not e[3]]          # draw order
    added = [e for e in eligible[100:] if e[3]]
    # each replacement must build a clean truth record (same checks as the original data)
    truth_dir = cfg.root / "truth"
    replacements = []
    for op, d, row, _ in added:
        r = {**row, "before_root_ids": json.loads(row["before_root_ids"]), "after_root_ids": json.loads(row["after_root_ids"])}
        truth, problems = build_one(r, d, cg, backend, seg_res, seg_source)
        if problems:
            print(f"  skip {op}: truth problems {problems}")
            continue
        (truth_dir / f"{op}.json").write_text(json.dumps(truth, indent=1) + "\n")
        replacements.append(row)
    if len(replacements) < len(dropped):
        raise SystemExit(f"only {len(replacements)} clean replacements for {len(dropped)} dropped splits; rerun")
    replacements = replacements[: len(dropped)]
    mapping = {old: new["operation_id"] for old, new in zip(dropped, replacements, strict=True)}

    new_rows = []
    by_id = {r["operation_id"]: r for r in replacements}
    for r in orig_rows:
        new_rows.append(by_id[mapping[r["operation_id"]]] if r["operation_id"] in mapping else r)
    out_csv = cfg.root / "data" / "operations_v2.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(new_rows)

    sp_path = cfg.root / "data" / "splits.json"
    old = json.loads(sp_path.read_text())
    types = {r["operation_id"]: r["operation_type"] for r in new_rows}
    doc = {"splits_version": SPLITS_VERSION, "seed": old["seed"],
           "operations_csv": "data/operations_v2.csv", "operations_csv_sha256": sha256_file(out_csv),
           **{k: sorted((mapping.get(o, o) for o in old[k]), key=int) for k in ("train", "val", "test")},
           "replaced": mapping,
           "replacement_rule": ("split operations whose source and sink supervoxels share a root at public "
                                f"materialization {cfg['negatives']['check_materialization']} were replaced, in the "
                                "source sampler's draw order, by the next eligible splits that did not; each "
                                "replacement takes the replaced split's slot"),
           "sampler": {**SAMPLER, "ids_drawn": considered, "eligible_splits": len(eligible),
                       "eligible_not_surviving": [str(e[0]) for e in eligible if not e[3]]},
           "previous": {"splits_version": old["splits_version"], "operations_csv_sha256": old["operations_csv_sha256"]}}
    doc["counts"] = {k: {t: sum(types[o] == t for o in doc[k]) for t in ("merge", "split")} for k in ("train", "val", "test")}
    sp_path.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"dropped": dropped, "replaced": mapping, "counts": doc["counts"],
                      "operations_csv_v2_sha256": doc["operations_csv_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
