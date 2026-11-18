"""Negative tasks: the same operations, shown just after the edit, where the right answer is "none".

Read-only toward FlyWire; reads truth/<operation_id>.json and writes truth/negatives/<operation_id>.json.
The task time is T_post = edit time + 1 ms (the truth's t_post_utc); the segments are the edit's
results, which are valid at T_post.

- merge operation -> `no_split` task: the merged segment (1 segment). Eligible only if the merge
  survived (its added edges still join one root at the check time) and nothing was split off the
  segment before then (a sample of its supervoxels all still share one root). Otherwise the segment
  still held a false merge, and "no split needed" would be a wrong label.
- split operation -> `no_merge` task: the two segments the split produced (the roots of its source
  and sink supervoxels at T_post). Eligible only if the split survived (sources and sinks on
  different roots at the check time).

The check time is the public materialization `negatives.check_materialization` (783: the published
FlyWire connectome). Each record also says whether the source edit survived, which is the label
quality of the positive task built from the same operation (that task is unchanged).

    .venv/bin/python -m build.build_negatives
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime
from typing import Any

import numpy as np

from build.make_splits import read_operations
from outer.config import load_config

NEGATIVE_VERSION = "1"


def iso(t: datetime) -> str:
    return t.isoformat(timespec="microseconds")


def _roots(cg: Any, svs: list[Any], t: datetime) -> list[int]:
    return [int(r) for r in cg.get_roots(np.asarray([int(s) for s in svs], dtype=np.uint64), timestamp=t)]


def build_one(truth: dict[str, Any], cg: Any, t_check: datetime, n_sample: int, min_together: float,
              rng: np.random.Generator) -> dict[str, Any]:
    op, typ = truth["operation_id"], truth["operation_type"]
    t_post = datetime.fromisoformat(truth["t_post_utc"])
    problems: list[str] = []
    rec: dict[str, Any] = {"negative_version": NEGATIVE_VERSION, "source_operation_id": op,
                           "source_operation_type": typ, "operation_type": "none",
                           "kind": "no_split" if typ == "merge" else "no_merge",
                           "t_utc": truth["t_post_utc"], "source_timestamp_utc": truth["timestamp_utc"],
                           "segmentation_source": truth["segmentation_source"],
                           "check_utc": iso(t_check)}
    if typ == "merge":
        segs = [str(r) for r in truth["post_edit_roots"]]
        if len(segs) != 1:
            problems.append(f"merge produced {len(segs)} roots")
        edge_svs = [s for e in truth["added_edges"] for s in e]
        survived = len(set(_roots(cg, edge_svs, t_check))) == 1 if edge_svs else False
        svs = np.asarray(cg.get_leaves(int(segs[0])), dtype=np.uint64)
        sample = svs if len(svs) <= n_sample else rng.choice(svs, n_sample, replace=False)
        cnt = collections.Counter(_roots(cg, sample, t_check))
        together = cnt.most_common(1)[0][1] / len(sample)
        rec.update(n_supervoxels=int(len(svs)), n_sampled=int(len(sample)), together_at_check=together,
                   roots_at_check=len(cnt))
        eligible = survived and together >= min_together
    else:
        # the operation's recorded source/sink supervoxels (the clicked points can sit off the segment)
        src = sorted({str(r) for r in _roots(cg, truth["source_supervoxels"], t_post)})
        snk = sorted({str(r) for r in _roots(cg, truth["sink_supervoxels"], t_post)})
        if len(src) != 1 or len(snk) != 1 or src == snk:
            problems.append(f"sides at T_post: sources {src}, sinks {snk}")
        segs = [str(src[0]), str(snk[0])]
        if not set(segs) <= set(truth["post_edit_roots"]):
            problems.append("side roots are not post-edit roots")
        s_check = set(_roots(cg, truth["source_supervoxels"], t_check))
        k_check = set(_roots(cg, truth["sink_supervoxels"], t_check))
        survived = not (s_check & k_check)
        eligible = survived
    # the segments must be valid at the task time (created by then, not yet replaced)
    ids = [int(s) for s in segs]
    created = cg.get_root_timestamps(ids)
    latest = cg.is_latest_roots(ids, timestamp=t_post)
    if not all(bool(x) for x in latest) or any(c > t_post for c in created):
        problems.append("segments not valid at T_post")
    rec.update(segments=segs, source_edit_survived=bool(survived), eligible=bool(eligible and not problems),
               problems=problems)
    return rec


def main(argv: list[str] | None = None) -> None:
    from caveclient import CAVEclient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="rebuild records that already exist")
    args = ap.parse_args(argv)
    cfg = load_config()
    neg = cfg["negatives"]
    client = CAVEclient(cfg["flywire"]["datastack"])
    cg = client.chunkedgraph
    t_check = client.materialize.get_timestamp(int(neg["check_materialization"]))
    out_dir = cfg.root / "truth" / "negatives"
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = sorted((r["operation_id"] for r in read_operations(cfg.path("data.operations_csv"))), key=int)
    rng = np.random.default_rng(0)
    stats: dict[str, collections.Counter] = {"merge": collections.Counter(), "split": collections.Counter()}
    for op in ops[: args.limit]:
        truth = json.loads((cfg.root / "truth" / f"{op}.json").read_text())
        out = out_dir / f"{op}.json"
        if out.exists() and not args.overwrite:
            rec = json.loads(out.read_text())
        else:
            rec = build_one(truth, cg, t_check, int(neg["sample_supervoxels"]), float(neg["no_split_min_together"]), rng)
            out.write_text(json.dumps(rec, indent=1) + "\n")
        s = stats[truth["operation_type"]]
        s["n"] += 1
        s["source_edit_survived"] += rec["source_edit_survived"]
        s["eligible"] += rec["eligible"]
        s["problems"] += bool(rec["problems"])
        extra = f" together {rec['together_at_check']:.4f}" if "together_at_check" in rec else ""
        print(f"  {op} {truth['operation_type']} -> {rec['kind']}: survived {rec['source_edit_survived']}"
              f"{extra} eligible {rec['eligible']} {rec['problems'] or ''}", flush=True)
    summary = {"negative_version": NEGATIVE_VERSION, "check_materialization": int(neg["check_materialization"]),
               "check_utc": iso(t_check), **{k: dict(v) for k, v in stats.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
