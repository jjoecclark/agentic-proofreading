"""Negative tasks (right answer "none"), built from proofread FlyWire neurons (negative_version 2).

Read-only toward FlyWire. Reads the 200 edit operations' truth (truth/<op>.json) only to date the
negatives and to keep them away from the edits; writes truth/negatives/<negative_id>.json and
truth/negatives/summary.json.

- `no_split`: one proofread neuron. It is one cell, so nothing should be split off.
- `no_merge`: two proofread neurons that share >= `negatives.min_synapses` synapses (A->B plus B->A,
  synapses_nt_v1 with cleft_score > `negatives.min_cleft_score`, no autapses). They touch where they
  synapse but are different cells, so they should not be merged.

Neuron pool: the `proofread_neurons` table at materialization `negatives.check_materialization`
(783), as roots at that version, minus
- anything tagged glia / not a neuron / trachea in `neuron_information_v2` (`negatives.exclude_tag_regex`);
  every proofread neuron also carries a neuron super_class (hierarchical_neuron_annotations), and roots
  without one are dropped, so glia never enter;
- the neurons that hold any of the 200 edits (their edited supervoxels' roots at 783) and every
  pre/post-edit root of an edit.
Each neuron is used at most once over all negatives, so no neuron is shared between splits.

Dating (no timestamp leak): edit tasks are dated 2019-2023, proofread roots are defined at 783 (2023-09).
A negative's time T is an edit time of the matching edit type (merge times for no_merge, split times for
no_split), drawn at random and moved by up to +-`negatives.time_jitter_days` (kept inside the edit times'
range). Only neurons whose 783 root already existed before T qualify; such a root was not edited between
its creation and 783, so the segment at T is exactly the proofread neuron.

Splits: within each kind, the tasks are shuffled and the first `split_counts.train` go to train, then
val, then test.

    .venv/bin/python -m build.build_negatives            # build (synapse queries are cached)
    .venv/bin/python -m build.build_negatives --report   # also compare segment sizes with edit tasks
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from outer.config import load_config

NEGATIVE_VERSION = "2"  # 2: proofread neurons (1: the state just after an edit)
KINDS = ("no_merge", "no_split")


def iso(t: datetime) -> str:
    return t.isoformat(timespec="microseconds")


def edit_records(root: Path) -> list[dict[str, Any]]:
    splits = json.loads((root / "data" / "splits.json").read_text())
    ops = [op for k in ("train", "val", "test") for op in splits[k]]
    return [json.loads((root / "truth" / f"{op}.json").read_text()) for op in ops]


def draw_time(rng: np.random.Generator, times: list[datetime], jitter_days: float) -> datetime:
    """An edit time moved by up to +-jitter_days, strictly inside the edit times' range (redrawn, not
    clipped, so it never equals an edit time) and in whole milliseconds like the edit tasks' times
    (CAVE timestamps have millisecond precision; finer digits would mark a task as a negative)."""
    lo, hi = min(times), max(times)
    while True:
        t = times[int(rng.integers(len(times)))] + timedelta(seconds=float(rng.uniform(-1, 1)) * jitter_days * 86400)
        t = t.replace(microsecond=t.microsecond // 1000 * 1000)
        if lo < t < hi and t not in times:
            return t


class SynapseCounts:
    """root -> {partner: [n_out, n_in]} at a materialization, cached in a JSON file."""

    def __init__(self, client: Any, version: int, min_cleft: float, cache: Path):
        self.c, self.v, self.min_cleft, self.cache = client, version, min_cleft, cache
        self.d: dict[str, dict[str, list[int]]] = json.loads(cache.read_text()) if cache.exists() else {}
        self._dirty = 0

    def _side(self, col: str, root: int) -> collections.Counter:
        df = self.c.materialize.query_table(
            "synapses_nt_v1", materialization_version=self.v, filter_equal_dict={col: root},
            select_columns=["pre_pt_root_id", "post_pt_root_id", "cleft_score"], limit=2_000_000)
        df = df[(df.cleft_score > self.min_cleft) & (df.pre_pt_root_id != df.post_pt_root_id)]
        other = "post_pt_root_id" if col == "pre_pt_root_id" else "pre_pt_root_id"
        return collections.Counter(int(x) for x in df[other])

    def partners(self, root: int) -> dict[int, list[int]]:
        k = str(root)
        if k not in self.d:
            out, inc = self._side("pre_pt_root_id", root), self._side("post_pt_root_id", root)
            self.d[k] = {str(p): [out.get(p, 0), inc.get(p, 0)] for p in set(out) | set(inc)}
            self._dirty += 1
            if self._dirty >= 20:
                self.save()
        return {int(p): v for p, v in self.d[k].items()}

    def save(self) -> None:
        tmp = self.cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d))
        tmp.replace(self.cache)
        self._dirty = 0


def neuron_pool(client: Any, version: int, exclude_regex: str
                ) -> tuple[dict[int, dict[str, Any]], dict[str, int], set[int], dict[int, str]]:
    """root -> {created, super_class, position_nm_xyz} for proofread, annotated, non-glia neurons at `version`
    (CAVE positions in this datastack are in nm: voxel_resolution 1 x 1 x 1), plus the stats, the excluded
    (tagged) roots, and every root's super_class."""
    m, cg = client.materialize, client.chunkedgraph
    pr = m.query_table("proofread_neurons", materialization_version=version)
    sc = m.query_table("hierarchical_neuron_annotations", materialization_version=version,
                       filter_equal_dict={"classification_system": "super_class"})
    info = m.query_table("neuron_information_v2", materialization_version=version)
    excluded = {int(r) for r in info[info.tag.str.contains(exclude_regex, flags=re.I, na=False)].pt_root_id}
    super_class = {int(r): str(c) for r, c in zip(sc.pt_root_id, sc.cell_type, strict=True)}
    roots = np.asarray(pr.pt_root_id, dtype=np.int64)
    created: list[datetime] = []
    for i in range(0, len(roots), 10000):
        created += list(cg.get_root_timestamps(roots[i:i + 10000]))
    pool, stats = {}, collections.Counter(proofread=len(roots))
    for r, t, pos in zip(roots, created, pr.pt_position, strict=True):
        r = int(r)
        if r in excluded:
            stats["excluded_tag"] += 1
        elif r not in super_class:
            stats["no_super_class"] += 1
        else:
            pool[r] = {"created": t, "super_class": super_class[r], "position_nm_xyz": [int(x) for x in pos]}
    stats["pool"] = len(pool)
    return pool, dict(stats), excluded, super_class


def edit_neurons(cg: Any, edits: list[dict[str, Any]], t_check: datetime) -> dict[str, list[int]]:
    """op -> the roots at the check time that hold the edit's supervoxels."""
    out = {}
    for tr in edits:
        svs = {int(s) for e in tr.get("added_edges", []) for s in e}
        svs |= {int(s) for s in tr.get("source_supervoxels", []) + tr.get("sink_supervoxels", [])}
        svs |= {int(s) for s in tr.get("edited_supervoxels", [])}
        svs.discard(0)
        roots = cg.get_roots(np.asarray(sorted(svs), dtype=np.uint64), timestamp=t_check) if svs else []
        out[tr["operation_id"]] = sorted({int(r) for r in roots})
    return out


def build(cfg: Any, client: Any, rng: np.random.Generator, report: bool) -> dict[str, Any]:
    neg = cfg["negatives"]
    root = cfg.root
    version = int(neg["check_materialization"])
    cg = client.chunkedgraph
    t_check = client.materialize.get_timestamp(version)
    out_dir = root / "truth" / "negatives"
    out_dir.mkdir(parents=True, exist_ok=True)
    edits = edit_records(root)
    seg_source = edits[0]["segmentation_source"]
    times = {typ: sorted(datetime.fromisoformat(t["t_utc"]) for t in edits if t["operation_type"] == typ)
             for typ in ("merge", "split")}
    pool, pool_stats, excluded, super_class = neuron_pool(client, version, str(neg["exclude_tag_regex"]))
    print(f"pool: {pool_stats}", flush=True)
    held = edit_neurons(cg, edits, t_check)
    edit_roots = {int(r) for t in edits for r in t["pre_edit_roots"] + t["post_edit_roots"]}
    used: set[int] = {r for rs in held.values() for r in rs} | edit_roots
    by_created = sorted(pool, key=lambda r: pool[r]["created"])
    created_arr = np.array([pool[r]["created"].timestamp() for r in by_created])
    syn = SynapseCounts(client, version, float(neg["min_cleft_score"]), out_dir / "synapse_cache.json")
    min_syn = int(neg["min_synapses"])
    margin = timedelta(seconds=60)

    def candidates(t: datetime) -> np.ndarray:
        return np.asarray(by_created[: int(np.searchsorted(created_arr, (t - margin).timestamp()))], dtype=np.int64)

    def pick(t: datetime) -> int | None:
        c = candidates(t)
        for _ in range(50):
            if not len(c):
                return None
            r = int(c[rng.integers(len(c))])
            if r not in used:
                return r
        free = [int(r) for r in c if int(r) not in used]
        return int(rng.choice(free)) if free else None

    recs: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
    n_want = {"no_merge": int(neg["n_no_merge"]), "no_split": int(neg["n_no_split"])}
    tries = collections.Counter()
    # interleave the kinds so neither gets the pick of the early (scarce) neurons
    order = [k for i in range(max(n_want.values())) for k in KINDS if i < n_want[k]]
    for kind in order:
        typ = "merge" if kind == "no_merge" else "split"
        while True:
            tries[kind] += 1
            if tries[kind] > 50 * n_want[kind]:
                raise SystemExit(f"could not build enough {kind} tasks")
            t = draw_time(rng, times[typ], float(neg["time_jitter_days"]))
            a = pick(t)
            if a is None:
                continue
            if kind == "no_split":
                segs, syn_info = [a], None
            else:
                ok = [(b, v) for b, v in syn.partners(a).items()
                      if sum(v) >= min_syn and b in pool and b not in used and b != a
                      and pool[b]["created"] < t - margin]
                if not ok:
                    used.add(a)  # not tried again (keeps the build bounded; a later T might have found one)
                    tries["no_partner"] += 1
                    continue
                b, v = ok[int(rng.integers(len(ok)))]
                segs, syn_info = [a, b], {"a_to_b": int(v[0]), "b_to_a": int(v[1])}
            used.update(segs)
            rec = {"negative_version": NEGATIVE_VERSION, "kind": kind, "operation_type": "none",
                   "t_utc": iso(t), "time_drawn_from": f"{typ} edit times", "segmentation_source": seg_source,
                   "segments": [str(s) for s in segs], "check_materialization": version, "check_utc": iso(t_check),
                   "neurons": [{"root_at_check": str(s), "root_created_utc": iso(pool[s]["created"]),
                                "super_class": pool[s]["super_class"],
                                "position_nm_xyz": pool[s]["position_nm_xyz"]} for s in segs]}
            if syn_info:
                rec["synapses"] = syn_info
            recs[kind].append(rec)
            if len(recs[kind]) % 50 == 0:
                print(f"  {kind}: {len(recs[kind])}/{n_want[kind]}", flush=True)
            break
    syn.save()

    # checks at T and at the check time; then splits
    counts = neg["split_counts"]
    problems_total = 0
    for kind in KINDS:
        rs = recs[kind]
        rng.shuffle(rs)
        bounds = {"train": counts["train"], "val": counts["train"] + counts["val"],
                  "test": counts["train"] + counts["val"] + counts["test"]}
        if bounds["test"] != len(rs):
            raise SystemExit(f"split_counts add up to {bounds['test']}, but {kind} has {len(rs)} tasks")
        for i, rec in enumerate(rs):
            rec["negative_id"] = f"{kind}_{i:04d}"
            rec["split"] = "train" if i < bounds["train"] else "val" if i < bounds["val"] else "test"
            ids = [int(s) for s in rec["segments"]]
            t = datetime.fromisoformat(rec["t_utc"])
            p = []
            if not all(bool(x) for x in cg.is_latest_roots(ids, timestamp=t)):
                p.append("a segment is not current at T")
            if not all(bool(x) for x in cg.is_latest_roots(ids, timestamp=t_check)):
                p.append("a segment is not current at the check time")
            if any(c >= t for c in cg.get_root_timestamps(ids)):
                p.append("a segment was created after T")
            if set(ids) & (edit_roots | excluded) or any(i_ not in super_class for i_ in ids):
                p.append("an edit root, an excluded (glia / not a neuron) root, or a root without a super_class")
            rec["problems"] = p
            rec["eligible"] = not p
            problems_total += bool(p)

    summary: dict[str, Any] = {
        "negative_version": NEGATIVE_VERSION, "check_materialization": version, "check_utc": iso(t_check),
        "pool": pool_stats, "edit_neurons_excluded": len({r for rs in held.values() for r in rs}),
        "tries": dict(tries), "with_problems": problems_total,
        "kinds": {k: {"n": len(v), "by_split": dict(collections.Counter(r["split"] for r in v)),
                      "super_class": dict(collections.Counter(n["super_class"] for r in v for n in r["neurons"])),
                      "t_quantiles": [iso(q) for q in quantiles([datetime.fromisoformat(r["t_utc"]) for r in v])]}
                  for k, v in recs.items()},
        "edit_t_quantiles": {typ: [iso(q) for q in quantiles(ts)] for typ, ts in times.items()},
        "edit_tasks_glia": glia_check(edits, held, excluded, super_class),
    }
    nm = [r["synapses"]["a_to_b"] + r["synapses"]["b_to_a"] for r in recs["no_merge"]]
    summary["kinds"]["no_merge"]["synapses_quantiles"] = [int(x) for x in np.quantile(nm, [0, .25, .5, .75, 1])]
    if report:
        summary["size_l2"] = size_report(cg, edits, recs, rng)
    return {"records": recs, "summary": summary}


def quantiles(ts: list[datetime]) -> list[datetime]:
    s = sorted(ts)
    return [s[int(q * (len(s) - 1))] for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)]


def glia_check(edits: list[dict[str, Any]], held: dict[str, list[int]], excluded: set[int],
               super_class: dict[int, str]) -> dict[str, Any]:
    """Edit tasks whose edit sits on a root (at the check time) tagged glia / not a neuron, or on a root
    that has no neuron super_class (an unannotated fragment, or possibly glia)."""
    tagged = sorted(t["operation_id"] for t in edits if set(held[t["operation_id"]]) & excluded)
    no_class = sorted(t["operation_id"] for t in edits
                      if not any(r in super_class for r in held[t["operation_id"]]))
    by_type = collections.Counter(t["operation_type"] for t in edits if t["operation_id"] in set(no_class))
    return {"tagged_not_neuron": tagged, "n_tagged_not_neuron": len(tagged),
            "no_neuron_super_class": no_class, "n_no_neuron_super_class": len(no_class),
            "no_neuron_super_class_by_type": dict(by_type)}


def size_report(cg: Any, edits: list[dict[str, Any]], recs: dict[str, list[dict[str, Any]]],
                rng: np.random.Generator, n: int = 40) -> dict[str, Any]:
    """L2-chunk counts of the shown segments: edit tasks (pre-edit roots) vs negatives (a sample)."""
    def l2(r: str) -> int:
        return len(cg.get_leaves(int(r), stop_layer=2))

    def q(xs: list[int]) -> list[int]:
        return [int(x) for x in np.quantile(xs, [0.1, 0.25, 0.5, 0.75, 0.9])] if xs else []

    out: dict[str, Any] = {}
    for typ in ("merge", "split"):
        es = [t for t in edits if t["operation_type"] == typ]
        es = [es[i] for i in rng.choice(len(es), min(n, len(es)), replace=False)]
        sizes = [sorted(l2(r) for r in t["pre_edit_roots"]) for t in es]
        out[typ] = {"n": len(es), "smaller": q([s[0] for s in sizes]), "larger": q([s[-1] for s in sizes])}
    for kind, rs in recs.items():
        rs = [rs[i] for i in rng.choice(len(rs), min(n, len(rs)), replace=False)]
        sizes = [sorted(l2(s) for s in r["segments"]) for r in rs]
        out[kind] = {"n": len(rs), "smaller": q([s[0] for s in sizes]), "larger": q([s[-1] for s in sizes])}
    return out


def main(argv: list[str] | None = None) -> None:
    from caveclient import CAVEclient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="also compare segment sizes (L2 chunks) with edit tasks")
    args = ap.parse_args(argv)
    cfg = load_config()
    client = CAVEclient(cfg["flywire"]["datastack"])
    rng = np.random.default_rng(int(cfg["negatives"]["seed"]))
    res = build(cfg, client, rng, args.report)
    out_dir = cfg.root / "truth" / "negatives"
    new = {r["negative_id"] for rs in res["records"].values() for r in rs}
    for p in out_dir.glob("*.json"):  # older records (negative_version 1, named by operation id) go
        if p.stem not in new and p.name not in ("summary.json", "synapse_cache.json"):
            p.unlink()
    for rs in res["records"].values():
        for r in rs:
            (out_dir / f"{r['negative_id']}.json").write_text(json.dumps(r, indent=1) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(res["summary"], indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
