"""Write data/splits.json: train / val / test operation IDs, stratified by type (PLAN.md Section 3.2).

The file is frozen once written. Changing the splits needs the user's approval (CLAUDE.md rule 7),
so overwriting requires --force. Version 2 (non-surviving splits replaced) is written by
build/replace_splits.py; do not regenerate it from scratch with this script.

    .venv/bin/python -m build.make_splits
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from outer.config import load_config, sha256_file

SPLITS_VERSION = "1"


def read_operations(csv_path: Path) -> list[dict[str, Any]]:
    """Rows of operations.csv. Root IDs stay decimal strings (64-bit safe)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "operation_id": str(int(r["operation_id"])),
                "operation_type": r["operation_type"],
                "timestamp_utc": r["timestamp_utc"],
                "before_root_ids": [str(x) for x in json.loads(r["before_root_ids"])],
                "after_root_ids": [str(x) for x in json.loads(r["after_root_ids"])],
            })
    return rows


def make_splits(rows: list[dict[str, Any]], sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    types = sorted({r["operation_type"] for r in rows})
    for t in types:
        ids = sorted((r["operation_id"] for r in rows if r["operation_type"] == t), key=int)
        rng.shuffle(ids)
        start = 0
        for name in ("train", "val", "test"):
            n = sizes[name] // len(types)
            out[name].extend(ids[start:start + n])
            start += n
        if start != len(ids):
            raise SystemExit(f"split sizes {sizes} do not use all {len(ids)} {t} operations")
    return {k: sorted(v, key=int) for k, v in out.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.json (ask the user first)")
    args = ap.parse_args(argv)
    cfg = load_config()
    out = cfg.root / "data" / "splits.json"
    if out.exists() and not args.force:
        print(f"{out} exists and is frozen; not changing it")
        return
    rows = read_operations(cfg.path("data.operations_csv"))
    s = cfg["splits"]
    splits = make_splits(rows, {k: int(s[k]) for k in ("train", "val", "test")}, int(s["seed"]))
    types = {r["operation_id"]: r["operation_type"] for r in rows}
    doc = {"splits_version": SPLITS_VERSION, "seed": int(s["seed"]),
           "operations_csv_sha256": sha256_file(cfg.path("data.operations_csv")),
           **splits,
           "counts": {k: {t: sum(types[o] == t for o in v) for t in ("merge", "split")} for k, v in splits.items()}}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}: " + ", ".join(f"{k} {len(v)} {doc['counts'][k]}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
