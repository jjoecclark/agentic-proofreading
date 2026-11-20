"""Build the cleaned task states tasks/<task_id>.json from the CSV + truth/ (PLAN.md Section 4.1).

- Task IDs are random ("t_" + 8 hex digits) and reveal nothing about the operation. The task-ID ↔
  operation map, with each task's shuffle seed, lives in truth/task_map.json and is reused on rebuild.
- Merges show both pre-edit segments; splits show the single pre-edit segment; order shuffled with
  the task seed. No navigation, camera position, annotations, title, or operation ID; nothing after T.
- Negative tasks (no edit needed; truth/negatives/<negative_id>.json, from build/build_negatives.py):
  proofread neurons at their time T, in the split their record names. Only eligible ones.
- tasks/index.json lists the task IDs of each split (no operation IDs), so the outer harness can find
  a split's tasks without reading truth/, and which of them are negatives.
- A harness never sees these raw segment ids: the outer harness hands it the task's aliases
  (services/aliases.py).

Checks: root counts match the type; no task file contains an operation ID or a coordinate; an edit
task holds no post-edit root and its time is before the edit; a negative task holds no root of any edit
task (pre or post) and no operation ID; only whitelisted keys appear.

    .venv/bin/python -m build.build_states
"""

from __future__ import annotations

import json
import random
import secrets
from datetime import datetime
from typing import Any

from build.build_negatives import NEGATIVE_VERSION
from build.make_splits import read_operations
from outer.config import load_config

STATE_VERSION = "1"
INDEX_VERSION = "3"     # 2: negatives; 3: proofread-neuron negatives (negative_version 2)
TASK_MAP_VERSION = "3"  # 2: negatives; 3: negatives keyed by negative_id

ALLOWED_KEYS = {"state_version", "task_id", "timestamp_utc", "neuroglancer", "layers", "type", "name",
                "source", "segments", "timestamp"}


def make_state(task_id: str, seed: int, truth: dict[str, Any], em_source: str) -> dict[str, Any]:
    """An edit task (truth/<op>.json: the pre-edit roots at T) or a negative (truth/negatives/<op>.json:
    the edit's resulting segments at T_post)."""
    segments = list(truth["segments"] if "kind" in truth else truth["pre_edit_roots"])
    random.Random(seed).shuffle(segments)
    t = datetime.fromisoformat(truth["t_utc"])
    return {
        "state_version": STATE_VERSION,
        "task_id": task_id,
        "timestamp_utc": t.isoformat(timespec="microseconds"),
        "neuroglancer": {"layers": [
            {"type": "image", "name": "em", "source": em_source},
            {"type": "segmentation", "name": "segments", "source": truth["segmentation_source"],
             "segments": segments, "timestamp": f"{t.timestamp():.6f}"},
        ]},
    }


def _walk(obj: Any, keys: set[str], values: list[Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            _walk(v, keys, values)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, keys, values)
    else:
        values.append(obj)


def check_state(state: dict[str, Any], truth: dict[str, Any]) -> list[str]:
    """Leak checks for one task file (empty list = clean)."""
    problems = []
    keys: set[str] = set()
    values: list[Any] = []
    _walk(state, keys, values)
    if keys - ALLOWED_KEYS:
        problems.append(f"unexpected keys {sorted(keys - ALLOWED_KEYS)}")
    strs = {str(v) for v in values}
    if truth["operation_id"] in strs:
        problems.append("contains the operation ID")
    post_only = set(truth["post_edit_roots"]) - set(truth["pre_edit_roots"])
    if post_only & strs:
        problems.append("contains a post-edit root")
    if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        problems.append("contains a number (task files hold only strings; no coordinates)")
    segs = state["neuroglancer"]["layers"][1]["segments"]
    n_expect = 2 if truth["operation_type"] == "merge" else 1
    if len(segs) != n_expect or set(segs) != set(truth["pre_edit_roots"]):
        problems.append(f"segments {segs} do not match the {truth['operation_type']} pre-edit roots")
    if datetime.fromisoformat(state["timestamp_utc"]) >= datetime.fromisoformat(truth["timestamp_utc"]):
        problems.append("task time is not before the edit")
    return problems


def check_negative_state(state: dict[str, Any], neg: dict[str, Any], edit_roots: set[str],
                         operation_ids: set[str]) -> list[str]:
    """Leak checks for a negative task file (neg: truth/negatives/<negative_id>.json; edit_roots: every
    pre- and post-edit root of the edit operations; operation_ids: theirs)."""
    problems = []
    keys: set[str] = set()
    values: list[Any] = []
    _walk(state, keys, values)
    if keys - ALLOWED_KEYS:
        problems.append(f"unexpected keys {sorted(keys - ALLOWED_KEYS)}")
    strs = {str(v) for v in values}
    if operation_ids & strs:
        problems.append("contains an operation ID")
    if edit_roots & strs:
        problems.append("contains a root of an edit task")
    if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        problems.append("contains a number (task files hold only strings; no coordinates)")
    segs = state["neuroglancer"]["layers"][1]["segments"]
    if len(segs) != (1 if neg["kind"] == "no_split" else 2) or set(segs) != set(neg["segments"]):
        problems.append(f"segments {segs} do not match the {neg['kind']} segments")
    if datetime.fromisoformat(state["timestamp_utc"]) != datetime.fromisoformat(neg["t_utc"]):
        problems.append("task time is not the negative's time")
    if datetime.fromisoformat(state["timestamp_utc"]).microsecond % 1000:
        problems.append("task time is finer than milliseconds (edit tasks' times are whole milliseconds)")
    return problems


def in_splits_of(splits: dict[str, Any]) -> list[str]:
    return [op for k in ("train", "val", "test") for op in splits[k]]


def _edit_roots(root: Any, op: str) -> list[str]:
    tr = json.loads((root / "truth" / f"{op}.json").read_text())
    return [str(r) for r in tr["pre_edit_roots"] + tr["post_edit_roots"]]


def new_task_id(used: set[str]) -> str:
    tid = "t_" + secrets.token_hex(4)
    while tid in used:
        tid = "t_" + secrets.token_hex(4)
    used.add(tid)
    return tid


def main() -> None:
    cfg = load_config()
    root = cfg.root
    splits = json.loads((root / "data" / "splits.json").read_text())
    ops = {r["operation_id"]: r for r in read_operations(cfg.path("data.operations_csv"))}
    map_path = root / "truth" / "task_map.json"
    old = json.loads(map_path.read_text()) if map_path.exists() else {}
    task_map: dict[str, Any] = old.get("operations", {})
    neg_map: dict[str, Any] = old.get("negatives", {})
    used = {v["task_id"] for v in task_map.values()} | {v["task_id"] for v in neg_map.values()}
    tasks_dir = root / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    index: dict[str, list[str]] = {}
    negatives: list[str] = []
    n_problems = 0
    for split in ("train", "val", "test"):
        index[split] = []
        for op in splits[split]:
            truth = json.loads((root / "truth" / f"{op}.json").read_text())
            if truth["problems"]:
                print(f"  warning: {op} has truth problems: {truth['problems']}")
            if op not in task_map:
                task_map[op] = {"task_id": new_task_id(used), "seed": secrets.randbits(32), "split": split}
            tm = task_map[op]
            tm["split"] = split
            state = make_state(tm["task_id"], tm["seed"], truth, cfg["flywire"]["em_source"])
            problems = check_state(state, truth)
            if set(state["neuroglancer"]["layers"][1]["segments"]) != set(ops[op]["before_root_ids"]):
                problems.append("segments do not match the CSV before_root_ids")
            if problems:
                n_problems += 1
                print(f"  {op}: {problems}")
                continue
            (tasks_dir / f"{tm['task_id']}.json").write_text(json.dumps(state, indent=1) + "\n")
            index[split].append(tm["task_id"])
    edit_roots = {r for op in in_splits_of(splits) for r in _edit_roots(root, op)}
    op_ids = set(ops) | set(in_splits_of(splits))
    live_negs: set[str] = set()
    for npath in sorted((root / "truth" / "negatives").glob("*.json")):
        neg = json.loads(npath.read_text())
        if neg.get("negative_version") != NEGATIVE_VERSION or not neg.get("eligible"):
            continue
        nid, split = neg["negative_id"], neg["split"]
        live_negs.add(nid)
        if nid not in neg_map:
            neg_map[nid] = {"task_id": new_task_id(used), "seed": secrets.randbits(32), "split": split}
        nm = neg_map[nid]
        nm["split"] = split
        state = make_state(nm["task_id"], nm["seed"], neg, cfg["flywire"]["em_source"])
        problems = check_negative_state(state, neg, edit_roots, op_ids)
        if problems:
            n_problems += 1
            print(f"  negative {nid}: {problems}")
            continue
        (tasks_dir / f"{nm['task_id']}.json").write_text(json.dumps(state, indent=1) + "\n")
        index[split].append(nm["task_id"])
        negatives.append(nm["task_id"])
    for split in ("train", "val", "test"):
        index[split].sort()
    for nid in [k for k in neg_map if k not in live_negs]:  # older negatives (by operation id) leave the map
        del neg_map[nid]
    in_splits = {op for k in ("train", "val", "test") for op in splits[k]}
    for op in [op for op in task_map if op not in in_splits]:  # replaced operations (build/replace_splits.py)
        del task_map[op]
    keep = {f"{v['task_id']}.json" for v in (*task_map.values(), *neg_map.values())} | {"index.json"}
    for p in tasks_dir.glob("*.json"):
        if p.name not in keep:
            p.unlink()
    map_path.write_text(json.dumps({"task_map_version": TASK_MAP_VERSION, "operations": task_map,
                                    "negatives": neg_map}, indent=1) + "\n")
    (tasks_dir / "index.json").write_text(json.dumps({"index_version": INDEX_VERSION, **index,
                                                      "negative": sorted(negatives)}, indent=1) + "\n")
    neg = set(negatives)
    print(f"wrote {sum(len(v) for v in index.values())} task files "
          f"({', '.join(f'{k} {len(v)} ({sum(t in neg for t in v)} negative)' for k, v in index.items())}); "
          f"{n_problems} with problems")


if __name__ == "__main__":
    main()
