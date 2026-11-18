"""Build the cleaned task states tasks/<task_id>.json from the CSV + truth/ (PLAN.md Section 4.1).

- Task IDs are random ("t_" + 8 hex digits) and reveal nothing about the operation. The task-ID ↔
  operation map, with each task's shuffle seed, lives in truth/task_map.json and is reused on rebuild.
- Merges show both pre-edit segments; splits show the single pre-edit segment; order shuffled with
  the task seed. No navigation, camera position, annotations, title, or operation ID; nothing after T.
- Negative tasks (no edit needed; truth/negatives/, from build/build_negatives.py): the edit's result
  just after it (T_post), in the source operation's split. Only eligible ones (the edit survived; for
  `no_split`, nothing was split off later).
- tasks/index.json lists the task IDs of each split (no operation IDs), so the outer harness can find
  a split's tasks without reading truth/, and which of them are negatives.
- A harness never sees these raw segment ids: the outer harness hands it the task's aliases
  (services/aliases.py).

Checks: root counts match the type; no task file contains an operation ID or a coordinate; an edit
task holds no post-edit root and its time is before the edit; a negative task holds no pre-edit root
and its time is after the edit; only whitelisted keys appear.

    .venv/bin/python -m build.build_states
"""

from __future__ import annotations

import json
import random
import secrets
from datetime import datetime
from typing import Any

from build.make_splits import read_operations
from outer.config import load_config

STATE_VERSION = "1"
INDEX_VERSION = "2"     # 2: negatives
TASK_MAP_VERSION = "2"  # 2: negatives

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


def check_negative_state(state: dict[str, Any], neg: dict[str, Any], source: dict[str, Any]) -> list[str]:
    """Leak checks for a negative task file (neg: truth/negatives/<op>.json, source: truth/<op>.json)."""
    problems = []
    keys: set[str] = set()
    values: list[Any] = []
    _walk(state, keys, values)
    if keys - ALLOWED_KEYS:
        problems.append(f"unexpected keys {sorted(keys - ALLOWED_KEYS)}")
    strs = {str(v) for v in values}
    if source["operation_id"] in strs:
        problems.append("contains the operation ID")
    if (set(source["pre_edit_roots"]) - set(source["post_edit_roots"])) & strs:
        problems.append("contains a pre-edit root")
    if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        problems.append("contains a number (task files hold only strings; no coordinates)")
    segs = state["neuroglancer"]["layers"][1]["segments"]
    if len(segs) != (1 if neg["kind"] == "no_split" else 2) or set(segs) != set(neg["segments"]):
        problems.append(f"segments {segs} do not match the {neg['kind']} segments")
    if datetime.fromisoformat(state["timestamp_utc"]) <= datetime.fromisoformat(source["timestamp_utc"]):
        problems.append("task time is not after the edit")
    return problems


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
            npath = root / "truth" / "negatives" / f"{op}.json"
            neg = json.loads(npath.read_text()) if npath.exists() else None
            if not neg or not neg["eligible"]:
                neg_map.pop(op, None)
                continue
            if op not in neg_map:
                neg_map[op] = {"task_id": new_task_id(used), "seed": secrets.randbits(32), "split": split}
            nm = neg_map[op]
            nm["split"] = split
            state = make_state(nm["task_id"], nm["seed"], neg, cfg["flywire"]["em_source"])
            problems = check_negative_state(state, neg, truth)
            if problems:
                n_problems += 1
                print(f"  negative of {op}: {problems}")
                continue
            (tasks_dir / f"{nm['task_id']}.json").write_text(json.dumps(state, indent=1) + "\n")
            index[split].append(nm["task_id"])
            negatives.append(nm["task_id"])
        index[split].sort()
    in_splits = {op for k in ("train", "val", "test") for op in splits[k]}
    for m in (task_map, neg_map):  # operations replaced in the splits (build/replace_splits.py) leave the map
        for op in [op for op in m if op not in in_splits]:
            del m[op]
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
