"""Phase 0-1 checks: splits, task-file leak checks, and (if built) the real tasks/ and truth/."""

from __future__ import annotations

import json

import pytest

from build.build_states import check_negative_state, check_state, make_state
from build.make_splits import make_splits, read_operations
from outer.config import ROOT, load_config, sha256_file

HAVE_DATA = (ROOT / "tasks" / "index.json").exists() and (ROOT / "truth" / "task_map.json").exists()


def test_operations_csv_pinned():
    cfg = load_config()  # checks the sha256 prefix
    assert sha256_file(cfg.path("data.operations_csv")).startswith(cfg["data"]["sha256_prefix"])
    rows = read_operations(cfg.path("data.operations_csv"))
    assert len(rows) == 200 and sum(r["operation_type"] == "merge" for r in rows) == 100
    assert all(isinstance(x, str) and len(x) >= 17 for r in rows for x in r["before_root_ids"])


def test_make_splits_stratified():
    rows = [{"operation_id": str(i), "operation_type": "merge" if i % 2 else "split"} for i in range(200)]
    s = make_splits(rows, {"train": 100, "val": 50, "test": 50}, 0)
    assert [len(s[k]) for k in ("train", "val", "test")] == [100, 50, 50]
    assert not (set(s["train"]) & set(s["val"])) and not (set(s["val"]) & set(s["test"]))
    for k, n in (("train", 50), ("val", 25), ("test", 25)):
        assert sum(int(x) % 2 for x in s[k]) == n
    assert make_splits(rows, {"train": 100, "val": 50, "test": 50}, 0) == s


def _truth():
    return {"operation_id": "123456", "operation_type": "merge", "t_utc": "2022-01-01T00:00:00.000000+00:00",
            "timestamp_utc": "2022-01-01T00:00:00.001000+00:00", "pre_edit_roots": ["720575940000000001", "720575940000000002"],
            "post_edit_roots": ["720575940000000003"], "segmentation_source": "graphene://x"}


def test_check_state_catches_leaks():
    tr = _truth()
    st = make_state("t_00000000", 1, tr, "precomputed://em")
    assert check_state(st, tr) == []
    assert "navigation" not in json.dumps(st)
    bad = json.loads(json.dumps(st))
    bad["title"] = "op 123456"
    assert check_state(bad, tr)
    bad = json.loads(json.dumps(st))
    bad["neuroglancer"]["layers"][1]["segments"].append("720575940000000003")
    assert any("post-edit" in p for p in check_state(bad, tr))
    bad = json.loads(json.dumps(st))
    bad["neuroglancer"]["layers"][0]["name"] = "123456"
    assert any("operation ID" in p for p in check_state(bad, tr))
    bad = json.loads(json.dumps(st))
    bad["neuroglancer"]["layers"][0]["source"] = [1.0, 2.0, 3.0]
    assert any("number" in p for p in check_state(bad, tr))


def test_check_negative_state_catches_leaks():
    tr = _truth()
    neg = {"kind": "no_merge", "segments": ["720575940000000003", "720575940000000004"],
           "t_utc": "2022-01-01T00:00:00.002000+00:00", "segmentation_source": "graphene://x"}
    st = make_state("t_00000001", 1, neg, "precomputed://em")
    assert check_negative_state(st, neg, tr) == []
    bad = json.loads(json.dumps(st))
    bad["neuroglancer"]["layers"][1]["segments"][0] = tr["pre_edit_roots"][0]
    assert any("pre-edit" in p for p in check_negative_state(bad, neg, tr))
    bad = json.loads(json.dumps(st))
    bad["timestamp_utc"] = tr["t_utc"]
    assert any("after the edit" in p for p in check_negative_state(bad, neg, tr))


@pytest.mark.skipif(not HAVE_DATA, reason="tasks/ and truth/ not built")
def test_real_tasks_are_clean_and_match_splits():
    idx = json.loads((ROOT / "tasks" / "index.json").read_text())
    tmd = json.loads((ROOT / "truth" / "task_map.json").read_text())
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    op_of = {v["task_id"]: op for op, v in tmd["operations"].items()}
    neg_of = {v["task_id"]: op for op, v in tmd.get("negatives", {}).items()}
    assert set(idx.get("negative", [])) == set(neg_of)
    for split in ("train", "val", "test"):
        pos = [t for t in idx[split] if t in op_of]
        neg = [t for t in idx[split] if t in neg_of]
        assert len(pos) + len(neg) == len(idx[split])
        assert sorted(op_of[t] for t in pos) == sorted(splits[split])
        assert all(neg_of[t] in splits[split] for t in neg)  # a negative stays in its operation's split
        for t in neg:
            st = json.loads((ROOT / "tasks" / f"{t}.json").read_text())
            n = json.loads((ROOT / "truth" / "negatives" / f"{neg_of[t]}.json").read_text())
            src = json.loads((ROOT / "truth" / f"{neg_of[t]}.json").read_text())
            assert n["eligible"] and check_negative_state(st, n, src) == [], t
        for t in pos:
            st = json.loads((ROOT / "tasks" / f"{t}.json").read_text())
            tr = json.loads((ROOT / "truth" / f"{op_of[t]}.json").read_text())
            assert check_state(st, tr) == [], t
            assert tr["problems"] == [], t
            assert st["task_id"] == t and t.startswith("t_") and op_of[t] not in t
