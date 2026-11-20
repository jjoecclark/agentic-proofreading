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
    neg = {"kind": "no_merge", "segments": ["720575940000000005", "720575940000000006"],
           "t_utc": "2021-05-01T00:00:00.000000+00:00", "segmentation_source": "graphene://x"}
    edit_roots = set(tr["pre_edit_roots"] + tr["post_edit_roots"])
    st = make_state("t_00000001", 1, neg, "precomputed://em")
    assert check_negative_state(st, neg, edit_roots, {"123456"}) == []
    for r in (tr["pre_edit_roots"][0], tr["post_edit_roots"][0]):
        bad = json.loads(json.dumps(st))
        bad["neuroglancer"]["layers"][1]["segments"][0] = r
        assert any("edit task" in p for p in check_negative_state(bad, neg, edit_roots, {"123456"}))
    bad = json.loads(json.dumps(st))
    bad["neuroglancer"]["layers"][0]["name"] = "123456"
    assert any("operation ID" in p for p in check_negative_state(bad, neg, edit_roots, {"123456"}))
    bad = json.loads(json.dumps(st))
    bad["timestamp_utc"] = tr["t_utc"]
    assert any("negative's time" in p for p in check_negative_state(bad, neg, edit_roots, {"123456"}))
    fine = {**neg, "t_utc": "2021-05-01T00:00:00.000123+00:00"}
    assert any("milliseconds" in p for p in check_negative_state(make_state("t_00000002", 1, fine, "precomputed://em"),
                                                                  fine, edit_roots, {"123456"}))
    one = {**neg, "kind": "no_split"}
    assert any("segments" in p for p in check_negative_state(st, one, edit_roots, {"123456"}))


def test_draw_time_stays_in_range():
    from datetime import UTC, datetime

    import numpy as np

    from build.build_negatives import draw_time

    ts = [datetime(2020, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC)]
    rng = np.random.default_rng(0)
    got = [draw_time(rng, ts, 14) for _ in range(500)]
    assert min(got) > ts[0] and max(got) < ts[1] and all(g.microsecond % 1000 == 0 for g in got)
    assert all(min(abs((g - t).total_seconds()) for t in ts) <= 14 * 86400 for g in got)


@pytest.mark.skipif(not HAVE_DATA, reason="tasks/ and truth/ not built")
def test_real_tasks_are_clean_and_match_splits():
    cfg = load_config()
    idx = json.loads((ROOT / "tasks" / "index.json").read_text())
    tmd = json.loads((ROOT / "truth" / "task_map.json").read_text())
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    op_of = {v["task_id"]: op for op, v in tmd["operations"].items()}
    neg_of = {v["task_id"]: nid for nid, v in tmd.get("negatives", {}).items()}
    assert set(idx.get("negative", [])) == set(neg_of)
    edit_roots: set[str] = set()
    for op in op_of.values():
        tr = json.loads((ROOT / "truth" / f"{op}.json").read_text())
        edit_roots |= set(tr["pre_edit_roots"] + tr["post_edit_roots"])
    seen_neurons: dict[str, str] = {}
    want = cfg["negatives"]["split_counts"]
    for split in ("train", "val", "test"):
        pos = [t for t in idx[split] if t in op_of]
        neg = [t for t in idx[split] if t in neg_of]
        assert len(pos) + len(neg) == len(idx[split])
        assert sorted(op_of[t] for t in pos) == sorted(splits[split])
        kinds = {"no_merge": 0, "no_split": 0}
        for t in neg:
            st = json.loads((ROOT / "tasks" / f"{t}.json").read_text())
            n = json.loads((ROOT / "truth" / "negatives" / f"{neg_of[t]}.json").read_text())
            assert n["eligible"] and n["split"] == split, t
            assert check_negative_state(st, n, edit_roots, set(op_of.values())) == [], t
            kinds[n["kind"]] += 1
            for s in n["segments"]:  # each neuron once, so never in two splits
                assert s not in seen_neurons, (t, s, seen_neurons.get(s))
                seen_neurons[s] = t
            assert all(x["super_class"] for x in n["neurons"])  # annotated neurons only (no glia)
            if n["kind"] == "no_merge":
                assert n["synapses"]["a_to_b"] + n["synapses"]["b_to_a"] >= cfg["negatives"]["min_synapses"]
        assert kinds == {"no_merge": want[split], "no_split": want[split]}, (split, kinds)
        for t in pos:
            st = json.loads((ROOT / "tasks" / f"{t}.json").read_text())
            tr = json.loads((ROOT / "truth" / f"{op_of[t]}.json").read_text())
            assert check_state(st, tr) == [], t
            assert tr["problems"] == [], t
            assert st["task_id"] == t and t.startswith("t_") and op_of[t] not in t


def test_edit_supervoxels_cover_every_side():
    from build.replace_edits import edit_supervoxels

    merge = {"added_edges": [["11", "12"], ["12", "13"]]}
    split = {"removed_edges": [[21, 22]], "source_ids": ["23"], "sink_ids": ["24", "0"]}
    assert edit_supervoxels(merge) == [11, 12, 13]
    assert edit_supervoxels(split) == [21, 22, 23, 24]  # 0 (background) dropped
