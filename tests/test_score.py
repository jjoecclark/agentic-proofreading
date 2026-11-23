"""Scoring rules (PLAN.md Section 5) on the fake world."""

from __future__ import annotations

import numpy as np
import pytest

from outer import score as S
from tests.fakes import FakeBackend, fake_truths

R = 2000.0


@pytest.fixture(scope="module")
def world():
    b = FakeBackend(2)
    return b, fake_truths(b)


def sc(dec, tr, b):
    d, why = S.validate_decision(dec)
    assert why is None, why
    return S.score_decision(d, tr, b, R)


def merge(p, q):
    return {"decision_version": "1", "action": "merge", "points_nm_xyz": [list(p), list(q)]}


def split(src, snk):
    return {"decision_version": "1", "action": "split", "sources_nm_xyz": [list(x) for x in src],
            "sinks_nm_xyz": [list(x) for x in snk]}


def test_merge_hit_either_order(world):
    b, trs = world
    tr = trs["t_fake0000"]
    c = np.array(tr["edit_center_nm_xyz"])
    for p, q in ((c - [300, 0, 0], c + [300, 0, 0]), (c + [300, 0, 0], c - [300, 0, 0])):
        r = sc(merge(p, q), tr, b)
        assert r["hit"] and r["distance_nm"] == pytest.approx(0.0)


def test_merge_misses(world):
    b, trs = world
    tr = trs["t_fake0000"]
    c = np.array(tr["edit_center_nm_xyz"])
    assert sc(merge(c - [300, 0, 0], c - [600, 0, 0]), tr, b)["miss_reason"] == S.OFF_SEGMENTS  # both on one root
    far = sc(merge(c + [-300, 2500, 0], c + [300, 2500, 0]), tr, b)
    assert far["miss_reason"] == S.TOO_FAR and far["distance_nm"] == pytest.approx(2500)
    # one point near, one point 2.5 um away: both must be within R
    assert sc(merge(c - [300, 0, 0], c + [2600, 0, 0]), tr, b)["miss_reason"] == S.TOO_FAR
    three = {"decision_version": "1", "action": "merge", "points_nm_xyz": [list(c)] * 3}
    assert sc(three, tr, b)["miss_reason"] == S.WRONG_POINT_COUNT
    assert sc(split([c - [300, 0, 0]], [c + [300, 0, 0]]), tr, b)["miss_reason"] == S.WRONG_ACTION
    assert sc({"decision_version": "1", "action": "none"}, tr, b)["miss_reason"] == S.NONE_ACTION


def test_split_hit_and_misses(world):
    b, trs = world
    tr = trs["t_fake0001"]
    c = np.array(tr["edit_center_nm_xyz"])
    hit = sc(split([c - [300, 0, 0], c - [500, 100, 0]], [c + [300, 0, 0]]), tr, b)
    assert hit["hit"] and hit["distance_nm"] == pytest.approx(0.0)
    # sources and sinks on the same side after the edit
    assert sc(split([c - [300, 0, 0]], [c - [600, 0, 0]]), tr, b)["miss_reason"] == S.NOT_SEPARATED
    # one source on the wrong side
    assert sc(split([c - [300, 0, 0], c + [400, 0, 0]], [c + [300, 0, 0]]), tr, b)["miss_reason"] == S.NOT_SEPARATED
    # closest pair midpoint too far
    far = sc(split([c + [-300, 3000, 0]], [c + [300, 3000, 0]]), tr, b)
    assert far["miss_reason"] == S.TOO_FAR
    # a point off the segment (outside the volume -> background)
    assert sc(split([c - [300, 0, 0]], [[-5000, 0, 0]]), tr, b)["miss_reason"] == S.OFF_SEGMENTS
    assert sc(merge(c - [300, 0, 0], c + [300, 0, 0]), tr, b)["miss_reason"] == S.WRONG_ACTION


@pytest.mark.parametrize("bad", [None, [], {"action": "merge"}, {"decision_version": "1", "action": "jump"},
                                 {"decision_version": "1", "action": "merge", "points_nm_xyz": [[1, 2]]},
                                 {"decision_version": "1", "action": "split", "sources_nm_xyz": [[1, 2, 3]]},
                                 {"decision_version": "1", "action": "merge", "points_nm_xyz": [[1, 2, float("nan")]] * 2}])
def test_invalid_decisions(bad):
    d, why = S.validate_decision(bad)
    assert d is None and why == S.INVALID_SCHEMA


def test_wilson_and_summary():
    lo, hi = S.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert S.wilson(0, 0) == (None, None)
    scores = [{"hit": True, "operation_type": "merge", "distance_nm": 100.0, "usd": 0.5, "wall_s": 10, "miss_reason": None},
              {"hit": False, "operation_type": "split", "distance_nm": 3000.0, "usd": 0.7, "wall_s": 20,
               "miss_reason": "too_far"},
              {"hit": False, "operation_type": "split", "distance_nm": None, "usd": 0.0, "wall_s": 5, "miss_reason": "crash"}]
    s = S.summarize(scores)
    assert s["all"]["hits"] == 1 and s["merge"]["hit_rate"] == 1.0 and s["split"]["n"] == 2
    assert s["all"]["median_distance_nm"] == 1550.0 and s["usd_per_task"] == pytest.approx(0.4)
    assert s["miss_reasons"] == {"too_far": 1, "crash": 1}


def test_paired_gain():
    a = {"t1": {"hit": True}, "t2": {"hit": True}, "t3": {"hit": False}}
    b = {"t1": {"hit": False}, "t2": {"hit": True}, "t3": {"hit": True}}
    g = S.paired_gain(a, b)
    assert {k: g[k] for k in ("n", "new_hits", "best_hits", "gain", "new_only", "best_only")} == \
        {"n": 3, "new_hits": 2, "best_hits": 2, "gain": 0, "new_only": 1, "best_only": 1}


def _s(kind, hit):
    return {"hit": hit, "kind": kind, "operation_type": "none" if kind.startswith("no_") else kind,
            "distance_nm": None, "usd": 0, "miss_reason": None if hit else "x"}


def test_kind_score_weighs_each_kind_the_same():
    # 'always none' on 1 merge, 1 split, 4 no_merge, 4 no_split: 8/10 hits, but score 50%
    always_none = [_s("merge", False), _s("split", False)] + [_s("no_merge", True)] * 4 + [_s("no_split", True)] * 4
    assert S.kind_score(always_none) == pytest.approx(0.5)
    s = S.summarize(always_none)
    assert s["score"] == pytest.approx(0.5) and s["all"]["hits"] == 8 and s["score_version"] == "4"
    # merge 1/2, no_split 1/4, other kinds absent: mean over present kinds only
    assert S.kind_score([_s("merge", True), _s("merge", False)] + [_s("no_split", True)] + [_s("no_split", False)] * 3) \
        == pytest.approx((0.5 + 0.25) / 2)
    assert S.kind_score([]) is None


def test_paired_gain_score_pp():
    new = {"a": _s("merge", True), "b": _s("no_merge", True), "c": _s("no_merge", True), "x": _s("split", True)}
    best = {"a": _s("merge", False), "b": _s("no_merge", True), "c": _s("no_merge", False)}
    g = S.paired_gain(new, best)  # common tasks a, b, c only
    assert g["n"] == 3 and g["new_score"] == pytest.approx(1.0) and g["best_score"] == pytest.approx(0.25)
    assert g["score_gain_pp"] == pytest.approx(75.0)


def test_negative_tasks(world):
    """No-edit tasks: only `none` hits; any edit, even a perfect one for the source edit, is a miss."""
    from tests.fakes import fake_negatives

    b, trs = world
    negs = fake_negatives(trs)
    for nid, neg in negs.items():
        src = trs[nid.replace("t_fneg", "t_fake")]
        c = np.array(src["edit_center_nm_xyz"])
        r = sc({"decision_version": "1", "action": "none"}, neg, b)
        assert r["hit"] and r["miss_reason"] is None
        if neg["kind"] == "no_merge":
            edit = merge(c - [300, 0, 0], c + [300, 0, 0])
        else:
            edit = split([c - [150, 0, 0]], [c + [150, 0, 0]])
        r = sc(edit, neg, b)
        assert not r["hit"] and r["miss_reason"] == S.EDIT_NOT_NEEDED
    s = S.summarize([{"hit": True, "kind": "no_split", "operation_type": "none", "distance_nm": None, "usd": 0,
                      "miss_reason": None},
                     {"hit": False, "kind": "no_merge", "operation_type": "none", "distance_nm": None, "usd": 0,
                      "miss_reason": "edit_not_needed"},
                     {"hit": True, "operation_type": "merge", "distance_nm": 10.0, "usd": 0, "miss_reason": None}])
    assert (s["all"]["hits"], s["no_edit"]["n"], s["no_edit"]["hits"], s["edit"]["n"], s["merge"]["hits"]) == (2, 2, 1, 1, 1)
