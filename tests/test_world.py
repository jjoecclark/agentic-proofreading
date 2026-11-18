"""Phase 2: the world service against a fake backend (time binding and refusals)."""

from __future__ import annotations

import http.client
import json
import socket
from datetime import datetime

import numpy as np
import pytest

from services.aliases import roots_of
from services.tokens import world_token
from services.world.client import WorldClient, WorldError, WorldRefused
from tests.fakes import FakeBackend, strip_time


def wc(env, task_id: str) -> WorldClient:
    tr = env.truths[task_id]
    return WorldClient(socket_path=str(env.world_socket), token=world_token(env.secret, task_id, tr["t_utc"]))


def ali(env, task_id: str, roots) -> set[str]:
    """The ids a task sees for these real roots (world API 2: per-task aliases)."""
    return {str(int(a)) for a in roots_of(env.secret, task_id).alias([int(r) for r in roots])}


def raw(env, method: str, path: str, body: dict | None, token: str | None):
    class C(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX)
            self.sock.connect(str(env.world_socket))
    c = C("localhost")
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    c.request(method, path, body=json.dumps(body or {}), headers=headers)
    r = c.getresponse()
    return r.status, r.read()


def test_info_is_bound_to_T(fwenv):
    w = wc(fwenv, "t_fake0000")
    info = w.info()
    assert datetime.fromisoformat(info["t_utc"]) == datetime.fromisoformat(fwenv.truths["t_fake0000"]["t_utc"])
    assert [16.0, 16.0, 40.0] in [lv["resolution_nm_xyz"] for lv in info["layers"]["em"]["levels"]]


def test_roots_and_segmentation_are_as_of_T(fwenv):
    tr = fwenv.truths["t_fake0000"]  # a merge: at T the two pre-edit roots are still separate
    w = wc(fwenv, "t_fake0000")
    c = np.array(tr["edit_center_nm_xyz"])
    box = [(c - [800, 800, 200]).tolist(), (c + [800, 800, 200]).tolist()]
    seg = w.segmentation(box, (16, 16, 40))
    ids = {str(int(x)) for x in np.unique(seg.data_zyx)}
    assert ids == ali(fwenv, "t_fake0000", tr["pre_edit_roots"])  # not the merged root
    sv = w.supervoxels(box, (16, 16, 40))
    assert sv.data_zyx.shape == seg.data_zyx.shape == (10, 100, 100)
    r = w.roots(np.unique(sv.data_zyx))
    assert {str(int(x)) for x in r} == ali(fwenv, "t_fake0000", tr["pre_edit_roots"])
    # the same supervoxels seen from a task whose T is after this edit map to the merged root
    later = WorldClient(socket_path=str(fwenv.world_socket),
                        token=world_token(fwenv.secret, "t_later", strip_time(3).isoformat()))
    assert {str(int(x)) for x in later.roots(np.unique(sv.data_zyx))} == ali(fwenv, "t_later", tr["post_edit_roots"])
    # index helpers: the voxel at the edit center belongs to the right-hand root
    z, y, x = seg.index_of(c + [100, 0, 0])
    assert {str(int(seg.data_zyx[z, y, x]))} == ali(fwenv, "t_fake0000", tr["pre_edit_roots"][1:])
    em = w.em(box, (32, 32, 40))
    assert em.data_zyx.dtype == np.uint8 and em.data_zyx.shape == (10, 50, 50)


def test_l2_graph_positions_mesh_for_valid_root(fwenv):
    tr = fwenv.truths["t_fake0001"]  # a split: the single pre-edit root
    w = wc(fwenv, "t_fake0001")
    (seg_id,) = ali(fwenv, "t_fake0001", tr["pre_edit_roots"])
    g = w.l2_graph(seg_id)
    assert len(g.l2_ids) > 100 and g.edges.shape[1] == 2
    pos, approx = w.l2_positions(g.l2_ids[:50])
    assert pos.shape == (50, 3) and not approx.any()
    m = w.mesh(seg_id)
    assert m.vertices_nm_xyz.shape[1] == 3 and m.faces.shape[1] == 3


def test_refuses_roots_not_valid_at_T(fwenv):
    tr = fwenv.truths["t_fake0001"]
    w = wc(fwenv, "t_fake0001")
    for post_root in ali(fwenv, "t_fake0001", tr["post_edit_roots"]):  # created by the edit, after T
        with pytest.raises(WorldRefused):
            w.l2_graph(post_root)
        with pytest.raises(WorldRefused):
            w.mesh(post_root)
    with pytest.raises(WorldRefused):
        w.mesh(12345)  # not a root at all


def test_ids_are_per_task_aliases(fwenv):
    """Raw root and L2 ids carry their creation time; each task sees its own opaque ids instead."""
    tr = fwenv.truths["t_fake0001"]
    w = wc(fwenv, "t_fake0001")
    with pytest.raises(WorldRefused):  # the raw id of a root that is valid at T means nothing here
        w.mesh(tr["pre_edit_roots"][0])
    (seg_id,) = ali(fwenv, "t_fake0001", tr["pre_edit_roots"])
    assert seg_id != ali(fwenv, "t_fake0003", tr["pre_edit_roots"]).pop()
    g = w.l2_graph(seg_id)
    real_l2, _ = fwenv.backend.l2_graph(int(tr["pre_edit_roots"][0]))
    assert not set(map(int, g.l2_ids)) & set(map(int, real_l2))
    assert set(map(int, g.edges.ravel())) <= set(map(int, g.l2_ids))
    pos, _ = w.l2_positions(g.l2_ids[:5])
    assert np.allclose(pos, fwenv.backend.l2_positions(np.asarray(real_l2[:5], dtype=np.uint64))[0])


def test_l2_positions_only_for_l2_ids_from_l2_graph(fwenv):
    w = wc(fwenv, "t_fake0002")
    fake_l2 = (1 << 40) + 77
    with pytest.raises(WorldRefused):
        w.l2_positions([fake_l2])


@pytest.mark.parametrize("path", ["/v1/lineage_graph", "/v1/operation_details", "/v1/latest_roots",
                                  "/v1/root_timestamps", "/v1/get_roots_at", "/v1/merge", "/v1/split",
                                  "/segmentation/api/v1/table/x/merge", "/v1/past_id_mapping"])
def test_refuses_history_and_writes(fwenv, path):
    tok = world_token(fwenv.secret, "t_fake0000", fwenv.truths["t_fake0000"]["t_utc"])
    status, body = raw(fwenv, "POST", path, {}, tok)
    assert status == 403 and json.loads(body)["error"]["type"] == "refused"


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "GET"])
def test_refuses_non_post(fwenv, method):
    tok = world_token(fwenv.secret, "t_fake0000", fwenv.truths["t_fake0000"]["t_utc"])
    status, _ = raw(fwenv, method, "/v1/roots", {"supervoxel_ids": ["1"]}, tok)
    assert status == 403


def test_refuses_other_times(fwenv):
    tok = world_token(fwenv.secret, "t_fake0000", fwenv.truths["t_fake0000"]["t_utc"])
    for body in ({"supervoxel_ids": ["1"], "timestamp": "2030-01-01T00:00:00"},
                 {"supervoxel_ids": ["1"], "options": {"as_of": 0}},
                 {"root_id": "1", "latest": True}):
        status, resp = raw(fwenv, "POST", "/v1/roots", body, tok)
        assert status == 403, body
        assert "time" in json.loads(resp)["error"]["message"] or "T" in json.loads(resp)["error"]["message"]


def test_rejects_missing_or_forged_tokens(fwenv):
    assert raw(fwenv, "POST", "/v1/info", {}, None)[0] == 401
    forged = world_token(b"not-the-secret", "t_fake0000", "2030-01-01T00:00:00+00:00")
    assert raw(fwenv, "POST", "/v1/info", {}, forged)[0] == 401


def test_size_limits_and_bad_args(fwenv):
    w = wc(fwenv, "t_fake0000")
    with pytest.raises(WorldError) as ei:
        w.em([[0, 0, 0], [100000, 100000, 10000]], (16, 16, 40))
    assert ei.value.status == 400
    with pytest.raises(WorldError):
        w.em([[0, 0, 0], [100, 100, 100]], (5, 5, 5))  # not an available level


def test_requests_are_logged_with_task(fwenv):
    w = wc(fwenv, "t_fake0003")
    w.info()
    log = [json.loads(x) for x in (fwenv.root / "runs/world_requests.jsonl").read_text().splitlines()]
    assert any(r.get("task_id") == "t_fake0003" and r["endpoint"] == "info" for r in log)


def test_fake_backend_matches_truth():
    b = FakeBackend(2)
    from tests.fakes import fake_truths

    for tr in fake_truths(b).values():
        t = datetime.fromisoformat(tr["t_utc"])
        sv = b.supervoxels_at(np.array(tr["sources_nm_xyz"] + tr["sinks_nm_xyz"]))
        assert {str(int(x)) for x in b.roots(sv, t)} == set(tr["pre_edit_roots"])
