"""Live checks against real FlyWire data (read-only). Run with:  .venv/bin/python -m pytest --live tests/test_live.py

- Phase 2: the world service on 2 train tasks: EM and segmentation cutouts at the edit location, the L2
  graph, L2 positions, and mesh of a historical root, and refusal of a root that is not valid at T.
  Ids are the task's aliases (world API 2).
- Negative tasks: the world at T_post serves the edit's resulting segments and refuses the pre-edit ones.
- Scoring on real data: a test-only oracle that snaps to the edited supervoxels runs in the real sandbox
  against the real world service and is scored by outer/score.py. No Claude API calls are made.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from outer.config import ROOT, load_config, token_secret
from outer.evaluate import Evaluator, task_ids
from outer.score import CachedLookup, Truth
from services.aliases import roots_of
from services.httpserver import serve
from services.tokens import world_token
from services.world import server as W
from services.world.client import WorldClient, WorldRefused

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live():
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="fwlive", dir="/tmp"))
    backend = W.FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
    secret = token_secret(cfg)
    app = W.WorldService(backend, secret, log_path=tmp / "world.jsonl", session_dir=tmp / "sessions")
    servers = serve(W.make_handler(app), tmp / "w.sock")
    yield cfg, backend, secret, tmp
    for s in servers:
        s.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


def pick(cfg, kind, n=1):
    tr = Truth(cfg.root / "truth")
    return [t for t in task_ids(cfg, "train") if (tr(t).get("kind") or tr(t)["operation_type"]) == kind][:n]


def ali(secret, tid, roots):
    return [str(int(a)) for a in roots_of(secret, tid).alias([int(r) for r in roots])]


@pytest.mark.parametrize("typ", ["merge", "split"])
def test_world_live(live, typ):
    cfg, backend, secret, tmp = live
    tid = pick(cfg, typ)[0]
    tr = Truth(cfg.root / "truth")(tid)
    st = json.loads((cfg.root / "tasks" / f"{tid}.json").read_text())
    w = WorldClient(socket_path=str(tmp / "w.sock"), token=world_token(secret, tid, st["timestamp_utc"]))
    info = w.info()
    assert info["t_utc"].startswith(st["timestamp_utc"][:19])
    c = np.array(tr["edit_center_nm_xyz"])
    box = [(c - [1024, 1024, 200]).tolist(), (c + [1024, 1024, 200]).tolist()]
    em = w.em(box, (8, 8, 40))
    assert em.data_zyx.shape[0] in (10, 11) and em.data_zyx.shape[1:] in ((256, 256), (257, 257)), em.data_zyx.shape
    assert em.data_zyx.std() > 5, "EM cutout looks empty"
    seg = w.segmentation(box, (16, 16, 40))
    found = {str(int(x)) for x in np.unique(seg.data_zyx)}
    assert set(ali(secret, tid, tr["pre_edit_roots"])) <= found, (tr["pre_edit_roots"], len(found))
    post_only = set(tr["post_edit_roots"]) - set(tr["pre_edit_roots"])
    assert not set(ali(secret, tid, post_only)) & found, "post-edit root visible at T"
    assert not set(tr["pre_edit_roots"]) & found, "raw root ids visible"
    assert set(st["neuroglancer"]["layers"][1]["segments"]) == set(tr["pre_edit_roots"])
    root = ali(secret, tid, tr["pre_edit_roots"][:1])[0]
    g = w.l2_graph(root)
    assert len(g.l2_ids) >= 100 and len(g.edges) >= len(g.l2_ids) - 1
    pos, approx = w.l2_positions(g.l2_ids)
    lo, hi = np.array(info["bounds_nm_xyz"])
    assert np.all(pos >= lo) and np.all(pos <= hi) and approx.mean() < 0.2
    m = w.mesh(root)
    d = np.linalg.norm(m.vertices_nm_xyz - c, axis=1).min()
    assert d < 3000, f"nearest mesh vertex {d:.0f} nm from the edit center"
    # L2 positions agree with the mesh: each non-approximate position is near some mesh vertex
    sample = pos[~approx][:: max(1, (~approx).sum() // 50)]
    near = [np.linalg.norm(m.vertices_nm_xyz - p, axis=1).min() for p in sample]
    assert np.median(near) < 1000
    for post in ali(secret, tid, post_only):
        with pytest.raises(WorldRefused):
            w.l2_graph(post)


@pytest.mark.parametrize("kind", ["no_merge", "no_split"])
def test_world_live_negative(live, kind):
    cfg, backend, secret, tmp = live
    tid = pick(cfg, kind)[0]
    neg = Truth(cfg.root / "truth")(tid)
    src = json.loads((cfg.root / "truth" / f"{neg['source_operation_id']}.json").read_text())
    st = json.loads((cfg.root / "tasks" / f"{tid}.json").read_text())
    w = WorldClient(socket_path=str(tmp / "w.sock"), token=world_token(secret, tid, st["timestamp_utc"]))
    c = np.array(src["edit_center_nm_xyz"])
    box = [(c - [1024, 1024, 200]).tolist(), (c + [1024, 1024, 200]).tolist()]
    found = {str(int(x)) for x in np.unique(w.segmentation(box, (16, 16, 40)).data_zyx)}
    segs = ali(secret, tid, neg["segments"])
    assert set(segs) & found, "the task's segments are not at the source edit"
    gone = set(src["pre_edit_roots"]) - set(src["post_edit_roots"])
    assert not set(ali(secret, tid, gone)) & found, "a pre-edit root is visible at T_post"
    for s_ in segs:
        assert len(w.l2_graph(s_).l2_ids) >= 1 and len(w.mesh(s_).vertices_nm_xyz) > 0
    for r in ali(secret, tid, gone):
        with pytest.raises(WorldRefused):
            w.mesh(r)


def snap_answers(cfg, tids):
    tr = Truth(cfg.root / "truth")
    out = {}
    for t in tids:
        r = tr(t)
        if r.get("kind") in ("no_merge", "no_split"):
            out[t] = {"type": "none"}
            continue
        out[t] = {"type": r["operation_type"], "center": r["edit_center_nm_xyz"], "added_edges": r.get("added_edges", []),
                  "sources": r.get("source_supervoxels", []), "sinks": r.get("sink_supervoxels", [])}
    return out


@pytest.mark.skipif(not (ROOT / "outer/sandbox.sif").exists(), reason="needs the sandbox image")
def test_snap_oracle_on_real_train_tasks(live):
    from outer.sandbox import ApptainerRunner
    from services.llm import proxy as P
    from services.llm.ledger import Ledger
    from tests.fakes import FakeUpstream

    cfg, backend, secret, tmp = live
    n = int(os.environ.get("FW_LIVE_N", "20"))
    tids = task_ids(cfg, "train")[:n]
    h = tmp / "harnesses" / "snap_oracle"
    shutil.copytree(ROOT / "tests/harnesses/snap_oracle", h)
    (h / "answers.json").write_text(json.dumps(snap_answers(cfg, tids)))
    up = FakeUpstream()
    papp = P.LLMProxy(secret, Ledger(tmp / "ledger.jsonl"), cfg["prices_usd_per_mtok"], up.url, "fake-key", tmp)
    servers = serve(P.make_handler(papp), tmp / "l.sock")
    try:
        runner = ApptainerRunner(ROOT / "outer/sandbox.sif", tmp / "w.sock", tmp / "l.sock")
        ev = Evaluator(cfg, runner, CachedLookup(backend))
        ev.ledger_path, ev.log_root = tmp / "ledger.jsonl", tmp
        s = ev.evaluate("snap_oracle", "train", tids, parallel=8, harness_dir=h, run_root=tmp / "runs", progress=False)
    finally:
        for x in servers:
            x.shutdown()
        up.close()
    misses = {t: json.loads((tmp / "runs" / t / "score.json").read_text()) for t in tids}
    misses = {t: (v["kind"], v["miss_reason"], v["distance_nm"]) for t, v in misses.items() if not v["hit"]}
    print(f"\nsnap oracle on {len(tids)} real train tasks: {s['all']['hits']}/{s['all']['n']} "
          f"(merge {s['merge']['hits']}/{s['merge']['n']}, split {s['split']['hits']}/{s['split']['n']}, "
          f"no-edit {s['no_edit']['hits']}/{s['no_edit']['n']}); misses {misses}")
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "live_oracle_check.json").write_text(json.dumps({"summary": s, "misses": misses}, indent=1))
    assert s["all"]["hits"] >= 0.9 * len(tids), misses
