"""Phases 5-6 dry run with no API spend: the real Claude Agent SDK + Claude Code CLI run in the sandbox
against a scripted fake upstream that writes a harness, calls score_train, and stops. Then the loop
evaluates it, builds feedback, runs a revise round, and applies the keep rule."""

from __future__ import annotations

import json

import pytest

from outer.evaluate import Evaluator
from outer.sandbox import ApptainerRunner
from outer.score import CachedLookup
from tests.conftest import APPTAINER_IMAGE, HARNESS_DIR, close_env, make_env, needs_apptainer

TRIVIAL = (HARNESS_DIR / "trivial" / "harness.py").read_text()


def tool_names(body):
    return {t.get("name") for t in body.get("tools") or []}


def n_tool_results(body):
    return sum(1 for m in body["messages"] if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result")


def scripted_meta(body):
    """Claude Code main-loop requests carry the Write/Bash tools; everything else gets plain text."""
    if not {"Write", "Bash"} <= tool_names(body):
        return "ok"
    k = n_tool_results(body)
    revising = "round 1" in json.dumps(body["messages"][0])
    if revising:  # round 1: read the feedback README, then stop without changes
        if k == 0:
            return {"content": [{"type": "tool_use", "id": "toolu_r1", "name": "Bash",
                                 "input": {"command": "cat /work/feedback/README.md; ls /work/feedback/train | head",
                                           "description": "read feedback"}}], "stop_reason": "tool_use"}
        return "No change this round."
    steps = [
        ("Write", {"file_path": "/work/harness/harness.py", "content": TRIVIAL}),
        ("Write", {"file_path": "/work/harness/NOTES.md", "content": "# h000\nnearest L2 pair / L2 centroid.\n"}),
        ("Bash", {"command": "score_train --all-examples", "timeout": 600000, "description": "probe"}),
    ]
    if k < len(steps):
        name, inp = steps[k]
        return {"content": [{"type": "text", "text": f"step {k}"},
                            {"type": "tool_use", "id": f"toolu_{k:02d}", "name": name, "input": inp}],
                "stop_reason": "tool_use"}
    return "Built h000."


@pytest.fixture
def meta_env():
    env, servers = make_env(n_strips=4, reply=scripted_meta)
    env.cfg["limits"]["meta_wall_s"] = 900
    env.cfg["budget_usd"]["total"] = 1000.0
    env.cfg["loop"].update(max_rounds=2, accept_min_gain_pp=1.0, plateau_rounds=3)
    yield env
    close_env(env, servers)


def test_loop_adopts_h000_from_outer_create():
    """After `outer.create` wrote h000, the loop evaluates it as round 0 instead of starting round 0 again."""
    from outer.loop import load_state, round_estimate, run_rounds
    from outer.sandbox import LocalRunner
    from tests.conftest import harness_copy

    env, servers = make_env(n_strips=4)
    try:
        env.cfg["budget_usd"]["total"] = 10.0  # far below a fresh round's worst case ($50 + 4 tasks x $2)
        env.cfg["loop"].update(max_rounds=1)
        harness_copy("trivial", env.root).rename(env.root / "harnesses/h000")
        sdir = env.root / "runs/meta/round_000"
        (sdir / "probe/results/170000_aaaaaa").mkdir(parents=True)
        (sdir / "probe/results/170000_aaaaaa/result.json").write_text(json.dumps({"tasks": [{"usd": 0.01}] * 4}))
        (sdir / "session.json").write_text(json.dumps({"session": "round_000", "round": 0, "harness": "h000", "ok": True,
                                                       "meta_usd": 2.5, "probe_usd": 0.04, "meta_requests": 9,
                                                       "probe_requests": 4, "wall_s": 600.0, "scan": [],
                                                       "has_notes": True}))
        assert round_estimate(env.cfg, {"rounds": []}, {"session": "round_000"}) == pytest.approx(4 * 0.015)
        ev = Evaluator(env.cfg, LocalRunner(env.world_socket, env.llm_socket), CachedLookup(env.backend), truth=env.truth)
        st = run_rounds(env.cfg, yes=True, allow_scan_hits=False, ev=ev)
        r0 = st["rounds"][0]
        assert r0["harness"] == "h000" and r0["session"] == "round_000" and r0["meta_usd"] == 2.5
        assert r0["evaluated"] and r0["kept"] and load_state(env.cfg)["best"] == "h000"
        assert not (env.root / "runs/meta/round_000_a1").exists()
        assert not any(json.loads(x)["owner"].startswith("meta/") for x in (env.root / "ledger.jsonl").read_text().splitlines())
    finally:
        close_env(env, servers)


def test_loop_records_test_every_round_but_never_uses_or_shows_it():
    """Each evaluated harness also runs on test; decide() uses val only and the feedback has no test."""
    from outer.feedback import build_feedback
    from outer.loop import decide, run_rounds
    from outer.sandbox import LocalRunner
    from tests.conftest import harness_copy

    env, servers = make_env(n_strips=6, test_strips=2, negatives=True)
    try:
        env.cfg["budget_usd"]["total"] = 1000.0
        env.cfg["loop"].update(max_rounds=1, evaluate_test_every_round=True)
        harness_copy("trivial", env.root).rename(env.root / "harnesses/h000")
        (env.root / "runs/meta/round_000").mkdir(parents=True)
        (env.root / "runs/meta/round_000/session.json").write_text(json.dumps(
            {"session": "round_000", "round": 0, "harness": "h000", "ok": True, "meta_usd": 0.0, "probe_usd": 0.0,
             "meta_requests": 0, "probe_requests": 0, "wall_s": 1.0, "scan": [], "has_notes": True}))
        ev = Evaluator(env.cfg, LocalRunner(env.world_socket, env.llm_socket), CachedLookup(env.backend), truth=env.truth)
        st = run_rounds(env.cfg, yes=True, allow_scan_hits=False, ev=ev)
        r0 = st["rounds"][0]
        idx = json.loads((env.root / "tasks/index.json").read_text())
        assert r0["test"]["all"]["n"] == len(idx["test"]) == 4 and r0["test"]["score"] is not None
        assert r0["val"]["score"] == pytest.approx(0.0)  # trivial: off-target edits, never none
        assert all((env.root / "runs/h000/test" / t / "score.json").exists() for t in idx["test"])
        # a second harness that differs from h000 only on test is not kept: decide() reads val only
        harness_copy("trivial", env.root).rename(env.root / "harnesses/h001")
        import shutil
        shutil.copytree(env.root / "runs/h000/val", env.root / "runs/h001/val")
        assert decide(env.cfg, st, "h001")[0] is False
        fb = build_feedback(env.cfg, 1, "h000", st, None)
        text = "".join(p.read_text(errors="replace") for p in fb.rglob("*") if p.is_file())
        assert not any(t in text for t in idx["test"] + idx["val"])
        assert all("test" not in r for r in json.loads((fb / "history.json").read_text())["rounds"])
        # no distance to the true location (trilateration) and no raw root ids reach the meta-agent
        for f in fb.rglob("*"):
            if f.is_file():
                t = f.read_text(errors="replace")
                assert "distance_nm" not in t and "point_roots_at_t" not in t, (f, t[:2000])
        assert all(json.loads(p.read_text()).get("hit") is not None for p in fb.glob("train/*/score.json"))
        from outer.evaluate import PROBE_FIELDS
        assert not any("distance" in f or "root" in f for f in PROBE_FIELDS)
        assert "test" in (env.root / "runs/report.md").read_text()
        assert "true_location_nm_xyz" not in text  # the monitor's field never reaches the meta-agent
        # the monitor export: every evaluated task once, with its true location and the round's decision
        from outer.dashboard_export import export
        m = export(env.cfg)
        res = json.loads((env.root / "runs/dashboard/results.json").read_text())
        rounds = json.loads((env.root / "runs/dashboard/rounds.json").read_text())
        assert m["files"]["results"]["rows"] == len(res)
        assert len([r for r in res if r["harness"] == "h000"]) == sum(len(idx[k]) for k in ("train", "val", "test"))
        assert len([r for r in res if r["harness"] == "h001"]) == len(idx["val"])  # copied in above
        assert all(r["x_nm"] is not None and r["kind"] for r in res)
        assert rounds[0]["harness"] == "h000" and rounds[0]["kept"] == "kept"
        assert any(r["harness"] == "h001" and r["kept"] == "evaluating" for r in rounds)  # runs/h001 exists, not recorded
    finally:
        close_env(env, servers)


@needs_apptainer
def test_meta_session_and_two_rounds(meta_env):
    from outer.loop import load_state, run_rounds

    env = meta_env
    runner = ApptainerRunner(APPTAINER_IMAGE, env.world_socket, env.llm_socket)
    ev = Evaluator(env.cfg, runner, CachedLookup(env.backend), truth=env.truth)
    st = run_rounds(env.cfg, yes=True, allow_scan_hits=False, ev=ev, session_kw={"runner": runner})
    r0 = st["rounds"][0]
    s0 = json.loads((env.root / "runs/meta/round_000/session.json").read_text())
    log = (env.root / "runs/meta/round_000/run.log").read_text()
    assert s0["ok"], (s0, log[-3000:])
    assert (env.root / "harnesses/h000/harness.py").read_text() == TRIVIAL
    assert (env.root / "harnesses/h000/NOTES.md").exists()
    # the meta-agent's requests went through the proxy under its own owner, with the allowed model only
    led = [json.loads(x) for x in (env.root / "ledger.jsonl").read_text().splitlines()]
    meta = [e for e in led if e["owner"] == "meta/round_000"]
    assert meta and all(e["model"] == "claude-opus-5-5" and e["status"] == 200 for e in meta), meta[-3:]
    assert s0["meta_usd"] > 0
    # score_train ran probes on train tasks, paid from the probe budget
    probes = [e for e in led if e["owner"].startswith("probe/round_000/")]
    assert probes and s0["probe_usd"] > 0
    res = list((env.root / "runs/meta/round_000/probe/results").glob("*/result.json"))
    assert res and "hits" in json.loads(res[0].read_text())["summary"]
    # the transcript of the meta session exists and the tool calls are in it
    tr = (env.root / "runs/meta/round_000/transcript.jsonl").read_text()
    assert "score_train" in tr
    # round 0 evaluated on train + val and kept; round 1 (no change) rejected under accept_min_gain_pp=1
    assert r0["kept"] and r0["evaluated"] and r0["train"]["all"]["n"] == 2 and r0["val"]["all"]["n"] == 2
    r1 = st["rounds"][1]
    assert r1["harness"] == "h001" and r1["evaluated"] and not r1["kept"], r1
    assert load_state(env.cfg)["best"] == "h000"
    fb = env.root / "feedback/round_001"
    assert (fb / "README.md").exists() and (fb / "train").exists()
    # the feedback packet never mentions val tasks
    val = json.loads((env.root / "tasks/index.json").read_text())["val"]
    text = "".join(p.read_text(errors="replace") for p in fb.rglob("*") if p.is_file())
    assert not any(t in text for t in val)
    # the revise session saw the feedback (it ran `cat /work/feedback/README.md`)
    tr1 = (env.root / "runs/meta/round_001/transcript.jsonl").read_text()
    assert "Feedback for round 1" in tr1
    assert (env.root / "runs/report.md").exists()
