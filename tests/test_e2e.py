"""Phase 4 end to end, with a fake LLM upstream and a fake world: evaluate -> sandbox -> score."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from outer.evaluate import SCORE_TRAIN, Evaluator, ProbeServer, load_scores, task_ids
from outer.sandbox import ApptainerRunner, LocalRunner, scan_harness
from outer.score import CachedLookup
from tests.conftest import APPTAINER_IMAGE, close_env, harness_copy, make_env, needs_apptainer


def evaluator(env, runner):
    return Evaluator(env.cfg, runner, CachedLookup(env.backend), truth=env.truth)


def local(env):
    return LocalRunner(env.world_socket, env.llm_socket)


def apptainer(env):
    return ApptainerRunner(APPTAINER_IMAGE, env.world_socket, env.llm_socket)


@pytest.fixture
def env6():
    env, servers = make_env(n_strips=6)
    yield env
    close_env(env, servers)


@pytest.mark.parametrize("runner_name", ["local", pytest.param("apptainer", marks=needs_apptainer)])
def test_oracle_scores_100_and_offset_scores_0(env6, runner_name):
    runner = local(env6) if runner_name == "local" else apptainer(env6)
    ev = evaluator(env6, runner)
    tids = sorted(env6.truths)
    harness_copy("oracle", env6.root, env6.truths)
    s = ev.evaluate("oracle", "train", tids, parallel=3, harness_dir=env6.root / "harnesses/oracle",
                    run_root=env6.root / "runs/oracle/all", progress=False)
    assert s["all"]["hits"] == len(tids), s
    assert s["merge"]["n"] == 3 and s["split"]["n"] == 3
    harness_copy("oracle@offset", env6.root, env6.truths, offset_nm=(0, 5000, 0))
    s = ev.evaluate("offset", "train", tids, parallel=3, harness_dir=env6.root / "harnesses/oracle@offset",
                    run_root=env6.root / "runs/offset/all", progress=False)
    assert s["all"]["hits"] == 0 and s["miss_reasons"] == {"too_far": len(tids)}, s
    assert s["all"]["median_distance_nm"] == pytest.approx(5000.0)


@pytest.mark.parametrize("runner_name", ["local", pytest.param("apptainer", marks=needs_apptainer)])
def test_trivial_harness_runs_on_all_train_tasks(env6, runner_name):
    runner = local(env6) if runner_name == "local" else apptainer(env6)
    ev = evaluator(env6, runner)
    harness_copy("trivial", env6.root)
    train = task_ids(env6.cfg, "train")
    s = ev.evaluate("trivial", "train", parallel=3, harness_dir=env6.root / "harnesses/trivial", progress=False)
    assert s["all"]["n"] == len(train)
    scores = load_scores(env6.cfg, "trivial", "train")
    assert set(scores) == set(train)
    for t, sc in scores.items():
        d = env6.root / "runs/trivial/train" / t
        assert (d / "decision.json").exists(), (d / "run.log").read_text()
        assert sc["miss_reason"] not in ("crash", "timeout", "no_decision", "invalid_schema"), sc
        assert sc["usd"] > 0 and sc["requests"] == 1
        assert (d / "transcript.jsonl").exists()
    # resumable: a second run skips finished tasks (no new LLM requests)
    n = len(env6.upstream.requests)
    ev.evaluate("trivial", "train", parallel=3, harness_dir=env6.root / "harnesses/trivial", progress=False)
    assert len(env6.upstream.requests) == n


def test_negative_tasks_score_only_none():
    """No-edit tasks: `none` is the hit, any edit is `edit_not_needed`; harnesses see aliased segment ids."""
    env, servers = make_env(n_strips=4, negatives=True)
    try:
        ev = evaluator(env, local(env))
        tids = sorted(env.truths) + sorted(env.negatives)
        harness_copy("oracle", env.root, env.truths, negatives=env.negatives)
        s = ev.evaluate("oracle", "train", tids, parallel=4, harness_dir=env.root / "harnesses/oracle",
                        run_root=env.root / "runs/oracle/all", progress=False)
        assert s["all"]["hits"] == 8 and s["no_edit"] == {**s["no_edit"], "n": 4, "hits": 4}, s
        assert s["no_merge"]["n"] == 2 and s["no_split"]["n"] == 2 and s["edit"]["hits"] == 4
        harness_copy("trivial", env.root)
        s = ev.evaluate("trivial", "train", tids, parallel=4, harness_dir=env.root / "harnesses/trivial",
                        run_root=env.root / "runs/trivial/all", progress=False)
        assert s["no_edit"]["hits"] == 0 and s["miss_reasons"].get("edit_not_needed") == 4, s
        for t in sorted(env.negatives):
            sc = json.loads((env.root / "runs/trivial/all" / t / "score.json").read_text())
            assert sc["kind"] in ("no_merge", "no_split") and sc["miss_reason"] == "edit_not_needed"
            seen = json.loads((env.root / "runs/trivial/all" / t / "task.json").read_text())
            segs = seen["neuroglancer"]["layers"][1]["segments"]
            assert not set(segs) & set(env.negatives[t]["segments"])  # the raw ids never reach the harness
    finally:
        close_env(env, servers)


def test_timeout_crash_and_budget_are_recorded():
    env, servers = make_env(n_strips=2, per_task_usd=0.5)
    try:
        env.cfg["limits"]["task_wall_s"] = 3
        ev = evaluator(env, local(env))
        t = sorted(env.truths)[0]
        for name, reason in (("sleeper", "timeout"), ("crasher", "crash"), ("spender", "budget_exhausted")):
            harness_copy(name, env.root)
            if name == "spender":
                env.cfg["limits"]["task_wall_s"] = 60
                env.upstream.full_output = True
            s = ev.run_task(env.root / "harnesses" / name, t, env.root / "runs" / name / t, f"run/{name}/train")
            assert s["miss_reason"] == reason, (name, s, (env.root / "runs" / name / t / "run.log").read_text()[-2000:])
            if name == "spender":
                assert s["usd"] <= 0.5
    finally:
        close_env(env, servers)


@needs_apptainer
def test_container_isolation(env6):
    from outer.config import ROOT

    harness_copy("isolation", env6.root)
    home = Path.home()
    probe = [str(ROOT / "truth" / "task_map.json"), str(ROOT / "data" / "operations.csv"),
             str(ROOT / "tasks" / "index.json"), str(ROOT / "config.yaml"), str(env6.root / "truth" / "task_map.json"),
             str(env6.root / "tasks" / "index.json"), str(env6.root / "ledger.jsonl"), str(ROOT / ".venv" / "pyvenv.cfg"),
             str(home / ".cloudvolume/secrets/cave-secret.json"), str(home / ".config/flywire_harness/anthropic_api_key"),
             str(home / ".config/flywire_harness/token_secret")]
    (env6.root / "harnesses/isolation/probe_paths.json").write_text(json.dumps(probe))
    ev = evaluator(env6, apptainer(env6))
    t = sorted(env6.truths)[0]
    s = ev.run_task(env6.root / "harnesses/isolation", t, env6.root / "runs/iso" / t, "run/iso/train")
    dec = json.loads((env6.root / "runs/iso" / t / "decision.json").read_text())
    rep = json.loads(dec["notes"])
    for p in probe:
        assert rep[p] != "READABLE", p
    for k in ("140.82.112.3:443", "8.8.8.8:53", "api.anthropic.com:443"):
        assert rep[k] != "CONNECTED", k
    assert rep["anthropic_key_is_run_token"]
    assert not any(k.startswith(("CAVE", "AWS", "GOOGLE")) for k in rep["env_keys"])
    assert s["miss_reason"] == "none"


@needs_apptainer
def test_score_train_probe_roundtrip(env6, tmp_path):
    """The meta-agent's score_train: a request file in /probe is evaluated by the host and answered."""
    import subprocess

    harness_copy("oracle", env6.root, env6.truths)
    ev = evaluator(env6, apptainer(env6))
    probe_dir = env6.root / "probe"
    work = env6.root / "metawork"
    (work / "harness").mkdir(parents=True)
    for f in (env6.root / "harnesses/oracle").iterdir():
        (work / "harness" / f.name).write_bytes(f.read_bytes())
    (work / "bin").mkdir()
    (work / "bin/score_train").write_text(SCORE_TRAIN)
    train = task_ids(env6.cfg, "train")
    srv = ProbeServer(ev, probe_dir, "sess1", train, 5.0, env6.root / "runs/probe/sess1").start()
    try:
        cmd = ["apptainer", "exec", "--containall", "--net", "--network", "none", "-B", f"{work}:/work",
               "-B", f"{probe_dir}:/probe", str(APPTAINER_IMAGE), "python", "/work/bin/score_train", *train[:2]]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "2/2 hits" in out.stdout, out.stdout
        cmd[-2:] = ["t_notatrain"]
        out = subprocess.run(cmd[:-1] + ["t_notatrain"], capture_output=True, text=True, timeout=300)
        assert "not train task ids" in out.stdout
    finally:
        srv.stop()


def test_stop_kills_running_containers_and_records_nothing(tmp_path):
    """Ctrl-C / SIGTERM on the outer harness (stop_all_containers) kills processes in their own process
    groups, and the interrupted run raises Interrupted instead of returning a result to be scored."""
    import threading

    from outer import sandbox

    out: dict[str, BaseException | dict] = {}

    def worker():
        try:
            out["r"] = sandbox.run_process(["sleep", "60"], None, 120, tmp_path / "run.log")
        except BaseException as e:  # noqa: BLE001
            out["r"] = e

    th = threading.Thread(target=worker)
    th.start()
    try:
        deadline = time.time() + 10
        while not sandbox._LIVE and time.time() < deadline:
            time.sleep(0.05)
        (p,) = list(sandbox._LIVE)
        t0 = time.time()
        assert sandbox.stop_all_containers() == 1
        th.join(15)
        assert not th.is_alive() and time.time() - t0 < 10
        assert isinstance(out["r"], sandbox.Interrupted), out
        assert p.poll() is not None and not sandbox._LIVE
        with pytest.raises(sandbox.Interrupted):  # nothing new starts while stopping
            sandbox.run_process(["true"], None, 10, tmp_path / "run2.log")
    finally:
        sandbox.STOPPING.clear()


def test_service_down_records_nothing():
    """If the world service stops, failed runs are not scored as the harness's crashes: evaluation stops,
    no score.json is written (so a resume reruns the task), and a new evaluation refuses to start."""
    from outer import sandbox
    from outer.evaluate import ServicesDown

    env, servers = make_env(n_strips=2)
    try:
        ev = evaluator(env, local(env))
        harness_copy("trivial", env.root)
        t = sorted(env.truths)[0]
        servers[1].shutdown()  # servers = [proxy, world]
        servers[1].server_close()
        with pytest.raises(ServicesDown):
            ev.run_task(env.root / "harnesses/trivial", t, env.root / "runs/trivial/train" / t, "run/trivial/train")
        assert not (env.root / "runs/trivial/train" / t / "score.json").exists()
        sandbox.STOPPING.clear()
        with pytest.raises(SystemExit, match="world"):
            ev.evaluate("trivial", "train", harness_dir=env.root / "harnesses/trivial", progress=False)
    finally:
        sandbox.STOPPING.clear()
        close_env(env, servers)


def test_static_scan_flags_network_code(tmp_path):
    d = tmp_path / "h"
    d.mkdir()
    (d / "harness.py").write_text("import requests\nimport numpy as np\nfrom urllib.request import urlopen\n"
                                  "import subprocess\nsubprocess.run(['curl ', 'x'])\nx = 1  # import socket\n")
    (d / "ok.py").write_text("import numpy\nfrom world_client import WorldClient\n")
    hits = scan_harness(d)
    whats = {(h["file"], h["line"]) for h in hits}
    assert ("harness.py", 1) in whats and ("harness.py", 3) in whats and ("harness.py", 5) in whats
    assert not any(h["file"] == "ok.py" for h in hits)
    time.sleep(0)
