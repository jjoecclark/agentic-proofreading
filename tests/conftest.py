"""Test fixtures: a temporary repo root with fake tasks/truth, and the real `world` and `llm` services
running against fakes (FakeBackend, FakeUpstream) on Unix sockets. Tests never touch the network."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from outer.config import ROOT, load_config, token_secret
from services.httpserver import serve
from services.llm import proxy as proxy_mod
from services.llm.ledger import Ledger
from services.world import server as world_mod
from tests.fakes import FakeBackend, FakeUpstream, fake_negatives, fake_state, fake_truths


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--live", action="store_true", help="run tests that read real FlyWire data")


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live (reads real FlyWire data)")
    for it in items:
        if "live" in it.keywords:
            it.add_marker(skip)


@dataclass
class Env:
    root: Path
    cfg: Any
    backend: FakeBackend
    truths: dict[str, dict[str, Any]]  # edit tasks
    upstream: FakeUpstream
    world_socket: Path
    llm_socket: Path
    secret: bytes
    proxy: Any
    world: Any
    negatives: dict[str, dict[str, Any]] = field(default_factory=dict)  # no-edit tasks

    def truth(self, task_id: str) -> dict[str, Any]:
        return self.truths[task_id] if task_id in self.truths else self.negatives[task_id]


def make_env(n_strips: int = 4, reply: Any = None, per_task_usd: float = 2.0,
             negatives: bool = False) -> tuple[Env, list[Any]]:
    root = Path(tempfile.mkdtemp(prefix="fw", dir="/tmp"))
    cfg_doc = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg_doc["world"].update(socket=str(root / "w.sock"), cache_dir=str(root / "cache"), port=None)
    cfg_doc["proxy"].update(socket=str(root / "l.sock"), port=None)
    cfg_doc["secrets"]["token_secret_file"] = str(root / "secret")
    cfg_doc["limits"].update(task_wall_s=60, parallel_tasks=4)
    cfg_doc["budget_usd"]["per_task"] = per_task_usd
    (root / "config.yaml").write_text(yaml.safe_dump(cfg_doc))
    cfg = load_config(root / "config.yaml", check_data=False)
    backend = FakeBackend(n_strips)
    truths = fake_truths(backend)
    (root / "tasks").mkdir()
    (root / "truth").mkdir()
    negs = fake_negatives(truths) if negatives else {}
    (root / "truth" / "negatives").mkdir()
    index: dict[str, Any] = {"index_version": "2", "train": [], "val": [], "test": [], "negative": sorted(negs)}
    for j, (tid, tr) in enumerate(sorted(truths.items())):
        split = "train" if j < max(2, len(truths) // 2) else "val"
        (root / "tasks" / f"{tid}.json").write_text(json.dumps(fake_state(tid, tr)))
        (root / "truth" / f"{tr['operation_id']}.json").write_text(json.dumps(tr))
        index[split].append(tid)
        nid = tid.replace("t_fake", "t_fneg")
        if nid in negs:  # a negative goes to its source operation's split
            (root / "tasks" / f"{nid}.json").write_text(json.dumps(fake_state(nid, negs[nid])))
            (root / "truth" / "negatives" / f"{tr['operation_id']}.json").write_text(json.dumps(negs[nid]))
            index[split].append(nid)
    (root / "tasks" / "index.json").write_text(json.dumps(index))
    (root / "truth" / "task_map.json").write_text(json.dumps({"task_map_version": "2", "operations": {
        tr["operation_id"]: {"task_id": tid, "seed": 7} for tid, tr in truths.items()}, "negatives": {
        n["source_operation_id"]: {"task_id": nid, "seed": 7} for nid, n in negs.items()}}))
    secret = token_secret(cfg)
    upstream = FakeUpstream(reply)
    papp = proxy_mod.LLMProxy(secret, Ledger(root / "ledger.jsonl"), cfg["prices_usd_per_mtok"], upstream.url,
                              "fake-key", root)
    wapp = world_mod.WorldService(backend, secret, log_path=root / "runs" / "world_requests.jsonl",
                                  session_dir=root / "cache" / "sessions")
    servers = serve(proxy_mod.make_handler(papp), root / "l.sock") + serve(world_mod.make_handler(wapp), root / "w.sock")
    env = Env(root, cfg, backend, truths, upstream, root / "w.sock", root / "l.sock", secret, papp, wapp, negs)
    return env, servers


def close_env(env: Env, servers: list[Any]) -> None:
    for s in servers:
        s.shutdown()
    env.upstream.close()
    shutil.rmtree(env.root, ignore_errors=True)


@pytest.fixture
def fwenv() -> Any:
    env, servers = make_env()
    yield env
    close_env(env, servers)


HARNESS_DIR = Path(__file__).parent / "harnesses"


def write_answers(dst: Path, truths: dict[str, dict[str, Any]], offset_nm: tuple[float, float, float] = (0, 0, 0),
                  negatives: dict[str, Any] | None = None) -> None:
    """Oracle answers for the fake tasks (tests only; real harnesses never see truth)."""
    import numpy as np

    ans = {}
    off = np.array(offset_nm, dtype=float)
    for tid, tr in truths.items():
        s = (np.array(tr["sources_nm_xyz"]) + off).tolist()
        k = (np.array(tr["sinks_nm_xyz"]) + off).tolist()
        if tr["operation_type"] == "merge":
            ans[tid] = {"decision_version": "1", "action": "merge", "points_nm_xyz": [s[0], k[0]]}
        else:
            ans[tid] = {"decision_version": "1", "action": "split", "sources_nm_xyz": s, "sinks_nm_xyz": k}
    for tid in negatives or {}:
        ans[tid] = {"decision_version": "1", "action": "none"}
    (dst / "answers.json").write_text(json.dumps(ans))


def harness_copy(name: str, dst_root: Path, truths: dict[str, Any] | None = None,
                 offset_nm: tuple[float, float, float] = (0, 0, 0), negatives: dict[str, Any] | None = None) -> Path:
    dst = dst_root / "harnesses" / name
    shutil.copytree(HARNESS_DIR / name.split("@")[0], dst)
    if truths is not None:
        write_answers(dst, truths, offset_nm, negatives)
    return dst


APPTAINER_IMAGE = ROOT / "outer" / "sandbox.sif"
needs_apptainer = pytest.mark.skipif(not (APPTAINER_IMAGE.exists() and shutil.which("apptainer")),
                                     reason="needs apptainer and outer/sandbox.sif")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
