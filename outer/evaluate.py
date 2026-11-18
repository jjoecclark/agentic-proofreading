"""Evaluate a harness on a split (PLAN.md Section 8.1), and serve ``score_train`` probes (8.2).

For each task: a fresh sandbox, a world session token bound to the task's T, a run token with the
per-task dollar cap (enforced by the proxy), and the wall-time limit. Results go to
runs/<harness>/<split>/<task_id>/: decision.json, transcript.jsonl (written by the proxy), usage.json
(from the ledger), run_meta.json, run.log, score.json. A task with a finished score.json is skipped.

    .venv/bin/python -m outer.evaluate h000 train [--tasks t_..] [--parallel 8]

Needs the world service and the llm proxy running (README.md).
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from outer.config import Config, load_config, token_secret
from outer.score import CachedLookup, Truth, read_json, score_run, summarize
from services.aliases import alias_state
from services.llm.ledger import read_entries
from services.llm.ledger import summarize as ledger_summary
from services.tokens import run_token, world_token

EVAL_VERSION = "1"


def task_ids(cfg: Config, split: str) -> list[str]:
    return list(json.loads((cfg.root / "tasks" / "index.json").read_text())[split])


def negative_ids(cfg: Config) -> list[str]:
    """Task ids (all splits) of the negative tasks: no edit needed. Known to the outer harness only."""
    return list(json.loads((cfg.root / "tasks" / "index.json").read_text()).get("negative", []))


def task_file(cfg: Config, task_id: str) -> Path:
    return cfg.root / "tasks" / f"{task_id}.json"


class ServicesDown(BaseException):
    """The world service or the proxy stopped. Runs from then on would fail through no fault of the
    harness, so nothing more is recorded (a BaseException: no `except Exception` turns it into a score)."""


def services_down(cfg: Config, runner: Any = None) -> list[str]:
    """The services that do not answer: the runner's sockets if it has them, else config.yaml's."""
    import socket

    if runner is not None and hasattr(runner, "world_socket"):
        socks = [("world", runner.world_socket), ("llm proxy", runner.llm_socket)]
    else:
        socks = [("world", cfg.path("world.socket")), ("llm proxy", cfg.path("proxy.socket"))]
    down = []
    for name, path in socks:
        s = socket.socket(socket.AF_UNIX)
        try:
            s.connect(str(path))
        except OSError:
            down.append(f"{name} ({path})")
        finally:
            s.close()
    return down


def check_services(cfg: Config, runner: Any = None) -> None:
    down = services_down(cfg, runner)
    if down:
        raise SystemExit(f"not running: {', '.join(down)}. Start the services (README.md), then rerun.")


class Evaluator:
    def __init__(self, cfg: Config, runner: Any, lookup: Any, truth: Callable[[str], dict[str, Any]] | None = None):
        self.cfg, self.runner, self.lookup = cfg, runner, lookup
        self.truth = truth or Truth(cfg.root / "truth")
        self.secret = token_secret(cfg)
        self.ledger_path = cfg.root / "ledger.jsonl"
        self.log_root = cfg.root  # the proxy writes transcripts relative to its own root (the repo)
        self.radius_nm = float(cfg["scoring"]["hit_radius_um"]) * 1000.0
        self._score_lock = threading.Lock()

    def run_task(self, harness_dir: Path, task_id: str, run_dir: Path, owner_prefix: str,
                 group: str | None = None, group_cap_usd: float | None = None) -> dict[str, Any]:
        run_dir = Path(run_dir)
        done = read_json(run_dir / "score.json")
        if done is not None:
            return done
        if run_dir.exists():
            shutil.rmtree(run_dir)
        workdir, outdir = run_dir / "workdir", run_dir / "out"
        for d in (workdir, outdir, workdir / ".home", workdir / ".tmp"):
            d.mkdir(parents=True, exist_ok=True)
        state = json.loads(task_file(self.cfg, task_id).read_text())
        tpath = run_dir / "task.json"  # what the harness sees: segment ids as this task's aliases
        tpath.write_text(json.dumps(alias_state(state, self.secret), indent=1) + "\n")
        attempt = secrets.token_hex(4)
        owner = f"{owner_prefix}/{task_id}/{attempt}"
        wall_s = float(self.cfg["limits"]["task_wall_s"])
        cap = float(self.cfg["budget_usd"]["per_task"])
        models = list(self.cfg["models"]["inner_allowed"])
        conf = {
            "world_token": world_token(self.secret, task_id, state["timestamp_utc"]),
            "llm_token": run_token(self.secret, owner, cap, models, str(run_dir.resolve().relative_to(self.log_root.resolve())),
                                   group=group, group_cap_usd=group_cap_usd),
            "limits": {"wall_s": wall_s, "deadline_unix": time.time() + wall_s, "usd_cap": cap, "models": models},
        }
        (outdir / "config.json").write_text(json.dumps(conf))
        meta = self.runner.run_harness(Path(harness_dir).resolve(), tpath.resolve(), workdir, outdir,
                                       wall_s + 30.0, run_dir / "run.log")
        down = services_down(self.cfg, self.runner)
        if down:  # the run may have failed because of it: record nothing, stop everything
            from outer.sandbox import stop_all_containers

            stop_all_containers()
            raise ServicesDown(f"{', '.join(down)} stopped during evaluation; nothing recorded for {task_id}. "
                               "Restart the services, then rerun the same command to resume.")
        meta.update(owner=owner, task_id=task_id, eval_version=EVAL_VERSION)
        if (outdir / "decision.json").exists():
            shutil.copy(outdir / "decision.json", run_dir / "decision.json")
        err = read_json(outdir / "error.json")
        if err is not None:
            meta["error"] = {k: err.get(k) for k in ("type", "message")}
            shutil.copy(outdir / "error.json", run_dir / "error.json")
        (outdir / "config.json").unlink(missing_ok=True)
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        usage = ledger_summary(self.ledger_path, owner_prefix=owner)
        usage["refused_budget"] = sum(1 for e in read_entries(self.ledger_path)
                                      if e.get("owner") == owner and e.get("status") == "refused_budget")
        usage["owner"] = owner
        (run_dir / "usage.json").write_text(json.dumps(usage, indent=1) + "\n")
        with self._score_lock:  # the lookup cache is shared
            return score_run(run_dir, task_id, self.truth(task_id), self.lookup, self.radius_nm)

    def evaluate(self, harness: str, split: str, tids: list[str] | None = None, parallel: int | None = None,
                 harness_dir: Path | None = None, run_root: Path | None = None,
                 progress: bool = True) -> dict[str, Any]:
        check_services(self.cfg, self.runner)
        tids = list(tids) if tids else task_ids(self.cfg, split)
        harness_dir = Path(harness_dir or self.cfg.root / "harnesses" / harness)
        run_root = Path(run_root or self.cfg.root / "runs" / harness / split)
        parallel = int(parallel or self.cfg["limits"]["parallel_tasks"])
        scores: dict[str, dict[str, Any]] = {}

        def one(t: str) -> None:
            try:
                s = self.run_task(harness_dir, t, run_root / t, f"run/{harness}/{split}")
            except Exception as e:  # noqa: BLE001  (an outer-harness failure, not the harness's)
                traceback.print_exc()
                s = {"task_id": t, "hit": False, "miss_reason": f"evaluator_error: {type(e).__name__}: {e}",
                     "kind": "?", "operation_type": "?", "distance_nm": None, "usd": 0.0}
            scores[t] = s
            if progress:
                print(f"  {harness}/{split}/{t}: {'HIT ' if s['hit'] else 'miss'} "
                      f"{s.get('miss_reason') or ''} d={_fmt(s.get('distance_nm'))} ${s.get('usd', 0):.3f} "
                      f"[{len(scores)}/{len(tids)}]", flush=True)

        with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            list(ex.map(one, tids))
        ordered = [scores[t] for t in tids]
        summary = {"eval_version": EVAL_VERSION, "harness": harness, "split": split, "task_ids": tids,
                   **summarize(ordered)}
        if run_root.exists():
            (run_root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        return summary


def _fmt(d: Any) -> str:
    return "-" if d is None else f"{d / 1000:.2f}um"


def load_scores(cfg: Config, harness: str, split: str) -> dict[str, dict[str, Any]]:
    out = {}
    for t in task_ids(cfg, split):
        s = read_json(cfg.root / "runs" / harness / split / t / "score.json")
        if s is not None:
            out[t] = s
    return out


# ============================================================================ score_train probes


SCORE_TRAIN = r'''#!/usr/bin/env python3
"""score_train: run the harness in /work/harness on train tasks and print hit/miss, distance, cost.

    score_train t_0123abcd t_4567ef01 ...     run these train tasks (each in a fresh sandbox)
    score_train --all-examples                run the example tasks listed in /work/tasks/EXAMPLES
    score_train --wait REQUEST_ID             keep waiting for an earlier request

The harness folder is snapshotted when you call this. Spending comes from the session's probe
budget. Per-task details (decision.json, transcript.jsonl, run.log, error.json) are copied to
/probe/results/<request>/<task_id>/.
"""
import json, os, shutil, sys, time, uuid
PROBE = "/probe"
def wait(rid, timeout):
    res = os.path.join(PROBE, "results", rid, "result.json")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(res):
            r = json.load(open(res))
            if r.get("error"):
                print("ERROR:", r["error"]); return 1
            print(f"request {rid}: {r['summary']}")
            print(f"{'task':12} {'type':6} {'result':5} {'miss_reason':24} {'distance_um':>11} {'usd':>7} {'wall_s':>7}")
            for t in r["tasks"]:
                d = t.get("distance_nm")
                print(f"{t['task_id']:12} {t.get('kind') or t.get('operation_type','?'):8} {'HIT' if t['hit'] else 'miss':5} "
                      f"{(t.get('miss_reason') or ''):24} {('-' if d is None else f'{d/1000:.2f}'):>11} "
                      f"{t.get('usd',0):7.3f} {(t.get('wall_s') or 0):7.0f}")
            print(f"details: {os.path.join(PROBE, 'results', rid)}/<task_id>/")
            print(f"probe budget left: ${r.get('probe_budget_left_usd', float('nan')):.2f}")
            return 0
        time.sleep(3)
    print(f"still running; run: score_train --wait {rid}")
    return 2
def main(a):
    timeout = float(os.environ.get("SCORE_TRAIN_TIMEOUT_S", "1500"))
    if a[:1] == ["--wait"]:
        return wait(a[1], timeout)
    if a[:1] == ["--all-examples"]:
        a = open("/work/tasks/EXAMPLES").read().split()
    if not a:
        print(__doc__); return 2
    rid = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:6]
    d = os.path.join(PROBE, "requests", rid)
    shutil.copytree("/work/harness", os.path.join(d, "harness"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    json.dump({"task_ids": a}, open(os.path.join(d, "request.json"), "w"))
    open(os.path.join(d, "READY"), "w").close()
    print(f"submitted request {rid} for {len(a)} task(s)", flush=True)
    return wait(rid, timeout)
sys.exit(main(sys.argv[1:]))
'''


class ProbeServer:
    """Watches <probe_dir>/requests/ for score_train requests from a meta-agent session and evaluates
    each snapshot on the requested train tasks (fresh sandboxes, the session's probe budget)."""

    def __init__(self, ev: Evaluator, probe_dir: Path, session: str, allowed_tasks: list[str],
                 probe_budget_usd: float, run_root: Path):
        self.ev, self.probe_dir, self.session = ev, Path(probe_dir), session
        self.allowed = set(allowed_tasks)
        self.budget = float(probe_budget_usd)
        self.group = f"probe/{session}"
        self.run_root = Path(run_root)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        (self.probe_dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.probe_dir / "results").mkdir(parents=True, exist_ok=True)

    def start(self) -> ProbeServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def spent(self) -> float:
        return ledger_summary(self.ev.ledger_path, owner_prefix=f"probe/{self.session}/")["usd"]

    def _loop(self) -> None:
        busy: set[str] = set()
        while not self._stop.is_set():
            for ready in sorted((self.probe_dir / "requests").glob("*/READY")):
                rid = ready.parent.name
                if rid in busy or (ready.parent / "CLAIMED").exists():
                    continue
                (ready.parent / "CLAIMED").touch()
                busy.add(rid)
                threading.Thread(target=self._handle, args=(rid,), daemon=True).start()
            self._stop.wait(2.0)

    def _handle(self, rid: str) -> None:
        req_dir = self.probe_dir / "requests" / rid
        out_dir = self.probe_dir / "results" / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {"request_id": rid}
        try:
            req = json.loads((req_dir / "request.json").read_text())
            tids = [str(t) for t in req.get("task_ids", [])]
            bad = [t for t in tids if t not in self.allowed]
            if bad:
                raise ValueError(f"not train task ids: {bad[:5]} (score_train runs train tasks only)")
            if self.spent() >= self.budget:
                raise ValueError(f"probe budget of ${self.budget:.2f} is used up")
            harness_dir = req_dir / "harness"
            if not (harness_dir / "harness.py").exists():
                raise ValueError("no harness.py in /work/harness")
            scores = []
            par = int(self.ev.cfg["limits"]["parallel_tasks"])

            def one(t: str) -> dict[str, Any]:
                rd = self.run_root / rid / t
                s = self.ev.run_task(harness_dir, t, rd, f"probe/{self.session}/{rid}", group=self.group,
                                     group_cap_usd=self.budget)
                dst = out_dir / t
                dst.mkdir(parents=True, exist_ok=True)
                for name in ("decision.json", "transcript.jsonl", "run.log", "error.json", "usage.json"):
                    if (rd / name).exists():
                        shutil.copy(rd / name, dst / name)
                if (rd / "media").exists():
                    shutil.copytree(rd / "media", dst / "media", dirs_exist_ok=True)
                keep = ("task_id", "kind", "operation_type", "action", "hit", "miss_reason", "distance_nm", "usd",
                        "wall_s", "tokens", "requests")
                return {k: s.get(k) for k in keep}

            with ThreadPoolExecutor(max_workers=max(1, par)) as ex:
                scores = list(ex.map(one, tids))
            n, k = len(scores), sum(bool(s["hit"]) for s in scores)
            result.update(tasks=scores, summary=f"{k}/{n} hits, ${sum(s['usd'] or 0 for s in scores):.3f}")
        except Exception as e:  # noqa: BLE001
            result["error"] = f"{type(e).__name__}: {e}"
        result["probe_budget_left_usd"] = max(0.0, self.budget - self.spent())
        tmp = out_dir / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        tmp.rename(out_dir / "result.json")


def main(argv: list[str] | None = None) -> None:
    from outer.sandbox import make_runner
    from services.world.server import FlyWireBackend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("harness")
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--yes", action="store_true", help="confirm running the test split")
    args = ap.parse_args(argv)
    cfg = load_config()
    from outer.sandbox import install_stop_handlers
    install_stop_handlers()  # Ctrl-C / kill also stops the containers
    if args.split == "test" and not args.yes:
        raise SystemExit("the test split runs once, at the end, after asking the user (PLAN.md 8.4); pass --yes")
    backend = FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
    ev = Evaluator(cfg, make_runner(cfg), CachedLookup(backend))
    s = ev.evaluate(args.harness, args.split, args.tasks, args.parallel)
    a = s["all"]
    print(f"{args.harness} {args.split}: {a['hits']}/{a['n']} hits, ${s['usd_total']:.2f}; misses {s['miss_reasons']}")


if __name__ == "__main__":
    main()
