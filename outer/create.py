"""Round 0 (PLAN.md Section 8.2): the meta-agent writes harnesses/h000.

The meta-agent (Claude Agent SDK, ``models.meta_agent``, no web tools) runs in the sandbox with:
TASK.md, API.md and the world client, the train task files (three marked as examples), world tokens
for train tasks, and ``score_train`` (runs the harness under construction on train tasks, paid from
``budget_usd.meta_probe``). Its own Claude usage is capped at ``budget_usd.meta_session``.

This step spends real money (up to meta_session + meta_probe), so it needs --yes.

    .venv/bin/python -m outer.create            # shows the plan and the worst-case cost
    .venv/bin/python -m outer.create --yes      # runs it
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from outer.config import ROOT, Config, load_config, token_secret
from outer.evaluate import SCORE_TRAIN, Evaluator, ProbeServer, check_services, negative_ids, task_ids
from outer.sandbox import copy_tree, scan_harness
from services.aliases import alias_state
from services.llm.ledger import summarize as ledger_summary
from services.tokens import run_token, world_token

META_VERSION = "1"

SYSTEM_APPEND = """
You are the meta-agent of an automated research loop. You run inside a sandbox with no network
access; the only services are the FlyWire `world` (see /work/API.md) and the Claude API proxy. Your
working directory is /work. Only /work/harness/ is kept when you finish. Do not try to reach the
internet or to find the answers to tasks by any route other than solving them from the data.
`score_train` (on PATH) runs a snapshot of /work/harness on train tasks; it can take many minutes,
so pass a generous Bash timeout (up to 1800000 ms) or rerun it with `--wait <request>`.
"""


def fill(template: str, **values: Any) -> str:
    """Fill ``{name}`` / ``{name:fmt}`` placeholders for known names only (JSON braces stay as they are)."""
    def sub(m: re.Match) -> str:
        if m.group(1) not in values:
            return m.group(0)
        return format(values[m.group(1)], m.group(2)[1:] if m.group(2) else "")
    return re.sub(r"\{(\w+)(:[^{}]*)?\}", sub, template)


def render_task_md(cfg: Config) -> str:
    model = cfg["models"]["inner_allowed"][0]
    p = cfg["prices_usd_per_mtok"][model]
    txt = re.sub(r"<!--.*?-->\n*", "", (ROOT / "outer" / "prompts" / "TASK.md").read_text(), flags=re.S)
    return fill(txt, radius_um=cfg["scoring"]["hit_radius_um"], model=model, wall_s=float(cfg["limits"]["task_wall_s"]),
                      per_task_usd=float(cfg["budget_usd"]["per_task"]), in_price=p["input"], out_price=p["output"],
                      cache_read=p["cache_read"], cache_write=p["cache_write_5m"],
                      probe_usd=float(cfg["budget_usd"]["meta_probe"]), session_usd=float(cfg["budget_usd"]["meta_session"]),
                      parallel=int(cfg["limits"]["parallel_tasks"]), n_train=len(task_ids(cfg, "train")),
                      n_train_negative=len(set(task_ids(cfg, "train")) & set(negative_ids(cfg))))


def example_tasks(cfg: Config, train: list[str]) -> list[str]:
    """Example train tasks, chosen deterministically: two merges, one split, and one no-edit task of each
    kind (2 segments: should not be merged; 1 segment: should not be split)."""
    neg = set(negative_ids(cfg))
    by: dict[tuple[int, bool], list[str]] = {(1, False): [], (2, False): [], (1, True): [], (2, True): []}
    for t in train:
        st = json.loads((cfg.root / "tasks" / f"{t}.json").read_text())
        by[(len(st["neuroglancer"]["layers"][1]["segments"]), t in neg)].append(t)
    return by[(2, False)][:2] + by[(1, False)][:1] + by[(2, True)][:1] + by[(1, True)][:1]


def worst_case_usd(cfg: Config) -> float:
    return float(cfg["budget_usd"]["meta_session"]) + float(cfg["budget_usd"]["meta_probe"])


def new_session_dir(cfg: Config, round_k: int) -> tuple[str, Path]:
    base = f"round_{round_k:03d}"
    for a in range(100):
        name = base if a == 0 else f"{base}_a{a}"
        d = cfg.root / "runs" / "meta" / name
        if not d.exists():
            return name, d
    raise SystemExit("too many attempts for this round")


def prepare_workdir(cfg: Config, work: Path, start_harness: Path | None, feedback_dir: Path | None) -> list[str]:
    """Assemble /work for the meta-agent. Only train tasks, never truth/, the CSV, val or test."""
    secret = token_secret(cfg)
    train = task_ids(cfg, "train")
    (work / "harness").mkdir(parents=True)
    if start_harness is not None:
        copy_tree(start_harness, work / "harness")
    (work / "TASK.md").write_text(render_task_md(cfg))
    (work / "docs").mkdir()
    shutil.copy(ROOT / "outer" / "prompts" / "proofreading_guide.md", work / "docs" / "proofreading_guide.md")
    shutil.copy(ROOT / "services" / "world" / "API.md", work / "API.md")
    shutil.copy(ROOT / "services" / "world" / "client.py", work / "world_client.py")
    (work / "tasks").mkdir()
    tokens = {}
    for t in train:
        state = json.loads((cfg.root / "tasks" / f"{t}.json").read_text())
        (work / "tasks" / f"{t}.json").write_text(json.dumps(alias_state(state, secret), indent=1) + "\n")
        tokens[t] = world_token(secret, t, state["timestamp_utc"])
    (work / "tasks" / "EXAMPLES").write_text("\n".join(example_tasks(cfg, train)) + "\n")
    (work / "world_tokens.json").write_text(json.dumps(tokens, indent=1))
    (work / "bin").mkdir()
    (work / "bin" / "score_train").write_text(SCORE_TRAIN)
    (work / "bin" / "score_train").chmod(0o755)
    for d in (".home", ".tmp"):
        (work / d).mkdir()
    if feedback_dir is not None:
        shutil.copytree(feedback_dir, work / "feedback")
    return train


def run_meta_session(cfg: Config, round_k: int, out_name: str, prompt: str, start_harness: Path | None = None,
                     feedback_dir: Path | None = None, runner: Any = None, ev: Evaluator | None = None) -> dict[str, Any]:
    from outer.sandbox import make_runner
    from outer.score import CachedLookup
    from services.world.server import FlyWireBackend

    out_dir = cfg.root / "harnesses" / out_name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; harness versions are never overwritten")
    check_services(cfg)
    runner = runner or make_runner(cfg)
    if ev is None:
        backend = FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
        ev = Evaluator(cfg, runner, CachedLookup(backend))
    session, sdir = new_session_dir(cfg, round_k)
    work, probe, out = sdir / "work", sdir / "probe", sdir / "out"
    train = prepare_workdir(cfg, work, start_harness, feedback_dir)
    probe.mkdir(parents=True)
    out.mkdir(parents=True)
    secret = token_secret(cfg)
    model = cfg["models"]["meta_agent"]
    conf = {"llm_token": run_token(secret, f"meta/{session}", float(cfg["budget_usd"]["meta_session"]), [model],
                                   str(sdir.relative_to(cfg.root))),
            "model": model, "tools": list(cfg["sandbox"]["meta_tools"]), "prompt": prompt,
            "system_append": SYSTEM_APPEND, "max_turns": None, "meta_version": META_VERSION}
    (out / "config.json").write_text(json.dumps(conf))
    srv = ProbeServer(ev, probe, session, train, float(cfg["budget_usd"]["meta_probe"]),
                      cfg.root / "runs" / "probe" / session).start()
    t0 = time.time()
    print(f"meta-agent session {session}: {sdir} (wall limit {cfg['limits']['meta_wall_s']} s)", flush=True)
    try:
        meta = runner.run_meta(work, probe, out, float(cfg["limits"]["meta_wall_s"]), sdir / "run.log")
    finally:
        srv.stop()
        (out / "config.json").unlink(missing_ok=True)
    meta_cost = ledger_summary(cfg.root / "ledger.jsonl", owner_prefix=f"meta/{session}")
    probe_cost = ledger_summary(cfg.root / "ledger.jsonl", owner_prefix=f"probe/{session}/")
    result: dict[str, Any] = {"meta_version": META_VERSION, "session": session, "round": round_k, "harness": out_name,
                              "wall_s": round(time.time() - t0, 1), **meta,
                              "meta_usd": meta_cost["usd"], "probe_usd": probe_cost["usd"],
                              "meta_requests": meta_cost["requests"], "probe_requests": probe_cost["requests"]}
    if (out / "error.json").exists():
        result["error"] = json.loads((out / "error.json").read_text()).get("message")
    if (work / "harness" / "harness.py").exists():
        copy_tree(work / "harness", out_dir)
        result["scan"] = scan_harness(out_dir)
        result["has_notes"] = (out_dir / "NOTES.md").exists()
        result["ok"] = True
    else:
        result["ok"] = False
        result["why"] = "the meta-agent left no /work/harness/harness.py"
    (sdir / "session.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def print_result(r: dict[str, Any]) -> None:
    print(f"session {r['session']}: {'wrote harnesses/' + r['harness'] if r['ok'] else 'FAILED: ' + r.get('why', '')}; "
          f"meta ${r['meta_usd']:.2f} ({r['meta_requests']} requests), probes ${r['probe_usd']:.2f}, "
          f"{r['wall_s'] / 60:.0f} min" + (" [timed out]" if r.get("timed_out") else ""))
    if r.get("error"):
        print(f"  meta-agent error: {r['error']}")
    for h in r.get("scan") or []:
        print(f"  SCAN: {h['file']}:{h['line']} {h['what']}: {h['text']}")
    if r.get("ok") and not r.get("has_notes"):
        print("  warning: no NOTES.md")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="approve the spend (up to meta_session + meta_probe)")
    ap.add_argument("--name", default="h000")
    args = ap.parse_args(argv)
    cfg = load_config()
    from outer.sandbox import install_stop_handlers
    install_stop_handlers()  # Ctrl-C / kill also stops the containers
    worst = worst_case_usd(cfg)
    print(f"round 0: the meta-agent ({cfg['models']['meta_agent']}) writes harnesses/{args.name}. Worst case "
          f"${worst:.2f} (session ${cfg['budget_usd']['meta_session']:.2f} + probes "
          f"${cfg['budget_usd']['meta_probe']:.2f}); ask_above is ${cfg['budget_usd']['ask_above']:.2f}.")
    if worst > float(cfg["budget_usd"]["ask_above"]) and not args.yes:
        raise SystemExit("not started: rerun with --yes once the user approves this spend")
    prompt = (ROOT / "outer" / "prompts" / "meta_create.md").read_text()
    prompt = prompt.split("-->", 1)[-1].strip()
    print_result(run_meta_session(cfg, 0, args.name, prompt))


if __name__ == "__main__":
    main()
