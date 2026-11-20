"""The outer loop (PLAN.md Section 8.4). Plain code, not an agent.

Each round: check the budget in the ledger; create (round 0) or revise (from the best harness);
evaluate on train and val; keep the revision only if its val score (mean of the per-kind hit rates,
outer/score.py) beats the best's by at least loop.accept_min_gain_pp percentage points on the same
tasks. With loop.evaluate_test_every_round (the user's approval, 2026-11-20), each evaluated harness is
also run on test, for the record only: test results never decide anything and never reach the
meta-agent (outer/feedback.py takes only train and aggregate val). Stop after loop.max_rounds, after loop.plateau_rounds rounds
in a row with nothing kept, or when budget_usd.total would be exceeded. State: runs/loop_state.json.

    .venv/bin/python -m outer.loop --yes             # run rounds within budget_usd.total
    .venv/bin/python -m outer.loop --final --yes     # at the end: best and h000 on test (reuses
                                                     # the per-round test runs when they exist)

Steps expected to cost more than budget_usd.ask_above need --yes (the user's approval of
budget_usd.total) or an interactive "y".
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from outer.config import ROOT, Config, load_config
from outer.evaluate import Evaluator, load_scores, task_ids
from outer.feedback import short
from outer.score import paired_gain
from services.llm.ledger import summarize as ledger_summary

LOOP_VERSION = "2"  # 2: per-kind score keep rule (accept_min_gain_pp); test recorded every round


def state_path(cfg: Config):
    return cfg.root / "runs" / "loop_state.json"


def load_state(cfg: Config) -> dict[str, Any]:
    p = state_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    return {"loop_version": LOOP_VERSION, "rounds": [], "best": None, "final": None}


def save_state(cfg: Config, st: dict[str, Any]) -> None:
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1) + "\n")
    tmp.replace(p)


def ledger_total(cfg: Config) -> float:
    return ledger_summary(cfg.root / "ledger.jsonl")["usd"]


def per_task_estimate(cfg: Config, st: dict[str, Any]) -> float:
    """Measured mean cost per evaluated task (x1.5 margin), or the per-task cap before any measurement."""
    usd, n = 0.0, 0
    for r in st["rounds"]:
        for k in ("train", "val", "test"):
            if r.get(k):
                usd += r[k].get("usd_total") or 0.0
                n += r[k]["all"]["n"]
    cap = float(cfg["budget_usd"]["per_task"])
    return min(cap, 1.5 * usd / n) if n else cap


def eval_splits(cfg: Config) -> tuple[str, ...]:
    """The splits every evaluated harness runs on."""
    return ("train", "val", "test") if cfg["loop"].get("evaluate_test_every_round") else ("train", "val")


def round_estimate(cfg: Config, st: dict[str, Any], session: dict[str, Any] | None = None) -> float:
    """Worst case of the next round. With `session` (this round's meta session, already finished), only
    its evaluation, priced from that session's last probe when it ran one."""
    n_eval = sum(len(task_ids(cfg, k)) for k in eval_splits(cfg))
    if session is None:
        return (float(cfg["budget_usd"]["meta_session"]) + float(cfg["budget_usd"]["meta_probe"])
                + n_eval * per_task_estimate(cfg, st))
    measured = last_probe_per_task_usd(cfg.root / "runs" / "meta" / session["session"])
    if measured is None:
        return n_eval * per_task_estimate(cfg, st)
    return n_eval * min(float(cfg["budget_usd"]["per_task"]), 1.5 * measured)


def finished_session(cfg: Config, k: int) -> dict[str, Any] | None:
    """The session.json of round k's meta session if it already finished and wrote harnesses/hK, but the
    round was not recorded: round 0 run by `outer.create`, or a loop stopped during evaluation. The loop
    then evaluates that harness instead of running the session again (harnesses are never overwritten)."""
    name = f"h{k:03d}"
    if not (cfg.root / "harnesses" / name / "harness.py").exists():
        return None
    for p in sorted((cfg.root / "runs" / "meta").glob(f"round_{k:03d}*/session.json")):
        r = json.loads(p.read_text())
        if r.get("ok") and r.get("harness") == name:
            return r
    return None


def last_probe_per_task_usd(session_dir) -> float | None:
    """Mean cost per task in the session's last score_train probe, or None if it ran none."""
    results = sorted((session_dir / "probe" / "results").glob("*/result.json"))
    if not results:
        return None
    tasks = json.loads(results[-1].read_text()).get("tasks") or []
    return sum(t.get("usd") or 0.0 for t in tasks) / len(tasks) if tasks else None


def confirm(question: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} -> not confirmed (no terminal; pass --yes once the user approves)")
        return False
    return input(f"{question} [y/N] ").strip().lower() == "y"


def decide(cfg: Config, st: dict[str, Any], name: str) -> tuple[bool, str, dict[str, Any] | None]:
    if st["best"] is None:
        return True, "first harness", None
    g = paired_gain(load_scores(cfg, name, "val"), load_scores(cfg, st["best"], "val"))
    need = float(cfg["loop"]["accept_min_gain_pp"])
    if g["score_gain_pp"] is None:
        return False, "no val scores to compare", g
    msg = (f"val score {100 * g['new_score']:.1f}% vs {100 * g['best_score']:.1f}% "
           f"({g['score_gain_pp']:+.1f} pp {'>=' if g['score_gain_pp'] >= need else '<'} +{need:g} pp)")
    return g["score_gain_pp"] >= need, msg, g


def run_rounds(cfg: Config, yes: bool, allow_scan_hits: bool, ev: Evaluator | None = None,
               session_kw: dict[str, Any] | None = None) -> dict[str, Any]:
    from outer.create import print_result, run_meta_session
    from outer.report import write_report
    from outer.revise import revise

    total = cfg["budget_usd"].get("total")
    if total is None:
        raise SystemExit("budget_usd.total is not set. The user sets it before Phase 6 (config.yaml).")
    st = load_state(cfg)
    lc = cfg["loop"]
    ev = ev or make_evaluator(cfg)
    session_kw = dict(session_kw or {}, ev=ev)
    while True:
        k = len(st["rounds"])
        if k >= int(lc["max_rounds"]):
            print(f"stop: {k} rounds done (max_rounds)")
            break
        recent = st["rounds"][1:][-int(lc["plateau_rounds"]):]
        if len(recent) >= int(lc["plateau_rounds"]) and not any(r.get("kept") for r in recent):
            print(f"stop: no kept revision in the last {lc['plateau_rounds']} rounds")
            break
        adopted = finished_session(cfg, k)
        spent, est = ledger_total(cfg), round_estimate(cfg, st, adopted)
        if spent + est > float(total):
            print(f"stop: spent ${spent:.2f} + round {k} worst case ${est:.2f} would exceed budget_usd.total ${total:.2f}")
            break
        if est > float(cfg["budget_usd"]["ask_above"]) and not confirm(
                f"round {k} may cost up to ${est:.2f} (spent so far ${spent:.2f} of ${total:.2f}). Run it?", yes):
            break
        name = f"h{k:03d}"
        if adopted:
            res = adopted
            print(f"round {k}: harnesses/{name} from session {res['session']} already exists; evaluating it")
        elif k == 0:
            prompt = (ROOT / "outer/prompts/meta_create.md").read_text().split("-->", 1)[-1].strip()
            res = run_meta_session(cfg, 0, name, prompt, **session_kw)
        else:
            res = revise(cfg, k, st, **session_kw)
        print_result(res)
        rec: dict[str, Any] = {"round": k, "harness": name if res["ok"] else None, "session": res["session"],
                               "meta_usd": res["meta_usd"], "probe_usd": res["probe_usd"], "kept": False,
                               "evaluated": False, "scan_hits": len(res.get("scan") or [])}
        if not res["ok"]:
            rec["reason"] = f"no harness: {res.get('why')}"
            st["rounds"].append(rec)
            save_state(cfg, st)
            continue
        if rec["scan_hits"] and not allow_scan_hits and not confirm(
                f"{name} has {rec['scan_hits']} static-scan hit(s) (listed above). Evaluate it anyway?", False):
            rec["reason"] = "stopped at static-scan hits"
            st["rounds"].append(rec)
            save_state(cfg, st)
            break
        before = ledger_total(cfg)
        tr = ev.evaluate(name, "train")
        va = ev.evaluate(name, "val")
        rec.update(evaluated=True, train=short(tr), val=short(va), train_summary=tr, val_summary_aggregate=short(va))
        if "test" in eval_splits(cfg):  # recorded only: decide() and the feedback never read it
            rec["test"] = short(ev.evaluate(name, "test"))
        rec["eval_usd"] = ledger_total(cfg) - before
        kept, why, gain = decide(cfg, st, name)
        rec.update(kept=kept, reason=why, gain=gain)
        if kept:
            st["best"] = name
        st["rounds"].append(rec)
        save_state(cfg, st)
        write_report(cfg, st)
        print(f"round {k}: {name} train score {_pct(tr.get('score'))}, val score {_pct(va.get('score'))}"
              f"{', test score ' + _pct(rec['test'].get('score')) if rec.get('test') else ''} "
              f"-> {'KEPT' if kept else 'rejected'} ({why}); best = {st['best']}", flush=True)
    write_report(cfg, st)
    return st


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def run_final(cfg: Config, yes: bool, ev: Evaluator | None = None) -> dict[str, Any]:
    from outer.report import write_report

    st = load_state(cfg)
    if not st.get("best"):
        raise SystemExit("no best harness")
    names = list(dict.fromkeys([st["best"], "h000"]))
    todo = [h for h in names if len(load_scores(cfg, h, "test")) < len(task_ids(cfg, "test"))]
    n = sum(len(task_ids(cfg, "test")) - len(load_scores(cfg, h, "test")) for h in todo)
    est = n * per_task_estimate(cfg, st)
    if todo and not confirm(f"run the test split on {todo} ({n} task runs, up to ${est:.2f})?", yes):
        raise SystemExit("not run")
    ev = ev or make_evaluator(cfg)
    st["final"] = {h: ev.evaluate(h, "test") for h in names}
    save_state(cfg, st)
    write_report(cfg, st)
    return st


def make_evaluator(cfg: Config) -> Evaluator:
    from outer.sandbox import make_runner
    from outer.score import CachedLookup
    from services.world.server import FlyWireBackend

    backend = FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
    return Evaluator(cfg, make_runner(cfg), CachedLookup(backend))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="the user approved spending up to budget_usd.total")
    ap.add_argument("--allow-scan-hits", action="store_true")
    ap.add_argument("--final", action="store_true", help="evaluate the best harness and h000 on test (once)")
    args = ap.parse_args(argv)
    cfg = load_config()
    from outer.sandbox import install_stop_handlers
    install_stop_handlers()  # Ctrl-C / kill also stops the containers
    if args.final:
        run_final(cfg, args.yes)
    else:
        run_rounds(cfg, args.yes, args.allow_scan_hits)


if __name__ == "__main__":
    main()
