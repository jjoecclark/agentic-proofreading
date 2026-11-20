"""Round k >= 1 (PLAN.md Section 8.3): the meta-agent revises the best harness into harnesses/hK.

It receives a copy of the current best harness and feedback/round_k/ (outer/feedback.py). It never
sees per-task val or test results. Normally called by outer/loop.py; runnable alone:

    .venv/bin/python -m outer.revise --yes      # next round, from runs/loop_state.json
"""

from __future__ import annotations

import argparse
from typing import Any

from outer.config import ROOT, Config, load_config
from outer.create import fill, print_result, run_meta_session, worst_case_usd
from outer.evaluate import task_ids
from outer.feedback import build_feedback


def revise(cfg: Config, round_k: int, state: dict[str, Any], **kw: Any) -> dict[str, Any]:
    best = state["best"]
    last = state["rounds"][-1] if state.get("rounds") else None
    rejected = last["harness"] if last and not last.get("kept") and last.get("harness") and last.get("evaluated") else None
    fb = build_feedback(cfg, round_k, best, state, rejected)
    tmpl = (ROOT / "outer" / "prompts" / "meta_revise.md").read_text().split("-->", 1)[-1].strip()
    prompt = fill(tmpl, round=round_k, best=best, accept_min_gain_pp=cfg["loop"]["accept_min_gain_pp"],
                         n_val=len(task_ids(cfg, "val")))
    return run_meta_session(cfg, round_k, f"h{round_k:03d}", prompt, start_harness=cfg.root / "harnesses" / best,
                            feedback_dir=fb, **kw)


def main(argv: list[str] | None = None) -> None:
    from outer.loop import load_state

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    from outer.sandbox import install_stop_handlers
    install_stop_handlers()  # Ctrl-C / kill also stops the containers
    state = load_state(cfg)
    if not state.get("best"):
        raise SystemExit("no best harness yet; run outer.create / outer.loop first")
    if worst_case_usd(cfg) > float(cfg["budget_usd"]["ask_above"]) and not args.yes:
        raise SystemExit(f"worst case ${worst_case_usd(cfg):.2f} > ask_above; rerun with --yes once approved")
    print_result(revise(cfg, len(state["rounds"]), state))


if __name__ == "__main__":
    main()
