"""Build feedback/round_k/: what the meta-agent sees before revising (PLAN.md Section 8.3).

Contents:
- train/<task_id>/: per-task train results of the current best harness: score.json, decision.json,
  transcript.jsonl (+ media/), run.log (tail), error.json
- train_summary.json: the best harness's aggregate train results
- history.json and README.md: aggregate train and val hit rates (overall and by type) of every past
  round, its cost, and whether the revision was kept and why
- notes/<harness>_NOTES.md: every earlier NOTES.md
- rejected/<harness>/ (if the last revision was rejected): its code, train summary, per-task train scores

Never included: per-task val or test results, truth, operation IDs, edit coordinates.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from outer.config import Config
from outer.evaluate import task_ids
from outer.sandbox import copy_tree

FEEDBACK_VERSION = "1"
RUN_FILES = ("score.json", "decision.json", "transcript.jsonl", "error.json", "usage.json")
LOG_TAIL_BYTES = 200_000


def short(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Aggregate-only view of an evaluation summary (safe for val)."""
    if not summary:
        return None
    out = {k: {"n": summary[k]["n"], "hits": summary[k]["hits"], "hit_rate": summary[k]["hit_rate"]}
           for k in ("all", "edit", "no_edit", "merge", "split", "no_merge", "no_split") if k in summary}
    out["usd_total"] = summary.get("usd_total")
    return out


def copy_run(src: Path, dst: Path, files: tuple[str, ...] = RUN_FILES, with_media: bool = True) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    log = src / "run.log"
    if log.exists():
        data = log.read_bytes()
        (dst / "run.log").write_bytes(data[-LOG_TAIL_BYTES:])
    if with_media and (src / "media").exists():
        shutil.copytree(src / "media", dst / "media", dirs_exist_ok=True)


def history_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in state.get("rounds", []):
        rows.append({"round": r["round"], "harness": r.get("harness"), "kept": r.get("kept"),
                     "reason": r.get("reason"), "train": r.get("train"), "val": r.get("val"),
                     "val_paired": {k: r["gain"][k] for k in ("n", "new_hits", "best_hits", "gain")} if r.get("gain") else None,
                     "meta_usd": r.get("meta_usd"), "probe_usd": r.get("probe_usd"), "eval_usd": r.get("eval_usd")})
    return rows


def _rate(b: dict[str, Any] | None, k: str) -> str:
    if not b or not b.get(k) or b[k]["n"] == 0:
        return "-"
    return f"{b[k]['hits']}/{b[k]['n']}"


def readme(round_k: int, best: str, rows: list[dict[str, Any]], train_summary: dict[str, Any] | None,
           rejected: str | None) -> str:
    lines = [f"# Feedback for round {round_k}", "",
             f"`/work/harness/` is a copy of **{best}**, the best harness so far (highest validation hits).", "",
             "## Files", "",
             f"- `train/<task_id>/`: {best}'s result on each train task: `score.json` (hit, miss_reason, distance_nm,"
             " usd, wall_s), `decision.json`, `transcript.jsonl` (every Claude request and response; images in"
             " `media/`), `run.log` (stdout/stderr tail), `error.json` (if it crashed).",
             "- `train_summary.json`: aggregate train results of the best harness (hit rates with 95% CIs, median"
             " distance, cost, miss-reason counts).",
             "- `history.json`: every round so far (below).",
             "- `notes/`: NOTES.md of every earlier harness."]
    if rejected:
        lines.append(f"- `rejected/{rejected}/`: the last revision, which was **not kept**: its code, train summary,"
                     " and per-task train scores. Learn from it; the next harness starts again from the best.")
    lines += ["", "## History (validation: aggregate only)", "",
              "| round | harness | train all | train merge | train split | val all | val merge | val split | kept | why |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['round']} | {r['harness']} | {_rate(r['train'], 'all')} | {_rate(r['train'], 'merge')} | "
                     f"{_rate(r['train'], 'split')} | {_rate(r['val'], 'all')} | {_rate(r['val'], 'merge')} | "
                     f"{_rate(r['val'], 'split')} | {'yes' if r['kept'] else 'no'} | {r.get('reason') or ''} |")
    if train_summary:
        lines += ["", f"## {best} on train", "",
                  f"- miss reasons: {json.dumps(train_summary.get('miss_reasons'))}",
                  f"- median distance to the true location: {train_summary['all'].get('median_distance_nm')} nm",
                  f"- cost per task: ${(train_summary.get('usd_per_task') or 0):.3f}; median wall time "
                  f"{train_summary.get('median_wall_s')} s"]
    return "\n".join(lines) + "\n"


def build_feedback(cfg: Config, round_k: int, best: str, state: dict[str, Any], rejected: str | None) -> Path:
    d = cfg.root / "feedback" / f"round_{round_k:03d}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    runs = cfg.root / "runs"
    for t in task_ids(cfg, "train"):
        src = runs / best / "train" / t
        if src.exists():
            copy_run(src, d / "train" / t)
    ts = None
    if (runs / best / "train" / "summary.json").exists():
        ts = json.loads((runs / best / "train" / "summary.json").read_text())
        ts.pop("task_ids", None)
        (d / "train_summary.json").write_text(json.dumps(ts, indent=1) + "\n")
    rows = history_rows(state)
    (d / "history.json").write_text(json.dumps({"feedback_version": FEEDBACK_VERSION, "rounds": rows}, indent=1) + "\n")
    (d / "notes").mkdir()
    for r in state.get("rounds", []):
        n = cfg.root / "harnesses" / str(r.get("harness")) / "NOTES.md"
        if n.exists():
            shutil.copy(n, d / "notes" / f"{r['harness']}_NOTES.md")
    if rejected:
        rd = d / "rejected" / rejected
        copy_tree(cfg.root / "harnesses" / rejected, rd / "harness")
        rs = runs / rejected / "train" / "summary.json"
        if rs.exists():
            s = json.loads(rs.read_text())
            s.pop("task_ids", None)
            (rd / "train_summary.json").write_text(json.dumps(s, indent=1) + "\n")
        for t in task_ids(cfg, "train"):
            src = runs / rejected / "train" / t
            if src.exists():
                copy_run(src, rd / "train" / t, files=("score.json", "decision.json", "error.json"), with_media=False)
    (d / "README.md").write_text(readme(round_k, best, rows, ts, rejected))
    assert_no_val_leak(cfg, d)
    return d


def assert_no_val_leak(cfg: Config, d: Path) -> None:
    """No val/test task id may appear anywhere in the feedback packet."""
    held_out = set(task_ids(cfg, "val")) | set(task_ids(cfg, "test"))
    for p in d.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".log", ".txt", ""):
            text = p.read_text(errors="replace")
            hit = [t for t in held_out if t in text or t in str(p)]
            if hit:
                raise RuntimeError(f"feedback leak: {p} mentions held-out task(s) {hit[:3]}")
