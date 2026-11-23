"""Build feedback/round_k/: what the meta-agent sees before revising (PLAN.md Section 8.3).

Contents:
- train/<task_id>/: per-task train results of the current best harness: score.json, decision.json,
  transcript.jsonl (+ media/), run.log (tail), error.json
- train_summary.json: the best harness's aggregate train results
- history.json and README.md: aggregate train and val hit rates (overall and by type) of every past
  round, its cost, and whether the revision was kept and why
- notes/<harness>_NOTES.md: every earlier NOTES.md
- rejected/<harness>/ (if the last revision was rejected): its code, train summary, per-task train scores

Scores are sanitized (`meta_score`): hit, miss reason, kind, cost and time only. No distance to the true
location (four probes at known points would trilaterate it exactly; h003/h004 did) and no raw root ids.

Never included: per-task val results, any test result (not even aggregate: test runs every round for
the record, outer/loop.py), truth, operation IDs, edit coordinates.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from outer.config import Config
from outer.evaluate import task_ids
from outer.sandbox import copy_tree

FEEDBACK_VERSION = "3"  # 2: per-kind score; test never included; 3: no distances, sanitized scores
RUN_FILES = ("score.json", "decision.json", "transcript.jsonl", "error.json", "usage.json")
LOG_TAIL_BYTES = 200_000


# what the meta-agent may see of a task's score (no distance_nm, no point_roots_at_t: raw ids)
META_SCORE_KEYS = ("score_version", "task_id", "kind", "operation_type", "action", "hit", "miss_reason", "usd",
                   "tokens", "requests", "wall_s", "radius_nm")
SUMMARY_BLOCK_KEYS = ("n", "hits", "hit_rate", "ci95")


def meta_score(score: dict[str, Any]) -> dict[str, Any]:
    return {k: score[k] for k in META_SCORE_KEYS if k in score}


def meta_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """An evaluation summary without distances or task ids."""
    out = {}
    for k, v in summary.items():
        if k in ("task_ids", "median_distance_nm"):
            continue
        out[k] = {b: v[b] for b in SUMMARY_BLOCK_KEYS if b in v} if isinstance(v, dict) and "hits" in v else v
    return out


def short(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Aggregate-only view of an evaluation summary (safe for val)."""
    if not summary:
        return None
    out = {k: {"n": summary[k]["n"], "hits": summary[k]["hits"], "hit_rate": summary[k]["hit_rate"]}
           for k in ("all", "edit", "no_edit", "merge", "split", "no_merge", "no_split") if k in summary}
    out["score"] = summary.get("score")
    out["usd_total"] = summary.get("usd_total")
    return out


def copy_run(src: Path, dst: Path, files: tuple[str, ...] = RUN_FILES, with_media: bool = True) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f == "score.json" and (src / f).exists():
            (dst / f).write_text(json.dumps(meta_score(json.loads((src / f).read_text())), indent=1) + "\n")
        elif (src / f).exists():
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
        rows.append({"round": r["round"], "harness": r.get("harness"), "kept": r.get("kept"),  # never r["test"]
                     "reason": r.get("reason"), "train": r.get("train"), "val": r.get("val"),
                     "val_paired": {k: r["gain"].get(k) for k in ("n", "new_hits", "best_hits", "gain", "new_score",
                                                                  "best_score", "score_gain_pp")} if r.get("gain") else None,
                     "meta_usd": r.get("meta_usd"), "probe_usd": r.get("probe_usd"), "eval_usd": r.get("eval_usd")})
    return rows


def _rate(b: dict[str, Any] | None, k: str) -> str:
    if not b or not b.get(k) or b[k]["n"] == 0:
        return "-"
    return f"{b[k]['hits']}/{b[k]['n']}"


def _score(b: dict[str, Any] | None) -> str:
    return "-" if not b or b.get("score") is None else f"{100 * b['score']:.1f}%"


def readme(round_k: int, best: str, rows: list[dict[str, Any]], train_summary: dict[str, Any] | None,
           rejected: str | None) -> str:
    lines = [f"# Feedback for round {round_k}", "",
             f"`/work/harness/` is a copy of **{best}**, the best harness so far (highest validation score).", "",
             "Score = the mean of the hit rates of the four task kinds (merge, split, no_merge, no_split).", "",
             "## Files", "",
             f"- `train/<task_id>/`: {best}'s result on each train task: `score.json` (hit, miss_reason,"
             " usd, wall_s; no distance to the true location), `decision.json`,"
             " `transcript.jsonl` (every Claude request and response; images in"
             " `media/`), `run.log` (stdout/stderr tail), `error.json` (if it crashed).",
             "- `train_summary.json`: aggregate train results of the best harness (score, hit rates with 95% CIs, cost,"
             " miss-reason counts).",
             "- `history.json`: every round so far (below).",
             "- `notes/`: NOTES.md of every earlier harness."]
    if rejected:
        lines.append(f"- `rejected/{rejected}/`: the last revision, which was **not kept**: its code, train summary,"
                     " and per-task train scores. Learn from it; the next harness starts again from the best.")
    lines += ["", "## History (validation: aggregate only)", "",
              "| round | harness | train score | train merge | train split | train no_merge | train no_split | "
              "val score | val merge | val split | val no_merge | val no_split | kept | why |",
              "|" + "---|" * 14]
    for r in rows:
        tr, va = r["train"], r["val"]
        lines.append(f"| {r['round']} | {r['harness']} | {_score(tr)} | {_rate(tr, 'merge')} | {_rate(tr, 'split')} | "
                     f"{_rate(tr, 'no_merge')} | {_rate(tr, 'no_split')} | {_score(va)} | {_rate(va, 'merge')} | "
                     f"{_rate(va, 'split')} | {_rate(va, 'no_merge')} | {_rate(va, 'no_split')} | "
                     f"{'yes' if r['kept'] else 'no'} | {r.get('reason') or ''} |")
    if train_summary:
        lines += ["", f"## {best} on train", "",
                  f"- miss reasons: {json.dumps(train_summary.get('miss_reasons'))}",
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
        ts = meta_summary(json.loads((runs / best / "train" / "summary.json").read_text()))
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
            s = meta_summary(json.loads(rs.read_text()))
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
