"""Export the run's progress for the monitor dashboard (user request, 2026-11-22).

Reads only what the outer harness writes (never truth/): tasks/ (segments, time, split), runs/loop_state.json
(rounds, decisions, costs), runs/<harness>/<split>/<task_id>/score.json and decision.json (per-task results;
score.json carries the true location, written by outer/score.py), and runs/meta/*/session.json (a round still
being evaluated). Writes runs/dashboard/{tasks,results,rounds}.json and manifest.json (sha256 per file, so a
sync uploads only what changed).

The output holds raw root ids and true edit locations: it is for the user's own dashboard only, and is never
bound into a sandbox or copied into feedback.

    .venv/bin/python -m outer.dashboard_export
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from outer.config import Config, load_config

EXPORT_VERSION = "1"
SPLITS = ("train", "val", "test")


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def task_rows(root: Path) -> list[dict[str, Any]]:
    idx = json.loads((root / "tasks" / "index.json").read_text())
    rows = []
    for split in SPLITS:
        for t in idx[split]:
            seg = json.loads((root / "tasks" / f"{t}.json").read_text())["neuroglancer"]["layers"][1]
            rows.append({"task_id": t, "split": split, "segments": " ".join(seg["segments"]), "t_unix": seg["timestamp"]})
    return rows


def predicted_points(decision: Any) -> list[list[float]]:
    if not isinstance(decision, dict):
        return []
    pts = decision.get("points_nm_xyz") or (decision.get("sources_nm_xyz") or []) + (decision.get("sinks_nm_xyz") or [])
    out = []
    for p in pts if isinstance(pts, list) else []:
        try:
            out.append([round(float(v)) for v in p][:3])
        except (TypeError, ValueError):
            continue
    return [p for p in out if len(p) == 3]


def round_list(root: Path) -> list[dict[str, Any]]:
    """Recorded rounds from the loop state, then any harness being evaluated that the state doesn't list yet."""
    st = _read(root / "runs" / "loop_state.json") or {"rounds": []}
    rounds = []
    for r in st.get("rounds", []):
        if not r.get("harness"):
            continue
        rounds.append({"round": r["round"], "harness": r["harness"], "kept": "kept" if r.get("kept") else "rejected",
                       "reason": r.get("reason") or "", "meta_usd": round(r.get("meta_usd") or 0.0, 4),
                       "probe_usd": round(r.get("probe_usd") or 0.0, 4), "eval_usd": round(r.get("eval_usd") or 0.0, 4)})
    known = {r["harness"] for r in rounds}
    for h in sorted(p.name for p in (root / "harnesses").glob("h[0-9][0-9][0-9]") if p.is_dir()):
        if h in known or not (root / "runs" / h).exists():
            continue
        k = int(h[1:])
        sess = [s for s in (_read(p) for p in sorted((root / "runs" / "meta").glob(f"round_{k:03d}*/session.json")))
                if s and s.get("harness") == h]
        s = sess[-1] if sess else {}
        rounds.append({"round": k, "harness": h, "kept": "evaluating", "reason": "being evaluated",
                       "meta_usd": round(s.get("meta_usd") or 0.0, 4), "probe_usd": round(s.get("probe_usd") or 0.0, 4),
                       "eval_usd": None})
    return sorted(rounds, key=lambda r: r["round"])


def result_rows(root: Path, rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = json.loads((root / "tasks" / "index.json").read_text())
    rows = []
    for r in rounds:
        spent = 0.0
        for split in SPLITS:
            for t in idx[split]:
                d = root / "runs" / r["harness"] / split / t
                sc = _read(d / "score.json")
                if sc is None:
                    continue
                loc = sc.get("true_location_nm_xyz") or [None, None, None]
                spent += float(sc.get("usd") or 0.0)
                rows.append({"round": r["round"], "harness": r["harness"], "task_id": t, "split": split,
                             "kind": sc.get("kind"), "hit": int(bool(sc.get("hit"))),
                             "action": sc.get("action") or "(no decision)", "miss_reason": sc.get("miss_reason") or "",
                             "pred_nm_xyz": json.dumps(predicted_points(_read(d / "decision.json"))),
                             "x_nm": None if loc[0] is None else round(loc[0]),
                             "y_nm": None if loc[1] is None else round(loc[1]),
                             "z_nm": None if loc[2] is None else round(loc[2]),
                             "usd": round(float(sc.get("usd") or 0.0), 4), "wall_s": sc.get("wall_s")})
        if r["eval_usd"] is None:  # a round still being evaluated: its spend so far
            r["eval_usd"] = round(spent, 4)
    return rows


def export(cfg: Config) -> dict[str, Any]:
    root = cfg.root
    out = root / "runs" / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    rounds = round_list(root)
    files = {"tasks": task_rows(root), "results": result_rows(root, rounds), "rounds": rounds}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest: dict[str, Any] = {"export_version": EXPORT_VERSION, "exported_at": now, "files": {}}
    for name, rows in files.items():
        text = json.dumps(rows, separators=(",", ":"))
        path = out / f"{name}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)
        manifest["files"][name] = {"path": str(path.relative_to(root)), "rows": len(rows),
                                   "sha256": hashlib.sha256(text.encode()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def changed(cfg: Config, manifest: dict[str, Any]) -> list[str]:
    """Files whose content differs from what runs/dashboard/synced.json says was last uploaded to the
    dashboard (that file is written by whoever uploads: {name: {sha256, blob_id, dataset_version}})."""
    synced = _read(cfg.root / "runs" / "dashboard" / "synced.json") or {}
    return [k for k, v in manifest["files"].items() if (synced.get(k) or {}).get("sha256") != v["sha256"]]


def main() -> None:
    cfg = load_config()
    m = export(cfg)
    print(json.dumps({"exported_at": m["exported_at"], "changed": changed(cfg, m),
                      **{k: {"rows": v["rows"], "sha256": v["sha256"][:12]} for k, v in m["files"].items()}}))


if __name__ == "__main__":
    main()
