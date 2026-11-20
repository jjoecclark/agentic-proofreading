"""Write runs/report.md: one row per round (train, val and test scores, hit rates by kind, cost,
kept/rejected) and the final test comparison (PLAN.md Section 8.4).

    .venv/bin/python -m outer.report
"""

from __future__ import annotations

from typing import Any

from outer.config import Config, load_config
from services.llm.ledger import summarize as ledger_summary

REPORT_VERSION = "3"  # 2: no-edit (negative) task columns; 3: per-kind score, test every round


def _r(b: dict[str, Any] | None, k: str) -> str:
    if not b or not b.get(k) or not b[k]["n"]:
        return "-"
    return f"{b[k]['hits']}/{b[k]['n']} ({100 * b[k]['hits'] / b[k]['n']:.0f}%)"


def _ci(b: dict[str, Any], k: str) -> str:
    lo, hi = b[k].get("ci95") or (None, None)
    return "-" if lo is None else f"{100 * lo:.0f}-{100 * hi:.0f}%"


def _s(b: dict[str, Any] | None) -> str:
    return "-" if not b or b.get("score") is None else f"{100 * b['score']:.1f}%"


def write_report(cfg: Config, st: dict[str, Any]) -> str:
    led = ledger_summary(cfg.root / "ledger.jsonl")
    lines = ["# Outer-harness report", "", f"report_version {REPORT_VERSION}. Best harness: **{st.get('best')}**. "
             f"Total spend in the ledger: **${led['usd']:.2f}** ({led['requests']} requests).", "",
             "## Rounds", "",
             "Score = mean of the per-kind hit rates (merge, split, no-merge, no-split). Test is run every round "
             "for the record only: it never decides what is kept and the meta-agent never sees it.", "",
             "| round | harness | train score | val score | val merge | val split | val no-merge | val no-split | "
             "test score | test merge | test split | test no-merge | test no-split | meta $ | probe $ | eval $ | kept | why |",
             "|" + "---|" * 18]
    for r in st.get("rounds", []):
        va, te = r.get("val"), r.get("test")
        lines.append(f"| {r['round']} | {r.get('harness') or '-'} | {_s(r.get('train'))} | {_s(va)} | "
                     f"{_r(va, 'merge')} | {_r(va, 'split')} | {_r(va, 'no_merge')} | {_r(va, 'no_split')} | "
                     f"{_s(te)} | {_r(te, 'merge')} | {_r(te, 'split')} | {_r(te, 'no_merge')} | {_r(te, 'no_split')} | "
                     f"{r.get('meta_usd', 0):.2f} | {r.get('probe_usd', 0):.2f} | {(r.get('eval_usd') or 0):.2f} | "
                     f"{'yes' if r.get('kept') else 'no'} | {r.get('reason') or ''} |")
    if st.get("final"):
        lines += ["", "## Final test (run once)", "",
                  "| harness | test score | test all | 95% CI | merge | split | no-merge | no-split | median distance | $/task | "
                  "misses |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for h, s in st["final"].items():
            md = s["all"].get("median_distance_nm")
            lines.append(f"| {h} | {_s(s)} | {_r(s, 'all')} | {_ci(s, 'all')} | {_r(s, 'merge')} | {_r(s, 'split')} | "
                         f"{_r(s, 'no_merge')} | {_r(s, 'no_split')} | "
                         f"{'-' if md is None else f'{md / 1000:.2f} µm'} | {(s.get('usd_per_task') or 0):.3f} | "
                         f"{s.get('miss_reasons')} |")
    text = "\n".join(lines) + "\n"
    out = cfg.root / "runs" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


def main() -> None:
    from outer.loop import load_state

    cfg = load_config()
    print(write_report(cfg, load_state(cfg)))


if __name__ == "__main__":
    main()
