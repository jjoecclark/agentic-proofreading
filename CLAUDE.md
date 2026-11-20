# CLAUDE.md

An outer harness that has a coding agent (the meta-agent) build and improve a FlyWire proofreading
harness. The design is PLAN.md; progress and decisions are in PROGRESS.md.

## Rules (PLAN.md Section 12)

1. Only `build/`, `services/world/`, and `outer/score.py` talk to FlyWire, and only to read.
   Nothing ever performs a FlyWire edit.
2. Only `services/llm/proxy.py` holds the Anthropic API key. All Claude calls, the meta-agent's
   included, go through the proxy.
3. Only `build/` and `outer/score.py` read `truth/`. Nothing a harness or the meta-agent can see
   contains operation IDs, edit coordinates, val/test tasks, or raw root/L2 ids (they see per-task
   aliases, `services/aliases.py`). An edit task never shows its own post-edit roots; a negative
   (no-edit) task shows proofread neurons (no glia) and never any edit task's roots.
4. Positions are in nm, (x, y, z) order, in all code and files. Arrays are indexed (z, y, x).
   Names end in `_nm_xyz` or `_zyx`.
5. Do not hand-edit generated harnesses under `harnesses/`. Fixes go into prompts, feedback, or
   the outer harness.
6. Tests never touch the network or the real API (fake world, fake upstream). Tests that read real
   FlyWire data are marked `live` and run only with `--live`; none of them call the Claude API.
7. Ask the user before: any step expected to cost more than `budget_usd.ask_above`; running the
   test split (except the per-round test runs of `outer/loop.py`, approved 2026-11-20: recorded only,
   never used to keep a revision, never shown to the meta-agent); changing the splits, the scoring rule, or `R`; adding files outside the layout in
   PLAN.md Section 9 (approved additions are listed in PROGRESS.md).
8. Every file format carries a version string (`state_version`, `decision_version`,
   `score_version`, world API version, ledger version, ...). Bump it on any change.

## Do not

- Read or reuse the older sibling project `../cos597/`. This project starts fresh.
- Put secrets in the repository. Secrets live in `~/.config/flywire_harness/` (Anthropic key, token
  secret) and `~/.cloudvolume/secrets/` (CAVE token); none of these is ever bound into a container.

## Working notes

- Python: `.venv/bin/python` (3.12, from `anaconda3/2024.10`). Tests: `.venv/bin/python -m pytest -q`
  (about 2.5 min; needs apptainer for the sandbox tests). Live checks: `.venv/bin/python -m pytest
  --live tests/test_live.py`. Lint: `.venv/bin/ruff check .`
- Sandbox image: `outer/sandbox.sif`, built from `outer/sandbox.def` with
  `apptainer build outer/sandbox.sif outer/sandbox.def` (about 1 min, on della-vis1).
- Services run on della-vis1 and listen on Unix sockets (`services/world/world.sock`,
  `services/llm/llm.sock`). Sandboxes run on della-vis1 too, with `--network none`; they reach the
  services only through those sockets (README.md).
- Every Claude call is in `ledger.jsonl`; `services.llm.ledger.summarize()` totals it.
