# Plan: an outer harness that builds and improves a FlyWire proofreading harness

Version 1 · 2026-11-17

## 1. Summary

We build two layers.

- **Inner harness:** code that receives one task (a JSON state naming FlyWire segments at one
  moment in time) and returns one proofreading decision (a merge or a split, with locations).
  It may call an LLM, render images, analyze meshes, run several agents — whatever it needs.
  **We do not write it.** An LLM writes it.
- **Outer harness:** our code. It asks a coding agent (the **meta-agent**) to write the first
  inner harness, runs it on tasks, scores the results against real FlyWire edits, and asks the
  meta-agent to revise the harness. It keeps a revision only if it scores better on held-out
  tasks.

```
 round 0:  meta-agent ── writes ──► harnesses/h000
 round k:  evaluate hK on train + val ──► score ──► feedback (train only) ──► meta-agent ── writes ──► hK+1
           keep hK+1 if val improves beyond noise; otherwise revise again from the best version
 end:      evaluate the best harness and h000 on test, once
```

Ground truth comes from 200 FlyWire operations (100 merges, 100 splits) sampled by
[brychang/flywire-edit-sampling](https://github.com/brychang/flywire-edit-sampling).

Two services sit between generated code and the outside world, and they are the core of the
design:

- **`world`**: read-only FlyWire data, frozen at the task's moment in time. It is the only
  holder of the CAVE token.
- **`llm` proxy**: forwards Claude API calls, meters them, and enforces budgets. It is the only
  holder of the Anthropic API key.

## 2. Terms

| Term | Meaning |
|---|---|
| Operation | One FlyWire edit from `operations.csv`: a 2→1 merge or a 1→2 split. |
| Task | What the inner harness receives for one operation: a cleaned JSON state (Section 4). Task IDs are random and do not reveal the operation ID. |
| T | The task's time: operation timestamp − 1 ms (just before the edit), for both types. |
| Pre-edit roots | Segments valid at T. Merges: the 2 segments that were joined. Splits: the 1 segment that was cut. |
| Post-edit roots | Segments just after the edit (T + 2 ms). Used only for scoring splits. |
| Edit center | Mean of the operation's recorded source and sink coordinates, in nm (the same definition flywire-edit-sampling uses). |
| Harness `hK` | Version K of the inner harness: a folder of generated code under `harnesses/`. |
| Meta-agent | The coding agent that writes and revises harnesses. |
| Truth | Per-operation answers (edit center, sources, sinks, pre/post-edit roots). Only the scorer and the build scripts read it. |

## 3. Data

### 3.1 Source

- `data/operations.csv`: a copy of `outputs/operations.csv` from flywire-edit-sampling,
  commit `05f9f38535748517937f630c4ccbc281cb7fc8eb` (sha256 prefix `00a51c046bee2682`). The commit
  and hash are recorded in `config.yaml` and checked at startup.
- 200 operations: 100 merges and 100 splits. Both pre-edit merge segments, and both post-edit
  split segments, have ≥ 100 L2 chunks.
- Columns used: `operation_id`, `operation_type`, `timestamp_utc`, `before_root_ids`,
  `after_root_ids`. Root IDs are decimal strings and must stay strings (64-bit rounding).
- Caveat (from the source repo): operations are edits, not independently verified errors. A
  few may be wrong or undone later. We keep all 200 and accept this noise.

### 3.2 Splits

`data/splits.json`: 100 train / 50 val / 50 test, stratified by type (50/25/25 merges, same for
splits), fixed seed, frozen after creation.

- **train**: the meta-agent sees these tasks and their per-task results.
- **val**: used by the outer loop to accept or reject a revision. The meta-agent sees only
  each round's aggregate val hit rate and whether its revision was kept.
- **test**: run once at the end, on the final harness and on `h000`.

### 3.3 Truth (`build/build_truth.py`)

Run once with the CAVE token (`flywire_fafb_public`). For each operation it reads the
operation details and writes `truth/<operation_id>.json`:

- operation type, timestamp, T
- pre-edit and post-edit root IDs (checked against the CSV)
- source and sink coordinates in nm, and the edit center in nm

`truth/` is gitignored and never visible to harnesses or the meta-agent (Section 7).

## 4. The task: input state and output decision

### 4.1 Input: `tasks/<task_id>.json`

Built by `build/build_states.py` from the CSV plus `truth/`.

```json
{
  "state_version": "1",
  "task_id": "t_8c41e2a7",
  "timestamp_utc": "2022-11-14T01:46:39.753000+00:00",
  "neuroglancer": {
    "layers": [
      {"type": "image", "name": "em",
       "source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"},
      {"type": "segmentation", "name": "segments",
       "source": "<graphene segmentation source>",
       "segments": ["720575940610776440", "720575940624101559"],
       "timestamp": "1668390399.753000"}
    ]
  }
}
```

Rules:

- **Merges show both pre-edit segments. Splits show the single pre-edit segment.** The segment
  count therefore tells the harness the edit type; that is accepted.
- **No position hints.** No `navigation`, camera position, annotations, title, or operation ID.
  The harness must find the edit location on its own.
- Segment order is shuffled with the task seed.
- The task does not include anything after T.

### 4.2 Output: `decision.json`

All positions are in nm, (x, y, z) order.

```json
{"decision_version": "1", "action": "merge", "points_nm_xyz": [[x, y, z], [x, y, z]]}
{"decision_version": "1", "action": "split", "sources_nm_xyz": [[x, y, z]], "sinks_nm_xyz": [[x, y, z]]}
{"decision_version": "1", "action": "none"}
```

An optional `"notes"` string is stored but not scored. Every task has exactly one true edit;
there are no negative tasks.

## 5. Scoring (`outer/score.py`, `score_version` 1)

Hit radius **R = 2 µm** (`scoring.hit_radius_um`).

A point "lies on root X at time t" when the supervoxel at that point (base resolution of the
supervoxel layer) maps to root X at time t. A point on background (supervoxel 0) lies on no root.

- **Merge hit**: `action` is `merge`, there are exactly 2 points, one point lies on each of the
  two pre-edit roots at T, and **both points are within R of the edit center**.
- **Split hit**: `action` is `split`, there is ≥ 1 source and ≥ 1 sink, every point lies on the
  pre-edit root at T, all sources lie on one post-edit root and all sinks on the other at
  T + 2 ms, and **the midpoint of the closest source–sink pair is within R of the edit center**.
- Everything else is a miss: `none`, a wrong action, invalid JSON, a crash, a timeout, or an
  exhausted budget. The reason is recorded.

Reported per task: hit, miss reason, distance from the predicted location to the edit center
(merge: midpoint of the two points; split: midpoint of the closest source–sink pair), cost,
tokens, wall time.

Reported per run: hit rate overall and per type with 95% CIs, median distance, cost per task,
and counts of miss reasons.

Checks in `tests/`: an oracle harness that reads truth must score 100%, and a harness that
places points 5 µm away must score 0%.

## 6. Services

### 6.1 `world`: time-frozen, read-only FlyWire data (`services/world/`)

A small HTTP server on a node with internet access, plus a Python client library. Each task
gets a **session token bound to that task's T**; every query is answered as of T.

API (documented for the meta-agent in `services/world/API.md`):

| Call | Returns |
|---|---|
| `info()` | T, dataset bounds (nm), available resolutions, layer descriptions |
| `em(bbox_nm, resolution)` | uint8 EM cutout, indexed (z, y, x) |
| `supervoxels(bbox_nm, resolution)` | uint64 supervoxel cutout |
| `roots(supervoxel_ids)` | root ID of each supervoxel at T |
| `segmentation(bbox_nm, resolution)` | root ID per voxel at T (supervoxels mapped through `roots`) |
| `l2_graph(root_id)` | L2 chunk IDs and edges of a root valid at T |
| `l2_positions(l2_ids)` | approximate position in nm of each L2 chunk |
| `mesh(root_id)` | mesh of a root valid at T |

The world provides data primitives only. Skeletons, renderings, measurements and search
strategies are the harness's job.

Refused, always:

- lineage, operation records, edit history, "latest root", root creation and expiry times
- any query about a time other than T
- any root not valid at T
- any write (merge or split) to FlyWire

The server caches chunks on disk (`world.cache_dir`) and logs every request with its task
token. Unknowns to settle in Phase 2: mesh access for historical roots on
`flywire_fafb_public`, and whether L2 positions come from an L2 cache or must be computed from
supervoxel chunks.

### 6.2 `llm` proxy: the only holder of the API key (`services/llm/`)

An HTTP proxy for `POST /v1/messages` (streaming and non-streaming).

- Clients use the standard Anthropic SDK with `ANTHROPIC_BASE_URL` pointed at the proxy and
  `ANTHROPIC_API_KEY` set to a **run token** issued by the outer harness. The proxy swaps in the
  real key.
- Each run token carries an owner (meta-agent session, or harness + split + task), a dollar cap,
  and a model allowlist. A request that would go over the cap gets an error response.
- Every request and response is logged: `runs/.../transcript.jsonl` for its owner, and one line
  in the spend ledger (`ledger.jsonl`) with model, request ID, input / output / cache-write /
  cache-read tokens, and dollars computed from the price table in `config.yaml`.
- The ledger is the single source of truth for spending. `outer/loop.py` reads it before each
  step and stops when a budget would be exceeded.

## 7. Isolation and leak control

The CSV is public, and root IDs plus a timestamp identify an operation in it. FlyWire's history
endpoints reveal what happened after T. Generated code must therefore have no path to either.

- Harnesses and the meta-agent run inside an **Apptainer container** (`--containall`) on della
  compute nodes, which have no internet access. Only these are bound into the container:
  - for the meta-agent: its working copy of the harness, the train task files, the feedback
    packet, the world client library, and `API.md`;
  - for a harness run: the harness folder (read-only), one task file, and an empty work directory.
- Inside, the only reachable services are `world` and the `llm` proxy (on the vis node). The
  environment contains no CAVE token and no real API key.
- `truth/`, `data/operations.csv`, and the val and test task files are never bound into a
  container.
- The meta-agent gets world session tokens for train tasks only.
- `outer/` statically scans each new harness for network calls other than the two clients
  (sockets, `requests`, `urllib`, `subprocess curl`, …) and reports any hits to the user before
  evaluation.

Phase 4 must confirm that compute nodes can reach the vis node. If they cannot, use the fallback:
run the containers on the vis node with networking limited to the two services.

## 8. The outer harness (`outer/`)

### 8.1 Interface of a harness

```
harnesses/hK/
  harness.py      # def run(state: dict, world: WorldClient, llm: anthropic.Anthropic,
                  #         workdir: pathlib.Path, limits: Limits) -> dict  (a decision)
  NOTES.md        # written by the meta-agent: design, what it tried, what it learned
  ...             # anything else it creates: prompts, tools, renderers, sub-agents
```

`outer/evaluate.py` calls `run` once per task in a fresh container with a wall-time limit
(`limits.wall_s`) and a per-task dollar cap (enforced by the proxy). Runs are resumable: a task
with a finished `score.json` is skipped.

### 8.2 Round 0: `outer/create.py`

The meta-agent (Claude Agent SDK, `models.meta_agent`, no web tools) receives:

- `TASK.md`: the task, the decision schema, the scoring rule in words, the harness interface,
  and limits
- `API.md` and the world client library
- 3 example train task files
- `./score_train <task_ids>`: runs the harness under construction on those train tasks and
  prints hit/miss, distance, and cost per task. Spending comes from the session's probe budget
  (`budget_usd.meta_probe`).

It writes `harnesses/h000/`.

### 8.3 Round k: `outer/revise.py`

The meta-agent receives a copy of the current best harness and `feedback/round_k/`:

- per-task train results: hit/miss, miss reason, distance, cost, and the full transcript
- aggregate train and val hit rates for every past round, and which revisions were kept
- its own earlier `NOTES.md`

It writes `harnesses/hK+1/`. It never sees per-task val or test results.

### 8.4 The loop: `outer/loop.py`

Plain code, not an agent.

1. Check the budget in the ledger.
2. Create or revise a harness (Sections 8.2–8.3).
3. Evaluate it on train and val.
4. **Keep it** if its val hits exceed the best harness's by at least `loop.accept_min_gain`
   tasks (default 3 of 50, paired on the same tasks). Otherwise the next round revises from the
   best harness, and the feedback says the attempt was rejected and why.
5. Stop after `loop.max_rounds` rounds, after `loop.plateau_rounds` rounds in a row with no
   kept revision, or when the budget runs out.
6. At the end, evaluate the best harness and `h000` on test (after asking the user) and write
   the report.

`outer/report.py` writes `runs/report.md`: one row per round with train and val hit rates (by
type), cost, and kept/rejected, plus the final test comparison.

## 9. Repo layout

```
cos597_new/
  PLAN.md  CLAUDE.md  README.md  config.yaml  pyproject.toml  .gitignore
  data/
    operations.csv            # pinned copy (Section 3.1)
    splits.json               # train / val / test operation IDs, frozen
  build/                      # run once; needs the CAVE token; read-only toward FlyWire
    build_truth.py            # operations → truth/
    build_states.py           # operations + truth → tasks/ and the task-ID ↔ operation map
    make_splits.py            # → data/splits.json
  truth/                      # gitignored; read only by build/ and outer/score.py
  tasks/                      # cleaned task states (Section 4.1); task-ID map lives in truth/
  services/
    world/  server.py  client.py  API.md  cache/
    llm/    proxy.py  ledger.py
  outer/
    create.py  revise.py  evaluate.py  score.py  feedback.py  loop.py  report.py  sandbox.py
    prompts/                  # TASK.md and the meta-agent's instructions, versioned
  harnesses/                  # generated; one folder per version; tracked in git
  feedback/round_k/           # what the meta-agent saw each round
  runs/<harness>/<split>/<task_id>/   decision.json  transcript.jsonl  usage.json  score.json
  ledger.jsonl                # spend ledger (gitignored)
  tests/                      # fake world, fake Anthropic upstream, oracle harnesses; no network
```

## 10. `config.yaml` (initial values)

```yaml
data:
  operations_csv: data/operations.csv
  source_repo: https://github.com/brychang/flywire-edit-sampling
  source_commit: 05f9f38535748517937f630c4ccbc281cb7fc8eb
splits: {train: 100, val: 50, test: 50, seed: 0}
scoring: {hit_radius_um: 2.0}
models:
  meta_agent: claude-opus-5-5
  inner_allowed: [claude-opus-5-5]       # the proxy rejects other models
prices_usd_per_mtok:                     # cache write = 1.25 × input, cache read = 0.1 × input
  claude-opus-5-5: {input: 4.00, output: 20.00}
limits: {task_wall_s: 1800}
budget_usd:                              # placeholders until Phase 5 measures real costs
  per_task: 2.00
  meta_session: 30.00
  meta_probe: 20.00
  total: null                            # set by the user before Phase 6
  ask_above: 20.00                       # ask the user before any step expected to cost more
loop: {max_rounds: 10, accept_min_gain: 3, plateau_rounds: 3}
world: {host: della-vis1, port: null, cache_dir: services/world/cache}
proxy: {host: della-vis1, port: null}
```

## 11. Build phases

Each phase ends with passing tests and a short note in `PROGRESS.md`.

### Phase 0: Scaffold
`pyproject.toml`, `config.yaml`, `.gitignore`, `CLAUDE.md`, `git init`. Copy `operations.csv`
and check its hash. `build/make_splits.py` writes `data/splits.json`.

### Phase 1: Truth and tasks
`build_truth.py` and `build_states.py`. Try 5 operations first, then all 200. Checks: root
counts match the type, the CSV and operation details agree, edit centers fall within the
dataset bounds, and no task file contains an operation ID, coordinates, or a post-edit root.

### Phase 2: `world` service
Server, client, `API.md`, chunk cache. Tests against a fake backend: time binding, refusal of
history queries, refusal of roots not valid at T, and refusal of writes. Then a live check on
2 train tasks: EM and segmentation cutouts at a known location, and the L2 graph and mesh of a
historical root.

### Phase 3: `llm` proxy
Proxy and ledger, tested against a fake upstream: streaming and non-streaming, usage
accounting, cap enforcement, model allowlist, transcript logging.

### Phase 4: Evaluate and score
`sandbox.py` (Apptainer), `evaluate.py`, `score.py`. End-to-end with a fake LLM and fake world:
the oracle harness scores 100%, the offset harness 0%, and a hand-written trivial harness runs
on all train tasks. Confirm compute node → vis node connectivity, and confirm that `truth/` and
the CSV are not readable inside the container.

### Phase 5: Round 0 (first real API spend; ask the user first)
`create.py` and the meta-agent prompts. Build `h000`, run it on 5 train tasks, and report the
measured cost per task and per meta-agent session. Use these numbers to set the budgets and
`budget_usd.total` with the user.

### Phase 6: The loop
`revise.py`, `feedback.py`, `loop.py`, `report.py`. Run rounds within the approved budget.

### Phase 7: Final evaluation (ask the user first)
Run the best harness and `h000` on test once, then write the report.

## 12. Rules for Claude Code (copy into `CLAUDE.md`)

1. Only `build/`, `services/world/`, and `outer/score.py` talk to FlyWire, and only to read.
   Nothing ever performs a FlyWire edit.
2. Only `services/llm/proxy.py` holds the Anthropic API key. All Claude calls, the meta-agent's
   included, go through the proxy.
3. Only `build/` and `outer/score.py` read `truth/`. Nothing a harness or the meta-agent can see
   contains operation IDs, edit coordinates, post-edit roots, or val/test tasks.
4. Positions are in nm, (x, y, z) order, in all code and files. Arrays are indexed (z, y, x).
   Names end in `_nm_xyz` or `_zyx`.
5. Do not hand-edit generated harnesses under `harnesses/`. Fixes go into prompts, feedback, or
   the outer harness.
6. Tests never touch the network or the real API (fake world, fake upstream).
7. Ask the user before: any step expected to cost more than `budget_usd.ask_above`; running the
   test split; changing the splits, the scoring rule, or `R`; adding files outside the layout in
   Section 9.
8. Every file format carries a version string (`state_version`, `decision_version`,
   `score_version`, world API version, ledger version). Bump it on any change.
