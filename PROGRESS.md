# Progress

| Phase | Status | Notes |
|---|---|---|
| 0. Scaffold | done | CSV pinned (sha256 `00a51c046bee2682…`), splits 100/50/50 stratified, seed 0 |
| 1. Truth and tasks | done | 200 edits (operations_v3) + 800 proofread-neuron negatives; 1000 task files pass the leak checks |
| 2. `world` service | done | fake-backend tests + live check on 2 train tasks |
| 3. `llm` proxy | done | fake-upstream tests: streaming, usage, caps, allowlist, transcripts |
| 4. Evaluate and score | done | oracle 100% / offset 0% / trivial harness on all train tasks, in Apptainer; real-data oracle 95/100 |
| 5. Round 0 | done | h000: train 42/100 (merge 41/50, split 1/50); round cost $6.48 |
| 6. The loop | code done, dry run passed | waiting for `budget_usd.total` |
| 7. Final evaluation | code ready | not run |

Tests: `.venv/bin/python -m pytest -q` → 72 passed, 5 skipped (live); `--live` → 5 passed (2026-11-20).

## Phase 0 (2026-11-17)

- `data/operations.csv` copied from flywire-edit-sampling at commit `05f9f38…`; hash checked at
  every `load_config()`. `build/make_splits.py` → `data/splits.json` (frozen; `--force` needed).
- Host Python: `.venv` from `anaconda3/2024.10` (Python 3.12; matches the sandbox image).

## Phase 1

- `build/build_truth.py`: operation details in batches of 40; pre-edit roots at T = op − 1 ms and
  post-edit roots at T + 2 ms, both checked against the CSV and the operation record (all 200 agree).
  Coordinates: recorded source/sink coordinates are in 16 × 16 × 40 nm segmentation voxels; edit
  center = mean of all source and sink points × that resolution. Matches the sampling repo's
  `edit_center_nm` (checked on 2 operations).
- **Finding: two 2019 operations (1703 split, 4557 merge) store float64 coordinates as raw 64-bit
  integers** (~4.7e18). The sampling repo's center for them is garbage. `build_truth` decodes them
  back to floats (`coords_decoded_from_float64_bits: true`); the decoded points land in the recorded
  supervoxels' chunks (one exact supervoxel match). Without this, those 2 tasks would be unscoreable.
- Coordinate check (supervoxel under each recorded point at base resolution): all points on the
  pre-edit roots for 66/100 merges and 75/100 splits. Recorded clicks often sit a few voxels off
  the segment, which is why the real-data oracle snaps to the edited supervoxels.
- Merge clicks are close together (median 0.45 µm apart) but 13/100 are > 2 µm apart and 4/100
  > 4 µm. For those few, the center (the midpoint) may be > 2 µm from one segment: a likely
  unhittable task under the scoring rule. This is part of the accepted noise.
- `build/build_states.py`: random task IDs (`t_` + 8 hex) and shuffle seeds in `truth/task_map.json`;
  `tasks/index.json` lists task IDs per split (so `outer/` never needs truth/ to find a split).

## Phase 2

- Unknowns settled (live, `flywire_fafb_public`): meshes of historical roots work (graphene
  manifest); `level2_chunk_graph` works for historical roots; **there is no L2 cache**, so
  `l2_positions` = centroid of each L2 node's own mesh fragment (~97% have one; the rest get their
  graph chunk's center with `approximate=True`). 5,164 fragments fetch in ~7 s; cached in SQLite.
- EM: the 4 nm level has missing chunks (read as 0); 8 nm and coarser are fine. Documented in API.md.
- Root validity at T = created ≤ T and `is_latest_roots(timestamp=T)`.
- `l2_positions` accepts only L2 ids that `l2_graph` returned for the same task: otherwise a harness
  could probe for L2 ids created by the future edit.
- Refused: unknown endpoints (lineage, operations, latest roots, timestamps, merge/split, ...), any
  non-POST method, any timestamp-like field anywhere in a request, roots not valid at T. Every
  request is logged with its task (`runs/world_requests.jsonl`), before the reply is sent.

## Phase 3

- Run tokens and world tokens are HMAC-signed (`services/tokens.py`); the secret never enters a
  container. Run token = owner, dollar cap, model allowlist, optional group + group cap (the
  meta-agent's probe budget), and the transcript directory.
- Cap rule: a request is refused (HTTP 402, `budget_exceeded`) if spent + in-flight reservations +
  its worst case (estimated input at the 5-minute cache-write price + `max_tokens` × output price)
  would exceed the cap. So the cap is never exceeded, even with concurrent requests.
- Also refused, to keep the sandbox off the web and the model allowlist tight: server-side tools
  (web search/fetch, code execution, ...), `mcp_servers`, `container`, `fallbacks`, `inference_geo`,
  `service_tier`, fast mode.
- Prices: explicit per field in config.yaml. **PLAN.md's "cache read = 0.1 × input" is outdated for
  Opus 5.5: cache reads are $0.20/M (0.05 ×)**; 1-hour cache writes are 2 × input.
- Transcripts: base64 images are saved to `media/` and referenced; a request that continues the
  previous conversation stores only its new messages (`messages_from`).

## Phase 4

- **Sandbox (Section 7, vis-node variant, chosen deliberately):** containers run on della-vis1 with
  `apptainer exec --containall --net --network none`: no interface but loopback, so no internet and
  no cluster network. The two services are bound in as Unix sockets; a relay inside exposes the
  proxy on 127.0.0.1 for the SDK and the Claude Code CLI. This is stricter than the compute-node
  design and needs no Slurm queue (the `cpu` partition had 0 idle nodes). Compute → vis-node
  connectivity was therefore not tested; it is not needed.
- Verified in the container: truth/, the CSV, tasks/index.json, config.yaml, the ledger, the venv,
  the CAVE secret, the API key file, and the token secret are unreadable; TCP connects fail; the
  only credential in the environment is the run token.
- `evaluate.py`: one fresh container per task, resumable (finished `score.json` is skipped; an
  unfinished task is rerun under a new attempt owner), parallel (`limits.parallel_tasks`, 8).
- Real-data check (`tests/test_live.py`, no Claude calls): an oracle that snaps to the edited
  supervoxels near the true center hits **95/100 train tasks** (2 misses are the wide-click merges
  above; 3 have the edited supervoxels > 3 µm from the center).

## Phases 5-6 (code; dry run only)

- `create.py` / `revise.py`: the meta-agent is the Claude Agent SDK 0.2.183 (bundled Claude Code
  CLI) in the sandbox, tools Read/Write/Edit/Bash/Glob/Grep/TodoWrite, no web tools, model
  `claude-opus-5-5` for every role, `bypassPermissions` inside the container, 4 h wall limit.
- `score_train` is file-based: it snapshots `/work/harness` into `/probe/requests/`; the host
  evaluates it on the requested train tasks in fresh containers (probe budget = group cap) and
  writes the results back. No extra network service.
- `feedback.py` checks that no val/test task ID appears anywhere in the packet.
- Dry run (`tests/test_meta.py`): the real CLI in the sandbox, driven by a scripted fake upstream,
  wrote h000, ran `score_train`, the loop evaluated h000 on train and val, built feedback, ran a
  revise round, and rejected the unchanged h001. Zero API spend.
- Claude Code requests `max_tokens` = 128000 (worst case $2.56). Fine for the meta-agent's $30
  cap; TASK.md tells harness authors to set `CLAUDE_CODE_MAX_OUTPUT_TOKENS` if they embed it.

## Phase 5 result (2026-11-17)

- `outer.create --yes` → h000 in 40 min: meta $2.78 (54 requests), probes $3.70. h000 on all 100
  train tasks (its last probe, identical code): **42/100, merge 41/50, split 1/50**, $0.83 total
  (~$0.017 per split task, $0 per merge task), max 71 s per task. No scan hits, no world refusals.
- Splits: mean distance 53.5 µm (median 41 µm) from the true edit. In 22/50 split tasks h000's
  Claude call stops at its own `max_tokens=600` (thinking uses part of it), the JSON never closes,
  and h000 silently falls back to its heuristic. A harness bug for the meta-agent to fix (rule 5).
- Session folders are ~0.9 GB, almost all the meta-agent's own `/work/.tmp` (cached features).

## Outer-harness fixes for long runs (2026-11-17)

- The loop evaluates a round whose meta session already finished (`finished_session`) instead of
  starting it again: round 0 run by `outer.create`, or a loop stopped during evaluation.
- Ctrl-C, SIGTERM and SIGHUP stop the containers too (`install_stop_handlers`); before, containers
  ran on in their own process groups (a meta session for up to 4 h). An interrupted task records
  nothing and reruns on resume.
- If the world service or the proxy stops, evaluation stops and records nothing (`ServicesDown`),
  instead of scoring every task as the harness's crash; evaluation refuses to start without them.

## Negative tasks, opaque ids, proofreading guide (2026-11-17, user request)

- **Negative (no-edit) tasks** (`build/build_negatives.py`, `truth/negatives/`): each operation shown
  just after its edit (T_post = edit time + 1 ms), with right answer `none`. A merge gives a `no_split`
  task (the merged segment), a split a `no_merge` task (the two pieces). Each goes to its source
  operation's split, so splits are unchanged. Checked against the public materialization 783 (the
  published connectome; all 200 operations are older):
  - the edit survived: 93/100 merges, 83/100 splits. **So 7 merge and 17 split edit tasks are edits
    that were later undone or redone** (possibly wrong labels; left unchanged, for the user to decide);
  - `no_split` also needs nothing split off the segment later (all 20,000 sampled supervoxels still on
    one root): only 31 of 93 qualify (50 at >= 99%, 69 at >= 95%). The rest still held a false merge,
    so "no split needed" would be a wrong label. `no_merge`: 83 (all surviving splits).
  - (Before the split replacement below:) 114 negatives: train 51 (13 no_split + 38 no_merge), val 33 (11 + 22), test 30 (7 + 23). Task
    files: train 151, val 83, test 80 (`tasks/index.json` lists the negatives under `negative`).
- **Scoring (score_version 2):** a negative is a hit only for `none`; any edit is `edit_not_needed`.
  Every task counts the same; the keep rule uses total val hits (now over 83 tasks). Summaries and the
  report break results down by kind (`merge`, `split`, `no_merge`, `no_split`).
- **Opaque ids (world API 2, `services/aliases.py`):** FlyWire numbers roots and L2 nodes from
  counters, so raw ids reveal creation time. The two pieces of a split have ids 256 apart in 54/100
  operations (pre-merge pairs: 4/100), so a harness could answer most `no_merge` tasks from the ids
  alone. Every task now sees root and L2 ids through a keyed 64-bit permutation (per task, per id kind);
  the outer harness hands harnesses and the meta-agent aliased task states. Supervoxel ids are
  unchanged (no edit time in them).
- **Proofreading guide:** the user's "Fly Connectome Proofreading — Base Instructions for Agents"
  transcribed to `outer/prompts/proofreading_guide.md`, with a note on top saying what applies here
  (no edits, frozen time, only the world API's data, opaque ids, one decision per task). Copied to
  `/work/docs/` for the meta-agent; TASK.md (version 2, with a "Changes" section) and both meta
  prompts point to it. The meta-agent decides what to pass on to the inner harness.
- h000 was written for task version 1 and never answers `none`; round 1 sees the new TASK.md.
- CLAUDE.md rule 3 reworded: negatives show post-edit roots by design (never next to the same edit's
  pre-edit roots), and nobody outside the services sees raw root or L2 ids.

## Splits replaced: 100 surviving splits (2026-11-18, user request)

- The 17 split operations whose edit did not survive to materialization 783 (sources and sinks back
  on one root; e.g. 407653, re-merged by the same proofreader 25 minutes later) are replaced by the
  next surviving splits of the source's own sampler (`build/replace_splits.py`). It replays
  flywire-edit-sampling's draw (seed 42, IDs 1..3,000,000, same eligibility filter: one root before,
  the two operation roots after, >= 100 L2 chunks each), checks that the first 100 eligible splits
  are exactly the original 100 (they are, after 13,024 IDs), and continues to 100 surviving ones
  (14,784 IDs, 118 eligible, 1 more non-survivor skipped). All 17 build clean truth records.
- Each replacement takes the replaced split's slot (train 12, val 3, test 2), so split sizes,
  stratification and the merges are unchanged; the other 183 tasks keep their task IDs.
  `data/operations_v2.csv` (sha256 `43828137…`) is the new configured CSV; `data/operations.csv` is the
  untouched source copy. `data/splits.json` version 2 records the mapping and the sampler settings.
- Merges are unchanged: 7 of 100 did not survive (not replaced; ask the user).
- Now: 131 negatives (all 100 splits give a `no_merge` task; 31 `no_split`). Task files: train 163
  (100 edit + 50 no_merge + 13 no_split), val 86 (50 + 25 + 11), test 82 (50 + 25 + 7).
- Checks: the real-data snap oracle hits 159/163 train tasks (merge 47/50, split 49/50, no-edit
  63/63); offline tests 67 passed; live tests 5 passed.

## Proofread-neuron negatives, per-kind score, test every round, glia out (2026-11-20, user request)

- **Old negatives removed** (negative_version 1: the state just after an edit). **New negatives**
  (`build/build_negatives.py`, negative_version 2), from `proofread_neurons` at materialization 783:
  - `no_merge` (400): two proofread neurons sharing >= 5 synapses (A->B + B->A, `synapses_nt_v1`,
    cleft_score > 50, no autapses); median 8 synapses (range 5-176).
  - `no_split` (400): one proofread neuron.
  - Glia out: every proofread neuron has a neuron super_class (none is glia); 137 more tagged "not a
    neuron" / glia / trachea in `neuron_information_v2` are dropped. Also dropped: the 305 neurons
    holding one of the 200 edits, and every edit root. Each neuron used once (no neuron in two splits).
  - **Dating:** T = a random edit time of the matching type (merge times for no_merge, split times for
    no_split) +- 14 days, strictly inside the edit range, in whole milliseconds; only neurons whose 783
    root existed before T. Quantiles match the edit tasks (medians: no_merge 2022-01 vs merge 2021-09;
    no_split 2022-02 vs split 2022-01). **Leak found and fixed:** CAVE edit times are whole
    milliseconds, so the first build's microsecond digits marked 790/800 negatives; `build_states`
    now refuses finer-than-ms negative times.
  - Splits (user): 200 / 100 / 100 per kind. Tasks: train 500 (100 edit + 400 negative), val 250
    (50 + 200), test 250 (50 + 200). index_version 3, task_map_version 3.
  - **Size cue (for the user):** sampled L2-chunk medians: split 667 vs no_split 141; merge larger
    segment 645 vs no_merge 287. Negatives are smaller, so "big single segment -> split" is a possible
    shortcut (edit segments were sampled with >= 100 L2 chunks per side, and a false merge holds two
    cells). Not corrected.
- **Glia in edit tasks:** 16 of the 200 edits sat on a root tagged not a neuron / glia / trachea at 783
  (2 merges of two such pieces; 14 splits cutting one off a neuron). Replaced (user) by
  `build/replace_edits.py`: replays the source sampler for merges and splits (reproduces the original
  100 + 100 exactly), takes the next eligible operations that are clean (splits: also survived), in
  each replaced one's slot. `data/operations_v3.csv` (sha256 `cc5a2dad…`), `data/splits.json`
  version 3. Now 0 edits on a tagged root; 2 (632411, 1715495) sit on roots with no annotation at all
  (untagged fragments; left in).
- **Scoring (score_version 3):** score = mean of the per-kind hit rates (merge, split, no_merge,
  no_split); "always none" = 50%. Keep rule (loop_version 2): val score gain >= 2 pp
  (`loop.accept_min_gain_pp`), paired on the same tasks.
- **Test every round** (`loop.evaluate_test_every_round`, user): every evaluated harness also runs on
  test; recorded in loop_state and report.md only. `decide()` reads val only; feedback (version 2)
  carries no test result, even aggregate (tested). `--final` reuses the per-round test runs.
  CLAUDE.md rule 7 and PLAN.md updated.
- TASK.md version 3 (new negatives described without the selection rule; per-kind score);
  meta_revise.md version 3; config_version 2; report_version 3.
- h001-h004 and runs/ are from the old tasks; not re-evaluated (user: optimization restarts from
  scratch on the new data). Move `runs/loop_state.json` (and old `runs/h*`) aside before restarting.
- Checks: offline tests 72 passed; live tests 5 passed (snap oracle 40/40 on the first 40 train tasks).

## Files added beyond PLAN.md Section 9 (for the user to confirm)

`PROGRESS.md`; `__init__.py` files; `outer/config.py` (config loading, CSV hash check, token secret);
`outer/sandbox.def` + `outer/sandbox.sif` (gitignored); `outer/prompts/{TASK.md, meta_create.md,
meta_revise.md}`; `services/tokens.py`; `services/httpserver.py`; `tasks/index.json`;
`truth/task_map.json`; `tests/{conftest.py, fakes.py, harnesses/}`; `.venv/` (gitignored);
`runs/loop_state.json`, `runs/world_requests.jsonl`, `runs/live_oracle_check.json`;
`build/build_negatives.py`, `build/replace_splits.py`, `data/operations_v2.csv`, `truth/negatives/`,
`services/aliases.py`,
`outer/prompts/proofreading_guide.md` (negatives, opaque ids, the guide);
`build/replace_edits.py`, `data/operations_v3.csv`, `truth/negatives/synapse_cache.json`,
`runs/build_negatives.log`, `runs/replace_edits.log` (2026-11-20).
New config keys: `config_version`, `data.source_path`, `data.sha256_prefix`, `flywire.*`,
explicit cache prices, `limits.parallel_tasks`, `limits.meta_wall_s`, `world.socket`,
`world.max_cutout_voxels`, `world.max_roots_per_call`, `proxy.socket`, `proxy.upstream`,
`proxy.api_key_file`, `secrets.*`, `sandbox.*`, `negatives.*`, `loop.accept_min_gain_pp`,
`loop.evaluate_test_every_round`.

## Open items for the user

1. Put the Anthropic API key in `~/.config/flywire_harness/anthropic_api_key` (README.md).
2. Approve Phase 5 (`outer.create --yes`, worst case $50), then measure cost per task on 5 train tasks.
3. Set `budget_usd.total` before Phase 6.
4. Confirm the vis-node sandbox (8 parallel containers on della-vis1) is acceptable on a shared node.
