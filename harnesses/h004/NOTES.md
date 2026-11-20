# FlyWire proofreading harness — NOTES (round 4, on top of h003)

## Round 4 change (merge_verify.py only)
- **Verifier v3: wide-context mesh view.** The LLM now also gets image 0: A (red) / B (blue) meshes in a
  40 um cube around the candidate (`WIDE_HALF = 20000`; the 12 um inner cube outlined), plus a prompt
  paragraph on reading it (would A+B form one plausible neuron piece; short fragment ending at the contact
  vs two segments both extending far in their own directions). Render failure falls back to the old
  two-image prompt; facts/blank bonus are unaffected. Decision rule and all constants unchanged.
- Cost ~$0.08 per LLM-zone merge task (was ~$0.06–0.08); splits still `none`.

## Round 4 measurements
- 52 merge-side train tasks that reach the LLM (VERIFY_LO <= score+bonus <= VERIFY_HI; the other 48 are
  deterministic and unchanged: 41/48 hits):
  | verifier | run | hits /52 |
  |---|---|---|
  | h003 v2 (feedback run) | — | 38 |
  | v3 (req 032842_c72c1c) | 1 | 39 |
  | v3 (req 033442_e20bf2) | 2 | 39 (same 13 misses as run 1) |
  => merge side ~80/100, train total ~93/163 (h003: 92). Within noise of h003, but run-to-run stable.
- AUC of p_same (merge vs no_merge, 52 LLM-zone tasks): v2 0.81, v3 0.83, avg of both 0.83; score alone 0.69.
  v3 p is lower overall: it rejects no_merges better (21f1861c .43->.20, d0bc5ba7 .57->.16, 527c26f5,
  f8293fde, 12c97180) but also lowers some true merges (35aa2f95 .70->.41, 741f6348, 25d17bff).
- **Decision rule is saturated:** sweeping P_W in {4,6,8,10} and COMB_TH in [-1,1] gives 36–39/52 everywhere
  (dev/round4/fit2.py). The remaining misses are consistent across runs/verifiers: 0eae6b70, 25d17bff,
  b08bfc41, 741f6348, d7b7f610, 0c53a032, 96c7270c (true merges the LLM calls different cells — e.g.
  0eae6b70: red fibre running beside a blue arbor in every section), 8e1a06ea (glia-like no_merge across a
  damaged section), plus wrong-candidate tasks (2104601b, 8060d42a, fd852a49, 7ec899df).

## Round 4 split analysis (all negative; split stays off)
Data: dev/true_locations_train.json + L2 graphs of all 63 one-segment train tasks (dev/round4/*.py).
- L2 `approximate` flags (chunks without a mesh fragment) are not near the true cut (nearest approximate
  node 2–34 um away, rank 3–427 by distance).
- Balanced-cut ranking (remove 2 um ball at each thinned L2 node, size of 2nd-largest piece): true cut is
  top-1 in 7/49, top-3 in ~15/49. Ties along backbones (whole paths give the same partition).
  Among the 17 split tasks with a single big mesh component (a weak split-vs-no_split gate: 17/49 split vs
  2/14 no_split), top-1 is right in only 2.
- Mesh connected components: no_split segments (freshly merged) tend to have more large mesh components
  (median 4 vs 2), but junctions between mesh components are not at the true cut (nearest 4–190 um).
- Same mesh-component stats on merge tasks: no merge/no_merge signal. Segment L2 sizes: no signal
  (every task segment has >= 100 L2 nodes).
- Economics reminder: answering split on every 1-segment task needs >~48% localisation accuracy to gain
  +3 on val (25 split vs ~9-11 no_split). Nothing found comes close.

## Ideas for next rounds
- Merge side is at ~80/100 on train with flat decision curves; the misses are hard cases. Remaining
  merge upside on val is ~6 tasks. A gain probably needs new evidence (e.g. EM reslices along the neurite
  axis, or a proper 3D EM continuity check across the z-break), not rule tuning.
- Splits (25/86 val, 0 hits) remain the only large pot; need a fundamentally better localiser.

---- (round 3 notes below) ----

# FlyWire proofreading harness — NOTES (round 3, h003)

## What this harness is
Merge tasks (2 segments): geometric contact candidates (geom.py, unchanged since h000) -> top candidate ->
blank-EM-section detector + LLM verifier v2 (merge_verify.py, from round 2) -> one linear rule.
Split tasks (1 segment): always `none` (SPLIT_ENABLED = False; split.py / split2.py / llm_split.py unused).

## Round 3 diagnosis
- **Which code was evaluated before.** The round-1 h001 train logs (feedback/train) come from a score-only
  snapshot (0 LLM requests, no "merge decision" log line). The evaluated h002 (rejected, val 45 vs 43) was
  *h001's v1 verifier* (`score + 3(p-0.5) >= -0.8`) **plus split2 enabled** (its run.logs show the h001 rule
  and split decisions), not the round-2 linear rule/blank code in its folder. split2 was 3/50 on split and
  cost 3/13 no_split on train, and 0/25 split with 1/11 no_split lost on val. So the round-2 merge-side code
  (blank detector, verifier v2, linear rule) had never been through validation. h003 ships it, with split off.
- **True locations via zero-cost probes** (dev/true_locations_train.json, dev/trilaterate.py): a probe harness
  that answers a merge/split with two points at a known spot gets back the distance to the true location even
  on a miss; 4 runs (different spots) trilaterate the location exactly (residual 0). $0 because no LLM calls.
  Located 50/50 merge and 49/50 split edits.
- **Merge ranking:** true location is the top candidate on 41/50 merges; rank 1-2 on 4 (2104601b, 8060d42a,
  af0bfdf9, fd852a49), rank 6-7 on 2 (7ec899df, f446907d), not a candidate on 2 (7bd56303: 12.5 um from the only
  gap candidate; c5d5b982: 3.4 um), 175f6fc3 is a 3.9 um gap (cannot hit). So the decision is the bottleneck.
- **Merge/no_merge features tried at the top candidate (100 train tasks, LOO logistic):** score alone 72,
  score+blank 74. No gain from: A/B z-range overlap in a 3x3 um x 31-section box (merges overlap less, but it is
  redundant with score), number of candidates (1 candidate: 18 merge vs 6 no_merge, but redundant), in-plane vs
  z-contact counts, in-plane A-B interface darkness in EM (membrane vs none: identical distributions),
  misalignment (phase-correlation shift between consecutive sections: >10 px at 17 merges/4 no_merges, but those
  merges already score high), candidate score margin.
- **Splits:** at the trilaterated true cuts, segments are whole neurons with dense arbors; the cut sits inside
  the arbor (renders: scratch only). Removing a 2 um ball at the true cut leaves two big pieces in ~1/3 of tasks
  (rank <= 4 among ~250-5000 thinned L2 nodes on 17/49), but the same "balanced cut" exists in every no_split
  segment. No precise enough localiser -> split stays off (any wrong split on no_split costs a hit; ~11 of 86
  val tasks are no_split).

## Round 3 changes (harness.py decision constants only, on top of the round-2 merge code)
1. `COMB_TH` -0.3 -> **0.0**. A merge answer only hits if the location is also right (~85% of true merges),
   while `none` always hits a no_merge, so the threshold should lean to `none`. On train the interval
   [-0.3, 0.14) held 3 no_merges (bf6bdc6c, 485d927c, 728643ff) and no merges.
2. `GAP_BLANK_BONUS` = 1.5: gap candidates get an extra +1.5 when blank sections lie within 3 sections (missing
   sections explain the gap; the gap penalty -4-gap/300 assumes it is unexplained). Fixes 5a959624; 32b49ab4 was
   already a gap+blank hit; no no_merge has a gap+blank top candidate.
3. `VERIFY_SAMPLES` 3 -> **5** (per-sample p_same spread is 0.15-0.3; same-task means moved by up to 0.25 between
   runs at 3 samples). `VERIFY_HI` 1.6 -> 2.0 so the LLM is asked wherever it can flip the decision.
4. Split tasks: `none` (as h001).

## Measured (score_train)
| version | merge | no_merge | merge-side /100 | $ (100 tasks) |
|---|---|---|---|---|
| h001 eval (feedback, score only) | 31 | 40 | 71 | 0 |
| h002 eval (v1 verifier, feedback) | 36 | 41 | 77 | — |
| round-2 merge code, split off (req 025035_c8f30a) | 37 | 41 | 78 | 2.3 |
| + changes 1-2 (req 025539_2c71cb, +13 no_split 13/13) | 34 | 45 | 79 | 2.3 |
| **+ change 3 = h003** (req 030200_3cb40f) | 35 | 46 | **81** | 4.1 |
Train total estimate: 81 + 13 no_split + 0 split = **94/163** (h001 84, h002 90). ~$0.04/task overall
(~$0.08 per merge task that reaches the LLM), merge-task wall 7-106 s, no crashes.
Run-to-run noise on the merge side is about +-3 (LLM p near the threshold: 741f6348, d7b7f610, 25d17bff,
0eae6b70 flipped between runs).

## Remaining merge-side misses (h003 run)
- FP no_merges that look like clean continuations (21f1861c, 485d927c, 8e1a06ea, d0bc5ba7): LLM p 0.4-0.7.
- True merges with correct top candidate but low p: 0c53a032, 0eae6b70 (p ~0.15), 96c7270c, b08bfc41.
- Wrong/no candidate: see ranking above.

## Ideas for next rounds
- Use dev/true_locations_train.json to train/inspect directly (true-location features vs top-candidate).
- Splits: still the big pot (25/86 val). A localiser would need >~45% precision when it fires, or a
  split-vs-no_split gate. Not found.
- Merge: run-to-run LLM noise is now a sizable part of the error; a better verifier view (e.g. EM
  reslices along the neurite axis) could matter more than more features.

---- (round 2 notes below; NB its "final" linear-rule code was not what got evaluated as h002) ----

# FlyWire proofreading harness — NOTES (v2, round 1)

## Round 1 changes (task version 2: no-edit tasks exist, ids opaque)
h000 never answered `none`, but 63/163 train tasks need none (train: 50 merge, 50 no_merge,
50 split, 13 no_split). Changes:
1. **Split tasks → always `none`** (`SPLIT_ENABLED = False` in harness.py). Split localisation was
   1/50 on train (0/25 val); `none` wins all no_split tasks (13/13 train). Split code kept, unused.
2. **Merge tasks can answer `none`.** The h000 best-candidate geometric score (exits/opposition) already
   separates merge (median −0.04) from no_merge (median −2.5). Score-only threshold −1.0: hits flat at
   70–71/98 for thresholds −1.5..−0.5 (from h000 logs); measured 71/100 on a full run.
3. **LLM verifier** (`merge_verify.py`): at the best candidate, renders (a) A/B mesh point clouds red/blue in
   a 12 µm cube, 3 projections, (b) 5 EM sections (8 nm, 4 µm FOV, z offsets −6,−2,0,+2,+6) with A/B mask
   outlines; Opus returns p_same. 2 samples averaged. Decision: merge iff
   `score + 3·(p_same − 0.5) ≥ −0.8`; verifier skipped when it cannot change the answer; falls back to
   score ≥ −1.0 if the LLM fails. Offline on 96 tasks: p alone (≥0.6) 67, score alone 68, combo 74–75
   in-sample; 2-fold CV (200 random halves) combo 36.2/48 vs score-only 32.0/48 — real but modest.
4. On any exception return `none` (no merge fallback).

## Measured (score_train, all 163 train)
- request 005930_9651af (changes 1,2,4): **84/163** — merge 31/50, no_merge 40/50, no_split 13/13, split 0/50.
- request 011331_08f2d6 (all changes): **90/163** — merge 36/50, no_merge 41/50, no_split 13/13, split 0/50.
  $2.24 total (~$0.04 per merge task, $0 split), max wall 128 s, no tracebacks.
- Remaining misses: 13 merge→none (≈4 had the right candidate, the rest had a wrong top candidate so
  were misses anyway), 1 too_far, 9 no_merge answered merge.

## Ideas for next rounds
- Splits are the big untapped pot (~30% of tasks), but only worth enabling with a confidence gate:
  a wrong split on a no_split task costs a hit, and `none` on a split task is a miss either way, so any
  split localiser with precision > 0 on split tasks is pure gain as long as it stays off no_split tasks.
  (Prior: 50 split vs 13 no_split.) I.e. try split only when a detector is confident; else `none`.
- Merge: verify top-2/3 candidates with the LLM and pick the best combined score (h000 had ~9 tasks with
  the wrong top candidate). Better verifier views (more EM sections, zoomed 3D).

---- (h000 notes below) ----

# FlyWire proofreading harness — NOTES (v1)

## Layout
- `harness.py` — `run()`: dispatches on 2 segments (merge) vs 1 (split); try/except with fallbacks
  (merge: closest L2-centroid pair; split: geometry-only). Never returns a crash.
- `geom.py` — mesh welding (duplicate chunk-boundary vertices merged by rounded coords), `MeshInfo`
  (KD-tree + vertex adjacency), contact/gap candidate generation, local "exits" topology feature,
  `snap_pair` / `snap_one` (turn a location into voxel centres that really lie on a root, using
  `segmentation` cutouts at 16x16x40, growing the box if needed).
- `split.py` — L2-graph analysis for split tasks: X-crossing candidates, branch grouping, point placement.
- `llm_split.py` — Opus prompts for splits (`pick_candidate` used; `ask_location`/`ask_zoom` kept but off).
- `render.py` — 3-view point-cloud projections of a mesh (PNG) with numbered markers.

## Merge (2 segments) — 41/50 train hits, $0, ~10-20 s/task
1. Fetch both meshes, weld vertices.
2. Candidates = clusters of A-vertices within 150 nm of B (single-link 500 nm; clusters > 3 µm are
   cut into 2 µm grid cells), plus "gap" candidates (closest approach up to 2.5 µm, >1 µm from contacts).
3. Features per candidate at r = 1, 2, 3 µm: number of times the *mesh-connected* local piece of A (and B)
   crosses the sphere of radius r ("exits": 1 = the segment ends here), and opposition of the two local
   centroid offsets (A and B lying on opposite sides).
4. Score (hand-tuned on my own train hits as positives): -Σ|exits-1| + 0.7·opp + 0.3·log(1+n);
   gap candidates get −4 − gap/300 nm (no true location among my hits was a non-contact gap).
5. Snap the best location to a touching voxel pair (A voxel adjacent to B nearest the centre).
   For gaps > 300 nm snap each side's own nearest vertex.
- Progress: plain contact-cluster + mesh-centroid "endness" 39/50 → subdivided clusters + exits
  features + gap penalty + robust snapping 41/50.
- Remaining misses: wrong contact chosen (5–55 µm off; many contacts between two neurons running
  together), and 2 tasks where the segments are ~3.9 µm apart, so both points can't be within 2 µm of the
  true location (t_175f6fc3 midpoint was 0.9 µm off).

## Split (1 segment) — 1–2/50 train hits, ~$0.016/task, ~20-70 s/task
Segments are whole neurons (260–6700 L2 chunks, 50–380 µm), so this is the hard half.
- Candidates: L2 nodes with degree ≥ 4, scored by how well their branches pair into two straight
  anti-parallel lines (an "X"); branch directions from walking ~6 µm out along the L2 graph.
- Opus is shown the whole-neuron 3-view render with the top 12 candidates (thinned to ≥ 8 µm apart)
  numbered, and picks one (`pick_candidate`). One call, ~$0.016.
- Points: **1 source + 1 sink** (more points only adds ways to straddle the cut wrongly), each ~1.5 µm
  from the junction inside its own branch's angular sector, snapped to a voxel of the root.
  Branch choice: deg ≥ 4 → one branch from each straight line; deg 3 → if two branches are near parallel
  (cos > 0.7, "two strands fusing") separate them, else the odd branch vs one through-branch.
- Measured variants (50 split tasks each):
  - X heuristic only, 2+2 points: 0/50 (median distance ~30 µm; 14 tasks < 10 µm).
  - X heuristic only, 1+1 points: 0/50 (two "too_far" at 4–6 µm with correct sides).
  - LLM free localisation (whole view → coordinates → zoom with numbered junctions): 1/50, $0.04/task,
    median ~35 µm. Opus reads coordinates off axes badly (≈10 µm errors).
  - LLM picks among X candidates: 2/50 (rerun: 1/50), $0.016/task. **Current default.**
- Almost every miss is `sides_not_separated`, i.e. the location is wrong (typically 5–80 µm off).
- Observation: t_bb0565ee is a "zipper" merge — the soma's primary neurite runs as two parallel strands
  that fuse; cut is at the fusion. Large L2 loops are a possible signal, but L2 loops of ~10 µm are
  common in every neuron, so not discriminative on their own. Thick-node (soma) detection via
  distance from L2 centroid to nearest mesh vertex found >2 µm-deep nodes in only ~4 tasks.

## Ideas for next revisions
- Splits: need a better candidate detector. Options: local mesh/segmentation skeletonization to get
  exact junctions (L2 centroids are only ±2 µm); EM-based checks (membrane gaps at a candidate via
  zoomed EM + mask overlays shown to Opus); multi-candidate zoom renders scored by the LLM; soma pairs.
- Merges: a learned ranker over contact features, or an LLM check on the top 2–3 when scores are close;
  smarter handling of long lengthwise contacts.
- Budget is barely used ($2/task allowed, ~$0.02 spent), as is wall time (max 71 s of 1800 s).

## Final measured result (request 173015_f7bde1, all 100 train tasks)
42/100 hits: merge 41/50, split 1/50. Total spend $0.83 (≈$0.017 per split task, $0 per merge task).
Max wall time 71 s. No crashes or timeouts.
