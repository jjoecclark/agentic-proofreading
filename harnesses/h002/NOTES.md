# FlyWire proofreading harness — NOTES (round 2)

## Round 2 changes (merge side only; split tasks still answer `none`)
1. **Blank-section detector** (`merge_verify.blank_sections`): fetch EM (8 nm; fallbacks 16 nm / 4 nm, because the
   server rejects some levels near the x-origin with OutOfBounds) in a 2x2 um box, +-10 sections around the
   best merge candidate; sections whose max is 0 are missing data. If a blank section lies within 3 sections
   of the candidate, the segmentation very likely broke there. Train: of the true merges whose top-1 candidate
   was right, 18/41 have a blank section at <=3 sections; among 50 no_merge tasks only 1 does (bf6bdc6c), 2 more
   at 4–6 sections. It is the clearest single signal found so far (LOO logistic: score+blank 76/91 vs score
   alone 70/91). All-blank boxes (missing chunk) count as "unknown".
2. **Verifier v2** (`merge_verify.py`): EM at native 8 nm (2.4 um FOV, 7 sections -5..+5 + a raw centre panel,
   1-px outlines drawn into the pixels with PIL) instead of matplotlib-downsampled 4 um panels; prompt now
   includes measured facts (blank sections, z-ranges of A/B near the contact, A-B touching faces) and explains
   how both kinds of case arise (DIFFERENT = a proofreader just cut an automatic false merge, so they always touch).
   3 samples, prompt cached (1st call writes, 2 more read). ~$0.025/merge-task.
3. **Decision = one linear rule** (fit by LOO logistic regression on train merge/no_merge logs, then rounded):
   `combined = score + 4*(p_same-0.5) + 3*[blank within 3 sections]`, merge iff `combined >= -0.3`.
   LLM asked only when it can flip the decision (`-2.2 <= score + blank bonus <= 1.6`); p=None counts as 0.5.
4. Removed: the h001 two-stage `score + 3(p-0.5) >= -0.8` rule.
   Tried and reverted: verifying the top-3 candidates and taking the best combined score (ALT_K=3, code kept,
   off). It recovered right-candidate-at-rank-1 cases (0c53a032, 8060d42a) but added as many no_merge FPs
   (459c37b9, 21f1861c, d5646fcd) → 75/100.

## Diagnosis that drove this (round-2 probes on train)
- The round-1 feedback logs were from a snapshot without the LLM (0 requests); the code here had the verifier.
- **Candidate ranking is fine; the merge/none decision is the bottleneck.** Forcing a merge at candidate rank k
  (geometry only): top-1 within 2 um on 41/50 merge tasks; rank 1/2/3 add 4 more (2104601b, 8060d42a: rank 1;
  fd852a49, af0bfdf9: rank 2). Ceiling with top-1 + perfect none decision = 91/100 on merge+no_merge.
- LLM p_same is noisy across runs (same task: 0.47 → 0.32, 0.49 → 0.29) and the no_merge cases that look like a
  clean end-to-end continuation (485d927c, 21f1861c, 8e1a06ea, d0bc5ba7, 728643ff) fool both geometry and the
  LLM. In LOO, adding p to score+blank gains only ~+1/91.
- Splits: checked cheap discriminators for split vs no_split (L2 long edges, loops, mesh disconnected
  components / z-gaps): none separates them. Split stays off.

## Measured (score_train, merge + no_merge = 100 train tasks; split/no_split unchanged: 0/50, 13/13)
| version | hits/100 | merge | no_merge | $ |
|---|---|---|---|---|
| h001 as shipped (old verifier), req 012528_28819e | 75 | 33 | 42 | 2.20 |
| v2 verifier + blank rule (thresholded), req 014927_3833c4 | 76 | 30 | 46 | 3.82 |
| + top-3 alternatives, req 015746_90ea9b | 75 | 33 | 42 | 5.26 |
| **final** (linear rule, top-1 only), req 022030_95ca02 | **76** | 33 | 43 | 2.62 (103 tasks) |
Offline replays of the final rule on logged p values: 76–80 depending on which run's p is used. Run-to-run
noise is about ±2, so the gain over h001 is small (+1–4 on 100 tasks) and the train total is about 89–92/163.
Max wall 49 s. No crashes.

## Ideas for next rounds
- Remaining merge-side errors are no_merges that look like clean continuations, plus true merges the LLM calls
  "side by side" (b08bfc41, 0eae6b70). Supervoxel-level evidence at the contact (how the cut surface runs, A
  supervoxels touching B) has not been tried.
- Misalignment detection (section-to-section shift) could extend the blank-section idea. Merges with no blank
  section but a z-direction contact (A and B in disjoint section ranges) may be misalignment breaks.
- Splits remain 0/50 (~30% of tasks). Any detector needs very high precision to beat answering `none`.

---- (round 1 notes below) ----


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
