# FlyWire proofreading harness — NOTES (round 3, h003 candidate)

## Round 3 changes (vs h001)
1. **Merge verifier v2** (`merge_verify2.py`, `shade.py`). Point-cloud renders were replaced by shaded 3D
   surface renders (painter's algorithm on mesh triangles, matplotlib PolyCollection). Three images: a 40 um
   context cube, a 12 um close-up, and the same 5 EM sections with A/B outlines. The prompt asks Opus to
   reason about what A does at the cross, then B, then the EM, then overall morphology.
2. **Up to 3 candidates verified.** Candidates are taken in score order, kept only if they could still pass
   (score + W/2 >= threshold) and are >= 3 um apart from those already kept. Each gets 2 samples.
   z = score + 3.5*(p_same - 0.5). Merge at argmax z iff z >= -1.2. If the LLM fails, fall back to h001's
   score-only rule (top score >= -1.0).
3. Splits stay **disabled** (always `none`), as in h001.

## Measured
- Full train run, request 031150_6c9a5b: **92/163**. merge 39/50, no_merge 40/50, no_split 13/13,
  split 0/50. Compare h001 84 and h002 90 (h002 had merge 36, no_merge 41, split 3, no_split 10).
  $4.86 total (max $0.16 per task), max wall 165 s, no crashes.
- Earlier run (threshold -1.0, merge tasks only), request 030452_7dab70: 77/100 merge+no_merge, the same
  as h002. Re-thresholding on that run's own z values: -1.3..-1.1 gives 79, -1.0 gives 77. So the
  merge-side gain is about +1..2 on 100 tasks, within LLM run-to-run noise. Don't expect much on val.
- Offline, same candidates (scratch eval, 49 middle-zone tasks):
  - p_same AUC: old point-cloud verifier 0.79, v2 0.83, few-shot v2 (6 worked train examples in a cached
    prefix) 0.85. Few-shot was not adopted: the gain is noise-level and it adds complexity.
  - A logistic regression on score + p + voxel-contact count + #candidates gave no gain over
    score + p (LOO).
  - Two samples per candidate are almost identical (p differs by about 0.05). The sampling noise that
    matters is between runs.
- Ceiling for merge localization: forced merge at the top candidate is 41/50. Of the 9 misses, 4 have the
  true location among candidates 1-2 (that is why multi-candidate verification is in), 2 are single
  gap candidates (score -17), and 3 are nowhere in the top 5.

## Train ground truth (devdata/)
Distances are reported even on misses, and a $0 (no-LLM) probe harness costs nothing from the probe budget.
So 4 runs of `devdata/trilateration_probe_harness.py.txt` (K=0..3: four non-coplanar L2 points per
segment) trilaterate the **true edit location** exactly (residual 0).
- `devdata/split_true.json`: true cut locations of all 50 train split tasks.
- `devdata/merge_true_missed.json`: true locations of the 9 merge tasks whose top candidate was wrong.
Use these for offline work instead of spending score_train budget.

## What we learned about splits (with the true locations)
- The "balanced cut" generator (remove a 2 um ball around an L2 node, take the 2nd-largest component)
  puts the truth at rank 0 in only 7/50. Spatially diverse top-k recall (truth within 2 um):
  k=3: 8-12, k=10: 13-14, k=15: 14-16 of 50. Balanced-cut statistics do not separate split from no_split
  segments; no_split segments look just as balanced.
- Opus *sees* that a segment is a false merge (e.g. two somas in t_3e4ba717) but rates decoy cut sites
  as high as the true one (0.72 true vs 0.85 / 0.70 decoys). It cannot localize the cut.
- Soma detection via "depth of L2 centroid below mesh surface" fails: the two-soma segment maxes at
  1.46 um. A coarse segmentation volume is not practical because z stays at 40 nm at every level.
- Break-even maths for val (25 split / 11 no_split): splitting on every split-looking task needs
  localization precision above 44%. Any gate must keep no_split false positives near 0. With recall@10
  of about 28%, this does not pay. Next time, start from a candidate generator with much better recall
  (e.g. mesh-skeleton X-crossings or EM-level checks), measured offline against `split_true.json`.

## Ideas not done
- Few-shot merge verifier (AUC 0.85 vs 0.83). Perhaps worth it with more examples or a larger eval set.
- Stabilize decisions near the threshold, e.g. more samples only when |z - th| < 0.5.

---- (h001 / round 1 notes below) ----

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
