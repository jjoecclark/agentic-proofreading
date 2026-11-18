<!-- guide_version 1: "Fly Connectome Proofreading — Base Instructions for Agents" (Nov 17, 2026), supplied
by the user as a PDF and transcribed here, with the outer harness's note on top. Copied into the
meta-agent's /work/docs/. Links are dropped: the sandbox has no internet. -->

# How this guide applies to this task (note from the outer harness)

The guide below was written for agents and people who proofread FlyWire for real. Use its biology,
error catalogue and decision rules; they are what experts look for. Here, though:

- **Nothing is edited.** Your harness returns one decision per task (a merge, a split, or `none`).
  Rules about editing (latest versions, sandbox datasets, logging, undo, previewing a multicut,
  chaining edits) describe the real workflow, not something to implement. There is no "hand to a
  human" option: when the guide says to flag a case, you still have to choose an edit or `none`.
- **Time is frozen** at the task's time T. The segments you are given exist at T. Edit history,
  lineage, latest roots and root timestamps are not available (the world API refuses them).
- **The data is only what `/work/API.md` offers:** EM, supervoxels, segmentation and roots at T,
  L2 graphs, approximate L2 positions, and meshes. Not available: synapse tables, nucleus
  detections, cell types or other annotations, homologue or mirror-image cells, L2-cache attributes
  (volume, area), neurotransmitter predictions, multicut previews. Where the guide relies on those,
  approximate from what you have (for example, a soma from mesh or L2-graph thickness) or skip it.
- **Segment and L2 ids are opaque**, different in every task; their bits mean nothing.
- **Each task is one decision about the segment(s) shown**, not a full proofread of a neuron. A
  1-segment task asks: does this segment contain a false merge, and if so where should it be split?
  A 2-segment task asks: are these two segments one cell, and if so where should they be joined?
  Some tasks need no edit; the right answer is then `none`.
- The figures in the "Automated and AI proofreading" section are from the literature; measure your
  own harness on the train tasks rather than assuming them.

The rest of this file is the guide as supplied.

---

# Fly Connectome Proofreading — Base Instructions for Agents

Nov 17, 2026

## Purpose and scope

Your mission is to make each fly neuron's reconstruction correct enough that its synaptic connections
can be trusted, at the lowest cost per neuron. Fix errors in order of their effect on connectivity,
not on how tidy the mesh looks.

This guide is written for agents that build a proofreading harness, including its visualization
tools, and for agents that will proofread with it. It condenses the four Seung-lab/FlyWire training
documents supplied with this brief, the FlyWire blog, and the published literature up to November 2026.

**How to use this guide**

- Background: *Datasets and infrastructure*, *Biology primer*, *Synapses*.
- What to look for: *Error taxonomy*.
- What to do: *Workflow and completion criteria*, then *Decision rules*. Load the decision rules into
  every proofreading agent's prompt.
- What to build: *Harness requirements*.
- How well machines do today, and how to measure yours: *Automated proofreading*, *Evaluation*.
- Links and definitions: *Resources*, *Glossary*.

**Non-negotiable rules**

1. Edit only the latest version of a neuron. Root IDs change after every edit, so re-resolve IDs
   immediately before acting.
2. Build and test against a sandbox dataset. Edit a production dataset only with explicit permission
   from its owners.
3. Log every edit with its inputs, the evidence viewed, the rationale and a confidence score. Every
   edit must be reversible.
4. When the evidence is ambiguous, flag the location for a human instead of editing.
5. Prefer leaving a small split over creating a merge. A false merge adds wrong connections; a missed
   twig only drops a few correct ones.
6. Follow each dataset's terms of use and citation requirements.

## Datasets and infrastructure

Fly proofreading happens on a supervoxel graph: AI over-splits the volume into supervoxels,
agglomeration joins them into segments, and every edit adds or removes edges between supervoxels.
The adult datasets below are the ones you are most likely to work on; FlyWire conventions are the
default in this guide.

| Dataset | Coverage | Scale | Notes for proofreaders |
|---|---|---|---|
| FAFB / FlyWire | Adult female brain | 139,255 neurons; ~50 million synapses | Serial-section TEM, 4×4 nm pixels, 40 nm sections, 7,062 sections. Misalignments were the dominant failure mode of automated reconstruction. ~33 person-years and 2,712,769 edits (to v630). Public releases 630 and 783. FAFB images are left–right inverted; side labels follow biology. |
| Hemibrain | About half the central brain | ~25,000 neurons; ~20 million synapses | ~50 person-years of proofreading. Defines a useful body-status vocabulary: Traced, Roughly traced, Orphan, Leaves, Unimportant. |
| FANC | Adult female ventral nerve cord | ~14,600 neuronal cell bodies; ~45 million synapses | CAVE-based, same edit model as FlyWire. |
| MANC | Adult male ventral nerve cord | Whole VNC | Janelia FlyEM pipeline (neuPrint). |
| BANC | Adult female brain plus ventral nerve cord | ~160,000 neurons | Published in Nature on 8 June 2026. Proofread in Spelunker, a Neuroglancer-based tool. Path swaps and dark-mitochondria truncations are common. |
| Male CNS | Adult male brain, optic lobes, VNC, intact neck | 166,700 neurons; ~125 million connections | 8 nm isotropic EM (~160 teravoxels), ~90 million FFN segments, ~300 million predicted synapses. ~44 person-years. v1.0 released 8 June 2026. |

**How the segmentation is represented** (FlyWire methods, CAVE, CAVEclient docs)

- **Supervoxel**: the smallest unit; voxels grouped because they almost certainly belong to one cell.
  Edits never cut a supervoxel.
- **Root ID**: one connected component of the supervoxel graph, i.e. one neuron as currently
  segmented. Every edit retires the old root ID and mints new ones, so store (root ID, timestamp)
  pairs, never bare IDs.
- **ChunkedGraph**: the supervoxel graph split into spatial chunks under an octree that caches
  connected components. Level-2 (L2) nodes are a neuron's pieces within one base chunk; the L2 cache
  holds their size and surface area, enough for fast coarse skeletons.
- **Merge**: pick one supervoxel on each segment; an edge is added. Median server latency was 940 ms
  in FlyWire's beta.
- **Split (multicut)**: place source points on one side and sink points on the other; the backend
  runs a min-cut on the local supervoxel graph, weighted by predicted affinities. Median latency was
  1,818 ms. A preview call returns the resulting components before you commit.
- **CAVE annotations**: stored as points and bound to whichever root ID covers that point at a given
  time. Materialized versions give consistent snapshots; live queries give the current state.
  FlyWire's "proofread/complete" markings are themselves a point-annotation table.
- **Synapse tables**: pre- and postsynaptic points, each with supervoxel and root IDs. FlyWire drops
  synapses with an unassigned end or a cleft score of 50 or below, and most analyses require at least
  5 synapses for a connection.
- **Meshes and skeletons**: meshes are built per chunk and stitched. Public snapshots ship
  precomputed skeletons for proofread neurons; fafbseg can build quick low-resolution skeletons from
  the L2 cache.
- **Coordinate spaces**: FlyWire uses FAFB14.1; older FAFB14 data (CATMAID tracings, Buhmann
  synapses) need the transforms in `fafbseg.xform`.

## Biology primer for tracing

A fly neuron is almost always a tree: one soma, one neurite, branches that diverge and never rejoin.
Any departure from that shape is your first clue to an error. Source: the Glossary of fly
ultrastructure and morphology unless another source is given.

| Feature | What it looks like | How to use it |
|---|---|---|
| Soma and proximal neurite | Most fly neurons are unipolar: one neurite leaves the cell body and runs some distance before joining the main backbone. The junction usually sits between a nearer, finer, input-rich (dendritic) arbor and a farther, bouton-rich (axonal) arbor. | Orient yourself by finding the soma and the junction. Input and output can share an arbor, so polarity is a tendency, not a rule. |
| Backbone vs twig | Backbones contain microtubules; twigs do not (Schneider-Mizell 2016). Twigs can be thinner than 40 nm, i.e. thinner than one section. | Backbones carry the neuron's shape: proofread them thoroughly. Twigs carry most inputs, but missing twigs caused 96% of errors in manual tracing with little effect on strong connections. Skip orphan twigs unless the task needs them. |
| Branching direction | Neurites diverge outward like a tree. | A branch that turns back toward the soma, or leaves at an odd angle, suggests a wrong attachment. |
| No loops | Two branches that have diverged do not rejoin. | A loop means a merge, except **autofasciculation**: a neuron running alongside itself, most often the proximal neurite beside its backbone. |
| Cytoplasm | Glia are often darker than neurons, but light glia and dark neurons (some sensory neurons) exist. | A sudden light/dark change at a contact suggests a merge between two cells. |
| Glia | Densely thorny; wrap bundles of neurons as dark boundaries. | Glia merge with many cells and are hard to proofread. Cut glia off neurons; ignore small glial fragments, which do not affect connectivity. |
| Microtubules | Long tubes slightly narrower than vesicles, running along backbones. | Follow them: they keep a steady angle, curve with the neurite, turn into new branches and never cross a membrane. Microtubules splaying in a new direction reveal a branch you may have missed. |
| Mitochondria | Round or oblong, with internal folds (cristae) that may look broken. | Their bulges mimic boutons. Dark mitochondria can make the AI end a branch early; look for the membrane wrapped around one to find the continuation. |
| Nucleus | Large round body inside the soma. | The AI sometimes segments the nucleus separately from its soma; merge them. |
| Invagination | One cell pushing into another; double bilayer membranes; sheets wrapping at neurite ends. | The neck joining the two parts may be invisible between sections, which causes splits or merges. |
| Synaptic spinules | Protrusions from other neurons penetrating a bouton; can pinch off into double-membrane vesicles. | Do not mistake a spinule for part of the bouton's neuron. |
| Tracheae | Air tubes with ribbed walls and a hollow core, running through the brain. | Often merged with glia or neurons; cut them off. |
| Vesicles | Small round clear vesicles near synapses; larger dense-core vesicles (DCVs), sometimes with a halo; large clear-core vesicles. | Vesicle content signals presynaptic tissue (see *Synapses*). |
| Natural ending | Soft, rounded tips; a bouton or spine can mark a real endpoint (BANC tutorial). | Sharp, flat or jagged ends, especially on thick branches, suggest a truncation. |

**Evidence that a segment belongs to your neuron**

- Cytoplasm colour stays consistent, with no sudden change from dark to light.
- Microtubules keep a steady angle along a branch and turn smoothly into side branches.
- Branches leave at plausible angles and point away from the soma.
- The segment does not suddenly look more presynaptic or more postsynaptic (judged by vesicles) than
  the rest of the arbor.
- Its 3D shape matches other neurons of the same type, including the mirror-image neuron in the
  other hemisphere.

## Synapses

You will mostly consume automated synapse predictions, but you must recognise synapses to judge
whether a fragment belongs to a neuron and whether a contact is real. Source: the Fly Synapses guide
unless another source is given.

**Structure.** Fly chemical synapses are polyadic: one presynaptic site faces several postsynaptic
partners (dyads, triads, or a "rosette" of many). All partners touched by the same cleft count as one
synapse. Gap junctions are unlikely to be visible in FAFB.

**The four indicators** (Zheng et al. 2018 convention). Accept a synapse only if the primary
indicator and at least two secondary indicators are present.

| Indicator | Role | What it looks like | Pitfalls |
|---|---|---|---|
| Vesicle cluster | Primary, required | A concentration of small clear vesicles against the presynaptic membrane; sparse axons may show only 1–6 vesicles. | Scattered vesicles near a border are not a cluster. Larger, darker dense-core vesicles are not a cluster. |
| Synaptic cleft | Secondary | Thickened apposed membranes, often with a thin gap; usually under 1 µm long. | False clefts: step through sections, as a cleft can vanish when a different segment replaces the partner. |
| T-bar | Secondary | The darkest feature: a pedestal topped by a "table" inside the vesicle cluster. Can be a dot, a ribbon, or long ("travelling"). | Rarely T-shaped when seen face-on. |
| Postsynaptic density (PSD) | Secondary | A dark band on the postsynaptic side of the cleft; prominence varies widely. | Face-on, a PSD resembles the shadow of a membrane cut parallel to the section. Confirm with vesicles or a T-bar in nearby sections. PSDs occur only on postsynaptic (dendritic) membrane. |

**Face-on synapses.** When a membrane lies parallel to the section plane, step through z until the
postsynaptic profiles move aside and the cleft and T-bar appear. Orthogonal XZ and YZ views help.

**Unusual polarity.** Axo-axonic and dendro-dendritic synapses exist. Connectivity analyses usually
disregard synapses that originate from dendrites.

**Automated predictions you will use**

- **Synaptic partners**: Buhmann et al. 2021 predicted pre/post partner pairs across FAFB, about 250
  million in total. FlyWire combines these with Heinrich et al. cleft scores and drops pairs with a
  score of 50 or below (Dorkenwald 2024 methods).
- **Neurotransmitters**: Eckstein et al. 2024 predict acetylcholine, glutamate, GABA, serotonin,
  dopamine or octopamine from EM, with 87% accuracy per synapse and 94% per neuron. Treat
  single-synapse calls as noisy.
- **Completeness proxy**: in FlyWire, 92.3% of presynapses but only 43.9% of postsynapses are attached
  to proofread neurons, because unattached twigs carry most inputs.

**How synapses feed proofreading**

- A synapse at the cut face of a fragment is evidence of a false split; Autoproof feeds synapse
  proximity to its merge classifier for this reason.
- Rank orphan fragments by synapse count: a fragment with many synapses is worth attaching; a bare
  twig rarely is.
- A branch whose synapse polarity or predicted transmitter differs sharply from the rest of the
  neuron may be a merge. Use this only as supporting evidence: Autoproof found transmitter-based merge
  searches too noisy on their own.

## Error taxonomy

Every error is either a **merge** (pieces of different cells joined) or a **split** (one cell broken
into pieces); data artefacts cause both. Fix merges on backbones first, then backbone splits; leave
twigs and small glial fragments unless the task needs them. Sources: Errors & issues in EM, FlyWire
Proofreading Tips, BANC expert tips and the FlyWire FAQ.

### Merge errors

| Error | Signature | Fix | Priority |
|---|---|---|---|
| X-shaped (perpendicular) merge | A branch crosses another at roughly 90°. At a T-junction, a branch arriving from a different plane than its neighbours is suspect. | One multicut with sources on one branch and sinks on the other. | High |
| H-shaped (parallel) merge | Two parallel branches joined somewhere along their length. There may be several merge points, or a "phantom" one that looks joined in 3D but connects elsewhere. A short contact before the branches diverge can look like a normal continuation when zoomed out. | Use find-path between the two branches to locate the bridge; place cut points on the direct contact area. Expect to need several cuts; keep the 3D view zoomed in. | High |
| Twig-to-twig merge | Two arbors joined at thin terminal twigs. A twig linking two backbones is abnormal: twigs end backbones, they do not connect them. | Find-path to the junction, then cut. Easy to fix, hard to find in dense arbors. | Medium |
| Glia merge | Darker, thorny, wrapping tissue attached to a soma or neurite; cytoplasm darkness changes at the contact. Intracellular membrane folds in glia confuse boundary detection. | Cut glia off the neuron. Ignore small glial fragments. | High for large pieces |
| Trachea merge | Ribbed, hollow tube attached to glia or a neuron. | Cut off. | Medium |
| Synaptic invagination merge | At a synapse one branch wraps its partner, blurring the boundary. | A clean cut may be impossible; accept a split that keeps most of each cell. | Medium |
| Multi-soma segment | Two or more somas (or nuclei) in one root ID. | Split until each segment has one soma. | High |
| Path swap | A false continuation onto a neighbouring neuron, common where neurons run in parallel and twist, and after misalignments. Looks like an H-merge in which only one of the two extensions continues; directionality is implausible for the cell type. | Merge the correct continuation, then split off the wrong one. Confirm in 2D. Common in BANC's brain–VNC tracts. | High |

### Split errors

| Error | Signature | Fix | Priority |
|---|---|---|---|
| Backbone truncation at a misalignment or bad section | A thick branch ends abruptly with a flat or jagged face; the EM jumps between sections (e.g. the bad slice at z=3345 in the supplied example). | Landmark realignment: annotate a distinctive nearby object before and after the jump, toggle between the points, then find your neuron's continuation. If no landmark is visible, search beyond the field of view. | High |
| Soma-tract break (autofasciculation) | Detached soma, or a thin proximal neurite running parallel to the backbone. | Search the EM for a thin object running beside the backbone, starting near the dendrites, where the soma tract usually joins. | High |
| Pinched ("MIA") neurite | Very thin neurite squeezed between structures until invisible. Common between the optic lobe's lamina and medulla, often leaving detached somas. | Use neighbouring neurons of the same type running in parallel as guides. Try several candidate extensions and keep the one that fits. | High |
| Intracellular-membrane split | Mitochondria or other internal membranes are mistaken for cell boundaries; dark mitochondria end branches early in BANC. | Select the mitochondrion in 2D and look for the profile wrapped around it; merge that. | Medium |
| Nucleus split from soma | Nucleus is its own segment. | Merge it into the soma. | Medium |
| Missing dendritic branch | A void in otherwise regular, fractal branching, or uneven sparseness. Some cell types really are sparse. | Inspect the void in 2D for an unattached branch. Large missing dendrites matter most. | Medium |
| Barely attached bouton | A thin raised bump on an axon branch in 3D; a small, nearly separate blob next to it in 2D. | Select the blob; attach it if it is the bouton. When most terminal twigs carry boutons, check the ones that do not. | Low–medium |
| Orphan twig | A small fragment, often under 40 nm thick, carrying a few synapses. | Usually skip; attach in descending order of synapse count when the task needs completeness. | Low |

### Data artefacts

| Artefact | Signature | What to do |
|---|---|---|
| Misalignment | Neighbouring sections shifted against each other; sometimes several bad sections or a blackout gap. | Landmark realignment (above). |
| Folds and pinched sections | Thick black bars across the EM. | Continue across with landmarks and mitochondria cues. |
| Unsegmented region | Square-edged holes in the mesh; no supervoxels appear on hover. Caused by bad sections or the tissue edge. | Cannot be fixed by editing; record and flag. |
| Stale view | You are looking at a past snapshot. | Only the current root can be edited: re-resolve to the latest root first. |

## Workflow and completion criteria

Remove merges, extend backbones, and stop when more edits will not change connectivity. The Seung
lab's fast protocol cut the time per neuron from 135 to 21 minutes on average, and its results were
about 99% identical to three rounds of thorough proofreading (Faster proofreading).

### What extra effort buys

| Study | Finding |
|---|---|
| FlyWire 2022: 183 neurons, each proofread in three successive rounds | Large corrections (over 1 µm³) had a median of 8 in round 1, then 1, then 0. After round 1 the median F1 against round 3 was 0.99 for both volume and synapse assignment. |
| FlyWire 2022: timing of 49 first-round neurons | Median 90 minutes, mean 135 minutes per neuron. |
| FlyWire whole brain: 826 random neurons re-proofread | First-round proofreading averaged 99.2% F1 by volume. |
| FlyWire whole brain: effort | About 79 seconds per edit; roughly 30 person-years for 2.7 million edits. |

### Order of work across a dataset

1. Every segment containing an automatically detected nucleus: extend it as far as possible and
   remove all false merges (FlyWire round 1).
2. Remaining segments in descending order of synapse count, down to a cut-off (FlyWire used 100
   synapses).
3. Orphan fragments by synaptic weight. Autoproof handled fragments with 10–100 synapses and
   auto-accepted only high-confidence merges.

### Loop for one neuron

1. Resolve the latest root ID and record the timestamp.
2. Review the whole neuron in 3D before zooming in. Count somas, compare the shape with other cells of
   the same type and with the mirror-image cell, and note loops, backward branches, voids and thick
   abrupt endings.
3. Remove merges first: other neurons, glia, tracheae, extra somas. Merges put wrong synapses on the
   neuron; splits only drop correct ones.
4. Extend backbone truncations. Visit each thick ending, inspect the EM at the ending and a few
   sections beyond it, and bridge misalignments with landmarks.
5. Reattach a detached soma or nucleus.
6. Skip short thin endings, orphan twigs and small glial fragments unless the task requires them.
7. Validate: one soma, no loops except autofasciculation, plausible synapse polarity, shape
   consistent with homologues.
8. Mark the status with a point annotation inside the backbone, recording root ID and timestamp.
   Re-mark after any later edit; a marked neuron that changes substantially without a new mark goes
   back for review.
9. Record anything unresolved with its location and reason, and hand it to a human.

### When a neuron is complete

- **FlyWire rule** (FAQ): mark a cell complete only when it has a soma, a backbone and all structures
  common to its cell type. If unsure, do not mark it.
- **Exceptions** (To trace, or not to trace?): photoreceptor (R) cells are only partly inside the
  volume and can be complete without a soma. Many cells have no distinct axon, and dendrites need not
  look "classic".
- **A neuron that seems to stop where it should continue is probably incomplete.** Use neighbouring
  cells as guides.
- **BANC**: completeness standards depend on the lab and the question; even thorough reference cells
  do not have every spine checked.

### Status vocabulary for the harness

Track axon and dendrite status separately, as MICrONS does, and reuse hemibrain's terms for bodies
that are not ordinary neurons.

| Status | Meaning | Usable for analysis |
|---|---|---|
| unreviewed | No comprehensive proofreading yet | No |
| clean | All merges removed; may still be truncated | Yes, for the synapses it has |
| backbone_extended | Clean, and every backbone ending checked and extended | Yes (FlyWire "complete" level) |
| extended | Clean, and every tip inspected and extended where possible | Yes |
| flagged | Unresolved issue recorded for a human | Not until resolved |
| orphan / leaves_volume / not_neuron | Untraceable fragment, exits the volume, or glia/trachea | Excluded or analysed separately |

## Decision rules for agents

Paste this section into every proofreading agent's instructions. Each rule names the evidence to
collect, the action, and when to hand the case to a human.

| If you see | Then | Hand to a human when |
|---|---|---|
| Two somas or nuclei in one segment | Split between them; follow each proximal neurite to decide which branches belong where. | Both somas share a long parallel tract and the cut location is unclear. |
| A branch crossing another at about 90° | Check the junction in 2D across several sections; if the profiles stay separate, cut. | The branches share membrane over several sections. |
| Two parallel branches touching | Run find-path to locate every contact; cut on direct contact areas; re-check for a second bridge. | More than two cut attempts fail. |
| A branch turning back toward the soma, or leaving at an odd angle | Treat it as a probable merge or path swap; confirm in 2D. | It is the only continuation of a backbone. |
| A loop in the skeleton | Find the closing edge and cut it, unless it is autofasciculation (same neuron running alongside itself). | Both paths are equally plausible. |
| A sudden change in cytoplasm darkness, or thorny dark tissue | Treat it as a glia merge and cut at the boundary. | The cell type is known to have dark cytoplasm (some sensory neurons). |
| A thick branch ending abruptly | Inspect the EM at the ending and up to a few sections beyond; look for misalignment and use landmarks. | Two or more candidate continuations fit equally well. |
| A candidate fragment for a split | Accept if it continues the branch's direction, has a similar calibre at the interface, and lines up across any gap. | Calibre or direction disagree, or the fragment would add a second soma. |
| Two segments side by side over a long stretch | Assume two different neurons; do not merge. | Autofasciculation is plausible (proximal neurite next to its own backbone). |
| A large gap between candidate pieces | Do not reject on the gap alone; missing or damaged sections are common. Judge by direction and calibre. | The gap spans many sections with no landmark. |
| A detached soma near the arbor | Search for a thin tract running parallel to the backbone near the dendrites; check neighbours of the same type. | No tract is visible in EM. |
| A short thin ending, orphan twig, or small glial scrap | Skip, unless the task requires completeness. | Never; just log it. |
| An unsegmented square-edged hole | Do not edit. Record the location. | Always report it. |
| A synapse at the cut face of a fragment | Treat it as evidence of a false split and look for the continuation. | The continuation would cross another neuron's backbone. |

**Guardrails for every edit**

1. Before a split, preview the multicut and check what leaves: if the removed part holds the soma or
   most of the synapses, stop and reconsider.
2. Before a merge, check that the result will have at most one soma and no loop.
3. After any edit, confirm the new root ID, the change in volume and synapse count, and that the
   change matches your intent. Undo if not.
4. Never chain more than a few unverified edits; re-render and re-check after each one.
5. Log the root ID before and after, edit type, points used, images viewed, rationale and confidence.

**Known model biases to correct for** (ConnectomeBench)

- Correct merge partners are not necessarily thin; split pieces are often as thick as the segment they
  belong to.
- Large gaps usually come from missing data, not from wrong candidates.
- Reason in 3D: a segment can look short in one view and long in another, so use several views.
- Models drift towards always accepting or always rejecting merges. Track your accept rate against a
  labelled set and recalibrate.

## Harness requirements

The harness must let an agent see what an expert sees — the whole neuron in 3D and the raw EM at the
suspect point — and act through guarded, logged, reversible edits. Build the components below; the
error catalogue and decision rules above say what each one is for.

Harness loop for one suspect location (7 steps, 1 confidence gate): **candidate queue** (ranked suspect
spots) → **render evidence** (3D views + EM slices) → **decide** (rules + scorer) → **confident?** If
yes: **guarded edit** (preview, then commit) → **verify** (undo if it is wrong) → **log + status** (IDs,
evidence, why) → next candidate. If no: **human review** (uncertain cases) → next candidate.
Candidates come from automatic flags. Only confident decisions become edits; every edit is previewed
and verified, and uncertain cases go to a human.

### Views to build

| View | Must show | Used for |
|---|---|---|
| 3D mesh | Whole neuron and zoomed regions; free rotation plus fixed top, side and front views; a candidate pair in two contrasting colours; cropping to a region; neighbouring segments on demand. | Overall shape, X- and H-merges, voids, branch direction. |
| 2D EM slice | Raw EM at full resolution (4×4 nm in FAFB) with a toggleable segmentation overlay; step forward and back by N sections; jump to a coordinate. | Membranes, boundaries, synapses, misalignments. |
| Orthogonal slices | XZ and YZ planes through the same point. | Face-on synapses and membranes lying parallel to the sections. |
| Skeleton graph | Soma, branch points and endpoints with radius; endpoints sorted by radius. | An ordered list of backbone endings to inspect; loop detection. |
| Synapse overlay | Pre- and postsynaptic points in different colours; optional transmitter predictions; counts per branch. | Polarity checks, synapses at cut faces, orphan value. |
| Homologue overlay | Another cell of the same type, or the mirror-image cell, in the same space. | Spotting missing or extra branches. |
| Landmark tool | Two linked annotation points on either side of a jump, with one-key toggling. | Bridging misalignments. |
| Find-path | Shortest path between two points through the neuron's L2 graph, highlighted in 3D. | Locating merge bridges in dense arbors. |
| Edit preview | Multicut components before commit; soma count of a merge result. | Enforcing the guardrails. |
| Edit history | Timeline of a neuron's edits and authors; differences between root versions. | Avoiding undoing others' work; training data. |

### Data layer

- Root-ID resolution at a timestamp, latest-root lookup and staleness checks (CAVEclient:
  `get_roots`, `get_latest_roots`, `is_latest_roots`).
- Per-object edit logs and undo.
- L2 graph and L2-cache attributes (volume, area) for cheap skeletons and features.
- Meshes, precomputed skeletons, and EM and segmentation cutouts (CloudVolume, fafbseg, navis).
- Annotation tables: nucleus detections, synapses (materialized and live), cell types, proofreading
  status.
- A sandbox/production switch, with production writes disabled by default.

### Automatic candidate generation

Agents should not search blindly. Generate a ranked queue of suspect locations per neuron:

- **Truncation candidates**: skeleton endpoints whose radius is backbone-sized.
- **Merge candidates**: skeleton junctions of degree three or more, especially near-perpendicular
  crossings; loops; two or more nuclei; abrupt calibre changes at a junction; branches whose synapse
  polarity differs from the rest.
- **Split candidates**: orphans touching an endpoint, ranked by synapse count; synapses at fragment
  cut faces.
- **Asymmetry**: cable length or synapse counts that differ strongly from the homologue.
- **No-go zones**: known misaligned or unsegmented regions, so effort is not wasted there.

Use specialist models to score candidates. Autoproof combined a 3D CNN on EM plus segment masks plus a
synapse-proximity channel with a point-cloud shape model and synapse-connectivity features.

### Rendering evidence for models

- Always include segment masks. Removing them cut split-correction accuracy from about 97% to 77.7% in
  ConnectomeBench2.
- Render several views and aggregate: averaging up to 64 geometry views added roughly 3–4 percentage
  points.
- Pair mesh geometry with EM slices. Mesh-based split correction transferred across species; EM-based
  and merge-error judgements did not.
- Crop to the interface. ConnectomeBench used top, side and front views at 1024×1024 pixels, cropped
  to a 4,096 nm cube around each split interface.

### Edit executor, logging and replay

- Wrap merge, multicut and undo behind the guardrails in *Decision rules*; rate-limit calls. Edits
  take about 1–2 seconds each.
- Record every decision with its evidence images, scores, rationale and outcome.
- Build an offline replay environment from CAVE edit histories. ConnectomeBench2 treated edits that
  were never reversed as positive labels; use the same idea to test the harness before it touches
  live data.

## Automated and AI proofreading

Specialist models trained on past edits now match or beat human experts on split correction, and on
merge detection in mouse. Fly merge detection remains the weakest case, and off-the-shelf language
models are near chance on it. Build the harness so that specialist scorers propose, the agent gathers
evidence and applies the rules, and humans see the uncertain middle.

| Work | Date | Approach | Key result | Lesson for the harness |
|---|---|---|---|---|
| VLM benchmark (Li, Pfister et al.) | Sep 2026 | 21 vision-language models on synapse detection; 5 on split and merge decisions from ConnectomeBench2. | Zero-shot, most models were at chance. LoRA fine-tuning on a few thousand labels reached specialist level; adapted models beat specialists on merge errors in unseen species. | Never rely on zero-shot judgements; fine-tune or give worked examples. |
| ConnectomeBench2 (Brown, Farkas et al.) | Jun 2026 | 716,000 expert decisions and 4.5 million images from fly, mouse, human and zebrafish; one vision transformer on mesh and EM views. | Split correction 97.0% vs 93.0% for human experts (balanced accuracy); merge identification 93.0% vs 84.1%. Fly merges were lowest at 85.5%. | Keep masks, aggregate views, check calibration, expect drops on new data. |
| ConnectomeBench (Brown et al.), the brief's first paper | Oct 2025 | Off-the-shelf multimodal LLMs on rendered 3D meshes from FlyWire and MICrONS; prompts with descriptions or heuristics. | Segment typing 0.52–0.82 balanced accuracy. Split multiple-choice up to 0.828 on FlyWire vs 0.90 for humans. Binary merge detection near chance (best 0.628 vs 0.74), and a fine-tuned ResNet-50 beat every LLM on it. | LLMs can triage and choose among split candidates; do not trust them alone on merges. |
| Autoproof (Huang et al., Janelia) | Sep 2025 | 3D CNN on EM plus segment masks plus a synapse channel; point-cloud shape model; connectivity features; trained on past decisions in the male CNS. | Over 90% precision at 90% recall on merge candidates: 90% of merges for about 20% of the effort. 200,000 orphan merges auto-accepted, worth about 4 person-years. | Train on your own edit history; auto-accept only above a measured precision. |
| PATHFINDER, the brief's third paper | May 2025 | Mouse cortex: flood-filling segmentation, a high-recall candidate generator (SENSE), a point-cloud plausibility filter (SHAPE) and combinatorial search. | Axon nERL of 94.2%; projected proofreading cost about 84 times lower. | Generate candidates with high recall, then filter with shape at a larger scale. |
| NEURD (Celii et al.) | Apr 2025 | Meshes decomposed into annotated graphs; rule-based merge correction in mouse. | Automated merge cleaning in MICrONS, but it caught none of 14 multi-soma examples in ConnectomeBench. | Hand-written graph rules are interpretable but brittle across datasets. |
| SegCLR (Dorkenwald et al.) | Nov 2023 | Self-supervised embeddings from EM and segmentation. | Cell type inferred from fragments as small as 10 µm, with far fewer labels. | Embeddings can flag fragments whose identity does not match their host neuron. |
| Bridging the Gap (Berman, Chklovskii, Wu), the brief's fourth paper | Dec 2021 | Point-cloud classifier (CurveNet) for merging fragments across missing sections. | Across 8 missing sections it merged just over 40% of neurons with merge errors under 5%. A nearest-neighbour heuristic failed beyond 1 section. | Gaps need learned shape priors; set thresholds that favour avoiding merges. |
| RLCorrector (Nguyen et al.) | Jun 2021 | Hierarchical reinforcement-learning agents for detection, classification and correction. | Fully automatic proofreading on benchmark volumes. | A precedent for splitting the task among specialised agents. |
| Guided proofreading (Haehn et al.) | 2018 | CNN recommends candidate merges and splits; the user answers yes or no. | Reduced variation of information 7.5 times faster than earlier methods. | Turn proofreading into ranked yes/no decisions. |
| Error detection and correction (Zung et al.) | 2017 | 3D CNNs that map errors in an object mask, then prune it to the true object. | High detection accuracy; correction improved when given the detector's output. | Keep detection and correction as separate stages. |

**Implications**

- Automate split correction first; it is the most reliable step and transfers best across datasets.
- Treat merge detection in fly as a human-in-the-loop task until your own measurements show otherwise.
- Use the language-model agent to orchestrate, gather evidence, apply the decision rules and explain
  its choices; use trained scorers for the yes/no judgements.
- Re-measure accuracy and calibration on every new dataset or brain region before trusting thresholds.

## Evaluation and quality control

Judge the harness by connectivity accuracy per unit of effort, measured against expert
re-proofreading, not by how many edits it makes. Report every metric below for each dataset and brain
region.

| Metric | What it measures | Reference value |
|---|---|---|
| Volume F1 against expert re-proofreading | Morphological correctness | 99.2% average for FlyWire first-round proofreading (826 neurons); median 0.99 in the 2022 study |
| Synapse-assignment F1 | Whether the right synapses sit on the neuron | Median 0.99 after one round (FlyWire 2022) |
| Synapse attachment rate per neuropil | Completeness | 92.3% of presynapses and 43.9% of postsynapses attached in FlyWire |
| Strong-connection recall | Whether strong connections survive | 99.7% of connections with more than 9 synapses kept at least one synapse; 83% of those with at least 3 (FlyWire 2022) |
| Edit precision | Share of agent edits that experts keep | Use "never reversed" as the label, as ConnectomeBench2 did |
| Decision accuracy and calibration | Quality of each yes/no merge or split call | Human experts: 93.0% on splits, 84.1% on merges (ConnectomeBench2); also report expected calibration error |
| Variation of information, ERL/nERL | Global split and merge rates of a segmentation | 94.2% axon nERL in PATHFINDER (mouse) |
| Cost | Minutes and edits per neuron | 21 minutes per neuron (fast protocol); about 79 seconds per edit (FlyWire) |
| Cross-hemisphere consistency | Biological plausibility | Connections over 10 synapses, or over 1% of a cell's input, are highly conserved across connectomes (Schlegel 2024) |

**Quality-control process**

- Have experts re-proofread a random sample plus neurons flagged by metrics. FlyWire re-checked 826
  random neurons and 3,106 flagged segments.
- Hold out spatially separate test regions; ConnectomeBench2 split data into 80 µm cubes to prevent
  leakage between nearby edits.
- Re-review neurons that changed substantially after being marked complete.
- Analyse connections of at least 5 synapses. Errors are roughly independent, so connections with many
  synapses are the most reliable.
- Track each agent's accept and reject rates over time to catch drift.

## Resources

Start with the training material, then the papers for the dataset you work on. (Links omitted here:
the sandbox has no internet.)

**Training material**: Glossary of fly ultrastructure and morphology (neuron morphology,
ultrastructure and tracing tips); Errors & issues in EM (linked examples of glia merges, bad slices,
autofasciculation, twig merges, unsegmented regions); Faster proofreading (the 21-minute-per-neuron
protocol); Fly synapses (synapse indicators with EM examples and counter-examples); FlyWire
Proofreading Tips (X and H merges, misalignments, twig merges, invaginations, soma tracts, pinched
neurites, path swaps); FlyWire FAQ (completion rule, R-cell exception, misalignment trick, display
fixes); To trace, or not to trace? (when to stop proofreading a cell); Expert proofreading tips for
BANC (mitochondria cues, path swaps, completeness in the brain–VNC dataset); BANC proofreading 101:
Spelunker (interface, split/merge keys, find-path, natural endings); FlyWire 101 (interface basics,
supervoxels, merging and splitting); MICrONS proofreading and data quality (compartment-level
clean/extended status model); Hemibrain dataset introduction (body-status definitions).

**Datasets and core papers**: Zheng et al. 2018, Cell (the FAFB image volume); Li et al., FFN
segmentation of FAFB (local realignment and synthesized imagery for bad sections); Dorkenwald et al.
2022, Nature Methods (FlyWire, the ChunkedGraph, proofreading accuracy and effort); Dorkenwald et al.
2024, Nature (the whole-brain connectome; proofreading rounds, accuracy, completeness); Schlegel et al.
2024, Nature (cell types, hemilineages, cross-connectome typing); Matsliah et al. 2024, Nature (optic
lobe cell types and wiring); Dorkenwald et al. 2025, Nature Methods (CAVE: versioned proofreading and
annotation infrastructure); Buhmann et al. 2021, Nature Methods (automated synaptic partner detection
in FAFB); Eckstein et al. 2024, Cell (neurotransmitter prediction from EM); Schneider-Mizell et al.
2016, eLife (backbones vs twigs and which errors matter for connectivity); Scheffer et al. 2020, eLife
(hemibrain pipeline, proofreading protocols and completeness tests); Azevedo et al. 2024, Nature (FANC
female VNC); Takemura et al., MANC (male VNC connectome); Bates et al. 2026, Nature (BANC); Berg et al.,
male CNS preprint (complete male CNS; 44 person-years of proofreading).

**Tools and data access**: Codex (browse FlyWire and other fly connectomes, cell types and
connections); CAVEclient (Python access to the ChunkedGraph, edits, annotations and materialization);
fafbseg-py (FlyWire meshes, skeletons, synapses, coordinate transforms; built on navis);
flywire_annotations (cell-type and class annotations for release 783); Neuroglancer (the web viewer
under FlyWire, Spelunker and Codex); Fly connectome data tutorial (loading BANC, FAFB, MANC and
hemibrain data); MICrONS materialization tutorial (CAVE versioning examples that carry over to fly
datasets); ConnectomeBench code and data (evaluation tasks for split, merge and segment typing); Male
CNS releases.

## Glossary

| Term | Meaning |
|---|---|
| Autofasciculation | Two parts of the same neuron running side by side; looks like a loop or merge but is real. |
| Backbone | Thick, microtubule-containing neurite; carries the neuron's overall shape. |
| Bouton | Swelling on an axon holding presynaptic sites. |
| CAVE | Connectome Annotation Versioning Engine: versioned segmentation, annotations and materialization. |
| ChunkedGraph | Chunked supervoxel graph with cached components; every edit changes it. |
| Cleft | Gap between pre- and postsynaptic membranes at a synapse. |
| Dense-core vesicle (DCV) | Large dark vesicle; not part of a synaptic vesicle cluster. |
| ERL / nERL | Expected (normalised) run length: average error-free path length in a segmentation. |
| Find-path | Tool that traces the route between two points through a neuron, used to locate merge bridges. |
| Hemilineage | Group of neurons born from one stem-cell branch; often share transmitter and projection. |
| L2 node | A neuron's piece within one base chunk of the ChunkedGraph. |
| Materialization | A versioned snapshot binding annotations, such as synapses, to the root IDs of that moment. |
| Merge error | Pieces of two or more cells joined in one segment. |
| Misalignment | Neighbouring sections shifted against each other; the main cause of splits and path swaps in FAFB. |
| Multicut | Split operation: source and sink points, then a min-cut on the supervoxel graph. |
| Neuropil | Synapse-dense brain region, e.g. medulla, lamina, mushroom body. |
| Orphan | Small fragment not attached to any proofread neuron. |
| Path swap | False continuation from one neuron onto a neighbouring, parallel one. |
| Polyadic synapse | One presynaptic site contacting several postsynaptic partners. |
| Proximal neurite (soma tract) | The single neurite linking a unipolar soma to the backbone. |
| PSD | Postsynaptic density: dark band on the receiving membrane. |
| Root ID | ID of a whole segment at a given time; changes after every edit. |
| Split error | One cell broken into two or more segments. |
| Supervoxel | Smallest indivisible segmentation unit. |
| T-bar | Dark presynaptic structure at fly active zones. |
| Twig | Thin, microtubule-free branch carrying most synaptic inputs. |
| Variation of information (VI) | Information-theoretic measure of split and merge error in a segmentation. |
