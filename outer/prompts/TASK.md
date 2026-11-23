# TASK: build a FlyWire proofreading harness

<!-- prompt_version 5 (task version 5; see "Changes" at the end). Placeholders in {braces} are filled from config.yaml by outer/create.py. -->

## The task

FlyWire is a whole-brain reconstruction of an adult fruit fly (FAFB electron-microscopy volume).
Automatic segmentation cuts the EM volume into **supervoxels**, and a graph joins supervoxels into
**segments** (root ids), ideally one per neuron. Human proofreaders fix two kinds of errors:

- a **merge** edit joins two segments that are really one neuron (the automatic segmentation had
  split it);
- a **split** edit cuts one segment that really contains parts of two different cells (the automatic
  segmentation had merged them). A split is specified by **source** points on one side and **sink**
  points on the other; FlyWire then cuts the segment between them.

Each task shows you one or two segments at a moment T of FlyWire's history and asks one question:

- **2 segments: should they be merged?** If they are parts of one cell, return two points, one on
  each segment, at the place where they should be joined. If they are different cells, return `none`.
- **1 segment: should it be split?** If it holds parts of two different cells (a false merge),
  return source points and sink points on the segment, on the two sides of the place where it should
  be cut. If it is one cell, return `none`.

When an edit is needed, it is a real edit that a human proofreader made right after T, and that is
the edit to find (there is one). When no edit is needed, the segments are neurons that FlyWire's
published connectome (materialization 783) lists as proofread, unchanged since before T: two different
neurons that touch (they form synapses with each other), or one neuron with nothing to split off.
{n_train_negative} of the {n_train} train tasks need no edit; held-out tasks have the same mix. The edits
are real proofreading actions and a few may themselves be mistakes, so no harness can score 100%.

## Your job

Write a harness: Python code in `/work/harness/` that solves one task per call. It may call Claude,
render images, analyze meshes, skeletonize, run several agents, anything that fits the limits below.
Also write `/work/harness/NOTES.md`: your design, what you tried, what you measured, what you learned.
Future revisions (by you) will read it.

## Input: the task state

`run()` receives the task as a dict (also in `/work/tasks/<task_id>.json` for train tasks):

```json
{"state_version": "1", "task_id": "t_8c41e2a7", "timestamp_utc": "2022-11-14T01:46:39.753000+00:00",
 "neuroglancer": {"layers": [
   {"type": "image", "name": "em", "source": "precomputed://..."},
   {"type": "segmentation", "name": "segments", "source": "graphene://...",
    "segments": ["720575940610776440", "720575940624101559"], "timestamp": "1668390399.753000"}]}}
```

There is no position hint: you must decide whether an edit is needed and find its location yourself.
Segment ids are decimal strings (64-bit; never convert them to float). They are opaque ids for this
task: the world API understands them, the same segment has a different id in every task, and their
digits carry no information.

## Output: the decision

`run()` returns a dict. Positions are in **nm, (x, y, z)**.

```json
{"decision_version": "1", "action": "merge", "points_nm_xyz": [[x, y, z], [x, y, z]]}
{"decision_version": "1", "action": "split", "sources_nm_xyz": [[x, y, z]], "sinks_nm_xyz": [[x, y, z]]}
{"decision_version": "1", "action": "none"}
```

An optional `"notes"` string is stored but not scored.

## Scoring (hit radius R = {radius_um} µm)

A point "lies on segment X" when the supervoxel at that point (16 x 16 x 40 nm supervoxel layer)
belongs to X at time T. A point on background lies on no segment.

- **Merge hit:** action `merge`, exactly 2 points, one on each of the two segments, and **both
  points within R of the true edit location**.
- **Split hit:** action `split`, at least one source and one sink, every point on the segment, all
  sources end up on one side and all sinks on the other side of the true cut (as FlyWire's own cut
  separated them just after the edit), and **the midpoint of the closest source-sink pair within R of
  the true edit location**.
- **No-edit hit:** action `none` on a task that needs no edit. Any merge or split there is a miss
  (`edit_not_needed`).
- Anything else is a miss: `none` on a task that needs an edit, the wrong action, invalid output, a
  crash, a timeout, or an exhausted budget. A crash or timeout is a miss on a no-edit task too.

Your **score** is the mean of the hit rates of the four task kinds (`merge`, `split`, `no_merge`,
`no_split`), so each kind weighs the same however many tasks it has. Always answering `none` scores 50%.

The true edit location is the mean of the points the human proofreader placed. Reported per task:
its kind (`merge`, `split`, `no_merge`, `no_split`), hit/miss, the miss reason, cost, and wall time.
The distance to the true location is not reported.

Your harness is judged on **held-out tasks** you never see. Train tasks are for development only.

## Harness interface

```
/work/harness/
  harness.py   # def run(state: dict, world: WorldClient, llm: anthropic.Anthropic,
               #         workdir: pathlib.Path, limits: Limits) -> dict
  NOTES.md
  ...          # anything else: prompts, modules, data files (keep the folder under 200 MB)
```

- `world`: a `world_client.WorldClient` bound to this task's time T. Read `/work/API.md`.
- `llm`: an `anthropic.Anthropic` client (standard SDK). Allowed model: **`{model}`** only.
- `workdir`: an empty, writable scratch directory (the current directory too).
- `limits`: `limits.wall_s`, `limits.deadline_unix`, `limits.remaining_s()`, `limits.usd_cap`,
  `limits.models`, `limits.usage()` (spend so far, from the proxy), `limits.remaining_usd()`.
- The harness folder is mounted read-only at `/harness` and is on `sys.path`; import your own modules
  normally. `world_client` is importable too.

## Limits and environment

- Wall time: **{wall_s:.0f} s per task**; the process is killed at the limit (a timeout is a miss).
  Return a best guess before the deadline.
- Money: **${per_task_usd:.2f} per task** for Claude calls, enforced by the proxy. Before each request
  the proxy reserves its worst case: estimated input at the cache-write price plus `max_tokens` of
  output at ${out_price:.0f}/M tokens. A request whose worst case would exceed what is left is refused
  with HTTP 402 (`anthropic.APIStatusError`, status 402, type `budget_exceeded`). Size `max_tokens`
  to what you need and catch that error.
- Pricing for `{model}`: input ${in_price:.2f}/M, output ${out_price:.2f}/M, cache read
  ${cache_read:.2f}/M, cache write ${cache_write:.2f}/M (5-minute TTL) tokens.
- Spending well (measured on earlier harnesses: about two thirds of their cost was output tokens, and
  only half of their input came from cache):
  - **Prompt caching is shared across tasks.** Every task uses the same account, and tasks run
    {parallel} at a time, so a prefix that is byte-identical across tasks (fixed instructions, reference
    text, examples) is written once and then read at ${cache_read:.2f}/M instead of ${in_price:.2f}/M. Put it
    first and mark its end with `cache_control: {"type": "ephemeral"}`; put the task's own images and
    data after it. Prefixes under 512 tokens are not cached, and any change in the prefix (a timestamp,
    a task id, a different tool list) makes it miss. In a multi-turn loop, mark the latest turn too.
  - **Output costs most.** `{model}` always thinks; `output_config: {"effort": "low" | "medium" |
    "high" | ...}` sets how much (the default is `medium`). Use the lowest effort that keeps accuracy
    for each kind of call, ask for short answers, and size `max_tokens` to the answer you need.
  - `usage.cache_read_input_tokens` in each response shows whether the cache was hit.
- If the harness runs Claude Code / the Claude Agent SDK, set `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (for
  example 16000) in its environment: by default it asks for 128000 output tokens per request, whose
  worst case ($2.56) exceeds the per-task budget, so every request would be refused.
- The proxy accepts only `POST /v1/messages` (streaming or not) and `count_tokens`. Server-side tools
  (web search, web fetch, code execution, MCP, fallbacks) are refused; define your own client tools.
  Images (base64 PNG/JPEG) are fine.
- **No network** besides the two services. The sandbox has no internet.
- Python 3.12 with numpy, scipy, scikit-image, pillow, matplotlib (Agg), networkx, trimesh, pyyaml,
  anthropic, and claude-agent-sdk. There is no GPU. You cannot install packages at evaluation time.
- Tasks run in parallel, each in a fresh sandbox; nothing persists between tasks.

## Tools you have now

- `/work/tasks/`: the train task files (`/work/tasks/EXAMPLES` lists a few to start with, covering
  every kind of task).
- `/work/docs/proofreading_guide.md`: a guide for agents and people who proofread FlyWire (biology,
  error types, decision rules, what has worked in the literature). Its first section says which parts
  apply here and what data you do not have. Your harness may use any of it, for example in prompts.
- `/work/API.md` and `/work/world_client.py`: the world API.
- `/work/world_tokens.json`: world tokens for train tasks, to explore data yourself:
  `WorldClient(token=json.load(open('/work/world_tokens.json'))['t_...'])` (WORLD_SOCKET is set).
- `score_train <task_id> ...`: snapshots `/work/harness`, runs it on those train tasks exactly as the
  evaluation does (fresh sandbox per task, same limits), and prints hit/miss, miss reason,
  cost, and wall time per task. Details per task (decision, transcript, log, error) land in
  `/probe/results/<request>/<task_id>/`. It spends from your probe budget
  (**${probe_usd:.2f}** for this session); runs of up to ~{parallel} tasks proceed in parallel.
- Your own Claude usage in this session is capped at **${session_usd:.2f}**.

## Changes

- Version 2: tasks that need no edit (answer `none`); segment and L2 ids are opaque per task (world API
  version 2); the proofreading guide in `/work/docs/`.
- Version 3: new no-edit tasks (proofread neurons: two synapsing neurons, or one neuron), four times as
  many as edit tasks; the score is the mean of the per-kind hit rates.
- Version 4: `score_train` and the feedback no longer report the distance to the true location.
- Version 5: concrete advice on prompt caching across tasks and on `effort` (Limits and environment).
