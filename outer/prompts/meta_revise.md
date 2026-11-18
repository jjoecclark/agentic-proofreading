<!-- prompt_version 2: the meta-agent's instructions for round k >= 1 (outer/revise.py) -->
You are revising a FlyWire proofreading harness (round {round}).

`/work/harness/` holds the current best harness ({best}), which you wrote in an earlier round. Read
`/work/TASK.md` (the task and rules; its "Changes" section lists what changed, possibly since the
harness was written), then `/work/feedback/README.md`, which explains the
feedback for this round: per-task train results with transcripts, the hit rates of every past round
on train and on held-out validation tasks, and which revisions were kept or rejected and why.

Improve the harness so that it scores better on **held-out** tasks. A revision is kept only if its
validation hits beat the current best by at least {accept_min_gain} of {n_val} tasks; otherwise the
next round starts again from the current best. You never see per-task validation results, so improve
what generalizes: diagnose the failure modes on train, fix their causes, and confirm with
`score_train`.

`/work/docs/proofreading_guide.md` is background on FlyWire proofreading; use what helps.

Update `/work/harness/NOTES.md`: what you changed, why, and what you measured. Only files in
`/work/harness/` are kept. Finish with a short summary of the change and its train results.
