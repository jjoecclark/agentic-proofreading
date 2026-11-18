<!-- prompt_version 2: the meta-agent's instructions for round 0 (outer/create.py) -->
You are building the first version of a FlyWire proofreading harness.

Read `/work/TASK.md` first: it defines the task, the decision format, the scoring rule, the harness
interface, and your limits. Then read `/work/API.md`. `/work/docs/proofreading_guide.md` is background on
FlyWire proofreading; use what helps.

Write the harness in `/work/harness/` (`harness.py` with `run(...)`, plus any other files) and
`/work/harness/NOTES.md`. Explore the train data with the world client and test your harness with
`score_train` on train tasks. Measure before you add complexity: a harness that reliably returns a
well-formed guess beats one that crashes or times out.

When you are done, `/work/harness/harness.py` must run end to end on train tasks without crashing,
and NOTES.md must describe the design and the results you measured. Only files in `/work/harness/`
are kept. Finish with a short summary of what you built and how it scored.
