# FlyWire outer harness

A coding agent (the meta-agent) writes a harness that proofreads FlyWire edits; this repository is
the outer harness that runs it, scores it against 200 real FlyWire edits, and asks the meta-agent to
improve it. Design: [PLAN.md](PLAN.md). Status: [PROGRESS.md](PROGRESS.md).

## Secrets (never in the repository)

| What | Where | Read by |
|---|---|---|
| Anthropic API key | `~/.config/flywire_harness/anthropic_api_key` (one line, `chmod 600`) | `services/llm/proxy.py` only |
| CAVE token | `~/.cloudvolume/secrets/cave-secret.json` (caveclient's default) | `build/`, `services/world/`, `outer/score.py` |
| token-signing secret | `~/.config/flywire_harness/token_secret` (created automatically) | outer harness, both services |

To add the API key without leaving it in your shell history:

```sh
mkdir -p ~/.config/flywire_harness && chmod 700 ~/.config/flywire_harness
install -m 600 /dev/null ~/.config/flywire_harness/anthropic_api_key
nano ~/.config/flywire_harness/anthropic_api_key      # paste the key, save
```

## Setup (della-vis1)

```sh
cd <your clone of this repository>
/usr/licensed/anaconda3/2024.10/bin/python -m venv .venv
.venv/bin/pip install numpy pyyaml httpx anthropic caveclient cloud-volume pytest ruff   # pyproject.toml lists them
apptainer build outer/sandbox.sif outer/sandbox.def  # the sandbox image (~350 MB)
.venv/bin/python -m pytest -q                        # offline tests (fake world, fake Claude upstream)
```

Data (already built; rerun only if needed):

```sh
.venv/bin/python -m build.make_splits      # data/splits.json (frozen)
.venv/bin/python -m build.build_truth      # truth/<operation_id>.json (gitignored)
.venv/bin/python -m build.replace_splits   # done once: data/operations_v2.csv, splits.json v2 (surviving splits)
.venv/bin/python -m build.build_negatives --report  # truth/negatives/<negative_id>.json: 800 proofread-neuron no-edit tasks
.venv/bin/python -m build.build_states     # tasks/<task_id>.json + tasks/index.json (edit + no-edit tasks)
```

## Start the services

Both run on della-vis1 and stay up while you work (use `tmux` so they survive logout):

```sh
.venv/bin/python -m services.world.server > runs/world_server.log 2>&1 &
.venv/bin/python -m services.llm.proxy    > runs/llm_proxy.log 2>&1 &
```

## Run

```sh
.venv/bin/python -m outer.create --yes                      # Phase 5: meta-agent writes harnesses/h000 (<= $50)
.venv/bin/python -m outer.evaluate h000 train --tasks t_... # evaluate on some train tasks
.venv/bin/python -m outer.loop --yes                        # Phase 6: rounds within budget_usd.total
.venv/bin/python -m outer.loop --final --yes                # Phase 7: best vs h000 on test (reuses per-round test runs)
.venv/bin/python -m outer.report                            # runs/report.md
```

Each step that may cost more than `budget_usd.ask_above` refuses to start without `--yes`.

## Layout

```
config.yaml              all settings (prices, budgets, limits, loop rules, service sockets)
data/                    operations.csv (source copy), operations_v2.csv (used), splits.json (frozen, v2)
build/                   make_splits, build_truth, build_negatives, build_states (run once; read-only FlyWire)
truth/                   per-operation truth, negatives/, task_map.json (gitignored; build/ and score.py only)
tasks/                   task states (PLAN.md 4.1) and index.json (task ids per split)
services/world/          the world server, client library, API.md, cache
services/llm/            the proxy and the spend ledger
services/aliases.py      per-task opaque root/L2 ids (raw FlyWire ids carry their creation time)
outer/                   sandbox, evaluate, score, create, revise, feedback, loop, report, prompts/
                         (prompts/ also holds proofreading_guide.md, copied to the meta-agent's /work/docs/)
harnesses/               generated harness versions (tracked)
feedback/round_k/        what the meta-agent saw each round
runs/                    per-task results, meta sessions, loop state, report.md
ledger.jsonl             every Claude request with tokens and dollars (gitignored)
tests/                   fakes, test harnesses, tests
```
