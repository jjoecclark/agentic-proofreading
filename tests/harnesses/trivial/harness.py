"""Test-only trivial harness: looks at the world, asks the LLM once, and guesses the middle of the
first segment's L2 positions (a merge pairs it with the second segment's nearest L2 node)."""
import numpy as np


def run(state, world, llm, workdir, limits):
    segs = state["neuroglancer"]["layers"][1]["segments"]
    info = world.info()
    assert info["t_utc"][:19] == state["timestamp_utc"][:19]
    r = llm.messages.create(model=limits.models[0], max_tokens=64,
                            messages=[{"role": "user", "content": f"Task {state['task_id']} with {len(segs)} segments."}])
    _ = r.content[0].text
    pos = []
    for s in segs:
        g = world.l2_graph(s)
        p, _approx = world.l2_positions(g.l2_ids[:200])
        pos.append(p)
    if len(segs) == 2:
        d = np.linalg.norm(pos[0][:, None, :] - pos[1][None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        return {"decision_version": "1", "action": "merge", "points_nm_xyz": [pos[0][i].tolist(), pos[1][j].tolist()],
                "notes": "nearest L2 pair"}
    c = pos[0].mean(axis=0)
    return {"decision_version": "1", "action": "split", "sources_nm_xyz": [(c - [500, 0, 0]).tolist()],
            "sinks_nm_xyz": [(c + [500, 0, 0]).tolist()]}
