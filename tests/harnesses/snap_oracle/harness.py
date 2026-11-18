"""Test-only oracle for REAL tasks: answers.json (written by tests/test_live.py from truth/) gives the
edit center and the edited supervoxels; the harness places each point on the voxel of those
supervoxels nearest to the center, using only world.supervoxels(). Real harnesses never see truth."""
import json
import pathlib

import numpy as np


def nearest_voxel(vol, ids, center):
    mask = np.isin(vol.data_zyx, np.array([int(i) for i in ids], dtype=np.uint64))
    zyx = np.argwhere(mask)
    if not len(zyx):
        return None
    pts = vol.origin_nm_xyz + (zyx[:, ::-1] + 0.5) * vol.resolution_nm_xyz
    return pts[int(np.argmin(np.linalg.norm(pts - center, axis=1)))].tolist()


def run(state, world, llm, workdir, limits):
    a = json.loads((pathlib.Path(__file__).parent / "answers.json").read_text())[state["task_id"]]
    if a["type"] == "none":  # a negative task: no edit needed
        return {"decision_version": "1", "action": "none"}
    c = np.array(a["center"], dtype=float)
    for half in (1500.0, 3000.0):  # 3 um half-width keeps the cutout under the 64 Mi-voxel limit
        h = np.array([half, half, max(half, 800.0)])
        vol = world.supervoxels([(c - h).tolist(), (c + h).tolist()], (16, 16, 40))
        if a["type"] == "merge":
            p = [nearest_voxel(vol, [e[0] for e in a["added_edges"]], c), nearest_voxel(vol, [e[1] for e in a["added_edges"]], c)]
            if all(x is not None for x in p):
                return {"decision_version": "1", "action": "merge", "points_nm_xyz": p}
        else:
            s, k = nearest_voxel(vol, a["sources"], c), nearest_voxel(vol, a["sinks"], c)
            if s is not None and k is not None:
                return {"decision_version": "1", "action": "split", "sources_nm_xyz": [s], "sinks_nm_xyz": [k]}
    return {"decision_version": "1", "action": "none", "notes": "edited supervoxels not found near the center"}
