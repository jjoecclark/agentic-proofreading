# `world` API (world_api_version 2)

The `world` service gives read-only FlyWire data **as of one moment, the task's time T**. Your
harness receives a ready `WorldClient` (module `world_client`) bound to the task it is solving.

Conventions:

- Positions are in **nanometers, (x, y, z) order**. Boxes are `[[x0, y0, z0], [x1, y1, z1]]` in nm.
- Arrays are indexed **(z, y, x)**.
- Root, supervoxel, and L2 ids are 64-bit unsigned integers. Keep them as Python `int`, `numpy.uint64`,
  or decimal strings; never as floats.
- **Root and L2 ids are opaque and per task** (since version 2): every call in a task uses the same id
  for the same segment or L2 node, including the segment ids in the task state, but another task sees
  different ids, and the digits carry no information (they are not FlyWire's ids). Supervoxel ids are
  FlyWire's own. 0 (background) is 0.
- Resolutions are `(x, y, z)` nm per voxel and must be one of the levels listed by `info()`.

## Calls

| Call | Returns |
|---|---|
| `info()` | dict: `t_utc` (T), `bounds_nm_xyz`, `layers.em.levels`, `layers.supervoxels.levels` (each level: `resolution_nm_xyz`, `voxel_offset_xyz`, `size_xyz`, `chunk_size_xyz`), `l2_chunk_size_nm_xyz` |
| `em(bbox_nm_xyz, resolution_nm_xyz=(16,16,40))` | `Volume`: uint8 EM image |
| `supervoxels(bbox_nm_xyz, resolution_nm_xyz=(16,16,40))` | `Volume`: uint64 supervoxel ids (0 = background) |
| `segmentation(bbox_nm_xyz, resolution_nm_xyz=(16,16,40))` | `Volume`: uint64 root id per voxel at T (supervoxels mapped through `roots`) |
| `roots(supervoxel_ids)` | uint64 array: root id of each supervoxel at T (0 stays 0) |
| `l2_graph(root_id)` | `L2Graph`: `l2_ids` (uint64, all L2 chunks of the root) and `edges` (uint64, shape (N, 2)) |
| `l2_positions(l2_ids)` | `(positions_nm_xyz (N, 3), approximate (N,) bool)` |
| `mesh(root_id)` | `Mesh`: `vertices_nm_xyz` (float32, (N, 3)) and `faces` (int32, (M, 3)) |

`Volume` has `data_zyx`, `origin_nm_xyz` (lower corner of voxel [0, 0, 0]), `resolution_nm_xyz`, and
helpers `index_of(point_nm_xyz) -> (z, y, x)` and `center_nm_of((z, y, x)) -> point_nm_xyz`. The box is
snapped outward to whole voxels; areas outside the dataset read as 0.

## Data notes (FlyWire FAFB, public datastack)

- EM levels run from 4 x 4 x 40 nm to 2048 x 2048 x 1280 nm. Some chunks are **missing at 4 nm** and
  read as all zeros; use 8 nm or coarser if a 4 nm cutout is blank. Supervoxel levels run from
  16 x 16 x 40 nm upward (z stays 40 nm).
- One cutout may hold at most 64 Mi voxels; `segmentation` may contain at most 100,000 distinct
  supervoxels. Use coarser levels for overviews.
- An **L2 chunk** is a piece of one root inside one graph chunk (4096 x 4096 x 20480 nm). The L2 graph
  is the root's skeleton-like adjacency at that granularity. Segments in these tasks have 100 to
  several thousand L2 chunks.
- `l2_positions`: there is no L2 cache on this datastack. Each position is the centroid of that L2
  chunk's own mesh fragment; chunks without a fragment (tiny ones, ~3%) get their graph chunk's center
  and `approximate=True`. Only ids returned by `l2_graph` for this task are accepted. First calls on a
  large root take ~10 s; results are cached.
- `mesh` returns the mesh fragments of the root concatenated, with duplicate vertices along chunk
  boundaries (not merged). A large neuron can have ~10^6 vertices; first calls take seconds.
- Every root you pass to `l2_graph` or `mesh` must be valid at T (the task's segments are).

## Refused, always (`WorldRefused`, HTTP 403)

- lineage, operation records, edit history, "latest root", root creation and expiry times
- any query about a time other than T (any `timestamp`-like field)
- any root not valid at T
- any write (merge or split) to FlyWire

Other failures raise `WorldError` (bad arguments: HTTP 400; server problems: HTTP 5xx, retried twice).

## Example

```python
import numpy as np

def look_around(world, root_id, center_nm_xyz, half_nm=(2000, 2000, 400)):
    c, h = np.asarray(center_nm_xyz, float), np.asarray(half_nm, float)
    box = [(c - h).tolist(), (c + h).tolist()]
    em = world.em(box, (8, 8, 40))                  # uint8, (z, y, x)
    seg = world.segmentation(box, (16, 16, 40))     # root ids at T, (z, y, x)
    mask = seg.data_zyx == np.uint64(root_id)
    return em, seg, mask

g = world.l2_graph(root_id)
pos, approx = world.l2_positions(g.l2_ids)
```
