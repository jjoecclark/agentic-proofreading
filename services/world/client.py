"""Client for the `world` service: read-only FlyWire data frozen at one task's time T.

Standard library + numpy only. Inside the sandbox it is importable as ``world_client``; the harness
receives a ready ``WorldClient`` in ``run(...)``. See API.md for every call.

    from world_client import WorldClient
    world = WorldClient()                                   # reads WORLD_SOCKET / WORLD_URL, WORLD_TOKEN
    info = world.info()
    em = world.em([[x0, y0, z0], [x1, y1, z1]], resolution_nm_xyz=(16, 16, 40))
    em.data_zyx.shape, em.origin_nm_xyz, em.resolution_nm_xyz

Positions are in nm, (x, y, z) order. Arrays are indexed (z, y, x).
"""

from __future__ import annotations

import http.client
import io
import json
import os
import socket
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

WORLD_API_VERSION = "2"  # 2: root and L2 ids are opaque per-task aliases


class WorldError(Exception):
    """The request failed (bad arguments, server error)."""

    def __init__(self, status: int, kind: str, message: str):
        super().__init__(f"[{status} {kind}] {message}")
        self.status, self.kind, self.message = status, kind, message


class WorldRefused(WorldError):
    """The world will not answer this (history, other times, roots not valid at T, writes)."""


@dataclass
class Volume:
    """A cutout. ``data_zyx[k, j, i]`` is the voxel whose lower corner is at
    ``origin_nm_xyz + (i, j, k) * resolution_nm_xyz``."""

    data_zyx: np.ndarray
    origin_nm_xyz: np.ndarray
    resolution_nm_xyz: np.ndarray

    def index_of(self, point_nm_xyz: Sequence[float]) -> tuple[int, int, int]:
        """(z, y, x) index of the voxel containing a point in nm (may be out of range)."""
        i = np.floor((np.asarray(point_nm_xyz, dtype=float) - self.origin_nm_xyz) / self.resolution_nm_xyz)
        return int(i[2]), int(i[1]), int(i[0])

    def center_nm_of(self, index_zyx: Sequence[int]) -> np.ndarray:
        """Center in nm (x, y, z) of the voxel at a (z, y, x) index."""
        z, y, x = index_zyx
        return self.origin_nm_xyz + (np.array([x, y, z], dtype=float) + 0.5) * self.resolution_nm_xyz


@dataclass
class L2Graph:
    l2_ids: np.ndarray   # uint64, every L2 node of the root
    edges: np.ndarray    # uint64 (N, 2), pairs of connected L2 ids


@dataclass
class Mesh:
    vertices_nm_xyz: np.ndarray  # float32 (N, 3)
    faces: np.ndarray            # int32 (M, 3), indices into vertices


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


def _ids(values: Iterable[Any]) -> list[str]:
    if isinstance(values, np.ndarray):
        values = values.astype(np.uint64).tolist()
    return [str(int(v)) for v in values]


class WorldClient:
    def __init__(self, socket_path: str | None = None, url: str | None = None, token: str | None = None,
                 timeout_s: float = 600.0, retries: int = 2):
        self.socket_path = socket_path or os.environ.get("WORLD_SOCKET")
        self.url = url or os.environ.get("WORLD_URL")
        self.token = token or os.environ.get("WORLD_TOKEN")
        if not (self.socket_path or self.url):
            raise ValueError("set WORLD_SOCKET or WORLD_URL (or pass socket_path / url)")
        if not self.token:
            raise ValueError("set WORLD_TOKEN (or pass token)")
        self.timeout_s, self.retries = timeout_s, retries

    # -- transport

    def _conn(self) -> http.client.HTTPConnection:
        if self.socket_path:
            return _UnixHTTPConnection(self.socket_path, self.timeout_s)
        u = self.url.split("://", 1)[-1]
        host, _, port = u.partition(":")
        return http.client.HTTPConnection(host, int(port.rstrip("/") or 80), timeout=self.timeout_s)

    def _call(self, endpoint: str, body: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                c = self._conn()
                c.request("POST", f"/v1/{endpoint}", body=data, headers={
                    "authorization": f"Bearer {self.token}", "content-type": "application/json"})
                r = c.getresponse()
                payload = r.read()
                ctype = r.getheader("content-type", "")
                meta = r.getheader("x-world-meta")
                c.close()
            except (TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
                last = e
                time.sleep(1.0 + attempt)
                continue
            if r.status == 200:
                if ctype.startswith("application/x-npz"):
                    z = np.load(io.BytesIO(payload))
                    return (json.loads(meta) if meta else None), {k: z[k] for k in z.files}
                return json.loads(payload), None
            try:
                err = json.loads(payload)["error"]
            except Exception:  # noqa: BLE001
                err = {"type": "unknown", "message": payload[:500].decode(errors="replace")}
            if r.status >= 500 and attempt < self.retries:
                last = WorldError(r.status, err.get("type", ""), err.get("message", ""))
                time.sleep(1.0 + attempt)
                continue
            cls = WorldRefused if r.status == 403 else WorldError
            raise cls(r.status, err.get("type", ""), err.get("message", ""))
        raise WorldError(0, "unreachable", f"world service unreachable: {last}")

    # -- API (see API.md)

    def info(self) -> dict[str, Any]:
        """T, dataset bounds (nm), available resolutions per layer, layer descriptions."""
        js, _ = self._call("info", {})
        return js

    def _volume(self, endpoint: str, bbox_nm_xyz: Any, resolution_nm_xyz: Sequence[float]) -> Volume:
        _, a = self._call(endpoint, {"bbox_nm_xyz": np.asarray(bbox_nm_xyz, dtype=float).tolist(),
                                     "resolution_nm_xyz": [float(x) for x in resolution_nm_xyz]})
        return Volume(a["data"], a["origin_nm_xyz"], a["resolution_nm_xyz"])

    def em(self, bbox_nm_xyz: Any, resolution_nm_xyz: Sequence[float] = (16, 16, 40)) -> Volume:
        """uint8 EM cutout of the box [[x0, y0, z0], [x1, y1, z1]] (nm), indexed (z, y, x)."""
        return self._volume("em", bbox_nm_xyz, resolution_nm_xyz)

    def supervoxels(self, bbox_nm_xyz: Any, resolution_nm_xyz: Sequence[float] = (16, 16, 40)) -> Volume:
        """uint64 supervoxel-id cutout (0 = background), indexed (z, y, x)."""
        return self._volume("supervoxels", bbox_nm_xyz, resolution_nm_xyz)

    def segmentation(self, bbox_nm_xyz: Any, resolution_nm_xyz: Sequence[float] = (16, 16, 40)) -> Volume:
        """uint64 root id per voxel at T (supervoxels mapped through roots()), indexed (z, y, x)."""
        return self._volume("segmentation", bbox_nm_xyz, resolution_nm_xyz)

    def roots(self, supervoxel_ids: Iterable[Any]) -> np.ndarray:
        """Root id (uint64) of each supervoxel at T; 0 stays 0."""
        _, a = self._call("roots", {"supervoxel_ids": _ids(supervoxel_ids)})
        return a["root_ids"]

    def l2_graph(self, root_id: Any) -> L2Graph:
        """L2 chunk ids and edges of a root that is valid at T."""
        _, a = self._call("l2_graph", {"root_id": str(int(root_id))})
        return L2Graph(a["l2_ids"], a["edges"].reshape(-1, 2))

    def l2_positions(self, l2_ids: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
        """(positions_nm_xyz (N, 3), approximate (N,) bool) for L2 ids returned by l2_graph."""
        _, a = self._call("l2_positions", {"l2_ids": _ids(l2_ids)})
        return a["positions_nm_xyz"].reshape(-1, 3), a["approximate"]

    def mesh(self, root_id: Any) -> Mesh:
        """Mesh of a root valid at T: vertices in nm (x, y, z) and triangle faces."""
        _, a = self._call("mesh", {"root_id": str(int(root_id))})
        return Mesh(a["vertices_nm_xyz"].reshape(-1, 3), a["faces"].reshape(-1, 3))
