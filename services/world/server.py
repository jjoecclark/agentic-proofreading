"""The `world` service (PLAN.md Section 6.1): read-only FlyWire data, frozen at a task's time T.

Every request carries a world session token (services/tokens.py) that names one task and its T.
All answers are as of T. Root and L2 ids are opaque per-task aliases (services/aliases.py): raw ids
carry their creation time, which would tell a harness which segments an edit had just produced.
There are no endpoints for lineage, operation records, edit history, latest roots, root timestamps,
other times, or writes; such requests are refused (HTTP 403), and so is any request that mentions a
timestamp or names a root that is not valid at T.

    WorldService  — the policy layer (time binding, refusals, size limits, request log)
    FlyWireBackend — real data via caveclient + cloud-volume, with an on-disk cache
    tests/fakes.py — FakeBackend with the same interface, used by the tests

Run:  .venv/bin/python -m services.world.server            (reads config.yaml)
"""

from __future__ import annotations

import argparse
import io
import json
import signal
import sqlite3
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import numpy as np

from services.aliases import l2_of, roots_of
from services.httpserver import JSONHandler, serve
from services.tokens import TokenError, verify

WORLD_API_VERSION = "2"  # 2: per-task opaque root and L2 ids

ENDPOINTS = ("info", "em", "supervoxels", "roots", "segmentation", "l2_graph", "l2_positions", "mesh")
FORBIDDEN_KEYS = {"timestamp", "time", "t", "t_utc", "timestamp_utc", "at", "as_of", "since", "until",
                  "start_timestamp", "end_timestamp", "latest"}


class Refused(Exception):
    """A request the world will not answer (HTTP 403)."""


class Invalid(Exception):
    """A malformed request (HTTP 400)."""


def parse_utc(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=UTC)


# ----------------------------------------------------------------------------- backend interface


class Backend(Protocol):
    def static_info(self) -> dict[str, Any]: ...
    def em(self, lo_vox_xyz: np.ndarray, hi_vox_xyz: np.ndarray, level: int) -> np.ndarray: ...
    def supervoxels(self, lo_vox_xyz: np.ndarray, hi_vox_xyz: np.ndarray, level: int) -> np.ndarray: ...
    def roots(self, supervoxel_ids: np.ndarray, t: datetime) -> np.ndarray: ...
    def roots_valid_at(self, root_ids: np.ndarray, t: datetime) -> np.ndarray: ...
    def l2_graph(self, root_id: int) -> tuple[np.ndarray, np.ndarray]: ...
    def l2_positions(self, l2_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def mesh(self, root_id: int) -> tuple[np.ndarray, np.ndarray]: ...


def levels_of(info: dict[str, Any], layer: str) -> list[list[float]]:
    return [lv["resolution_nm_xyz"] for lv in info["layers"][layer]["levels"]]


# ----------------------------------------------------------------------------- the policy layer


class LRU(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def put(self, k: Any, v: Any) -> None:
        self[k] = v
        self.move_to_end(k)
        while len(self) > self.maxsize:
            self.popitem(last=False)


class WorldService:
    def __init__(self, backend: Backend, secret: bytes, log_path: Path | None = None,
                 max_cutout_voxels: int = 64 * 2**20, max_roots_per_call: int = 100_000,
                 session_dir: Path | None = None):
        self.backend, self.secret = backend, secret
        self.max_cutout_voxels, self.max_roots_per_call = max_cutout_voxels, max_roots_per_call
        self.log_path = Path(log_path) if log_path else None
        self._log_lock = threading.Lock()
        self._valid = LRU(200_000)
        self._vlock = threading.Lock()
        self.session_dir = Path(session_dir) if session_dir else None
        self._l2_seen: dict[str, set[int]] = {}
        self._l2_lock = threading.Lock()
        self._info = backend.static_info()

    # -- auth / logging

    def session(self, headers: Any) -> dict[str, Any]:
        auth = headers.get("authorization") or ""
        tok = auth[7:] if auth.lower().startswith("bearer ") else headers.get("x-world-token")
        if not tok:
            raise PermissionError("missing world session token")
        try:
            p = verify(tok, self.secret, "world")
        except TokenError as e:
            raise PermissionError(str(e)) from e
        p["t"] = parse_utc(p["t_utc"])
        p["roots"], p["l2"] = roots_of(self.secret, p["task_id"]), l2_of(self.secret, p["task_id"])
        return p

    def log(self, rec: dict[str, Any]) -> None:
        if not self.log_path:
            return
        line = json.dumps({"world_log_version": WORLD_API_VERSION, "ts": time.time(), **rec}, default=str)
        with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    # -- checks

    @staticmethod
    def check_no_time(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in FORBIDDEN_KEYS:
                    raise Refused(f"queries about other times are refused (field {k!r}); every answer is as of "
                                  f"the task's time T")
                WorldService.check_no_time(v)
        elif isinstance(obj, list):
            for v in obj:
                WorldService.check_no_time(v)

    def _level(self, layer: str, resolution: Any) -> tuple[int, np.ndarray]:
        levels = levels_of(self._info, layer)
        if resolution is None:
            raise Invalid(f"resolution is required; available for {layer}: {levels}")
        try:
            res = [float(x) for x in resolution]
        except (TypeError, ValueError) as e:
            raise Invalid("resolution must be [x, y, z] in nm") from e
        for i, lv in enumerate(levels):
            if np.allclose(lv, res):
                return i, np.asarray(lv, dtype=float)
        raise Invalid(f"resolution {res} is not available for {layer}; available: {levels}")

    def _bbox(self, bbox_nm: Any, res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            b = np.asarray(bbox_nm, dtype=float)
        except (TypeError, ValueError) as e:
            raise Invalid("bbox_nm must be [[x0, y0, z0], [x1, y1, z1]] in nm") from e
        if b.shape != (2, 3) or not np.all(np.isfinite(b)):
            raise Invalid("bbox_nm must be [[x0, y0, z0], [x1, y1, z1]] in nm")
        lo = np.floor(b[0] / res).astype(np.int64)
        hi = np.ceil(b[1] / res).astype(np.int64)
        if np.any(hi <= lo):
            raise Invalid("bbox_nm must have max > min on every axis")
        n = int(np.prod(hi - lo))
        if n > self.max_cutout_voxels:
            raise Invalid(f"cutout of {n} voxels exceeds the limit of {self.max_cutout_voxels}; "
                          f"use a coarser resolution or a smaller box")
        return lo, hi

    @staticmethod
    def _ids(values: Any, name: str) -> np.ndarray:
        if not isinstance(values, list):
            raise Invalid(f"{name} must be a list of ids (strings or integers)")
        try:
            return np.array([int(v) for v in values], dtype=np.uint64)
        except (TypeError, ValueError, OverflowError) as e:
            raise Invalid(f"{name} must be a list of ids (strings or integers)") from e

    def require_valid_roots(self, roots: np.ndarray, t: datetime, shown: np.ndarray | None = None) -> None:
        """Refuse unless every (real) root is valid at t; `shown` are the ids as the client sent them."""
        key_t = t.isoformat()
        unknown = [int(r) for r in roots if (int(r), key_t) not in self._valid]
        if unknown:
            ok = self.backend.roots_valid_at(np.array(unknown, dtype=np.uint64), t)
            with self._vlock:
                for r, v in zip(unknown, ok, strict=True):
                    self._valid.put((r, key_t), bool(v))
        shown = roots if shown is None else shown
        bad = [str(int(s)) for r, s in zip(roots, shown, strict=True) if not self._valid.get((int(r), key_t), False)]
        if bad:
            raise Refused(f"root(s) {bad[:5]} are not valid at the task's time T; only roots that exist at T "
                          f"can be queried")

    def _remember_l2(self, task_id: str, l2_ids: np.ndarray) -> None:
        with self._l2_lock:
            s = self._l2_seen.setdefault(task_id, set())
            s.update(int(x) for x in l2_ids)
            if self.session_dir:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                np.save(self.session_dir / f"{task_id}.l2.npy", np.array(sorted(s), dtype=np.uint64))

    def _known_l2(self, task_id: str) -> set[int]:
        with self._l2_lock:
            if task_id not in self._l2_seen and self.session_dir:
                p = self.session_dir / f"{task_id}.l2.npy"
                if p.exists():
                    self._l2_seen[task_id] = {int(x) for x in np.load(p)}
            return self._l2_seen.get(task_id, set())

    # -- endpoints: each returns (json_dict or None, arrays dict or None)

    def handle(self, endpoint: str, sess: dict[str, Any], req: dict[str, Any]) -> tuple[Any, Any]:
        self.check_no_time(req)
        t: datetime = sess["t"]
        if endpoint == "info":
            return {"world_api_version": WORLD_API_VERSION, "task_id": sess["task_id"],
                    "t_utc": t.isoformat(), **self._info}, None
        if endpoint in ("em", "supervoxels", "segmentation"):
            layer = "em" if endpoint == "em" else "supervoxels"
            level, res = self._level(layer, req.get("resolution_nm_xyz", req.get("resolution")))
            lo, hi = self._bbox(req.get("bbox_nm_xyz", req.get("bbox_nm")), res)
            if endpoint == "em":
                data = self.backend.em(lo, hi, level)
            else:
                data = self.backend.supervoxels(lo, hi, level)
                if endpoint == "segmentation":
                    uniq, inv = np.unique(data, return_inverse=True)
                    if len(uniq) > self.max_roots_per_call:
                        raise Invalid(f"{len(uniq)} supervoxels in the box exceed the limit "
                                      f"{self.max_roots_per_call}; use a smaller box")
                    roots = sess["roots"].alias(self.backend.roots(uniq, t))
                    data = roots[inv].reshape(data.shape)
            return {"shape_zyx": list(data.shape)}, {
                "data": data, "origin_nm_xyz": (lo * res).astype(float), "resolution_nm_xyz": res}
        if endpoint == "roots":
            svs = self._ids(req.get("supervoxel_ids"), "supervoxel_ids")
            if len(svs) > self.max_roots_per_call:
                raise Invalid(f"at most {self.max_roots_per_call} supervoxel ids per call")
            return None, {"root_ids": sess["roots"].alias(self.backend.roots(svs, t))}
        if endpoint in ("l2_graph", "mesh"):
            shown = self._ids([req.get("root_id")], "root_id")
            root = sess["roots"].real(shown)
            self.require_valid_roots(root, t, shown)
            if endpoint == "l2_graph":
                l2, edges = self.backend.l2_graph(int(root[0]))
                self._remember_l2(sess["task_id"], l2)
                al = sess["l2"]
                return None, {"l2_ids": al.alias(l2), "edges": al.alias(np.asarray(edges, dtype=np.uint64))}
            v, f = self.backend.mesh(int(root[0]))
            return None, {"vertices_nm_xyz": v, "faces": f}
        if endpoint == "l2_positions":
            shown = self._ids(req.get("l2_ids"), "l2_ids")
            ids = sess["l2"].real(shown)
            known = self._known_l2(sess["task_id"])
            unknown = [str(int(s)) for x, s in zip(ids, shown, strict=True) if int(x) not in known]
            if unknown:
                raise Refused(f"l2_positions accepts only L2 ids returned by l2_graph for this task "
                              f"(unknown: {unknown[:5]})")
            pos, approx = self.backend.l2_positions(ids)
            return None, {"l2_ids": shown, "positions_nm_xyz": pos, "approximate": approx}
        raise Refused(refusal_message(endpoint))


def refusal_message(endpoint: str) -> str:
    return (f"/v1/{endpoint} is not available. The world answers only {', '.join(ENDPOINTS)}, as of the "
            f"task's time T. Lineage, operation records, edit history, latest roots, root timestamps, "
            f"other times, and writes are refused.")


def encode_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class WorldHandler(JSONHandler):
    app: WorldService

    def _dispatch(self, method: str) -> None:
        t0 = time.time()
        path = urlsplit(self.path).path.rstrip("/")
        endpoint = path.split("/")[-1] if path.startswith("/v1/") else path
        rec: dict[str, Any] = {"endpoint": endpoint, "method": method}
        hdr = {"x-world-api-version": WORLD_API_VERSION}
        try:
            raw = self.read_body()
            sess = self.app.session(self.headers)
            rec["task_id"] = sess["task_id"]
            if method != "POST" or endpoint not in ENDPOINTS or not path.startswith("/v1/"):
                raise Refused(refusal_message(endpoint))
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                raise Invalid("body must be JSON") from e
            if not isinstance(req, dict):
                raise Invalid("body must be a JSON object")
            rec["params"] = {k: (v if not isinstance(v, list) or len(v) <= 6 else f"<{len(v)} items>")
                             for k, v in req.items()}
            js, arrays = self.app.handle(endpoint, sess, req)
            if arrays is not None:
                status, ctype, data = 200, "application/x-npz", encode_arrays(arrays)
                hdr["x-world-meta"] = json.dumps(js or {})
            else:
                status, ctype, data = 200, "application/json", json.dumps(js).encode()
            rec.update(status=200, bytes=len(data))
        except Exception as e:  # noqa: BLE001
            status, etype = {PermissionError: (401, "unauthorized"), Refused: (403, "refused"),
                             Invalid: (400, "invalid")}.get(type(e), (500, "internal"))
            msg = str(e) if status != 500 else f"{type(e).__name__}: {e}"
            rec.update(status=status, error=msg)
            ctype, data = "application/json", json.dumps({"error": {"type": etype, "message": msg}}).encode()
        rec["seconds"] = round(time.time() - t0, 3)
        self.app.log(rec)  # logged before the reply, so a client never sees an unlogged answer
        self.send_bytes(status, data, ctype, hdr)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")


def make_handler(app: WorldService) -> type[WorldHandler]:
    return type("BoundWorldHandler", (WorldHandler,), {"app": app})


# ----------------------------------------------------------------------------- FlyWire backend


class FlyWireBackend:
    """Read-only FlyWire access. Only this class (and build/, outer/score.py) talks to FlyWire.

    - EM: the precomputed EM layer (config flywire.em_source); missing chunks read as 0.
    - supervoxels: the watershed layer under the graphene segmentation (its info ``data_dir``).
    - roots at T: chunkedgraph ``get_roots(..., timestamp=T)``.
    - root validity at T: created at or before T, and still the latest version at T.
    - L2 graph: ``get_leaves(root, stop_layer=2)`` + ``level2_chunk_graph(root)``.
    - L2 positions: there is no L2 cache on flywire_fafb_public, so each L2 node's position is the
      centroid of its own L2 mesh fragment; nodes with no fragment get their graph chunk's center
      and ``approximate=True``.
    - mesh: the graphene mesh of the root (fragments concatenated; vertices in nm).
    Immutable results (L2 graphs, meshes, L2 positions, volume chunks) are cached on disk.
    """

    def __init__(self, datastack: str, em_source: str, cache_dir: Path):
        self.datastack, self.em_source = datastack, em_source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tl = threading.local()
        cg = self._client().chunkedgraph
        self.seg_source = self._client().info.segmentation_source()
        seg = self._seg()
        self.ws_source = "precomputed://" + seg.meta.info["data_dir"]
        self.graph_chunk_vox = np.asarray(cg.segmentation_info["graph"]["chunk_size"], dtype=np.int64)
        self._db_lock = threading.Lock()
        self._db = sqlite3.connect(str(self.cache_dir / "l2pos.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS l2pos (l2 INTEGER PRIMARY KEY, x REAL, y REAL, z REAL, "
                         "approx INTEGER)")
        self._info = self._make_info()

    # -- per-thread clients

    def _client(self) -> Any:
        if not hasattr(self._tl, "client"):
            from caveclient import CAVEclient
            self._tl.client = CAVEclient(self.datastack)
        return self._tl.client

    def _cv(self, name: str, path: str, **kw: Any) -> Any:
        key = f"cv_{name}"
        if not hasattr(self._tl, key):
            from cloudvolume import CloudVolume
            setattr(self._tl, key, CloudVolume(path, use_https=True, progress=False, fill_missing=True,
                                               cache=str(self.cache_dir / "cv"), **kw))
        return getattr(self._tl, key)

    def _seg(self) -> Any:
        if not hasattr(self._tl, "cv_seg"):
            from cloudvolume import CloudVolume
            self._tl.cv_seg = CloudVolume(self.seg_source, use_https=True, progress=False)
        return self._tl.cv_seg

    # -- info

    def _make_info(self) -> dict[str, Any]:
        em, ws = self._cv("em", self.em_source), self._cv("ws", self.ws_source)

        def levels(vol: Any) -> list[dict[str, Any]]:
            return [{"resolution_nm_xyz": [float(x) for x in s["resolution"]],
                     "voxel_offset_xyz": [int(x) for x in s["voxel_offset"]],
                     "size_xyz": [int(x) for x in s["size"]],
                     "chunk_size_xyz": [int(x) for x in s["chunk_sizes"][0]]} for s in vol.meta.scales]

        s0 = ws.meta.scales[0]
        r0 = np.asarray(s0["resolution"], dtype=float)
        lo = np.asarray(s0["voxel_offset"]) * r0
        hi = (np.asarray(s0["voxel_offset"]) + np.asarray(s0["size"])) * r0
        return {
            "dataset": f"FlyWire FAFB ({self.datastack})",
            "bounds_nm_xyz": [lo.tolist(), hi.tolist()],
            "layers": {
                "em": {"description": "EM image, uint8. Some chunks are missing at the finest level and read as 0.",
                       "source": self.em_source, "dtype": "uint8", "levels": levels(em)},
                "supervoxels": {"description": "watershed supervoxel ids, uint64; 0 is background. Roots come "
                                               "from mapping supervoxels through roots() at T.",
                                "dtype": "uint64", "levels": levels(ws)},
            },
            "l2_chunk_size_nm_xyz": (self.graph_chunk_vox * r0).tolist(),
        }

    def static_info(self) -> dict[str, Any]:
        return self._info

    # -- volumes

    def _download(self, vol: Any, lo: np.ndarray, hi: np.ndarray, level: int, dtype: Any) -> np.ndarray:
        """[lo, hi) in voxels of ``level`` → array (z, y, x); outside the dataset reads as 0."""
        from cloudvolume import Bbox

        s = vol.meta.scales[level]
        off, size = np.asarray(s["voxel_offset"]), np.asarray(s["size"])
        out = np.zeros(tuple((hi - lo)[::-1]), dtype=dtype)
        a, b = np.maximum(lo, off), np.minimum(hi, off + size)
        if np.any(b <= a):
            return out
        arr = np.asarray(vol.download(Bbox(a.tolist(), b.tolist()), mip=level))[..., 0]
        sl = tuple(slice(int(x0 - l0), int(x1 - l0)) for x0, x1, l0 in zip(a[::-1], b[::-1], lo[::-1], strict=True))
        out[sl] = arr.transpose(2, 1, 0)
        return out

    def em(self, lo: np.ndarray, hi: np.ndarray, level: int) -> np.ndarray:
        return self._download(self._cv("em", self.em_source), lo, hi, level, np.uint8)

    def supervoxels(self, lo: np.ndarray, hi: np.ndarray, level: int) -> np.ndarray:
        return self._download(self._cv("ws", self.ws_source), lo, hi, level, np.uint64)

    def supervoxels_at(self, points_nm_xyz: np.ndarray) -> np.ndarray:
        """Supervoxel id at each point (base resolution of the supervoxel layer); used by scoring."""
        res = np.asarray(self._info["layers"]["supervoxels"]["levels"][0]["resolution_nm_xyz"], dtype=float)
        pts = np.asarray(points_nm_xyz, dtype=float).reshape(-1, 3)
        out = np.zeros(len(pts), dtype=np.uint64)
        for i, p in enumerate(pts):
            v = np.floor(p / res).astype(np.int64)
            out[i] = self.supervoxels(v, v + 1, 0)[0, 0, 0]
        return out

    # -- graph

    def roots(self, supervoxel_ids: np.ndarray, t: datetime) -> np.ndarray:
        svs = np.asarray(supervoxel_ids, dtype=np.uint64)
        out = np.zeros(len(svs), dtype=np.uint64)
        nz = svs != 0
        if nz.any():
            uniq, inv = np.unique(svs[nz], return_inverse=True)
            r = np.asarray(self._client().chunkedgraph.get_roots(uniq, timestamp=t), dtype=np.uint64)
            out[nz] = r[inv]
        return out

    def roots_valid_at(self, root_ids: np.ndarray, t: datetime) -> np.ndarray:
        cg = self._client().chunkedgraph
        ids = [int(r) for r in root_ids]
        ok = np.zeros(len(ids), dtype=bool)
        for i, r in enumerate(ids):
            try:
                created = cg.get_root_timestamps([r])[0]
                latest = bool(cg.is_latest_roots([r], timestamp=t)[0])
            except Exception:  # noqa: BLE001  (not a root id at all)
                continue
            created = created if created.tzinfo else created.replace(tzinfo=UTC)
            ok[i] = created <= t and latest
        return ok

    def l2_graph(self, root_id: int) -> tuple[np.ndarray, np.ndarray]:
        p = self.cache_dir / "l2graph" / f"{root_id}.npz"
        if p.exists():
            z = np.load(p)
            return z["l2_ids"], z["edges"]
        cg = self._client().chunkedgraph
        l2 = np.unique(np.asarray(cg.get_leaves(root_id, stop_layer=2), dtype=np.uint64))
        edges = np.asarray(cg.level2_chunk_graph(root_id), dtype=np.uint64).reshape(-1, 2)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, l2_ids=l2, edges=edges)
        return l2, edges

    def _chunk_center_nm(self, l2: int) -> np.ndarray:
        seg = self._seg()
        pos = np.asarray(seg.meta.decode_chunk_position(l2), dtype=np.int64)
        start = pos * self.graph_chunk_vox
        if seg.meta.chunks_start_at_voxel_offset:
            start = start + np.asarray(seg.meta.voxel_offset(0))
        res = np.asarray(seg.meta.resolution(0), dtype=float)
        return (start + self.graph_chunk_vox / 2.0) * res

    def l2_positions(self, l2_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from cloudfiles import CloudFiles
        from cloudvolume.mesh import Mesh

        ids = [int(x) for x in l2_ids]
        found: dict[int, tuple[float, float, float, int]] = {}
        with self._db_lock:
            for i in range(0, len(ids), 900):
                q = ids[i:i + 900]
                rows = self._db.execute(f"SELECT l2, x, y, z, approx FROM l2pos WHERE l2 IN ({','.join('?' * len(q))})",
                                        q).fetchall()
                found.update({r[0]: r[1:] for r in rows})
        todo = sorted(set(ids) - set(found))
        if todo:
            seg = self._seg()
            names = {seg.mesh.compute_filename(x): x for x in todo}
            cp = seg.meta.join(seg.meta.cloudpath, seg.mesh.meta.mesh_path)
            got: dict[int, tuple[float, float, float, int]] = {}
            for i in range(0, len(names), 2000):
                batch = list(names)[i:i + 2000]
                for r in CloudFiles(cp, parallel=16).get(batch):
                    l2 = names[r["path"].split("/")[-1]]
                    m = None
                    if r.get("content"):
                        try:
                            m = Mesh.from_draco(r["content"])
                        except Exception:  # noqa: BLE001
                            try:
                                m = Mesh.from_precomputed(r["content"])
                            except Exception:  # noqa: BLE001
                                m = None
                    if m is not None and len(m.vertices):
                        c = np.asarray(m.vertices, dtype=float).mean(axis=0)
                        got[l2] = (float(c[0]), float(c[1]), float(c[2]), 0)
            for l2 in todo:
                if l2 not in got:
                    c = self._chunk_center_nm(l2)
                    got[l2] = (float(c[0]), float(c[1]), float(c[2]), 1)
            with self._db_lock:
                self._db.executemany("INSERT OR REPLACE INTO l2pos VALUES (?, ?, ?, ?, ?)",
                                     [(k, *v) for k, v in got.items()])
                self._db.commit()
            found.update(got)
        pos = np.array([found[x][:3] for x in ids], dtype=float).reshape(-1, 3)
        approx = np.array([bool(found[x][3]) for x in ids], dtype=bool)
        return pos, approx

    def mesh(self, root_id: int) -> tuple[np.ndarray, np.ndarray]:
        p = self.cache_dir / "mesh" / f"{root_id}.npz"
        if p.exists():
            z = np.load(p)
            return z["vertices_nm_xyz"], z["faces"]
        m = self._seg().mesh.get(int(root_id), deduplicate_chunk_boundaries=False)
        m = m[int(root_id)] if isinstance(m, dict) else m
        v, f = np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int32)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, vertices_nm_xyz=v, faces=f)
        return v, f


def make_service(cfg: Any, backend: Backend | None = None) -> WorldService:
    from outer.config import token_secret

    w = cfg["world"]
    if backend is None:
        backend = FlyWireBackend(cfg["flywire"]["datastack"], cfg["flywire"]["em_source"], cfg.path("world.cache_dir"))
    return WorldService(backend, token_secret(cfg), log_path=cfg.root / "runs" / "world_requests.jsonl",
                        max_cutout_voxels=int(w["max_cutout_voxels"]), max_roots_per_call=int(w["max_roots_per_call"]),
                        session_dir=cfg.path("world.cache_dir") / "sessions")


def main(argv: list[str] | None = None) -> None:
    from outer.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    app = make_service(cfg)
    sock = cfg.path("world.socket")
    servers = serve(make_handler(app), sock, cfg["world"].get("host"), cfg["world"].get("port"))
    print(f"world listening on {sock}" + (f" and port {cfg['world']['port']}" if cfg["world"].get("port") else ""),
          flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    for s in servers:
        s.shutdown()


if __name__ == "__main__":
    main()
