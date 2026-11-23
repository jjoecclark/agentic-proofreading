"""Fakes for tests: an analytic FlyWire-like world and a fake Anthropic upstream. No network."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------- fake world
#
# Space: x in [0, 1024), y in [0, K*640), z in [0, 256) voxels at 16 x 16 x 40 nm. Supervoxels are
# 32 x 32 x 8 blocks. Strip i covers y in [640 i, 640 (i+1)) and has one edit at time T_i, along the
# plane x = 512:
#   even i (merge): before T_i roots L_i (x < 512) and R_i (x >= 512); after T_i one root M_i.
#   odd i (split):  before T_i one root S_i; after T_i roots SL_i (x < 512) and SR_i (x >= 512).
# The edit center of strip i is (8192, 10240 i + 5120, 5120) nm.

RES = np.array([16.0, 16.0, 40.0])
BLOCK = np.array([32, 32, 8])
STRIP_Y = 640
NX, NZ = 1024, 256
BASE_TIME = datetime(2023, 1, 1, tzinfo=UTC)
L2_OFFSET = 1 << 40


def strip_time(i: int) -> datetime:
    return BASE_TIME + timedelta(days=int(i))


class FakeBackend:
    def __init__(self, n_strips: int = 4):
        self.k = n_strips
        self.ny = n_strips * STRIP_Y
        self.nb = np.array([NX // BLOCK[0], self.ny // BLOCK[1], NZ // BLOCK[2]])
        self.calls: list[str] = []

    # -- geometry helpers
    def sv_of_vox(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        inside = (x >= 0) & (x < NX) & (y >= 0) & (y < self.ny) & (z >= 0) & (z < NZ)
        bx, by, bz = x // BLOCK[0], y // BLOCK[1], z // BLOCK[2]
        sv = 1 + bx + self.nb[0] * (by + self.nb[1] * bz)
        return np.where(inside, sv, 0).astype(np.uint64)

    def block_of_sv(self, sv: int) -> tuple[int, int, int]:
        s = int(sv) - 1
        return int(s % self.nb[0]), int((s // self.nb[0]) % self.nb[1]), int(s // (self.nb[0] * self.nb[1]))

    @staticmethod
    def root_id(i: int, k: int) -> int:
        return 720575940000000000 + 10 * i + k

    def strip_roots(self, i: int) -> dict[str, list[int]]:
        r = [self.root_id(i, k) for k in (1, 2, 3)]
        if i % 2 == 0:
            return {"pre": [r[0], r[1]], "post": [r[2]]}
        return {"pre": [r[0]], "post": [r[1], r[2]]}

    def root_of_block(self, bx: int, by: int, t: datetime) -> int:
        i = by * BLOCK[1] // STRIP_Y
        left = bx * BLOCK[0] < NX // 2
        before = t < strip_time(i)
        if i % 2 == 0:
            return self.root_id(i, (1 if left else 2) if before else 3)
        return self.root_id(i, 1 if before else (2 if left else 3))

    def blocks_of_root(self, root: int) -> list[tuple[int, int, int]]:
        i, k = (root - 720575940000000000) // 10, (root - 720575940000000000) % 10
        y0, y1 = i * STRIP_Y // BLOCK[1], (i + 1) * STRIP_Y // BLOCK[1]
        half = self.nb[0] // 2
        if i % 2 == 0:
            xs = range(0, half) if k == 1 else range(half, self.nb[0]) if k == 2 else range(self.nb[0])
        else:
            xs = range(self.nb[0]) if k == 1 else range(0, half) if k == 2 else range(half, self.nb[0])
        return [(bx, by, bz) for bz in range(self.nb[2]) for by in range(y0, y1) for bx in xs]

    def validity(self, root: int) -> tuple[datetime, datetime]:
        i, k = (root - 720575940000000000) // 10, (root - 720575940000000000) % 10
        early, late = BASE_TIME - timedelta(days=365), BASE_TIME + timedelta(days=3650)
        t = strip_time(i)
        if i % 2 == 0:
            return (early, t) if k in (1, 2) else (t, late)
        return (early, t) if k == 1 else (t, late)

    # -- Backend interface
    def static_info(self) -> dict[str, Any]:
        def lv(r: list[float]) -> dict[str, Any]:
            f = np.array(r) / RES
            return {"resolution_nm_xyz": r, "voxel_offset_xyz": [0, 0, 0],
                    "size_xyz": [int(NX / f[0]), int(self.ny / f[1]), int(NZ / f[2])], "chunk_size_xyz": [64, 64, 64]}
        return {"dataset": "fake", "bounds_nm_xyz": [[0, 0, 0], (np.array([NX, self.ny, NZ]) * RES).tolist()],
                "layers": {"em": {"description": "fake EM", "dtype": "uint8",
                                  "levels": [lv([16.0, 16.0, 40.0]), lv([32.0, 32.0, 40.0])]},
                           "supervoxels": {"description": "fake supervoxels", "dtype": "uint64",
                                           "levels": [lv([16.0, 16.0, 40.0]), lv([32.0, 32.0, 40.0])]}},
                "l2_chunk_size_nm_xyz": (BLOCK * RES).tolist()}

    def _grid(self, lo: np.ndarray, hi: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = 2 ** level
        z, y, x = np.meshgrid(np.arange(lo[2], hi[2]), np.arange(lo[1], hi[1]) * f, np.arange(lo[0], hi[0]) * f,
                              indexing="ij")
        return x, y, z

    def supervoxels(self, lo: np.ndarray, hi: np.ndarray, level: int) -> np.ndarray:
        self.calls.append("supervoxels")
        x, y, z = self._grid(lo, hi, level)
        return self.sv_of_vox(x, y, z)

    def em(self, lo: np.ndarray, hi: np.ndarray, level: int) -> np.ndarray:
        self.calls.append("em")
        x, y, z = self._grid(lo, hi, level)
        img = 140 + 40 * np.sin(x / 7.0) * np.cos(y / 5.0)
        img[(x % BLOCK[0] == 0) | (y % BLOCK[1] == 0)] = 40
        return img.astype(np.uint8)

    def supervoxels_at(self, points_nm_xyz: np.ndarray) -> np.ndarray:
        v = np.floor(np.asarray(points_nm_xyz, dtype=float).reshape(-1, 3) / RES).astype(np.int64)
        return self.sv_of_vox(v[:, 0], v[:, 1], v[:, 2])

    def roots(self, supervoxel_ids: np.ndarray, t: datetime) -> np.ndarray:
        self.calls.append("roots")
        out = np.zeros(len(supervoxel_ids), dtype=np.uint64)
        for j, sv in enumerate(supervoxel_ids):
            if int(sv):
                bx, by, _ = self.block_of_sv(int(sv))
                out[j] = self.root_of_block(bx, by, t)
        return out

    def roots_valid_at(self, root_ids: np.ndarray, t: datetime) -> np.ndarray:
        out = []
        for r in root_ids:
            r = int(r)
            i = (r - 720575940000000000) // 10
            if not (0 <= i < self.k) or (r % 10) not in (1, 2, 3):
                out.append(False)
                continue
            a, b = self.validity(r)
            out.append(a <= t < b)
        return np.array(out, dtype=bool)

    def l2_graph(self, root_id: int) -> tuple[np.ndarray, np.ndarray]:
        blocks = self.blocks_of_root(root_id)
        s = set(blocks)
        ids = {b: 1 + b[0] + self.nb[0] * (b[1] + self.nb[1] * b[2]) + L2_OFFSET for b in blocks}
        edges = [(ids[b], ids[(b[0] + 1, b[1], b[2])]) for b in blocks if (b[0] + 1, b[1], b[2]) in s]
        edges += [(ids[b], ids[(b[0], b[1] + 1, b[2])]) for b in blocks if (b[0], b[1] + 1, b[2]) in s]
        edges += [(ids[b], ids[(b[0], b[1], b[2] + 1)]) for b in blocks if (b[0], b[1], b[2] + 1) in s]
        return np.array(sorted(ids.values()), dtype=np.uint64), np.array(edges, dtype=np.uint64).reshape(-1, 2)

    def l2_positions(self, l2_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = []
        for l2 in l2_ids:
            bx, by, bz = self.block_of_sv(int(l2) - L2_OFFSET)
            pos.append((np.array([bx, by, bz]) + 0.5) * BLOCK * RES)
        return np.array(pos, dtype=float).reshape(-1, 3), np.zeros(len(pos), dtype=bool)

    def mesh(self, root_id: int) -> tuple[np.ndarray, np.ndarray]:
        b = np.array(self.blocks_of_root(root_id))
        lo, hi = b.min(0) * BLOCK * RES, (b.max(0) + 1) * BLOCK * RES
        v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                     dtype=np.float32)
        f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
        return v, f


def fake_truths(backend: FakeBackend) -> dict[str, dict[str, Any]]:
    """task_id -> truth record (same schema as truth/<op>.json) for each strip."""
    out = {}
    for i in range(backend.k):
        t = strip_time(i)
        t_pre, t_post = t - timedelta(milliseconds=1), t + timedelta(milliseconds=1)
        c = np.array([NX / 2 * RES[0], (i * STRIP_Y + STRIP_Y / 2) * RES[1], NZ / 2 * RES[2]])
        roots = backend.strip_roots(i)
        typ = "merge" if i % 2 == 0 else "split"
        src = [(c + [-150, 0, 0]).tolist()]
        snk = [(c + [150, 0, 0]).tolist()]
        out[f"t_fake{i:04d}"] = {
            "truth_version": "1", "operation_id": str(900000 + i), "operation_type": typ,
            "timestamp_utc": t.isoformat(), "t_utc": t_pre.isoformat(), "t_post_utc": t_post.isoformat(),
            "pre_edit_roots": [str(r) for r in roots["pre"]], "post_edit_roots": [str(r) for r in roots["post"]],
            "sources_nm_xyz": src, "sinks_nm_xyz": snk, "edit_center_nm_xyz": c.tolist(),
            "segmentation_source": "graphene://fake", "problems": []}
    return out


def fake_negatives(truths: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """task_id -> negative truth (the fields of truth/negatives/<negative_id>.json that scoring and the
    world use). The fake world has no proofread neurons, so each strip's state just after its edit stands
    in: a merge strip gives a no_split task (one root), a split strip a no_merge task (two roots)."""
    out = {}
    for i, (tid, tr) in enumerate(sorted(truths.items())):
        kind = "no_split" if tr["operation_type"] == "merge" else "no_merge"
        out[tid.replace("t_fake", "t_fneg")] = {
            "negative_version": "2", "negative_id": f"{kind}_{i:04d}", "kind": kind, "operation_type": "none",
            "t_utc": tr["t_post_utc"], "segmentation_source": tr["segmentation_source"],
            "segments": list(tr["post_edit_roots"]), "eligible": True, "problems": [],
            "neurons": [{"root_at_check": r, "position_nm_xyz": tr["edit_center_nm_xyz"]} for r in tr["post_edit_roots"]]}
    return out


def fake_state(task_id: str, truth: dict[str, Any]) -> dict[str, Any]:
    from build.build_states import make_state

    return make_state(task_id, 7, truth, "precomputed://fake-em")


# ----------------------------------------------------------------------------- fake Anthropic upstream


class FakeUpstream:
    """Speaks enough of POST /v1/messages (streaming and not) for the proxy and the SDK.

    ``reply(body) -> str`` picks the response text. Usage: input = len(json(messages)) // 4,
    output = 10 + len(text) // 4, plus fixed cache numbers so all price terms are exercised."""

    def __init__(self, reply: Any = None):
        self.reply = reply or (lambda body: "hello from fake upstream")
        self.full_output = False  # bill max_tokens of output (to exercise caps)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                up.requests.append(body)
                up.headers.append({k.lower(): v for k, v in self.headers.items()})
                if self.headers.get("x-api-key") != "fake-key":
                    return self._json(401, {"type": "error", "error": {"type": "authentication_error",
                                                                       "message": "bad key"}})
                if self.path.startswith("/v1/messages/count_tokens"):
                    return self._json(200, {"input_tokens": 42})
                r = up.reply(body)  # a string, or {"content": [blocks], "stop_reason": ...}
                if isinstance(r, str):
                    content, stop = [{"type": "text", "text": r}], "end_turn"
                else:
                    content, stop = r["content"], r.get("stop_reason", "end_turn")
                text = json.dumps(content) if not isinstance(r, str) else r
                n_out = int(body.get("max_tokens", 0)) if up.full_output else 10 + len(text) // 4
                usage = {"input_tokens": len(json.dumps(body.get("messages"))) // 4, "output_tokens": n_out,
                         "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
                         "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 0}}
                msg = {"id": f"msg_fake{len(up.requests)}", "type": "message", "role": "assistant",
                       "model": body["model"], "content": content,
                       "stop_reason": stop, "stop_sequence": None, "usage": usage}
                if not body.get("stream"):
                    return self._json(200, msg, {"request-id": f"req_fake{len(up.requests)}"})
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("request-id", f"req_fake{len(up.requests)}")
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                start = dict(msg, content=[], stop_reason=None,
                             usage={**usage, "output_tokens": 1})
                events = [("message_start", {"type": "message_start", "message": start})]
                for i, b in enumerate(content):
                    if b["type"] == "text":
                        events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                               "content_block": {"type": "text", "text": ""}}))
                        for j in range(0, len(b["text"]), 7):
                            events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {
                                "type": "text_delta", "text": b["text"][j:j + 7]}}))
                    else:  # tool_use
                        events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                               "content_block": {**b, "input": {}}}))
                        js = json.dumps(b["input"])
                        for j in range(0, len(js), 40):
                            events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {
                                "type": "input_json_delta", "partial_json": js[j:j + 40]}}))
                    events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
                events += [("message_delta", {"type": "message_delta",
                                              "delta": {"stop_reason": stop, "stop_sequence": None},
                                              "usage": {"output_tokens": usage["output_tokens"]}}),
                           ("message_stop", {"type": "message_stop"})]
                for name, data in events:
                    chunk = f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                    time.sleep(0.001)
                self.wfile.write(b"0\r\n\r\n")

            def _json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
