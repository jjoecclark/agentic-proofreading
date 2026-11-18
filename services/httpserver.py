"""Threaded HTTP servers on a Unix socket (what sandboxes reach) and optionally on TCP."""

from __future__ import annotations

import json
import os
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, path: Path, handler: type[BaseHTTPRequestHandler]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_socket():
            path.unlink()
        super().__init__(str(path), handler)
        os.chmod(path, 0o600)  # only this user's processes (and its containers) may connect
        self.server_name, self.server_port = "localhost", 0


class TCPHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class JSONHandler(BaseHTTPRequestHandler):
    """Base handler: JSON helpers, quiet logs, Unix-socket-safe client address."""

    protocol_version = "HTTP/1.1"
    app: Any = None  # set on the subclass

    def address_string(self) -> str:  # client_address is '' on Unix sockets
        return "unix" if not self.client_address else str(self.client_address[0])

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def read_body(self) -> bytes:
        n = int(self.headers.get("content-length") or 0)
        return self.rfile.read(n) if n else b""

    def send_json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_bytes(status, data, "application/json", headers)

    def send_bytes(self, status: int, data: bytes, content_type: str,
                   headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)


def serve(handler: type[BaseHTTPRequestHandler], socket_path: Path | None, host: str | None = None,
          port: int | None = None) -> list[socketserver.BaseServer]:
    """Start servers in daemon threads; returns them (call ``.shutdown()`` to stop)."""
    servers: list[socketserver.BaseServer] = []
    if socket_path is not None:
        servers.append(UnixHTTPServer(Path(socket_path), handler))
    if port is not None:
        servers.append(TCPHTTPServer((host or "127.0.0.1", int(port)), handler))
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    return servers
