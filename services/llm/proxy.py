"""The `llm` proxy (PLAN.md Section 6.2): the only holder of the Anthropic API key.

Clients use the standard Anthropic SDK with ``ANTHROPIC_BASE_URL`` pointed here and
``ANTHROPIC_API_KEY`` set to a run token (services/tokens.py). For each request the proxy

1. verifies the run token, the model allowlist, and that the request uses no server-side tool or
   option that could reach the web or another model (web search/fetch, MCP, fallbacks, ...);
2. refuses it (HTTP 402, error type ``budget_exceeded``) if its worst case — estimated input at the
   5-minute cache-write price plus ``max_tokens`` of output — would take the owner or its group over
   the cap, counting requests still in flight;
3. swaps in the real key, forwards it (streaming or not), and relays the response;
4. appends one ledger line (tokens and dollars) and one transcript line for the owner.

Run:  .venv/bin/python -m services.llm.proxy            (reads config.yaml)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from services.httpserver import JSONHandler, serve
from services.llm.ledger import Ledger, cost_usd, normalize_usage
from services.tokens import TokenError, verify

TRANSCRIPT_VERSION = "1"
PROXY_API_VERSION = "1"

# Tools a client executes itself are fine; server-side tools could reach the web or other models.
CLIENT_TOOL_TYPE_PREFIXES = ("custom", "bash_", "text_editor_", "memory_")
REFUSED_BODY_KEYS = ("mcp_servers", "container", "fallbacks", "inference_geo", "service_tier")
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding",
               "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"}
IMAGE_TOKENS_ESTIMATE = 5000      # per image/document block, for the pre-request worst case
CHARS_PER_TOKEN_ESTIMATE = 3.0    # conservative (real text averages ~3.5-4 chars per token)


class Refusal(Exception):
    def __init__(self, status: int, etype: str, message: str):
        super().__init__(message)
        self.status, self.etype, self.message = status, etype, message


def error_body(etype: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": etype, "message": message}}


# ----------------------------------------------------------------------------- request checks


def check_request(body: dict[str, Any], allowed_models: list[str]) -> None:
    model = body.get("model")
    if model not in allowed_models:
        raise Refusal(403, "permission_error", f"model {model!r} is not allowed (allowed: {allowed_models})")
    for k in REFUSED_BODY_KEYS:
        if k in body:
            raise Refusal(403, "permission_error", f"request field {k!r} is not allowed through this proxy")
    if body.get("speed") not in (None, "standard"):
        raise Refusal(403, "permission_error", "only standard speed is allowed through this proxy")
    for tool in body.get("tools") or []:
        ttype = tool.get("type") if isinstance(tool, dict) else None
        if ttype and not str(ttype).startswith(CLIENT_TOOL_TYPE_PREFIXES):
            raise Refusal(403, "permission_error",
                          f"server-side tool type {ttype!r} is not allowed; define client tools instead")


def _text_chars(obj: Any) -> tuple[int, int]:
    """(characters of non-media JSON, number of media blocks) in a request body."""
    chars, media = 0, 0
    if isinstance(obj, dict):
        if obj.get("type") in ("image", "document") and isinstance(obj.get("source"), dict):
            return 0, 1
        for k, v in obj.items():
            c, m = _text_chars(v)
            chars += c + len(k)
            media += m
    elif isinstance(obj, list):
        for v in obj:
            c, m = _text_chars(v)
            chars, media = chars + c, media + m
    elif isinstance(obj, str):
        chars += len(obj)
    else:
        chars += len(str(obj))
    return chars, media


def worst_case_usd(body: dict[str, Any], price: dict[str, float]) -> float:
    chars, media = _text_chars({k: body.get(k) for k in ("system", "messages", "tools")})
    in_tokens = chars / CHARS_PER_TOKEN_ESTIMATE + media * IMAGE_TOKENS_ESTIMATE
    out_tokens = int(body.get("max_tokens") or 0)
    return (in_tokens * price.get("cache_write_5m", 1.25 * price["input"]) + out_tokens * price["output"]) / 1e6


# ----------------------------------------------------------------------------- SSE accumulation


class StreamAccumulator:
    """Parses an SSE byte stream into the final message and its usage."""

    def __init__(self) -> None:
        self.buf = b""
        self.message: dict[str, Any] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.partial_json: dict[int, str] = {}
        self.usage: dict[str, Any] = {}
        self.error: Any = None

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk.replace(b"\r\n", b"\n")
        while b"\n\n" in self.buf:
            raw, self.buf = self.buf.split(b"\n\n", 1)
            data = [ln[5:].strip() for ln in raw.split(b"\n") if ln.startswith(b"data:")]
            if data:
                try:
                    self._event(json.loads(b"\n".join(data)))
                except json.JSONDecodeError:
                    pass

    def _event(self, ev: dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "message_start":
            self.message = dict(ev.get("message") or {})
            self.usage.update(self.message.get("usage") or {})
        elif t == "content_block_start":
            self.blocks[ev["index"]] = dict(ev.get("content_block") or {})
        elif t == "content_block_delta":
            b = self.blocks.setdefault(ev["index"], {})
            d = ev.get("delta") or {}
            dt = d.get("type")
            if dt == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif dt == "input_json_delta":
                self.partial_json[ev["index"]] = self.partial_json.get(ev["index"], "") + d.get("partial_json", "")
            elif dt == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
            elif dt == "signature_delta":
                b["signature"] = d.get("signature")
            else:
                b.setdefault("_deltas", []).append(d)
        elif t == "content_block_stop":
            i = ev["index"]
            if i in self.partial_json:
                try:
                    self.blocks[i]["input"] = json.loads(self.partial_json[i] or "{}")
                except json.JSONDecodeError:
                    self.blocks[i]["input"] = {"_unparsed": self.partial_json[i]}
        elif t == "message_delta":
            self.message.update({k: v for k, v in (ev.get("delta") or {}).items()})
            for k, v in (ev.get("usage") or {}).items():
                if v is not None:
                    self.usage[k] = v
        elif t == "error":
            self.error = ev.get("error")

    def final(self) -> dict[str, Any]:
        msg = dict(self.message)
        msg["content"] = [self.blocks[i] for i in sorted(self.blocks)]
        msg["usage"] = self.usage
        if self.error:
            msg["stream_error"] = self.error
        return msg


# ----------------------------------------------------------------------------- transcripts


class Transcripts:
    """Writes ``<log_dir>/transcript.jsonl``. Base64 media is saved to ``<log_dir>/media/`` and
    replaced by a reference; a request that continues the owner's previous conversation stores
    only its new messages (``messages_from``)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._locks: dict[str, threading.Lock] = {}
        self._prev: dict[str, tuple[str, list[str]]] = {}
        self._glock = threading.Lock()

    def _lock(self, key: str) -> threading.Lock:
        with self._glock:
            return self._locks.setdefault(key, threading.Lock())

    def _elide(self, obj: Any, media_dir: Path) -> Any:
        if isinstance(obj, dict):
            src = obj.get("source") if obj.get("type") in ("image", "document") else None
            if isinstance(src, dict) and src.get("type") == "base64" and isinstance(src.get("data"), str):
                raw = base64.b64decode(src["data"], validate=False)
                h = hashlib.sha256(raw).hexdigest()[:16]
                ext = (src.get("media_type") or "application/octet-stream").split("/")[-1]
                media_dir.mkdir(parents=True, exist_ok=True)
                p = media_dir / f"{h}.{ext}"
                if not p.exists():
                    p.write_bytes(raw)
                return {**obj, "source": {**src, "data": f"<media/{p.name} ({len(raw)} bytes)>"}}
            return {k: self._elide(v, media_dir) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._elide(v, media_dir) for v in obj]
        return obj

    def write(self, owner: str, log_dir: str | None, request: dict[str, Any], record: dict[str, Any]) -> None:
        if not log_dir:
            return
        d = (self.root / log_dir).resolve()
        if self.root.resolve() not in d.parents and d != self.root.resolve():
            return  # never write outside the repository
        d.mkdir(parents=True, exist_ok=True)
        req = self._elide(request, d / "media")
        msgs = req.get("messages") or []
        hashes = [hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest() for m in msgs]
        sys_hash = hashlib.sha256(json.dumps([req.get("system"), req.get("tools")], sort_keys=True,
                                             default=str).encode()).hexdigest()
        with self._lock(owner):
            prev = self._prev.get(owner)
            k = 0
            if prev and prev[0] == sys_hash and len(prev[1]) <= len(hashes) and hashes[:len(prev[1])] == prev[1]:
                k = len(prev[1])
            stored = {kk: v for kk, v in req.items() if kk != "messages"}
            if k:
                stored = {kk: v for kk, v in stored.items() if kk not in ("system", "tools")}
            stored["messages_from"] = k
            stored["messages"] = msgs[k:]
            self._prev[owner] = (sys_hash, hashes)
            line = {"transcript_version": TRANSCRIPT_VERSION, "ts": time.time(), "request": stored, **record}
            with open(d / "transcript.jsonl", "a") as f:
                f.write(json.dumps(line, default=str) + "\n")


# ----------------------------------------------------------------------------- the proxy


class LLMProxy:
    def __init__(self, secret: bytes, ledger: Ledger, prices: dict[str, dict[str, float]], upstream: str,
                 api_key: str, repo_root: Path, timeout_s: float = 900.0):
        self.secret, self.ledger, self.prices = secret, ledger, prices
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.transcripts = Transcripts(repo_root)
        self.http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0),
                                 limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
        self._rlock = threading.Lock()
        self.reserved_owner: dict[str, float] = {}
        self.reserved_group: dict[str, float] = {}

    # -- auth and budget

    def authenticate(self, headers: Any) -> dict[str, Any]:
        tok = headers.get("x-api-key")
        if not tok:
            auth = headers.get("authorization") or ""
            tok = auth[7:] if auth.lower().startswith("bearer ") else None
        if not tok:
            raise Refusal(401, "authentication_error", "missing run token")
        try:
            return verify(tok, self.secret, "llm")
        except TokenError as e:
            raise Refusal(401, "authentication_error", str(e)) from e

    def usage_report(self, tok: dict[str, Any]) -> dict[str, Any]:
        owner, group = tok["owner"], tok.get("group")
        spent = self.ledger.spent(owner)
        out = {"proxy_api_version": PROXY_API_VERSION, "owner": owner, "spent_usd": spent,
               "cap_usd": tok["cap_usd"], "remaining_usd": max(0.0, tok["cap_usd"] - spent),
               "models": tok["models"]}
        if group:
            gs = self.ledger.spent_group(group)
            out.update(group=group, group_spent_usd=gs, group_cap_usd=tok.get("group_cap_usd"))
            if tok.get("group_cap_usd") is not None:
                out["remaining_usd"] = min(out["remaining_usd"], max(0.0, tok["group_cap_usd"] - gs))
        return out

    def reserve(self, tok: dict[str, Any], est: float) -> None:
        owner, group = tok["owner"], tok.get("group")
        with self._rlock:
            o_res = self.reserved_owner.get(owner, 0.0)
            if self.ledger.spent(owner) + o_res + est > tok["cap_usd"]:
                raise Refusal(402, "budget_exceeded",
                              f"run token budget: spent ${self.ledger.spent(owner):.4f} + in flight ${o_res:.4f}"
                              f" + this request's worst case ${est:.4f} would exceed the cap ${tok['cap_usd']:.2f}")
            if group and tok.get("group_cap_usd") is not None:
                g_res = self.reserved_group.get(group, 0.0)
                if self.ledger.spent_group(group) + g_res + est > tok["group_cap_usd"]:
                    raise Refusal(402, "budget_exceeded",
                                  f"group budget {group!r}: spent ${self.ledger.spent_group(group):.4f} + in flight "
                                  f"${g_res:.4f} + worst case ${est:.4f} would exceed ${tok['group_cap_usd']:.2f}")
                self.reserved_group[group] = g_res + est
            self.reserved_owner[owner] = o_res + est

    def release(self, tok: dict[str, Any], est: float) -> None:
        with self._rlock:
            self.reserved_owner[tok["owner"]] = self.reserved_owner.get(tok["owner"], 0.0) - est
            if tok.get("group"):
                self.reserved_group[tok["group"]] = self.reserved_group.get(tok["group"], 0.0) - est

    def record(self, tok: dict[str, Any], model: str, usage: dict[str, Any] | None, status: Any,
               stream: bool, request_id: str | None) -> dict[str, Any]:
        tokens = normalize_usage(usage)
        usd = cost_usd(model, tokens, self.prices) if model in self.prices else 0.0
        return self.ledger.append({"owner": tok["owner"], "group": tok.get("group"), "model": model,
                                   "request_id": request_id, "status": status, "stream": stream,
                                   **tokens, "usd": usd})

    # -- forwarding

    def upstream_headers(self, headers: Any) -> dict[str, str]:
        out = {"x-api-key": self.api_key, "content-type": "application/json", "accept-encoding": "identity"}
        for k in ("anthropic-version", "anthropic-beta", "accept", "user-agent"):
            if headers.get(k):
                out[k] = headers.get(k)
        out.setdefault("anthropic-version", "2023-06-01")
        return out


class ProxyHandler(JSONHandler):
    app: LLMProxy

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            tok = self.app.authenticate(self.headers)
            if path == "/_proxy/usage":
                return self.send_json(200, self.app.usage_report(tok))
            if path.startswith("/v1/models"):
                r = self.app.http.get(self.app.upstream + self.path, headers=self.app.upstream_headers(self.headers))
                return self.send_bytes(r.status_code, r.content, r.headers.get("content-type", "application/json"))
            raise Refusal(404, "not_found_error", f"{path} is not available through this proxy")
        except Refusal as e:
            self.send_json(e.status, error_body(e.etype, e.message))

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        raw = self.read_body()
        try:
            tok = self.app.authenticate(self.headers)
            body = json.loads(raw or b"{}")
            if path == "/v1/messages/count_tokens":
                check_request(body, tok["models"])
                r = self.app.http.post(self.app.upstream + self.path, content=raw,
                                       headers=self.app.upstream_headers(self.headers))
                return self.send_bytes(r.status_code, r.content, r.headers.get("content-type", "application/json"))
            if path != "/v1/messages":
                raise Refusal(404, "not_found_error", f"{path} is not available through this proxy")
            try:
                check_request(body, tok["models"])
            except Refusal:
                self.app.record(tok, str(body.get("model")), None, "refused_request", bool(body.get("stream")),
                                None)
                raise
            price = self.app.prices.get(body["model"])
            if price is None:
                raise Refusal(403, "permission_error", f"no price configured for {body['model']}")
            est = worst_case_usd(body, price)
            try:
                self.app.reserve(tok, est)
            except Refusal:
                self.app.record(tok, body["model"], None, "refused_budget", bool(body.get("stream")), None)
                raise
            try:
                if body.get("stream"):
                    self._stream(tok, body, raw)
                else:
                    self._plain(tok, body, raw)
            finally:
                self.app.release(tok, est)
        except Refusal as e:
            self.send_json(e.status, error_body(e.etype, e.message))
        except json.JSONDecodeError:
            self.send_json(400, error_body("invalid_request_error", "request body is not JSON"))
        except httpx.HTTPError as e:
            self.send_json(502, error_body("api_error", f"upstream error: {type(e).__name__}: {e}"))

    def _relay_headers(self, r: httpx.Response) -> dict[str, str]:
        return {k: v for k, v in r.headers.items() if k.lower() not in HOP_HEADERS and k.lower() != "content-type"}

    def _plain(self, tok: dict[str, Any], body: dict[str, Any], raw: bytes) -> None:
        t0 = time.time()
        r = self.app.http.post(self.app.upstream + self.path, content=raw,
                               headers=self.app.upstream_headers(self.headers))
        rid = r.headers.get("request-id")
        try:
            resp = r.json()
        except ValueError:
            resp = {"_raw": r.text[:2000]}
        model = resp.get("model") if isinstance(resp, dict) and resp.get("model") in self.app.prices else body["model"]
        entry = self.app.record(tok, model, resp.get("usage") if r.status_code == 200 else None,
                                r.status_code, False, rid)
        self.app.transcripts.write(tok["owner"], tok.get("log_dir"), body, {
            "status": r.status_code, "request_id": rid, "response": resp, "usd": entry["usd"],
            "seconds": round(time.time() - t0, 3)})
        self.send_bytes(r.status_code, r.content, r.headers.get("content-type", "application/json"),
                        self._relay_headers(r))

    def _stream(self, tok: dict[str, Any], body: dict[str, Any], raw: bytes) -> None:
        t0 = time.time()
        acc = StreamAccumulator()
        client_ok = True
        with self.app.http.stream("POST", self.app.upstream + self.path, content=raw,
                                  headers=self.app.upstream_headers(self.headers)) as r:
            rid = r.headers.get("request-id")
            if r.status_code != 200:
                data = r.read()
                self.app.record(tok, body["model"], None, r.status_code, True, rid)
                try:
                    resp = json.loads(data)
                except ValueError:
                    resp = {"_raw": data[:2000].decode(errors="replace")}
                self.app.transcripts.write(tok["owner"], tok.get("log_dir"), body, {
                    "status": r.status_code, "request_id": rid, "response": resp, "usd": 0.0,
                    "seconds": round(time.time() - t0, 3)})
                return self.send_bytes(r.status_code, data, r.headers.get("content-type", "application/json"),
                                       self._relay_headers(r))
            self.send_response(200)
            self.send_header("content-type", r.headers.get("content-type", "text/event-stream"))
            self.send_header("transfer-encoding", "chunked")
            for k, v in self._relay_headers(r).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                for chunk in r.iter_bytes():
                    if not chunk:
                        continue
                    acc.feed(chunk)
                    if client_ok:
                        try:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            client_ok = False  # keep draining so the usage is still recorded
            except httpx.HTTPError as e:
                acc.error = acc.error or {"type": "upstream_disconnect", "message": f"{type(e).__name__}: {e}"}
            finally:
                final = acc.final()
                model = final.get("model") if final.get("model") in self.app.prices else body["model"]
                entry = self.app.record(tok, model, final.get("usage"), 200 if not acc.error else "stream_error",
                                        True, rid)
                self.app.transcripts.write(tok["owner"], tok.get("log_dir"), body, {
                    "status": 200, "request_id": rid, "response": final, "usd": entry["usd"],
                    "client_disconnected": not client_ok, "seconds": round(time.time() - t0, 3)})
        if client_ok:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass


def make_handler(app: LLMProxy) -> type[ProxyHandler]:
    return type("BoundProxyHandler", (ProxyHandler,), {"app": app})


def read_api_key(path: Path) -> str:
    env = os.environ.get("ANTHROPIC_API_KEY_FOR_PROXY")
    if env:
        return env.strip()
    if not path.exists():
        raise SystemExit(f"Anthropic API key not found. Put it in {path} (one line, chmod 600), "
                         f"or export ANTHROPIC_API_KEY_FOR_PROXY for the proxy process only.")
    mode = path.stat().st_mode & 0o077
    if mode:
        print(f"warning: {path} is readable by group/others; run: chmod 600 {path}", file=sys.stderr)
    return path.read_text().strip()


def main(argv: list[str] | None = None) -> None:
    from outer.config import load_config, token_secret

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--upstream", default=None, help="override proxy.upstream (tests use a fake)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    upstream = args.upstream or cfg["proxy"]["upstream"]
    api_key = read_api_key(cfg.path("proxy.api_key_file")) if args.upstream is None else "fake-key"
    app = LLMProxy(token_secret(cfg), Ledger(cfg.root / "ledger.jsonl"), cfg["prices_usd_per_mtok"],
                   upstream, api_key, cfg.root)
    sock = cfg.path("proxy.socket")
    servers = serve(make_handler(app), sock, cfg["proxy"].get("host"), cfg["proxy"].get("port"))
    print(f"llm proxy listening on {sock}" + (f" and port {cfg['proxy']['port']}" if cfg["proxy"].get("port") else "")
          + f"; upstream {upstream}; ledger {cfg.root / 'ledger.jsonl'}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    for s in servers:
        s.shutdown()


if __name__ == "__main__":
    main()
