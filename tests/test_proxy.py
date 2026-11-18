"""Phase 3: the llm proxy against a fake upstream (streaming, usage accounting, caps, allowlist, logs)."""

from __future__ import annotations

import json
import time

import anthropic
import httpx
import pytest

from outer.sandbox import start_relay
from services.llm.ledger import cost_usd, normalize_usage, read_entries
from services.tokens import TokenError, run_token, sign, verify, world_token

MODEL = "claude-opus-5-5"


def client(env, owner="t/owner", cap=2.0, models=(MODEL,), group=None, group_cap=None, log_dir="runs/x"):
    port = start_relay(str(env.llm_socket))  # the same relay the sandbox uses
    tok = run_token(env.secret, owner, cap, list(models), log_dir, group=group, group_cap_usd=group_cap)
    return anthropic.Anthropic(base_url=f"http://127.0.0.1:{port}", api_key=tok, max_retries=0), tok, port


def test_tokens_sign_verify(fwenv):
    tok = world_token(fwenv.secret, "t_1", "2023-01-01T00:00:00+00:00")
    assert verify(tok, fwenv.secret, "world")["task_id"] == "t_1"
    with pytest.raises(TokenError):
        verify(tok, fwenv.secret, "llm")
    with pytest.raises(TokenError):
        verify(tok, b"other-secret", "world")
    body, mac = tok.split(".")
    with pytest.raises(TokenError):
        verify(body[:-2] + "AA." + mac, fwenv.secret, "world")
    with pytest.raises(TokenError):
        verify(sign({"kind": "world"}, fwenv.secret, ttl_s=-1), fwenv.secret, "world")


def test_nonstreaming_forward_and_ledger(fwenv):
    c, tok, _ = client(fwenv)
    r = c.messages.create(model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}])
    assert r.content[0].text == "hello from fake upstream"
    # the upstream saw the real key, never the run token
    h = fwenv.upstream.headers[-1]
    assert h["x-api-key"] == "fake-key" and tok not in json.dumps(h)
    e = read_entries(fwenv.root / "ledger.jsonl")[-1]
    assert e["owner"] == "t/owner" and e["status"] == 200 and e["stream"] is False
    expect = cost_usd(MODEL, normalize_usage(r.usage.model_dump()), fwenv.cfg["prices_usd_per_mtok"])
    assert e["usd"] == pytest.approx(expect) and e["usd"] > 0
    assert e["cache_read_input_tokens"] == 1000 and e["cache_creation_5m"] == 100
    tr = [json.loads(x) for x in (fwenv.root / "runs/x/transcript.jsonl").read_text().splitlines()]
    assert tr[-1]["response"]["content"][0]["text"] == "hello from fake upstream"
    assert tr[-1]["request"]["messages"][0]["content"] == "hi"


def test_price_arithmetic():
    prices = {"m": {"input": 4.0, "output": 20.0, "cache_write_5m": 5.0, "cache_write_1h": 8.0, "cache_read": 0.2}}
    u = normalize_usage({"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
                         "cache_creation_input_tokens": 2_000_000,
                         "cache_creation": {"ephemeral_5m_input_tokens": 1_000_000, "ephemeral_1h_input_tokens": 1_000_000}})
    assert cost_usd("m", u, prices) == pytest.approx(4 + 20 + 0.2 + 5 + 8)


def test_streaming_relay_and_usage(fwenv):
    fwenv.upstream.reply = lambda body: "a streamed reply that spans several chunks"
    c, _, _ = client(fwenv)
    with c.messages.stream(model=MODEL, max_tokens=100, messages=[{"role": "user", "content": "hi"}]) as s:
        text = "".join(s.text_stream)
        final = s.get_final_message()
    assert text == "a streamed reply that spans several chunks"
    e = read_entries(fwenv.root / "ledger.jsonl")[-1]
    assert e["stream"] is True and e["output_tokens"] == final.usage.output_tokens > 1
    assert e["input_tokens"] == final.usage.input_tokens
    tr = json.loads((fwenv.root / "runs/x/transcript.jsonl").read_text().splitlines()[-1])
    assert tr["response"]["content"][0]["text"] == text


def test_model_allowlist_and_forbidden_features(fwenv):
    c, _, _ = client(fwenv)
    with pytest.raises(anthropic.PermissionDeniedError):
        c.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[{"role": "user", "content": "x"}])
    with pytest.raises(anthropic.PermissionDeniedError):
        c.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "x"}],
                          tools=[{"type": "web_search_20260209", "name": "web_search"}])
    with pytest.raises(anthropic.PermissionDeniedError):
        c.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "x"}],
                          extra_body={"mcp_servers": [{"type": "url", "url": "http://x", "name": "x"}]})
    with pytest.raises(anthropic.PermissionDeniedError):
        c.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "x"}],
                          extra_body={"fallbacks": "default"})
    n_up = len(fwenv.upstream.requests)
    assert n_up == 0  # nothing reached the upstream
    # client-side tools are fine
    c.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "x"}],
                      tools=[{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}])
    assert len(fwenv.upstream.requests) == 1


def test_bad_token(fwenv):
    port = start_relay(str(fwenv.llm_socket))
    c = anthropic.Anthropic(base_url=f"http://127.0.0.1:{port}", api_key="sk-ant-not-a-run-token", max_retries=0)
    with pytest.raises(anthropic.AuthenticationError):
        c.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "x"}])


def test_cap_enforced(fwenv):
    # each request bills its full max_tokens: 20000 x $20/M = $0.40; cap $1 allows two, not three
    fwenv.upstream.full_output = True
    c, _, port = client(fwenv, owner="capped", cap=1.0)
    ok = 0
    with pytest.raises(anthropic.APIStatusError) as ei:
        for _ in range(10):
            c.messages.create(model=MODEL, max_tokens=20000, messages=[{"role": "user", "content": "x"}])
            ok += 1
    assert ei.value.status_code == 402 and "budget" in str(ei.value)
    assert ok == 2
    entries = [e for e in read_entries(fwenv.root / "ledger.jsonl") if e["owner"] == "capped"]
    assert entries[-1]["status"] == "refused_budget"
    assert sum(e["usd"] for e in entries) <= 1.0
    u = httpx.get(f"http://127.0.0.1:{port}/_proxy/usage", headers={"x-api-key": c.api_key}).json()
    assert u["cap_usd"] == 1.0 and u["spent_usd"] == pytest.approx(sum(e["usd"] for e in entries))


def test_group_cap(fwenv):
    fwenv.upstream.full_output = True  # each request bills $0.20
    a, _, _ = client(fwenv, owner="g/a", cap=5.0, group="g", group_cap=0.3)
    b, _, _ = client(fwenv, owner="g/b", cap=5.0, group="g", group_cap=0.3)
    a.messages.create(model=MODEL, max_tokens=10000, messages=[{"role": "user", "content": "x"}])
    with pytest.raises(anthropic.APIStatusError) as ei:
        b.messages.create(model=MODEL, max_tokens=10000, messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code == 402 and "group" in str(ei.value)


def test_count_tokens_passthrough_free(fwenv):
    c, _, _ = client(fwenv, owner="ct")
    r = c.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": "x"}])
    assert r.input_tokens == 42
    assert not [e for e in read_entries(fwenv.root / "ledger.jsonl") if e["owner"] == "ct"]


def test_media_elided_in_transcript(fwenv):
    import base64

    png = base64.b64encode(b"\x89PNG fake image bytes" * 100).decode()
    c, _, _ = client(fwenv, log_dir="runs/media")
    c.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
        {"type": "text", "text": "what is this"}]}])
    line = (fwenv.root / "runs/media/transcript.jsonl").read_text()
    assert png not in line and "media/" in line
    assert list((fwenv.root / "runs/media/media").glob("*.png"))


def test_transcript_stores_only_new_messages(fwenv):
    c, _, _ = client(fwenv, log_dir="runs/conv", owner="conv")
    msgs = [{"role": "user", "content": "first"}]
    r = c.messages.create(model=MODEL, max_tokens=10, messages=msgs)
    msgs += [{"role": "assistant", "content": r.content[0].text}, {"role": "user", "content": "second"}]
    c.messages.create(model=MODEL, max_tokens=10, messages=msgs)
    lines = [json.loads(x) for x in (fwenv.root / "runs/conv/transcript.jsonl").read_text().splitlines()]
    assert lines[0]["request"]["messages_from"] == 0 and len(lines[0]["request"]["messages"]) == 1
    assert lines[1]["request"]["messages_from"] == 1 and len(lines[1]["request"]["messages"]) == 2


def test_concurrent_requests_respect_cap(fwenv):
    import threading

    fwenv.upstream.reply = lambda body: (time.sleep(0.3), "slow")[1]
    c, _, _ = client(fwenv, owner="conc", cap=1.0)
    results = []

    def go():
        try:
            c.messages.create(model=MODEL, max_tokens=20000, messages=[{"role": "user", "content": "x"}])
            results.append("ok")
        except anthropic.APIStatusError as e:
            results.append(e.status_code)

    ts = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("ok") == 2 and results.count(402) == 4
