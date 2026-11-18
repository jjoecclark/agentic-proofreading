"""Sandboxes for generated code (PLAN.md Section 7).

Harnesses and the meta-agent run in an Apptainer container started with ``--containall --net
--network none``: no network interface but loopback, a clean environment, no home directory, and only
these paths bound in:

- harness run: the harness folder (read-only), one task file (read-only), an empty work directory,
  an output directory, the world client library + API.md, and the two service sockets;
- meta-agent: its own work directory (working copy of the harness, train task files, feedback packet,
  TASK.md, API.md, world client, ``score_train``), a probe directory, and the two sockets.

``truth/``, ``data/operations.csv``, and val/test task files are never bound. The only reachable
services are `world` and the `llm` proxy, through Unix sockets. Inside, a small relay exposes the proxy
on 127.0.0.1 so the Anthropic SDK and the Claude Code CLI can use ``ANTHROPIC_BASE_URL``.

This file has two halves. The host half (``ApptainerRunner``, ``LocalRunner``, ``scan_harness``) is
used by outer/. The container half (``main``: ``harness`` and ``meta`` modes) is bound into the
container as /opt/fw/fw_runner.py; it uses only the standard library at import time.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

SANDBOX_VERSION = "1"

# In-container paths
C_HARNESS, C_TASK, C_WORK, C_OUT, C_PROBE = "/harness", "/task/task.json", "/work", "/out", "/probe"
C_WORLD_SOCK, C_LLM_SOCK, C_LIB = "/run/fw/world.sock", "/run/fw/llm.sock", "/opt/fw"


# ============================================================================ shared: limits


@dataclass
class Limits:
    """What a harness may use for one task. ``remaining_s()`` is wall time left; ``usage()`` asks the
    proxy how much of the dollar cap is spent."""

    wall_s: float
    deadline_unix: float
    usd_cap: float
    models: list[str] = field(default_factory=list)

    def remaining_s(self) -> float:
        return self.deadline_unix - time.time()

    def usage(self) -> dict[str, Any]:
        import urllib.request

        base = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
        req = urllib.request.Request(base + "/_proxy/usage",
                                     headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", "")})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def remaining_usd(self) -> float:
        return float(self.usage()["remaining_usd"])


# ============================================================================ container half


def _pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def start_relay(unix_path: str) -> int:
    """Listen on 127.0.0.1:<port> and forward each connection to a Unix socket. Returns the port."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(256)

    def loop() -> None:
        while True:
            c, _ = srv.accept()
            u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                u.connect(unix_path)
            except OSError:
                c.close()
                continue
            threading.Thread(target=_pipe, args=(c, u), daemon=True).start()
            threading.Thread(target=_pipe, args=(u, c), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    return srv.getsockname()[1]


def _load_config(out_dir: str) -> dict[str, Any]:
    with open(os.path.join(out_dir, "config.json")) as f:
        return json.load(f)


def _setup_env(conf: dict[str, Any]) -> None:
    llm_sock = os.environ.get("FW_LLM_SOCKET", C_LLM_SOCK)
    port = start_relay(llm_sock)
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    os.environ["ANTHROPIC_API_KEY"] = conf["llm_token"]
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    os.environ["WORLD_SOCKET"] = os.environ.get("FW_WORLD_SOCKET", C_WORLD_SOCK)
    if conf.get("world_token"):
        os.environ["WORLD_TOKEN"] = conf["world_token"]
    if os.environ.get("FW_HOME"):  # apptainer does not let --env override HOME
        os.environ["HOME"] = os.environ["FW_HOME"]
    for k in ("HOME", "TMPDIR"):
        if os.environ.get(k):
            os.makedirs(os.environ[k], exist_ok=True)


def harness_main() -> int:
    out = os.environ.get("FW_OUTDIR", C_OUT)
    conf = _load_config(out)
    _setup_env(conf)
    t0 = time.time()
    try:
        harness_dir = os.environ.get("FW_HARNESS_DIR", C_HARNESS)
        task_file = os.environ.get("FW_TASK_FILE", C_TASK)
        workdir = Path(os.environ.get("FW_WORKDIR", C_WORK))
        sys.path.insert(0, harness_dir)
        os.chdir(workdir)
        import importlib

        import anthropic
        from world_client import WorldClient

        with open(task_file) as f:
            state = json.load(f)
        harness = importlib.import_module("harness")
        world = WorldClient(socket_path=os.environ["WORLD_SOCKET"], token=os.environ["WORLD_TOKEN"])
        llm = anthropic.Anthropic()
        limits = Limits(**conf["limits"])
        decision = harness.run(state, world, llm, workdir, limits)
        text = json.dumps(decision)
        with open(os.path.join(out, "decision.json"), "w") as f:
            f.write(text)
        return 0
    except BaseException as e:  # noqa: BLE001
        with open(os.path.join(out, "error.json"), "w") as f:
            json.dump({"type": type(e).__name__, "message": str(e)[:4000],
                       "traceback": traceback.format_exc()[-20000:], "after_s": time.time() - t0}, f)
        traceback.print_exc()
        return 1


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {"_type": type(obj).__name__, **{k: _jsonable(v) for k, v in asdict(obj).items()}}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def meta_main() -> int:
    """Run the meta-agent (Claude Agent SDK) in /work with the prompt in /out/config.json."""
    out = os.environ.get("FW_OUTDIR", C_OUT)
    conf = _load_config(out)
    _setup_env(conf)
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, query

    model = conf["model"]
    env = {"ANTHROPIC_BASE_URL": os.environ["ANTHROPIC_BASE_URL"], "ANTHROPIC_API_KEY": conf["llm_token"],
           "ANTHROPIC_MODEL": model, "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
           "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model,
           "CLAUDE_CODE_SUBAGENT_MODEL": model,
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1",
           "DISABLE_ERROR_REPORTING": "1", "WORLD_SOCKET": os.environ["WORLD_SOCKET"],
           "BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "1800000",
           "HOME": os.environ.get("HOME", "/work/.home"), "PATH": "/work/bin:" + os.environ.get("PATH", "")}
    opts = ClaudeAgentOptions(
        cwd=os.environ.get("FW_WORKDIR", C_WORK), model=model, tools=list(conf["tools"]),
        allowed_tools=list(conf["tools"]), disallowed_tools=["WebFetch", "WebSearch"],
        permission_mode="bypassPermissions", system_prompt={"type": "preset", "preset": "claude_code",
                                                            "append": conf.get("system_append", "")},
        setting_sources=[], env=env, max_turns=conf.get("max_turns"))
    log_path = os.path.join(out, "agent_messages.jsonl")

    async def go() -> int:
        n = 0
        with open(log_path, "a") as log:
            async for msg in query(prompt=conf["prompt"], options=opts):
                n += 1
                log.write(json.dumps({"ts": time.time(), "message": _jsonable(msg)}) + "\n")
                log.flush()
        return n

    try:
        n = asyncio.run(go())
        with open(os.path.join(out, "meta_done.json"), "w") as f:
            json.dump({"messages": n}, f)
        return 0
    except BaseException as e:  # noqa: BLE001
        with open(os.path.join(out, "error.json"), "w") as f:
            json.dump({"type": type(e).__name__, "message": str(e)[:4000],
                       "traceback": traceback.format_exc()[-20000:]}, f)
        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv else ""
    if mode == "harness":
        return harness_main()
    if mode == "meta":
        return meta_main()
    print("usage: fw_runner.py harness|meta", file=sys.stderr)
    return 2


# ============================================================================ host half


REPO = Path(__file__).resolve().parent.parent


def _kill_tree(p: subprocess.Popen, grace_s: float = 10.0) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


class Interrupted(BaseException):
    """The outer harness is stopping. A BaseException, so no `except Exception` records the killed run
    as the harness's crash: the task stays unfinished and is rerun on resume."""


_LIVE: set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()
STOPPING = threading.Event()


def stop_all_containers(grace_s: float = 5.0) -> int:
    """Kill every sandbox process started by this Python process; later run_process calls raise Interrupted."""
    STOPPING.set()
    with _LIVE_LOCK:
        procs = list(_LIVE)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace_s
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(0.1)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return len(procs)


def install_stop_handlers() -> None:
    """Make Ctrl-C, SIGTERM and SIGHUP stop the containers too. Containers run in their own process
    groups (so a timeout kills a whole tree), so these signals would not reach them otherwise, and a
    meta-agent container could keep running, and spending, for hours after the outer harness is gone."""
    def handler(signum: int, frame: Any) -> None:
        n = stop_all_containers()
        print(f"\nstopping on signal {signum}: killed {n} container(s). Rerun the same command to resume.",
              file=sys.stderr, flush=True)
        raise KeyboardInterrupt

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, handler)


def run_process(cmd: list[str], env: dict[str, str] | None, wall_s: float, log_path: Path,
                cwd: Path | None = None) -> dict[str, Any]:
    """Run ``cmd`` in its own process group with a wall-time limit; stdout+stderr go to ``log_path``.
    Raises Interrupted if the outer harness is stopping (stop_all_containers)."""
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if STOPPING.is_set():
        raise Interrupted("the outer harness is stopping")
    with open(log_path, "ab") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        with _LIVE_LOCK:
            _LIVE.add(p)
        timed_out = False
        try:
            if STOPPING.is_set():  # stopped between the check above and registration
                _kill_tree(p, grace_s=5.0)
            p.wait(timeout=wall_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(p)
        except BaseException:  # e.g. KeyboardInterrupt in this thread
            _kill_tree(p, grace_s=5.0)
            raise
        finally:
            with _LIVE_LOCK:
                _LIVE.discard(p)
    if STOPPING.is_set():
        raise Interrupted("the outer harness is stopping; this run's result is not recorded")
    return {"sandbox_version": SANDBOX_VERSION, "exit_code": p.returncode, "timed_out": timed_out,
            "wall_s": round(time.time() - t0, 3)}


class ApptainerRunner:
    """Runs harnesses and the meta-agent in Apptainer containers with no network."""

    def __init__(self, image: Path, world_socket: Path, llm_socket: Path, apptainer: str = "apptainer"):
        self.image, self.world_socket, self.llm_socket = Path(image), Path(world_socket), Path(llm_socket)
        self.apptainer = apptainer
        if not self.image.exists():
            raise SystemExit(f"sandbox image {self.image} is missing; build it with: "
                             f"apptainer build {self.image} {REPO / 'outer' / 'sandbox.def'}")

    def _base(self) -> list[str]:
        return [self.apptainer, "exec", "--containall", "--cleanenv", "--net", "--network", "none",
                "--no-privs", "--pwd", C_WORK,
                "-B", f"{self.world_socket}:{C_WORLD_SOCK}", "-B", f"{self.llm_socket}:{C_LLM_SOCK}",
                "-B", f"{REPO / 'services/world/client.py'}:{C_LIB}/world_client.py:ro",
                "-B", f"{REPO / 'services/world/API.md'}:{C_LIB}/API.md:ro",
                "-B", f"{REPO / 'outer/sandbox.py'}:{C_LIB}/fw_runner.py:ro",
                "--env", f"PYTHONPATH={C_LIB}", "--env", f"FW_HOME={C_WORK}/.home", "--env", f"TMPDIR={C_WORK}/.tmp"]

    def run_harness(self, harness_dir: Path, task_file: Path, workdir: Path, outdir: Path, wall_s: float,
                    log_path: Path) -> dict[str, Any]:
        cmd = self._base() + ["-B", f"{harness_dir}:{C_HARNESS}:ro", "-B", f"{task_file}:{C_TASK}:ro",
                              "-B", f"{workdir}:{C_WORK}", "-B", f"{outdir}:{C_OUT}",
                              str(self.image), "python", f"{C_LIB}/fw_runner.py", "harness"]
        return run_process(cmd, None, wall_s, log_path)

    def run_meta(self, workdir: Path, probedir: Path, outdir: Path, wall_s: float, log_path: Path) -> dict[str, Any]:
        cmd = self._base() + ["-B", f"{workdir}:{C_WORK}", "-B", f"{probedir}:{C_PROBE}", "-B", f"{outdir}:{C_OUT}",
                              str(self.image), "python", f"{C_LIB}/fw_runner.py", "meta"]
        return run_process(cmd, None, wall_s, log_path)


class LocalRunner:
    """No isolation: runs the container half with the host Python. For tests only."""

    def __init__(self, world_socket: Path, llm_socket: Path, python: str = sys.executable):
        self.world_socket, self.llm_socket, self.python = Path(world_socket), Path(llm_socket), python

    def _lib(self, outdir: Path) -> Path:
        lib = outdir / ".lib"
        lib.mkdir(parents=True, exist_ok=True)
        for name, src in (("world_client.py", REPO / "services/world/client.py"),
                          ("fw_runner.py", REPO / "outer/sandbox.py")):
            if not (lib / name).exists():
                (lib / name).symlink_to(src)
        return lib

    def run_harness(self, harness_dir: Path, task_file: Path, workdir: Path, outdir: Path, wall_s: float,
                    log_path: Path) -> dict[str, Any]:
        lib = self._lib(outdir)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(lib), "HOME": str(workdir / ".home"),
               "TMPDIR": str(workdir / ".tmp"), "FW_OUTDIR": str(outdir), "FW_HARNESS_DIR": str(harness_dir),
               "FW_TASK_FILE": str(task_file), "FW_WORKDIR": str(workdir), "FW_LLM_SOCKET": str(self.llm_socket),
               "FW_WORLD_SOCKET": str(self.world_socket)}
        for k in ("FW_TEST_TRUTH_DIR",):  # oracle harness in tests
            if k in os.environ:
                env[k] = os.environ[k]
        return run_process([self.python, str(lib / "fw_runner.py"), "harness"], env, wall_s, log_path)

    def run_meta(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError("the meta-agent runs only in the Apptainer sandbox")


def make_runner(cfg: Any) -> ApptainerRunner:
    return ApptainerRunner(cfg.path("sandbox.image"), cfg.path("world.socket"), cfg.path("proxy.socket"))


# ============================================================================ static scan

NETWORK_PATTERNS = [
    (r"^\s*(import|from)\s+(socket|ssl|requests|urllib3?|http\.client|httpx|httpx2|aiohttp|websockets?|pycurl|"
     r"ftplib|smtplib|telnetlib|paramiko|grpc|zmq)\b", "network module import"),
    (r"\burllib\.request\b|\burlopen\s*\(", "urllib"),
    (r"\bsocket\.(socket|create_connection|getaddrinfo)\b", "raw socket"),
    (r"\basyncio\.open_(unix_)?connection\b", "asyncio connection"),
    (r"\b(curl|wget|nc|ncat|netcat|ssh|scp|rsync|telnet)\b\s", "network command"),
    (r"ANTHROPIC_BASE_URL\s*=|base_url\s*=", "overrides the Claude API endpoint"),
    (r"WORLD_(SOCKET|URL)\s*=", "overrides the world endpoint"),
]


def scan_harness(harness_dir: Path) -> list[dict[str, Any]]:
    """Report lines that look like network access other than the two clients."""
    hits = []
    for p in sorted(Path(harness_dir).rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".sh", ".bash", ".js", ".ts", "") or p.stat().st_size > 2_000_000:
            continue
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("#") and p.suffix == ".py":
                continue
            for pat, what in NETWORK_PATTERNS:
                if re.search(pat, line):
                    hits.append({"file": str(p.relative_to(harness_dir)), "line": i, "what": what,
                                 "text": line.strip()[:200]})
    return hits


def copy_tree(src: Path, dst: Path, max_mb: float = 200.0) -> None:
    """Copy a harness folder (skipping caches); refuse folders over ``max_mb``."""
    size = sum(p.stat().st_size for p in Path(src).rglob("*") if p.is_file())
    if size > max_mb * 1e6:
        raise ValueError(f"{src} is {size / 1e6:.0f} MB, over the {max_mb:.0f} MB limit for a harness")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
                    dirs_exist_ok=True)


if __name__ == "__main__":
    sys.exit(main())
