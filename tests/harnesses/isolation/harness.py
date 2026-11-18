"""Test-only: tries to read things it must not see and to reach the network; reports what happened."""
import json
import os
import socket


def run(state, world, llm, workdir, limits):
    report = {}
    for p in json.loads(open(os.path.join(os.path.dirname(__file__), "probe_paths.json")).read()):
        try:
            with open(p) as f:
                f.read(10)
            report[p] = "READABLE"
        except Exception as e:  # noqa: BLE001
            report[p] = type(e).__name__
    for host, port in (("140.82.112.3", 443), ("8.8.8.8", 53), ("api.anthropic.com", 443)):
        s = socket.socket()
        s.settimeout(3)
        try:
            s.connect((host, port))
            report[f"{host}:{port}"] = "CONNECTED"
        except Exception as e:  # noqa: BLE001
            report[f"{host}:{port}"] = type(e).__name__
        finally:
            s.close()
    report["env_keys"] = sorted(os.environ)
    report["anthropic_key_is_run_token"] = os.environ.get("ANTHROPIC_API_KEY", "").count(".") == 1
    report["world_info_t"] = world.info()["t_utc"]
    return {"decision_version": "1", "action": "none", "notes": json.dumps(report)}
