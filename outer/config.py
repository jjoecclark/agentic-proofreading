"""Load config.yaml, resolve paths, and check the pinned operations CSV (PLAN.md Sections 3.1, 10)."""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


class Config(dict):
    """config.yaml as a dict, plus helpers. ``cfg.path("world.socket")`` resolves a path setting."""

    root: Path

    def get_key(self, dotted: str) -> Any:
        node: Any = self
        for part in dotted.split("."):
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        return resolve_path(self.get_key(dotted), self.root)


def resolve_path(value: str | os.PathLike, root: Path = ROOT) -> Path:
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else (root / p)


def load_config(path: Path | None = None, check_data: bool = True) -> Config:
    path = Path(path) if path else CONFIG_PATH
    with open(path) as f:
        cfg = Config(yaml.safe_load(f))
    cfg.root = path.resolve().parent
    if check_data:
        check_operations_csv(cfg)
    return cfg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_operations_csv(cfg: Config) -> None:
    """The CSV must be the pinned copy (sha256 prefix in config.yaml)."""
    csv = cfg.path("data.operations_csv")
    if not csv.exists():
        raise SystemExit(f"{csv} is missing; copy it from {cfg['data']['source_repo']} at commit "
                         f"{cfg['data']['source_commit']}")
    digest = sha256_file(csv)
    if not digest.startswith(cfg["data"]["sha256_prefix"]):
        raise SystemExit(f"{csv}: sha256 {digest[:16]} does not match the pinned prefix "
                         f"{cfg['data']['sha256_prefix']}")


def token_secret(cfg: Config, create: bool = True) -> bytes:
    """The HMAC key that signs world session tokens and proxy run tokens. Lives outside the
    repository (never bound into a container); created on first use with mode 600."""
    path = cfg.path("secrets.token_secret_file")
    if not path.exists():
        if not create:
            raise SystemExit(f"token secret {path} is missing")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    return path.read_text().strip().encode()
