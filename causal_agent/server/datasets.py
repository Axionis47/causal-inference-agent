"""Metadata still used by the separate conversation runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path

from causal_agent.server.settings import Settings


class NotFound(Exception):
    pass


# ------------------------------------------------------------------ index and meta


def meta_path(s: Settings, name: str) -> Path:
    return s.web_root / name / "meta.json"


def read_meta(s: Settings, name: str) -> dict | None:
    p = meta_path(s, name)
    return json.loads(p.read_text()) if p.exists() else None


def write_meta(s: Settings, name: str, meta: dict) -> None:
    """Written whole or not at all: the worker thread writes while the page's poll reads, and a reader must never see half a file."""
    p = meta_path(s, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, p)
