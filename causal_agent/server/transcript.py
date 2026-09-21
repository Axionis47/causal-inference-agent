"""The transcript on disk: one JSON line per turn under data/web/<name>/, appended as the conversation goes and read whole for the
page."""

from __future__ import annotations

from pathlib import Path

from causal_agent.server.models import Turn
from causal_agent.server.settings import Settings


def path(settings: Settings, name: str) -> Path:
    return settings.web_root / name / "transcript.jsonl"


def append(settings: Settings, name: str, turn: Turn) -> None:
    p = path(settings, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(turn.model_dump_json() + "\n")


def read(settings: Settings, name: str) -> list[Turn]:
    """Every turn on disk; a line that no longer parses is skipped, never a reason to lose the rest."""
    p = path(settings, name)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(Turn.model_validate_json(line))
            except Exception:
                continue
    return out
