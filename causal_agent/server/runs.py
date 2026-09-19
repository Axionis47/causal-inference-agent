"""Run directories as files. Names are checked, paths are resolved inside the run root, nothing else is served."""

from __future__ import annotations

import re
from pathlib import Path

from causal_agent.server.models import FileEntry, RunFiles
from causal_agent.server.settings import Settings

RUN_ID = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
KNOWN = {"artifacts.json": "application/json", "design.json": "application/json", "design.md": "text/markdown", "report.md": "text/plain", "figures.json": "application/json",
         "table.csv": "text/csv", "bins.csv": "text/csv", "canon.csv": "text/csv", "scores.csv": "text/csv", "panel.csv": "text/csv"}
THOUGHTS_MARK = "MODEL THOUGHTS"


class NotFound(Exception):
    pass


def run_dir(s: Settings, run_id: str) -> Path:
    if not RUN_ID.match(run_id or ""):
        raise NotFound(run_id)
    root = s.run_root.resolve()
    p = (root / run_id).resolve()
    if p.parent != root or not p.is_dir():
        raise NotFound(run_id)
    return p


def list_files(s: Settings, run_id: str) -> RunFiles:
    d = run_dir(s, run_id)
    files = [FileEntry(name=p.name, size=p.stat().st_size) for p in sorted(d.iterdir()) if p.is_file() and p.name in KNOWN]
    return RunFiles(run_id=run_id, files=files)


def file_path(s: Settings, run_id: str, filename: str) -> tuple[Path, str]:
    if filename not in KNOWN:
        raise NotFound(filename)
    p = run_dir(s, run_id) / filename
    if not p.is_file():
        raise NotFound(filename)
    return p, KNOWN[filename]


def report_text(path: Path, raw: bool = False) -> str:
    text = path.read_text(errors="replace")
    if raw:
        return text
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(THOUGHTS_MARK):
            return "\n".join(lines[:i]).rstrip() + "\n"
    return text
