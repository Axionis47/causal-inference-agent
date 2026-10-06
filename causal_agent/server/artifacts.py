"""Artifacts as the page reads them: the view of an artifact with the addresses its files are served from, and the file behind
such an address. Names are checked; a path resolves only inside the dataset's memory home and only to a file the artifact
lists."""

from __future__ import annotations

import re
from pathlib import Path

from causal_agent.memory import artifacts as VA
from causal_agent.memory.artifacts import Artifact
from causal_agent.server.models import ArtifactView
from causal_agent.server.settings import Settings

NAME = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
ID = re.compile(r"^[0-9a-f]{8}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9_\-.]{1,120}$")
MEDIA = {".png": "image/png", ".csv": "text/csv", ".json": "application/json", ".txt": "text/plain", ".py": "text/plain"}


class NotFound(Exception):
    pass


def url(name: str, a: Artifact, file: str) -> str:
    return f"/api/artifacts/{name}/{a.id}/{file}"


def view(name: str, a: Artifact) -> ArtifactView:
    return ArtifactView(
        id=a.id,
        address=a.address,
        kind=a.kind,
        design=a.design,
        ask=a.ask,
        summary=a.summary,
        facts=dict(a.facts),
        figure=a.figure,
        table=a.table,
        made_at=a.made_at,
        url=url(name, a, "figure.png") if a.figure else None,
        table_url=url(name, a, a.table) if a.table else None,
    )


def view_of(name: str, raw: dict | None) -> ArtifactView | None:
    """The view of an artifact a graph payload carried, or None."""
    return view(name, Artifact.model_validate(raw)) if raw else None


def of_memory(s: Settings, name: str) -> list[ArtifactView]:
    """The artifacts about the data before any run."""
    return [view(name, a) for a in VA.list_artifacts(name, root=s.root) if a.design is None]


def of_design(s: Settings, name: str, design: int) -> list[ArtifactView]:
    """The artifacts about one design run."""
    return [view(name, a) for a in VA.list_artifacts(name, design=design, root=s.root)]


def file_path(s: Settings, name: str, artifact_id: str, file: str) -> tuple[Path, str]:
    """The file behind a served address and its media type: a file the artifact lists, inside the memory home."""
    if not NAME.match(name or "") or not ID.match(artifact_id or "") or not FILE_NAME.match(file or "") or file.startswith("."):
        raise NotFound(artifact_id)
    folder = VA.folder(name, artifact_id, root=s.root)
    try:
        a = VA.load(name, artifact_id, root=s.root)
    except (OSError, ValueError):
        raise NotFound(artifact_id) from None
    if file not in a.files:
        raise NotFound(file)
    p = folder / file
    home = VA.home(s.root).resolve()
    if home not in p.resolve().parents or not p.is_file():
        raise NotFound(file)
    return p, MEDIA.get(p.suffix, "application/octet-stream")
