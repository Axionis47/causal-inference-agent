"""Drawn pictures as the page reads them: the view of an artifact with the address it is served from, and the file behind
that address. Names are checked; a path resolves only inside the dataset's memory home."""

from __future__ import annotations

import re
from pathlib import Path

from causal_agent.server.models import ArtifactView
from causal_agent.server.settings import Settings
from causal_agent.viz import store as VS
from causal_agent.viz.store import Artifact, Moment

NAME = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
ID = re.compile(r"^[0-9a-f]{8}$")


class NotFound(Exception):
    pass


def url(name: str, a: Artifact) -> str:
    return f"/api/artifacts/{name}/{a.moment}/{a.design or 0}/{a.id}/figure.png"


def view(name: str, a: Artifact) -> ArtifactView:
    return ArtifactView(
        id=a.id, address=a.address, moment=a.moment, design=a.design, ask=a.ask, caption=a.caption, facts=dict(a.facts), made_at=a.made_at, url=url(name, a)
    )


def view_of(name: str, raw: dict | None) -> ArtifactView | None:
    """The view of an artifact a graph payload carried, or None."""
    return view(name, Artifact.model_validate(raw)) if raw else None


def before_runs(s: Settings, name: str) -> list[ArtifactView]:
    return [view(name, a) for a in VS.list_artifacts(name, "pre", root=s.root)]


def after_run(s: Settings, name: str, design: int) -> list[ArtifactView]:
    return [view(name, a) for a in VS.list_artifacts(name, "post", design, root=s.root)]


def figure_path(s: Settings, name: str, moment: str, design: int, artifact_id: str) -> Path:
    """The picture behind a served address; design 0 means a pre artifact."""
    if not NAME.match(name or "") or not ID.match(artifact_id or "") or moment not in ("pre", "post") or (moment == "post" and design < 1):
        raise NotFound(artifact_id)
    when: Moment = "pre" if moment == "pre" else "post"
    p = VS.folder(name, when, None if when == "pre" else design, artifact_id, root=s.root) / "figure.png"
    home = VS.home(s.root).resolve()
    if home not in p.resolve().parents or not p.is_file():
        raise NotFound(artifact_id)
    return p
