"""The artifacts shelf: what a tool made from the file on request, by what it is, not when. An artifact is a picture, a table
of facts, or both, with the script that made it, a summary in words, and every number it stands behind kept beside it, so
the chat can cite the artifact and each number by address.

data/memory/<name>/artifacts/<id>/        artifact.json, facts.json, code.py, request.json, figure.png?, the table?, thoughts.json?

The folder hangs off the dataset's memory, so it is deleted with it. `design` tags the run an artifact is about; None means it
is about the data before any run. The address of an artifact is artifact:<id>; each number is artifact:<id>.<name>; the table,
when there is one, is artifact:<id>.table and the picture artifact:<id>.figure."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.common import config, trace

Kind = Literal["exploration", "picture"]
FILE = "artifact.json"


class Artifact(BaseModel):
    id: str = Field(description="short, url-safe, unique: eight hex characters")
    dataset: str
    kind: Kind = Field(description="picture: a figure was asked for and drawn; exploration: facts, a table, or both, with or without a figure")
    design: int | None = Field(description="the design run the artifact is about; null when it is about the data before any run")
    memory_version: int
    data_version: int = Field(description="the version of the table it was made from; 0 is the uploaded file")
    ask: str = Field(description="what was asked for, in the person's words")
    summary: str = Field(description="what it shows, in words; a picture's caption")
    facts: dict[str, float] = Field(description="every number it stands behind, by name")
    files: list[str] = Field(description="the file names in its folder")
    table: str | None = Field(default=None, description="the table file's name in the folder, when it wrote one")
    figure: bool = Field(default=False, description="figure.png is in the folder")
    sandbox: str = Field(default="subprocess", description="the fence the script ran in: seatbelt, bwrap, unshare, subprocess or docker")
    made_at: str = Field(description="ISO time")

    @property
    def address(self) -> str:
        return f"artifact:{self.id}"

    def addresses(self) -> set[str]:
        out = {self.address} | {f"{self.address}.{name}" for name in self.facts}
        if self.table:
            out.add(f"{self.address}.table")
        if self.figure:
            out.add(f"{self.address}.figure")
        return out

    def when(self) -> str:
        return "before any run" if self.design is None else f"design {self.design}"

    def render(self) -> str:
        """The artifact as lines with addresses, for the chat's material."""
        lines = [f"[{self.address}] {self.summary} ({self.kind}; {self.when()})"]
        lines.extend(f"  [{self.address}.{name}] {name}: {value:.4g}" for name, value in self.facts.items())
        if self.table:
            lines.append(f"  [{self.address}.table] a table: {self.table}")
        if self.figure:
            lines.append(f"  [{self.address}.figure] a picture")
        return "\n".join(lines)


def home(root: Path | None = None) -> Path:
    """The memory home: data/memory under the repository root, or under `root`. Read from the config on every call, so a test
    that points config.ROOT elsewhere moves the shelf with it."""
    return config.load(root=root).paths.memory


def folder(dataset: str, artifact_id: str, root: Path | None = None) -> Path:
    return home(root) / dataset / "artifacts" / artifact_id


def new_id() -> str:
    return secrets.token_hex(4)


def save(a: Artifact, root: Path | None = None) -> Path:
    d = folder(a.dataset, a.id, root)
    d.mkdir(parents=True, exist_ok=True)
    trace.write(d / FILE, a.model_dump_json(indent=2))
    return d


def load(dataset: str, artifact_id: str, root: Path | None = None) -> Artifact:
    return Artifact.model_validate_json((folder(dataset, artifact_id, root) / FILE).read_text())


def list_artifacts(dataset: str, kind: Kind | None = None, design: int | None = None, root: Path | None = None) -> list[Artifact]:
    """The dataset's artifacts, oldest first; by kind when one is given; about one design run when a design is given."""
    d = home(root) / dataset / "artifacts"
    if not d.is_dir():
        return []
    found = [Artifact.model_validate_json((p / FILE).read_text()) for p in d.iterdir() if (p / FILE).is_file()]
    if kind is not None:
        found = [a for a in found if a.kind == kind]
    if design is not None:
        found = [a for a in found if a.design == design]
    return sorted(found, key=lambda a: a.made_at)


def about(dataset: str, design: int | None, root: Path | None = None) -> list[Artifact]:
    """What the chat may cite at this point: every artifact about the data, plus the ones about the given run."""
    return [a for a in list_artifacts(dataset, root=root) if a.design is None or a.design == design]
