"""Where a drawn artifact lives and how it is addressed. An artifact is a picture the drawing tool made from the file,
with every number the picture shows kept beside it, so the chat can cite the picture and each number by address.

data/memory/<name>/viz/pre/<id>/                    drawn before any design run
data/memory/<name>/designs/<n>/viz/<id>/            drawn after design run n

Each folder holds request.json, code.py, figure.png, facts.json and artifact.json. The folder hangs off the dataset's
memory, or off the design run, so it is deleted with them. The address of an artifact is artifact:<id>; each number is
artifact:<id>.<name>."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.common import config

Moment = Literal["pre", "post"]
FILE = "artifact.json"


class Artifact(BaseModel):
    id: str = Field(description="short, url-safe, unique: eight hex characters")
    dataset: str
    moment: Moment
    design: int | None = Field(description="the design run a post artifact belongs to; null for pre")
    memory_version: int
    ask: str = Field(description="what was asked for, in the person's words")
    caption: str = Field(description="one sentence on what the picture shows")
    facts: dict[str, float] = Field(description="every number the picture shows, by name")
    made_at: str = Field(description="ISO time")
    files: list[str] = Field(description="the file names in its folder")
    sandbox: str = Field(default="subprocess", description="the fence the script ran in: seatbelt, bwrap, unshare, subprocess or docker")

    @property
    def address(self) -> str:
        return f"artifact:{self.id}"

    def addresses(self) -> set[str]:
        return {self.address} | {f"{self.address}.{name}" for name in self.facts}

    def render(self) -> str:
        """The artifact as lines with addresses, for the chat's material."""
        when = "before the run" if self.moment == "pre" else f"design {self.design}"
        lines = [f"[{self.address}] {self.caption} ({when})"]
        lines.extend(f"  [{self.address}.{name}] {name}: {value:.4g}" for name, value in self.facts.items())
        return "\n".join(lines)


def home(root: Path | None = None) -> Path:
    """The memory home: data/memory under the repository root, or under `root`."""
    return config.load(root=root).paths.memory


def folder(dataset: str, moment: Moment, design: int | None, artifact_id: str, root: Path | None = None) -> Path:
    base = home(root) / dataset
    if moment == "pre":
        return base / "viz" / "pre" / artifact_id
    if design is None:
        raise ValueError("a post artifact needs the design run it belongs to")
    return base / "designs" / str(design) / "viz" / artifact_id


def new_id() -> str:
    return secrets.token_hex(4)


def save(a: Artifact, root: Path | None = None) -> Path:
    d = folder(a.dataset, a.moment, a.design, a.id, root)
    d.mkdir(parents=True, exist_ok=True)
    (d / FILE).write_text(a.model_dump_json(indent=2))
    return d


def load(dataset: str, moment: Moment, design: int | None, artifact_id: str, root: Path | None = None) -> Artifact:
    return Artifact.model_validate_json((folder(dataset, moment, design, artifact_id, root) / FILE).read_text())


def list_artifacts(dataset: str, moment: Moment, design: int | None = None, root: Path | None = None) -> list[Artifact]:
    """Every artifact of the dataset at that moment (and design run, for post), oldest first."""
    base = home(root) / dataset
    d = base / "viz" / "pre" if moment == "pre" else base / "designs" / str(design) / "viz"
    if not d.is_dir():
        return []
    found = [Artifact.model_validate_json((p / FILE).read_text()) for p in d.iterdir() if (p / FILE).is_file()]
    return sorted(found, key=lambda a: a.made_at)
