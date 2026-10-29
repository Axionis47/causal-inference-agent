"""Artifact contracts and the owner-bound publication interface.

The server supplies a conversation repository. PostgreSQL and private object
storage are authoritative; local folders are task scratch and download caches.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from causal_agent.common import config, trace

Origin = Literal["exploration", "analysis"]
Format = Literal["figure", "table", "report", "result"]
FILE = "artifact.json"
_repository: ContextVar = ContextVar("artifact_repository", default=None)


@contextmanager
def using(repository):
    """The integration service supplies an owner-checked repository per conversation."""
    token = _repository.set(repository)
    try:
        yield
    finally:
        _repository.reset(token)


def repository():
    value = _repository.get()
    if value is None:
        raise RuntimeError("Artifact publication requires an owned conversation")
    return value


def supplied_context(dataset: str) -> str | None:
    repo = _repository.get()
    return repo.context if repo is not None and repo.conversation == dataset else None


class ArtifactElement(BaseModel):
    id: str
    label: str
    kind: Literal["fact", "table_column", "chart_series"]
    file: str
    selector: str


class Artifact(BaseModel):
    id: str = Field(description="URL-safe UUID, stored as 32 hexadecimal characters")
    dataset: str
    owner_id: str
    dataset_version_id: str
    origin: Origin
    format: Format
    design: int | None = Field(description="the design run the artifact is about; null when it is about the data before any run")
    memory_version: int
    data_version: int = Field(description="the version of the table it was made from; 0 is the uploaded file")
    ask: str = Field(description="what was asked for, in the person's words")
    summary: str = Field(description="what it shows, in words; a picture's caption")
    method: str
    findings: list[str]
    limitations: list[str]
    tags: list[str]
    source_sha256: str
    parent_refs: list[str]
    conversation_id: str
    request_id: str
    attempt_id: str
    facts: dict[str, float] = Field(description="every number it stands behind, by name")
    files: list[str] = Field(description="the file names in its folder")
    table: str | None = Field(default=None, description="the table file's name in the folder, when it wrote one")
    figure: bool = Field(default=False, description="figure.png is in the folder")
    sandbox: Literal["docker"] | None = None
    made_at: str = Field(description="ISO time")
    elements: list[ArtifactElement] = Field(default_factory=list)
    comparison_basis: dict | None = None

    @property
    def address(self) -> str:
        return f"artifact:{self.id}"

    def addresses(self) -> set[str]:
        out = {self.address} | {f"{self.address}.{name}" for name in self.facts}
        out.add(f"{self.address}.method")
        out.add(f"{self.address}.provenance")
        out.update(f"{self.address}.finding.{i}" for i in range(len(self.findings)))
        out.update(f"{self.address}.limitation.{i}" for i in range(len(self.limitations)))
        if self.table:
            out.add(f"{self.address}.table")
        if self.figure:
            out.add(f"{self.address}.figure")
        return out

    def when(self) -> str:
        return "before any run" if self.design is None else f"design {self.design}"

    def provenance(self) -> str:
        return f"created {self.made_at}; tags: {', '.join(self.tags)}; input version: {self.dataset_version_id}; parents: {', '.join(self.parent_refs)}"

    def render(self) -> str:
        """The artifact as lines with addresses, for the chat's material."""
        lines = [f"[{self.address}] {self.summary} ({self.origin}; {self.format}; {self.when()})"]
        lines.append(f"[{self.address}.provenance] {self.provenance()}")
        lines.append(f"[{self.address}.method] {self.method}")
        lines.extend(f"[{self.address}.finding.{i}] {text}" for i, text in enumerate(self.findings))
        lines.extend(f"[{self.address}.limitation.{i}] {text}" for i, text in enumerate(self.limitations))
        lines.extend(f"  [{self.address}.{name}] {name}: {value:.4g}" for name, value in self.facts.items())
        if self.table:
            lines.append(f"  [{self.address}.table] a table: {self.table}")
        if self.figure:
            lines.append(f"  [{self.address}.figure] a picture")
        return "\n".join(lines)


def home(root: Path | None = None) -> Path:
    """Local scratch/cache root, never a catalogue."""
    return config.load(root=root).paths.runs / "explorations"


def folder(dataset: str, artifact_id: str, root: Path | None = None) -> Path:
    return home(root) / dataset / artifact_id


def new_id() -> str:
    return uuid4().hex


def save(a: Artifact, root: Path | None = None) -> Path:
    d = folder(a.dataset, a.id, root)
    d.mkdir(parents=True, exist_ok=True)
    trace.write(d / FILE, a.model_dump_json(indent=2))
    repository().publish(a, d)
    return d


def load(dataset: str, artifact_id: str, root: Path | None = None) -> Artifact:
    a = repository().load(dataset, artifact_id)
    repository().materialize(a, folder(dataset, artifact_id, root))
    return a


def list_artifacts(dataset: str, origin: Origin | None = None, design: int | None = None, root: Path | None = None) -> list[Artifact]:
    """Published artifacts in the bound conversation, oldest first."""
    if _repository.get() is None:
        return []
    found = repository().list(dataset)
    if origin is not None:
        found = [a for a in found if a.origin == origin]
    if design is not None:
        found = [a for a in found if a.design == design]
    return sorted(found, key=lambda a: a.made_at)


def about(dataset: str, design: int | None, root: Path | None = None) -> list[Artifact]:
    """What the chat may cite at this point: every artifact about the data, plus the ones about the given run."""
    return [a for a in list_artifacts(dataset, root=root) if a.design is None or a.design == design]
