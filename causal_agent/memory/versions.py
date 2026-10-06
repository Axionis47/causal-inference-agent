"""The versions shelf: every table the data agent wrote, by what made it. Version 0 is the uploaded file and has no folder; a
version v from 1 up lives under the dataset's memory with the table, the script that made it, and a record naming its parent,
what was asked, what changed and the memory version it was made at. Never in place: a new version is written beside its
parent, and switching the version in play is the person's word, a step in the journal and a move of the memory's version.

data/memory/<name>/versions/<v>/        table.csv, code.py, version.json, thoughts.json?

The address of a version is version:<v>; its rows are the parent's rows by identity (a row kept keeps its place, a row added
is refused), so a design on version v and one on version w compare like for like."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from causal_agent.common import config, trace

FILE = "version.json"
TABLE = "table.csv"


class Version(BaseModel):
    v: int = Field(description="1 and up; 0 is the uploaded file and is never written")
    dataset: str
    parent: int = Field(description="the version this one was made from")
    ask: str = Field(description="what was asked, in the person's words")
    summary: str = Field(description="what changed, in words")
    rows: int
    columns: list[str] = Field(description="the file's own column names, in order")
    added: list[str] = Field(default_factory=list, description="columns not in the parent")
    dropped: list[str] = Field(default_factory=list, description="the parent's columns not here")
    memory_version: int
    sandbox: str = "subprocess"
    made_at: str

    @property
    def address(self) -> str:
        return f"version:{self.v}"

    def render(self) -> str:
        """The version as a line with its address, for the chat's material."""
        diff = "; ".join(p for p in (f"added {', '.join(self.added)}" if self.added else "", f"dropped {', '.join(self.dropped)}" if self.dropped else "") if p)
        return f"[{self.address}] table version {self.v} from {self.parent}: {self.summary} ({self.rows} rows{'; ' + diff if diff else ''})"


def home(root: Path | None = None) -> Path:
    return config.load(root=root).paths.memory


def folder(dataset: str, v: int, root: Path | None = None) -> Path:
    return home(root) / dataset / "versions" / str(v)


def table_path(dataset: str, v: int, root: Path | None = None) -> Path | None:
    """Where version v's table is; None for version 0, whose table is the dataset's own file."""
    return None if v == 0 else folder(dataset, v, root) / TABLE


def next_version(dataset: str, root: Path | None = None) -> int:
    d = home(root) / dataset / "versions"
    return max((int(p.name) for p in d.iterdir() if p.name.isdigit()), default=0) + 1 if d.is_dir() else 1


def save(version: Version, root: Path | None = None) -> Path:
    d = folder(version.dataset, version.v, root)
    d.mkdir(parents=True, exist_ok=True)
    trace.write(d / FILE, version.model_dump_json(indent=2))
    return d


def load(dataset: str, v: int, root: Path | None = None) -> Version:
    return Version.model_validate_json((folder(dataset, v, root) / FILE).read_text())


def list_versions(dataset: str, root: Path | None = None) -> list[Version]:
    """The dataset's written versions, oldest first; version 0 is not among them."""
    d = home(root) / dataset / "versions"
    if not d.is_dir():
        return []
    return sorted((Version.model_validate_json((p / FILE).read_text()) for p in d.iterdir() if (p / FILE).is_file()), key=lambda x: x.v)
