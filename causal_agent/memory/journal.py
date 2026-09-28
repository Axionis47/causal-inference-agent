"""The analysis journal: one append-only record per conversation of the steps it took, each with an address, what it read,
and what it left on disk. It records what was done and seen; it never writes what is known.

data/memory/<name>/analyses/<id>/journal.jsonl     one Step per line, in order
data/memory/<name>/analyses/<id>/explore/<k>/       what an exploration before any design drew (reserved)

A step's design is the design run it belongs to: a step after a run belongs to the last design written; a step before
a run leads to the next design, which is not known yet, so it is null and a reader attaches it to the design step that
follows. The address of a step is step:<n>."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.memory import store

StepKind = Literal["question", "claim", "fit", "explain", "design", "run", "brief", "answer", "what_if", "revise", "requestion", "explore"]
By = Literal["code", "model", "person"]
_MAX_NOTE = 300


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class Step(BaseModel):
    n: int
    kind: StepKind
    by: By
    at: str
    memory_version: int
    design: int | None = None
    read: list[str] = Field(default_factory=list, description="addresses it used: user:turn:<n>, col:, claim:, step:<n>, figure:, probe:, family names")
    left: list[str] = Field(default_factory=list, description="paths it wrote, relative to the dataset's memory dir when inside it, else as recorded")
    note: str = ""

    @property
    def address(self) -> str:
        return f"step:{self.n}"

    def line(self) -> str:
        """The step as the after-run material and the explain prompt read it."""
        parts = [f"{self.kind} by {self.by}", f"v{self.memory_version}"]
        if self.design is not None:
            parts.append(f"design {self.design}")
        if self.read:
            parts.append("read " + ", ".join(self.read))
        if self.note:
            parts.append(self.note)
        if self.left:
            parts.append("left " + ", ".join(self.left))
        return " · ".join(parts)


class Journal:
    """One conversation's record. Reading tolerates a partial last line; appending starts on a fresh line after one."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def home(self) -> Path:
        """The dataset's memory dir: analyses/<id>/journal.jsonl sits two levels under it."""
        return self.path.parents[2]

    def steps(self) -> list[Step]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(Step.model_validate_json(line))
            except ValueError:
                continue
        return out

    def last(self, kind: StepKind | None = None) -> Step | None:
        for s in reversed(self.steps()):
            if kind is None or s.kind == kind:
                return s
        return None

    def resolve(self, address: str) -> Step | None:
        a = address.strip()
        n = a[len("step:") :] if a.startswith("step:") else a
        if not n.isdigit():
            return None
        want = int(n)
        return next((s for s in self.steps() if s.n == want), None)

    def addresses(self) -> set[str]:
        return {s.address for s in self.steps()}

    def rel(self, path: str | Path) -> str:
        """A path relative to the dataset's memory dir when it lies inside it, else as given."""
        p = Path(path)
        try:
            return str(p.resolve().relative_to(self.home.resolve()))
        except ValueError:
            return str(p)

    def append(
        self,
        kind: StepKind,
        *,
        by: By,
        memory_version: int,
        design: int | None = None,
        read: Iterable[str] = (),
        left: Iterable[str] = (),
        note: str = "",
        at: str | None = None,
    ) -> Step:
        last = self.last()
        step = Step(
            n=(last.n if last else 0) + 1,
            kind=kind,
            by=by,
            at=at or _now(),
            memory_version=memory_version,
            design=design,
            read=list(dict.fromkeys(read)),
            left=[self.rel(p) for p in left],
            note=note.strip()[:_MAX_NOTE],
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lead = ""
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    lead = "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(lead + step.model_dump_json() + "\n")
        return step


# ------------------------------------------------------------------ where a journal lives


def home(name: str, analysis_id: str, root: Path | None = None) -> Path:
    return store.home(name, root) / "analyses" / analysis_id


def open_journal(name: str, analysis_id: str, root: Path | None = None) -> Journal:
    """The journal for one analysis; nothing is created until the first step is appended."""
    return Journal(home(name, analysis_id, root) / "journal.jsonl")


def analysis_ids(name: str, root: Path | None = None) -> list[str]:
    """Every analysis the dataset has had, a1, a2, ..., in order."""
    d = store.home(name, root) / "analyses"
    if not d.is_dir():
        return []
    ids = [p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("a") and p.name[1:].isdigit()]
    return sorted(ids, key=lambda a: int(a[1:]))


def next_analysis_id(name: str, root: Path | None = None) -> str:
    """One past the highest analysis on disk."""
    ids = analysis_ids(name, root)
    return f"a{(int(ids[-1][1:]) if ids else 0) + 1}"
