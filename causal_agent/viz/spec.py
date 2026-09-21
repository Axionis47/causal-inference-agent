"""The figure contract. A figure is data with addresses, never an image: the page draws it, the chat cites it, a run
keeps it. The desk asks for a Point to make; the viz tool answers with a Figure, made or not, and says why.

Addresses: figure:<id> for the figure, figure:<id>.<series>.<i> for one drawn value, and every number the figure shows is
also a probe address the desk already has (probe:<family>.<name>), from the same computation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import Probe

Kind = Literal["bars", "lines", "points", "density", "interval", "graph"]
Role = Literal["treatment", "outcome", "confounder", "driver", "mediator", "instrument", "hidden", "excluded", "other"]
Moment = Literal["ready", "run"]


class Series(BaseModel):
    """One drawn set of values. `x` is a level, a period, or a bin; `y` the number; `lo`/`hi` an interval; `n` the rows behind each point."""

    name: str
    x: Sequence[str | float] = Field(default_factory=list)
    y: Sequence[float | None] = Field(default_factory=list)
    lo: Sequence[float | None] | None = None
    hi: Sequence[float | None] | None = None
    n: Sequence[int] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def key(self) -> str:
        """The series' address segment, computed here once so the page never rebuilds it."""
        return _key(self.name)


class Mark(BaseModel):
    """A reference line: the cutoff, the change period, a floor."""

    kind: Literal["vline", "hline"] = "vline"
    at: str | float
    label: str = ""


class Node(BaseModel):
    """One node of a graph figure: a column, or the hidden factor."""

    id: str
    label: str = ""
    role: Role = "other"


class Edge(BaseModel):
    """One arrow of a graph figure, with the addresses it rests on."""

    src: str
    dst: str
    cites: list[str] = Field(default_factory=list)


class FigureSpec(BaseModel):
    id: str
    kind: Kind
    title: str
    x_label: str = ""
    y_label: str = ""
    series: list[Series] = Field(default_factory=list)
    marks: list[Mark] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list, description="for kind graph: the nodes")
    edges: list[Edge] = Field(default_factory=list, description="for kind graph: the arrows")
    moment: Moment = Field(default="run", description="ready: made at the ready moment, before the run; run: made by the run")
    note: str = Field(default="", description="one sentence on what the figure shows, in the question's words")
    draws_on: list[str] = Field(default_factory=list, description="the memory and probe addresses the figure rests on")

    @property
    def address(self) -> str:
        return f"figure:{self.id}"

    def addresses(self) -> set[str]:
        out = {self.address}
        for s in self.series:
            out.update(f"{self.address}.{s.key}.{i}" for i in range(len(s.x)))
        out.update(f"{self.address}.node.{i}" for i in range(len(self.nodes)))
        out.update(f"{self.address}.edge.{i}" for i in range(len(self.edges)))
        return out

    def render(self) -> str:
        """The figure as lines with addresses, for the chat's material."""
        lines = [
            f"[{self.address}] {self.kind}: {self.title}" + (f" — {self.note}" if self.note else "") + (" (before the run)" if self.moment == "ready" else "")
        ]
        for i, n in enumerate(self.nodes):
            lines.append(f"  [{self.address}.node.{i}] {n.label or n.id} ({n.role})")
        for i, e in enumerate(self.edges):
            lines.append(f"  [{self.address}.edge.{i}] {e.src} -> {e.dst}" + (f" [{', '.join(e.cites)}]" if e.cites else ""))
        for s in self.series:
            for i, (x, y) in enumerate(zip(s.x, s.y)):
                tail = ""
                if s.lo is not None and s.hi is not None and s.lo[i] is not None and s.hi[i] is not None:
                    tail += f" [{s.lo[i]:.4g}, {s.hi[i]:.4g}]"
                if s.n is not None:
                    tail += f" (n={s.n[i]})"
                yv = "—" if y is None else f"{y:.4g}"
                lines.append(f"  [{self.address}.{s.key}.{i}] {s.name} · {x}: {yv}{tail}")
        for m in self.marks:
            lines.append(f"  mark: {m.label or m.kind} at {m.at}")
        if self.draws_on:
            lines.append("  draws on: " + ", ".join(f"[{a}]" for a in self.draws_on))
        return "\n".join(lines)


class Point(BaseModel):
    """What the desk wants shown: a claim to make visible, and what it is about. Never a figure name."""

    family: str = Field(description="the family the point serves, by its registry name")
    claim: str = Field(description="the point, in the question's words: 'the two arms overlap on lunch'")
    about: list[str] = Field(default_factory=list, description="memory or probe addresses the point rests on")
    columns: list[str] = Field(default_factory=list, description="columns the point names, if any")


class Figure(BaseModel):
    """The viz tool's answer: a figure with the number it shows, or a refusal with why."""

    made: bool
    spec: FigureSpec | None = None
    probe: Probe | None = Field(default=None, description="the number the figure shows, from the same computation")
    why: str = ""
    function: str = ""

    @classmethod
    def refused(cls, why: str, function: str = "") -> Figure:
        return cls(made=False, why=why, function=function)
