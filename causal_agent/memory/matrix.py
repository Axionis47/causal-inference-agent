"""The matrix: every family against every claim kind, kept as a record with an update rule. A cell carries its value, the
address that decided it, and the memory version it was set at; a cell that did not change keeps its version. The chat cites
a cell as matrix:<family>.<kind>. The cell logic lives in `table.compute`; this module only remembers and compares."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.memory import ops
from causal_agent.memory.catalogue import Catalogue, FamilyNeeds
from causal_agent.memory.claims import ProbeResult
from causal_agent.memory.records import Memory

Value = Literal["not_needed", "unknown", "does_not_fit", "fits"]


class Cell(BaseModel):
    value: Value
    set_by: str | None = Field(default=None, description="the address whose value decided the cell: a fits field, an empty or refuted claim")
    at: str | None = Field(default=None, description="the memory version the cell took this value at, as v<n>")


class CellChange(BaseModel):
    family: str
    kind: str
    before: Value | None
    after: Value | None
    set_by: str | None = None

    def line(self) -> str:
        return f"{self.family}.{self.kind}: {self.before or 'none'} -> {self.after or 'none'}" + (f" ({self.set_by})" if self.set_by else "")


class Matrix(BaseModel):
    memory_version: int = -1
    cells: dict[str, dict[str, Cell]] = Field(default_factory=dict, description="family -> kind -> cell")
    struck: dict[str, str] = Field(default_factory=dict, description="family -> why it is out: a cell that does not fit, or a failed probe")
    surviving: list[str] = Field(default_factory=list)
    ready: bool = False

    def cell(self, family: str, kind: str) -> Cell | None:
        return self.cells.get(family, {}).get(kind)

    def update(
        self,
        memory: Memory,
        probes: list[ProbeResult],
        needs: Mapping[str, FamilyNeeds],
        columns: list[str] | None = None,
        cat: Catalogue | None = None,
    ) -> Matrix:
        """The matrix as the memory now stands. A cell whose value did not change keeps the version it was set at."""
        status = ops.fit(memory, probes, needs, columns=columns, cat=cat)
        at = f"v{memory.version}"
        cells: dict[str, dict[str, Cell]] = {}
        for family, row in status.table.items():
            cells[family] = {}
            for kind, value in row.items():
                old = self.cell(family, kind)
                set_by = status.set_by.get(family, {}).get(kind)
                cells[family][kind] = Cell(value=value, set_by=set_by, at=old.at if old is not None and old.value == value else at)
        return Matrix(memory_version=memory.version, cells=cells, struck=dict(status.struck), surviving=list(status.surviving), ready=status.ready)

    def diff(self, prev: Matrix) -> list[CellChange]:
        """The cells whose value differs from `prev`. A cell absent on one side and not needed on the other did not change."""
        out: list[CellChange] = []
        families = list(dict.fromkeys(list(prev.cells) + list(self.cells)))
        for family in families:
            kinds = list(dict.fromkeys(list(prev.cells.get(family, {})) + list(self.cells.get(family, {}))))
            for kind in kinds:
                a, b = prev.cell(family, kind), self.cell(family, kind)
                before, after = (a.value if a else None), (b.value if b else None)
                if before == after or {before, after} <= {None, "not_needed"}:
                    continue
                out.append(CellChange(family=family, kind=kind, before=before, after=after, set_by=b.set_by if b else None))
        return out

    def render(self) -> list[str]:
        """One line per cell in play, each with its address, then one per struck family, then the verdict."""
        lines: list[str] = []
        for family, row in self.cells.items():
            for kind, c in row.items():
                if c.value == "not_needed":
                    continue
                lines.append(f"[matrix:{family}.{kind}] {c.value}" + (f" · set by {c.set_by}" if c.set_by else "") + (f" · {c.at}" if c.at else ""))
        for family, why in self.struck.items():
            lines.append(f"[matrix:{family}.struck] {why}")
        lines.append(f"[matrix] {'ready' if self.ready else 'not ready'} · in play: {', '.join(self.surviving) or 'none'} · v{self.memory_version}")
        return lines
