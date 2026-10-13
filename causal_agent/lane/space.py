"""A lane's design space, declared once: the characteristics of a design with the values the library runs, which
rung decides each, the fact it is read from, and the shapes the library does not run with the reason a stop carries. The
dimension names are the catalogue's own `applies_when` keys, so the estimator and refuter entries are the allowed tuples.

Four pure functions read it. `cells` enumerates the space with what serves each cell; `coverage` proves, before any run, that
every cell is served by an estimator and a refuter or is declared unsupported. `bounds` projects one file onto the space at load, so a file that can reach no cell stops before
any model call. `offer` gives a deciding rung only the values still runnable given what the rungs below fixed. `cell` names
the shape a design reached, or says where it left the space.
"""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from causal_agent.lane.knowledge import matches


class Dimension(BaseModel):
    """One characteristic: a list of values the library runs, or a range whose bound may be a number, `inf`, or `<fact>/<n>`."""

    model_config = ConfigDict(extra="forbid")
    values: list[Any] | None = None
    range: list[Any] | None = None
    decided_by: str
    read_from: str | None = None
    soft: bool = False  # offered, never a gap: a value no entry serves is a decline at the freeze, not a wall
    unsupported: dict[str, str] = Field(default_factory=dict)  # value, or `above`/`below` for a range -> the reason

    def discrete(self) -> bool:
        return self.values is not None


class Unsupported(BaseModel):
    """A combination the library does not run, and the reason."""

    model_config = ConfigDict(extra="forbid")
    when: dict[str, Any]
    reason: str


class Space(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimensions: dict[str, Dimension]
    unsupported: list[Unsupported] = Field(default_factory=list)

    def reason_for(self, point: dict[str, Any]) -> str | None:
        """The declared reason a point is not run: a dimension's own value, or a combination; None when nothing declares it."""
        for name, v in point.items():
            d = self.dimensions.get(name)
            if d is not None and str(v) in d.unsupported:
                return d.unsupported[str(v)]
        for u in self.unsupported:
            if all(point.get(k) == v for k, v in u.when.items()):
                return u.reason
        return None


class Outside(BaseModel):
    """Where a file or a design left the space, with the reason the declaration gives."""

    dimension: str
    value: Any
    reason: str

    def render(self) -> str:
        return f"{self.dimension} = {self.value!r}: {self.reason}"


class Gap(BaseModel):
    """A cell of the space nothing serves and nothing declares unsupported."""

    cell: dict[str, Any]
    missing: str  # estimator | refuter


class Covered(BaseModel):
    cells: int
    unsupported: int


class Cell(BaseModel):
    """One point of the keyed dimensions: the entries whose rule applies to it, or the declared reason it is not run."""

    point: dict[str, Any]
    estimators: list[str] = Field(default_factory=list)
    refuters: list[str] = Field(default_factory=list)
    unsupported: str | None = None


def load_space(path: Path) -> Space:
    return Space.model_validate(yaml.safe_load(Path(path).read_text()) or {})


# ------------------------------------------------------------------ the bounds of a range, read from facts


def _bound(raw: Any, facts: dict[str, Any]) -> float:
    if isinstance(raw, int | float):
        return float(raw)
    s = str(raw).strip()
    if s == "inf":
        return math.inf
    name, _, div = s.partition("/")
    v = facts.get(name)
    if v is None:
        raise KeyError(f"the bound {s!r} reads the fact {name!r}, which was not given")
    return float(v) / float(div) if div else float(v)


def _outside_range(name: str, d: Dimension, v: Any, facts: dict[str, Any]) -> Outside | None:
    assert d.range is not None
    lo, hi = _bound(d.range[0], facts), _bound(d.range[1], facts)
    if float(v) < lo:
        return Outside(dimension=name, value=v, reason=d.unsupported.get("below", f"{name} below {lo:g}"))
    if float(v) > hi:
        return Outside(dimension=name, value=v, reason=d.unsupported.get("above", f"{name} above {hi:g}"))
    return None


# ------------------------------------------------------------------ the four readers


def bounds(space: Space, observed: dict[str, Any]) -> Outside | None:
    """Whether a file's observed characteristics lie inside the space; the first that does not, with its reason. Ranges that
    read a fact take it from `observed`."""
    for name, v in observed.items():
        d = space.dimensions.get(name)
        if d is None:
            continue
        if d.discrete():
            assert d.values is not None
            if v not in d.values:
                return Outside(dimension=name, value=v, reason=space.reason_for({name: v}) or f"{name} = {v!r} is not a shape this lane runs")
        elif d.range is not None:
            out = _outside_range(name, d, v, observed)
            if out is not None:
                return out
    return None


def _keyed(space: Space, entries: list[Any]) -> list[str]:
    """The discrete dimensions the catalogue keys on, in the space's order."""
    named = {k for e in entries for k in e.applies_when}
    return [n for n, d in space.dimensions.items() if d.discrete() and n in named]


def _name(entry: Any) -> str:
    return str(getattr(entry, "name", entry.applies_when))


def cells(space: Space, estimators: list[Any], refuters: list[Any]) -> list[Cell]:
    """Every cell of the keyed dimensions, in the space's order: the entries whose rule applies to it for some value of the soft
    dimensions, or the declared reason it is not run. The end states a design can reach, enumerated from the declaration."""
    entries = list(estimators) + list(refuters)
    hard = [n for n in _keyed(space, entries) if not space.dimensions[n].soft]
    soft = [n for n in _keyed(space, entries) if space.dimensions[n].soft]
    out: list[Cell] = []
    for values in product(*[space.dimensions[n].values or [] for n in hard]):
        point = dict(zip(hard, values, strict=True))
        reason = space.reason_for(point)
        if reason is not None:
            out.append(Cell(point=point, unsupported=reason))
            continue
        variants = [dict(point, **dict(zip(soft, sv, strict=True))) for sv in product(*[space.dimensions[n].values or [] for n in soft])] or [point]
        out.append(
            Cell(
                point=point,
                estimators=[_name(e) for e in estimators if any(matches(e.applies_when, v) for v in variants)],
                refuters=[_name(r) for r in refuters if any(matches(r.applies_when, v) for v in variants)],
            )
        )
    return out


def coverage(space: Space, estimators: list[Any], refuters: list[Any]) -> Covered | Gap:
    """Every cell of the keyed dimensions is served by an estimator and a refuter, or is declared unsupported."""
    all_cells = cells(space, estimators, refuters)
    for c in all_cells:
        if c.unsupported is not None:
            continue
        if not c.estimators:
            return Gap(cell=c.point, missing="estimator")
        if not c.refuters:
            return Gap(cell=c.point, missing="refuter")
    return Covered(cells=len(all_cells), unsupported=sum(1 for c in all_cells if c.unsupported is not None))


def _as_list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list) else [v]


def _consistent(applies_when: dict[str, Any], fixed: dict[str, Any]) -> bool:
    return all(fixed[k] in _as_list(w) for k, w in applies_when.items() if k in fixed and not k.endswith(("_min", "_max")))


def offer(space: Space, entries: list[Any], fixed: dict[str, Any]) -> dict[str, list[Any]]:
    """For each dimension not yet fixed, the values that still lead to a cell some entry serves; a range as its two bounds."""
    out: dict[str, list[Any]] = {}
    live = [e for e in entries if _consistent(e.applies_when, fixed)]
    for name, d in space.dimensions.items():
        if name in fixed:
            continue
        if d.discrete():
            assert d.values is not None
            allowed = {v for e in live for v in _as_list(e.applies_when.get(name, d.values))}
            out[name] = [v for v in d.values if v in allowed and space.reason_for(dict(fixed, **{name: v})) is None]
        elif d.range is not None:
            try:
                out[name] = [_bound(d.range[0], fixed), _bound(d.range[1], fixed)]
            except KeyError:  # a bound that reads a fact not yet fixed: nothing to offer until it is
                continue
    return out


def limit(space: Space, name: str, facts: dict[str, Any] | None = None) -> float:
    """The upper bound of a range dimension, read from `facts` when the declaration names one."""
    d = space.dimensions[name]
    if d.range is None:
        raise KeyError(f"{name} is not a range")
    return _bound(d.range[1], facts or {})


def cell(space: Space, facts: dict[str, Any]) -> dict[str, Any] | Outside:
    """The shape a design reached, as one point of the space; or where it left the space, with the declared reason."""
    out = bounds(space, facts)
    if out is not None:
        return out
    point = {n: facts[n] for n in space.dimensions if n in facts}
    reason = space.reason_for(point)
    if reason is not None:
        return Outside(dimension=", ".join(point), value=point, reason=reason)
    return point
