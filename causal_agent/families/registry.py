"""The registry: every family the desk knows, in the order the routing lists them. An explicit list, built once at import;
nothing is scanned. A built family's package contributes its FAMILY; the declared ones come from families/declared."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from causal_agent.families.base import BlockInputs, Family, FamilyDef, stub_lane
from causal_agent.memory.catalogue import FamilyNeeds
from causal_agent.viz.graph import register_figures

__all__ = ["REGISTRY", "BlockInputs", "FamilyDef", "family", "knowledge", "lanes", "needs", "stub_lane"]


def _build() -> dict[str, FamilyDef]:
    from causal_agent.families import adjustment, declared, diff_in_diff, discontinuity

    out: dict[str, FamilyDef] = {}
    for f in [adjustment.FAMILY, diff_in_diff.FAMILY, discontinuity.FAMILY, *declared.FAMILIES]:
        out[f.name] = f
        if f.previz:
            register_figures(f.name, f.previz)
    return out


REGISTRY: dict[str, FamilyDef] = _build()


def family(name: str) -> FamilyDef:
    return REGISTRY[name]


def needs() -> dict[str, FamilyNeeds]:
    """The claims each family needs, for the fit grid and the open questions; a family with none listed is not in the grid."""
    return {n: f.needs for n, f in REGISTRY.items() if f.needs is not None}


def knowledge() -> list[Family]:
    """The prose the routing judges the data against, in registry order."""
    return [f.knowledge for f in REGISTRY.values()]


@lru_cache(maxsize=1)
def lanes() -> dict[str, Any]:
    """Every family's compiled lane, keyed by family name; a declared family gets its stub."""
    return {n: f.lane() for n, f in REGISTRY.items() if f.lane is not None}
