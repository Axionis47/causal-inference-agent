"""The registry: every family the desk knows, in the order the routing lists them. An explicit list, built once at import;
nothing is scanned. A family that lives in its own package contributes its FAMILY; the rest are still read from the old
places until each moves."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from causal_agent.families.base import BlockInputs, FamilyDef, stub_lane
from causal_agent.knowledge import Family, load_registry
from causal_agent.memory.catalogue import FamilyNeeds, load_catalogue
from causal_agent.viz.graph import register_figures

__all__ = ["REGISTRY", "BlockInputs", "FamilyDef", "family", "knowledge", "lanes", "needs", "stub_lane"]


def _diff_in_diff_lane():
    from causal_agent.specialists.did.graph import compile_subgraph

    return compile_subgraph()


def _discontinuity_lane():
    from causal_agent.specialists.rd.graph import compile_subgraph

    return compile_subgraph()


def _legacy() -> list[FamilyDef]:
    """The families not yet in their own package: knowledge from knowledge/families.yaml, needs from memory/fields.yaml."""
    from causal_agent.common.contracts import DidDesign, RdDesign
    from causal_agent.families import blocks, previz, probes

    needs = load_catalogue().families
    built: dict[str, dict[str, Any]] = {
        "diff_in_diff": dict(
            design_cls=DidDesign, design_block=blocks.diff_in_diff, probes=probes.diff_in_diff, previz=previz.DIFF_IN_DIFF, lane=_diff_in_diff_lane
        ),
        "discontinuity": dict(
            design_cls=RdDesign, design_block=blocks.discontinuity, probes=probes.discontinuity, previz=previz.DISCONTINUITY, lane=_discontinuity_lane
        ),
        "synthetic_control": dict(probes=probes.synthetic_control),
        "interrupted_series": dict(probes=probes.interrupted_series),
        "instrument": dict(probes=probes.instrument),
    }
    out = []
    for fam in load_registry():
        extra = dict(built.get(fam.name, {}))
        if "lane" not in extra:
            extra["lane"] = lambda n=fam.name, s=fam.specialist, b=fam.status == "built": stub_lane(n, s, b)
        out.append(FamilyDef(name=fam.name, knowledge=fam, needs=needs.get(fam.name), **extra))
    return out


def _build() -> dict[str, FamilyDef]:
    from causal_agent.families import adjustment

    out: dict[str, FamilyDef] = {}
    for f in [adjustment.FAMILY, *_legacy()]:
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
