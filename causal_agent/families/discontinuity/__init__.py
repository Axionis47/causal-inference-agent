"""Discontinuity: the effect of a change assigned by a strict cutoff on one score, from rows just either side of it.
Runs on rdrobust."""

from __future__ import annotations

from pathlib import Path

from causal_agent.families.base import FamilyDef, load_family_yaml
from causal_agent.families.discontinuity import design, handoff, previz, probes

_KNOWLEDGE, _NEEDS = load_family_yaml(Path(__file__).parent / "family.yaml")


def _lane():
    from causal_agent.families.discontinuity.lane.graph import compile_subgraph

    return compile_subgraph()


FAMILY = FamilyDef(
    name="discontinuity",
    knowledge=_KNOWLEDGE,
    needs=_NEEDS,
    design_cls=design.RdDesign,
    design_block=handoff.design_block,
    probes=probes.probes,
    previz=previz.FIGURES,
    lane=_lane,
)
