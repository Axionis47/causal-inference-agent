"""Adjustment: the effect of a change when what drove the change is measured. Runs on DoWhy."""

from __future__ import annotations

from pathlib import Path

from causal_agent.families.adjustment import design, handoff, previz, probes
from causal_agent.families.base import FamilyDef, load_family_yaml

_KNOWLEDGE, _NEEDS = load_family_yaml(Path(__file__).parent / "family.yaml")


def _lane():
    from causal_agent.families.adjustment.lane.graph import compile_subgraph

    return compile_subgraph()


FAMILY = FamilyDef(
    name="adjustment",
    knowledge=_KNOWLEDGE,
    needs=_NEEDS,
    design_cls=design.AdjustmentDesign,
    design_block=handoff.design_block,
    probes=probes.probes,
    previz=previz.FIGURES,
    lane=_lane,
)
