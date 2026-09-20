"""Diff-in-diff: the effect of a change that reached some units and not others, from how outcomes moved in each group.
Runs on pyfixest."""

from __future__ import annotations

from pathlib import Path

from causal_agent.families.base import FamilyDef, load_family_yaml
from causal_agent.families.diff_in_diff import design, handoff, previz, probes

_KNOWLEDGE, _NEEDS = load_family_yaml(Path(__file__).parent / "family.yaml")


def _lane():
    from causal_agent.families.diff_in_diff.lane.graph import compile_subgraph

    return compile_subgraph()


FAMILY = FamilyDef(
    name="diff_in_diff",
    knowledge=_KNOWLEDGE,
    needs=_NEEDS,
    design_cls=design.DidDesign,
    design_block=handoff.design_block,
    probes=probes.probes,
    previz=previz.FIGURES,
    lane=_lane,
)
