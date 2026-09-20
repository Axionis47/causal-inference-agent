"""What a family is to the desk: its knowledge, the claims it needs, its design block and how the desk fills it, its probes,
its pre-run figures, and the lane that runs it. REGISTRY is an explicit list, built once at import; nothing is scanned.

A declared family has knowledge, needs and perhaps probes, and a stub lane that says it is not supported yet."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from causal_agent.common.contracts import Belief, ColumnBrief, Design, Handoff, Probe, Scope
from causal_agent.knowledge import Family, load_registry
from causal_agent.memory.catalogue import FamilyNeeds, load_catalogue
from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.viz.graph import PrevizFigure, register_figures


@dataclass(frozen=True)
class BlockInputs:
    """What the desk hands a family to fill its design block: the briefs, the settled claims, the beliefs, the probes."""

    briefs: list[ColumnBrief]
    claims: dict[str, dict[str, Any]]  # assignment, change, grain, sampling, missing
    beliefs: dict[str, Belief]
    probes: list[Probe]
    entry: dict[str, Any]  # the dataset's index entry
    scope: Scope
    treatment: str | None


ProbeFn = Callable[[pd.DataFrame, ClaimTable, dict], list[ProbeResult]]
LaneFactory = Callable[[], Any]


@dataclass(frozen=True)
class FamilyDef:
    name: str
    knowledge: Family
    needs: FamilyNeeds | None = None  # None: the family takes part in no fit grid (declared, with no claims listed)
    design_cls: type[Design] | None = None
    design_block: Callable[[BlockInputs], Design] | None = None
    probes: ProbeFn | None = None
    previz: list[PrevizFigure] = field(default_factory=list)
    lane: LaneFactory | None = None

    @property
    def specialist(self) -> str:
        return self.knowledge.specialist

    @property
    def built(self) -> bool:
        return self.knowledge.status == "built"


# ------------------------------------------------------------------ the stub lane for a declared family


class StubState(TypedDict, total=False):
    handoff: Handoff | None
    dataset: str
    specialist_result: dict | None


def stub_lane(family: str, specialist: str, built: bool):
    """A lane that reports 'not supported yet' with the hand-off preserved."""

    def run(state: StubState) -> dict:
        h = state.get("handoff")
        if h is None:
            return {"specialist_result": {"status": "no_handoff", "family": family}}
        summary = {
            "family": h.family,
            "specialist": h.specialist,
            "outcome": h.outcome,
            "treatment": h.treatment,
            "relevant_columns": [c.column for c in h.relevant_columns],
            "chosen_assumption": h.chosen_assumption,
        }
        if built:
            return {"specialist_result": {"status": "stub", "message": f"{specialist} specialist not implemented yet; would run on this hand-off", **summary}}
        return {"specialist_result": {"status": "not_supported", "message": f"family {family} has no specialist yet ({specialist})", **summary}}

    g = StateGraph(StubState)
    g.add_node("run", run)
    g.add_edge(START, "run")
    g.add_edge("run", END)
    return g.compile(checkpointer=False)


# ------------------------------------------------------------------ the built lanes, compiled on first use


def _adjustment_lane():
    from causal_agent.specialists.dowhy.graph import compile_subgraph

    return compile_subgraph()


def _diff_in_diff_lane():
    from causal_agent.specialists.did.graph import compile_subgraph

    return compile_subgraph()


def _discontinuity_lane():
    from causal_agent.specialists.rd.graph import compile_subgraph

    return compile_subgraph()


# ------------------------------------------------------------------ the registry


def _build() -> dict[str, FamilyDef]:
    from causal_agent.common.contracts import AdjustmentDesign, DidDesign, RdDesign
    from causal_agent.families import blocks, previz, probes

    prose = {f.name: f for f in load_registry()}
    needs = load_catalogue().families
    built: dict[str, dict[str, Any]] = {
        "adjustment": dict(
            design_cls=AdjustmentDesign, design_block=blocks.adjustment, probes=probes.adjustment, previz=previz.ADJUSTMENT, lane=_adjustment_lane
        ),
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
    out: dict[str, FamilyDef] = {}
    for name, fam in prose.items():
        extra = dict(built.get(name, {}))
        if "lane" not in extra:
            extra["lane"] = lambda n=name, s=fam.specialist, b=fam.status == "built": stub_lane(n, s, b)
        out[name] = FamilyDef(name=name, knowledge=fam, needs=needs.get(name), **extra)
    for fam_def in out.values():
        if fam_def.previz:
            register_figures(fam_def.name, fam_def.previz)
    return out


REGISTRY: dict[str, FamilyDef] = _build()


def family(name: str) -> FamilyDef:
    return REGISTRY[name]


def needs() -> dict[str, FamilyNeeds]:
    """The claims each family needs, for the fit grid and the open questions; a family with none listed is not in the grid."""
    return {n: f.needs for n, f in REGISTRY.items() if f.needs is not None}


def knowledge(path: str | None = None) -> list[Family]:
    """The prose the routing judges the data against; a path names another registry file for tests."""
    return load_registry(path) if path else [f.knowledge for f in REGISTRY.values()]


@lru_cache(maxsize=1)
def lanes() -> dict[str, Any]:
    """Every family's compiled lane, keyed by family name; a declared family gets its stub."""
    return {n: f.lane() for n, f in REGISTRY.items() if f.lane is not None}
