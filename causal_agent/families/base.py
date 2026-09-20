"""What a family is to the desk: its knowledge, the claims it needs, its design block and how the desk fills it, its probes,
its pre-run figures, and the lane that runs it. A family package builds one FamilyDef; the registry lists them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from causal_agent.common.contracts import Belief, ColumnBrief, Design, Handoff, Probe, Scope
from causal_agent.knowledge import Family
from causal_agent.memory.catalogue import FamilyNeeds
from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.viz.graph import PrevizFigure


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


def load_family_yaml(path: Path) -> tuple[Family, FamilyNeeds | None]:
    """A family's own file: the knowledge the routing reads, and under `needs_claims` the claims it needs for the fit grid."""
    raw = yaml.safe_load(Path(path).read_text())
    name = raw.pop("name", None) or Path(path).parent.name
    needs = raw.pop("needs_claims", None)
    return Family(name=name, **raw), (FamilyNeeds(name=name, **needs) if needs else None)


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
