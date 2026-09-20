"""What a family is to the desk: its knowledge, the claims it needs, its design block and how the desk fills it, its probes,
its pre-run figures, and the lane that runs it. A family package builds one FamilyDef; the registry lists them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from causal_agent.common.contracts import Belief, ColumnBrief, Design, Handoff, Probe, Scope
from causal_agent.memory.catalogue import FamilyNeeds
from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.viz.graph import PrevizFigure


class Family(BaseModel):
    """What the routing knows about a family: written as method knowledge it judges the data against, never as rules."""

    name: str
    applies_to: list[str]
    answers: str
    needs: list[str]
    look_for: str
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    convince: str = Field(default="", description="the point a figure makes at the ready moment, in the question's words")
    specialist: str
    status: Literal["built", "declared"]

    def render(self) -> str:
        needs = "\n".join(f"    - {n}" for n in self.needs)
        return (
            f"family: {self.name}  (status: {self.status})\n"
            f"  applies to questions of kind: {', '.join(self.applies_to)}\n"
            f"  answers: {self.answers}\n"
            f"  needs:\n{needs}\n"
            f"  where the evidence usually lives: {self.look_for}\n"
            f"  assumes: {self.assumes}\n"
            f"  weak when: {self.weak_when}"
        )


def render_preferences(registry: list[Family]) -> str:
    """The families' stated preferences over one another, for the routing prompt."""
    lines = []
    for f in registry:
        for other, why in f.prefer_over.items():
            lines.append(f"- prefer {f.name} over {other}: {why}")
    return "\n".join(lines) or "(none recorded)"


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
    refutation_prefix: str = "placebo"  # how the lane addresses its falsifications: refute:<c>.<name> or placebo:<c>.<name>

    @property
    def specialist(self) -> str:
        return self.knowledge.specialist

    @property
    def built(self) -> bool:
        return self.knowledge.status == "built"


def load_family_yaml(path: Path, name: str | None = None) -> tuple[Family, FamilyNeeds | None]:
    """A family's own file: the knowledge the routing reads, and under `needs_claims` the claims it needs for the fit grid.
    The name is the one given, else the file's `name`, else the package folder's."""
    raw = yaml.safe_load(Path(path).read_text())
    name = name or raw.pop("name", None) or Path(path).parent.name
    raw.pop("name", None)
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
