"""Adjustment reasoning certificates and immutable execution designs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from causal_agent.common.contracts import Checks, Contrast, FrozenDesign
from causal_agent.families.adjustment.lane.knowledge import EstimatorEntry, FalsificationEntry, SensitivityEntry
from causal_agent.lane.ladder import LadderBase


class Contrasts(BaseModel):
    items: list[Contrast]


class Decision(BaseModel):
    reason: str
    cites: list[str] = Field(default_factory=list)
    values: dict = Field(default_factory=dict)

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:{k}", str(v)) for k, v in self.values.items()]


class Ladder(LadderBase):
    ORDER: ClassVar[tuple[str, ...]] = ("pair", "mechanism", "relationships", "probe", "rivals", "checks", "picks")
    pair: Decision | None = None
    mechanism: Decision | None = None
    relationships: Decision | None = None
    probe: Decision | None = None
    rivals: Decision | None = None
    checks: Decision | None = None
    picks: Decision | None = None


class Edge(BaseModel):
    src: str
    dst: str
    cites: list[str] = Field(default_factory=list)
    status: str = "proposed"


class Excluded(BaseModel):
    column: str
    why: str


class Graph(BaseModel):
    treatment: str
    outcome: str
    nodes: list[str]
    edges: list[Edge]
    excluded: list[Excluded] = Field(default_factory=list)

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from((e.src, e.dst) for e in self.edges)
        return g

    def render(self) -> str:
        return "\n".join(
            [f"treatment {self.treatment}, outcome {self.outcome}"] + [f"{e.src} -> {e.dst} ({e.status}) [{', '.join(e.cites)}]" for e in self.edges]
        )


class Estimand(BaseModel):
    kind: str
    target: str = "ate"
    identification_method: str = "minimal-adjustment"
    adjustment_set: list[str] = Field(default_factory=list)
    model_covariates: list[str] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    frontdoor_set: list[str] = Field(default_factory=list)
    mediator: str | None = None
    first_stage_set: list[str] = Field(default_factory=list)
    second_stage_set: list[str] = Field(default_factory=list)
    total_set: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    symbolic: str = ""
    structural_zero: bool = False


class Candidate(BaseModel):
    id: str
    graph_id: str
    endpoint_id: str
    graph: Graph
    estimand: Estimand
    target: dict
    assumptions: list[str] = Field(default_factory=list)
    cites: list[str] = Field(default_factory=list)
    status: Literal["identified", "dropped", "frozen", "diagnostic"] = "identified"
    reason: str = ""
    checks: Checks = Field(default_factory=Checks)
    allowed_estimators: list[str] = Field(default_factory=list)
    chosen_estimator: str | None = None


class EstimatorChoice(BaseModel):
    design_id: str
    name: str = Field(
        description="Plain offered estimator name, for example propensity_score_weighting. Never put estimator: or an address prefix in name; those belong only in cites."
    )
    reason: str
    cites: list[str] = Field(
        description="Only displayed address strings beginning design:, check:, estimator: or rel:. Never append plain column names such as c or x."
    )


class ComparisonInterpretation(BaseModel):
    answer: str
    caveats: list[str]
    cites: list[str] = Field(
        description="Raw displayed evidence addresses, such as claim:story.text or col:c.when. Never quote sentences or include square brackets."
    )
    comparisons: list[str] = Field(default_factory=list, description="descriptive comparisons only between compatible targets; no universal agreement verdict")


class Design(FrozenDesign):
    """No mutable reasoning or catalogue lookup is needed once this record exists."""

    model_config = ConfigDict(frozen=True)
    id: str
    endpoint_id: str
    context_id: str
    ledger_revision: int
    target: dict
    identification_certificate: dict = Field(default_factory=dict)
    support: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    inference: dict = Field(default_factory=dict)
    weight_policy: dict = Field(default_factory=dict)
    contrasts: list[Contrast]
    graph: Graph
    estimand: Estimand
    checks: Checks
    estimator: EstimatorEntry
    fallback: EstimatorEntry | None = None
    falsifications: list[FalsificationEntry] = Field(default_factory=list)
    sensitivities: list[SensitivityEntry] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    modifier_groups: dict[str, list[dict]] = Field(default_factory=dict)
    bootstrap_draws: int = 100
    bootstrap_seed: int = 7
    target_units: str = "ate"
    frozen_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def render(self) -> str:
        return "\n".join(
            [
                f"DESIGN {self.id}: {self.endpoint_id}",
                f"context {self.context_id}; ledger revision {self.ledger_revision}",
                f"target {self.target}",
                self.graph.render(),
                f"identification {self.estimand.model_dump()}",
                f"estimator {self.estimator.name}; fallback {self.fallback.name if self.fallback else 'none'}",
            ]
        )


class Mechanism(BaseModel):
    drivers: list[str]
    offer_column: str | None = None
    uptake_column: str | None = None
    self_selection: bool | None = None
    reason: str
    cites: list[str] = Field(
        description="Raw displayed evidence addresses, such as claim:story.text or col:c.when. Never quote sentences or include square brackets."
    )


class RelationshipProposal(BaseModel):
    kind: Literal["edge", "no_edge", "hidden", "when", "same_as", "nested_in", "modifies", "outcome_measure"]
    src: str = Field(description="Raw column key, for example c or t, never col:c. A hidden factor uses hidden:name.")
    dst: str = Field(
        description='Required string. For edge/no_edge supply the raw destination key, e.g. y for t to y or m for t to m. For unary attributes (when, hidden, modifies, outcome_measure, same_as, nested_in), supply an empty string: dst="". Never omit this property or return null.'
    )
    value: str | bool | list[str] | None = Field(
        default=None,
        description="when: before/at/after/unknown; hidden: list of affected raw column keys; outcome_measure/modifies: boolean; edge/no_edge: null.",
    )
    status: Literal["proposed", "supported", "contested"]
    reason: str
    cites: list[str] = Field(description="Raw displayed source addresses, not quoted sentences. Code derives source/category and ledger IDs.")


class Relationships(BaseModel):
    entries: list[RelationshipProposal] = Field(description="Only new or changed hypotheses. Omit unchanged confirmed ledger entries.")
    considered: list[str] = Field(description="every non-pair column examined exactly once; never include treatment or outcome")
    reason: str
    cites: list[str] = Field(description="Raw displayed evidence addresses, such as claim:story.text or col:c.when.")
