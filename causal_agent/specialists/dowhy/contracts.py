"""Adjustment-lane contracts. What each DoWhy specialist node writes.

Lane-invariant artifacts (Contrast, Checks, Estimate, Refutation, Interpretation, Feasibility)
live in causal_agent.common.contracts. These are the ones only this lane needs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import networkx as nx
from pydantic import BaseModel, Field

from causal_agent.common.contracts import Checks, Cited, Contrast


class Contrasts(BaseModel):
    items: list[Contrast] = Field(description="one per comparison; for a two-level treatment exactly one")


class Relation(BaseModel):
    """One column's relation to the treatment and the outcome. Four claims, each cited when true."""

    column: str
    affects_treatment: bool = Field(description="the note says this column fed the decision to make the change")
    affects_outcome: bool = Field(description="this column plausibly moves the outcome on its own")
    affected_by_treatment: bool = Field(description="this column's value came after, and could be changed by, the treatment")
    is_outcome_measure: bool = Field(description="this column is another measurement of the same outcome, not a cause")
    reasons: list[Cited] = Field(description="one entry per claim marked true, each citing the card that supports it")


class Edge(BaseModel):
    src: str
    dst: str
    cites: list[str] = Field(default_factory=list)


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

    def parents(self, node: str) -> list[str]:
        return sorted(e.src for e in self.edges if e.dst == node)

    def render(self) -> str:
        lines = [f"treatment {self.treatment} -> outcome {self.outcome}"]
        for n in self.nodes:
            if n in (self.treatment, self.outcome):
                continue
            to_t = any(e.src == n and e.dst == self.treatment for e in self.edges)
            to_y = any(e.src == n and e.dst == self.outcome for e in self.edges)
            from_t = any(e.src == self.treatment and e.dst == n for e in self.edges)
            role = "confounder" if to_t and to_y else "outcome driver" if to_y else "treatment driver" if to_t else "mediator" if from_t else "isolated"
            lines.append(f"  {n}: {role}" + (f"  [{', '.join(sorted({c for e in self.edges if n in (e.src, e.dst) for c in e.cites}))}]"))
        for x in self.excluded:
            lines.append(f"  excluded {x.column}: {x.why}")
        return "\n".join(lines)


class Estimand(BaseModel):
    kind: Literal["backdoor", "none"]
    adjustment_set: list[str] = Field(default_factory=list)
    alternatives: dict[str, list[str]] = Field(default_factory=dict)
    dowhy_text: str = ""


class Revision(BaseModel):
    column: str
    change: Literal["add_to_treatment", "remove_to_treatment", "add_to_outcome", "remove_to_outcome", "exclude"]
    reason: str
    cites: list[str] = Field(default_factory=list)


class DesignAssessment(BaseModel):
    action: Literal["proceed", "revise", "stop"]
    revisions: list[Revision] = Field(default_factory=list, description="only when action is revise; each must touch a flagged column")
    reason: str = Field(description="one or two sentences citing the flag numbers")
    cites: list[str] = Field(default_factory=list, description="check addresses such as check:<contrast>.overlap")


class EstimatorPick(BaseModel):
    name: str = Field(description="one of the names offered")
    reason: str
    cites: list[str] = Field(default_factory=list, description="check addresses and pack addresses that support the pick")


class Design(BaseModel):
    """The frozen design. Everything after this is deterministic execution."""

    contrasts: list[Contrast]
    graph: Graph
    estimand: Estimand
    checks: Checks
    estimator: str
    params: dict = Field(default_factory=dict)
    also_run: str | None = None
    refuters: list[str]
    target_units: str
    frozen_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def render(self) -> str:
        lines = [
            "DESIGN",
            "  contrasts    " + "; ".join(f"{c.treated} vs {c.control} ({c.reason})" for c in self.contrasts),
            "  graph",
        ]
        lines += ["    " + l for l in self.graph.render().splitlines()]
        lines.append(f"  estimand     {self.estimand.kind}; adjust for {', '.join(self.estimand.adjustment_set) or 'nothing'}")
        for r in self.checks.results:
            lines.append(f"  check        {r.level:4} {r.address}  {r.detail}")
        lines.append(f"  estimator    {self.estimator}" + (f" (+ {self.also_run} as secondary)" if self.also_run else "") + f"  params {self.params}")
        lines.append(f"  refuters     {', '.join(self.refuters)}")
        lines.append(f"  target       {self.target_units}")
        lines.append(f"  frozen at    {self.frozen_at}")
        return "\n".join(lines)
