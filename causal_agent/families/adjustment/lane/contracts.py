"""Adjustment-lane contracts. What each DoWhy specialist node writes, and the ladder the design is climbed on.

Lane-invariant artifacts (Contrast, Checks, Estimate, Refutation, Interpretation, Feasibility)
live in causal_agent.common.contracts. These are the ones only this lane needs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Checks, Cited, Contrast, Departure


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
    departures: list[Departure] = Field(
        default_factory=list, description="one entry per claim where you departed from THE LAST READING, naming the claim and citing what changed it"
    )


# ------------------------------------------------------------------ the ladder: one record per rung, each line addressed


class Unsure(BaseModel):
    """One thing a rung would not guess: what it is about (an address when there is one) and why. A flag, never a question."""

    about: str = Field(description="the address of the claim or item you could not settle, such as col:lunch.may_modify, or a short name")
    reason: str


class Pair(BaseModel):
    """Rung 0: the outcome, the treatment, which levels are compared, and for whom the effect is wanted."""

    outcome: str
    treatment: str
    contrasts: list[Contrast]
    target_asked: str
    by: Literal["pack", "judgement"] = "pack"

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("ladder:pair.outcome", self.outcome),
            ("ladder:pair.treatment", self.treatment),
            ("ladder:pair.contrasts", "; ".join(f"{c.treated!r} versus {c.control!r}" for c in self.contrasts) + f" (set by the {self.by})"),
            ("ladder:pair.target", self.target_asked),
        ]


class Mechanism(BaseModel):
    """Rung 1: what set the treatment. Code fills the kind from the pack; a judgement fills the rest only when the pack leaves it open."""

    kind: str | None = None
    drivers: list[str] = Field(default_factory=list, description="the columns the decision or the offer looked at")
    offer_column: str | None = Field(default=None, description="the column recording who was offered the change, when the offer and the taking are two columns")
    uptake_column: str | None = Field(default=None, description="the column recording who took it, when the offer and the taking are two columns")
    self_selection: bool | None = Field(default=None, description="whether units could move their own assignment after the offer or the rule")
    reason: str = ""
    cites: list[str] = Field(default_factory=list)
    unsure: list[Unsure] = Field(default_factory=list, description="what you would not guess, with why")
    by: Literal["pack", "judgement"] = "pack"

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("ladder:mechanism.kind", f"{self.kind or 'not said'} (set by the {self.by})"),
            ("ladder:mechanism.drivers", ", ".join(self.drivers) or "none named"),
            ("ladder:mechanism.offer", f"offer {self.offer_column or 'none'}; uptake {self.uptake_column or 'none'}"),
            ("ladder:mechanism.self_selection", "yes" if self.self_selection else "no" if self.self_selection is False else "not known"),
        ] + ([("ladder:mechanism.reason", self.reason)] if self.reason else [])


class Timing(BaseModel):
    """Rung 2: every other column's place in time against the treatment, by code from the pack."""

    before: list[str] = Field(default_factory=list)
    at: list[str] = Field(default_factory=list)
    after: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:timing.{w}", ", ".join(getattr(self, w)) or "none") for w in ("before", "at", "after", "unknown")]


class Role(Relation):
    """Rung 3: one pre-treatment column's role, read with every other pre-treatment column in view."""

    stands_for: str | None = Field(default=None, description="the thing outside the file this column stands in for, or null")
    redundant_with: str | None = Field(default=None, description="another listed column that carries the same information, or null")
    nested_in: str | None = Field(default=None, description="a listed column this one is a finer version of, or null")
    modifier_candidate: bool = Field(default=False, description="the effect could plausibly differ across this column's values")
    links: list[Cited] = Field(default_factory=list, description="the reasons for stands_for, redundant_with and nested_in, each cited")

    def word(self) -> str:
        if self.is_outcome_measure:
            return "another measure of the outcome"
        to_t, to_y = self.affects_treatment, self.affects_outcome
        base = (
            "confounder"
            if to_t and to_y
            else "outcome driver"
            if to_y
            else "treatment driver"
            if to_t
            else "changed by the treatment"
            if self.affected_by_treatment
            else "no role"
        )
        extra = []
        if self.stands_for:
            extra.append(f"stands for {self.stands_for}")
        if self.redundant_with:
            extra.append(f"same information as {self.redundant_with}")
        if self.nested_in:
            extra.append(f"sits inside {self.nested_in}")
        if self.modifier_candidate:
            extra.append("a candidate modifier")
        return base + (f"; {'; '.join(extra)}" if extra else "")


class Roles(BaseModel):
    items: list[Role] = Field(description="one per column listed, all of them")
    unsure: list[Unsure] = Field(default_factory=list, description="what you would not guess, with why; the answer above still stands")

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:roles.{r.column}", r.word()) for r in self.items]


PostKind = Literal["mediator", "outcome_measure", "consequence_of_treatment", "consequence_of_outcome", "background", "unrelated"]


class PostRole(BaseModel):
    """Rung 4: one column set at or after the treatment."""

    column: str
    kind: PostKind = Field(
        description="mediator: the treatment changed it and it moves the outcome; outcome_measure: another measurement of the outcome; "
        "consequence_of_treatment: the treatment changed it and it does not move the outcome; consequence_of_outcome: the outcome moved it; "
        "background: recorded late but fixed before the treatment, a cause of the outcome; unrelated: none of these"
    )
    reason: str
    cites: list[str]
    departures: list[Departure] = Field(default_factory=list)

    def relation(self) -> Relation:
        k = self.kind
        return Relation(
            column=self.column,
            affects_treatment=False,
            affects_outcome=k in ("mediator", "background"),
            affected_by_treatment=k in ("mediator", "consequence_of_treatment", "consequence_of_outcome"),
            is_outcome_measure=k == "outcome_measure",
            reasons=[Cited(reason=self.reason, cites=self.cites)],
            departures=self.departures,
        )


class PostRoles(BaseModel):
    items: list[PostRole] = Field(description="one per column listed, all of them")
    unsure: list[Unsure] = Field(default_factory=list, description="what you would not guess, with why; the answer above still stands")

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:post_roles.{r.column}", f"{r.kind.replace('_', ' ')}: {r.reason}") for r in self.items]


RoadKind = Literal["backdoor", "frontdoor", "iv"]


class Road(BaseModel):
    """Rung 6: the road taken among the roads the graph opens, argued from the hidden factors and the rungs below."""

    taken: RoadKind = Field(description="the road the design takes: backdoor (adjust), frontdoor (through the mediator), iv (through the instrument)")
    why: str = Field(description="one or two sentences: why this road over the others open")
    cites: list[str] = Field(default_factory=list)
    unsure: list[Unsure] = Field(default_factory=list)
    open: list[str] = Field(default_factory=list, description="filled by code: every road the graph opens")
    hidden_factor: bool = Field(default=False, description="filled by code: the person says a hidden factor exists")
    by: Literal["pack", "code", "judgement"] = "code"

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("ladder:road.taken", f"{self.taken} (set by the {self.by})"),
            ("ladder:road.open", ", ".join(self.open) or "none"),
            ("ladder:road.hidden_factor", "the person says one exists" if self.hidden_factor else "none declared"),
            ("ladder:road.why", self.why),
        ]


class Modifier(Cited):
    column: str


class Heterogeneity(BaseModel):
    """Rung 7: where the effect could differ, and for whom it is wanted."""

    modifiers: list[Modifier] = Field(default_factory=list, description="at most the number allowed, each a listed candidate, each cited")
    why: str = ""
    cites: list[str] = Field(default_factory=list)
    unsure: list[Unsure] = Field(default_factory=list)
    target_units: str = Field(default="ate", description="filled by code from the mechanism and the question's scope")
    by: Literal["code", "judgement"] = "code"

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("ladder:heterogeneity.modifiers", ", ".join(m.column for m in self.modifiers) or "none"),
            ("ladder:heterogeneity.target", self.target_units),
        ] + ([("ladder:heterogeneity.why", self.why)] if self.why else [])


class Threat(BaseModel):
    """One risk of this design, named by code from the pack and the rungs below, with the addresses it rests on."""

    name: str
    level: Literal["soft", "hard"]
    text: str
    cites: list[str] = Field(default_factory=list)


class Threats(BaseModel):
    """Rung 8: the risks of this design, each a flag the assessment must answer and the interpretation must cite."""

    items: list[Threat] = Field(default_factory=list)

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:threats.{t.name}", f"{t.level}: {t.text}") for t in self.items] or [("ladder:threats.none", "no threat named by the pack")]


class Ladder(BaseModel):
    """The rungs climbed so far. A rung reads the rungs below it; every line has an address the record, the report and the
    chat after can cite."""

    pair: Pair | None = None
    mechanism: Mechanism | None = None
    timing: Timing | None = None
    roles: Roles | None = None
    post_roles: PostRoles | None = None
    road: Road | None = None
    heterogeneity: Heterogeneity | None = None
    threats: Threats | None = None

    def rungs(self) -> list[tuple[str, Any]]:
        names = ("pair", "mechanism", "timing", "roles", "post_roles", "road", "heterogeneity", "threats")
        return [(n, r) for n in names if (r := getattr(self, n)) is not None]

    def lines(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for _, rung in self.rungs():
            out += rung.lines()
        return out

    def unsure_all(self) -> list[tuple[str, Unsure]]:
        """Every item a rung would not guess, with the rung's name."""
        return [(name, u) for name, rung in self.rungs() for u in getattr(rung, "unsure", [])]

    def addresses(self) -> set[str]:
        return {a for a, _ in self.lines()}

    def resolve(self, address: str) -> bool:
        a = norm_address(address)
        return any(norm_address(x) == a for x in self.addresses())

    def render(self) -> str:
        return ("THE LADDER SO FAR\n" + "\n".join(f"[{a}] {t}" for a, t in self.lines())) if self.lines() else ""


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
    """What identifies the effect on this graph: every road DoWhy finds, and the one the design takes."""

    kind: Literal["backdoor", "frontdoor", "iv", "none"]
    roads: list[str] = Field(default_factory=list, description="every road that identifies the effect: backdoor, frontdoor, iv")
    adjustment_set: list[str] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    frontdoor_set: list[str] = Field(default_factory=list)
    alternatives: dict[str, list[str]] = Field(default_factory=dict)
    sensitivity_required: bool = Field(default=False, description="the person says a hidden factor exists and no road avoids it: the caveat must say so")
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
    modifiers: list[str] = Field(default_factory=list, description="the columns the effect is also estimated within, level by level")
    frozen_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def render(self) -> str:
        lines = [
            "DESIGN",
            "  contrasts    " + "; ".join(f"{c.treated} vs {c.control} ({c.reason})" for c in self.contrasts),
            "  graph",
        ]
        lines += ["    " + ln for ln in self.graph.render().splitlines()]
        e = self.estimand
        road = {
            "backdoor": f"adjust for {', '.join(e.adjustment_set) or 'nothing'}",
            "iv": f"through the instrument {', '.join(e.instruments)}",
            "frontdoor": f"through the mediator {', '.join(e.frontdoor_set)}",
            "none": "nothing identifies it",
        }[e.kind]
        lines.append(
            f"  estimand     {e.kind}; {road}"
            + (f"; roads open: {', '.join(e.roads)}" if len(e.roads) > 1 else "")
            + ("; a hidden factor is believed to exist and no road avoids it" if e.sensitivity_required else "")
        )
        for r in self.checks.results:
            lines.append(f"  check        {r.level:4} {r.address}  {r.detail}")
        lines.append(f"  estimator    {self.estimator}" + (f" (+ {self.also_run} as secondary)" if self.also_run else "") + f"  params {self.params}")
        lines.append(f"  refuters     {', '.join(self.refuters)}")
        lines.append(f"  target       {self.target_units}")
        lines.append(f"  modifiers    {', '.join(self.modifiers) or 'none'}")
        lines.append(f"  frozen at    {self.frozen_at}")
        return "\n".join(lines)
