"""The design brief: what the design rests on, in the dataset's terms. The Designer writes one per hand-off; the pack carries it,
the record prints it, the chat cites its lines."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts.base import Cited

Road = Literal["backdoor", "frontdoor", "iv"]


class DecisionMade(BaseModel):
    """One of the family's decisions, decided for this dataset."""

    name: str = Field(description="a decision name from the family's list")
    choice: str = Field(description="what was decided, in the dataset's words: 'compare completed against none'")
    rests_on: list[str] = Field(description="addresses in the memory or the probes it rests on; at least one")
    reason: str = Field(description="one sentence")

    @property
    def address(self) -> str:
        return f"design.brief.{self.name}"

    def render(self) -> str:
        return f"[{self.address}] {self.choice} (rests on {', '.join(f'[{a}]' for a in self.rests_on)}): {self.reason}"


class DesignBrief(BaseModel):
    """What the design rests on: the family, the road when the family has roads, the target, every decision the family lists,
    the threats with their cites, the checks that would show them, and the one sentence the design bets on."""

    family: str
    road: Road | None = Field(default=None, description="the identification road, for a family that has roads; null otherwise")
    target: Literal["average", "treated"] = "average"
    decisions: list[DecisionMade]
    threats: list[Cited] = Field(default_factory=list, description="what would break the design, each cited")
    checks: list[str] = Field(default_factory=list, description="the checks that would show a threat, in words")
    bets_on: str = Field(description="the assumption, one sentence, in this dataset's terms")

    def decision(self, name: str) -> DecisionMade | None:
        return next((d for d in self.decisions if d.name == name), None)

    def lines(self) -> list[tuple[str, str]]:
        """Every line with its address, in the order the brief reads. A family with a decision named road says the road on that
        decision's line, first; one without gets a road line of its own only when a road is set."""
        out = []
        for d in self.decisions:
            head = f"{self.road}: " if d.name == "road" and self.road is not None else ""
            out.append((d.address, f"{head}{d.choice} (rests on {', '.join(f'[{a}]' for a in d.rests_on)}): {d.reason}"))
        if self.road is not None and self.decision("road") is None:
            out.append(("design.brief.road", self.road))
        out.append(("design.brief.target", self.target))
        for i, t in enumerate(self.threats, start=1):
            out.append((f"design.brief.threat:{i}", f"{t.reason} (cites {', '.join(f'[{a}]' for a in t.cites)})"))
        for i, c in enumerate(self.checks, start=1):
            out.append((f"design.brief.check:{i}", c))
        out.append(("design.brief.bets_on", self.bets_on))
        return out

    def render(self) -> str:
        return "\n".join(f"[{a}] {text}" for a, text in self.lines())

    def addresses(self) -> set[str]:
        return {"design.brief"} | {a for a, _ in self.lines()}

    def cites(self) -> list[str]:
        """Every address a decision or a threat cites, in order, once each."""
        return list(dict.fromkeys([a for d in self.decisions for a in d.rests_on] + [a for t in self.threats for a in t.cites]))
