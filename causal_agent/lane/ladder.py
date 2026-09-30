"""The shape every lane's ladder shares: one record per rung, each line addressed `ladder:<rung>.<field>`, a rung reading the
rungs below it; the records that mean the same thing in every design (what a rung would not guess, a threat, a modifier); and
the code that turns the threats and the unsure items into flags the assessment must answer, and an unsure claim the interview
could have settled into a decline. A lane keeps its own rungs and their gates in its package; what is here is shape, not method.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import CheckResult, Cited, ColumnBrief, Decline, Handoff
from causal_agent.common.contracts.base import _slug
from causal_agent.lane.case import Case
from causal_agent.memory.catalogue import load_catalogue

_CATALOGUE = load_catalogue()  # to tell a claim the interview could have settled from a rung's own item


class Unsure(BaseModel):
    """One thing a rung would not guess: what it is about (an address when there is one) and why. A flag, never a question."""

    about: str = Field(description="the address of the claim or item you could not settle, such as col:lunch.may_modify, or a short name")
    reason: str


class Threat(BaseModel):
    """One risk of this design, named by code from the pack and the rungs below, with the addresses it rests on."""

    name: str
    level: Literal["soft", "hard"]
    text: str
    cites: list[str] = Field(default_factory=list)


class Threats(BaseModel):
    """The risks of this design, each a flag the assessment must answer and the interpretation must cite."""

    items: list[Threat] = Field(default_factory=list)

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:threats.{t.name}", f"{t.level}: {t.text}") for t in self.items] or [("ladder:threats.none", "no threat named by the pack")]


class Modifier(Cited):
    column: str


class Heterogeneity(BaseModel):
    """Where the effect could differ, and for whom it is wanted."""

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


class LadderBase(BaseModel):
    """The rungs climbed so far. A subclass lists its rung fields in `ORDER`; each rung has `lines()` and may have `unsure`."""

    ORDER: ClassVar[tuple[str, ...]] = ()

    def rungs(self) -> list[tuple[str, Any]]:
        return [(n, r) for n in self.ORDER if (r := getattr(self, n, None)) is not None]

    def lines(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for _, rung in self.rungs():
            out += rung.lines()
        return out

    def addresses(self) -> set[str]:
        return {a for a, _ in self.lines()}

    def resolve(self, address: str) -> bool:
        a = norm_address(address)
        return any(norm_address(x) == a for x in self.addresses())

    def render(self) -> str:
        return ("THE LADDER SO FAR\n" + "\n".join(f"[{a}] {t}" for a, t in self.lines())) if self.lines() else ""

    def unsure_all(self) -> list[tuple[str, Unsure]]:
        """Every item a rung would not guess, with the rung's name."""
        return [(name, u) for name, rung in self.rungs() for u in getattr(rung, "unsure", [])]


# ------------------------------------------------------------------ the threats every design shares, by code from the pack

_SAMPLING_WORDS = {
    "by_arm": "by whether the unit got the change",
    "by_group": "by group, region or type",
    "by_period": "by period",
    "unknown": "in a way the person could not say",
}


def pack_threats(h: Handoff, case: Case, outcome_key: str, columns: dict[str, str], *, outcome_timing: bool = True) -> list[Threat]:
    """Selection into the file, the outcome's timing, and gaps whose reason is not settled: the risks any design carries, read
    from the pack. A lane appends the risks of its own design; a design that observes the outcome on both sides of the change
    passes `outcome_timing=False`."""
    items: list[Threat] = []
    how = (h.sampling or {}).get("how")
    if how == "by_outcome":
        items.append(
            Threat(
                name="selection",
                level="hard",
                text="rows were drawn by how the outcome turned out; a comparison among the selected does not reach the population the question asks about",
                cites=["claim:sampling.how"],
            )
        )
    elif how in _SAMPLING_WORDS:
        items.append(
            Threat(
                name="selection",
                level="soft",
                text=f"rows were drawn {_SAMPLING_WORDS[how]}; who is in the file may differ by arm, and the estimate speaks for the file, not the population",
                cites=["claim:sampling.how"],
            )
        )
    yb: ColumnBrief | None = h.column(outcome_key)
    yw = (case.fact(f"col:{outcome_key}.when") or (yb.when if yb is not None else "unknown")) if outcome_timing else "after"
    if yw == "before":
        items.append(
            Threat(
                name="outcome_timing",
                level="hard",
                text="the outcome was measured before the change; it cannot carry its effect",
                cites=[f"col:{outcome_key}.when"],
            )
        )
    elif yw == "unknown":
        items.append(
            Threat(
                name="outcome_timing",
                level="soft",
                text="when the outcome was measured is not settled; an outcome measured before the change cannot carry its effect",
                cites=[f"col:{outcome_key}.when"],
            )
        )
    gaps = [b for b in h.columns if b.key in columns and b.facts.nulls > 0]
    if gaps and not (h.missing or {}).get("why"):
        items.append(
            Threat(
                name="missingness",
                level="soft",
                text=f"values are missing in {', '.join(b.name for b in gaps)} and the reason is not settled; a gap that differs by arm biases the comparison",
                cites=[b.address for b in gaps],
            )
        )
    return items


# ------------------------------------------------------------------ flags and declines from the ladder


def catalogue_address(address: str) -> bool:
    """Whether an address names a claim the interview could have settled: a field of a kind in the catalogue."""
    a = norm_address(address)
    if a.startswith("claim:"):
        kind, _, field = a[6:].partition(".")
        spec = _CATALOGUE.kinds.get(kind)
        return spec is not None and field in spec.fields
    if a.startswith("col:"):
        _, _, field = a[4:].partition(".")
        return field in _CATALOGUE.kinds["measured"].fields
    return False


def checks_and_declines(ladder: LadderBase, threats: Threats | None, existing: list[Decline]) -> tuple[list[CheckResult], list[Decline]]:
    """The threats as checks, and every item a rung would not guess as a soft flag; an unsure on a claim the interview could have
    settled is also a decline, `needs.unsettled`, which says the family's decisions miss a claim the reasoning needed. A decline
    already recorded is not recorded twice."""
    out = [CheckResult(contrast="all", name=f"threat.{t.name}", level=t.level, detail=t.text) for t in (threats.items if threats else [])]
    declines: list[Decline] = []
    have = {(d.about, d.check) for d in existing}
    for rung, u in ladder.unsure_all():
        out.append(CheckResult(contrast="all", name=f"unsure.{_slug(u.about)}", level="soft", detail=f"the {rung} rung would not settle {u.about}: {u.reason}"))
        if catalogue_address(u.about) and (u.about, "needs.unsettled") not in have:
            declines.append(
                Decline(
                    stage=rung,
                    kind="declined",
                    about=u.about,
                    reason=f"{u.reason}; the interview could have settled this before the run, so a decision in family.yaml should rest on it",
                    check="needs.unsettled",
                )
            )
    return out, declines
