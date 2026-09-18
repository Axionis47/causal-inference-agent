"""Router contracts. What each node writes. Every claim carries the addresses it cites."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["effect_of_change", "driver_search", "root_cause", "not_causal"]


class Cited(BaseModel):
    reason: str = Field(description="one or two sentences")
    cites: list[str] = Field(description="addresses from the pack, e.g. col:lunch.note, change:1.note, dataset.profile.grain")


class Candidate(Cited):
    column: str


class Scope(BaseModel):
    population_filter: str | None = Field(default=None, description="a filter on rows the question implies, in words, or null")
    window: str | None = Field(default=None, description="time window the question implies, or null")
    contrast: Literal["switch", "dose", "level_vs_level", "none"] = "switch"
    target: Literal["average", "on_treated", "on_untreated", "conditional", "counterfactual"] = "average"


class QuestionFrame(BaseModel):
    intent: Intent
    decision_served: str = Field(description="what decision the answer would inform, in one sentence")
    outcome_candidates: list[Candidate] = Field(description="ranked, best first")
    cause_candidates: list[Candidate] = Field(description="ranked, best first; empty if intent is driver_search or root_cause")
    scope: Scope
    relevant_columns: list[Candidate] = Field(
        description="every column that matters to this question, including the outcome and cause, each with why and a citation"
    )
    reasons: list[Cited] = Field(description="anything else the reader should know about how the question was read")

    @property
    def outcome(self) -> str | None:
        return self.outcome_candidates[0].column if self.outcome_candidates else None

    @property
    def cause(self) -> str | None:
        return self.cause_candidates[0].column if self.cause_candidates else None


class PrefilterVote(Cited):
    column: str
    relevant: bool


class NeedCheck(BaseModel):
    need: str
    met: bool
    cites: list[str] = Field(default_factory=list)
    note: str = Field(default="", description="why met or unmet, one sentence")


class FamilyVerdict(BaseModel):
    family: str
    admissible: bool
    needs: list[NeedCheck]
    concern: str = Field(default="", description="if admissible, the weak_when condition that applies here, or empty")


class Rejection(BaseModel):
    family: str
    reason: str
    cites: list[str] = Field(default_factory=list)


class FamilyDecision(BaseModel):
    admissible: list[str]
    chosen: str = Field(description="one of admissible; the literal string 'none' if no family is admissible")
    chosen_assumption: str = Field(description="the assumption this choice bets on, for this data; 'none' if no family chosen")
    why_over_alternatives: str = Field(description="if more than one family was admissible, why this one; else 'only admissible family'")
    rejected: list[Rejection]
    cites: list[str] = Field(default_factory=list, description="pack addresses supporting the choice; may be empty")


class Handoff(BaseModel):
    family: str
    specialist: str
    supported_now: bool
    outcome: str
    treatment: str | None
    scope: Scope
    pack_name: str
    relevant_columns: list[Candidate] = Field(description="the slice the specialist starts from; it may pull other cards by key")
    chosen_assumption: str
    reasons: list[Cited]


class Thought(BaseModel):
    """Debug only. Never gated, cited, or read by another node."""

    node: str
    text: str
    thinking_tokens: int | None = None
    output_tokens: int | None = None


# ----------------------------------------------------------------- specialist artifacts
# Lane-invariant. Every specialist produces these shapes; lane-specific parts sit inside.


class Contrast(Cited):
    """One comparison: rows at `control` versus rows at `treated`."""

    control: str
    treated: str

    @property
    def key(self) -> str:
        return f"{_slug(self.treated)}_vs_{_slug(self.control)}"


class CheckResult(BaseModel):
    contrast: str = Field(description="contrast key, or 'all' when the check does not depend on the contrast")
    name: str
    level: Literal["pass", "soft", "hard"]
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    @property
    def address(self) -> str:
        return f"check:{self.contrast}.{self.name}"


class Checks(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)

    @property
    def flags(self) -> list[CheckResult]:
        return [r for r in self.results if r.level != "pass"]

    @property
    def hard(self) -> list[CheckResult]:
        return [r for r in self.results if r.level == "hard"]


class Estimate(BaseModel):
    contrast: str
    method: str
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_treated: int = 0
    n_control: int = 0
    target_units: str = "ate"
    error: str | None = None
    secondary: bool = False


class Refutation(BaseModel):
    contrast: str
    refuter: str
    kind: Literal["falsification", "sensitivity"]
    new_effect: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    p_value: float | None = None
    passed: bool | None = Field(default=None, description="None for sensitivity; it reports a range, not a verdict")
    detail: str = ""


class Interpretation(BaseModel):
    contrast: str
    answer: str = Field(description="the answer to the question for this contrast, two or three sentences, in the outcome's units")
    effect_stated: float = Field(description="the effect size you are reporting, copied from the estimate")
    caveats: list[str] = Field(description="what the reader must know: the assumption bet on, flags, refuters that failed")
    cites: list[str] = Field(description="artifact addresses such as estimate:<contrast>.value, check:<contrast>.overlap, refute:<contrast>.<refuter>.p_value")


class Feasibility(BaseModel):
    """An honest stop. Which stage, why, the facts, and what would make the analysis possible."""

    stage: str
    reason: str
    facts: list[str] = Field(default_factory=list)
    what_would_fix: str = ""


def _slug(s: str) -> str:
    import re

    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(s).strip()).strip("_").lower()
    return out or "x"
