"""The lane's knowledge files: the entry models, and the shared loader bound to this folder. Read once, rendered for the model, filtered by facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from causal_agent.lane.knowledge import Knowledge, matches

_HERE = Path(__file__).parent


class EstimatorEntry(BaseModel):
    """One estimator the catalogue offers. `engine` names the library surface: feols runs `formula`; did2s runs `first_stage`
    on the untreated observations and `second_stage` on the residual; lpdid and saturated take the canonical panel whole.
    `applies_when` is keyed on the facts the lane computes (cohorts, never_treated, periods, units) and matched by one rule."""

    name: str
    in_words: str
    engine: Literal["feols", "did2s", "lpdid", "saturated"] = "feols"
    formula: str = ""
    first_stage: str = ""
    second_stage: str = ""
    controls_mode: str = "plain"  # plain | csw0 | none | first_stage | xfml
    params: dict[str, Any] = Field(default_factory=dict)
    reports: list[str] = Field(default_factory=list, description="what the fit yields beside the estimate: dynamic, by_cohort")
    applies_when: dict[str, Any]
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    rank: int = 99
    also_run: str | None = None

    def applies(self, **facts: Any) -> bool:
        return matches(self.applies_when, facts)

    def render(self) -> str:
        return f"estimator: {self.name}\n  what it does: {self.in_words}\n  assumes: {self.assumes}\n  weak when: {self.weak_when}"


class InferenceEntry(BaseModel):
    name: str
    in_words: str
    source: str = ""
    applies_when: dict[str, Any]
    vcov: Any
    resample: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    def applies(self, **facts: Any) -> bool:
        return matches(self.applies_when, facts)


class PlaceboEntry(BaseModel):
    """One falsification or sensitivity. A falsification carries a verdict under `pass_when`; a sensitivity reports a range and
    no verdict. `applies_when` is keyed on facts of the frozen design and matched by one rule."""

    name: str
    in_words: str
    kind: Literal["falsification", "sensitivity"] = "falsification"
    source: str = ""
    applies_when: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    def applies(self, **facts: Any) -> bool:
        return matches(self.applies_when, facts)


# ------------------------------------------------------------------ the files, through the shared loader

K = Knowledge(_HERE, estimator=EstimatorEntry, inference=InferenceEntry, placebo=PlaceboEntry)
load_estimators, load_inference, load_placebos, load_checks, load_beliefs = K.estimators, K.inference, K.placebos, K.checks, K.beliefs
estimator, placebo, pick_inference, render_preferences = K.estimator, K.placebo, K.pick_inference, K.render_preferences
