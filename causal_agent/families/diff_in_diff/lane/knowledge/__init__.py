"""The lane's knowledge files: the entry models, and the shared loader bound to this folder. Read once, rendered for the model, filtered by facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from causal_agent.lane.knowledge import Knowledge

_HERE = Path(__file__).parent


class EstimatorEntry(BaseModel):
    name: str
    in_words: str
    formula: str
    controls_mode: str = "plain"  # plain | csw0 | none
    applies_when: dict[str, Any]
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    rank: int = 99
    also_run: str | None = None

    def applies(self, *, cohorts: int, periods_pre: int, periods_post: int) -> bool:
        w = self.applies_when
        if "cohorts" in w and cohorts not in w["cohorts"]:
            return False
        if "cohorts_min" in w and cohorts < w["cohorts_min"]:
            return False
        return periods_pre >= w.get("periods_pre_min", 0) and periods_post >= w.get("periods_post_min", 0)

    def render(self) -> str:
        return f"estimator: {self.name}\n  what it does: {self.in_words}\n  assumes: {self.assumes}\n  weak when: {self.weak_when}"


class InferenceEntry(BaseModel):
    name: str
    in_words: str
    applies_when: dict[str, Any]
    vcov: Any
    resample: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    def applies(self, *, units_treated: int, kind: str) -> bool:
        w = self.applies_when
        if "kind" in w and kind not in w["kind"]:
            return False
        return w.get("units_treated_min", 0) <= units_treated <= w.get("units_treated_max", 10**9)


class PlaceboEntry(BaseModel):
    name: str
    in_words: str
    applies_when: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    def applies(self, *, units: int, periods_pre: int) -> bool:
        w = self.applies_when
        return units >= w.get("units_min", 0) and periods_pre >= w.get("periods_pre_min", 0)


# ------------------------------------------------------------------ the files, through the shared loader

K = Knowledge(_HERE, estimator=EstimatorEntry, inference=InferenceEntry, placebo=PlaceboEntry)
load_estimators, load_inference, load_placebos, load_checks, load_beliefs = K.estimators, K.inference, K.placebos, K.checks, K.beliefs
estimator, placebo, pick_inference, render_preferences = K.estimator, K.placebo, K.pick_inference, K.render_preferences
