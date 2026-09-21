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
    params: dict[str, Any] = Field(default_factory=dict)
    fuzzy: Any = False  # True | False | "inherit"
    covs: bool = False
    pickable: bool = True
    applies_when: dict[str, Any] = Field(default_factory=dict)
    runs_when: dict[str, Any] = Field(default_factory=dict)
    estimand: str
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    rank: int = 99
    also_run: list[str] = Field(default_factory=list)

    def applies(self, *, kind: str, first_stage: str | None, distinct_scores: int) -> bool:
        w = self.applies_when
        if "kind" in w and kind not in w["kind"]:
            return False
        if "first_stage" in w and first_stage not in w["first_stage"]:
            return False
        if "distinct_scores_max" in w and distinct_scores > w["distinct_scores_max"]:
            return False
        return True

    def render(self) -> str:
        return f"estimator: {self.name}\n  what it does: {self.in_words}\n  assumes: {self.assumes}\n  weak when: {self.weak_when}"


class InferenceEntry(BaseModel):
    name: str
    in_words: str
    applies_when: dict[str, Any]
    vce: str
    cluster: str | None = None
    point_row: str = "Conventional"
    interval_row: str = "Robust"

    def applies(self, *, cluster_column: bool) -> bool:
        w = self.applies_when
        return "cluster_column" not in w or w["cluster_column"] == cluster_column


class PlaceboEntry(BaseModel):
    name: str
    in_words: str
    source: str = ""
    spec: str = "primary"
    placement: str | None = None
    grid: list[str] = Field(default_factory=list)
    hold_b: bool = False
    radii_share_of_h: list[float] = Field(default_factory=list)
    applies_when: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)

    def applies(self) -> bool:
        return True


# ------------------------------------------------------------------ the files, through the shared loader

K = Knowledge(_HERE, estimator=EstimatorEntry, inference=InferenceEntry, placebo=PlaceboEntry)
load_estimators, load_inference, load_placebos, load_checks, load_beliefs = K.estimators, K.inference, K.placebos, K.checks, K.beliefs
estimator, placebo, pick_inference, render_preferences = K.estimator, K.placebo, K.pick_inference, K.render_preferences
